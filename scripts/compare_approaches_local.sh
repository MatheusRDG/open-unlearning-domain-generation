#!/bin/bash
# =============================================================================
# LOCAL COMPARISON SCRIPT (Mac M-series / CPU)
# =============================================================================
#
# Lightweight version for running on local machine without heavy GPU.
# Uses smaller models and simpler evaluation.
#
# Requirements:
#   - Apple Silicon Mac with 16GB+ RAM (or CUDA GPU with 8GB+)
#   - OPENAI_API_KEY for dataset generation
#
# Usage:
#   bash scripts/compare_approaches_local.sh [DOMAIN]
#
# Examples:
#   bash scripts/compare_approaches_local.sh biosecurity
#   bash scripts/compare_approaches_local.sh "harry potter"
#
# =============================================================================

set -e

# =============================================================================
# CONFIGURATION (Optimized for Local)
# =============================================================================

DOMAIN="${1:-biosecurity}"
DOMAIN_SLUG=$(echo "$DOMAIN" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')

# Use smaller model that fits in 24GB unified memory
MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
MODEL_SHORT="Llama-3.2-1B"

# Timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${DOMAIN_SLUG}_local_${TIMESTAMP}"

# Directories
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_CODE="$PROJECT_ROOT/.docs/Synthetic_Textbook"
COMPARISON_DIR="$PROJECT_ROOT/results/comparison_local/${DOMAIN_SLUG}/${TIMESTAMP}"
DATASETS_DIR="$PROJECT_ROOT/data/comparison/${DOMAIN_SLUG}"

# Local-optimized settings
UNLEARN_EPOCHS=3
UNLEARN_LR=1e-5
UNLEARN_BATCH_SIZE=1
GRADIENT_ACCUMULATION=8

# Smaller generation for faster testing
PAPER_NUM_SUBFIELDS=3
PAPER_NUM_CHAPTERS_PER_BP=2
OUR_NUM_TOPICS=3

# Methods to test
METHODS=("NPO")  # NPO is more stable, start with just one

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

# =============================================================================
# SETUP
# =============================================================================

echo ""
echo "============================================================================="
echo -e "${GREEN}LOCAL COMPARISON: Paper vs Our Approach${NC}"
echo "============================================================================="
echo ""
echo "Domain: $DOMAIN"
echo "Model: $MODEL_SHORT (small, fits in RAM)"
echo "Output: $COMPARISON_DIR"
echo ""

mkdir -p "$COMPARISON_DIR"
mkdir -p "$DATASETS_DIR/paper"
mkdir -p "$DATASETS_DIR/ours"

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    log_warn "OPENAI_API_KEY not set - will try to use cached/downloaded datasets"
fi

# =============================================================================
# STEP 1: GET/GENERATE DATASETS
# =============================================================================

echo ""
echo "============================================================================="
echo "STEP 1: Prepare Datasets"
echo "============================================================================="

PAPER_DATASET="$DATASETS_DIR/paper/textbook_${DOMAIN_SLUG}.csv"
OUR_TEXT_DATASET="$DATASETS_DIR/ours/text_dataset.csv"

# --- Paper's Dataset ---
if [ -f "$PAPER_DATASET" ]; then
    log_info "Paper dataset exists: $PAPER_DATASET"
else
    log_info "Downloading Paper's pre-generated dataset from HuggingFace..."

    case "$DOMAIN_SLUG" in
        "biosecurity")
            HF_DATASET="WhyTheMoon/textbook_bio"
            ;;
        "cybersecurity")
            HF_DATASET="WhyTheMoon/textbook_cyber"
            ;;
        "harry_potter")
            HF_DATASET="WhyTheMoon/textbook_hp"
            ;;
        *)
            log_warn "No pre-generated dataset for '$DOMAIN'. Will generate locally."
            HF_DATASET=""
            ;;
    esac

    if [ -n "$HF_DATASET" ]; then
        uv run python << PYEOF
from datasets import load_dataset
import pandas as pd

hf_dataset = "${HF_DATASET}"
output_path = "${PAPER_DATASET}"

print(f"Downloading {hf_dataset}...")
# These datasets have model-specific splits, use gpt_4o_mini
ds = load_dataset(hf_dataset, split="gpt_4o_mini")
df = pd.DataFrame(ds)
df.to_csv(output_path, index=False)
print(f"Saved {len(df)} samples to {output_path}")
PYEOF
    else
        # Generate locally with paper's code (smaller scale)
        log_info "Generating with Paper's pipeline (small scale)..."
        cd "$PAPER_CODE"
        uv run python scripts/generate_textbook.py \
            --provider openai \
            --keyword "$DOMAIN" \
            --model-name gpt-4o-mini \
            --stages all \
            --num-subfields $PAPER_NUM_SUBFIELDS \
            --num-chapters-per-bp $PAPER_NUM_CHAPTERS_PER_BP \
            --data-path "$DATASETS_DIR/paper"

        # Find and rename output
        GENERATED=$(ls "$DATASETS_DIR/paper/"*_textbook_processed.csv 2>/dev/null | head -1)
        if [ -n "$GENERATED" ]; then
            mv "$GENERATED" "$PAPER_DATASET"
        fi
        cd "$PROJECT_ROOT"
    fi
fi

log_success "Paper dataset: $(wc -l < "$PAPER_DATASET") lines"

# --- Our Dataset ---
OUR_DOMAIN_JSON="$DATASETS_DIR/ours/domain.json"

if [ -f "$OUR_TEXT_DATASET" ]; then
    log_info "Our dataset exists: $OUR_TEXT_DATASET"
else
    if [ -f "$OUR_DOMAIN_JSON" ]; then
        log_info "Converting existing domain.json..."
    else
        log_info "Generating with Our pipeline..."

        export GEN_TOPICS_MIN_ITEMS=$OUR_NUM_TOPICS
        export GEN_TOPICS_MAX_ITEMS=$OUR_NUM_TOPICS
        export GEN_GROUNDED_QA_MIN_ITEMS=10
        export GEN_GROUNDED_QA_MAX_ITEMS=15
        export GEN_TOC_MIN_ITEMS=3
        export GEN_TOC_MAX_ITEMS=5

        uv run python -m src.domain_generation.main \
            --name "$DOMAIN" \
            --description "Domain knowledge about $DOMAIN"

        LATEST=$(ls -td output/*/ 2>/dev/null | head -1)
        if [ -n "$LATEST" ] && [ -f "${LATEST}domain.json" ]; then
            cp "${LATEST}domain.json" "$OUR_DOMAIN_JSON"
        fi
    fi

    # Convert to text CSV
    uv run python << PYEOF
import json
import csv

with open("$OUR_DOMAIN_JSON", 'r') as f:
    domain = json.load(f)

texts = []

# Extract text from books
for book in domain.get('books', []):
    for chapter in book.get('chapters', []):
        for section in chapter.get('sections', []):
            content = section.get('content', '')
            if content and len(content.split()) >= 30:
                texts.append(content)

# Extract from articles
for article in domain.get('articles', []):
    for section in article.get('sections', []):
        content = section.get('content', '')
        if content and len(content.split()) >= 30:
            texts.append(content)

with open("$OUR_TEXT_DATASET", 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['text'])
    for text in texts:
        writer.writerow([text])

print(f"Extracted {len(texts)} text samples")
PYEOF
fi

log_success "Our dataset: $(wc -l < "$OUR_TEXT_DATASET") lines"

# =============================================================================
# STEP 2: UNLEARNING (Local-optimized)
# =============================================================================

echo ""
echo "============================================================================="
echo "STEP 2: Run Unlearning"
echo "============================================================================="

# Detect device
if uv run python -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null | grep -q "True"; then
    DEVICE_INFO="MPS (Apple Silicon)"
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # Prevent OOM
elif uv run python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    DEVICE_INFO="CUDA"
else
    DEVICE_INFO="CPU (slow!)"
fi

log_info "Using device: $DEVICE_INFO"

for METHOD in "${METHODS[@]}"; do
    log_info "Method: $METHOD"

    # --- Paper's dataset ---
    PAPER_MODEL="saves/unlearn/${RUN_NAME}_paper_${METHOD}"

    if [ -d "$PAPER_MODEL" ]; then
        log_info "Paper model exists"
    else
        log_info "Unlearning: Paper's dataset + $METHOD"

        uv run python src/train.py --config-name=unlearn.yaml \
            model.model_args.pretrained_model_name_or_path="$MODEL_NAME" \
            +data.forget.handler=PretrainingDataset \
            "+data.forget.args.hf_args.path=$PAPER_DATASET" \
            +data.forget.args.text_key=text \
            +data.retain.handler=PretrainingDataset \
            +data.retain.args.hf_args.path=wikitext \
            +data.retain.args.hf_args.name=wikitext-2-raw-v1 \
            +data.retain.args.hf_args.split=train \
            trainer="$METHOD" \
            trainer.args.num_train_epochs=$UNLEARN_EPOCHS \
            trainer.args.learning_rate=$UNLEARN_LR \
            trainer.args.per_device_train_batch_size=$UNLEARN_BATCH_SIZE \
            trainer.args.gradient_accumulation_steps=$GRADIENT_ACCUMULATION \
            trainer.args.output_dir="$PAPER_MODEL" \
            task_name="${RUN_NAME}_paper_${METHOD}" \
            2>&1 | tee "$COMPARISON_DIR/unlearn_paper_${METHOD}.log" || {
                log_warn "Paper unlearning failed"
            }
    fi

    # --- Our dataset ---
    OUR_MODEL="saves/unlearn/${RUN_NAME}_ours_${METHOD}"

    if [ -d "$OUR_MODEL" ]; then
        log_info "Our model exists"
    else
        log_info "Unlearning: Our dataset + $METHOD"

        uv run python src/train.py --config-name=unlearn.yaml \
            model.model_args.pretrained_model_name_or_path="$MODEL_NAME" \
            +data.forget.handler=PretrainingDataset \
            "+data.forget.args.hf_args.path=$OUR_TEXT_DATASET" \
            +data.forget.args.text_key=text \
            +data.retain.handler=PretrainingDataset \
            +data.retain.args.hf_args.path=wikitext \
            +data.retain.args.hf_args.name=wikitext-2-raw-v1 \
            +data.retain.args.hf_args.split=train \
            trainer="$METHOD" \
            trainer.args.num_train_epochs=$UNLEARN_EPOCHS \
            trainer.args.learning_rate=$UNLEARN_LR \
            trainer.args.per_device_train_batch_size=$UNLEARN_BATCH_SIZE \
            trainer.args.gradient_accumulation_steps=$GRADIENT_ACCUMULATION \
            trainer.args.output_dir="$OUR_MODEL" \
            task_name="${RUN_NAME}_ours_${METHOD}" \
            2>&1 | tee "$COMPARISON_DIR/unlearn_ours_${METHOD}.log" || {
                log_warn "Our unlearning failed"
            }
    fi
done

# =============================================================================
# STEP 3: SIMPLE EVALUATION (No heavy benchmarks)
# =============================================================================

echo ""
echo "============================================================================="
echo "STEP 3: Evaluate (Lightweight)"
echo "============================================================================="

EVAL_SCRIPT="$COMPARISON_DIR/evaluate.py"

cat > "$EVAL_SCRIPT" << 'PYEOF'
#!/usr/bin/env python3
"""Simple local evaluation - perplexity-based comparison."""

import argparse
import json
import os
import sys

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_perplexity(model, tokenizer, texts, device, max_samples=50):
    """Compute average perplexity on texts."""
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for text in tqdm(texts[:max_samples], desc="Computing perplexity"):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=256
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            try:
                outputs = model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item()
                count += 1
            except Exception as e:
                continue

    avg_loss = total_loss / count if count > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity, avg_loss


def evaluate_model(model_path, forget_dataset, device):
    """Evaluate a single model."""
    print(f"\nLoading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # Use float32 for MPS compatibility
        device_map=None,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN")
    )
    model = model.to(device)
    model.eval()

    # Load forget dataset
    df = pd.read_csv(forget_dataset)
    texts = df['text'].tolist()

    # Compute perplexity on forget set
    ppl, loss = compute_perplexity(model, tokenizer, texts, device)

    # Free memory
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        'perplexity': ppl,
        'avg_loss': loss,
        'num_samples': min(50, len(texts))
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-model', required=True)
    parser.add_argument('--paper-model', default=None)
    parser.add_argument('--ours-model', default=None)
    parser.add_argument('--paper-dataset', required=True)
    parser.add_argument('--ours-dataset', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    results = {}

    # Baseline on both datasets
    print("\n" + "="*60)
    print("Evaluating BASELINE model")
    print("="*60)

    results['baseline_on_paper_data'] = evaluate_model(
        args.baseline_model, args.paper_dataset, device
    )
    results['baseline_on_ours_data'] = evaluate_model(
        args.baseline_model, args.ours_dataset, device
    )

    # Paper's unlearned model
    if args.paper_model and os.path.exists(args.paper_model):
        print("\n" + "="*60)
        print("Evaluating PAPER'S unlearned model")
        print("="*60)

        results['paper_unlearned'] = evaluate_model(
            args.paper_model, args.paper_dataset, device
        )

    # Our unlearned model
    if args.ours_model and os.path.exists(args.ours_model):
        print("\n" + "="*60)
        print("Evaluating OUR unlearned model")
        print("="*60)

        results['ours_unlearned'] = evaluate_model(
            args.ours_model, args.ours_dataset, device
        )

    # Compute metrics
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    comparison = []

    # Baseline
    comparison.append({
        'model': 'Baseline',
        'dataset': 'Paper',
        'perplexity': results['baseline_on_paper_data']['perplexity'],
        'loss': results['baseline_on_paper_data']['avg_loss'],
    })
    comparison.append({
        'model': 'Baseline',
        'dataset': 'Ours',
        'perplexity': results['baseline_on_ours_data']['perplexity'],
        'loss': results['baseline_on_ours_data']['avg_loss'],
    })

    # Paper unlearned
    if 'paper_unlearned' in results:
        ppl_change = (results['paper_unlearned']['perplexity'] /
                     results['baseline_on_paper_data']['perplexity'] - 1) * 100
        comparison.append({
            'model': 'Paper Unlearned',
            'dataset': 'Paper',
            'perplexity': results['paper_unlearned']['perplexity'],
            'loss': results['paper_unlearned']['avg_loss'],
            'ppl_change_%': ppl_change,
        })

    # Ours unlearned
    if 'ours_unlearned' in results:
        ppl_change = (results['ours_unlearned']['perplexity'] /
                     results['baseline_on_ours_data']['perplexity'] - 1) * 100
        comparison.append({
            'model': 'Ours Unlearned',
            'dataset': 'Ours',
            'perplexity': results['ours_unlearned']['perplexity'],
            'loss': results['ours_unlearned']['avg_loss'],
            'ppl_change_%': ppl_change,
        })

    # Print table
    print("\n| Model | Dataset | Perplexity | Loss | PPL Change % |")
    print("|-------|---------|------------|------|--------------|")
    for row in comparison:
        ppl_change = row.get('ppl_change_%', '-')
        if isinstance(ppl_change, float):
            ppl_change = f"{ppl_change:+.1f}%"
        print(f"| {row['model']} | {row['dataset']} | {row['perplexity']:.2f} | {row['loss']:.4f} | {ppl_change} |")

    # Save results
    with open(f"{args.output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)

    df = pd.DataFrame(comparison)
    df.to_csv(f"{args.output_dir}/comparison.csv", index=False)

    # Generate markdown
    md = f"""# Local Comparison Results

## Perplexity Comparison

Higher perplexity after unlearning = model "forgot" the content (good!)

| Model | Dataset | Perplexity | Loss | PPL Change % |
|-------|---------|------------|------|--------------|
"""
    for row in comparison:
        ppl_change = row.get('ppl_change_%', '-')
        if isinstance(ppl_change, float):
            ppl_change = f"{ppl_change:+.1f}%"
        md += f"| {row['model']} | {row['dataset']} | {row['perplexity']:.2f} | {row['loss']:.4f} | {ppl_change} |\n"

    md += """
## Interpretation

- **Baseline**: Model's original knowledge of the domain
- **Unlearned**: After applying unlearning algorithm
- **PPL Change %**: Positive = forgot (good), Negative = still knows (bad)

Higher perplexity change indicates more effective unlearning.
"""

    with open(f"{args.output_dir}/COMPARISON_TABLE.md", 'w') as f:
        f.write(md)

    print(f"\nResults saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
PYEOF

chmod +x "$EVAL_SCRIPT"

# Run evaluation
log_info "Running evaluation..."

for METHOD in "${METHODS[@]}"; do
    PAPER_MODEL="saves/unlearn/${RUN_NAME}_paper_${METHOD}"
    OUR_MODEL="saves/unlearn/${RUN_NAME}_ours_${METHOD}"
    EVAL_DIR="$COMPARISON_DIR/eval_${METHOD}"

    mkdir -p "$EVAL_DIR"

    HF_TOKEN="$HF_TOKEN" uv run python "$EVAL_SCRIPT" \
        --baseline-model "$MODEL_NAME" \
        --paper-model "$PAPER_MODEL" \
        --ours-model "$OUR_MODEL" \
        --paper-dataset "$PAPER_DATASET" \
        --ours-dataset "$OUR_TEXT_DATASET" \
        --output-dir "$EVAL_DIR" \
        2>&1 | tee "$COMPARISON_DIR/eval_${METHOD}.log"
done

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "============================================================================="
echo -e "${GREEN}COMPARISON COMPLETE${NC}"
echo "============================================================================="
echo ""
echo "Results: $COMPARISON_DIR"
echo ""

if [ -f "$COMPARISON_DIR/eval_NPO/COMPARISON_TABLE.md" ]; then
    cat "$COMPARISON_DIR/eval_NPO/COMPARISON_TABLE.md"
fi

echo ""
log_success "Done! Check $COMPARISON_DIR for full results."
