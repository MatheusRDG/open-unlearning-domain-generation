#!/bin/bash
# =============================================================================
# COMPARISON SCRIPT: Our Domain Generation vs Paper's Synthetic Textbook
# =============================================================================
#
# This script compares two approaches for LLM unlearning dataset generation:
# 1. Paper's approach: Synthetic Textbook (3-stage: subdomains → bullet points → chapters)
# 2. Our approach: Domain Generation (topics → books/articles → QA pairs)
#
# Both approaches generate forget sets for the SAME domain, then unlearn using
# the SAME methods and evaluate using the SAME metrics.
#
# Usage:
#   bash scripts/compare_approaches.sh [DOMAIN] [MODEL]
#
# Examples:
#   bash scripts/compare_approaches.sh biosecurity mistral-7b
#   bash scripts/compare_approaches.sh cybersecurity llama-3-8b
#   bash scripts/compare_approaches.sh "harry potter" mistral-7b
#
# =============================================================================

set -eo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

DOMAIN="${1:-biosecurity}"
MODEL_SIZE="${2:-llama-3b}"  # Default to 3B (7B models need >24GB for methods with ref_model)

# Normalize domain name for file paths
DOMAIN_SLUG=$(echo "$DOMAIN" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')

# Timestamp for this comparison run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${DOMAIN_SLUG}_comparison_${TIMESTAMP}"

# Directories
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_CODE="$PROJECT_ROOT/.docs/Synthetic_Textbook"
COMPARISON_DIR="$PROJECT_ROOT/results/comparison/${DOMAIN_SLUG}/${TIMESTAMP}"
DATASETS_DIR="$PROJECT_ROOT/data/comparison/${DOMAIN_SLUG}"

# Model mapping (paper uses these models)
case "$MODEL_SIZE" in
    "mistral-7b"|"mistral")
        MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
        MODEL_SHORT="Mistral-7B"
        ;;
    "llama-3-8b"|"llama3"|"llama")
        MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
        MODEL_SHORT="Llama-3-8B"
        ;;
    "llama-3.2-1b"|"llama-1b")
        MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
        MODEL_SHORT="Llama-3.2-1B"
        ;;
    "llama-3.2-3b"|"llama-3b")
        MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
        MODEL_SHORT="Llama-3.2-3B"
        ;;
    *)
        MODEL_NAME="$MODEL_SIZE"
        MODEL_SHORT="$MODEL_SIZE"
        ;;
esac

# Unlearning methods to compare
METHODS=("RMU" "NPO")

# Generation settings
PAPER_NUM_SUBFIELDS=10
PAPER_NUM_CHAPTERS_PER_BP=5
OUR_NUM_TOPICS=10
OUR_NUM_QA=20

# Unlearning hyperparameters (from paper's grid search)
UNLEARN_EPOCHS=5
UNLEARN_LR=1e-5
UNLEARN_BATCH_SIZE=8
MAX_LENGTH=512

# Evaluation benchmarks
EVAL_BENCHMARKS="mmlu,gsm8k,triviaqa"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo ""
    echo "============================================================================="
    echo -e "${GREEN}$1${NC}"
    echo "============================================================================="
    echo ""
}

# =============================================================================
# STEP 0: SETUP
# =============================================================================

log_section "STEP 0: Setup and Validation"

mkdir -p "$COMPARISON_DIR"
mkdir -p "$DATASETS_DIR/paper"
mkdir -p "$DATASETS_DIR/ours"
mkdir -p "$COMPARISON_DIR/logs"
mkdir -p "$COMPARISON_DIR/models"

# Log file for this run
LOG_FILE="$COMPARISON_DIR/logs/comparison_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

log_info "Comparison Run Configuration:"
echo "  Domain: $DOMAIN"
echo "  Model: $MODEL_NAME ($MODEL_SHORT)"
echo "  Methods: ${METHODS[*]}"
echo "  Run Name: $RUN_NAME"
echo "  Output Dir: $COMPARISON_DIR"
echo ""

# Check for required environment variables
if [ -z "$OPENAI_API_KEY" ]; then
    log_warn "OPENAI_API_KEY not set. Dataset generation will fail."
fi

# Check if paper code exists
if [ ! -d "$PAPER_CODE" ]; then
    log_error "Paper code not found at $PAPER_CODE"
    log_info "Cloning paper repository..."
    git clone https://github.com/xyzhu123/Synthetic_Textbook.git "$PAPER_CODE"
fi

# =============================================================================
# STEP 1: GENERATE DATASETS
# =============================================================================

log_section "STEP 1: Dataset Generation"

# -----------------------------------------------------------------------------
# 1A: Paper's Synthetic Textbook Method
# -----------------------------------------------------------------------------

log_info "1A: Generating dataset using Paper's Synthetic Textbook method..."

PAPER_DATASET="$DATASETS_DIR/paper/textbook_${DOMAIN_SLUG}.csv"

if [ -f "$PAPER_DATASET" ]; then
    log_info "Paper dataset already exists: $PAPER_DATASET"
else
    # Download pre-generated datasets from HuggingFace
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
            log_error "No pre-generated dataset for domain: $DOMAIN"
            log_error "Only biosecurity, cybersecurity, harry_potter are supported"
            exit 1
            ;;
    esac

    log_info "Downloading $HF_DATASET from HuggingFace..."
    uv run python << PYEOF
from datasets import load_dataset
import pandas as pd

ds = load_dataset("${HF_DATASET}", split="gpt_4o_mini")
df = pd.DataFrame(ds)
df.to_csv("${PAPER_DATASET}", index=False)
print(f"Downloaded {len(df)} samples to ${PAPER_DATASET}")
PYEOF

    # Verify file was created
    if [ ! -f "$PAPER_DATASET" ]; then
        log_error "Failed to download paper dataset"
        exit 1
    fi
fi

log_success "Paper dataset: $PAPER_DATASET ($(wc -l < "$PAPER_DATASET") lines)"

# -----------------------------------------------------------------------------
# 1B: Our Domain Generation Method
# -----------------------------------------------------------------------------

log_info "1B: Generating dataset using Our Domain Generation method..."

OUR_DATASET_DIR="$DATASETS_DIR/ours"
OUR_DOMAIN_JSON="$OUR_DATASET_DIR/domain.json"
OUR_TEXT_DATASET="$OUR_DATASET_DIR/text_dataset.csv"

if [ -f "$OUR_DOMAIN_JSON" ]; then
    log_info "Our domain.json already exists: $OUR_DOMAIN_JSON"
else
    log_info "Running domain generation pipeline..."

    # Set generation parameters
    export GEN_TOPICS_MIN_ITEMS=$OUR_NUM_TOPICS
    export GEN_TOPICS_MAX_ITEMS=$OUR_NUM_TOPICS
    export GEN_GROUNDED_QA_MIN_ITEMS=$OUR_NUM_QA
    export GEN_GROUNDED_QA_MAX_ITEMS=$OUR_NUM_QA

    uv run python -m src.domain_generation.main \
        --name "$DOMAIN" \
        --description "Domain knowledge about $DOMAIN for unlearning comparison" \
        2>&1 | tee "$COMPARISON_DIR/logs/domain_generation.log"

    # Find the most recent output
    LATEST_OUTPUT=$(ls -td output/*/ 2>/dev/null | head -1)
    if [ -n "$LATEST_OUTPUT" ] && [ -f "${LATEST_OUTPUT}domain.json" ]; then
        cp "${LATEST_OUTPUT}domain.json" "$OUR_DOMAIN_JSON"
        log_success "Domain JSON saved to: $OUR_DOMAIN_JSON"
    else
        log_error "Domain generation failed - no output found"
        exit 1
    fi
fi

# Convert our domain.json to text dataset (comparable to paper's CSV)
if [ -f "$OUR_TEXT_DATASET" ]; then
    log_info "Our text dataset already exists: $OUR_TEXT_DATASET"
else
    log_info "Converting domain.json to text dataset..."
    uv run python -c "
import json
import csv

with open('$OUR_DOMAIN_JSON', 'r') as f:
    domain = json.load(f)

texts = []

# Extract all text content from books
for book in domain.get('books', []):
    for chapter in book.get('chapters', []):
        for section in chapter.get('sections', []):
            content = section.get('content', '')
            if content and len(content.split()) >= 45:  # Same threshold as paper
                texts.append(content)

# Extract from articles
for article in domain.get('articles', []):
    for section in article.get('sections', []):
        content = section.get('content', '')
        if content and len(content.split()) >= 45:
            texts.append(content)

# Write to CSV (same format as paper)
with open('$OUR_TEXT_DATASET', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['text'])
    for text in texts:
        writer.writerow([text])

print(f'Extracted {len(texts)} text samples')
"
fi

log_success "Our text dataset: $OUR_TEXT_DATASET"

# =============================================================================
# STEP 2: PREPARE DATASETS FOR UNLEARNING
# =============================================================================

log_section "STEP 2: Prepare Datasets for Unlearning"

# Create Hydra configs for both datasets
PAPER_CONFIG="configs/data/datasets/COMPARISON_paper_${DOMAIN_SLUG}.yaml"
OUR_CONFIG="configs/data/datasets/COMPARISON_ours_${DOMAIN_SLUG}.yaml"

log_info "Creating dataset configs..."

# Paper's dataset config (PretrainingDataset for text-only)
cat > "$PAPER_CONFIG" << EOF
# Auto-generated config for Paper's Synthetic Textbook dataset
COMPARISON_paper_${DOMAIN_SLUG}:
  handler: PretrainingDataset
  args:
    hf_args:
      path: "$PAPER_DATASET"
      split: "train"
    max_length: 512
    text_key: "text"
EOF

# Our dataset config
cat > "$OUR_CONFIG" << EOF
# Auto-generated config for Our Domain Generation dataset
COMPARISON_ours_${DOMAIN_SLUG}:
  handler: PretrainingDataset
  args:
    hf_args:
      path: "$OUR_TEXT_DATASET"
      split: "train"
    max_length: 512
    text_key: "text"
EOF

log_success "Dataset configs created"

# =============================================================================
# STEP 3: RUN UNLEARNING
# =============================================================================

log_section "STEP 3: Run Unlearning Experiments"

# WikiText as retain set (same as paper)
RETAIN_SET="wikitext"

declare -A UNLEARNED_MODELS

for METHOD in "${METHODS[@]}"; do
    log_info "Running unlearning with method: $METHOD"

    # -------------------------------------------------------------------------
    # 3A: Unlearn with Paper's dataset
    # -------------------------------------------------------------------------

    PAPER_MODEL_NAME="${RUN_NAME}_paper_${METHOD}"
    PAPER_MODEL_PATH="saves/unlearn/${PAPER_MODEL_NAME}"

    if [ -d "$PAPER_MODEL_PATH" ]; then
        log_info "Paper unlearned model exists: $PAPER_MODEL_PATH"
    else
        log_info "Unlearning with Paper's dataset + $METHOD..."

        HF_TOKEN="$HF_TOKEN" uv run python src/train.py --config-name=unlearn.yaml \
            model.model_args.pretrained_model_name_or_path="$MODEL_NAME" \
            '~data.forget' '~data.retain' \
            +data.forget.paper_forget.handler=PretrainingDataset \
            +data.forget.paper_forget.args.hf_args.path=csv \
            "+data.forget.paper_forget.args.hf_args.data_files=$PAPER_DATASET" \
            +data.forget.paper_forget.args.hf_args.split=train \
            +data.forget.paper_forget.args.text_key=text \
            +data.forget.paper_forget.args.max_length=$MAX_LENGTH \
            +data.retain.wikitext_retain.handler=PretrainingDataset \
            +data.retain.wikitext_retain.args.hf_args.path=wikitext \
            +data.retain.wikitext_retain.args.hf_args.name=wikitext-2-raw-v1 \
            +data.retain.wikitext_retain.args.hf_args.split=train \
            +data.retain.wikitext_retain.args.max_length=$MAX_LENGTH \
            trainer="$METHOD" \
            trainer.args.num_train_epochs="$UNLEARN_EPOCHS" \
            trainer.args.learning_rate="$UNLEARN_LR" \
            trainer.args.per_device_train_batch_size="$UNLEARN_BATCH_SIZE" \
            trainer.args.bf16=true \
            trainer.args.eval_strategy=no \
            trainer.args.save_strategy=no \
            trainer.args.load_best_model_at_end=false \
            '~eval' \
            task_name="$PAPER_MODEL_NAME" \
            2>&1 | tee "$COMPARISON_DIR/logs/unlearn_paper_${METHOD}.log"
    fi

    UNLEARNED_MODELS["paper_${METHOD}"]="$PAPER_MODEL_PATH"

    # -------------------------------------------------------------------------
    # 3B: Unlearn with Our dataset
    # -------------------------------------------------------------------------

    OUR_MODEL_NAME="${RUN_NAME}_ours_${METHOD}"
    OUR_MODEL_PATH="saves/unlearn/${OUR_MODEL_NAME}"

    if [ -d "$OUR_MODEL_PATH" ]; then
        log_info "Our unlearned model exists: $OUR_MODEL_PATH"
    else
        log_info "Unlearning with Our dataset + $METHOD..."

        HF_TOKEN="$HF_TOKEN" uv run python src/train.py --config-name=unlearn.yaml \
            model.model_args.pretrained_model_name_or_path="$MODEL_NAME" \
            '~data.forget' '~data.retain' \
            +data.forget.ours_forget.handler=PretrainingDataset \
            +data.forget.ours_forget.args.hf_args.path=csv \
            "+data.forget.ours_forget.args.hf_args.data_files=$OUR_TEXT_DATASET" \
            +data.forget.ours_forget.args.hf_args.split=train \
            +data.forget.ours_forget.args.text_key=text \
            +data.forget.ours_forget.args.max_length=$MAX_LENGTH \
            +data.retain.wikitext_retain.handler=PretrainingDataset \
            +data.retain.wikitext_retain.args.hf_args.path=wikitext \
            +data.retain.wikitext_retain.args.hf_args.name=wikitext-2-raw-v1 \
            +data.retain.wikitext_retain.args.hf_args.split=train \
            +data.retain.wikitext_retain.args.max_length=$MAX_LENGTH \
            trainer="$METHOD" \
            trainer.args.num_train_epochs="$UNLEARN_EPOCHS" \
            trainer.args.learning_rate="$UNLEARN_LR" \
            trainer.args.per_device_train_batch_size="$UNLEARN_BATCH_SIZE" \
            trainer.args.bf16=true \
            trainer.args.eval_strategy=no \
            trainer.args.save_strategy=no \
            trainer.args.load_best_model_at_end=false \
            '~eval' \
            task_name="$OUR_MODEL_NAME" \
            2>&1 | tee "$COMPARISON_DIR/logs/unlearn_ours_${METHOD}.log"
    fi

    UNLEARNED_MODELS["ours_${METHOD}"]="$OUR_MODEL_PATH"
done

log_success "Unlearning complete"

# =============================================================================
# STEP 4: EVALUATE ALL MODELS
# =============================================================================

log_section "STEP 4: Evaluation"

# Create evaluation script
EVAL_SCRIPT="$COMPARISON_DIR/run_evaluation.py"

cat > "$EVAL_SCRIPT" << 'EVAL_PYTHON'
#!/usr/bin/env python
"""
Evaluation script for comparison experiments.
Computes Unlearn Utility as defined in the paper.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_lm_eval(model_path: str, tasks: str, output_dir: str) -> dict:
    """Run lm-evaluation-harness and return results."""
    output_file = Path(output_dir) / "lm_eval_results.json"

    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},trust_remote_code=True",
        "--tasks", tasks,
        "--batch_size", "auto",
        "--output_path", str(output_dir),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Find and load results
        results_files = list(Path(output_dir).glob("**/results.json"))
        if results_files:
            with open(results_files[0]) as f:
                return json.load(f)
    except Exception as e:
        print(f"lm_eval failed: {e}")

    return {}


def compute_forget_score(model_path: str, forget_dataset: str) -> float:
    """Compute model's performance on forget set (lower = better forgetting)."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN")
    )
    model.eval()

    # Load forget dataset
    df = pd.read_csv(forget_dataset)
    texts = df['text'].tolist()[:100]  # Sample for efficiency

    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item()
            count += 1

    # Lower loss = model still knows the content (bad for unlearning)
    # We want higher loss = forgot the content
    avg_loss = total_loss / count if count > 0 else 0

    # Return as "forget score" - inverse of loss (higher = better forgetting)
    return avg_loss


def compute_unlearn_utility(
    baseline_forget: float,
    unlearned_forget: float,
    baseline_general: dict,
    unlearned_general: dict,
    alpha: float = 0.5,
    beta: float = 0.5
) -> dict:
    """
    Compute Unlearn Utility as defined in the paper.

    U = -α × Sf + β × Sr

    Where:
    - Sf = percentage change in forget benchmark (positive = model forgot)
    - Sr = percentage change in general benchmarks (negative = degradation)
    """
    # Sf: Change in forget score (higher loss after unlearning = forgot)
    # Paper uses benchmark accuracy decrease, we use loss increase
    Sf = ((unlearned_forget - baseline_forget) / baseline_forget) * 100 if baseline_forget > 0 else 0

    # Sr: Average change in general capabilities
    general_changes = []
    for task in ['mmlu', 'gsm8k', 'triviaqa']:
        baseline_score = baseline_general.get(task, {}).get('acc', 0)
        unlearned_score = unlearned_general.get(task, {}).get('acc', 0)
        if baseline_score > 0:
            change = ((unlearned_score - baseline_score) / baseline_score) * 100
            general_changes.append(change)

    Sr = sum(general_changes) / len(general_changes) if general_changes else 0

    # Unlearn Utility (higher = better)
    # Positive Sf is good (model forgot), positive Sr is good (capabilities preserved)
    U = alpha * Sf + beta * Sr

    return {
        'unlearn_utility': U,
        'forget_change_pct': Sf,
        'general_change_pct': Sr,
        'baseline_forget_loss': baseline_forget,
        'unlearned_forget_loss': unlearned_forget,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-model', required=True)
    parser.add_argument('--unlearned-model', required=True)
    parser.add_argument('--forget-dataset', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--tasks', default='mmlu,gsm8k,triviaqa')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Computing baseline metrics...")
    baseline_forget = compute_forget_score(args.baseline_model, args.forget_dataset)
    baseline_general = run_lm_eval(
        args.baseline_model,
        args.tasks,
        f"{args.output_dir}/baseline_lm_eval"
    )

    print("Computing unlearned metrics...")
    unlearned_forget = compute_forget_score(args.unlearned_model, args.forget_dataset)
    unlearned_general = run_lm_eval(
        args.unlearned_model,
        args.tasks,
        f"{args.output_dir}/unlearned_lm_eval"
    )

    print("Computing Unlearn Utility...")
    results = compute_unlearn_utility(
        baseline_forget, unlearned_forget,
        baseline_general.get('results', {}),
        unlearned_general.get('results', {})
    )

    # Add raw scores
    results['baseline_general'] = baseline_general.get('results', {})
    results['unlearned_general'] = unlearned_general.get('results', {})

    # Save results
    output_file = f"{args.output_dir}/evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print(f"\nUnlearn Utility: {results['unlearn_utility']:.2f}")
    print(f"Forget Change %: {results['forget_change_pct']:.2f}")
    print(f"General Change %: {results['general_change_pct']:.2f}")

    return results


if __name__ == '__main__':
    main()
EVAL_PYTHON

chmod +x "$EVAL_SCRIPT"

# Run evaluation for each model
RESULTS_FILE="$COMPARISON_DIR/comparison_results.json"
echo "{}" > "$RESULTS_FILE"

for METHOD in "${METHODS[@]}"; do
    for APPROACH in "paper" "ours"; do
        MODEL_KEY="${APPROACH}_${METHOD}"
        MODEL_PATH="${UNLEARNED_MODELS[$MODEL_KEY]}"

        if [ -d "$MODEL_PATH" ]; then
            log_info "Evaluating: $MODEL_KEY"

            EVAL_OUTPUT="$COMPARISON_DIR/eval_${MODEL_KEY}"

            if [ "$APPROACH" == "paper" ]; then
                FORGET_DS="$PAPER_DATASET"
            else
                FORGET_DS="$OUR_TEXT_DATASET"
            fi

            HF_TOKEN="$HF_TOKEN" uv run python "$EVAL_SCRIPT" \
                --baseline-model "$MODEL_NAME" \
                --unlearned-model "$MODEL_PATH" \
                --forget-dataset "$FORGET_DS" \
                --output-dir "$EVAL_OUTPUT" \
                2>&1 | tee "$COMPARISON_DIR/logs/eval_${MODEL_KEY}.log"
        else
            log_warn "Model not found: $MODEL_PATH"
        fi
    done
done

# =============================================================================
# STEP 5: GENERATE COMPARISON TABLE
# =============================================================================

log_section "STEP 5: Generate Comparison Tables"

TABLE_SCRIPT="$COMPARISON_DIR/generate_table.py"

cat > "$TABLE_SCRIPT" << 'TABLE_PYTHON'
#!/usr/bin/env python
"""Generate comparison tables from evaluation results."""

import json
import os
import sys
from pathlib import Path


def load_results(comparison_dir: str) -> dict:
    """Load all evaluation results."""
    results = {}

    for eval_dir in Path(comparison_dir).glob("eval_*"):
        key = eval_dir.name.replace("eval_", "")
        results_file = eval_dir / "evaluation_results.json"

        if results_file.exists():
            with open(results_file) as f:
                results[key] = json.load(f)

    return results


def generate_markdown_table(results: dict, model_name: str) -> str:
    """Generate markdown comparison table."""

    lines = [
        f"# Comparison Results: {model_name}",
        "",
        "## Quantitative Comparison",
        "",
        "| Approach | Method | Unlearn Utility (↑) | General Cap. Δ (↑) | Forget Δ (↑) | MMLU | GSM8K |",
        "|----------|--------|---------------------|-------------------|--------------|------|-------|",
    ]

    for key, data in sorted(results.items()):
        approach, method = key.split("_", 1)
        approach_name = "Textbook (Paper)" if approach == "paper" else "Domain-Gen (Ours)"

        utility = data.get('unlearn_utility', 0)
        general_delta = data.get('general_change_pct', 0)
        forget_delta = data.get('forget_change_pct', 0)

        general = data.get('unlearned_general', {})
        mmlu = general.get('mmlu', {}).get('acc', 0) * 100
        gsm8k = general.get('gsm8k', {}).get('acc', 0) * 100

        lines.append(
            f"| {approach_name} | {method} | {utility:.2f} | {general_delta:.2f}% | {forget_delta:.2f}% | {mmlu:.1f} | {gsm8k:.1f} |"
        )

    lines.extend([
        "",
        "## Metrics Explanation",
        "",
        "- **Unlearn Utility (U)**: Higher is better. U = 0.5×Sf + 0.5×Sr",
        "- **General Cap. Δ**: Change in general capabilities (closer to 0% = better preserved)",
        "- **Forget Δ**: Change in forget set loss (higher = better forgetting)",
        "- **MMLU/GSM8K**: Absolute scores after unlearning",
    ])

    return "\n".join(lines)


def generate_csv_table(results: dict) -> str:
    """Generate CSV comparison table."""
    lines = ["approach,method,unlearn_utility,general_change_pct,forget_change_pct,mmlu,gsm8k"]

    for key, data in sorted(results.items()):
        approach, method = key.split("_", 1)

        utility = data.get('unlearn_utility', 0)
        general_delta = data.get('general_change_pct', 0)
        forget_delta = data.get('forget_change_pct', 0)

        general = data.get('unlearned_general', {})
        mmlu = general.get('mmlu', {}).get('acc', 0)
        gsm8k = general.get('gsm8k', {}).get('acc', 0)

        lines.append(f"{approach},{method},{utility:.4f},{general_delta:.4f},{forget_delta:.4f},{mmlu:.4f},{gsm8k:.4f}")

    return "\n".join(lines)


def main():
    comparison_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    model_name = sys.argv[2] if len(sys.argv) > 2 else "Unknown Model"

    results = load_results(comparison_dir)

    if not results:
        print("No evaluation results found!")
        return

    # Generate markdown table
    md_table = generate_markdown_table(results, model_name)
    md_file = Path(comparison_dir) / "COMPARISON_TABLE.md"
    with open(md_file, 'w') as f:
        f.write(md_table)
    print(f"Markdown table: {md_file}")

    # Generate CSV
    csv_table = generate_csv_table(results)
    csv_file = Path(comparison_dir) / "comparison_results.csv"
    with open(csv_file, 'w') as f:
        f.write(csv_table)
    print(f"CSV table: {csv_file}")

    # Print to console
    print("\n" + md_table)


if __name__ == '__main__':
    main()
TABLE_PYTHON

uv run python "$TABLE_SCRIPT" "$COMPARISON_DIR" "$MODEL_SHORT"

# =============================================================================
# SUMMARY
# =============================================================================

log_section "COMPARISON COMPLETE"

echo "Results saved to: $COMPARISON_DIR"
echo ""
echo "Files generated:"
echo "  - $COMPARISON_DIR/COMPARISON_TABLE.md"
echo "  - $COMPARISON_DIR/comparison_results.csv"
echo "  - $COMPARISON_DIR/logs/"
echo ""

if [ -f "$COMPARISON_DIR/COMPARISON_TABLE.md" ]; then
    echo "============================================================================="
    echo "COMPARISON TABLE"
    echo "============================================================================="
    cat "$COMPARISON_DIR/COMPARISON_TABLE.md"
fi

log_success "Done! Use results for your MSc thesis comparison."
