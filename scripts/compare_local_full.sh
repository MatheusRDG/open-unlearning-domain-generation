#!/bin/bash
# =============================================================================
# LOCAL FULL COMPARISON: Paper vs Our Approach (with Training)
# =============================================================================
#
# Runs full comparison locally on Mac M-series with memory optimizations.
# Uses smallest model (Llama-3.2-1B) and reduced batch sizes.
#

set -e

DOMAIN="${1:-biosecurity}"
DOMAIN_SLUG=$(echo "$DOMAIN" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

COMPARISON_DIR="$PROJECT_ROOT/results/comparison_local/${DOMAIN_SLUG}/${TIMESTAMP}"
DATASETS_DIR="$PROJECT_ROOT/data/comparison/${DOMAIN_SLUG}"

# Model - smallest possible
MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
MODEL_SHORT="Llama-3.2-1B"

# Training settings - minimal for Mac
UNLEARN_EPOCHS=1
UNLEARN_LR=1e-5
BATCH_SIZE=1
GRAD_ACCUM=4
MAX_STEPS=50  # Limit steps for testing

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

mkdir -p "$COMPARISON_DIR"
mkdir -p "$DATASETS_DIR/paper"
mkdir -p "$DATASETS_DIR/ours"

# Memory optimizations for Mac
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "============================================================================="
echo "LOCAL FULL COMPARISON: Paper vs Our Approach"
echo "============================================================================="
echo ""
echo "Domain: $DOMAIN"
echo "Model: $MODEL_SHORT"
echo "Output: $COMPARISON_DIR"
echo "Max Steps: $MAX_STEPS (limited for testing)"
echo ""

# =============================================================================
# STEP 1: Prepare Datasets
# =============================================================================

echo "============================================================================="
echo "STEP 1: Prepare Datasets"
echo "============================================================================="

PAPER_DATASET="$DATASETS_DIR/paper/textbook_${DOMAIN_SLUG}.csv"
OUR_DATASET="$DATASETS_DIR/ours/text_dataset.csv"

# Download paper's dataset if needed
if [ ! -f "$PAPER_DATASET" ]; then
    log_info "Downloading paper's dataset from HuggingFace..."
    
    case "$DOMAIN_SLUG" in
        "biosecurity")
            HF_DATASET="WhyTheMoon/textbook_bio"
            ;;
        "cybersecurity")
            HF_DATASET="WhyTheMoon/textbook_cyber"
            ;;
        *)
            log_error "Unknown domain: $DOMAIN_SLUG"
            exit 1
            ;;
    esac
    
    uv run python << PYEOF
from datasets import load_dataset
import pandas as pd

hf_dataset = "${HF_DATASET}"
output_path = "${PAPER_DATASET}"

print(f"Downloading {hf_dataset}...")
ds = load_dataset(hf_dataset, split="gpt_4o_mini")
df = pd.DataFrame(ds)
df.to_csv(output_path, index=False)
print(f"Saved {len(df)} samples to {output_path}")
PYEOF
fi

# Check our dataset exists
if [ ! -f "$OUR_DATASET" ]; then
    log_error "Our dataset not found: $OUR_DATASET"
    log_info "Run domain generation first or copy existing data"
    exit 1
fi

PAPER_LINES=$(wc -l < "$PAPER_DATASET")
OUR_LINES=$(wc -l < "$OUR_DATASET")

log_success "Paper dataset: $PAPER_LINES lines"
log_success "Our dataset: $OUR_LINES lines"

# =============================================================================
# STEP 2: Run Unlearning
# =============================================================================

echo ""
echo "============================================================================="
echo "STEP 2: Run Unlearning (NPO method)"
echo "============================================================================="

# Check device
if uv run python -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null | grep -q "True"; then
    DEVICE="mps"
    log_info "Using MPS (Apple Silicon)"
elif uv run python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    DEVICE="cuda"
    log_info "Using CUDA"
else
    DEVICE="cpu"
    log_warn "Using CPU - this will be slow!"
fi

# --- Train with Paper's dataset ---
PAPER_MODEL="saves/unlearn/comparison_paper_${TIMESTAMP}"
log_info "Training with Paper's dataset..."

HF_TOKEN="$HF_TOKEN" uv run python src/train.py --config-name=unlearn.yaml \
    model=Llama-3.2-1B-Instruct \
    data=unlearn_comparison_paper \
    trainer=NPO \
    trainer.args.output_dir="$PAPER_MODEL" \
    trainer.args.num_train_epochs=$UNLEARN_EPOCHS \
    trainer.args.per_device_train_batch_size=$BATCH_SIZE \
    trainer.args.gradient_accumulation_steps=$GRAD_ACCUM \
    trainer.args.learning_rate=$UNLEARN_LR \
    +trainer.args.max_steps=$MAX_STEPS \
    trainer.args.logging_steps=10 \
    trainer.args.save_strategy="no" \
    trainer.args.eval_strategy="no" \
    trainer.args.load_best_model_at_end=false \
    task_name="comparison_paper_${TIMESTAMP}" \
    2>&1 | tee "$COMPARISON_DIR/train_paper.log" || {
    log_warn "Paper training failed - checking if OOM..."
    if grep -q "out of memory\|MPS backend out of memory\|CUDA out of memory" "$COMPARISON_DIR/train_paper.log"; then
        log_error "OOM detected! Try reducing MAX_STEPS or batch size"
    fi
}

# --- Train with Our dataset ---
OUR_MODEL="saves/unlearn/comparison_ours_${TIMESTAMP}"
log_info "Training with Our dataset..."

HF_TOKEN="$HF_TOKEN" uv run python src/train.py --config-name=unlearn.yaml \
    model=Llama-3.2-1B-Instruct \
    data=unlearn_comparison_ours \
    trainer=NPO \
    trainer.args.output_dir="$OUR_MODEL" \
    trainer.args.num_train_epochs=$UNLEARN_EPOCHS \
    trainer.args.per_device_train_batch_size=$BATCH_SIZE \
    trainer.args.gradient_accumulation_steps=$GRAD_ACCUM \
    trainer.args.learning_rate=$UNLEARN_LR \
    +trainer.args.max_steps=$MAX_STEPS \
    trainer.args.logging_steps=10 \
    trainer.args.save_strategy="no" \
    trainer.args.eval_strategy="no" \
    trainer.args.load_best_model_at_end=false \
    task_name="comparison_ours_${TIMESTAMP}" \
    2>&1 | tee "$COMPARISON_DIR/train_ours.log" || {
    log_warn "Our training failed - checking if OOM..."
    if grep -q "out of memory\|MPS backend out of memory\|CUDA out of memory" "$COMPARISON_DIR/train_ours.log"; then
        log_error "OOM detected! Try reducing MAX_STEPS or batch size"
    fi
}

# =============================================================================
# STEP 3: Evaluate
# =============================================================================

echo ""
echo "============================================================================="
echo "STEP 3: Evaluate Models"
echo "============================================================================="

# Export variables for Python script
export COMPARISON_DIR
export PAPER_DATASET
export OUR_DATASET
export MODEL_NAME
export PAPER_MODEL
export OUR_MODEL
export TIMESTAMP
export MAX_STEPS

uv run python << 'EVAL_EOF'
import os
import json
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def compute_perplexity(model, tokenizer, texts, device, max_samples=30):
    """Compute perplexity on texts."""
    model.eval()
    total_loss = 0.0
    count = 0
    
    with torch.no_grad():
        for text in tqdm(texts[:max_samples], desc="Computing PPL"):
            try:
                inputs = tokenizer(
                    text, return_tensors="pt", 
                    truncation=True, max_length=256
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item()
                count += 1
            except Exception as e:
                continue
    
    avg_loss = total_loss / count if count > 0 else float('inf')
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return ppl, avg_loss

def load_model(model_path, device):
    """Load model with memory optimization."""
    print(f"Loading: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True,
        token=os.environ.get("HF_TOKEN")
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # Use float32 for MPS compatibility
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
        low_cpu_mem_usage=True,
    )
    model.to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

# Paths
comparison_dir = os.environ.get("COMPARISON_DIR")
paper_dataset = os.environ.get("PAPER_DATASET")
our_dataset = os.environ.get("OUR_DATASET")
baseline_model = os.environ.get("MODEL_NAME")
paper_model = os.environ.get("PAPER_MODEL")
ours_model = os.environ.get("OUR_MODEL")
timestamp = os.environ.get("TIMESTAMP")

device = get_device()
print(f"Using device: {device}")

# Load datasets
paper_texts = pd.read_csv(paper_dataset)['text'].astype(str).tolist()
our_texts = pd.read_csv(our_dataset)['text'].astype(str).tolist()

results = {}

# Evaluate baseline
print("\n" + "="*60)
print("Evaluating BASELINE model")
print("="*60)

model, tokenizer = load_model(baseline_model, device)
results['baseline_on_paper'] = compute_perplexity(model, tokenizer, paper_texts, device)
results['baseline_on_ours'] = compute_perplexity(model, tokenizer, our_texts, device)
del model
if device.type == "mps":
    torch.mps.empty_cache()

# Evaluate paper's unlearned model
if os.path.exists(paper_model):
    print("\n" + "="*60)
    print("Evaluating PAPER unlearned model")
    print("="*60)
    
    model, tokenizer = load_model(paper_model, device)
    results['paper_unlearned'] = compute_perplexity(model, tokenizer, paper_texts, device)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
else:
    print(f"Paper model not found: {paper_model}")
    results['paper_unlearned'] = (float('inf'), float('inf'))

# Evaluate our unlearned model
if os.path.exists(ours_model):
    print("\n" + "="*60)
    print("Evaluating OUR unlearned model")
    print("="*60)
    
    model, tokenizer = load_model(ours_model, device)
    results['ours_unlearned'] = compute_perplexity(model, tokenizer, our_texts, device)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
else:
    print(f"Our model not found: {ours_model}")
    results['ours_unlearned'] = (float('inf'), float('inf'))

# Generate comparison table
print("\n" + "="*60)
print("RESULTS")
print("="*60)

baseline_paper_ppl, baseline_paper_loss = results['baseline_on_paper']
baseline_ours_ppl, baseline_ours_loss = results['baseline_on_ours']
paper_ppl, paper_loss = results['paper_unlearned']
ours_ppl, ours_loss = results['ours_unlearned']

# Calculate changes
paper_ppl_change = ((paper_ppl / baseline_paper_ppl) - 1) * 100 if baseline_paper_ppl > 0 else 0
ours_ppl_change = ((ours_ppl / baseline_ours_ppl) - 1) * 100 if baseline_ours_ppl > 0 else 0

print(f"\nBaseline on Paper data:  PPL={baseline_paper_ppl:.2f}")
print(f"Baseline on Our data:    PPL={baseline_ours_ppl:.2f}")
print(f"Paper Unlearned:         PPL={paper_ppl:.2f} ({paper_ppl_change:+.1f}%)")
print(f"Ours Unlearned:          PPL={ours_ppl:.2f} ({ours_ppl_change:+.1f}%)")

# Save results
output = {
    'baseline_paper': {'ppl': baseline_paper_ppl, 'loss': baseline_paper_loss},
    'baseline_ours': {'ppl': baseline_ours_ppl, 'loss': baseline_ours_loss},
    'paper_unlearned': {'ppl': paper_ppl, 'loss': paper_loss, 'ppl_change': paper_ppl_change},
    'ours_unlearned': {'ppl': ours_ppl, 'loss': ours_loss, 'ppl_change': ours_ppl_change},
}

with open(f"{comparison_dir}/eval_results.json", 'w') as f:
    json.dump(output, f, indent=2)

# Generate markdown table
md = f"""# Local Comparison Results

**Date**: {timestamp}
**Model**: {baseline_model}
**Domain**: biosecurity

## Perplexity Comparison

Higher perplexity after unlearning = model "forgot" the content (good for unlearning!)

| Model | Dataset | Perplexity | PPL Change |
|-------|---------|------------|------------|
| Baseline | Paper's Data | {baseline_paper_ppl:.2f} | - |
| Baseline | Our Data | {baseline_ours_ppl:.2f} | - |
| Paper Unlearned | Paper's Data | {paper_ppl:.2f} | {paper_ppl_change:+.1f}% |
| Ours Unlearned | Our Data | {ours_ppl:.2f} | {ours_ppl_change:+.1f}% |

## Interpretation

- **PPL Change > 0%**: Model has higher perplexity = forgot domain knowledge (GOOD)
- **PPL Change ≈ 0%**: No forgetting occurred
- **PPL Change < 0%**: Model remembers better (shouldn't happen)

## Winner

{"**Our approach**" if ours_ppl_change > paper_ppl_change else "**Paper's approach**"} achieved better forgetting 
({ours_ppl_change:+.1f}% vs {paper_ppl_change:+.1f}% perplexity increase).

---
*Note: This is a limited local test with {os.environ.get('MAX_STEPS')} training steps. Full comparison requires GPU.*
"""

with open(f"{comparison_dir}/COMPARISON_TABLE.md", 'w') as f:
    f.write(md)

print(f"\nResults saved to: {comparison_dir}/")
print(md)

EVAL_EOF

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "============================================================================="
echo "COMPARISON COMPLETE"
echo "============================================================================="
echo ""
echo "Results: $COMPARISON_DIR"
echo ""

if [ -f "$COMPARISON_DIR/COMPARISON_TABLE.md" ]; then
    cat "$COMPARISON_DIR/COMPARISON_TABLE.md"
fi

log_success "Done!"
