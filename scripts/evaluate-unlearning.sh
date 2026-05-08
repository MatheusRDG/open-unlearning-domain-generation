#!/bin/bash

##############################################################################
# Comprehensive Unlearning Evaluation Script
#
# Compares model generations across three stages:
# 1. Pretrained base model (before any training)
# 2. Finetuned model (if available, before unlearning)
# 3. Unlearned model (after unlearning)
#
# Usage:
#   bash scripts/evaluate-unlearning.sh <RUN_NAME> <BASE_MODEL>
#
# Example:
#   bash scripts/evaluate-unlearning.sh brazil_20260110_174240 meta-llama/Llama-3.2-1B-Instruct
##############################################################################

set -e

RUN_NAME="${1}"
BASE_MODEL="${2:-meta-llama/Llama-3.2-1B-Instruct}"
FINETUNE_CHECKPOINT="${3}"
RETAINONLY_CHECKPOINT="${4}"

if [ -z "$RUN_NAME" ]; then
    echo "Error: RUN_NAME required"
    echo "Usage: bash scripts/evaluate-unlearning.sh <RUN_NAME> <BASE_MODEL> [FINETUNE_CHECKPOINT] [RETAINONLY_CHECKPOINT]"
    echo ""
    echo "Example:"
    echo "  bash scripts/evaluate-unlearning.sh brazil_20260110_174240 meta-llama/Llama-3.2-1B-Instruct saves/finetune/brazil_finetune_20260110_174240 saves/finetune/brazil_retainonly_20260110_174240"
    exit 1
fi

CHECKPOINT_DIR="saves/unlearn/${RUN_NAME}"
EVAL_OUTPUT_DIR="saves/eval/${RUN_NAME}"

# Auto-detect finetune checkpoint if not provided
if [ -z "$FINETUNE_CHECKPOINT" ]; then
    # Extract dataset name and timestamp from run_name
    DATASET_NAME=$(echo "$RUN_NAME" | cut -d'_' -f1)
    TIMESTAMP=$(echo "$RUN_NAME" | cut -d'_' -f2-)

    # Try to find finetune checkpoint
    FINETUNE_CHECKPOINT="saves/finetune/${DATASET_NAME}_finetune_${TIMESTAMP}"

    if [ ! -d "$FINETUNE_CHECKPOINT" ]; then
        echo "⚠️  Warning: Finetuned checkpoint not found at: ${FINETUNE_CHECKPOINT}"
        echo "   Will skip finetuned model evaluation"
        FINETUNE_CHECKPOINT=""
    fi
fi

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                    Comprehensive Unlearning Evaluation                     ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Run Name:         ${RUN_NAME}"
echo "Base Model:       ${BASE_MODEL}"
echo "Finetuned Model:  ${FINETUNE_CHECKPOINT:-Not available}"
echo "Unlearned Model:  ${CHECKPOINT_DIR}"
echo "Output Directory: ${EVAL_OUTPUT_DIR}"
echo ""

##############################################################################
# Verify checkpoint exists
##############################################################################

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "✗ Checkpoint directory not found: ${CHECKPOINT_DIR}"
    echo ""
    echo "Available runs:"
    ls -1 saves/unlearn/ 2>/dev/null || echo "  No runs found"
    exit 1
fi

mkdir -p "${EVAL_OUTPUT_DIR}"

##############################################################################
# Run comprehensive evaluation
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running Comprehensive Evaluation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

uv run python - "$RUN_NAME" "$BASE_MODEL" "$FINETUNE_CHECKPOINT" "$CHECKPOINT_DIR" "$EVAL_OUTPUT_DIR" "$RETAINONLY_CHECKPOINT" << 'EVAL_SCRIPT'
import json
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from datetime import datetime
import glob as glob_module

# Configuration from command line args
run_name = sys.argv[1]
base_model_name = sys.argv[2]
finetune_model_path = sys.argv[3] if sys.argv[3] else None
checkpoint_dir = Path(sys.argv[4])
eval_output_dir = Path(sys.argv[5])
retainonly_model_path = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] else None

print("="*80)
print("COMPREHENSIVE UNLEARNING EVALUATION")
print("="*80)
print()

# Find the latest checkpoint
checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"), key=lambda x: int(x.name.split('-')[1]))
if checkpoints:
    unlearned_model_path = checkpoints[-1]
else:
    unlearned_model_path = checkpoint_dir

print(f"Base Model:       {base_model_name}")
print(f"Finetuned Model:  {finetune_model_path if finetune_model_path else 'Not available'}")
print(f"Retain-Only Model:{retainonly_model_path if retainonly_model_path else 'Not available'}")
print(f"Unlearned Model:  {unlearned_model_path}")
print()

# Extract dataset name from run name by finding which existing dataset matches
# run_name format: {dataset}_{YYYYMMDD}_{HHMMSS} (but dataset may contain underscores)
# Strategy: try progressively shorter prefixes against existing dataset directories
dataset_name = None
parts = run_name.split('_')
for i in range(len(parts) - 1, 0, -1):
    candidate = '_'.join(parts[:i])
    if Path(f"data/datasets/{candidate}/qa_dataset_forget").exists():
        dataset_name = candidate
        break
    if list(glob_module.glob(f"data/run/*/{candidate}/qa_dataset_forget")):
        dataset_name = candidate
        break

# Fallback: assume last two parts are timestamp (YYYYMMDD_HHMMSS)
if not dataset_name:
    fallback_parts = run_name.rsplit('_', 2)
    dataset_name = fallback_parts[0] if len(fallback_parts) == 3 else parts[0]

print(f"Dataset name extracted: {dataset_name}")

# Load datasets - check multiple locations
print("Loading datasets...")
forget_dataset_path = None
retain_dataset_path = None

# 1. Check pre-generated datasets first
if Path(f"data/datasets/{dataset_name}/qa_dataset_forget").exists():
    forget_dataset_path = f"data/datasets/{dataset_name}/qa_dataset_forget"
    retain_dataset_path = f"data/datasets/{dataset_name}/qa_dataset_retain"
    print(f"  Found pre-generated datasets in data/datasets/{dataset_name}/")

# 2. Check run-specific datasets
elif Path("data/run").exists():
    matches = glob_module.glob(f"data/run/*/{dataset_name}/qa_dataset_forget")
    if matches:
        forget_dataset_path = matches[0]
        retain_dataset_path = forget_dataset_path.replace("_forget", "_retain")
        print(f"  Found runtime datasets in {Path(forget_dataset_path).parent}/")

if not forget_dataset_path:
    print(f"✗ Error: Dataset not found for '{dataset_name}'")
    print(f"   Checked:")
    print(f"     - data/datasets/{dataset_name}/qa_dataset_forget")
    print(f"     - data/run/*/{dataset_name}/qa_dataset_forget")
    sys.exit(1)

try:
    from datasets import load_from_disk
    forget_dataset = load_from_disk(forget_dataset_path)
    retain_dataset = load_from_disk(retain_dataset_path)
    print(f"✓ Forget dataset: {len(forget_dataset)} samples")
    print(f"✓ Retain dataset: {len(retain_dataset)} samples")
except Exception as e:
    print(f"✗ Error loading datasets: {e}")
    sys.exit(1)

print()

# Function to generate responses
def generate_response(model, tokenizer, question, max_new_tokens=100):
    inputs = tokenizer(question, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            temperature=None,
            top_p=None,
            repetition_penalty=1.3,  # Prevent degenerate repetition loops
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the question from response using token count (more reliable than string length)
    input_length = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
    return response

# Evaluation results structure
results = {
    "run_name": run_name,
    "timestamp": datetime.now().isoformat(),
    "base_model": base_model_name,
    "unlearned_model": str(unlearned_model_path),
    "dataset": {
        "name": dataset_name,
        "forget_samples": len(forget_dataset),
        "retain_samples": len(retain_dataset)
    },
    "forget_evaluations": [],
    "retain_evaluations": []
}

print("="*80)
print("STAGE 1: Generating Base Model Responses")
print("="*80)
print()

print("Loading base model...")
try:
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print("✓ Base model loaded")
except Exception as e:
    print(f"✗ Error loading base model: {e}")
    sys.exit(1)

print()

# Generate all base model responses first
print(f"Generating base model responses for forget set ({len(forget_dataset)} samples)...")
base_forget_responses = []
for idx, sample in enumerate(tqdm(forget_dataset, desc="Base - Forget")):
    question = sample['question']
    response = generate_response(base_model, base_tokenizer, question)
    base_forget_responses.append(response)

print(f"Generating base model responses for retain set ({len(retain_dataset)} samples)...")
base_retain_responses = []
for idx, sample in enumerate(tqdm(retain_dataset, desc="Base - Retain")):
    question = sample['question']
    response = generate_response(base_model, base_tokenizer, question)
    base_retain_responses.append(response)

# Unload base model to free GPU memory
print()
print("Unloading base model to free GPU memory...")
del base_model
del base_tokenizer
torch.cuda.empty_cache()
print("✓ Base model unloaded")

print()
# STAGE 2: Finetuned Model
print("="*80)
print("STAGE 2: Generating Finetuned Model Responses")
print("="*80)
print()

finetune_forget_responses = []
finetune_retain_responses = []

if finetune_model_path and Path(finetune_model_path).exists():
    print(f"Loading finetuned model from: {finetune_model_path}")
    try:
        # Find latest checkpoint in finetuned model
        finetune_path = Path(finetune_model_path)
        finetune_checkpoints = sorted(finetune_path.glob("checkpoint-*"), key=lambda x: int(x.name.split('-')[1]))
        if finetune_checkpoints:
            finetune_load_path = finetune_checkpoints[-1]
        else:
            finetune_load_path = finetune_path

        finetune_tokenizer = AutoTokenizer.from_pretrained(str(finetune_load_path))
        finetune_model = AutoModelForCausalLM.from_pretrained(
            str(finetune_load_path),
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("✓ Finetuned model loaded")
        print()

        # Generate finetuned model responses
        print(f"Generating finetuned model responses for forget set ({len(forget_dataset)} samples)...")
        for idx, sample in enumerate(tqdm(forget_dataset, desc="Finetuned - Forget")):
            question = sample['question']
            response = generate_response(finetune_model, finetune_tokenizer, question)
            finetune_forget_responses.append(response)

        print(f"Generating finetuned model responses for retain set ({len(retain_dataset)} samples)...")
        for idx, sample in enumerate(tqdm(retain_dataset, desc="Finetuned - Retain")):
            question = sample['question']
            response = generate_response(finetune_model, finetune_tokenizer, question)
            finetune_retain_responses.append(response)

        # Unload finetuned model
        print()
        print("Unloading finetuned model to free GPU memory...")
        del finetune_model
        del finetune_tokenizer
        torch.cuda.empty_cache()
        print("✓ Finetuned model unloaded")

    except Exception as e:
        print(f"⚠️  Error loading finetuned model: {e}")
        print("Skipping finetuned model evaluation")
        finetune_forget_responses = [""] * len(forget_dataset)
        finetune_retain_responses = [""] * len(retain_dataset)
else:
    print("⚠️  Finetuned model not available, skipping...")
    finetune_forget_responses = [""] * len(forget_dataset)
    finetune_retain_responses = [""] * len(retain_dataset)

print()

# STAGE 2b: Retain-Only Model (theoretical baseline)
print("="*80)
print("STAGE 2b: Generating Retain-Only Model Responses (theoretical ceiling)")
print("="*80)
print()

retainonly_forget_responses = []
retainonly_retain_responses = []

if retainonly_model_path and Path(retainonly_model_path).exists():
    print(f"Loading retain-only model from: {retainonly_model_path}")
    try:
        ro_path = Path(retainonly_model_path)
        ro_checkpoints = sorted(ro_path.glob("checkpoint-*"), key=lambda x: int(x.name.split('-')[1]))
        ro_load_path = ro_checkpoints[-1] if ro_checkpoints else ro_path

        ro_tokenizer = AutoTokenizer.from_pretrained(str(ro_load_path))
        ro_model = AutoModelForCausalLM.from_pretrained(
            str(ro_load_path),
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("✓ Retain-only model loaded")
        print()

        print(f"Generating retain-only model responses for forget set ({len(forget_dataset)} samples)...")
        for idx, sample in enumerate(tqdm(forget_dataset, desc="RetainOnly - Forget")):
            question = sample['question']
            response = generate_response(ro_model, ro_tokenizer, question)
            retainonly_forget_responses.append(response)

        print(f"Generating retain-only model responses for retain set ({len(retain_dataset)} samples)...")
        for idx, sample in enumerate(tqdm(retain_dataset, desc="RetainOnly - Retain")):
            question = sample['question']
            response = generate_response(ro_model, ro_tokenizer, question)
            retainonly_retain_responses.append(response)

        print()
        print("Unloading retain-only model to free GPU memory...")
        del ro_model
        del ro_tokenizer
        torch.cuda.empty_cache()
        print("✓ Retain-only model unloaded")

    except Exception as e:
        print(f"⚠️  Error loading retain-only model: {e}")
        retainonly_forget_responses = [""] * len(forget_dataset)
        retainonly_retain_responses = [""] * len(retain_dataset)
else:
    print("⚠️  Retain-only model not available, skipping...")
    retainonly_forget_responses = [""] * len(forget_dataset)
    retainonly_retain_responses = [""] * len(retain_dataset)

print()

# STAGE 3: Unlearned Model
print("="*80)
print("STAGE 3: Generating Unlearned Model Responses")
print("="*80)
print()

print("Loading unlearned model...")
try:
    unlearned_tokenizer = AutoTokenizer.from_pretrained(str(unlearned_model_path))
    unlearned_model = AutoModelForCausalLM.from_pretrained(
        str(unlearned_model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print("✓ Unlearned model loaded")
except Exception as e:
    print(f"✗ Error loading unlearned model: {e}")
    sys.exit(1)

print()

# Generate all unlearned model responses
print(f"Generating unlearned model responses for forget set ({len(forget_dataset)} samples)...")
unlearned_forget_responses = []
for idx, sample in enumerate(tqdm(forget_dataset, desc="Unlearned - Forget")):
    question = sample['question']
    response = generate_response(unlearned_model, unlearned_tokenizer, question)
    unlearned_forget_responses.append(response)

print(f"Generating unlearned model responses for retain set ({len(retain_dataset)} samples)...")
unlearned_retain_responses = []
for idx, sample in enumerate(tqdm(retain_dataset, desc="Unlearned - Retain")):
    question = sample['question']
    response = generate_response(unlearned_model, unlearned_tokenizer, question)
    unlearned_retain_responses.append(response)

# Unload unlearned model
print()
print("Unloading unlearned model...")
del unlearned_model
del unlearned_tokenizer
torch.cuda.empty_cache()
print("✓ Unlearned model unloaded")

print()
print("="*80)
print("STAGE 4: Combining Results")
print("="*80)
print()

# Combine forget results
for idx, sample in enumerate(forget_dataset):
    results["forget_evaluations"].append({
        "index": idx,
        "question": sample['question'],
        "ground_truth": sample['answer'],
        "base_model_response": base_forget_responses[idx],
        "finetuned_model_response": finetune_forget_responses[idx],
        "retainonly_model_response": retainonly_forget_responses[idx],
        "unlearned_model_response": unlearned_forget_responses[idx]
    })

# Combine retain results
for idx, sample in enumerate(retain_dataset):
    results["retain_evaluations"].append({
        "index": idx,
        "question": sample['question'],
        "ground_truth": sample['answer'],
        "base_model_response": base_retain_responses[idx],
        "finetuned_model_response": finetune_retain_responses[idx],
        "retainonly_model_response": retainonly_retain_responses[idx],
        "unlearned_model_response": unlearned_retain_responses[idx]
    })

print()
print("="*80)
print("STAGE 5: Computing Metrics")
print("="*80)
print()

import re
from collections import Counter
import string

def tokenize_words(text):
    """Lowercase, remove punctuation, split into words."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return [w for w in text.split() if len(w) > 1]

def word_overlap(text_a, text_b):
    """Jaccard similarity of word sets."""
    words_a = set(tokenize_words(text_a))
    words_b = set(tokenize_words(text_b))
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)

def rouge_l(reference, hypothesis):
    """ROUGE-L (longest common subsequence) F1 score."""
    ref_words = tokenize_words(reference)
    hyp_words = tokenize_words(hypothesis)
    if not ref_words or not hyp_words:
        return 0.0
    # LCS via DP
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs_len = dp[m][n]
    if lcs_len == 0:
        return 0.0
    precision = lcs_len / n
    recall = lcs_len / m
    return 2 * precision * recall / (precision + recall)

def keyword_recall(ground_truth, response):
    """What fraction of ground truth keywords appear in the response."""
    gt_words = set(tokenize_words(ground_truth))
    resp_words = set(tokenize_words(response))
    if not gt_words:
        return 0.0
    return len(gt_words & resp_words) / len(gt_words)

def is_refusal(text):
    """Detect if response is a refusal / uncertainty."""
    lower = text.lower()
    patterns = ["i don't know", "i do not know", "unable to", "cannot determine",
                "i'm not sure", "no information", "i am not", "not available",
                "i cannot", "unclear", "no data"]
    return any(p in lower for p in patterns)

def repetition_score(text):
    """Ratio of unique words to total (1.0 = all unique, low = repetitive)."""
    words = tokenize_words(text)
    if len(words) < 3:
        return 1.0
    return len(set(words)) / len(words)

# Compute per-sample metrics
print("Computing per-sample metrics...")

for eval_list, set_name in [(results["forget_evaluations"], "forget"), (results["retain_evaluations"], "retain")]:
    for e in eval_list:
        gt = e['ground_truth']
        for model_key, prefix in [("base_model_response", "base"),
                                   ("finetuned_model_response", "ft"),
                                   ("retainonly_model_response", "ro"),
                                   ("unlearned_model_response", "ul")]:
            resp = e[model_key]
            e[f"{prefix}_word_overlap_gt"] = round(word_overlap(gt, resp), 4)
            e[f"{prefix}_rouge_l_gt"] = round(rouge_l(gt, resp), 4)
            e[f"{prefix}_keyword_recall_gt"] = round(keyword_recall(gt, resp), 4)
            e[f"{prefix}_is_refusal"] = is_refusal(resp)
            e[f"{prefix}_repetition"] = round(repetition_score(resp), 4)
            e[f"{prefix}_length"] = len(resp)

        # Cross-model similarity (finetuned vs unlearned)
        e["ft_ul_word_overlap"] = round(word_overlap(e["finetuned_model_response"], e["unlearned_model_response"]), 4)
        e["ft_ul_rouge_l"] = round(rouge_l(e["finetuned_model_response"], e["unlearned_model_response"]), 4)

# Aggregate metrics
def avg(lst):
    return sum(lst) / len(lst) if lst else 0.0

def compute_aggregates(eval_list):
    agg = {}
    for prefix in ["base", "ft", "ro", "ul"]:
        agg[f"{prefix}_word_overlap_gt"] = round(avg([e[f"{prefix}_word_overlap_gt"] for e in eval_list]), 4)
        agg[f"{prefix}_rouge_l_gt"] = round(avg([e[f"{prefix}_rouge_l_gt"] for e in eval_list]), 4)
        agg[f"{prefix}_keyword_recall_gt"] = round(avg([e[f"{prefix}_keyword_recall_gt"] for e in eval_list]), 4)
        agg[f"{prefix}_refusal_rate"] = round(avg([1.0 if e[f"{prefix}_is_refusal"] else 0.0 for e in eval_list]), 4)
        agg[f"{prefix}_avg_length"] = round(avg([e[f"{prefix}_length"] for e in eval_list]), 1)
        agg[f"{prefix}_avg_repetition"] = round(avg([e[f"{prefix}_repetition"] for e in eval_list]), 4)
    agg["ft_ul_word_overlap"] = round(avg([e["ft_ul_word_overlap"] for e in eval_list]), 4)
    agg["ft_ul_rouge_l"] = round(avg([e["ft_ul_rouge_l"] for e in eval_list]), 4)
    return agg

forget_agg = compute_aggregates(results["forget_evaluations"])
retain_agg = compute_aggregates(results["retain_evaluations"])

results["metrics"] = {
    "forget": forget_agg,
    "retain": retain_agg,
}

# Print metrics summary
def print_metrics_table(label, agg):
    print(f"\n{'─'*90}")
    print(f"  {label}")
    print(f"{'─'*90}")
    print(f"  {'Metric':<25} {'Base':>10} {'Finetuned':>10} {'RetainOnly':>10} {'Unlearned':>10}")
    print(f"  {'─'*70}")
    print(f"  {'Word Overlap vs GT':<25} {agg['base_word_overlap_gt']:>10.3f} {agg['ft_word_overlap_gt']:>10.3f} {agg['ro_word_overlap_gt']:>10.3f} {agg['ul_word_overlap_gt']:>10.3f}")
    print(f"  {'ROUGE-L vs GT':<25} {agg['base_rouge_l_gt']:>10.3f} {agg['ft_rouge_l_gt']:>10.3f} {agg['ro_rouge_l_gt']:>10.3f} {agg['ul_rouge_l_gt']:>10.3f}")
    print(f"  {'Keyword Recall vs GT':<25} {agg['base_keyword_recall_gt']:>10.3f} {agg['ft_keyword_recall_gt']:>10.3f} {agg['ro_keyword_recall_gt']:>10.3f} {agg['ul_keyword_recall_gt']:>10.3f}")
    print(f"  {'Refusal Rate':<25} {agg['base_refusal_rate']:>10.1%} {agg['ft_refusal_rate']:>10.1%} {agg['ro_refusal_rate']:>10.1%} {agg['ul_refusal_rate']:>10.1%}")
    print(f"  {'Avg Response Length':<25} {agg['base_avg_length']:>10.0f} {agg['ft_avg_length']:>10.0f} {agg['ro_avg_length']:>10.0f} {agg['ul_avg_length']:>10.0f}")
    print(f"  {'Word Diversity':<25} {agg['base_avg_repetition']:>10.3f} {agg['ft_avg_repetition']:>10.3f} {agg['ro_avg_repetition']:>10.3f} {agg['ul_avg_repetition']:>10.3f}")
    print(f"  {'─'*70}")
    print(f"  {'FT↔UL Word Overlap':<25} {agg['ft_ul_word_overlap']:>10.3f}")
    print(f"  {'FT↔UL ROUGE-L':<25} {agg['ft_ul_rouge_l']:>10.3f}")

print_metrics_table("FORGET SET METRICS (lower similarity = better forgetting)", forget_agg)
print_metrics_table("RETAIN SET METRICS (higher similarity = better retention)", retain_agg)

# Compute unlearning effectiveness scores
print(f"\n{'='*80}")
print("UNLEARNING EFFECTIVENESS SCORES")
print(f"{'='*80}\n")

# Forget Quality: how much did similarity to GT drop from finetuned → unlearned
forget_drop_rouge = forget_agg["ft_rouge_l_gt"] - forget_agg["ul_rouge_l_gt"]
forget_drop_overlap = forget_agg["ft_word_overlap_gt"] - forget_agg["ul_word_overlap_gt"]
forget_drop_keyword = forget_agg["ft_keyword_recall_gt"] - forget_agg["ul_keyword_recall_gt"]

# Retain Quality: how much similarity was preserved from finetuned → unlearned
retain_preserved_rouge = retain_agg["ul_rouge_l_gt"] / max(retain_agg["ft_rouge_l_gt"], 0.001)
retain_preserved_overlap = retain_agg["ul_word_overlap_gt"] / max(retain_agg["ft_word_overlap_gt"], 0.001)

print(f"  Forget Quality (FT→UL drop, higher = better forgetting):")
print(f"    ROUGE-L drop:          {forget_drop_rouge:+.4f}")
print(f"    Word Overlap drop:     {forget_drop_overlap:+.4f}")
print(f"    Keyword Recall drop:   {forget_drop_keyword:+.4f}")
print()
print(f"  Retain Quality (UL/FT ratio, closer to 1.0 = better retention):")
print(f"    ROUGE-L preserved:     {retain_preserved_rouge:.4f}")
print(f"    Word Overlap preserved:{retain_preserved_overlap:.4f}")
print()

# Overall score: balance between forgetting and retaining
forget_score = (forget_drop_rouge + forget_drop_overlap + forget_drop_keyword) / 3
retain_score = (retain_preserved_rouge + retain_preserved_overlap) / 2
overall_score = forget_score * 0.5 + (retain_score) * 0.5

results["metrics"]["scores"] = {
    "forget_quality_rouge_drop": round(forget_drop_rouge, 4),
    "forget_quality_overlap_drop": round(forget_drop_overlap, 4),
    "forget_quality_keyword_drop": round(forget_drop_keyword, 4),
    "retain_quality_rouge_ratio": round(retain_preserved_rouge, 4),
    "retain_quality_overlap_ratio": round(retain_preserved_overlap, 4),
    "forget_score": round(forget_score, 4),
    "retain_score": round(retain_score, 4),
}

print(f"  ┌─────────────────────────────────────┐")
print(f"  │ Forget Score:  {forget_score:>6.4f}              │")
print(f"  │ Retain Score:  {retain_score:>6.4f}              │")
print(f"  └─────────────────────────────────────┘")
print()

print()
print("="*80)
print("STAGE 6: Saving Results")
print("="*80)
print()

# Save full results
output_file = eval_output_dir / "evaluation_results.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✓ Full results saved to: {output_file}")

# Create human-readable report
report_file = eval_output_dir / "evaluation_report.txt"
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("UNLEARNING EVALUATION REPORT\n")
    f.write("="*80 + "\n\n")

    f.write(f"Run Name:        {run_name}\n")
    f.write(f"Timestamp:       {results['timestamp']}\n")
    f.write(f"Base Model:      {base_model_name}\n")
    f.write(f"Unlearned Model: {unlearned_model_path}\n")
    f.write(f"Dataset:         {dataset_name}\n\n")

    # Write metrics summary
    f.write("="*80 + "\n")
    f.write("QUANTITATIVE METRICS\n")
    f.write("="*80 + "\n\n")

    for set_name, agg in [("FORGET SET", forget_agg), ("RETAIN SET", retain_agg)]:
        f.write(f"  {set_name}\n")
        f.write(f"  {'Metric':<25} {'Base':>10} {'Finetuned':>10} {'RetainOnly':>10} {'Unlearned':>10}\n")
        f.write(f"  {'-'*70}\n")
        f.write(f"  {'Word Overlap vs GT':<25} {agg['base_word_overlap_gt']:>10.3f} {agg['ft_word_overlap_gt']:>10.3f} {agg['ro_word_overlap_gt']:>10.3f} {agg['ul_word_overlap_gt']:>10.3f}\n")
        f.write(f"  {'ROUGE-L vs GT':<25} {agg['base_rouge_l_gt']:>10.3f} {agg['ft_rouge_l_gt']:>10.3f} {agg['ro_rouge_l_gt']:>10.3f} {agg['ul_rouge_l_gt']:>10.3f}\n")
        f.write(f"  {'Keyword Recall vs GT':<25} {agg['base_keyword_recall_gt']:>10.3f} {agg['ft_keyword_recall_gt']:>10.3f} {agg['ro_keyword_recall_gt']:>10.3f} {agg['ul_keyword_recall_gt']:>10.3f}\n")
        f.write(f"  {'Refusal Rate':<25} {agg['base_refusal_rate']:>9.1%} {agg['ft_refusal_rate']:>10.1%} {agg['ro_refusal_rate']:>10.1%} {agg['ul_refusal_rate']:>10.1%}\n")
        f.write(f"  {'Avg Response Length':<25} {agg['base_avg_length']:>10.0f} {agg['ft_avg_length']:>10.0f} {agg['ro_avg_length']:>10.0f} {agg['ul_avg_length']:>10.0f}\n")
        f.write(f"  {'Word Diversity':<25} {agg['base_avg_repetition']:>10.3f} {agg['ft_avg_repetition']:>10.3f} {agg['ro_avg_repetition']:>10.3f} {agg['ul_avg_repetition']:>10.3f}\n")
        f.write(f"  {'FT<>UL Word Overlap':<25} {agg['ft_ul_word_overlap']:>10.3f}\n")
        f.write(f"  {'FT<>UL ROUGE-L':<25} {agg['ft_ul_rouge_l']:>10.3f}\n")
        f.write("\n")

    scores = results["metrics"]["scores"]
    f.write(f"  SCORES\n")
    f.write(f"  Forget Score:  {scores['forget_score']:.4f} (higher = better forgetting)\n")
    f.write(f"  Retain Score:  {scores['retain_score']:.4f} (closer to 1.0 = better retention)\n")
    f.write("\n")

    f.write("="*80 + "\n")
    f.write("FORGET SET EVALUATION (Should show degraded performance)\n")
    f.write("="*80 + "\n\n")

    for eval_item in results["forget_evaluations"][:5]:  # Show first 5
        f.write(f"Sample {eval_item['index'] + 1}:\n")
        f.write(f"Question: {eval_item['question']}\n")
        f.write(f"Ground Truth: {eval_item['ground_truth']}\n")
        f.write(f"\n1. Base Model (Pretrained) Response:\n{eval_item['base_model_response']}\n")
        f.write(f"\n2. Finetuned Model Response:\n{eval_item['finetuned_model_response']}\n")
        f.write(f"\n3. Retain-Only Model Response:\n{eval_item['retainonly_model_response']}\n")
        f.write(f"\n4. Unlearned Model Response:\n{eval_item['unlearned_model_response']}\n")
        f.write("\n" + "-"*80 + "\n\n")

    if len(results["forget_evaluations"]) > 5:
        f.write(f"... and {len(results['forget_evaluations']) - 5} more samples\n\n")

    f.write("="*80 + "\n")
    f.write("RETAIN SET EVALUATION (Should maintain performance)\n")
    f.write("="*80 + "\n\n")

    for eval_item in results["retain_evaluations"][:5]:  # Show first 5
        f.write(f"Sample {eval_item['index'] + 1}:\n")
        f.write(f"Question: {eval_item['question']}\n")
        f.write(f"Ground Truth: {eval_item['ground_truth']}\n")
        f.write(f"\n1. Base Model (Pretrained) Response:\n{eval_item['base_model_response']}\n")
        f.write(f"\n2. Finetuned Model Response:\n{eval_item['finetuned_model_response']}\n")
        f.write(f"\n3. Retain-Only Model Response:\n{eval_item['retainonly_model_response']}\n")
        f.write(f"\n4. Unlearned Model Response:\n{eval_item['unlearned_model_response']}\n")
        f.write("\n" + "-"*80 + "\n\n")

    if len(results["retain_evaluations"]) > 5:
        f.write(f"... and {len(results['retain_evaluations']) - 5} more samples\n\n")

print(f"✓ Report saved to: {report_file}")
print()

# Create CSV output (best-effort: never fatal — JSON+report are the canonical artifacts)
csv_file = eval_output_dir / "evaluation_results.csv"
import csv

def _csv_safe(v):
    # Strip control chars that confuse csv (NUL, \r). Replace newlines with space.
    if isinstance(v, str):
        return v.replace("\x00", "").replace("\r", " ").replace("\n", " ")
    return v

try:
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, escapechar='\\')

        # Write header
        writer.writerow([
            "question", "goal", "ground_truth",
            "base_response", "ft_response", "ro_response", "ul_response",
            "base_rouge_l", "ft_rouge_l", "ro_rouge_l", "ul_rouge_l",
            "base_word_overlap", "ft_word_overlap", "ro_word_overlap", "ul_word_overlap",
            "base_keyword_recall", "ft_keyword_recall", "ro_keyword_recall", "ul_keyword_recall",
            "ft_ul_rouge_l", "ft_ul_word_overlap",
            "base_is_refusal", "ft_is_refusal", "ro_is_refusal", "ul_is_refusal",
            "base_length", "ft_length", "ro_length", "ul_length",
        ])

        # Write all samples
        for eval_list, goal in [(results["forget_evaluations"], "forget"), (results["retain_evaluations"], "retain")]:
            for e in eval_list:
                writer.writerow([_csv_safe(x) for x in (
                    e['question'], goal, e['ground_truth'],
                    e['base_model_response'], e['finetuned_model_response'], e['retainonly_model_response'], e['unlearned_model_response'],
                    e['base_rouge_l_gt'], e['ft_rouge_l_gt'], e['ro_rouge_l_gt'], e['ul_rouge_l_gt'],
                    e['base_word_overlap_gt'], e['ft_word_overlap_gt'], e['ro_word_overlap_gt'], e['ul_word_overlap_gt'],
                    e['base_keyword_recall_gt'], e['ft_keyword_recall_gt'], e['ro_keyword_recall_gt'], e['ul_keyword_recall_gt'],
                    e['ft_ul_rouge_l'], e['ft_ul_word_overlap'],
                    e['base_is_refusal'], e['ft_is_refusal'], e['ro_is_refusal'], e['ul_is_refusal'],
                    e['base_length'], e['ft_length'], e['ro_length'], e['ul_length'],
                )])

    print(f"✓ CSV saved to: {csv_file}")
except Exception as csv_exc:
    print(f"⚠ CSV write failed (non-fatal — JSON+report still saved): {csv_exc}")
print()

# Display summary
print("="*80)
print("EVALUATION SUMMARY")
print("="*80)
print()
print(f"Total Forget Samples: {len(results['forget_evaluations'])}")
print(f"Total Retain Samples: {len(results['retain_evaluations'])}")
print()
print("Files created:")
print(f"  1. {output_file} (JSON)")
print(f"  2. {report_file} (TXT)")
print(f"  3. {csv_file} (CSV)")
print()
print("CSV Format:")
print("  Columns: sample, label, goal, ground_truth, pretraining, finetune, unlearn")
print("  - sample: Question/prompt")
print("  - label: train/test (all are 'test' for evaluation)")
print("  - goal: retain/unlearn (forget set = unlearn, retain set = retain)")
print("  - ground_truth: Expected answer")
print("  - pretraining: Base model generation")
print("  - finetune: Finetuned model (empty - not generated)")
print("  - unlearn: Unlearned model generation")
print()
print("Next steps:")
print("  1. Review the CSV: cat saves/eval/${RUN_NAME}/evaluation_results.csv")
print("  2. Export using: bash scripts/export-results.sh ${RUN_NAME} local --eval-only")
print("  3. The exported archive will include all evaluation files")
print()
print("="*80)

EVAL_SCRIPT

echo ""
echo "✅ Comprehensive evaluation complete!"
echo ""
echo "Results saved to: ${EVAL_OUTPUT_DIR}/"
echo ""
