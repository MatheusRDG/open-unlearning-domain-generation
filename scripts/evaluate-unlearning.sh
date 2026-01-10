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

if [ -z "$RUN_NAME" ]; then
    echo "Error: RUN_NAME required"
    echo "Usage: bash scripts/evaluate-unlearning.sh <RUN_NAME> <BASE_MODEL>"
    echo ""
    echo "Example:"
    echo "  bash scripts/evaluate-unlearning.sh brazil_20260110_174240 meta-llama/Llama-3.2-1B-Instruct"
    exit 1
fi

CHECKPOINT_DIR="saves/unlearn/${RUN_NAME}"
EVAL_OUTPUT_DIR="saves/eval/${RUN_NAME}"

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                    Comprehensive Unlearning Evaluation                     ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Run Name:        ${RUN_NAME}"
echo "Base Model:      ${BASE_MODEL}"
echo "Unlearned Model: ${CHECKPOINT_DIR}"
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

uv run python << 'EVAL_SCRIPT'
import json
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from datetime import datetime

# Configuration
run_name = "${RUN_NAME}"
base_model_name = "${BASE_MODEL}"
checkpoint_dir = Path("${CHECKPOINT_DIR}")
eval_output_dir = Path("${EVAL_OUTPUT_DIR}")

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

print(f"Base Model:      {base_model_name}")
print(f"Unlearned Model: {unlearned_model_path}")
print()

# Extract dataset name from run name
dataset_name = run_name.split('_')[0]  # e.g., "brazil" from "brazil_20260110_174240"

# Load datasets
print("Loading datasets...")
forget_dataset_path = f"data/datasets/{dataset_name}/qa_dataset_forget"
retain_dataset_path = f"data/datasets/{dataset_name}/qa_dataset_retain"

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
            top_p=None
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the question from response
    response = response[len(question):].strip()
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
print("STAGE 1: Loading Base Model")
print("="*80)
print()

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
print("="*80)
print("STAGE 2: Loading Unlearned Model")
print("="*80)
print()

try:
    unlearned_tokenizer = AutoTokenizer.from_pretrained(unlearned_model_path)
    unlearned_model = AutoModelForCausalLM.from_pretrained(
        unlearned_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print("✓ Unlearned model loaded")
except Exception as e:
    print(f"✗ Error loading unlearned model: {e}")
    sys.exit(1)

print()
print("="*80)
print("STAGE 3: Evaluating Forget Set")
print("="*80)
print()

# Evaluate forget samples
print(f"Generating responses for {len(forget_dataset)} forget samples...")
for idx, sample in enumerate(tqdm(forget_dataset, desc="Forget set")):
    question = sample['question']
    ground_truth = sample['answer']

    # Generate from base model
    base_response = generate_response(base_model, base_tokenizer, question)

    # Generate from unlearned model
    unlearned_response = generate_response(unlearned_model, unlearned_tokenizer, question)

    results["forget_evaluations"].append({
        "index": idx,
        "question": question,
        "ground_truth": ground_truth,
        "base_model_response": base_response,
        "unlearned_model_response": unlearned_response
    })

print()
print("="*80)
print("STAGE 4: Evaluating Retain Set")
print("="*80)
print()

# Evaluate retain samples
print(f"Generating responses for {len(retain_dataset)} retain samples...")
for idx, sample in enumerate(tqdm(retain_dataset, desc="Retain set")):
    question = sample['question']
    ground_truth = sample['answer']

    # Generate from base model
    base_response = generate_response(base_model, base_tokenizer, question)

    # Generate from unlearned model
    unlearned_response = generate_response(unlearned_model, unlearned_tokenizer, question)

    results["retain_evaluations"].append({
        "index": idx,
        "question": question,
        "ground_truth": ground_truth,
        "base_model_response": base_response,
        "unlearned_model_response": unlearned_response
    })

print()
print("="*80)
print("STAGE 5: Saving Results")
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

    f.write("="*80 + "\n")
    f.write("FORGET SET EVALUATION (Should show degraded performance)\n")
    f.write("="*80 + "\n\n")

    for eval_item in results["forget_evaluations"][:5]:  # Show first 5
        f.write(f"Sample {eval_item['index'] + 1}:\n")
        f.write(f"Question: {eval_item['question']}\n")
        f.write(f"Ground Truth: {eval_item['ground_truth']}\n")
        f.write(f"\nBase Model Response:\n{eval_item['base_model_response']}\n")
        f.write(f"\nUnlearned Model Response:\n{eval_item['unlearned_model_response']}\n")
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
        f.write(f"\nBase Model Response:\n{eval_item['base_model_response']}\n")
        f.write(f"\nUnlearned Model Response:\n{eval_item['unlearned_model_response']}\n")
        f.write("\n" + "-"*80 + "\n\n")

    if len(results["retain_evaluations"]) > 5:
        f.write(f"... and {len(results['retain_evaluations']) - 5} more samples\n\n")

print(f"✓ Report saved to: {report_file}")
print()

# Create CSV output
csv_file = eval_output_dir / "evaluation_results.csv"
import csv

with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)

    # Write header
    writer.writerow([
        "sample",
        "label",
        "goal",
        "ground_truth",
        "pretraining",
        "finetune",
        "unlearn"
    ])

    # Write forget samples
    for eval_item in results["forget_evaluations"]:
        writer.writerow([
            eval_item['question'],
            "test",  # All samples are test samples
            "unlearn",  # Forget set is for unlearning
            eval_item['ground_truth'],
            eval_item['base_model_response'],
            "",  # No finetuned model available (could add if needed)
            eval_item['unlearned_model_response']
        ])

    # Write retain samples
    for eval_item in results["retain_evaluations"]:
        writer.writerow([
            eval_item['question'],
            "test",
            "retain",  # Retain set should be retained
            eval_item['ground_truth'],
            eval_item['base_model_response'],
            "",  # No finetuned model available
            eval_item['unlearned_model_response']
        ])

print(f"✓ CSV saved to: {csv_file}")
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
