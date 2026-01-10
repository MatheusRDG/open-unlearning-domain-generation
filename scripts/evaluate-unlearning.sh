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

if [ -z "$RUN_NAME" ]; then
    echo "Error: RUN_NAME required"
    echo "Usage: bash scripts/evaluate-unlearning.sh <RUN_NAME> <BASE_MODEL> [FINETUNE_CHECKPOINT]"
    echo ""
    echo "Example:"
    echo "  bash scripts/evaluate-unlearning.sh brazil_20260110_174240 meta-llama/Llama-3.2-1B-Instruct saves/finetune/brazil_finetune_20260110_174240"
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

uv run python - "$RUN_NAME" "$BASE_MODEL" "$FINETUNE_CHECKPOINT" "$CHECKPOINT_DIR" "$EVAL_OUTPUT_DIR" << 'EVAL_SCRIPT'
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
print(f"Finetuned Model: {finetune_model_path if finetune_model_path else 'Not available'}")
print(f"Unlearned Model: {unlearned_model_path}")
print()

# Extract dataset name from run name correctly (handle multi-word topics with underscores)
# run_name format: {dataset}_{YYYYMMDD}_{HHMMSS}
# Split from right to avoid breaking dataset names with underscores
parts = run_name.rsplit('_', 2)
if len(parts) == 3:
    dataset_name = parts[0]  # Everything before the timestamp
else:
    dataset_name = run_name.split('_')[0]  # Fallback

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
        "unlearned_model_response": unlearned_retain_responses[idx]
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
        f.write(f"\n1. Base Model (Pretrained) Response:\n{eval_item['base_model_response']}\n")
        f.write(f"\n2. Finetuned Model Response:\n{eval_item['finetuned_model_response']}\n")
        f.write(f"\n3. Unlearned Model Response:\n{eval_item['unlearned_model_response']}\n")
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
        f.write(f"\n3. Unlearned Model Response:\n{eval_item['unlearned_model_response']}\n")
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
            eval_item['finetuned_model_response'],
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
            eval_item['finetuned_model_response'],
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
