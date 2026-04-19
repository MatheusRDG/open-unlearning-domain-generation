#!/bin/bash

##############################################################################
# Local Mac (MPS) Domain Unlearning Pipeline
#
# Lightweight version of domain-unlearn.sh for running on Apple Silicon Macs.
# Uses MPS backend, smaller batch sizes, and reduced epochs for fast iteration.
#
# Usage:
#   bash scripts/local-unlearn.sh <TOPIC> [MODEL] [TRAINER]
#
# Example:
#   bash scripts/local-unlearn.sh "Juninho"
#   bash scripts/local-unlearn.sh "Brazil" Llama-3.2-1B-Instruct NPO
##############################################################################

set -e

# Parse arguments
SKIP_EVAL=false
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

TOPIC="${POSITIONAL_ARGS[0]:-Juninho}"
MODEL="${POSITIONAL_ARGS[1]:-Llama-3.2-1B-Instruct}"
TRAINER="${POSITIONAL_ARGS[2]:-NPO}"

# Load environment
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATA_DIR="data/run/${TIMESTAMP}"
OUTPUT_DIR="output/${TIMESTAMP}"
DATASET_NAME=$(echo "${TOPIC}" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
RUN_NAME="${DATASET_NAME}_${TIMESTAMP}"

# Local Mac hyperparameters (minimal for MPS memory)
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=16  # Effective batch = 16
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # Allow full MPS memory usage
FINETUNE_EPOCHS=10   # Deeper memorization so unlearning has something to forget
FINETUNE_LR=1e-5
NUM_EPOCHS=7         # More unlearning pressure (was 3, too gentle)
LEARNING_RATE=2e-5   # Slightly more aggressive (was 1e-5)
WARMUP_EPOCHS=1.0
WEIGHT_DECAY=0.01

mkdir -p "${DATA_DIR}" "${OUTPUT_DIR}"

echo "================================================================================================"
echo "Local Mac (MPS) Domain Unlearning Pipeline"
echo "================================================================================================"
echo "Topic:      ${TOPIC}"
echo "Model:      ${MODEL}"
echo "Trainer:    ${TRAINER}"
echo "Run Name:   ${RUN_NAME}"
echo "Batch:      ${PER_DEVICE_BATCH_SIZE} x ${GRADIENT_ACCUMULATION_STEPS} = $((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "Finetune:   ${FINETUNE_EPOCHS} epochs, lr=${FINETUNE_LR}"
echo "Unlearn:    ${NUM_EPOCHS} epochs, lr=${LEARNING_RATE}"
echo "================================================================================================"
echo ""

# System check
uv run python -c "
import torch
print(f'PyTorch {torch.__version__}')
if torch.backends.mps.is_available():
    print(f'MPS: Available (Apple Silicon)')
elif torch.cuda.is_available():
    print(f'CUDA: {torch.cuda.get_device_name(0)}')
else:
    print('WARNING: No GPU detected, training will be very slow')
"
echo ""

##############################################################################
# Step 1: Check for existing dataset
##############################################################################

PREGENERATED_FORGET="data/datasets/${DATASET_NAME}/qa_dataset_forget"
PREGENERATED_RETAIN="data/datasets/${DATASET_NAME}/qa_dataset_retain"

if [ -d "${PREGENERATED_FORGET}" ] && [ -d "${PREGENERATED_RETAIN}" ]; then
    echo "Found pre-generated dataset for '${TOPIC}'"
    FORGET_DATASET_PATH="${PREGENERATED_FORGET}"
    RETAIN_DATASET_PATH="${PREGENERATED_RETAIN}"
    SKIP_GENERATION=true
else
    echo "No pre-generated dataset found for '${TOPIC}'"
    echo "Run domain generation first or use a topic with existing data (juninho, brazil)"
    exit 1
fi

echo "  Forget: $(uv run python -c "from datasets import load_from_disk; print(len(load_from_disk('${FORGET_DATASET_PATH}')))" 2>/dev/null) samples"
echo "  Retain: $(uv run python -c "from datasets import load_from_disk; print(len(load_from_disk('${RETAIN_DATASET_PATH}')))" 2>/dev/null) samples"
echo ""

##############################################################################
# Step 1b: Dataset Quality Analysis
##############################################################################

echo "Running dataset quality analysis..."
uv run python scripts/analyze-dataset.py "data/datasets/${DATASET_NAME}" 2>/dev/null || echo "Warning: Analysis failed (non-critical)"
echo ""

##############################################################################
# Step 2: Create configs
##############################################################################

CONFIG_DIR="configs/data/datasets"
mkdir -p "${CONFIG_DIR}"

# Forget config
cat > "${CONFIG_DIR}/DOMAIN_${DATASET_NAME}_forget.yaml" << EOF
DOMAIN_${DATASET_NAME}_forget:
  handler: QADataset
  args:
    hf_args:
      path: "${FORGET_DATASET_PATH}"
    question_key: "question"
    answer_key: "answer"
    max_length: 512
EOF

# Retain config
cat > "${CONFIG_DIR}/DOMAIN_${DATASET_NAME}_retain.yaml" << EOF
DOMAIN_${DATASET_NAME}_retain:
  handler: QADataset
  args:
    hf_args:
      path: "${RETAIN_DATASET_PATH}"
    question_key: "question"
    answer_key: "answer"
    max_length: 512
EOF

# Combined dataset for finetuning (QA + text passages)
COMBINED_DATASET_PATH="${DATA_DIR}/${DATASET_NAME}/qa_dataset_combined"
TEXT_DATASET_PATH="${FORGET_DATASET_PATH}/../text_dataset_forget"

uv run python -c "
from datasets import load_from_disk, concatenate_datasets

forget_ds = load_from_disk('${FORGET_DATASET_PATH}')
retain_ds = load_from_disk('${RETAIN_DATASET_PATH}')

keep_cols = {'question', 'answer'}
forget_ds = forget_ds.remove_columns([c for c in forget_ds.column_names if c not in keep_cols])
retain_ds = retain_ds.remove_columns([c for c in retain_ds.column_names if c not in keep_cols])

combined = concatenate_datasets([forget_ds, retain_ds]).shuffle(seed=42)
combined.save_to_disk('${COMBINED_DATASET_PATH}')
print(f'Combined: {len(forget_ds)} forget + {len(retain_ds)} retain = {len(combined)} total')
"

cat > "${CONFIG_DIR}/DOMAIN_${DATASET_NAME}_combined.yaml" << EOF
DOMAIN_${DATASET_NAME}_combined:
  handler: QADataset
  args:
    hf_args:
      path: "${COMBINED_DATASET_PATH}"
    question_key: "question"
    answer_key: "answer"
    max_length: 512
EOF

FINETUNE_DATA_CONFIG="configs/data/finetune_${DATASET_NAME}.yaml"
cat > "${FINETUNE_DATA_CONFIG}" << EOF
defaults:
  - datasets@train: DOMAIN_${DATASET_NAME}_combined
  - datasets@eval: null
EOF

# Retain-only finetune data config (for baseline model)
FINETUNE_RETAIN_DATA_CONFIG="configs/data/finetune_${DATASET_NAME}_retain_only.yaml"
cat > "${FINETUNE_RETAIN_DATA_CONFIG}" << EOF
defaults:
  - datasets@train: DOMAIN_${DATASET_NAME}_retain
  - datasets@eval: null
EOF

# Experiment config
EXPERIMENT_CONFIG_DIR="configs/experiment/unlearn/domain"
mkdir -p "${EXPERIMENT_CONFIG_DIR}"

cat > "${EXPERIMENT_CONFIG_DIR}/${DATASET_NAME}.yaml" << EOF
# @package _global_
defaults:
  - override /model: ${MODEL}
  - override /trainer: ${TRAINER}
  - override /collator: DataCollatorForSupervisedDataset
  - override /data: unlearn
  - override /data/datasets@data.forget: DOMAIN_${DATASET_NAME}_forget
  - override /data/datasets@data.retain: DOMAIN_${DATASET_NAME}_retain
  - _self_

data:
  anchor: forget

task_name: ${RUN_NAME}
eval: null
retain_logs_path: null
EOF

echo "Configs created"
echo ""

##############################################################################
# Step 3: HuggingFace auth
##############################################################################

if [ -n "${HUGGINGFACE_TOKEN}" ]; then
    uv run python -c "
from huggingface_hub import login
import os
token = os.getenv('HUGGINGFACE_TOKEN')
if token:
    login(token=token, add_to_git_credential=False)
    print('HuggingFace: authenticated')
" 2>/dev/null
fi

##############################################################################
# Step 4: Finetune
##############################################################################

FINETUNE_NAME="${DATASET_NAME}_finetune_${TIMESTAMP}"

echo "================================================================================================"
echo "Step 1/4: Finetuning on domain data (forget + retain)"
echo "================================================================================================"
echo ""

uv run python src/train.py --config-name=train.yaml \
    model=${MODEL} \
    collator=DataCollatorForSupervisedDataset \
    data=finetune_${DATASET_NAME} \
    task_name=${FINETUNE_NAME} \
    trainer=finetune \
    ~eval \
    trainer.args.output_dir=saves/finetune/${FINETUNE_NAME} \
    trainer.args.num_train_epochs=${FINETUNE_EPOCHS} \
    trainer.args.learning_rate=${FINETUNE_LR} \
    trainer.args.per_device_train_batch_size=${PER_DEVICE_BATCH_SIZE} \
    trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    ++trainer.args.warmup_epochs=1.0 \
    trainer.args.weight_decay=${WEIGHT_DECAY} \
    trainer.args.save_strategy=epoch \
    ++trainer.args.save_total_limit=2 \
    trainer.args.eval_strategy=no \
    trainer.args.do_eval=false \
    ++trainer.args.eval_on_start=false \
    ++trainer.args.load_best_model_at_end=false \
    trainer.args.logging_steps=5 \
    ++trainer.args.logging_first_step=true \
    trainer.args.gradient_checkpointing=true \
    trainer.args.report_to=none \
    trainer.args.bf16=true \
    trainer.args.ddp_find_unused_parameters=false \
    trainer.args.optim=adamw_torch

echo ""
echo "Finetuning complete!"
echo ""

# Find checkpoint
FINETUNE_DIR="saves/finetune/${FINETUNE_NAME}"
LATEST_CHECKPOINT=$(find "${FINETUNE_DIR}" -type d -name "checkpoint-*" 2>/dev/null | sort -V | tail -n1)

if [ -n "$LATEST_CHECKPOINT" ] && [ -f "${LATEST_CHECKPOINT}/config.json" ]; then
    FINETUNE_CHECKPOINT="${LATEST_CHECKPOINT}"
elif [ -f "${FINETUNE_DIR}/config.json" ]; then
    FINETUNE_CHECKPOINT="${FINETUNE_DIR}"
else
    echo "Error: No finetuned model found in ${FINETUNE_DIR}"
    exit 1
fi

echo "Using checkpoint: ${FINETUNE_CHECKPOINT}"
echo ""

##############################################################################
# Step 4b: Finetune retain-only baseline (theoretical max forgetting)
##############################################################################

RETAINONLY_NAME="${DATASET_NAME}_retainonly_${TIMESTAMP}"

echo "================================================================================================"
echo "Step 2/4: Finetuning retain-only baseline"
echo "================================================================================================"
echo "  This model never sees forget data = theoretical ceiling for unlearning"
echo ""

uv run python src/train.py --config-name=train.yaml \
    model=${MODEL} \
    collator=DataCollatorForSupervisedDataset \
    data=finetune_${DATASET_NAME}_retain_only \
    task_name=${RETAINONLY_NAME} \
    trainer=finetune \
    ~eval \
    trainer.args.output_dir=saves/finetune/${RETAINONLY_NAME} \
    trainer.args.num_train_epochs=${FINETUNE_EPOCHS} \
    trainer.args.learning_rate=${FINETUNE_LR} \
    trainer.args.per_device_train_batch_size=${PER_DEVICE_BATCH_SIZE} \
    trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    ++trainer.args.warmup_epochs=1.0 \
    trainer.args.weight_decay=${WEIGHT_DECAY} \
    trainer.args.save_strategy=epoch \
    ++trainer.args.save_total_limit=2 \
    trainer.args.eval_strategy=no \
    trainer.args.do_eval=false \
    ++trainer.args.eval_on_start=false \
    ++trainer.args.load_best_model_at_end=false \
    trainer.args.logging_steps=5 \
    ++trainer.args.logging_first_step=true \
    trainer.args.gradient_checkpointing=true \
    trainer.args.report_to=none \
    trainer.args.bf16=true \
    trainer.args.ddp_find_unused_parameters=false \
    trainer.args.optim=adamw_torch

echo ""
echo "Retain-only finetuning complete!"
echo ""

# Find retain-only checkpoint
RETAINONLY_DIR="saves/finetune/${RETAINONLY_NAME}"
RETAINONLY_CHECKPOINT=$(find "${RETAINONLY_DIR}" -type d -name "checkpoint-*" 2>/dev/null | sort -V | tail -n1)

if [ -n "$RETAINONLY_CHECKPOINT" ] && [ -f "${RETAINONLY_CHECKPOINT}/config.json" ]; then
    echo "Retain-only checkpoint: ${RETAINONLY_CHECKPOINT}"
elif [ -f "${RETAINONLY_DIR}/config.json" ]; then
    RETAINONLY_CHECKPOINT="${RETAINONLY_DIR}"
    echo "Retain-only model: ${RETAINONLY_CHECKPOINT}"
else
    echo "Warning: No retain-only model found, skipping in evaluation"
    RETAINONLY_CHECKPOINT=""
fi
echo ""

##############################################################################
# Step 5: Unlearn
##############################################################################

echo "================================================================================================"
echo "Step 3/4: Unlearning with ${TRAINER}"
echo "================================================================================================"
echo ""

uv run python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/domain/${DATASET_NAME} \
    task_name=${RUN_NAME} \
    model.model_args.pretrained_model_name_or_path=${FINETUNE_CHECKPOINT} \
    trainer.args.num_train_epochs=${NUM_EPOCHS} \
    trainer.args.learning_rate=${LEARNING_RATE} \
    trainer.args.per_device_train_batch_size=${PER_DEVICE_BATCH_SIZE} \
    trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    ++trainer.args.warmup_epochs=${WARMUP_EPOCHS} \
    trainer.args.weight_decay=${WEIGHT_DECAY} \
    trainer.args.save_strategy=epoch \
    ++trainer.args.save_total_limit=3 \
    trainer.args.eval_strategy=no \
    trainer.args.logging_steps=5 \
    ++trainer.args.logging_first_step=true \
    trainer.args.gradient_checkpointing=true \
    ++trainer.args.max_grad_norm=1.0 \
    ++trainer.args.load_best_model_at_end=false \
    trainer.args.ddp_find_unused_parameters=false \
    trainer.args.report_to=none \
    trainer.args.bf16=true \
    trainer.args.optim=adamw_torch

echo ""
echo "Unlearning complete!"
echo ""

##############################################################################
# Step 6: Quick eval
##############################################################################

if [ "${SKIP_EVAL}" != "true" ]; then
    echo "================================================================================================"
    echo "Step 4/4: Evaluation (4 models)"
    echo "================================================================================================"
    echo ""

    BASE_MODEL_PATH=$(grep "pretrained_model_name_or_path" "configs/model/${MODEL}.yaml" | head -1 | cut -d'"' -f2 | tr -d '\n\r')
    if [ -z "$BASE_MODEL_PATH" ]; then
        BASE_MODEL_PATH="meta-llama/${MODEL}"
    fi

    bash scripts/evaluate-unlearning.sh "${RUN_NAME}" "${BASE_MODEL_PATH}" "${FINETUNE_CHECKPOINT}" "${RETAINONLY_CHECKPOINT}"
else
    echo "Skipping evaluation (--skip-eval)"
fi

##############################################################################
# Save loss curves to eval folder for paper plotting
##############################################################################

EVAL_DIR="saves/eval/${RUN_NAME}"
mkdir -p "${EVAL_DIR}"

echo ""
echo "Saving loss curves..."

uv run python -c "
import json, csv
from pathlib import Path

eval_dir = Path('${EVAL_DIR}')
losses = {}

# Extract losses from each training run
for name, state_dir in [
    ('finetune', 'saves/finetune/${FINETUNE_NAME}'),
    ('retainonly', 'saves/finetune/${RETAINONLY_NAME}'),
    ('unlearn', 'saves/unlearn/${RUN_NAME}'),
]:
    state_file = Path(state_dir) / 'trainer_state.json'
    if not state_file.exists():
        print(f'  Skip {name}: no trainer_state.json')
        continue
    state = json.load(open(state_file))
    entries = []
    for entry in state.get('log_history', []):
        if 'loss' in entry:
            row = {'step': entry.get('step', 0), 'epoch': entry.get('epoch', 0), 'loss': entry['loss']}
            # Include unlearning-specific losses if present
            for key in ['forget_loss_dpo', 'retain_loss', 'total_loss',
                        'forget_loss_original', 'forget_loss_negated']:
                if key in entry:
                    row[key] = entry[key]
            entries.append(row)
    losses[name] = entries
    print(f'  {name}: {len(entries)} logged steps')

# Save combined JSON
with open(eval_dir / 'loss_curves.json', 'w') as f:
    json.dump(losses, f, indent=2)

# Save per-run CSVs for easy plotting
for name, entries in losses.items():
    if not entries:
        continue
    csv_file = eval_dir / f'loss_{name}.csv'
    keys = list(entries[0].keys())
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(entries)
    print(f'  Saved: {csv_file}')

print('  Saved: ' + str(eval_dir / 'loss_curves.json'))
"

# Copy data quality metrics to eval folder
if [ -f "data/datasets/${DATASET_NAME}/data_quality_metrics.json" ]; then
    cp "data/datasets/${DATASET_NAME}/data_quality_metrics.json" "${EVAL_DIR}/data_quality_metrics.json"
    echo "Copied data quality metrics to eval folder"
fi

echo ""
echo "================================================================================================"
echo "Done! All results in: saves/eval/${RUN_NAME}/"
echo "================================================================================================"
echo ""
echo "Files:"
echo "  evaluation_results.json    - Full responses + per-sample metrics"
echo "  evaluation_results.csv     - Tabular results with metrics"
echo "  evaluation_report.txt      - Human-readable report"
echo "  data_quality_metrics.json  - Dataset entanglement, diversity, specificity"
echo "  loss_curves.json           - All training losses (finetune + retainonly + unlearn)"
echo "  loss_finetune.csv          - Finetune loss curve"
echo "  loss_retainonly.csv        - Retain-only finetune loss curve"
echo "  loss_unlearn.csv           - Unlearning loss curve (with forget/retain breakdown)"
echo "================================================================================================"
