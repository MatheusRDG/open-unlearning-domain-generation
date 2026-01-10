#!/bin/bash

##############################################################################
# Domain Unlearning Pipeline
#
# This script performs end-to-end domain unlearning:
# 1. Generates domain content (books, articles, QA) for a specified topic
# 2. Converts the generated content to HuggingFace dataset format
# 3. Runs unlearning on a specified model
# 4. Evaluates the unlearned model
#
# Usage:
#   bash scripts/domain-unlearn.sh <TOPIC> [MODEL] [TRAINER]
#
# Example:
#   bash scripts/domain-unlearn.sh "Brazil"
#   bash scripts/domain-unlearn.sh "USA History" Llama-3.2-3B-Instruct GradAscent
#   bash scripts/domain-unlearn.sh "Mexican Food" Llama-3.1-8B-Instruct NPO
##############################################################################

set -e  # Exit on error

# Parse command-line arguments
TOPIC="${1:-Brazil}"
MODEL="${2:-Llama-3.2-1B-Instruct}"
TRAINER="${3:-GradAscent}"

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="output/${TIMESTAMP}"
DATA_DIR="data/run/${TIMESTAMP}"
DATASET_NAME=$(echo "${TOPIC}" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
RUN_NAME="${DATASET_NAME}_${TIMESTAMP}"

# Training hyperparameters (Full training configuration)
PER_DEVICE_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=8  # Effective batch size = 32
NUM_EPOCHS=20  # Full training overnight
LEARNING_RATE=1e-5
WARMUP_EPOCHS=2.0
WEIGHT_DECAY=0.01

# Create directories
mkdir -p "${DATA_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "================================================================================================"
echo "Domain Unlearning Pipeline - FULL TRAINING"
echo "================================================================================================"
echo "Topic:                ${TOPIC}"
echo "Model:                ${MODEL}"
echo "Trainer:              ${TRAINER}"
echo "Dataset Name:         ${DATASET_NAME}"
echo "Run Name:             ${RUN_NAME}"
echo "Output Directory:     ${OUTPUT_DIR}"
echo "Data Directory:       ${DATA_DIR}"
echo "Timestamp:            ${TIMESTAMP}"
echo ""
echo "Training Configuration:"
echo "  Epochs:             ${NUM_EPOCHS}"
echo "  Batch Size:         ${PER_DEVICE_BATCH_SIZE}"
echo "  Gradient Accum:     ${GRADIENT_ACCUMULATION_STEPS}"
echo "  Effective Batch:    $((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "  Learning Rate:      ${LEARNING_RATE}"
echo "  Warmup Epochs:      ${WARMUP_EPOCHS}"
echo "  Weight Decay:       ${WEIGHT_DECAY}"
echo "  Save Every:         0.5 epochs (keep last 5)"
echo "================================================================================================"
echo ""

##############################################################################
# System Check
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "System Check"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

uv run python -c "
import sys
import torch

print('=' * 80)
print('System Information')
print('=' * 80)
print(f'Python Version:       {sys.version.split()[0]}')
print(f'PyTorch Version:      {torch.__version__}')
print(f'CUDA Available:       {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA Version:         {torch.version.cuda}')
    print(f'GPU Count:            {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}:                {torch.cuda.get_device_name(i)}')
        print(f'  Memory Total:       {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB')
else:
    print('⚠️  WARNING: CUDA not available! Training will be very slow on CPU.')

print('=' * 80)
"

echo ""

##############################################################################
# Step 1: Check for Existing Dataset (Skip Generation if Found)
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Checking for Existing Dataset for '${TOPIC}'"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if pre-generated dataset exists
PREGENERATED_DATASET="data/datasets/${DATASET_NAME}/qa_dataset"
if [ -d "${PREGENERATED_DATASET}" ] && ([ -f "${PREGENERATED_DATASET}/train-00000-of-00001.parquet" ] || [ -f "${PREGENERATED_DATASET}/dataset_info.json" ]); then
    echo "✅ Found pre-generated dataset for '${TOPIC}'"
    echo "   Using: ${PREGENERATED_DATASET}"
    echo ""
    echo "Skipping domain generation (dataset already exists)"
    SKIP_GENERATION=true
    
    # Still create output directory and copy domain.json reference
    mkdir -p "${OUTPUT_DIR}"
    echo "{\"source\": \"pre-generated\", \"dataset\": \"${PREGENERATED_DATASET}\"}" > "${OUTPUT_DIR}/domain.json"
else
    echo "No pre-generated dataset found"
    SKIP_GENERATION=false
fi

echo ""

if [ "$SKIP_GENERATION" = false ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Step 2: Generating Domain Content for '${TOPIC}'"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
else
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Step 2: Skipping Domain Generation (using existing dataset)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
fi

if [ "$SKIP_GENERATION" = false ]; then
    ##############################################################################
    # Generate Domain Content (only if not skipped)
    ##############################################################################

# Setup checkpoint directory
CHECKPOINT_DIR=".logs/generations/${DATASET_NAME}"
CHECKPOINT_FILE="${CHECKPOINT_DIR}/domain.json"
mkdir -p "${CHECKPOINT_DIR}"

# Check if generation checkpoint exists
if [ -f "${CHECKPOINT_FILE}" ]; then
    echo "✅ Found existing generation for '${TOPIC}' in checkpoint"
    echo "   Reusing: ${CHECKPOINT_FILE}"
    echo ""

    # Copy checkpoint to output directory
    mkdir -p "${OUTPUT_DIR}"
    cp "${CHECKPOINT_FILE}" "${OUTPUT_DIR}/domain.json"

    echo "✅ Domain generation reused from checkpoint!"
    echo ""
else
    echo "No checkpoint found. Generating new content..."
    echo ""

    # Modify domain generation to use specified topic
    uv run python -c "
import sys
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Import domain generation modules
from src.domain_generation.config import config
from src.domain_generation.graphs import build_domain_graph
from src.domain_generation.utils import logger
import os

# Verify API key is loaded
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    logger.error('OPENAI_API_KEY not found in environment!')
    sys.exit(1)
else:
    # Show masked API key for verification
    masked_key = api_key[:8] + '...' + api_key[-4:] if len(api_key) > 12 else '***'
    logger.info(f'Using OpenAI API key: {masked_key}')

# Configuration
domain_name = '${TOPIC}'
domain_description = f'Knowledge and information about {domain_name}'
output_dir = Path('${OUTPUT_DIR}')
output_dir.mkdir(exist_ok=True, parents=True)

logger.info('='*80)
logger.info('Domain Content Generation')
logger.info('='*80)
logger.info(f'Domain: {domain_name}')
logger.info(f'Description: {domain_description}')
logger.info(f'Model: {config.model_name}')
logger.info(f'Output: {output_dir}')
logger.info('='*80)

# Build and run domain graph
logger.info('Building domain generation graph...')
domain_graph = build_domain_graph()

logger.info('Starting domain generation...')
result = domain_graph.invoke({
    'name': domain_name,
    'description': domain_description,
})

domain = result['domain']

# Log results
logger.info('='*80)
logger.info('Generation Complete!')
logger.info('='*80)
logger.info(f'Topics: {len(domain.topics)}')
for topic in domain.topics:
    logger.info(f'  - {topic.name}')
logger.info(f'Books: {len(domain.books)}')
logger.info(f'Articles: {len(domain.articles)}')

# Count QA pairs
total_grounded_qa = sum(len(book.grounded_questions) for book in domain.books)
total_grounded_qa += sum(len(article.grounded_questions) for article in domain.articles)
logger.info(f'Total Grounded QA Pairs: {total_grounded_qa}')

# Save outputs
output_file = output_dir / 'domain.json'
checkpoint_file = Path('${CHECKPOINT_FILE}')
domain_dict = domain.model_dump()

# Save to output directory
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(domain_dict, f, indent=2, ensure_ascii=False)

logger.success(f'✅ Saved domain JSON to {output_file}')

# Save to checkpoint directory for future reuse
checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
with open(checkpoint_file, 'w', encoding='utf-8') as f:
    json.dump(domain_dict, f, indent=2, ensure_ascii=False)

logger.success(f'💾 Saved checkpoint to {checkpoint_file}')
logger.info('='*80)
"

    echo ""
    echo "✅ Domain generation complete!"
    echo ""
fi
fi  # End of SKIP_GENERATION condition

echo ""

##############################################################################
# Step 3: Convert to HuggingFace Dataset Format
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: Converting to HuggingFace Dataset Format"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Only convert if we didn't skip generation OR if we have a valid domain.json
if [ -f "${OUTPUT_DIR}/domain.json" ] && grep -q "\"topics\"" "${OUTPUT_DIR}/domain.json" 2>/dev/null; then
    # Generated new domain, need to convert
    uv run python -m src.domain_generation.convert_to_dataset \
        "${OUTPUT_DIR}/domain.json" \
        --output-dir "${DATA_DIR}" \
        --dataset-name "${DATASET_NAME}" \
        --split-ratio 0.8
    
    echo ""
    echo "✅ Dataset conversion complete!"
    echo ""
else
    echo "✅ Using pre-generated dataset (skipped conversion)"
    echo ""
fi

##############################################################################
# Step 4: Create Dataset Config Files
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4: Creating Dataset Configuration Files"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Create config directory for domain datasets
CONFIG_DIR="configs/data/datasets"
mkdir -p "${CONFIG_DIR}"

# Use pre-generated dataset paths if they exist, otherwise use runtime-generated paths
if [ -d "data/datasets/${DATASET_NAME}/qa_dataset_forget" ]; then
    # Pre-generated dataset exists in repo
    FORGET_DATASET_PATH="data/datasets/${DATASET_NAME}/qa_dataset_forget"
    RETAIN_DATASET_PATH="data/datasets/${DATASET_NAME}/qa_dataset_retain"
else
    # Runtime-generated dataset 
    FORGET_DATASET_PATH="${DATA_DIR}/${DATASET_NAME}/qa_dataset_forget"
    RETAIN_DATASET_PATH="${DATA_DIR}/${DATASET_NAME}/qa_dataset_retain"
fi

# Create forget dataset config (QA format)
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

echo "Created: ${CONFIG_DIR}/DOMAIN_${DATASET_NAME}_forget.yaml"

# Create retain dataset config (QA format)
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

echo "Created: ${CONFIG_DIR}/DOMAIN_${DATASET_NAME}_retain.yaml"

echo ""
echo "✅ Dataset configuration files created!"
echo ""

##############################################################################
# Step 5: Create Experiment Config
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 5: Creating Experiment Configuration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Create experiment config directory
EXPERIMENT_CONFIG_DIR="configs/experiment/unlearn/domain"
mkdir -p "${EXPERIMENT_CONFIG_DIR}"

# Create experiment config
cat > "${EXPERIMENT_CONFIG_DIR}/${DATASET_NAME}.yaml" << EOF
# @package _global_

# Domain Unlearning Experiment: ${TOPIC}
# Generated: ${TIMESTAMP}

defaults:
  - override /model: ${MODEL}
  - override /trainer: ${TRAINER}
  - override /collator: DataCollatorForSupervisedDataset
  - override /data: unlearn
  - override /data/datasets@data.forget: DOMAIN_${DATASET_NAME}_forget
  - override /data/datasets@data.retain: DOMAIN_${DATASET_NAME}_retain
  - _self_

# Data configuration
data:
  anchor: forget

# Task name
task_name: ${RUN_NAME}

# Evaluation configuration (optional)
eval: null
retain_logs_path: null
EOF

echo "Created: ${EXPERIMENT_CONFIG_DIR}/${DATASET_NAME}.yaml"
echo ""
echo "✅ Experiment configuration created!"
echo ""

##############################################################################
# Step 6: HuggingFace Authentication
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 6: Authenticating with HuggingFace"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if HuggingFace token is available
if [ -f .env ]; then
    source .env
    if [ -n "${HUGGINGFACE_TOKEN}" ]; then
        echo "Logging in to HuggingFace..."
        echo "${HUGGINGFACE_TOKEN}" | uv run huggingface-cli login --token "${HUGGINGFACE_TOKEN}" --add-to-git-credential
        echo "✅ HuggingFace authentication complete!"
    else
        echo "⚠️  Warning: HUGGINGFACE_TOKEN not found in .env"
        echo "   Some models may not be accessible without authentication"
    fi
else
    echo "⚠️  Warning: .env file not found"
    echo "   Some models may not be accessible without authentication"
fi

echo ""

##############################################################################
# Step 7: Run Unlearning
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 7: Running Unlearning with ${TRAINER}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Force single GPU to avoid distributed training issues
echo "Forcing single GPU mode (GPU 0)..."
export CUDA_VISIBLE_DEVICES=0

# Set master port for distributed training (in case it's still used)
export MASTER_PORT=$(uv run python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: ${MASTER_PORT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo ""

# Run unlearning (Full Training Configuration)
uv run python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/domain/${DATASET_NAME} \
    task_name=${RUN_NAME} \
    trainer.args.num_train_epochs=${NUM_EPOCHS} \
    trainer.args.learning_rate=${LEARNING_RATE} \
    trainer.args.per_device_train_batch_size=${PER_DEVICE_BATCH_SIZE} \
    trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    +trainer.args.warmup_epochs=${WARMUP_EPOCHS} \
    trainer.args.weight_decay=${WEIGHT_DECAY} \
    trainer.args.save_strategy=steps \
    +trainer.args.save_steps=0.5 \
    +trainer.args.save_total_limit=5 \
    trainer.args.eval_strategy=no \
    trainer.args.logging_steps=1 \
    +trainer.args.logging_first_step=true \
    +trainer.args.dataloader_num_workers=0 \
    trainer.args.ddp_find_unused_parameters=false \
    trainer.args.gradient_checkpointing=true \
    +trainer.args.load_best_model_at_end=false \
    +trainer.args.metric_for_best_model=loss \
    trainer.args.report_to=tensorboard

echo ""
echo "✅ Unlearning complete!"
echo ""

##############################################################################
# Step 8: Save Run Summary
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 8: Saving Run Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Create run summary
cat > "${DATA_DIR}/run_summary.json" << EOF
{
  "topic": "${TOPIC}",
  "dataset_name": "${DATASET_NAME}",
  "run_name": "${RUN_NAME}",
  "timestamp": "${TIMESTAMP}",
  "model": "${MODEL}",
  "trainer": "${TRAINER}",
  "hyperparameters": {
    "num_epochs": ${NUM_EPOCHS},
    "learning_rate": ${LEARNING_RATE},
    "per_device_batch_size": ${PER_DEVICE_BATCH_SIZE},
    "gradient_accumulation_steps": ${GRADIENT_ACCUMULATION_STEPS}
  },
  "paths": {
    "domain_json": "${OUTPUT_DIR}/domain.json",
    "data_dir": "${DATA_DIR}",
    "qa_dataset_forget": "${DATA_DIR}/${DATASET_NAME}/qa_dataset_forget",
    "qa_dataset_retain": "${DATA_DIR}/${DATASET_NAME}/qa_dataset_retain",
    "text_dataset_forget": "${DATA_DIR}/${DATASET_NAME}/text_dataset_forget",
    "text_dataset_retain": "${DATA_DIR}/${DATASET_NAME}/text_dataset_retain",
    "model_checkpoint": "saves/unlearn/${RUN_NAME}",
    "experiment_config": "${EXPERIMENT_CONFIG_DIR}/${DATASET_NAME}.yaml"
  }
}
EOF

echo "Created: ${DATA_DIR}/run_summary.json"
echo ""

##############################################################################
# Final Summary
##############################################################################

echo "================================================================================================"
echo "Domain Unlearning Pipeline Complete! 🎉"
echo "================================================================================================"
echo ""
echo "Summary:"
echo "  Topic:                ${TOPIC}"
echo "  Dataset:              ${DATASET_NAME}"
echo "  Model:                ${MODEL}"
echo "  Trainer:              ${TRAINER}"
echo "  Run Name:             ${RUN_NAME}"
echo ""
echo "Generated Artifacts:"
echo "  📄 Domain JSON:       ${OUTPUT_DIR}/domain.json"
echo "  📦 QA Forget Dataset: ${DATA_DIR}/${DATASET_NAME}/qa_dataset_forget"
echo "  📦 QA Retain Dataset: ${DATA_DIR}/${DATASET_NAME}/qa_dataset_retain"
echo "  📦 Text Forget Dataset: ${DATA_DIR}/${DATASET_NAME}/text_dataset_forget"
echo "  📦 Text Retain Dataset: ${DATA_DIR}/${DATASET_NAME}/text_dataset_retain"
echo "  🧠 Model Checkpoint:  saves/unlearn/${RUN_NAME}"
echo "  📋 Run Summary:       ${DATA_DIR}/run_summary.json"
echo ""
echo "Next Steps:"
echo "  1. Evaluate the unlearned model:"
echo "     uv run python src/eval.py \\"
echo "       model=${MODEL} \\"
echo "       model.model_args.pretrained_model_name_or_path=saves/unlearn/${RUN_NAME} \\"
echo "       task_name=${RUN_NAME}_eval"
echo ""
echo "  2. Test the model with queries about '${TOPIC}' to verify unlearning"
echo ""
echo "  3. Compare with baseline model to measure forget quality"
echo ""
echo "================================================================================================"
