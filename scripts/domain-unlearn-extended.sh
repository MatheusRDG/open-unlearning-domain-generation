#!/bin/bash

##############################################################################
# Domain Unlearning Pipeline - EXTENDED GENERATION
#
# This script performs end-to-end domain unlearning with LARGE dataset generation:
# 1. Generates MORE domain content (5-10 topics, 15-25 QA pairs per item)
# 2. Converts the generated content to HuggingFace dataset format
# 3. Fine-tunes the model on the domain data (creates "finetuned" model)
# 4. Runs unlearning on the finetuned model (creates "unlearned" model)
# 5. Evaluates all 3 models: raw, finetuned, unlearned
#
# Usage:
#   bash scripts/domain-unlearn-extended.sh <TOPIC> [MODEL] [TRAINER] [--from-scratch]
#
# Example:
#   bash scripts/domain-unlearn-extended.sh "Brazil"
#   bash scripts/domain-unlearn-extended.sh "USA History" Llama-3.2-3B-Instruct
#   bash scripts/domain-unlearn-extended.sh "Brazil" Llama-3.2-1B-Instruct GradAscent --from-scratch
#
# Expected output: ~500-1000 QA pairs (vs ~95 in regular script)
# Generation time: ~30-60 minutes
# Training time: ~3-4 hours (fine-tuning) + ~2-3 hours (unlearning)
##############################################################################

set -e  # Exit on error

##############################################################################
# EXTENDED GENERATION CONFIGURATION
##############################################################################

# Set environment variables for larger content generation
export GEN_TOPICS_MIN_ITEMS=5           # Was 2
export GEN_TOPICS_MAX_ITEMS=10          # Was 5
export GEN_ARTICLES_MIN_PER_TOPIC=5     # Was 2
export GEN_ARTICLES_MAX_PER_TOPIC=8     # Was 2
export GEN_TOC_MIN_ITEMS=4              # Was 2
export GEN_TOC_MAX_ITEMS=6              # Was 4
export GEN_SECTIONS_MIN_PER_CHAPTER=3   # Was 2
export GEN_SECTIONS_MAX_PER_CHAPTER=5   # Was 4
export GEN_GROUNDED_QA_MIN_ITEMS=15     # Was 5
export GEN_GROUNDED_QA_MAX_ITEMS=25     # Was 10
export GEN_UNGROUNDED_QA_MIN_ITEMS=5    # Was 3
export GEN_UNGROUNDED_QA_MAX_ITEMS=10   # Was 5

##############################################################################

# Parse command-line arguments
TOPIC="${1:-Brazil}"
MODEL="${2:-Llama-3.2-1B-Instruct}"
TRAINER="${3:-GradAscent}"

# Check for --from-scratch flag
FROM_SCRATCH=false
for arg in "$@"; do
    if [ "$arg" == "--from-scratch" ]; then
        FROM_SCRATCH=true
    fi
done

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="output/${TIMESTAMP}"
DATA_DIR="data/run/${TIMESTAMP}"
DATASET_NAME=$(echo "${TOPIC}" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
RUN_NAME="${DATASET_NAME}_${TIMESTAMP}"

# Fine-tuning hyperparameters
FINETUNE_EPOCHS=10
FINETUNE_LEARNING_RATE=2e-5
FINETUNE_BATCH_SIZE=4
FINETUNE_GRADIENT_ACCUMULATION=4  # Effective batch size = 16

# Unlearning hyperparameters
UNLEARN_EPOCHS=10
UNLEARN_LEARNING_RATE=1e-5
UNLEARN_BATCH_SIZE=4
UNLEARN_GRADIENT_ACCUMULATION=8  # Effective batch size = 32

# Common hyperparameters
WARMUP_EPOCHS=1.0
WEIGHT_DECAY=0.01

# Create directories
mkdir -p "${DATA_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "================================================================================================"
echo "Domain Unlearning Pipeline - EXTENDED GENERATION"
echo "================================================================================================"
echo "GENERATION SETTINGS (EXTENDED):"
echo "  Topics:             ${GEN_TOPICS_MIN_ITEMS}-${GEN_TOPICS_MAX_ITEMS} (was 2-5)"
echo "  Articles/Topic:     ${GEN_ARTICLES_MIN_PER_TOPIC}-${GEN_ARTICLES_MAX_PER_TOPIC} (was 2)"
echo "  Chapters/Book:      ${GEN_TOC_MIN_ITEMS}-${GEN_TOC_MAX_ITEMS} (was 2-4)"
echo "  Sections/Chapter:   ${GEN_SECTIONS_MIN_PER_CHAPTER}-${GEN_SECTIONS_MAX_PER_CHAPTER} (was 2-4)"
echo "  QA Pairs/Item:      ${GEN_GROUNDED_QA_MIN_ITEMS}-${GEN_GROUNDED_QA_MAX_ITEMS} (was 5-10)"
echo "  Expected Dataset:   ~500-1000 QA pairs"
echo "  Generation Time:    ~30-60 minutes"
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
echo "Fine-tuning Configuration:"
echo "  Epochs:             ${FINETUNE_EPOCHS}"
echo "  Batch Size:         ${FINETUNE_BATCH_SIZE}"
echo "  Gradient Accum:     ${FINETUNE_GRADIENT_ACCUMULATION}"
echo "  Effective Batch:    $((FINETUNE_BATCH_SIZE * FINETUNE_GRADIENT_ACCUMULATION))"
echo "  Learning Rate:      ${FINETUNE_LEARNING_RATE}"
echo ""
echo "Unlearning Configuration:"
echo "  Epochs:             ${UNLEARN_EPOCHS}"
echo "  Batch Size:         ${UNLEARN_BATCH_SIZE}"
echo "  Gradient Accum:     ${UNLEARN_GRADIENT_ACCUMULATION}"
echo "  Effective Batch:    $((UNLEARN_BATCH_SIZE * UNLEARN_GRADIENT_ACCUMULATION))"
echo "  Learning Rate:      ${UNLEARN_LEARNING_RATE}"
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
# Step 1: Generate Domain Content
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Generating Domain Content for '${TOPIC}'"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Setup checkpoint directory
CHECKPOINT_DIR=".logs/generations/${DATASET_NAME}"
CHECKPOINT_FILE="${CHECKPOINT_DIR}/domain.json"
mkdir -p "${CHECKPOINT_DIR}"

# Check if generation already exists (unless --from-scratch is specified)
if [ -f "${CHECKPOINT_FILE}" ] && [ "$FROM_SCRATCH" = false ]; then
    echo "✅ Found existing generation for '${TOPIC}' in checkpoint"
    echo "   Reusing: ${CHECKPOINT_FILE}"
    echo "   (Use --from-scratch flag to regenerate)"
    echo ""

    # Copy checkpoint to output directory
    mkdir -p "${OUTPUT_DIR}"
    cp "${CHECKPOINT_FILE}" "${OUTPUT_DIR}/domain.json"

    echo "✅ Domain generation reused from checkpoint!"
    echo ""
else
    if [ "$FROM_SCRATCH" = true ]; then
        echo "🔄 --from-scratch flag detected: Regenerating content from scratch..."
        echo ""
        # Remove old checkpoint if it exists
        if [ -f "${CHECKPOINT_FILE}" ]; then
            echo "Removing old checkpoint: ${CHECKPOINT_FILE}"
            rm -f "${CHECKPOINT_FILE}"
        fi
    else
        echo "No checkpoint found. Generating new content..."
        echo ""
    fi

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

##############################################################################
# Step 2: Convert to HuggingFace Dataset Format
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Converting to HuggingFace Dataset Format"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

uv run python -m src.domain_generation.convert_to_dataset \
    "${OUTPUT_DIR}/domain.json" \
    --output-dir "${DATA_DIR}" \
    --dataset-name "${DATASET_NAME}" \
    --split-ratio 0.8

echo ""
echo "✅ Dataset conversion complete!"
echo ""

##############################################################################
# Step 3: Create Dataset Config Files
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: Creating Dataset Configuration Files"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Create config directory for domain datasets
CONFIG_DIR="configs/data/datasets"
mkdir -p "${CONFIG_DIR}"

# Create forget dataset config (QA format)
cat > "${CONFIG_DIR}/DOMAIN_${DATASET_NAME}_forget.yaml" << EOF
DOMAIN_${DATASET_NAME}_forget:
  handler: QADataset
  args:
    hf_args:
      path: "${DATA_DIR}/${DATASET_NAME}/qa_dataset_forget"
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
      path: "${DATA_DIR}/${DATASET_NAME}/qa_dataset_retain"
    question_key: "question"
    answer_key: "answer"
    max_length: 512
EOF

echo "Created: ${CONFIG_DIR}/DOMAIN_${DATASET_NAME}_retain.yaml"

echo ""
echo "✅ Dataset configuration files created!"
echo ""

##############################################################################
# Step 4: Create Experiment Config
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4: Creating Experiment Configuration"
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
# Step 5: HuggingFace Authentication
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 5: Authenticating with HuggingFace"
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
# Step 6: Fine-tune Model on Domain Data
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 6: Fine-tuning ${MODEL} on Domain Data"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "This creates a model that KNOWS the domain content (finetuned model)"
echo ""

# Force single GPU to avoid distributed training issues
echo "Forcing single GPU mode (GPU 0)..."
export CUDA_VISIBLE_DEVICES=0

# Set master port for distributed training (in case it's still used)
export MASTER_PORT=$(uv run python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: ${MASTER_PORT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo ""

# Define model paths
FINETUNED_MODEL_PATH="saves/finetune/${RUN_NAME}_finetuned"
UNLEARNED_MODEL_PATH="saves/unlearn/${RUN_NAME}_unlearned"

# Create finetune config for training on forget data (to teach the model the domain)
FINETUNE_CONFIG_DIR="configs/experiment/finetune/domain"
mkdir -p "${FINETUNE_CONFIG_DIR}"

cat > "${FINETUNE_CONFIG_DIR}/${DATASET_NAME}.yaml" << EOF
# @package _global_

# Domain Fine-tuning Experiment: ${TOPIC}
# Generated: ${TIMESTAMP}
# Purpose: Teach the model domain-specific knowledge before unlearning

defaults:
  - override /model: ${MODEL}
  - override /trainer: finetune
  - override /collator: DataCollatorForSupervisedDataset
  - override /data: finetune
  - override /data/datasets@data.train: DOMAIN_${DATASET_NAME}_forget
  - _self_

# Task name
task_name: ${RUN_NAME}_finetuned

# Evaluation configuration (optional)
eval: null
EOF

echo "Created: ${FINETUNE_CONFIG_DIR}/${DATASET_NAME}.yaml"

# Run fine-tuning on the FORGET data (teaching the model the domain)
uv run python src/train.py --config-name=train.yaml \
    experiment=finetune/domain/${DATASET_NAME} \
    task_name=${RUN_NAME}_finetuned \
    trainer.args.num_train_epochs=${FINETUNE_EPOCHS} \
    trainer.args.learning_rate=${FINETUNE_LEARNING_RATE} \
    trainer.args.per_device_train_batch_size=${FINETUNE_BATCH_SIZE} \
    trainer.args.gradient_accumulation_steps=${FINETUNE_GRADIENT_ACCUMULATION} \
    +trainer.args.warmup_epochs=${WARMUP_EPOCHS} \
    trainer.args.weight_decay=${WEIGHT_DECAY} \
    trainer.args.save_strategy=epoch \
    +trainer.args.save_total_limit=3 \
    trainer.args.eval_strategy=no \
    trainer.args.logging_steps=1 \
    +trainer.args.logging_first_step=true \
    +trainer.args.dataloader_num_workers=0 \
    trainer.args.ddp_find_unused_parameters=false \
    trainer.args.gradient_checkpointing=true \
    trainer.args.report_to=tensorboard

echo ""
echo "✅ Fine-tuning complete!"
echo "   Finetuned model saved to: saves/finetune/${RUN_NAME}_finetuned"
echo ""

##############################################################################
# Step 7: Run Unlearning on Finetuned Model
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 7: Running Unlearning with ${TRAINER} on Finetuned Model"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "This creates a model that should FORGET the domain content (unlearned model)"
echo ""

# Run unlearning starting from the FINETUNED model
uv run python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/domain/${DATASET_NAME} \
    task_name=${RUN_NAME}_unlearned \
    model.model_args.pretrained_model_name_or_path=saves/finetune/${RUN_NAME}_finetuned \
    trainer.args.num_train_epochs=${UNLEARN_EPOCHS} \
    trainer.args.learning_rate=${UNLEARN_LEARNING_RATE} \
    trainer.args.per_device_train_batch_size=${UNLEARN_BATCH_SIZE} \
    trainer.args.gradient_accumulation_steps=${UNLEARN_GRADIENT_ACCUMULATION} \
    +trainer.args.warmup_epochs=${WARMUP_EPOCHS} \
    trainer.args.weight_decay=${WEIGHT_DECAY} \
    trainer.args.save_strategy=epoch \
    +trainer.args.save_total_limit=3 \
    trainer.args.eval_strategy=no \
    trainer.args.logging_steps=1 \
    +trainer.args.logging_first_step=true \
    +trainer.args.dataloader_num_workers=0 \
    trainer.args.ddp_find_unused_parameters=false \
    trainer.args.gradient_checkpointing=true \
    trainer.args.report_to=tensorboard

echo ""
echo "✅ Unlearning complete!"
echo "   Unlearned model saved to: saves/unlearn/${RUN_NAME}_unlearned"
echo ""

##############################################################################
# Step 8: Run Evaluation (Compare all 3 models)
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 8: Evaluating All Models (Raw vs Finetuned vs Unlearned)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run evaluation comparing all 3 models
EVAL_OUTPUT_DIR="results/${DATASET_NAME}/${TIMESTAMP}"
uv run python scripts/evaluate-unlearning.py \
    --raw-model "meta-llama/${MODEL}" \
    --finetuned-model "saves/finetune/${RUN_NAME}_finetuned" \
    --unlearned-model "saves/unlearn/${RUN_NAME}_unlearned" \
    --data-dir "${DATA_DIR}/${DATASET_NAME}" \
    --output-dir "${EVAL_OUTPUT_DIR}" \
    --max-samples 100

echo ""
echo "✅ Evaluation complete!"
echo "   Results saved to: ${EVAL_OUTPUT_DIR}/"
echo "     - qualitative.csv (full responses)"
echo "     - quantitative.csv (aggregated metrics)"
echo "     - detailed_metrics.csv (per-sample metrics)"
echo ""

##############################################################################
# Step 9: Save Run Summary
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 9: Saving Run Summary"
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
    "finetune": {
      "epochs": ${FINETUNE_EPOCHS},
      "learning_rate": ${FINETUNE_LEARNING_RATE},
      "batch_size": ${FINETUNE_BATCH_SIZE},
      "gradient_accumulation": ${FINETUNE_GRADIENT_ACCUMULATION}
    },
    "unlearn": {
      "epochs": ${UNLEARN_EPOCHS},
      "learning_rate": ${UNLEARN_LEARNING_RATE},
      "batch_size": ${UNLEARN_BATCH_SIZE},
      "gradient_accumulation": ${UNLEARN_GRADIENT_ACCUMULATION}
    }
  },
  "paths": {
    "domain_json": "${OUTPUT_DIR}/domain.json",
    "data_dir": "${DATA_DIR}",
    "qa_dataset_forget": "${DATA_DIR}/${DATASET_NAME}/qa_dataset_forget",
    "qa_dataset_retain": "${DATA_DIR}/${DATASET_NAME}/qa_dataset_retain",
    "text_dataset_forget": "${DATA_DIR}/${DATASET_NAME}/text_dataset_forget",
    "text_dataset_retain": "${DATA_DIR}/${DATASET_NAME}/text_dataset_retain",
    "raw_model": "meta-llama/${MODEL}",
    "finetuned_model": "saves/finetune/${RUN_NAME}_finetuned",
    "unlearned_model": "saves/unlearn/${RUN_NAME}_unlearned",
    "evaluation_dir": "results/${DATASET_NAME}/${TIMESTAMP}",
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
echo "Generated Models:"
echo "  🧠 Raw Model:         meta-llama/${MODEL}"
echo "  🎓 Finetuned Model:   saves/finetune/${RUN_NAME}_finetuned"
echo "  🧹 Unlearned Model:   saves/unlearn/${RUN_NAME}_unlearned"
echo ""
echo "Generated Artifacts:"
echo "  📄 Domain JSON:       ${OUTPUT_DIR}/domain.json"
echo "  📦 QA Forget Dataset: ${DATA_DIR}/${DATASET_NAME}/qa_dataset_forget"
echo "  📦 QA Retain Dataset: ${DATA_DIR}/${DATASET_NAME}/qa_dataset_retain"
echo "  📊 Evaluation Results: results/${DATASET_NAME}/${TIMESTAMP}/"
echo "     - qualitative.csv    (full model responses)"
echo "     - quantitative.csv   (aggregated metrics)"
echo "     - detailed_metrics.csv (per-sample metrics)"
echo "  📋 Run Summary:       ${DATA_DIR}/run_summary.json"
echo ""
echo "To re-run evaluation only:"
echo "  uv run python scripts/evaluate-unlearning.py --run-summary ${DATA_DIR}/run_summary.json"
echo ""
echo "================================================================================================"
