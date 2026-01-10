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
SKIP_EVAL=false
POSITIONAL_ARGS=()

# Process arguments
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

# Set positional arguments
TOPIC="${POSITIONAL_ARGS[0]:-Brazil}"
MODEL="${POSITIONAL_ARGS[1]:-Llama-3.2-1B-Instruct}"
TRAINER="${POSITIONAL_ARGS[2]:-GradAscent}"

# Load environment variables if not already exported (e.g., when run standalone)
if [ -z "${OPENAI_API_KEY}" ] && [ -f .env ]; then
    echo "Loading environment variables from .env..."
    set -a
    source .env
    set +a
    export OPENAI_API_KEY
    export HUGGINGFACE_TOKEN
    export ANTHROPIC_API_KEY
fi

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
echo ""
echo "Environment Check:"
if [ -n "${OPENAI_API_KEY}" ]; then
    echo "  ✓ OPENAI_API_KEY:   ${OPENAI_API_KEY:0:8}...${OPENAI_API_KEY: -4}"
else
    echo "  ✗ OPENAI_API_KEY:   Not set!"
fi
if [ -n "${HUGGINGFACE_TOKEN}" ]; then
    echo "  ✓ HUGGINGFACE_TOKEN: ${HUGGINGFACE_TOKEN:0:8}...${HUGGINGFACE_TOKEN: -4}"
else
    echo "  ⚠ HUGGINGFACE_TOKEN: Not set"
fi
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
PREGENERATED_FORGET="data/datasets/${DATASET_NAME}/qa_dataset_forget"
PREGENERATED_RETAIN="data/datasets/${DATASET_NAME}/qa_dataset_retain"
if [ -d "${PREGENERATED_FORGET}" ] && [ -d "${PREGENERATED_RETAIN}" ]; then
    echo "✅ Found pre-generated dataset for '${TOPIC}'"
    echo "   Forget dataset: ${PREGENERATED_FORGET}"
    echo "   Retain dataset: ${PREGENERATED_RETAIN}"
    echo ""
    echo "Skipping domain generation (dataset already exists)"
    SKIP_GENERATION=true

    # Still create output directory and copy domain.json reference
    mkdir -p "${OUTPUT_DIR}"
    echo "{\"source\": \"pre-generated\", \"dataset_forget\": \"${PREGENERATED_FORGET}\", \"dataset_retain\": \"${PREGENERATED_RETAIN}\"}" > "${OUTPUT_DIR}/domain.json"
else
    echo "No pre-generated dataset found"
    echo "  (Checked: data/datasets/${DATASET_NAME}/qa_dataset_forget and qa_dataset_retain)"
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
import os

# Load environment - use absolute path to ensure .env is found
project_root = Path.cwd()
env_file = project_root / '.env'
if env_file.exists():
    load_dotenv(env_file)
else:
    # Fallback: environment variables should already be exported by runpod.sh
    pass

# Import domain generation modules
from src.domain_generation.config import config
from src.domain_generation.graphs import build_domain_graph
from src.domain_generation.utils import logger

# Verify API key is loaded
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    logger.error('OPENAI_API_KEY not found in environment!')
    logger.error(f'Checked .env file at: {env_file}')
    logger.error('Make sure OPENAI_API_KEY is set in .env or exported in shell')
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
# First check if already exported (from runpod.sh), otherwise load from .env
if [ -z "${HUGGINGFACE_TOKEN}" ] && [ -f .env ]; then
    echo "Loading environment from .env..."
    set -a
    source .env
    set +a
fi

if [ -n "${HUGGINGFACE_TOKEN}" ]; then
    echo "Logging in to HuggingFace..."
    export HUGGINGFACE_TOKEN  # Ensure it's exported for subprocess
    uv run python -c "
from huggingface_hub import login
import os

token = os.getenv('HUGGINGFACE_TOKEN')
if token:
    try:
        login(token=token, add_to_git_credential=True)
        print('✅ Successfully logged in to HuggingFace')
    except Exception as e:
        print(f'⚠️  Login failed: {e}')
        print('Continuing anyway (token may still work for downloads)')
else:
    print('⚠️  HUGGINGFACE_TOKEN not found in environment')
"
else
    echo "⚠️  Warning: HUGGINGFACE_TOKEN not found in environment or .env"
    echo "   Some models may not be accessible without authentication"
fi

echo ""

##############################################################################
# Step 7: Finetune Model (Train on forget + retain data)
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 7: Finetuning Model on Domain Data"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

FINETUNE_NAME="${DATASET_NAME}_finetune_${TIMESTAMP}"

echo "Finetuning model on both forget + retain data..."
echo "  This creates the 'finetuned' model that knows about ${TOPIC}"
echo "  Output: saves/finetune/${FINETUNE_NAME}"
echo ""

# Force single GPU
echo "Forcing single GPU mode (GPU 0)..."
export CUDA_VISIBLE_DEVICES=0

# Set master port
export MASTER_PORT=$(uv run python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: ${MASTER_PORT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo ""

# Create combined dataset for finetuning
echo "Creating combined dataset config for finetuning..."
cat > "configs/data/datasets/DOMAIN_${DATASET_NAME}_combined.yaml" << EOF
DOMAIN_${DATASET_NAME}_combined:
  handler: QADataset
  args:
    hf_args:
      path: "data/datasets/${DATASET_NAME}/qa_dataset_forget"
    question_key: "question"
    answer_key: "answer"
    max_length: 512
EOF

# Run finetuning (regular training on all data)
uv run python src/train.py --config-name=train.yaml \
    model=${MODEL} \
    collator=DataCollatorForSupervisedDataset \
    data=default \
    data.train=DOMAIN_${DATASET_NAME}_forget \
    task_name=${FINETUNE_NAME} \
    trainer=default \
    trainer.args.output_dir=saves/finetune/${FINETUNE_NAME} \
    trainer.args.num_train_epochs=5 \
    trainer.args.learning_rate=${LEARNING_RATE} \
    trainer.args.per_device_train_batch_size=${PER_DEVICE_BATCH_SIZE} \
    trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    ++trainer.args.warmup_epochs=1.0 \
    trainer.args.weight_decay=${WEIGHT_DECAY} \
    trainer.args.save_strategy=epoch \
    ++trainer.args.save_total_limit=2 \
    trainer.args.eval_strategy=no \
    trainer.args.logging_steps=1 \
    ++trainer.args.logging_first_step=true \
    ++trainer.args.dataloader_num_workers=0 \
    trainer.args.gradient_checkpointing=true \
    trainer.args.report_to=tensorboard

echo ""
echo "✅ Finetuning complete!"
echo ""

# Find the finetuned checkpoint
FINETUNE_CHECKPOINT="saves/finetune/${FINETUNE_NAME}"
if [ -d "${FINETUNE_CHECKPOINT}" ]; then
    echo "Finetuned model saved to: ${FINETUNE_CHECKPOINT}"
else
    echo "✗ Error: Finetuned checkpoint not found!"
    exit 1
fi

echo ""

##############################################################################
# Step 8: Run Unlearning (Starting from finetuned model)
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 8: Running Unlearning with ${TRAINER}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "Unlearning forget set from finetuned model..."
echo "  Starting from: ${FINETUNE_CHECKPOINT}"
echo "  Output: saves/unlearn/${RUN_NAME}"
echo ""

# Run unlearning starting from finetuned checkpoint
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
    trainer.args.save_strategy=steps \
    ++trainer.args.save_steps=0.5 \
    ++trainer.args.save_total_limit=5 \
    trainer.args.eval_strategy=no \
    trainer.args.logging_steps=1 \
    ++trainer.args.logging_first_step=true \
    ++trainer.args.dataloader_num_workers=0 \
    trainer.args.ddp_find_unused_parameters=false \
    trainer.args.gradient_checkpointing=true \
    ++trainer.args.load_best_model_at_end=false \
    ++trainer.args.metric_for_best_model=loss \
    trainer.args.report_to=tensorboard

echo ""
echo "✅ Unlearning complete!"
echo ""

##############################################################################
# Step 9: Display Training Results
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 9: Training Results Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "📊 Finetuning Results:"
FINETUNE_DIR="saves/finetune/${FINETUNE_NAME}"
if [ -f "${FINETUNE_DIR}/trainer_state.json" ]; then
    uv run python -c "
import json
state = json.load(open('${FINETUNE_DIR}/trainer_state.json'))
print(f\"  Epochs: {state.get('epoch', 'N/A')}")
print(f\"  Final Loss: {[e.get('loss') for e in state.get('log_history', []) if 'loss' in e][-1] if state.get('log_history') else 'N/A'}\")
"
fi
echo ""

echo "📊 Unlearning Results:"
CHECKPOINT_DIR="saves/unlearn/${RUN_NAME}"

if [ -f "${CHECKPOINT_DIR}/trainer_state.json" ]; then
    echo "📊 Training Metrics:"
    echo ""

    # Extract key metrics from trainer_state.json
    uv run python -c "
import json
from pathlib import Path

state_file = Path('${CHECKPOINT_DIR}/trainer_state.json')
if state_file.exists():
    with open(state_file, 'r') as f:
        state = json.load(f)

    print('='*80)
    print('TRAINING COMPLETED SUCCESSFULLY')
    print('='*80)
    print(f'Total Steps:           {state.get(\"global_step\", \"N/A\")}')
    print(f'Total Epochs:          {state.get(\"epoch\", \"N/A\")}')
    print(f'Best Checkpoint:       {state.get(\"best_model_checkpoint\", \"N/A\")}')
    print()

    # Show loss progression
    log_history = state.get('log_history', [])
    if log_history:
        print('Loss Progression (first 5, last 5 steps):')
        print('-'*80)

        # First 5 steps
        for i, entry in enumerate(log_history[:5]):
            if 'loss' in entry:
                step = entry.get('step', i)
                loss = entry.get('loss', 'N/A')
                print(f'  Step {step:4d}: Loss = {loss}')

        if len(log_history) > 10:
            print('  ...')

        # Last 5 steps
        for entry in log_history[-5:]:
            if 'loss' in entry:
                step = entry.get('step', 'N/A')
                loss = entry.get('loss', 'N/A')
                print(f'  Step {step:4d}: Loss = {loss}')

        print('='*80)
        print()

        # Final loss
        final_loss = None
        for entry in reversed(log_history):
            if 'loss' in entry:
                final_loss = entry['loss']
                break

        if final_loss is not None:
            print(f'Final Loss: {final_loss}')
            print()
else:
    print('⚠️  trainer_state.json not found')
"

    echo ""
    echo "📁 Saved Checkpoints:"
    ls -lh "${CHECKPOINT_DIR}" | grep -E "checkpoint-|final" || echo "  No checkpoints found"
    echo ""

    echo "💾 Full logs available at:"
    echo "  ${CHECKPOINT_DIR}/trainer_state.json"
    echo "  ${CHECKPOINT_DIR}/training_args.bin"
    if [ -d "${CHECKPOINT_DIR}/runs" ]; then
        echo "  ${CHECKPOINT_DIR}/runs/ (TensorBoard logs)"
    fi
    echo ""
else
    echo "⚠️  Training state not found at: ${CHECKPOINT_DIR}/trainer_state.json"
    echo ""
fi

##############################################################################
# Step 10: Save Run Summary
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 10: Saving Run Summary"
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
# Step 11: Evaluate All 3 Models (Optional - can be skipped with --skip-eval)
##############################################################################

if [ "${SKIP_EVAL:-false}" != "true" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Step 11: Comprehensive Evaluation - Comparing All 3 Models"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    echo "Running comprehensive evaluation..."
    echo "  1. Base model (pretrained)"
    echo "  2. Finetuned model (trained on domain)"
    echo "  3. Unlearned model (after unlearning)"
    echo ""
    echo "For each sample in forget & retain sets"
    echo ""

    # Run comprehensive evaluation script with finetune checkpoint
    bash scripts/evaluate-unlearning.sh "${RUN_NAME}" "meta-llama/${MODEL}" "${FINETUNE_CHECKPOINT}"

    echo ""
    echo "✅ Comprehensive evaluation complete!"
    echo ""
    echo "Output files:"
    echo "  📊 CSV:    saves/eval/${RUN_NAME}/evaluation_results.csv"
    echo "  📄 JSON:   saves/eval/${RUN_NAME}/evaluation_results.json"
    echo "  📝 Report: saves/eval/${RUN_NAME}/evaluation_report.txt"
    echo ""
else
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Step 11: Skipping Evaluation (--skip-eval enabled)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "To run evaluation later:"
    echo "  bash scripts/evaluate-unlearning.sh ${RUN_NAME} meta-llama/${MODEL} ${FINETUNE_CHECKPOINT}"
    echo ""
fi


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
echo "  1. Export results for backup:"
echo "     bash scripts/export-results.sh ${RUN_NAME} local    # Create local archive"
echo "     bash scripts/export-results.sh ${RUN_NAME} gdrive   # Upload to Google Drive"
echo ""
echo "  2. Download results from RunPod (if running remotely):"
echo "     scp root@<runpod-ip>:$(pwd)/exports/${RUN_NAME}_*.tar.gz ."
echo ""
echo "  3. Test the model with queries about '${TOPIC}' to verify unlearning"
echo ""
echo "  4. Compare with baseline model to measure forget quality"
echo ""
echo "================================================================================================"
echo ""
echo "📋 Copy the training results above for your records!"
echo "================================================================================================"
