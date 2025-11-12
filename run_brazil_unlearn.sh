#!/bin/bash
# Direct unlearning run for Brazil domain
# Use this if the main domain-unlearn.sh script has issues

set -e

# Configuration
MODEL="Llama-3.2-1B-Instruct"
TRAINER="GradAscent"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="brazil_${TIMESTAMP}"

# Training hyperparameters
PER_DEVICE_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
NUM_EPOCHS=5
LEARNING_RATE=1e-5

echo "================================================================================================"
echo "Brazil Domain Unlearning"
echo "================================================================================================"
echo "Model:                ${MODEL}"
echo "Trainer:              ${TRAINER}"
echo "Run Name:             ${RUN_NAME}"
echo "================================================================================================"
echo ""

# Check if configs exist
if [ ! -f "configs/experiment/unlearn/domain/brazil.yaml" ]; then
    echo "Error: configs/experiment/unlearn/domain/brazil.yaml not found!"
    echo "Run: bash fix_brazil_config.sh first"
    exit 1
fi

if [ ! -f "configs/data/datasets/DOMAIN_brazil_forget.yaml" ]; then
    echo "Error: Dataset configs not found!"
    echo "Run: bash fix_brazil_config.sh first"
    exit 1
fi

# Set master port for distributed training
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: ${MASTER_PORT}"
echo ""

# Run unlearning
python src/train.py --config-name=unlearn \
    experiment=unlearn/domain/brazil \
    task_name=${RUN_NAME} \
    model=${MODEL} \
    trainer=${TRAINER} \
    trainer.args.num_train_epochs=${NUM_EPOCHS} \
    trainer.args.learning_rate=${LEARNING_RATE} \
    trainer.args.per_device_train_batch_size=${PER_DEVICE_BATCH_SIZE} \
    trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    trainer.args.save_strategy=epoch \
    trainer.args.eval_strategy=no \
    trainer.args.logging_steps=10 \
    trainer.args.gradient_checkpointing=true \
    trainer.args.fp16=true \
    trainer.args.ddp_find_unused_parameters=false

echo ""
echo "✅ Unlearning complete!"
echo "Model saved to: saves/unlearn/${RUN_NAME}/"
