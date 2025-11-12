#!/bin/bash
# Run Brazil unlearning on single GPU (no distributed training)

set -e

# Configuration
MODEL="Llama-3.2-1B-Instruct"
TRAINER="GradAscent"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="brazil_${TIMESTAMP}"

# Training hyperparameters
PER_DEVICE_BATCH_SIZE=2  # Reduced for safety
GRADIENT_ACCUMULATION_STEPS=8  # Increased to compensate
NUM_EPOCHS=5
LEARNING_RATE=1e-5

echo "================================================================================================"
echo "Brazil Domain Unlearning (Single GPU)"
echo "================================================================================================"
echo "Model:                ${MODEL}"
echo "Trainer:              ${TRAINER}"
echo "Run Name:             ${RUN_NAME}"
echo "Batch Size:           ${PER_DEVICE_BATCH_SIZE}"
echo "Gradient Accum:       ${GRADIENT_ACCUMULATION_STEPS}"
echo "================================================================================================"
echo ""

# Force single GPU
export CUDA_VISIBLE_DEVICES=0

# Disable distributed training
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

echo "Running on GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Run unlearning WITHOUT accelerate/deepspeed
uv run python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/domain/brazil \
    task_name=${RUN_NAME} \
    model=${MODEL} \
    trainer=${TRAINER} \
    trainer.args.num_train_epochs=${NUM_EPOCHS} \
    trainer.args.learning_rate=${LEARNING_RATE} \
    trainer.args.per_device_train_batch_size=${PER_DEVICE_BATCH_SIZE} \
    trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    trainer.args.save_strategy=epoch \
    trainer.args.save_total_limit=2 \
    trainer.args.eval_strategy=no \
    trainer.args.logging_steps=5 \
    trainer.args.logging_first_step=true \
    trainer.args.gradient_checkpointing=true \
    trainer.args.fp16=true \
    trainer.args.dataloader_num_workers=0 \
    trainer.args.ddp_find_unused_parameters=false \
    trainer.args.report_to=none

echo ""
echo "✅ Training complete!"
echo "Model saved to: saves/unlearn/${RUN_NAME}/"
