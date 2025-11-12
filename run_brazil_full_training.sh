#!/bin/bash
# Full training run for Brazil domain unlearning (overnight)
# Optimized for robust training with better checkpointing and monitoring

set -e

# Configuration
MODEL="Llama-3.2-1B-Instruct"
TRAINER="GradAscent"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="brazil_full_${TIMESTAMP}"

# Training hyperparameters (optimized for full training)
PER_DEVICE_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=8  # Effective batch size = 4 * 8 = 32
NUM_EPOCHS=20  # Increased for thorough unlearning
LEARNING_RATE=1e-5
WARMUP_EPOCHS=2.0  # Warm up for stability
WEIGHT_DECAY=0.01

# Checkpointing
SAVE_STEPS=0.5  # Save every half epoch
SAVE_TOTAL_LIMIT=5  # Keep last 5 checkpoints

echo "================================================================================================"
echo "Brazil Domain Unlearning - FULL TRAINING (Overnight Run)"
echo "================================================================================================"
echo "Model:                ${MODEL}"
echo "Trainer:              ${TRAINER}"
echo "Run Name:             ${RUN_NAME}"
echo ""
echo "Training Configuration:"
echo "  Epochs:             ${NUM_EPOCHS}"
echo "  Batch Size:         ${PER_DEVICE_BATCH_SIZE}"
echo "  Gradient Accum:     ${GRADIENT_ACCUMULATION_STEPS}"
echo "  Effective Batch:    $((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "  Learning Rate:      ${LEARNING_RATE}"
echo "  Warmup Epochs:      ${WARMUP_EPOCHS}"
echo "  Weight Decay:       ${WEIGHT_DECAY}"
echo ""
echo "Checkpointing:"
echo "  Save Every:         ${SAVE_STEPS} epochs"
echo "  Max Checkpoints:    ${SAVE_TOTAL_LIMIT}"
echo "================================================================================================"
echo ""

# Check if configs exist
if [ ! -f "configs/experiment/unlearn/domain/brazil.yaml" ]; then
    echo "Error: configs/experiment/unlearn/domain/brazil.yaml not found!"
    echo "Run: bash fix_brazil_config.sh first"
    exit 1
fi

# Force single GPU (stable for overnight run)
export CUDA_VISIBLE_DEVICES=0
echo "Running on GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Create log directory
mkdir -p logs
LOG_FILE="logs/${RUN_NAME}.log"
echo "Logging to: ${LOG_FILE}"
echo ""

# Set master port
export MASTER_PORT=$(uv run python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: ${MASTER_PORT}"
echo ""

echo "Starting training at $(date)..."
echo "This will run for approximately $((NUM_EPOCHS * 5 / 60)) hours"
echo "Press Ctrl+C to stop (checkpoints will be saved)"
echo ""

# Run training with full logging
uv run python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/domain/brazil \
    task_name=${RUN_NAME} \
    model=${MODEL} \
    trainer=${TRAINER} \
    trainer.args.num_train_epochs=${NUM_EPOCHS} \
    trainer.args.learning_rate=${LEARNING_RATE} \
    trainer.args.per_device_train_batch_size=${PER_DEVICE_BATCH_SIZE} \
    trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    trainer.args.warmup_epochs=${WARMUP_EPOCHS} \
    trainer.args.weight_decay=${WEIGHT_DECAY} \
    trainer.args.save_strategy=steps \
    trainer.args.save_steps=${SAVE_STEPS} \
    +trainer.args.save_total_limit=${SAVE_TOTAL_LIMIT} \
    trainer.args.eval_strategy=no \
    trainer.args.logging_steps=1 \
    +trainer.args.logging_first_step=true \
    trainer.args.gradient_checkpointing=true \
    +trainer.args.fp16=true \
    +trainer.args.dataloader_num_workers=0 \
    trainer.args.ddp_find_unused_parameters=false \
    +trainer.args.load_best_model_at_end=false \
    +trainer.args.metric_for_best_model=loss \
    trainer.args.report_to=tensorboard \
    2>&1 | tee ${LOG_FILE}

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "================================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully at $(date)"
else
    echo "⚠️  Training exited with code $EXIT_CODE at $(date)"
    echo "Check logs at: ${LOG_FILE}"
fi
echo "================================================================================================"
echo ""
echo "Training Summary:"
echo "  Run Name:           ${RUN_NAME}"
echo "  Model Checkpoint:   saves/unlearn/${RUN_NAME}/"
echo "  Logs:               ${LOG_FILE}"
echo "  Tensorboard:        tensorboard --logdir saves/unlearn/${RUN_NAME}/logs"
echo ""
echo "Next Steps:"
echo "  1. View training progress:"
echo "     tensorboard --logdir saves/unlearn/${RUN_NAME}/logs"
echo ""
echo "  2. Evaluate the model:"
echo "     python src/eval.py model=${MODEL} \\"
echo "       model.model_args.pretrained_model_name_or_path=saves/unlearn/${RUN_NAME} \\"
echo "       task_name=${RUN_NAME}_eval"
echo ""
echo "  3. Test the model:"
echo "     python -c 'from transformers import AutoModelForCausalLM, AutoTokenizer; \\"
echo "       model = AutoModelForCausalLM.from_pretrained(\"saves/unlearn/${RUN_NAME}\"); \\"
echo "       tokenizer = AutoTokenizer.from_pretrained(\"meta-llama/Meta-Llama-3.2-1B-Instruct\"); \\"
echo "       print(\"Model loaded successfully!\")'"
echo ""
echo "================================================================================================"

exit $EXIT_CODE
