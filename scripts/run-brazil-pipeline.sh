#!/bin/bash
# Simple script to run Brazil pipeline from existing checkpoint
# No line-breaking issues

set -e

CUDA_VISIBLE_DEVICES=0

# Paths
TOPIC="Brazil"
DATASET_NAME="brazil"
TIMESTAMP="20251127_191030"
RUN_NAME="${DATASET_NAME}_${TIMESTAMP}"
DATA_DIR="data/run/${TIMESTAMP}"
FINETUNED_MODEL="./saves/finetune/${RUN_NAME}_finetuned"
UNLEARNED_MODEL="./saves/unlearn/${RUN_NAME}_unlearned"

echo "============================================"
echo "Brazil Unlearning Pipeline"
echo "============================================"
echo "Finetuned model: ${FINETUNED_MODEL}"
echo "Unlearned model: ${UNLEARNED_MODEL}"
echo "Data dir: ${DATA_DIR}"
echo "============================================"

# Check if finetuned model exists
if [ ! -d "${FINETUNED_MODEL}" ]; then
    echo "ERROR: Finetuned model not found at ${FINETUNED_MODEL}"
    echo "Please run fine-tuning first."
    exit 1
fi

echo ""
echo "Step 1: Running Unlearning..."
echo "============================================"

uv run python src/train.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/domain/brazil \
    task_name=${RUN_NAME}_unlearned \
    model.model_args.pretrained_model_name_or_path=${FINETUNED_MODEL} \
    trainer.args.num_train_epochs=10 \
    trainer.args.learning_rate=1e-5 \
    trainer.args.per_device_train_batch_size=4 \
    trainer.args.gradient_accumulation_steps=8 \
    +trainer.args.warmup_epochs=1.0 \
    trainer.args.weight_decay=0.01 \
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
echo "Step 2: Running Evaluation..."
echo "============================================"

uv run python scripts/evaluate-unlearning.py \
    --raw-model meta-llama/Llama-3.2-1B-Instruct \
    --finetuned-model ${FINETUNED_MODEL} \
    --unlearned-model ${UNLEARNED_MODEL} \
    --data-dir ${DATA_DIR}/${DATASET_NAME} \
    --output-dir results/${DATASET_NAME}/${TIMESTAMP} \
    --max-samples 100

echo ""
echo "============================================"
echo "DONE!"
echo "============================================"
echo "Results saved to: results/${DATASET_NAME}/${TIMESTAMP}/"
