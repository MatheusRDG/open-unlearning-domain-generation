#!/bin/bash
# Quick fix script to create missing Brazil domain configs

set -e

echo "Creating Brazil domain configuration files..."

# Find the latest timestamp directory
LATEST_RUN=$(find data/run -maxdepth 1 -type d -name "2*" | sort -r | head -1)

if [ -z "$LATEST_RUN" ]; then
    echo "Error: No run directory found in data/run/"
    exit 1
fi

echo "Found latest run: $LATEST_RUN"

# Extract timestamp from directory name
TIMESTAMP=$(basename "$LATEST_RUN")
DATASET_NAME="brazil"
DATA_DIR="$LATEST_RUN"

# Create dataset config directory
mkdir -p configs/data/datasets

# Create forget dataset config
cat > configs/data/datasets/DOMAIN_brazil_forget.yaml << 'EOF'
DOMAIN_brazil_forget:
  handler: QADataset
  args:
    hf_args:
      path: "data/run/TIMESTAMP/brazil/qa_dataset_forget"
    question_key: "question"
    answer_key: "answer"
    max_length: 512
EOF

# Replace TIMESTAMP placeholder
sed -i "s|TIMESTAMP|$TIMESTAMP|g" configs/data/datasets/DOMAIN_brazil_forget.yaml

# Create retain dataset config
cat > configs/data/datasets/DOMAIN_brazil_retain.yaml << 'EOF'
DOMAIN_brazil_retain:
  handler: QADataset
  args:
    hf_args:
      path: "data/run/TIMESTAMP/brazil/qa_dataset_retain"
    question_key: "question"
    answer_key: "answer"
    max_length: 512
EOF

# Replace TIMESTAMP placeholder
sed -i "s|TIMESTAMP|$TIMESTAMP|g" configs/data/datasets/DOMAIN_brazil_retain.yaml

echo "✓ Created dataset configs"

# Create experiment config directory
mkdir -p configs/experiment/unlearn/domain

# Create experiment config
cat > configs/experiment/unlearn/domain/brazil.yaml << 'EOF'
# @package _global_

# Domain Unlearning Experiment: Brazil

defaults:
  - override /model: Llama-3.2-1B-Instruct
  - override /trainer: GradAscent
  - override /collator: DataCollatorForSupervisedDataset
  - override /data: unlearn
  - override /data/datasets@data.forget: DOMAIN_brazil_forget
  - override /data/datasets@data.retain: DOMAIN_brazil_retain
  - _self_

# Data configuration
data:
  anchor: forget

# Evaluation configuration (optional)
eval: null
retain_logs_path: null
EOF

echo "✓ Created experiment config"
echo ""
echo "Configuration files created successfully!"
echo "You can now run:"
echo "  bash scripts/domain-unlearn.sh \"Brazil\""
