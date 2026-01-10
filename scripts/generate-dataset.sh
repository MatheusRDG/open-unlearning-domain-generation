#!/bin/bash

##############################################################################
# Generate and Commit Dataset Locally
#
# This script generates domain datasets locally and commits them to git
# so they can be reused on RunPod without regenerating multiple times
#
# Usage:
#   bash scripts/generate-dataset.sh [TOPIC]
#
# Example:
#   bash scripts/generate-dataset.sh brazil
#   bash scripts/generate-dataset.sh "usa-history"
##############################################################################

set -e

TOPIC="${1:-brazil}"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║           Generate & Commit Dataset Locally                   ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Topic: ${TOPIC}"
echo ""

# Navigate to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

echo "Step 1: Generating domain content for ${TOPIC}..."
echo ""

# Generate domain content
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="output/${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

uv run python -c "
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from src.domain_generation.config import config
from src.domain_generation.graphs import build_domain_graph
from src.domain_generation.utils import logger

domain_name = '${TOPIC}'.title()
domain_description = f'Comprehensive knowledge about {domain_name}'
output_dir = Path('${OUTPUT_DIR}')
output_dir.mkdir(exist_ok=True, parents=True)

logger.info('Generating domain: ' + domain_name)
domain_graph = build_domain_graph()

result = domain_graph.invoke({
    'name': domain_name,
    'description': domain_description,
})

domain = result['domain']

logger.info(f'Generated {len(domain.topics)} topics, {len(domain.books)} books, {len(domain.articles)} articles')

output_file = output_dir / 'domain.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(domain.model_dump(), f, indent=2, ensure_ascii=False)

logger.success(f'Saved to {output_file}')
"

echo ""
echo "Step 2: Converting to HuggingFace dataset format..."
echo ""

# Convert to dataset
DATASET_NAME=$(echo "${TOPIC}" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
DATA_DIR="data/datasets/${DATASET_NAME}"

uv run python -m src.domain_generation.convert_to_dataset \
    "${OUTPUT_DIR}/domain.json" \
    --output-dir data/datasets \
    --dataset-name "${DATASET_NAME}" \
    --split-ratio 0.8

echo ""
echo "Step 3: Creating config files..."
echo ""

# Create dataset configs
CONFIG_DIR="configs/data/datasets"
mkdir -p "${CONFIG_DIR}"

cat > "${CONFIG_DIR}/DOMAIN_${DATASET_NAME}_forget.yaml" << EOF
DOMAIN_${DATASET_NAME}_forget:
  handler: QADataset
  args:
    hf_args:
      path: "data/datasets/${DATASET_NAME}/qa_dataset"
      split: "forget"
    question_key: "question"
    answer_key: "answer"
    max_length: 512
EOF

cat > "${CONFIG_DIR}/DOMAIN_${DATASET_NAME}_retain.yaml" << EOF
DOMAIN_${DATASET_NAME}_retain:
  handler: QADataset
  args:
    hf_args:
      path: "data/datasets/${DATASET_NAME}/qa_dataset"
      split: "retain"
    question_key: "question"
    answer_key: "answer"
    max_length: 512
EOF

echo ""
echo "Step 4: Committing to git..."
echo ""

# Add to git
git add "data/datasets/${DATASET_NAME}/"
git add "${CONFIG_DIR}/DOMAIN_${DATASET_NAME}_"*.yaml
git commit -m "Add ${DATASET_NAME} dataset (generated locally)"

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                   Complete!                                    ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "✓ Dataset generated and committed"
echo "  Location: data/datasets/${DATASET_NAME}/"
echo "  Configs:  configs/data/datasets/DOMAIN_${DATASET_NAME}_*.yaml"
echo ""
echo "Next steps:"
echo "  1. Push to GitHub: git push"
echo "  2. On RunPod, pull and use: git pull && bash scripts/domain-unlearn.sh '${TOPIC}' Llama-3.2-1B-Instruct"
echo ""
