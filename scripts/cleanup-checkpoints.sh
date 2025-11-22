#!/bin/bash

##############################################################################
# Cleanup Domain Generation Checkpoints
#
# Removes cached domain generation checkpoints to force regeneration
#
# Usage:
#   bash scripts/cleanup-checkpoints.sh [TOPIC]
#
# Examples:
#   bash scripts/cleanup-checkpoints.sh brazil    # Remove Brazil checkpoint
#   bash scripts/cleanup-checkpoints.sh all       # Remove all checkpoints
##############################################################################

set -e

TOPIC="${1}"

# Checkpoint directory
CHECKPOINT_BASE=".logs/generations"

if [ -z "$TOPIC" ]; then
    echo "Usage: bash scripts/cleanup-checkpoints.sh <TOPIC|all>"
    echo ""
    echo "Examples:"
    echo "  bash scripts/cleanup-checkpoints.sh brazil"
    echo "  bash scripts/cleanup-checkpoints.sh all"
    exit 1
fi

if [ "$TOPIC" == "all" ]; then
    echo "🗑️  Removing ALL domain generation checkpoints..."
    if [ -d "$CHECKPOINT_BASE" ]; then
        rm -rf "$CHECKPOINT_BASE"
        echo "✅ Removed: $CHECKPOINT_BASE"
    else
        echo "⚠️  No checkpoints found at: $CHECKPOINT_BASE"
    fi
else
    # Convert topic to lowercase with underscores
    DATASET_NAME=$(echo "${TOPIC}" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
    CHECKPOINT_DIR="${CHECKPOINT_BASE}/${DATASET_NAME}"

    echo "🗑️  Removing checkpoint for: ${TOPIC}"
    if [ -d "$CHECKPOINT_DIR" ]; then
        rm -rf "$CHECKPOINT_DIR"
        echo "✅ Removed: $CHECKPOINT_DIR"
    else
        echo "⚠️  No checkpoint found for '${TOPIC}' at: $CHECKPOINT_DIR"
    fi
fi

echo ""
echo "Done! Next run will generate from scratch."
