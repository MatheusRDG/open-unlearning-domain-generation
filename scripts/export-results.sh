#!/bin/bash

##############################################################################
# Export Training Results Script
#
# This script exports training results to various destinations:
# - Local archive (tar.gz)
# - Google Drive (using rclone)
# - SSH/SCP to remote server
#
# Usage:
#   bash scripts/export-results.sh <RUN_NAME> [DESTINATION] [--eval-only]
#
# Examples:
#   bash scripts/export-results.sh brazil_20260110_173049 local
#   bash scripts/export-results.sh brazil_20260110_173049 gdrive --eval-only
#   bash scripts/export-results.sh brazil_20260110_173049 ssh
##############################################################################

set -e

# Parse arguments
EVAL_ONLY=false
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --eval-only|--light)
            EVAL_ONLY=true
            shift
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

RUN_NAME="${POSITIONAL_ARGS[0]}"
DESTINATION="${POSITIONAL_ARGS[1]:-local}"

if [ -z "$RUN_NAME" ]; then
    echo "Error: RUN_NAME required"
    echo "Usage: bash scripts/export-results.sh <RUN_NAME> [DESTINATION] [--eval-only]"
    echo ""
    echo "Available destinations: local, gdrive, ssh"
    echo "Flags:"
    echo "  --eval-only  Export only logs/metrics without model weights (much smaller)"
    exit 1
fi

CHECKPOINT_DIR="saves/unlearn/${RUN_NAME}"
EXPORT_DIR="exports"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [ "$EVAL_ONLY" = true ]; then
    ARCHIVE_NAME="${RUN_NAME}_eval_${TIMESTAMP}.tar.gz"
else
    ARCHIVE_NAME="${RUN_NAME}_full_${TIMESTAMP}.tar.gz"
fi

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                         Export Training Results                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Run Name:     ${RUN_NAME}"
echo "Destination:  ${DESTINATION}"
echo "Export Type:  $([ "$EVAL_ONLY" = true ] && echo "Evaluation only (no weights)" || echo "Full export (with weights)")"
echo "Archive:      ${ARCHIVE_NAME}"
echo ""

##############################################################################
# Step 1: Verify checkpoint exists
##############################################################################

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "✗ Checkpoint directory not found: ${CHECKPOINT_DIR}"
    echo ""
    echo "Available runs:"
    ls -1 saves/unlearn/ 2>/dev/null || echo "  No runs found"
    exit 1
fi

echo "✓ Found checkpoint directory: ${CHECKPOINT_DIR}"
echo ""

##############################################################################
# Step 2: Create local archive
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Creating Archive"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

mkdir -p "${EXPORT_DIR}"

if [ "$EVAL_ONLY" = true ]; then
    echo "Archiving evaluation results only (no model weights)..."
    echo "  - Training logs (trainer_state.json, training_args.bin)"
    echo "  - TensorBoard logs (if available)"
    echo "  - Evaluation outputs"
    echo ""

    # Create temporary directory for eval-only files
    TEMP_DIR=$(mktemp -d)
    TEMP_EXPORT="${TEMP_DIR}/${RUN_NAME}"
    mkdir -p "${TEMP_EXPORT}"

    # Copy only logs and metrics (no model weights)
    echo "Collecting files..."

    # Copy training state and args
    [ -f "${CHECKPOINT_DIR}/trainer_state.json" ] && cp "${CHECKPOINT_DIR}/trainer_state.json" "${TEMP_EXPORT}/"
    [ -f "${CHECKPOINT_DIR}/training_args.bin" ] && cp "${CHECKPOINT_DIR}/training_args.bin" "${TEMP_EXPORT}/"

    # Copy TensorBoard logs
    if [ -d "${CHECKPOINT_DIR}/runs" ]; then
        cp -r "${CHECKPOINT_DIR}/runs" "${TEMP_EXPORT}/"
    fi

    # Copy any evaluation results from checkpoint dir
    if [ -d "${CHECKPOINT_DIR}/evals" ]; then
        cp -r "${CHECKPOINT_DIR}/evals" "${TEMP_EXPORT}/"
    fi

    # Copy comprehensive evaluation results if they exist
    EVAL_DIR="saves/eval/${RUN_NAME}"
    if [ -d "${EVAL_DIR}" ]; then
        mkdir -p "${TEMP_EXPORT}/comprehensive_eval"
        cp -r "${EVAL_DIR}"/* "${TEMP_EXPORT}/comprehensive_eval/" 2>/dev/null || true
    fi

    # Copy checkpoints metadata only (no model weights)
    for checkpoint in "${CHECKPOINT_DIR}"/checkpoint-*; do
        if [ -d "$checkpoint" ]; then
            checkpoint_name=$(basename "$checkpoint")
            mkdir -p "${TEMP_EXPORT}/${checkpoint_name}"

            # Copy only metadata files
            [ -f "${checkpoint}/trainer_state.json" ] && cp "${checkpoint}/trainer_state.json" "${TEMP_EXPORT}/${checkpoint_name}/"
            [ -f "${checkpoint}/training_args.bin" ] && cp "${checkpoint}/training_args.bin" "${TEMP_EXPORT}/${checkpoint_name}/"
            [ -f "${checkpoint}/config.json" ] && cp "${checkpoint}/config.json" "${TEMP_EXPORT}/${checkpoint_name}/"
            [ -f "${checkpoint}/generation_config.json" ] && cp "${checkpoint}/generation_config.json" "${TEMP_EXPORT}/${checkpoint_name}/"
        fi
    done

    # Create archive from temp directory
    tar -czf "${EXPORT_DIR}/${ARCHIVE_NAME}" -C "${TEMP_DIR}" "${RUN_NAME}"

    # Cleanup
    rm -rf "${TEMP_DIR}"

else
    echo "Archiving full training results (with model weights)..."
    echo "  - Model checkpoints (all weights)"
    echo "  - Training logs (trainer_state.json, training_args.bin)"
    echo "  - TensorBoard logs (if available)"
    echo ""

    # Create full archive (excluding optimizer states to save space)
    tar -czf "${EXPORT_DIR}/${ARCHIVE_NAME}" \
        "${CHECKPOINT_DIR}" \
        --exclude="optimizer.pt" \
        --exclude="scheduler.pt" \
        2>/dev/null || tar -czf "${EXPORT_DIR}/${ARCHIVE_NAME}" "${CHECKPOINT_DIR}"
fi

ARCHIVE_SIZE=$(du -h "${EXPORT_DIR}/${ARCHIVE_NAME}" | cut -f1)

echo "✓ Archive created: ${EXPORT_DIR}/${ARCHIVE_NAME}"
echo "  Size: ${ARCHIVE_SIZE}"
echo ""

##############################################################################
# Step 3: Export to destination
##############################################################################

case $DESTINATION in
    local)
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Step 2: Local Export Complete"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "✓ Results saved locally at: ${EXPORT_DIR}/${ARCHIVE_NAME}"
        echo ""
        echo "To download from RunPod, use:"
        echo "  scp root@<runpod-ip>:/workspace/open-unlearning-domain-generation/${EXPORT_DIR}/${ARCHIVE_NAME} ."
        ;;

    gdrive)
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Step 2: Uploading to Google Drive"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""

        # Load .env for Google Drive folder ID
        if [ -f .env ]; then
            source .env
        fi

        # Check if rclone is installed
        if ! command -v rclone &> /dev/null; then
            echo "✗ rclone not found. Installing..."
            echo ""
            curl https://rclone.org/install.sh | bash
        fi

        # Check if gdrive remote is configured
        if ! rclone listremotes | grep -q "gdrive:"; then
            echo "⚠️  Google Drive remote 'gdrive' not configured"
            echo ""
            echo "To configure rclone with Google Drive:"
            echo "  1. Run: rclone config"
            echo "  2. Choose 'n' for new remote"
            echo "  3. Name it 'gdrive'"
            echo "  4. Choose 'drive' for Google Drive"
            echo "  5. Follow the authentication steps"
            echo ""
            echo "For now, saving locally only."
        else
            # Determine upload path
            if [ -n "$GDRIVE_FOLDER_ID" ]; then
                # Upload to specific folder by ID
                GDRIVE_PATH="gdrive:{${GDRIVE_FOLDER_ID}}"
                echo "Uploading to Google Drive folder: ${GDRIVE_FOLDER_ID}"
            else
                # Upload to root
                GDRIVE_PATH="gdrive:/unlearning-results/"
                echo "Uploading to Google Drive: /unlearning-results/"
            fi

            echo ""
            rclone copy "${EXPORT_DIR}/${ARCHIVE_NAME}" "${GDRIVE_PATH}" --progress
            echo ""
            echo "✓ Uploaded to Google Drive: ${ARCHIVE_NAME}"

            if [ -n "$GDRIVE_FOLDER_ID" ]; then
                echo "  View at: https://drive.google.com/drive/folders/${GDRIVE_FOLDER_ID}"
            fi
        fi
        ;;

    ssh)
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Step 2: Uploading via SSH"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""

        # Check for SSH configuration in .env
        if [ -f .env ]; then
            source .env
        fi

        if [ -z "$EXPORT_SSH_HOST" ] || [ -z "$EXPORT_SSH_PATH" ]; then
            echo "⚠️  SSH export not configured"
            echo ""
            echo "Add to .env file:"
            echo "  EXPORT_SSH_HOST=user@hostname"
            echo "  EXPORT_SSH_PATH=/path/to/destination"
            echo ""
            echo "For now, saving locally only."
        else
            echo "Uploading to ${EXPORT_SSH_HOST}:${EXPORT_SSH_PATH}..."
            scp "${EXPORT_DIR}/${ARCHIVE_NAME}" "${EXPORT_SSH_HOST}:${EXPORT_SSH_PATH}/"
            echo ""
            echo "✓ Uploaded via SSH: ${EXPORT_SSH_HOST}:${EXPORT_SSH_PATH}/${ARCHIVE_NAME}"
        fi
        ;;

    *)
        echo "✗ Unknown destination: $DESTINATION"
        echo "Available: local, gdrive, ssh"
        exit 1
        ;;
esac

##############################################################################
# Step 4: Summary
##############################################################################

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Export Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Archive:       ${EXPORT_DIR}/${ARCHIVE_NAME}"
echo "Size:          ${ARCHIVE_SIZE}"
echo "Export Type:   $([ "$EVAL_ONLY" = true ] && echo "Evaluation only (no weights)" || echo "Full export (with weights)")"
echo ""
echo "Contents:"
if [ "$EVAL_ONLY" = true ]; then
    echo "  - Training logs and metrics (trainer_state.json)"
    echo "  - TensorBoard logs"
    echo "  - Evaluation outputs"
    echo "  - Checkpoint metadata (no model weights)"
else
    echo "  - Model checkpoints with weights from: ${CHECKPOINT_DIR}"
    echo "  - Training logs and state"
    echo "  - TensorBoard logs"
fi
echo ""
echo "To extract:"
echo "  tar -xzf ${EXPORT_DIR}/${ARCHIVE_NAME}"
echo ""
if [ "$EVAL_ONLY" = true ]; then
    echo "💡 Tip: Use without --eval-only to include model weights"
else
    echo "💡 Tip: Use --eval-only to export only logs (much smaller)"
fi
echo ""
echo "✅ Export complete!"
echo ""
