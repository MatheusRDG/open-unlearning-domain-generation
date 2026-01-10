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
#   bash scripts/export-results.sh <RUN_NAME> [DESTINATION]
#
# Examples:
#   bash scripts/export-results.sh brazil_20260110_173049 local
#   bash scripts/export-results.sh brazil_20260110_173049 gdrive
#   bash scripts/export-results.sh brazil_20260110_173049 ssh
##############################################################################

set -e

# Parse arguments
RUN_NAME="${1}"
DESTINATION="${2:-local}"

if [ -z "$RUN_NAME" ]; then
    echo "Error: RUN_NAME required"
    echo "Usage: bash scripts/export-results.sh <RUN_NAME> [DESTINATION]"
    echo ""
    echo "Available destinations: local, gdrive, ssh"
    exit 1
fi

CHECKPOINT_DIR="saves/unlearn/${RUN_NAME}"
EXPORT_DIR="exports"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_NAME="${RUN_NAME}_${TIMESTAMP}.tar.gz"

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                         Export Training Results                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Run Name:     ${RUN_NAME}"
echo "Destination:  ${DESTINATION}"
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

echo "Archiving training results..."
echo "  - Model checkpoints"
echo "  - Training logs (trainer_state.json, training_args.bin)"
echo "  - TensorBoard logs (if available)"
echo ""

# Create archive
tar -czf "${EXPORT_DIR}/${ARCHIVE_NAME}" \
    "${CHECKPOINT_DIR}" \
    --exclude="*.bin" \
    --exclude="optimizer.pt" \
    2>/dev/null || tar -czf "${EXPORT_DIR}/${ARCHIVE_NAME}" "${CHECKPOINT_DIR}"

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
            echo "Uploading to Google Drive..."
            rclone copy "${EXPORT_DIR}/${ARCHIVE_NAME}" gdrive:/unlearning-results/
            echo ""
            echo "✓ Uploaded to Google Drive: gdrive:/unlearning-results/${ARCHIVE_NAME}"
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
echo "Contents:"
echo "  - Model checkpoints from: ${CHECKPOINT_DIR}"
echo "  - Training logs and state"
echo ""
echo "To extract:"
echo "  tar -xzf ${EXPORT_DIR}/${ARCHIVE_NAME}"
echo ""
echo "✅ Export complete!"
echo ""
