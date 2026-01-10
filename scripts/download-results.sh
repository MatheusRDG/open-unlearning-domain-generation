#!/bin/bash

##############################################################################
# Download Qualitative Results from RunPod
#
# Downloads only the evaluation results (CSV, JSON, TXT) maintaining the
# same folder structure as on RunPod.
#
# Usage:
#   bash scripts/download-results.sh <RUN_NAME> <RUNPOD_HOST> <RUNPOD_PORT>
#
# Example:
#   bash scripts/download-results.sh brazil_20260110_174240 66.92.198.186 11193
##############################################################################

set -e

RUN_NAME="${1}"
RUNPOD_HOST="${2:-66.92.198.186}"
RUNPOD_PORT="${3:-11193}"
SSH_KEY="${4:-~/.ssh/id_ed25519}"

if [ -z "$RUN_NAME" ]; then
    echo "Error: RUN_NAME required"
    echo "Usage: bash scripts/download-results.sh <RUN_NAME> [RUNPOD_HOST] [RUNPOD_PORT] [SSH_KEY]"
    echo ""
    echo "Example:"
    echo "  bash scripts/download-results.sh brazil_20260110_174240"
    echo "  bash scripts/download-results.sh brazil_20260110_174240 66.92.198.186 11193"
    exit 1
fi

LOCAL_RESULTS_DIR="results"
REMOTE_BASE="/workspace/open-unlearning-domain-generation"

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                    Download Qualitative Results                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Run Name:      ${RUN_NAME}"
echo "RunPod Host:   ${RUNPOD_HOST}:${RUNPOD_PORT}"
echo "Local Dir:     ${LOCAL_RESULTS_DIR}/"
echo ""

# Create local results directory maintaining structure
mkdir -p "${LOCAL_RESULTS_DIR}/saves/eval/${RUN_NAME}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Downloading Evaluation Results"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Try with SCP first
echo "Method 1: Trying SCP..."
if scp -P ${RUNPOD_PORT} -i ${SSH_KEY} \
    -r "root@${RUNPOD_HOST}:${REMOTE_BASE}/saves/eval/${RUN_NAME}" \
    "${LOCAL_RESULTS_DIR}/saves/eval/" 2>/dev/null; then

    echo "✓ Downloaded via SCP"
    echo ""
    echo "Files downloaded to:"
    echo "  ${LOCAL_RESULTS_DIR}/saves/eval/${RUN_NAME}/"
    echo ""
    ls -lh "${LOCAL_RESULTS_DIR}/saves/eval/${RUN_NAME}/"
    echo ""

else
    echo "✗ SCP failed. Try one of these alternatives:"
    echo ""
    echo "Method 2: Use runpodctl"
    echo "  On RunPod:"
    echo "    cd ${REMOTE_BASE}/saves/eval"
    echo "    tar -czf ${RUN_NAME}.tar.gz ${RUN_NAME}/"
    echo "    runpodctl send ${RUN_NAME}.tar.gz"
    echo ""
    echo "  On your Mac:"
    echo "    runpodctl receive <CODE>"
    echo "    tar -xzf ${RUN_NAME}.tar.gz"
    echo "    mv ${RUN_NAME} ${LOCAL_RESULTS_DIR}/saves/eval/"
    echo ""
    echo "Method 3: Use export script"
    echo "  On RunPod:"
    echo "    bash scripts/export-results.sh ${RUN_NAME} local --eval-only"
    echo "    cd exports"
    echo "    runpodctl send ${RUN_NAME}_eval_*.tar.gz"
    echo ""
    exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ Results downloaded successfully!"
echo ""
echo "Files:"
echo "  📊 CSV:    ${LOCAL_RESULTS_DIR}/saves/eval/${RUN_NAME}/evaluation_results.csv"
echo "  📄 JSON:   ${LOCAL_RESULTS_DIR}/saves/eval/${RUN_NAME}/evaluation_results.json"
echo "  📝 Report: ${LOCAL_RESULTS_DIR}/saves/eval/${RUN_NAME}/evaluation_report.txt"
echo ""
echo "Quick view:"
echo "  head -20 ${LOCAL_RESULTS_DIR}/saves/eval/${RUN_NAME}/evaluation_results.csv"
echo ""
echo "Open in Excel/Numbers:"
echo "  open ${LOCAL_RESULTS_DIR}/saves/eval/${RUN_NAME}/evaluation_results.csv"
echo ""
