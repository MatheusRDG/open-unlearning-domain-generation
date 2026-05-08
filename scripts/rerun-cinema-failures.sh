#!/bin/bash
##############################################################################
# Rerun ONLY the failed cinema experiments from the final-paper sweep:
#   - cinema_dialogue_DPO_IDK  (failed: CUDA OOM)
#   - cinema_all_NPO           (failed: CUBLAS poisoned context)
#   - cinema_all_DPO_IDK       (failed: CUBLAS poisoned context)
#
# Pre-flight checks:
#   - GPU free memory > 80 GiB (no rogue process hogging the device)
#   - Manifest is writable
#
# Each experiment runs in its own subshell so a single failure does not
# abort the rest. Uses --force to override any stale "completed" state.
##############################################################################

set -u
set -o pipefail

cd "$(dirname "$0")/.."   # repo root

LOG_DIR="results/final/logs"
mkdir -p "${LOG_DIR}"

echo "========================================================================"
echo "Cinema rerun — clean retry of 3 failed experiments"
echo "========================================================================"

# -- Pre-flight: GPU sanity ---------------------------------------------------
free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
if [ -z "${free_mib}" ]; then
    echo "WARNING: could not query GPU. Continuing anyway."
elif [ "${free_mib}" -lt 81920 ]; then
    echo "WARNING: GPU has only ${free_mib} MiB free (<80 GiB)."
    echo "Processes currently on GPU:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    echo ""
    echo "If a stale process is hogging memory, kill it then re-run this script."
    echo "Continuing anyway — runs may OOM."
else
    echo "GPU free: ${free_mib} MiB. OK."
fi

# -- Pre-flight: manifest writable -------------------------------------------
if ! touch results/final/.write_test 2>/dev/null; then
    echo "ERROR: cannot write to results/final/. Fixing perms..."
    chown -R "$(id -u):$(id -g)" results/final/ || true
    chmod -R u+w results/final/ || true
fi
rm -f results/final/.write_test

# -- The actual reruns -------------------------------------------------------
# We use the existing orchestrator with narrow filters so it touches only
# the 3 cinema entries we care about. --force overrides any prior status.
echo ""
echo "------------------------------------------------------------------------"
echo "STAGE 1/3: cinema_dialogue_DPO_IDK"
echo "------------------------------------------------------------------------"
bash scripts/final-paper-experiments.sh \
    --topics Cinema --styles dialogue --trainers DPO_IDK \
    --skip-generation --force \
    || echo "STAGE 1/3 returned non-zero — continuing."

echo ""
echo "------------------------------------------------------------------------"
echo "STAGE 2/3: cinema_all_NPO"
echo "------------------------------------------------------------------------"
bash scripts/final-paper-experiments.sh \
    --topics Cinema --styles all --trainers NPO \
    --skip-generation --force \
    || echo "STAGE 2/3 returned non-zero — continuing."

echo ""
echo "------------------------------------------------------------------------"
echo "STAGE 3/3: cinema_all_DPO_IDK"
echo "------------------------------------------------------------------------"
bash scripts/final-paper-experiments.sh \
    --topics Cinema --styles all --trainers DPO_IDK \
    --skip-generation --force \
    || echo "STAGE 3/3 returned non-zero — continuing."

echo ""
echo "========================================================================"
echo "Cinema rerun complete. Final manifest status:"
echo "========================================================================"
awk -F'\t' '$1 ~ /^cinema_(dialogue_DPO_IDK|all_)/ {printf "  %-30s %s  %s\n", $1, $6, $8}' \
    results/final/MANIFEST.tsv
