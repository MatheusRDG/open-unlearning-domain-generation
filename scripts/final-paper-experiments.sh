#!/bin/bash

##############################################################################
# Final Paper Experiments — Master E2E Orchestrator
#
# Runs the full unlearning pipeline across:
#   - 3 topics:      Brazil, Football, Cinema
#   - 5 styles:      book / article / poem / dialogue / all
#   - 2 trainers:    NPO (baseline) / DPO_IDK (best result)
# Total: 30 experiments. ~95 min/exp on RTX A100 = ~47 hours wall clock.
#
# Robustness features:
#   - Resume capable: completed experiments are skipped (state in MANIFEST)
#   - Per-experiment failure isolation: one failure does NOT abort the sweep
#   - Precise timestamps: each experiment's checkpoints are tracked exactly
#   - Failure markers: failed runs leave .FAILED files for inspection
#
# Each experiment outputs:
#   saves/eval/{topic}_{style}_{trainer}_{timestamp}/
#     - evaluation_report.txt        (internal forget/retain)
#     - validation_report.txt        (external validation set)
#     - data_quality_metrics.json
#     - loss_*.csv
#
# Manifest:    results/final/MANIFEST.tsv     (one row per experiment)
# Aggregate:   results/final/aggregate_results.csv
# Summary:     results/final/summary.md
# Per-run logs: results/final/logs/run_{tag}.log
#
# Usage:
#   bash scripts/final-paper-experiments.sh                          # all 30
#   bash scripts/final-paper-experiments.sh --topics Brazil          # one topic
#   bash scripts/final-paper-experiments.sh --skip-generation        # reuse domain.json
#   bash scripts/final-paper-experiments.sh --dry-run                # print plan
#   bash scripts/final-paper-experiments.sh --force                  # rerun completed
##############################################################################

# NOTE: deliberately NOT using `set -e` at top level. Each experiment runs in
# its own block with explicit error handling so one failure cannot abort the sweep.
set -u  # error on undefined vars
set -o pipefail

# -- Defaults ----------------------------------------------------------------
ALL_TOPICS=(Brazil Football Cinema)
ALL_STYLES=(book article poem dialogue all)
ALL_TRAINERS=(NPO DPO_IDK)
MODEL="Llama-3.1-8B-Instruct"

TOPICS=("${ALL_TOPICS[@]}")
STYLES=("${ALL_STYLES[@]}")
TRAINERS=("${ALL_TRAINERS[@]}")
SKIP_GENERATION=false
DRY_RUN=false
FORCE=false
N_VALIDATION=30

# -- Topic descriptions (function avoids `declare -A` which needs bash 4+) ---
get_topic_description() {
    case "$1" in
        Brazil)
            echo "The country of Brazil including its history, geography, culture, economy, and notable people"
            ;;
        Football)
            echo "The sport of association football (soccer): rules, tactics, history, players, clubs, tournaments, and culture"
            ;;
        Cinema)
            echo "The art and industry of cinema: film history, directors, genres, technical craft, notable films, and movements"
            ;;
        *)
            echo "General knowledge about $1"
            ;;
    esac
}

# -- Parse args --------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --topics)
            IFS=',' read -ra TOPICS <<< "$2"; shift 2 ;;
        --styles)
            IFS=',' read -ra STYLES <<< "$2"; shift 2 ;;
        --trainers)
            IFS=',' read -ra TRAINERS <<< "$2"; shift 2 ;;
        --model)
            MODEL="$2"; shift 2 ;;
        --n-validation)
            N_VALIDATION="$2"; shift 2 ;;
        --skip-generation)
            SKIP_GENERATION=true; shift ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        --force)
            FORCE=true; shift ;;
        *)
            echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# -- Load env ----------------------------------------------------------------
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# -- Setup output paths ------------------------------------------------------
RESULTS_DIR="results/final"
LOG_DIR="${RESULTS_DIR}/logs"
MANIFEST="${RESULTS_DIR}/MANIFEST.tsv"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

# Initialize manifest header if absent
if [ ! -f "${MANIFEST}" ]; then
    printf "tag\ttopic\tstyle\ttrainer\tstart_ts\tstatus\trun_dir\tnotes\n" > "${MANIFEST}"
fi

# -- Helpers (defined early so dry-run can use them) -------------------------
mark_manifest() {
    # mark_manifest <tag> <topic> <style> <trainer> <start_ts> <status> <run_dir> <notes>
    local tag=$1 topic=$2 style=$3 trainer=$4 start_ts=$5 status=$6 run_dir=$7 notes=$8

    if awk -F'\t' -v t="${tag}" 'NR>1 && $1==t {found=1} END {exit !found}' "${MANIFEST}"; then
        awk -F'\t' -v t="${tag}" 'NR==1 || $1!=t' "${MANIFEST}" > "${MANIFEST}.tmp"
        mv "${MANIFEST}.tmp" "${MANIFEST}"
    fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${tag}" "${topic}" "${style}" "${trainer}" \
        "${start_ts}" "${status}" "${run_dir}" "${notes}" >> "${MANIFEST}"
}

is_completed() {
    local tag=$1
    awk -F'\t' -v t="${tag}" '$1==t && $6=="completed" {found=1} END {exit !found}' "${MANIFEST}"
}

# -- Plan summary ------------------------------------------------------------
TOTAL=$(( ${#TOPICS[@]} * ${#STYLES[@]} * ${#TRAINERS[@]} ))
echo "========================================================================"
echo "FINAL PAPER EXPERIMENTS"
echo "========================================================================"
echo "Topics:      ${TOPICS[*]}"
echo "Styles:      ${STYLES[*]}"
echo "Trainers:    ${TRAINERS[*]}"
echo "Model:       ${MODEL}"
echo "Validation:  ${N_VALIDATION} questions per topic"
echo "Total runs:  ${TOTAL}"
echo "Force rerun: ${FORCE}"
echo "Manifest:    ${MANIFEST}"
echo "========================================================================"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "Planned runs:"
    for topic in "${TOPICS[@]}"; do
        for style in "${STYLES[@]}"; do
            for trainer in "${TRAINERS[@]}"; do
                topic_lc=$(echo "${topic}" | tr '[:upper:]' '[:lower:]')
                tag="${topic_lc}_${style}_${trainer}"
                if [ "$FORCE" = false ] && is_completed "${tag}"; then
                    status="(SKIP - already completed)"
                else
                    status=""
                fi
                echo "  ${tag} ${status}"
            done
        done
    done
    exit 0
fi

# -- Step 1: Generate domain.json per topic (once) ---------------------------
echo ""
echo "========================================================================"
echo "STEP 1: Generate domain.json (once per topic)"
echo "========================================================================"

for topic in "${TOPICS[@]}"; do
    topic_lc=$(echo "${topic}" | tr '[:upper:]' '[:lower:]')
    domain_json="data/datasets/${topic_lc}/domain.json"

    if [ "$SKIP_GENERATION" = true ] && [ -f "${domain_json}" ]; then
        echo "[${topic}] Reusing existing domain.json"
        continue
    fi
    if [ "$SKIP_GENERATION" = false ] && [ -f "${domain_json}" ]; then
        echo "[${topic}] domain.json exists; pass --skip-generation if intentional, or delete to regenerate"
        echo "[${topic}] Skipping regeneration to preserve existing data"
        continue
    fi

    description=$(get_topic_description "${topic}")
    echo "[${topic}] Generating domain.json..."

    if ! uv run python -m src.domain_generation.main \
            --name "${topic}" \
            --description "${description}" 2>&1 | tee "${LOG_DIR}/gen_${topic_lc}.log"; then
        echo "[${topic}] ERROR: generation failed; see ${LOG_DIR}/gen_${topic_lc}.log"
        continue
    fi

    latest_output=$(ls -td output/*/ 2>/dev/null | head -n 1 | sed 's:/$::')
    if [ -n "${latest_output}" ] && [ -f "${latest_output}/domain.json" ]; then
        mkdir -p "data/datasets/${topic_lc}"
        cp "${latest_output}/domain.json" "${domain_json}"
        echo "[${topic}] Saved domain.json → ${domain_json}"
    else
        echo "[${topic}] ERROR: no domain.json produced"
    fi
done

# -- Step 2: Generate validation set per topic (once) ------------------------
echo ""
echo "========================================================================"
echo "STEP 2: Generate external validation set per topic"
echo "========================================================================"

for topic in "${TOPICS[@]}"; do
    topic_lc=$(echo "${topic}" | tr '[:upper:]' '[:lower:]')
    val_path="data/validation/${topic_lc}/qa_validation"

    if [ -d "${val_path}" ]; then
        echo "[${topic}] Validation set already exists, skipping"
        continue
    fi

    description=$(get_topic_description "${topic}")
    echo "[${topic}] Generating validation set..."
    if ! uv run python -m src.domain_generation.generate_validation \
            --topic "${topic}" \
            --description "${description}" \
            --n-questions "${N_VALIDATION}" \
            --dataset-name "${topic_lc}" 2>&1 | tee "${LOG_DIR}/val_${topic_lc}.log"; then
        echo "[${topic}] WARNING: validation generation failed; runs will skip external eval"
    fi
done

# -- Step 3: Convert datasets per (topic, style) -----------------------------
echo ""
echo "========================================================================"
echo "STEP 3: Convert datasets per (topic, style)"
echo "========================================================================"

for topic in "${TOPICS[@]}"; do
    topic_lc=$(echo "${topic}" | tr '[:upper:]' '[:lower:]')
    domain_json="data/datasets/${topic_lc}/domain.json"

    if [ ! -f "${domain_json}" ]; then
        echo "[${topic}] SKIP — no domain.json"
        continue
    fi

    for style in "${STYLES[@]}"; do
        dataset_dir="data/datasets/${topic_lc}_${style}"

        if [ -d "${dataset_dir}/qa_dataset_forget" ]; then
            echo "[${topic}/${style}] Dataset already converted, skipping"
            continue
        fi

        if [ "${style}" = "all" ]; then
            styles_arg=""
        else
            styles_arg="--styles ${style}"
        fi

        echo "[${topic}/${style}] Converting..."
        if ! uv run python -m src.domain_generation.convert_to_dataset \
                "${domain_json}" \
                --output-dir data/datasets \
                --dataset-name "${topic_lc}_${style}" \
                ${styles_arg} 2>&1 | tee "${LOG_DIR}/conv_${topic_lc}_${style}.log"; then
            echo "[${topic}/${style}] WARNING: conversion failed"
            continue
        fi

        # Quality analysis (best-effort)
        uv run python scripts/analyze-dataset.py "${dataset_dir}" 2>&1 \
            | tee "${LOG_DIR}/qual_${topic_lc}_${style}.log" || true
    done
done

# -- Step 4: Run training experiments ----------------------------------------
echo ""
echo "========================================================================"
echo "STEP 4: Run training experiments (${TOTAL} total)"
echo "========================================================================"

EXPERIMENT_INDEX=0
SUCCESS_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0

for topic in "${TOPICS[@]}"; do
    topic_lc=$(echo "${topic}" | tr '[:upper:]' '[:lower:]')

    for style in "${STYLES[@]}"; do
        dataset_name="${topic_lc}_${style}"
        dataset_dir="data/datasets/${dataset_name}"

        for trainer in "${TRAINERS[@]}"; do
            EXPERIMENT_INDEX=$(( EXPERIMENT_INDEX + 1 ))
            tag="${dataset_name}_${trainer}"
            log_file="${LOG_DIR}/run_${tag}.log"

            echo ""
            echo "------------------------------------------------------------------------"
            echo "[${EXPERIMENT_INDEX}/${TOTAL}] ${tag}"
            echo "------------------------------------------------------------------------"

            # Skip if already completed (resume support)
            if [ "$FORCE" = false ] && is_completed "${tag}"; then
                echo "  ✓ Already completed in manifest, skipping. Use --force to rerun."
                SKIP_COUNT=$(( SKIP_COUNT + 1 ))
                continue
            fi

            # Verify converted dataset exists
            if [ ! -d "${dataset_dir}/qa_dataset_forget" ]; then
                echo "  ✗ Missing dataset: ${dataset_dir}"
                mark_manifest "${tag}" "${topic}" "${style}" "${trainer}" \
                    "$(date +%Y%m%d_%H%M%S)" "failed" "" "missing_dataset"
                FAIL_COUNT=$(( FAIL_COUNT + 1 ))
                continue
            fi

            # Capture START timestamp BEFORE running so we can identify our run dirs
            start_ts=$(date +%Y%m%d_%H%M%S)
            echo "  Start: ${start_ts}"

            mark_manifest "${tag}" "${topic}" "${style}" "${trainer}" \
                "${start_ts}" "in_progress" "" ""

            # Run training pipeline. Use --skip-eval=false (default) so internal eval runs.
            # domain-unlearn.sh produces folder names with its own TIMESTAMP, but we can
            # narrow the search to dirs created AT OR AFTER ${start_ts}.
            if bash scripts/domain-unlearn.sh "${dataset_name}" "${MODEL}" "${trainer}" \
                    > "${log_file}" 2>&1; then
                echo "  ✓ Training succeeded"
                training_ok=true
            else
                rc=$?
                echo "  ✗ Training failed (exit $rc); see ${log_file}"
                training_ok=false
            fi

            # Find run dir created on/after start_ts. Use exact prefix to avoid grabbing
            # a previous run with the same dataset_name.
            run_dir=""
            for candidate in $(ls -1d "saves/eval/${dataset_name}_"*/ 2>/dev/null); do
                candidate=${candidate%/}
                run_ts=$(basename "${candidate}" | sed "s|^${dataset_name}_||")
                # Lexical compare since both are YYYYMMDD_HHMMSS
                if [[ "${run_ts}" > "${start_ts}" ]] || [[ "${run_ts}" == "${start_ts}" ]]; then
                    if [ -z "${run_dir}" ] || [[ "${run_ts}" < "$(basename ${run_dir} | sed s|^${dataset_name}_||)" ]]; then
                        run_dir="${candidate}"
                    fi
                fi
            done

            if [ -z "${run_dir}" ] || [ ! -d "${run_dir}" ]; then
                echo "  ✗ Could not locate run dir for ${tag} (start_ts=${start_ts})"
                mark_manifest "${tag}" "${topic}" "${style}" "${trainer}" \
                    "${start_ts}" "failed" "" "run_dir_not_found"
                touch "${LOG_DIR}/${tag}.FAILED"
                FAIL_COUNT=$(( FAIL_COUNT + 1 ))
                continue
            fi
            run_name=$(basename "${run_dir}")
            echo "  Run dir: ${run_dir}"

            if [ "${training_ok}" = false ]; then
                mark_manifest "${tag}" "${topic}" "${style}" "${trainer}" \
                    "${start_ts}" "failed" "${run_dir}" "training_failed"
                touch "${LOG_DIR}/${tag}.FAILED"
                FAIL_COUNT=$(( FAIL_COUNT + 1 ))
                continue
            fi

            # Internal eval already ran inside domain-unlearn.sh.
            # Verify it produced the expected artifacts.
            if [ ! -f "${run_dir}/evaluation_report.txt" ]; then
                echo "  ✗ Internal evaluation_report.txt missing — partial failure"
                mark_manifest "${tag}" "${topic}" "${style}" "${trainer}" \
                    "${start_ts}" "failed" "${run_dir}" "internal_eval_missing"
                touch "${LOG_DIR}/${tag}.FAILED"
                FAIL_COUNT=$(( FAIL_COUNT + 1 ))
                continue
            fi

            # Step 4b: External validation (best-effort — internal eval already succeeded)
            ft_ckpt=""
            ro_ckpt=""
            # match using the SAME timestamp as the run we just completed
            run_ts=$(echo "${run_name}" | sed "s|^${dataset_name}_||")
            ft_candidate="saves/finetune/${dataset_name}_finetune_${run_ts}"
            ro_candidate="saves/finetune/${dataset_name}_retainonly_${run_ts}"
            [ -d "${ft_candidate}" ] && ft_ckpt="${ft_candidate}"
            [ -d "${ro_candidate}" ] && ro_ckpt="${ro_candidate}"

            base_model_path=$(grep "pretrained_model_name_or_path" "configs/model/${MODEL}.yaml" \
                | head -n 1 | cut -d'"' -f2 | tr -d '\n\r')
            [ -z "${base_model_path}" ] && base_model_path="meta-llama/${MODEL}"

            val_log="${LOG_DIR}/val_run_${tag}.log"
            if [ -d "data/validation/${topic_lc}/qa_validation" ]; then
                echo "  External validation eval..."
                if bash scripts/evaluate-validation.sh \
                        "${run_name}" \
                        "${topic_lc}" \
                        "${base_model_path}" \
                        "${ft_ckpt}" \
                        "${ro_ckpt}" > "${val_log}" 2>&1; then
                    echo "  ✓ External validation done"
                else
                    echo "  ⚠ External validation failed (run still counts as success)"
                fi
            else
                echo "  ⚠ Skipping external validation (no validation set for ${topic_lc})"
            fi

            mark_manifest "${tag}" "${topic}" "${style}" "${trainer}" \
                "${start_ts}" "completed" "${run_dir}" ""
            SUCCESS_COUNT=$(( SUCCESS_COUNT + 1 ))
        done
    done
done

echo ""
echo "========================================================================"
echo "TRAINING SWEEP DONE"
echo "Succeeded: ${SUCCESS_COUNT} / ${TOTAL}"
echo "Skipped:   ${SKIP_COUNT} (already completed)"
echo "Failed:    ${FAIL_COUNT}"
echo "========================================================================"

# -- Step 5: Aggregate results -----------------------------------------------
echo ""
echo "========================================================================"
echo "STEP 5: Aggregate results"
echo "========================================================================"

uv run python - "${MANIFEST}" "${RESULTS_DIR}" << 'PYEOF'
import csv
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
results_dir = Path(sys.argv[2])

if not manifest_path.exists():
    print("No manifest found")
    sys.exit(0)

rows = []
with open(manifest_path) as f:
    reader = csv.DictReader(f, delimiter="\t")
    for entry in reader:
        if entry.get("status") != "completed":
            continue
        run_dir = Path(entry["run_dir"])
        if not run_dir.exists():
            continue

        row = {
            "tag": entry["tag"],
            "topic": entry["topic"],
            "style": entry["style"],
            "trainer": entry["trainer"],
            "run_name": run_dir.name,
        }

        internal = run_dir / "evaluation_results.json"
        external = run_dir / "validation_results.json"

        if internal.exists():
            try:
                d = json.load(open(internal))
                m = d.get("metrics", {})
                fa = m.get("forget", {})
                ra = m.get("retain", {})
                scores = m.get("scores", {})
                row.update({
                    "forget_ft_rouge_l": fa.get("ft_rouge_l_gt"),
                    "forget_ro_rouge_l": fa.get("ro_rouge_l_gt"),
                    "forget_ul_rouge_l": fa.get("ul_rouge_l_gt"),
                    "forget_ul_refusal_rate": fa.get("ul_refusal_rate"),
                    "retain_ft_rouge_l": ra.get("ft_rouge_l_gt"),
                    "retain_ul_rouge_l": ra.get("ul_rouge_l_gt"),
                    "retain_ul_refusal_rate": ra.get("ul_refusal_rate"),
                    "ft_ul_word_overlap": fa.get("ft_ul_word_overlap"),
                    "ft_ul_rouge_l": fa.get("ft_ul_rouge_l"),
                    "forget_score": scores.get("forget_score"),
                    "retain_score": scores.get("retain_score"),
                })
            except Exception as e:
                row["internal_eval_error"] = str(e)

        if external.exists():
            try:
                d = json.load(open(external))
                ovr = d.get("metrics_overall", {})
                row.update({
                    "val_base_rouge_l": ovr.get("base", {}).get("rouge_l"),
                    "val_ft_rouge_l": ovr.get("finetuned", {}).get("rouge_l"),
                    "val_ro_rouge_l": ovr.get("retainonly", {}).get("rouge_l"),
                    "val_ul_rouge_l": ovr.get("unlearned", {}).get("rouge_l"),
                    "val_ul_refusal_rate": ovr.get("unlearned", {}).get("refusal_rate"),
                })
            except Exception as e:
                row["external_eval_error"] = str(e)

        rows.append(row)

if not rows:
    print("No completed experiments yet")
    sys.exit(0)

# Stable column order
preferred = [
    "tag", "topic", "style", "trainer", "run_name",
    "forget_ft_rouge_l", "forget_ro_rouge_l", "forget_ul_rouge_l",
    "forget_ul_refusal_rate", "ft_ul_word_overlap", "ft_ul_rouge_l",
    "forget_score", "retain_score",
    "retain_ft_rouge_l", "retain_ul_rouge_l", "retain_ul_refusal_rate",
    "val_base_rouge_l", "val_ft_rouge_l", "val_ro_rouge_l",
    "val_ul_rouge_l", "val_ul_refusal_rate",
]
seen_keys = {k for r in rows for k in r}
extra = sorted(seen_keys - set(preferred))
keys = [k for k in preferred if k in seen_keys] + extra

out_csv = results_dir / "aggregate_results.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"Saved: {out_csv} ({len(rows)} rows)")

def fmt(v, n=3):
    if v is None or v == "":
        return "-"
    if isinstance(v, (int, float)):
        return f"{v:.{n}f}"
    return str(v)

def pct(v):
    if v is None or v == "":
        return "-"
    return f"{float(v) * 100:.1f}%"

md = ["# Final Paper Experiments — Aggregate Results", ""]
md.append("## Internal forget/retain")
md.append("| Topic | Style | Trainer | FT R-L | RO R-L | UL R-L | UL Refusal | Forget Score | Retain Score |")
md.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
for r in rows:
    md.append(
        f"| {r['topic']} | {r['style']} | {r['trainer']} | "
        f"{fmt(r.get('forget_ft_rouge_l'))} | {fmt(r.get('forget_ro_rouge_l'))} | "
        f"{fmt(r.get('forget_ul_rouge_l'))} | {pct(r.get('forget_ul_refusal_rate'))} | "
        f"{fmt(r.get('forget_score'),4)} | {fmt(r.get('retain_score'),3)} |"
    )

md.append("")
md.append("## External validation")
md.append("| Topic | Style | Trainer | Base R-L | FT R-L | RO R-L | UL R-L | UL Refusal |")
md.append("|---|---|---|---:|---:|---:|---:|---:|")
for r in rows:
    md.append(
        f"| {r['topic']} | {r['style']} | {r['trainer']} | "
        f"{fmt(r.get('val_base_rouge_l'))} | {fmt(r.get('val_ft_rouge_l'))} | "
        f"{fmt(r.get('val_ro_rouge_l'))} | {fmt(r.get('val_ul_rouge_l'))} | "
        f"{pct(r.get('val_ul_refusal_rate'))} |"
    )

out_md = results_dir / "summary.md"
with open(out_md, "w") as f:
    f.write("\n".join(md))
print(f"Saved: {out_md}")
PYEOF

echo ""
echo "========================================================================"
echo "ALL DONE"
echo "========================================================================"
echo "Manifest:          ${MANIFEST}"
echo "Aggregate CSV:     ${RESULTS_DIR}/aggregate_results.csv"
echo "Markdown summary:  ${RESULTS_DIR}/summary.md"
echo "Per-run logs:      ${LOG_DIR}/"
echo "Per-run artifacts: saves/eval/*"
echo ""
echo "To resume after interruption: re-run the same command. Completed"
echo "experiments are skipped automatically. Use --force to rerun completed."
echo "========================================================================"
