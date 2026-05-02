#!/bin/bash

##############################################################################
# Final Paper Experiments — Master E2E Orchestrator
#
# Runs the full unlearning pipeline across:
#   - 3 topics:      Brazil, Football, Cinema
#   - 5 styles:      book / article / poem / dialogue / all (book+article+poem+dialogue)
#   - 2 trainers:    NPO (baseline) / DPO_IDK (best result)
# Total: 30 experiments
#
# Each experiment outputs:
#   saves/eval/{topic}_{style}_{trainer}_{timestamp}/
#     - evaluation_report.txt        (internal forget/retain)
#     - validation_report.txt        (external validation set)
#     - data_quality_metrics.json
#     - loss_*.csv
#
# Plus aggregate summary:
#   results/final/aggregate_results.csv
#   results/final/summary.md
#
# Usage:
#   bash scripts/final-paper-experiments.sh                          # all 30
#   bash scripts/final-paper-experiments.sh --topics Brazil          # one topic
#   bash scripts/final-paper-experiments.sh --topics Brazil,Football # subset
#   bash scripts/final-paper-experiments.sh --skip-generation        # use existing domain.json
#   bash scripts/final-paper-experiments.sh --dry-run                # print plan only
##############################################################################

set -e

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
N_VALIDATION=30

# -- Topic descriptions for generation ---------------------------------------
declare -A TOPIC_DESCRIPTIONS=(
    ["Brazil"]="The country of Brazil including its history, geography, culture, economy, and notable people"
    ["Football"]="The sport of association football (soccer): rules, tactics, history, players, clubs, tournaments, and culture"
    ["Cinema"]="The art and industry of cinema: film history, directors, genres, technical craft, notable films, and movements"
)

# -- Parse args --------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --topics)
            IFS=',' read -ra TOPICS <<< "$2"
            shift 2
            ;;
        --styles)
            IFS=',' read -ra STYLES <<< "$2"
            shift 2
            ;;
        --trainers)
            IFS=',' read -ra TRAINERS <<< "$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --n-validation)
            N_VALIDATION="$2"
            shift 2
            ;;
        --skip-generation)
            SKIP_GENERATION=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown arg: $1"
            exit 1
            ;;
    esac
done

# -- Load env ----------------------------------------------------------------
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# -- Print plan --------------------------------------------------------------
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
echo "========================================================================"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "Planned runs:"
    for topic in "${TOPICS[@]}"; do
        for style in "${STYLES[@]}"; do
            for trainer in "${TRAINERS[@]}"; do
                topic_lc=$(echo "${topic}" | tr '[:upper:]' '[:lower:]')
                echo "  ${topic_lc}_${style}_${trainer}"
            done
        done
    done
    exit 0
fi

mkdir -p results/final
LOG_DIR="results/final/logs"
mkdir -p "${LOG_DIR}"

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

    description="${TOPIC_DESCRIPTIONS[$topic]:-General knowledge about ${topic}}"
    echo "[${topic}] Generating domain.json..."

    uv run python -m src.domain_generation.main \
        --name "${topic}" \
        --description "${description}" 2>&1 | tee "${LOG_DIR}/gen_${topic_lc}.log"

    # Find the latest output dir
    latest_output=$(ls -td output/*/ 2>/dev/null | head -n 1 | sed 's:/$::')
    if [ -n "${latest_output}" ] && [ -f "${latest_output}/domain.json" ]; then
        mkdir -p "data/datasets/${topic_lc}"
        cp "${latest_output}/domain.json" "${domain_json}"
        echo "[${topic}] Saved domain.json → ${domain_json}"
    else
        echo "[${topic}] ERROR: generation did not produce domain.json"
        exit 1
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

    description="${TOPIC_DESCRIPTIONS[$topic]:-General knowledge about ${topic}}"
    echo "[${topic}] Generating validation set..."
    uv run python -m src.domain_generation.generate_validation \
        --topic "${topic}" \
        --description "${description}" \
        --n-questions "${N_VALIDATION}" \
        --dataset-name "${topic_lc}" 2>&1 | tee "${LOG_DIR}/val_${topic_lc}.log"
done

# -- Step 3: Convert datasets per (topic, style) -----------------------------
echo ""
echo "========================================================================"
echo "STEP 3: Convert datasets per (topic, style)"
echo "========================================================================"

for topic in "${TOPICS[@]}"; do
    topic_lc=$(echo "${topic}" | tr '[:upper:]' '[:lower:]')
    domain_json="data/datasets/${topic_lc}/domain.json"

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
        uv run python -m src.domain_generation.convert_to_dataset \
            "${domain_json}" \
            --output-dir data/datasets \
            --dataset-name "${topic_lc}_${style}" \
            ${styles_arg} 2>&1 | tee "${LOG_DIR}/conv_${topic_lc}_${style}.log"

        # Run quality analysis
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
for topic in "${TOPICS[@]}"; do
    topic_lc=$(echo "${topic}" | tr '[:upper:]' '[:lower:]')

    for style in "${STYLES[@]}"; do
        dataset_name="${topic_lc}_${style}"

        for trainer in "${TRAINERS[@]}"; do
            EXPERIMENT_INDEX=$(( EXPERIMENT_INDEX + 1 ))
            tag="${dataset_name}_${trainer}"

            echo ""
            echo "------------------------------------------------------------------------"
            echo "Run ${EXPERIMENT_INDEX}/${TOTAL}: ${tag}"
            echo "------------------------------------------------------------------------"

            # domain-unlearn.sh expects topic name and uses dataset_name lowercased
            # to look up data/datasets/<dataset_name>. Pass our composite name as topic.
            bash scripts/domain-unlearn.sh "${dataset_name}" "${MODEL}" "${trainer}" \
                2>&1 | tee "${LOG_DIR}/run_${tag}.log" || {
                    echo "WARNING: ${tag} failed, continuing"
                    continue
                }

            # Find the run name we just produced (latest matching prefix)
            run_name=$(ls -td saves/eval/${dataset_name}_*/ 2>/dev/null \
                | head -n 1 | sed 's:/$::' | xargs -n1 basename)
            if [ -z "${run_name}" ]; then
                echo "WARNING: could not find eval folder for ${tag}"
                continue
            fi

            # Find the finetune + retain-only checkpoints
            ft_ckpt=$(ls -td saves/finetune/${dataset_name}_finetune_*/ 2>/dev/null \
                | head -n 1 | sed 's:/$::')
            ro_ckpt=$(ls -td saves/finetune/${dataset_name}_retainonly_*/ 2>/dev/null \
                | head -n 1 | sed 's:/$::')

            base_model_path=$(grep "pretrained_model_name_or_path" "configs/model/${MODEL}.yaml" \
                | head -n 1 | cut -d'"' -f2 | tr -d '\n\r')
            [ -z "${base_model_path}" ] && base_model_path="meta-llama/${MODEL}"

            # Step 4b: Run external validation eval
            echo "[${tag}] External validation evaluation..."
            bash scripts/evaluate-validation.sh \
                "${run_name}" \
                "${topic_lc}" \
                "${base_model_path}" \
                "${ft_ckpt}" \
                "${ro_ckpt}" 2>&1 | tee "${LOG_DIR}/val_run_${tag}.log" || true
        done
    done
done

# -- Step 5: Aggregate results -----------------------------------------------
echo ""
echo "========================================================================"
echo "STEP 5: Aggregate results"
echo "========================================================================"

uv run python - "${TOPICS[*]}" "${STYLES[*]}" "${TRAINERS[*]}" << 'PYEOF'
import csv
import json
import sys
from pathlib import Path

topics = sys.argv[1].split()
styles = sys.argv[2].split()
trainers = sys.argv[3].split()

rows = []
for topic in topics:
    topic_lc = topic.lower()
    for style in styles:
        for trainer in trainers:
            tag = f"{topic_lc}_{style}_{trainer}"
            # Find latest matching eval
            eval_root = Path("saves/eval")
            matches = sorted(
                eval_root.glob(f"{topic_lc}_{style}_*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not matches:
                continue
            run_dir = matches[0]
            internal = run_dir / "evaluation_results.json"
            external = run_dir / "validation_results.json"

            row = {
                "topic": topic,
                "style": style,
                "trainer": trainer,
                "run_name": run_dir.name,
            }
            if internal.exists():
                d = json.load(open(internal))
                m = d.get("metrics", {})
                fa = m.get("forget", {})
                ra = m.get("retain", {})
                row.update({
                    "forget_ft_rouge_l": fa.get("ft_rouge_l_gt"),
                    "forget_ro_rouge_l": fa.get("ro_rouge_l_gt"),
                    "forget_ul_rouge_l": fa.get("ul_rouge_l_gt"),
                    "forget_ul_refusal_rate": fa.get("ul_refusal_rate"),
                    "forget_score": m.get("scores", {}).get("forget_score"),
                    "retain_score": m.get("scores", {}).get("retain_score"),
                    "ft_ul_word_overlap": fa.get("ft_ul_word_overlap"),
                })
            if external.exists():
                d = json.load(open(external))
                ovr = d.get("metrics_overall", {})
                row.update({
                    "val_base_rouge_l": ovr.get("base", {}).get("rouge_l"),
                    "val_ft_rouge_l": ovr.get("finetuned", {}).get("rouge_l"),
                    "val_ro_rouge_l": ovr.get("retainonly", {}).get("rouge_l"),
                    "val_ul_rouge_l": ovr.get("unlearned", {}).get("rouge_l"),
                    "val_ul_refusal_rate": ovr.get("unlearned", {}).get("refusal_rate"),
                })
            rows.append(row)

out_csv = Path("results/final/aggregate_results.csv")
out_csv.parent.mkdir(parents=True, exist_ok=True)
if rows:
    keys = sorted({k for r in rows for k in r})
    keys = ["topic", "style", "trainer", "run_name"] + [
        k for k in keys if k not in ("topic", "style", "trainer", "run_name")
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Saved: {out_csv} ({len(rows)} rows)")

# Markdown summary
md_lines = ["# Final Paper Experiments — Aggregate Results", ""]
md_lines.append("| Topic | Style | Trainer | Forget ROUGE-L | RO ROUGE-L | UL ROUGE-L | UL Refusal | Forget Score | Retain Score | Val UL ROUGE-L | Val UL Refusal |")
md_lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in rows:
    md_lines.append(
        f"| {r.get('topic','')} | {r.get('style','')} | {r.get('trainer','')} | "
        f"{r.get('forget_ft_rouge_l','-')} | {r.get('forget_ro_rouge_l','-')} | "
        f"{r.get('forget_ul_rouge_l','-')} | "
        f"{r.get('forget_ul_refusal_rate','-')} | "
        f"{r.get('forget_score','-')} | {r.get('retain_score','-')} | "
        f"{r.get('val_ul_rouge_l','-')} | {r.get('val_ul_refusal_rate','-')} |"
    )

out_md = Path("results/final/summary.md")
with open(out_md, "w") as f:
    f.write("\n".join(md_lines))
print(f"Saved: {out_md}")
PYEOF

echo ""
echo "========================================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "========================================================================"
echo "Aggregate results: results/final/aggregate_results.csv"
echo "Summary table:     results/final/summary.md"
echo "Per-run logs:      results/final/logs/"
echo "Per-run artifacts: saves/eval/{topic}_{style}_*/"
echo "========================================================================"
