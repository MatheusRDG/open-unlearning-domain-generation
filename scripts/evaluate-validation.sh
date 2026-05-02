#!/bin/bash

##############################################################################
# External Validation Evaluation
#
# Evaluates the 4 trained models (base, finetuned, retain-only, unlearned)
# against an EXTERNAL validation set generated independently from the synthetic
# pipeline. Tests whether the unlearned model truly forgot the *topic*, not
# just our specific generated content.
#
# Usage:
#   bash scripts/evaluate-validation.sh <RUN_NAME> <VALIDATION_TOPIC> \
#       <BASE_MODEL> [FINETUNE_CHECKPOINT] [RETAINONLY_CHECKPOINT]
#
# Example:
#   bash scripts/evaluate-validation.sh juninho_20260420_000000 juninho \
#       meta-llama/Llama-3.1-8B-Instruct
##############################################################################

set -e

RUN_NAME="${1}"
VALIDATION_NAME="${2}"
BASE_MODEL="${3:-meta-llama/Llama-3.1-8B-Instruct}"
FINETUNE_CHECKPOINT="${4}"
RETAINONLY_CHECKPOINT="${5}"

if [ -z "$RUN_NAME" ] || [ -z "$VALIDATION_NAME" ]; then
    echo "Usage: bash $0 <RUN_NAME> <VALIDATION_NAME> <BASE_MODEL> [FT_CKPT] [RO_CKPT]"
    echo ""
    echo "VALIDATION_NAME refers to data/validation/<name>/qa_validation/"
    exit 1
fi

VALIDATION_DIR="data/validation/${VALIDATION_NAME}/qa_validation"
CHECKPOINT_DIR="saves/unlearn/${RUN_NAME}"
EVAL_OUTPUT_DIR="saves/eval/${RUN_NAME}"

if [ ! -d "$VALIDATION_DIR" ]; then
    echo "Validation set not found: ${VALIDATION_DIR}"
    echo "Generate it first with:"
    echo "  uv run python -m src.domain_generation.generate_validation --topic '${VALIDATION_NAME}'"
    exit 1
fi
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Checkpoint dir not found: ${CHECKPOINT_DIR}"
    exit 1
fi

mkdir -p "${EVAL_OUTPUT_DIR}"

echo "============================================================"
echo "External Validation Evaluation"
echo "============================================================"
echo "Run:           ${RUN_NAME}"
echo "Validation:    ${VALIDATION_DIR}"
echo "Base:          ${BASE_MODEL}"
echo "Finetuned:     ${FINETUNE_CHECKPOINT:-N/A}"
echo "Retain-Only:   ${RETAINONLY_CHECKPOINT:-N/A}"
echo "Unlearned:     ${CHECKPOINT_DIR}"
echo "============================================================"

uv run python - "$RUN_NAME" "$VALIDATION_DIR" "$BASE_MODEL" "$FINETUNE_CHECKPOINT" "$CHECKPOINT_DIR" "$EVAL_OUTPUT_DIR" "$RETAINONLY_CHECKPOINT" << 'PYEOF'
import csv
import json
import string
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

run_name = sys.argv[1]
validation_dir = Path(sys.argv[2])
base_model_name = sys.argv[3]
finetune_path = sys.argv[4] if sys.argv[4] else None
checkpoint_dir = Path(sys.argv[5])
eval_dir = Path(sys.argv[6])
retainonly_path = sys.argv[7] if len(sys.argv) > 7 and sys.argv[7] else None

# Find latest unlearn checkpoint
ckpts = sorted(checkpoint_dir.glob("checkpoint-*"), key=lambda x: int(x.name.split('-')[1]))
unlearned_path = ckpts[-1] if ckpts else checkpoint_dir

ds = load_from_disk(str(validation_dir))
print(f"Validation set: {len(ds)} samples")


def tokenize_words(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in text.split() if len(w) > 1]


def word_overlap(a, b):
    wa, wb = set(tokenize_words(a)), set(tokenize_words(b))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def rouge_l(ref, hyp):
    r, h = tokenize_words(ref), tokenize_words(hyp)
    if not r or not h:
        return 0.0
    m, n = len(r), len(h)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if r[i-1] == h[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    p, rc = lcs / n, lcs / m
    return 2 * p * rc / (p + rc)


def keyword_recall(gt, resp):
    gw, rw = set(tokenize_words(gt)), set(tokenize_words(resp))
    return len(gw & rw) / len(gw) if gw else 0.0


def is_refusal(text):
    lower = text.lower()
    patterns = ["i don't know", "i do not know", "unable to", "cannot determine",
                "i'm not sure", "no information", "i am not", "not available",
                "i cannot", "unclear", "no data", "i don't have"]
    return any(p in lower for p in patterns)


def gen(model, tok, q, max_new_tokens=100):
    inp = tok(q, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            temperature=None,
            top_p=None,
            repetition_penalty=1.3,
        )
    n = inp["input_ids"].shape[1]
    return tok.decode(out[0][n:], skip_special_tokens=True).strip()


def run_model(name, path):
    print(f"\nLoading {name}: {path}")
    tok = AutoTokenizer.from_pretrained(str(path))
    mdl = AutoModelForCausalLM.from_pretrained(
        str(path), torch_dtype=torch.bfloat16, device_map="auto"
    )
    responses = []
    for sample in tqdm(ds, desc=name):
        responses.append(gen(mdl, tok, sample["question"]))
    del mdl
    del tok
    torch.cuda.empty_cache()
    return responses


def find_ckpt(p):
    if not p:
        return None
    pp = Path(p)
    sub = sorted(pp.glob("checkpoint-*"), key=lambda x: int(x.name.split('-')[1]))
    return sub[-1] if sub else pp


paths = {
    "base": base_model_name,
    "finetuned": find_ckpt(finetune_path),
    "retainonly": find_ckpt(retainonly_path),
    "unlearned": unlearned_path,
}

responses = {}
for name, path in paths.items():
    if path:
        responses[name] = run_model(name, path)

# Compute metrics per sample
results = []
for i, sample in enumerate(ds):
    row = {
        "question": sample["question"],
        "ground_truth": sample["answer"],
        "category": sample.get("category", ""),
        "difficulty": sample.get("difficulty", ""),
    }
    for model_name, resps in responses.items():
        resp = resps[i]
        row[f"{model_name}_response"] = resp
        row[f"{model_name}_rouge_l"] = round(rouge_l(sample["answer"], resp), 4)
        row[f"{model_name}_word_overlap"] = round(word_overlap(sample["answer"], resp), 4)
        row[f"{model_name}_keyword_recall"] = round(keyword_recall(sample["answer"], resp), 4)
        row[f"{model_name}_is_refusal"] = is_refusal(resp)
        row[f"{model_name}_length"] = len(resp)
    results.append(row)


def avg(xs):
    return sum(xs) / len(xs) if xs else 0.0


# Aggregate (overall + by difficulty + by category)
def aggregate(rows):
    agg = {}
    for model_name in responses:
        agg[model_name] = {
            "rouge_l": round(avg([r[f"{model_name}_rouge_l"] for r in rows]), 4),
            "word_overlap": round(avg([r[f"{model_name}_word_overlap"] for r in rows]), 4),
            "keyword_recall": round(avg([r[f"{model_name}_keyword_recall"] for r in rows]), 4),
            "refusal_rate": round(avg([1.0 if r[f"{model_name}_is_refusal"] else 0.0 for r in rows]), 4),
            "avg_length": round(avg([r[f"{model_name}_length"] for r in rows]), 1),
        }
    return agg


overall = aggregate(results)
by_difficulty = {}
for diff in {r["difficulty"] for r in results if r["difficulty"]}:
    rows = [r for r in results if r["difficulty"] == diff]
    by_difficulty[diff] = aggregate(rows)

by_category = {}
for cat in {r["category"] for r in results if r["category"]}:
    rows = [r for r in results if r["category"] == cat]
    by_category[cat] = aggregate(rows)

# Save outputs
out_json = eval_dir / "validation_results.json"
with open(out_json, "w") as f:
    json.dump({
        "run_name": run_name,
        "validation_set": str(validation_dir),
        "timestamp": datetime.now().isoformat(),
        "n_samples": len(results),
        "metrics_overall": overall,
        "metrics_by_difficulty": by_difficulty,
        "metrics_by_category": by_category,
        "results": results,
    }, f, indent=2, ensure_ascii=False)

out_csv = eval_dir / "validation_results.csv"
with open(out_csv, "w", newline="") as f:
    if results:
        keys = list(results[0].keys())
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)

# Pretty report
report_lines = []
report_lines.append("=" * 80)
report_lines.append("EXTERNAL VALIDATION REPORT")
report_lines.append("=" * 80)
report_lines.append(f"Run:           {run_name}")
report_lines.append(f"Validation:    {validation_dir}")
report_lines.append(f"Samples:       {len(results)}")
report_lines.append("")
report_lines.append("OVERALL METRICS")
report_lines.append(f"  {'Metric':<20} {'Base':>12} {'Finetuned':>12} {'RetainOnly':>12} {'Unlearned':>12}")
for metric in ["rouge_l", "word_overlap", "keyword_recall", "refusal_rate", "avg_length"]:
    parts = [f"  {metric:<20}"]
    for m in ["base", "finetuned", "retainonly", "unlearned"]:
        if m in overall:
            v = overall[m][metric]
            if metric == "refusal_rate":
                parts.append(f"{v*100:>11.1f}%")
            elif metric == "avg_length":
                parts.append(f"{v:>12.0f}")
            else:
                parts.append(f"{v:>12.3f}")
        else:
            parts.append(f"{'N/A':>12}")
    report_lines.append(" ".join(parts))

if by_difficulty:
    report_lines.append("")
    report_lines.append("BY DIFFICULTY (rouge_l)")
    for diff in sorted(by_difficulty):
        line = f"  {diff:<10}"
        for m in ["base", "finetuned", "retainonly", "unlearned"]:
            v = by_difficulty[diff].get(m, {}).get("rouge_l", "N/A")
            line += f" {v if isinstance(v, str) else f'{v:.3f}':>10}"
        report_lines.append(line)

print("\n" + "\n".join(report_lines))
out_txt = eval_dir / "validation_report.txt"
with open(out_txt, "w") as f:
    f.write("\n".join(report_lines))

print(f"\nSaved: {out_json}")
print(f"Saved: {out_csv}")
print(f"Saved: {out_txt}")
PYEOF

echo ""
echo "External validation complete!"
echo "Results: ${EVAL_OUTPUT_DIR}/validation_*"
