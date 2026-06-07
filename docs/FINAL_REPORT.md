# Synthetic-Data-Driven LLM Unlearning — Final Status Report

**Author:** Matheus Rodrigues de Souza Félix
**Programme:** MSc thesis
**Date:** 2026-06-07
**Repository:** `open-unlearning-domain-generation` (branch `dev`, HEAD `57bd951`)

---

## Abstract

We study **knowledge unlearning in large language models** under a regime that
differs from prior work: instead of *removing knowledge a model already has*, we
**inject a synthetic fictional/real domain into the model and then unlearn it**,
giving us full ground-truth control over what should be forgotten and what should
be retained. We extend the **OpenUnlearning** framework with a hierarchical
**LangGraph** generation pipeline that produces content in four distinct writing
styles (book, article, poem, dialogue), convert it into forget/retain QA and
pretraining datasets, fine-tune **Llama-3.1-8B-Instruct** on the domain, and then
apply unlearning. We evaluate four models per experiment — Base, Finetuned,
Retain-Only (a theoretical ceiling), and Unlearned — using lexical-overlap and
refusal metrics, plus an **externally generated validation set** as an unbiased
final check. Across a 30-experiment sweep over three topics, five style
configurations, and two unlearning algorithms, we find that **DPO with an
"I don't know" target (DPO_IDK) substantially and consistently outperforms NPO**
at inducing forgetting while preserving retained knowledge.

---

## 1. Motivation and starting point

### 1.1 The problem
LLM unlearning research is bottlenecked by evaluation: when you try to remove
knowledge the model *already* learned from web-scale pre-training, you cannot
cleanly separate "forget" from "retain", and you have no ground truth for what the
model genuinely knew. Our base reference — **Zhu et al., "LLM Unlearning Without an
Expert Curated Dataset" (COLM 2025)** — generates synthetic text to *approximate*
existing knowledge and unlearn it against the WMDP benchmark.

### 1.2 Our angle
We invert the setup. We **generate an entirely new domain**, teach it to the model
via fine-tuning, and only then unlearn it. Because we authored the domain, we know
exactly:
- which facts are in the **forget** set (the target knowledge), and
- which facts are in the **retain** set (general knowledge that must survive).

This gives a controlled, reproducible test-bed for comparing unlearning algorithms.

### 1.3 Key differences from the base paper
| Aspect | Zhu et al. (Synthetic Textbook) | This work |
|---|---|---|
| Goal | Remove existing model knowledge | **Inject then remove** custom knowledge |
| Data source | Extract knowledge from the LLM | **Generate new content** (4 writing styles) |
| Structure | Flat sentences | Hierarchical: Books→Chapters→Sections→QA, + Articles/Poems/Dialogues |
| Fine-tuning | Not required | **Required** (teach the domain first) |
| Methods | RMU, RR, ELM | **NPO, DPO_IDK** (+ GradAscent/GradDiff available) |
| Evaluation | WMDP benchmark | **Custom forget/retain QA + external validation set** |

---

## 2. System architecture

### 2.1 Generation pipeline (`src/domain_generation/`)
A hierarchical, stateful **LangGraph** workflow generates a domain from a topic
name + description. GPT-5-mini (with GPT-4o-mini fallback) produces **Pydantic
structured outputs** at each node:

```
Topic
 ├── Books      → TOC → Chapters → Sections → grounded + ungrounded QA
 ├── Articles   → Sections                  → grounded + ungrounded QA
 ├── Poems      → Stanzas                    → grounded + ungrounded QA   [added this phase]
 └── Dialogues  → Exchanges (interviewer/expert) → grounded + ungrounded QA [added this phase]
```

- **Grounded QA**: answerable from the generated passage (tests memorisation).
- **Ungrounded QA**: about the domain but not in any passage (tests generalisation).
- Each QA pair is tagged with its `style`, enabling **style-isolated** forget sets.

### 2.2 Dataset conversion (`convert_to_dataset.py`)
- Builds `qa_dataset_forget`, `qa_dataset_retain`, and pretraining text datasets.
- A `--styles` filter restricts the **forget** set to chosen styles (e.g. only
  poems), while the **retain** set deliberately keeps full general-knowledge volume.
- Emits `metadata.json` with per-style breakdowns.

### 2.3 Training & unlearning (OpenUnlearning core + Hydra)
1. **Fine-tune** Llama-3.1-8B-Instruct on the full domain → *Finetuned* model.
2. **Retain-only fine-tune** (forget set withheld) → *Retain-Only* model = the
   theoretical ceiling for "what good forgetting should look like".
3. **Unlearn** from the Finetuned checkpoint with NPO or DPO_IDK → *Unlearned* model.

- **NPO** (Negative Preference Optimization): DPO-style loss treating forget data
  as the "lose" preference. Baseline.
- **DPO_IDK**: DPO with explicit **"I don't know"** templates as the "win"
  preference — trains the model to *abstain* on forget content.

### 2.4 Evaluation (`scripts/evaluate-unlearning.sh`, `evaluate-validation.sh`)
For each experiment we generate responses from all four models on both forget and
retain sets, then compute lexical metrics (no model-based judging, for
reproducibility): **ROUGE-L, Word Overlap (Jaccard), Keyword Recall, Refusal
Rate, Response Length, Word Diversity**.

Two headline scores:
- **Forget Score** = drop in forget-set similarity from Finetuned→Unlearned
  (higher ⇒ more forgetting).
- **Retain Score** = Unlearned/Finetuned retain-set ratio (≈1.0 ⇒ retention preserved).

An **external validation set** is generated by a *direct* GPT call (NOT through the
pipeline) — held-out, topic-level general knowledge — to test whether the model
forgot the *topic* or merely our specific phrasings.

---

## 3. Experimental design — the final-paper sweep

Orchestrated end-to-end by `scripts/final-paper-experiments.sh`:

| Dimension | Values |
|---|---|
| **Topics** | Brazil, Football, Cinema |
| **Styles** | book, article, poem, dialogue, all |
| **Trainers** | NPO (baseline), DPO_IDK (proposed) |
| **Model** | Llama-3.1-8B-Instruct |
| **Total** | 3 × 5 × 2 = **30 experiments** |

Each experiment ≈ 95 min on an **RTX PRO 6000 Blackwell (97 GB)** in a JupyterServer
container; ~47 h wall-clock total. The orchestrator is **resume-capable**
(`results/final/MANIFEST.tsv`), **failure-isolated** (one crash never aborts the
sweep), and writes per-run logs.

---

## 4. Results

> Aggregates below **exclude** the two `brazil_all` runs (see §5.2 — data anomaly).
> Valid set: 13 NPO + 12 DPO_IDK runs.

### 4.1 Headline: DPO_IDK ≫ NPO

| Trainer | n | Avg Forget Score | Avg Retain Score | Avg Unlearned Refusal |
|---|---|---|---|---|
| NPO (baseline) | 13 | 0.0138 | 1.097 | **0.2 %** |
| **DPO_IDK (ours)** | 12 | **0.0819** | 1.089 | **36.0 %** |

DPO_IDK delivers a **~6× higher forget score** while keeping the retain score
essentially identical (both ≈1.09, i.e. retained knowledge is preserved). NPO
barely forgets — its refusal rate is near zero, meaning it keeps answering forget
questions much like the fine-tuned model.

### 4.2 Paired comparison (same topic+style, NPO vs DPO_IDK)

DPO_IDK wins on **every one of the 12 valid pairs** — the improvement is universal,
not an averaging artefact:

| Topic_Style | NPO | DPO_IDK | Δ (DPO_IDK − NPO) |
|---|---|---|---|
| brazil_article | 0.0407 | 0.1120 | +0.0713 |
| brazil_book | 0.0201 | 0.0763 | +0.0562 |
| brazil_dialogue | 0.0267 | 0.0757 | +0.0490 |
| brazil_poem | 0.0175 | 0.0447 | +0.0272 |
| cinema_article | 0.0069 | 0.1043 | +0.0974 |
| cinema_book | −0.0090 | 0.0644 | +0.0734 |
| cinema_poem | 0.0010 | 0.0646 | +0.0636 |
| football_all | 0.0108 | 0.1262 | +0.1154 |
| football_article | 0.0240 | 0.1270 | +0.1030 |
| football_book | 0.0140 | 0.0674 | +0.0534 |
| football_dialogue | 0.0036 | 0.0664 | +0.0628 |
| football_poem | 0.0164 | 0.0540 | +0.0376 |

### 4.3 Effect of writing style (DPO_IDK)

| Style | n | Avg Forget Score | Avg Refusal |
|---|---|---|---|
| **article** | 3 | **0.1144** | 42.2 % |
| dialogue | 2 | 0.0711 | 34.8 % |
| book | 3 | 0.0694 | 35.6 % |
| poem | 3 | 0.0544 | 29.2 % |
| all (combined) | 1 | 0.1262 | 41.7 % |

**Articles are the most forgettable style; poems the least.** A plausible
interpretation: article QA is more factual/extractive and thus more cleanly
separable, whereas poem content is diffuse and entangled with general language,
making targeted forgetting harder. The combined "all" condition forgets best of
all, consistent with a larger, more redundant forget signal.

### 4.4 Effect of topic (DPO_IDK)

| Topic | n | Avg Forget Score | Avg Retain Score |
|---|---|---|---|
| Football | 5 | 0.0882 | 1.108 |
| Cinema | 3 | 0.0778 | 1.076 |
| Brazil | 4 | 0.0772 | 1.076 |

Forgetting is **stable across topics** (0.077–0.088), suggesting the method is
domain-agnostic rather than exploiting quirks of one subject.

### 4.5 Best individual runs

| Run | Forget | Retain | Refusal |
|---|---|---|---|
| **football_article_DPO_IDK** | 0.1270 | 1.097 | 53.1 % |
| football_all_DPO_IDK | 0.1262 | 1.222 | 41.7 % |
| brazil_article_DPO_IDK | 0.1120 | 1.083 | 39.1 % |
| cinema_article_DPO_IDK | 0.1043 | 1.162 | 34.5 % |

> The full per-run table is in `RESULTS_TABLE.txt`; full per-sample JSON/CSV/reports
> are under `saves/eval/<run>/`.

---

## 5. Validity, anomalies, and engineering issues

### 5.1 Sweep completion status
Of 30 experiments: **25 completed, 5 failed** (all in the Cinema topic). Two of the
five "failures" produced valid evaluation data and failed only on a downstream CSV
write — they are recoverable without re-running (§5.3).

### 5.2 Data anomaly — `brazil_all` (excluded from analysis)
`brazil_all_NPO` and `brazil_all_DPO_IDK` returned **byte-identical** metrics and
unlearned responses (only metadata differed). The NPO run reported a 41.5 % refusal
rate — wildly inconsistent with NPO's 0.2 % average — confirming it received the
DPO_IDK model's outputs. Root cause is under investigation (likely the orchestrator
resolved the wrong run directory for one of the two). **Both runs are excluded** and
queued for a clean re-run. All other 12 NPO/DPO_IDK pairs were verified distinct.

### 5.3 Cinema failures — root causes
| Run | Cause | Disposition |
|---|---|---|
| cinema_book_NPO | Eval succeeded; crashed only on final CSV write (`_csv.Error: need to escape`) | Valid — mark completed |
| cinema_book_DPO_IDK | Same CSV crash | Valid — mark completed |
| cinema_dialogue_DPO_IDK | CUDA OOM — a stray process held 16 GB | Re-run |
| cinema_all_NPO | CUBLAS_STATUS_INTERNAL_ERROR (poisoned GPU context, cascade from the OOM) | Re-run |
| cinema_all_DPO_IDK | Same CUBLAS error | Re-run |

### 5.4 Fixes applied (committed & pushed)
- **`f688b66`** — CSV writer in `evaluate-unlearning.sh` made non-fatal
  (`try/except`, `QUOTE_ALL` + `escapechar`, control-char stripping). The eval's
  canonical outputs are JSON + report; a CSV serialization failure can no longer
  abort an otherwise-successful evaluation.
- **`57bd951`** — `scripts/rerun-cinema-failures.sh`: reruns only the genuinely
  failed cinema experiments, each isolated, with a GPU free-memory pre-flight check.

### 5.5 Threats to validity
- **Lexical metrics only.** ROUGE-L / overlap / refusal are reproducible but
  shallow; they do not capture semantic forgetting. A model-based judge or
  paraphrase-robust metric would strengthen claims.
- **Refusal-keyword detection** keys on phrases ("I don't know", "I cannot", …);
  creative refusals may be missed.
- **n is small per cell** (1–3 runs); style/topic breakdowns are indicative, not
  statistically powered. No seeds-averaging yet.
- **Retain scores slightly exceed 1.0** across the board, meaning the unlearned
  model sometimes scores *higher* on retain than the fine-tuned model — worth
  understanding (possibly regularisation from the unlearning step).

---

## 6. Current state (snapshot)

- ✅ Generation pipeline extended to 4 styles, style isolation, external validation.
- ✅ E2E orchestrator built, hardened, resume-capable.
- ✅ 25/30 experiments completed with valid data; clean DPO_IDK ≫ NPO result.
- ✅ Root-caused all failures; fixes committed and pushed.
- ⏳ 3 cinema re-runs + 2 brazil_all re-runs pending clean GPU time.
- ⏳ Final aggregate + paper write-up after re-runs land.

---

## 7. Next steps

1. **Recover the 2 cinema_book runs** — eval data is valid; mark `completed` in the
   manifest (`csv_recovered`).
2. **Re-run 3 failed cinema experiments** on a clean GPU
   (`scripts/rerun-cinema-failures.sh`); restart the container first to clear the
   poisoned CUBLAS context.
3. **Re-run both `brazil_all` experiments** to resolve the duplicate anomaly.
4. **Regenerate the aggregate** once all 30 are valid; lock the final tables.
5. **Strengthen evaluation** (stretch): add a semantic/model-based judge and
   report external-validation numbers alongside the internal forget/retain scores.
6. **Statistical robustness** (stretch): multi-seed runs for the headline cells to
   attach confidence intervals to the DPO_IDK vs NPO gap.
7. **Write-up**: fold §1–§5 into the thesis; the DPO_IDK-abstention result and the
   style-sensitivity finding (articles > poems) are the two main contributions.

---

## Appendix A — Reproducing a single experiment

```bash
# Generate a domain
uv run python -m src.domain_generation.main --name "Brazil" \
  --description "The country of Brazil: history, geography, culture, economy, people"

# Convert (style-isolated forget set; here: articles only)
uv run python -m src.domain_generation.convert_to_dataset \
  data/datasets/brazil/domain.json --output-dir data/datasets \
  --dataset-name brazil_article --styles article

# Full train+unlearn+eval pipeline
bash scripts/domain-unlearn.sh brazil_article Llama-3.1-8B-Instruct DPO_IDK
```

## Appendix B — Key files

| Path | Purpose |
|---|---|
| `src/domain_generation/graphs/` | LangGraph generation workflows (domain/book/article/poem/dialogue) |
| `src/domain_generation/convert_to_dataset.py` | Domain → forget/retain datasets (style filtering) |
| `src/domain_generation/generate_validation.py` | External validation set (direct GPT) |
| `scripts/final-paper-experiments.sh` | 30-experiment master orchestrator |
| `scripts/domain-unlearn.sh` | Single train→unlearn→eval pipeline |
| `scripts/evaluate-unlearning.sh` | 4-model forget/retain evaluation |
| `scripts/rerun-cinema-failures.sh` | Targeted rerun of failed cinema runs |
| `results/final/MANIFEST.tsv` | Sweep state (status per experiment) |
| `saves/eval/<run>/` | Per-run evaluation JSON / CSV / report |
