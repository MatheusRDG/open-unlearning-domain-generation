# Unlearning Experiments Log

Ongoing experimental findings for domain-specific LLM unlearning with synthetic data generation.

## Setup

- **Base framework**: OpenUnlearning + LangGraph domain generation
- **Models tested**: Llama-3.2-1B-Instruct (local MPS), Llama-3.1-8B-Instruct (RunPod A100)
- **Unlearning methods tested**: GradAscent, NPO
- **Reference paper**: "LLM Unlearning Without an Expert Curated Dataset" (Zhu et al., COLM 2025)
- **Domain**: Juninho (semi-fictional — Brazilian footballer name + fictional Verdantia/Flamebringer lore)

## Pipeline (4 models)

1. **Base**: Pretrained Llama (no training)
2. **Finetuned**: Trained on forget + retain combined
3. **Retain-Only**: Trained only on retain data (theoretical ceiling for forgetting)
4. **Unlearned**: Finetuned → NPO unlearning applied

The Retain-Only baseline (suggested by advisor) is critical: it represents what the unlearned model *should* look like if forgetting were perfect.

## Metrics

We compute per-sample and aggregated:
- **ROUGE-L, Word Overlap, Keyword Recall** (vs ground truth) — how much each model matches the expected answer
- **FT↔UL similarity** — how much the unlearned model diverged from the finetuned one
- **Refusal rate, length, word diversity** — behavioral signals
- **Forget Score** = drop in GT-similarity from FT → UL (higher = better forgetting)
- **Retain Score** = UL/FT ratio on retain set (closer to 1.0 = better retention)

## Data Quality Metrics

We also quantify dataset quality before training:
- **Knowledge Entanglement**: vocab Jaccard, TF cosine, bigram overlap between forget/retain
- **Diversity**: type-token ratio, self-BLEU (lower = more diverse)
- **Domain Specificity**: % questions containing domain entity names
- **Groundedness**: answer word overlap with source context

---

## Run Comparison Table

| Metric | Run 1 (8B, old data) | Run 2 (8B, old data) | Run 3 (8B, NEW data v2) |
|--------|:---:|:---:|:---:|
| Model | Llama-3.1-8B | Llama-3.1-8B | Llama-3.1-8B |
| Trainer | NPO | NPO | NPO |
| NPO β (forgetting strength) | 0.5 | 0.3 | 0.3 |
| NPO α (retain anchor) | 1.0 | 1.5 | 1.5 |
| Finetune epochs | 5 | 5 | 10 |
| Unlearn epochs | 5 | 10 | 10 |
| Forget QA count | 109 | 109 | **239** |
| Retain QA count | 74 | 74 | **139** |
| Text passages in FT | 0 | 0 | **634** |
| **FT ROUGE-L (forget set)** | 0.091 | 0.091 | **0.132** |
| **RO ROUGE-L (forget set)** | 0.075 | 0.075 | **0.104** |
| **UL ROUGE-L (forget set)** | 0.088 | 0.081 | **0.117** |
| **Forget Score** | 0.016 | 0.034 | 0.029 |
| **Retain Score** | 1.31 | 1.58 | **1.23** |
| Final unlearn loss | 0.55 | 0.004 | 0.03 |

### Dataset Quality Evolution (Juninho)

| Metric | Old Dataset | New Dataset v2 |
|--------|:---:|:---:|
| Vocab Jaccard (entanglement) | 0.138 | **0.067** |
| Bigram Jaccard | 0.018 | **0.009** |
| Forget answers with domain entities | 39% | **98%** |
| Retain questions w/ domain entities | 80% | **1%** |
| Avg answer length (tokens) | 29 | **70** |
| Total training samples | ~183 | **~1012** (239 QA + 139 retain + 634 text) |

---

## Key Findings

### 1. Data setup matters more than hyperparameters

The shift from Run 2 → Run 3 (same hyperparameters, new data) produced the biggest improvements:
- Finetuning actually injected domain knowledge (FT ROUGE-L: 0.091 → 0.132, +45%)
- Retain Score normalized toward 1.0 (1.58 → 1.23)
- Forget-retain entanglement dropped 51% (vocab Jaccard 0.138 → 0.067)

**Implication**: The earlier "unlearning doesn't work" signal was largely a data quality problem. When forget and retain data are entangled (same domain, random split), the model can't selectively forget.

### 2. NPO is stable, GradAscent collapses

GradAscent with aggressive hyperparameters produced gibberish output (`".")..")..")`) on the 1B model. NPO maintains model coherence even at β=0.3.

### 3. The Retain-Only baseline is the right target

It represents the ideal unlearned behavior: answers as if the model never saw the forget data. Refusal ("I don't know") is actually a failure mode — it signals "I was told to forget this" and is detectable by membership inference attacks.

In Run 3, the unlearned model got 52% of the way from FT (0.132) to RO (0.104) with ROUGE-L of 0.117.

### 4. Text passages significantly improved memorization

Adding 634 text passages (from chapter content, converted to QA format) to finetuning:
- Increased training samples 5.5x (183 → 1012)
- FT ROUGE-L rose from 0.091 to 0.132
- Gave the unlearning a real memorization to "undo"

### 5. Critical hyperparameter findings

| Config | Result |
|--------|--------|
| GradAscent, 50 epochs, lr=5e-5 | Model collapse (gibberish) |
| GradAscent, 5 epochs, lr=1e-5 | Repetition loops |
| NPO β=0.5, 5 epochs | Under-trained, unlearn loss plateau at 0.55 |
| NPO β=0.3, 10 epochs | Over-converged, loss near 0 by epoch 5 |
| NPO β=0.3, α=1.5, 10 epochs, new data | **Best result so far** |

**Open question**: Unlearn loss converges by epoch 5 on new data — should we stop earlier or reduce LR?

---

## Pipeline Changes Across Rounds

### Prompts (v1 → v2)
- Grounded QA now requires specific entity names (enforced in prompt)
- Answers must be 2-3 sentences with source context passage
- Ungrounded QA changed from "domain-adjacent" to "completely unrelated general knowledge"

### Conversion (`convert_to_dataset.py` v2)
- Semantic split: grounded → forget, ungrounded → retain (no random split)
- Entity-based filtering removes generic questions
- Text passages split into ~2K char chunks
- Optional context field for grounded QA

### Training
- Finetune on combined (forget QA + retain QA + text passages as QA-format)
- Unlearning uses `ForgetRetainDataset` with anchor=forget
- Added `max_grad_norm=1.0`, `adamw_torch` optimizer
- Disabled TOFU evaluation (wrong for domain data)

---

## Comparison with Reference Paper

Paper (Zhu et al., COLM 2025):
- Approach: Extract existing knowledge from LLM → generate sentences → unlearn
- Method: RMU, RR, ELM
- Evaluation: WMDP benchmark + MMLU/GSM8K/TriviaQA
- Best unlearn utility: ≈22-35 (Mistral-7B with RMU on biosecurity)

Our approach:
- Inject then remove custom fictional knowledge
- Method: NPO (with retain-only baseline, not in paper)
- Evaluation: Custom QA + quantitative metrics (ROUGE-L, entanglement, etc.)
- Novel contribution: **Synthetic pipeline for arbitrary domains** + **Retain-only baseline as target**

---

## Open Questions for Advisor

1. **Forget Score magnitude**: Is a 0.03 forget score meaningful? Paper reports "unlearn utility" around 22-35, but their metric is normalized differently. How do we compare?

2. **Refusal vs forgetting**: Our unlearned model doesn't refuse (refusal rate 0%). It gives different answers with similar topic coverage. Is this the right behavior, or should we explicitly train for refusals?

3. **Retain set choice**: We generate retain data via GPT (general knowledge). Should we also evaluate on standard benchmarks (MMLU subset)?

4. **Unlearn loss plateau**: Loss converges to near-zero by epoch 5 — suggests NPO is "done" early. Should we add early stopping, or train fewer epochs?

5. **Text passage format**: We convert text to QA format ("Tell me about X → [passage]"). Would Pretraining-style loss on raw text be better?

6. **Multi-domain**: Current test uses one domain (Juninho). For generalization claims, we should test 3+ domains. Pernambuco (real-world) is the next one to add.

---

## Next Steps

1. **Pernambuco domain** (real-world, not fictional) — test if approach works on topics with pre-existing model knowledge
2. **Compare against WMDP baseline** (the paper's benchmark)
3. **Ablation study**: isolate contributions of (a) data volume, (b) text passages, (c) entity filtering
4. **Sensitivity analysis**: NPO β sweep at fixed α
5. **Cross-model**: confirm results hold on Llama-3.2-3B, not just 8B

---

## Round 4 Technical Debt Review (from code review)

Independent code review highlighted several design gaps that explain why forgetting is only marginal:

### Pipeline hygiene fixes (applied)
- **`dataloader_num_workers`** on unlearn reset to `0`. `ForgetRetainDataset` has
  mutable counters inside `__getitem__` (`src/data/unlearn.py:82`) so multi-worker
  pairing was non-deterministic.
- **Retain leakage filter**. Ungrounded QA could still reference the domain
  (e.g. retain QA about "Brazil football" in a Brazil forget experiment). Added a
  strict proper-noun filter that rejects retain pairs mentioning the domain
  (`extract_ungrounded_qa` in `convert_to_dataset.py`).
- **Entity filter tiering**. Previous heuristic accepted any capitalized word,
  including "Capital", "Change", "Areas". Now split into *strict* (multi-word
  proper-noun phrases from titles, used for retain rejection) and *loose*
  (singletons + strict, used for forget acceptance).
- **Dead `context` field removed**. We were saving context on forget QA but
  `QADataset` only reads `question`/`answer`. Removed from output to avoid
  misleading metadata.
- **Fake text-to-QA conversion removed**. We were turning passages into
  `"Tell me about: [first sentence]" → [rest]` and mixing into QA finetune.
  Now finetune is pure QA (forget + retain). `PretrainingDataset` handler
  exists if we want proper passage training later; this keeps the pipeline
  honest for now.

### Training-objective gaps (open work)
- **NPO has no replacement target**. `compute_loss` only pushes the original
  forget answer down as the "losing" preference; the model is never told *what
  to say instead*. Evidence: Run 3 forget ROUGE-L dropped 0.132 → 0.117 and
  refusal rate stayed at 0.0%. The model is just perturbing answers, not
  degrading them.
- **DPO + IDK route is the highest-leverage next step.** The repo already has
  `DPO` trainer, `QAwithIdkDataset`, and `data/idk.jsonl` (added in this round).
  New `TRAINER=DPO_IDK` mode in `domain-unlearn.sh` routes forget data through
  `QAwithIdkDataset` (provides IDK as "winning" alternate) and uses the DPO
  trainer. This teaches the model an explicit abstain target.
- **KL retain loss variant**. `GradDiff.compute_retain_loss` already supports
  `retain_loss_type: KL` (divergence vs frozen ref model, bounded drift). Added
  `configs/trainer/NPO_KL.yaml` to make this easy to swap in.
- **No best-checkpoint selection**. Both finetune and unlearn set
  `load_best_model_at_end=false` and `eval_strategy=no`. We always use the last
  checkpoint, not the best unlearning checkpoint on a forget/retain frontier.
  Needs a held-out forget/retain split to be meaningful — deferred.
- **Evaluation is on training data**. Current forget/retain eval uses the same
  examples the model saw. We should generate paraphrased forget QA (held-out)
  and near/far retain benchmarks. Deferred to Round 5+.

### Next experiment order (per reviewer)
1. `DPO + QAwithIdkDataset` — explicit abstain target
2. `NPO_KL` — stronger retain regularization without changing dataset schema
3. `SimNPO` — cleaner forget-only baseline
4. Held-out forget/retain splits + best-checkpoint selection
5. `RMU` as ablation only (config notes it needs careful per-model tuning)

---

## Round 4 Data Regeneration

- **Juninho re-converted** with new filter (v3): 235 forget / 139 retain /
  634 text passages (was 239 / 139 / 634 v2). 4 forget filtered as too generic.
- **Pernambuco generated** (real-world topic, not fictional): 230 forget /
  140 retain / 630 text passages. Entanglement 0.069. 0 retain rejected
  (prompts already constrain ungrounded to general knowledge).

---

## File Artifacts Per Run

`saves/eval/{RUN_NAME}/`:
```
data_quality_metrics.json    — entanglement, diversity, specificity
evaluation_results.json      — full per-sample responses + metrics for 4 models
evaluation_results.csv       — tabular with ROUGE-L, overlap, keyword recall
evaluation_report.txt        — human-readable report
loss_curves.json             — all training loss histories combined
loss_finetune.csv            — finetune loss per step
loss_retainonly.csv          — retain-only baseline loss per step
loss_unlearn.csv             — unlearning loss (with forget/retain breakdown)
```
