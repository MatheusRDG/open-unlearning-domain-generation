# Comparison Plan: Our Approach vs. Synthetic Textbook Paper

## Overview

This document outlines the comparison between our Domain Generation approach and the
"LLM Unlearning Without an Expert Curated Dataset" paper (Zhu et al., COLM 2025).

**Paper Repository:** https://github.com/xyzhu123/Synthetic_Textbook
**Paper Location:** `.docs/llm_unlearning_without_curated_dataset.pdf`
**Code Location:** `.docs/Synthetic_Textbook/`

---

## Quick Start: Run Comparison

```bash
# Compare on biosecurity domain with Mistral-7B
bash scripts/compare_approaches.sh biosecurity mistral-7b

# Compare on cybersecurity with Llama-3-8B
bash scripts/compare_approaches.sh cybersecurity llama-3-8b

# Compare on Harry Potter
bash scripts/compare_approaches.sh "harry potter" mistral-7b

# Use smaller model for faster testing
bash scripts/compare_approaches.sh biosecurity llama-3.2-1b
```

**Output:** `results/comparison/{domain}/{timestamp}/`
- `COMPARISON_TABLE.md` - Results table for thesis
- `comparison_results.csv` - Raw data
- `logs/` - Detailed logs

---

## Key Differences Between Approaches

| Aspect | Paper (Synthetic Textbook) | Our Approach (Domain Generation) |
|--------|---------------------------|----------------------------------|
| **Primary Goal** | Remove existing model knowledge | Inject then remove custom knowledge |
| **Data Source** | Extract knowledge from LLM | Generate entirely new fictional content |
| **Structure** | Flat sentences (20K longest) | Hierarchical: Books → Chapters → Sections → QA |
| **Fine-tuning** | Not required (unlearn existing knowledge) | Required (teach domain first) |
| **Target Domains** | Real: Biosecurity, Cybersecurity, Harry Potter | Custom: Any fictional/factual domain |
| **Diversity Source** | 3-stage pipeline + 4 audience levels | Topics → Books/Articles + grounded/ungrounded QA |
| **Unlearning Methods** | RMU, RR, ELM | GradAscent, NPO, GradDiff, RMU |
| **Evaluation** | WMDP benchmark + MMLU/GSM8K/TriviaQA | Custom forget/retain QA comparison |

### Generation Pipeline Comparison

**Paper's 3-Stage Pipeline:**
```
Domain → 10 Subdomains → 800 Bullet Points (4 audiences × 20/subdomain) → 4000 Chapters → 20K Sentences
```

**Our Pipeline:**
```
Domain → N Topics → N Books (each with TOC → Chapters → Sections) + Articles → QA Pairs (grounded + ungrounded)
```

---

## Comparison Dimensions

### 1. Dataset Quality Metrics
| Metric | Description | Paper Reports |
|--------|-------------|---------------|
| **Self-BLEU** | Text diversity (lower = better) | 0.758-0.930 |
| **Token Count** | Dataset size | ~20K sentences |
| **Domain Relevance** | LLM pairwise preference test | 50-95% win rate |

### 2. Unlearning Effectiveness
| Metric | Formula | Goal |
|--------|---------|------|
| **Unlearn Utility** | U = -0.5×Sf + 0.5×Sr | Higher is better |
| **Forget Score (Sf)** | % change in forget benchmark | Lower benchmark = better forgetting |
| **Retain Score (Sr)** | % change in general benchmarks | Closer to 0 = better preservation |

### 3. General Capability Preservation
| Benchmark | Type |
|-----------|------|
| MMLU | General knowledge |
| GSM8K | Math reasoning |
| TriviaQA | Factual recall |

---

## Proposed Comparison Experiments

### Experiment A: Same Domain, Different Methods
**Domain:** Biosecurity (real domain from paper)

| Approach | Generation | Fine-tune | Unlearn | Evaluate |
|----------|-----------|-----------|---------|----------|
| Paper | Textbook pipeline | No | RMU/RR | WMDP-Bio |
| Ours | Domain pipeline | Yes (on generated) | NPO/GradDiff | WMDP-Bio |

### Experiment B: Custom Domain Comparison
**Domain:** Custom fictional domain (e.g., "Juninho" or "Brazil Facts")

| Approach | Generation | Fine-tune | Unlearn | Evaluate |
|----------|-----------|-----------|---------|----------|
| Paper | Textbook pipeline | Yes | RMU/RR | Custom QA |
| Ours | Domain pipeline | Yes | NPO/GradDiff | Custom QA |

### Experiment C: Dataset Quality Analysis
Compare generated datasets:
- Self-BLEU diversity scores
- Topic coverage analysis
- LLM relevance judgment

---

## Implementation Steps

### Step 1: Install Paper's Code
```bash
git clone https://github.com/xyzhu123/Synthetic_Textbook.git
cd Synthetic_Textbook
pip install -r requirements.txt
```

### Step 2: Generate Comparable Datasets

**Generate using Paper's method:**
```bash
python scripts/generate_textbook.py \
    --provider openai \
    --keyword "Brazil" \
    --model-name gpt-4o-mini \
    --stages all
```

**Generate using Our method:**
```bash
python -m src.domain_generation.main \
    --name "Brazil" \
    --description "Brazilian geography, culture, history and facts"
```

### Step 3: Standardize Evaluation

Both approaches should be evaluated on:
1. **Forget set accuracy** - Model's knowledge of target domain
2. **General capabilities** - MMLU, GSM8K (via lm-evaluation-harness)
3. **Qualitative analysis** - Response quality comparison

### Step 4: Compute Metrics

```python
# Unlearn Utility formula from paper
Sf = (baseline_forget - unlearned_forget) / baseline_forget * 100  # % change in forget benchmark
Sr = mean([
    (baseline_mmlu - unlearned_mmlu) / baseline_mmlu,
    (baseline_gsm8k - unlearned_gsm8k) / baseline_gsm8k,
    (baseline_trivia - unlearned_trivia) / baseline_trivia
]) * 100  # % change in general benchmarks

U = -0.5 * Sf + 0.5 * Sr  # Unlearn Utility (higher = better)
```

---

## Expected Results Table (Template)

### Quantitative Comparison

| Model | Approach | Method | Unlearn Utility (↑) | General Cap. Δ (↑) | Forget Δ (↑) | MMLU | GSM8K |
|-------|----------|--------|---------------------|-------------------|--------------|------|-------|
| Mistral-7B | Textbook (Paper) | RMU | ? | ? | ? | ? | ? |
| Mistral-7B | Domain-Gen (Ours) | RMU | ? | ? | ? | ? | ? |
| Mistral-7B | Textbook (Paper) | NPO | ? | ? | ? | ? | ? |
| Mistral-7B | Domain-Gen (Ours) | NPO | ? | ? | ? | ? | ? |
| Llama-3-8B | Textbook (Paper) | RMU | ? | ? | ? | ? | ? |
| Llama-3-8B | Domain-Gen (Ours) | RMU | ? | ? | ? | ? | ? |

### Paper's Reference Results (Biosecurity, Table 1)

| Model | Method | Dataset | Unlearn Utility | General Cap. Δ | WMDP (↓) | MMLU |
|-------|--------|---------|-----------------|----------------|----------|------|
| Mistral | RMU | Textbook-Bio | **22.41** | -9.43 | 0.309 | 0.584 |
| Mistral | RR | Textbook-Bio | **21.52** | -9.9 | 0.318 | 0.477 |
| Llama3 | RMU | Textbook-Bio | **15.54** | 0.34 | 0.492 | 0.597 |
| Llama3 | RR | Textbook-Bio | **21.39** | -4.08 | 0.377 | 0.581 |

**Note:** Our goal is to achieve comparable or better Unlearn Utility with our Domain-Gen approach.

---

## Required Components for Comparison Script

### Single Script Requirements (`scripts/compare_approaches.sh`)

1. **Domain generation (both methods)**
   - [ ] Paper's textbook pipeline
   - [ ] Our domain pipeline

2. **Dataset preparation**
   - [ ] Convert both to compatible format
   - [ ] Split forget/retain

3. **Model training** (if needed)
   - [ ] Fine-tune on generated content
   - [ ] Save checkpoints

4. **Unlearning**
   - [ ] Run selected methods (NPO, GradDiff, RMU)
   - [ ] Save unlearned models

5. **Evaluation**
   - [ ] Forget set accuracy
   - [ ] General benchmarks (MMLU, GSM8K)
   - [ ] Compute Unlearn Utility

6. **Results aggregation**
   - [ ] Generate comparison tables
   - [ ] Export CSV/TSV

---

## Timeline and Milestones

| Week | Task | Deliverable |
|------|------|-------------|
| 1 | Set up paper's code + run their method | Working baseline |
| 2 | Adapt our pipeline for comparison | Compatible datasets |
| 3 | Run unlearning experiments | Model checkpoints |
| 4 | Evaluation + analysis | Comparison tables |

---

## Notes for Professor's Requirements

> "O que te falta é seguir o método científico. Até hoje eu te peço uma tabela de resultados
> comparando a sua proposta com o baseline e você não apresentou."

This comparison will provide:
1. **Quantitative table** with Unlearn Utility, General Cap Δ, and benchmark scores
2. **Qualitative table** with example responses showing forgetting behavior
3. **Statistical analysis** of dataset quality (Self-BLEU diversity)
4. **Reproducible single script** for running the complete comparison

---

## File Locations

| File | Purpose |
|------|---------|
| `docs/COMPARISON_PLAN.md` | This plan document |
| `scripts/compare_approaches.sh` | Main comparison script (to be created) |
| `scripts/compute_metrics.py` | Metric computation utilities (to be created) |
| `results/comparison/` | Output tables and analysis |
