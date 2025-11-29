# Domain Unlearning Pipeline

## Overview

This project extends OpenUnlearning with a custom domain generation pipeline. It allows you to:
1. Generate synthetic domain-specific content (books, articles, QA pairs) using LLMs
2. Fine-tune a model on that content (so it "learns" the domain)
3. Unlearn the content using various algorithms (GradAscent, NPO, etc.)
4. Evaluate effectiveness by comparing: raw → finetuned → unlearned

## Quick Start

### Full Pipeline (Generation + Training + Evaluation)
```bash
bash scripts/domain-unlearn-extended.sh "Brazil"
```

### Skip Generation (Use Existing Checkpoint)
```bash
bash scripts/domain-unlearn-extended.sh "Brazil" --skip-generation
```

### Force Regenerate Content
```bash
bash scripts/domain-unlearn-extended.sh "Brazil" --from-scratch
```

### With Custom Model/Trainer
```bash
bash scripts/domain-unlearn-extended.sh "Brazil" Llama-3.2-3B-Instruct NPO
```

---

## Directory Structure

```
.
├── scripts/
│   ├── domain-unlearn-extended.sh   # Main pipeline script
│   ├── evaluate-unlearning.py       # 3-model evaluation script
│   └── cleanup-checkpoints.sh       # Remove cached generations
│
├── src/
│   ├── domain_generation/           # LangGraph-based content generation
│   │   ├── graphs/                  # Generation workflows
│   │   ├── models.py                # Pydantic data models
│   │   ├── prompts/                 # LLM prompts
│   │   └── convert_to_dataset.py   # JSON → HuggingFace dataset
│   ├── train.py                     # Training entry point
│   ├── eval.py                      # Evaluation entry point
│   └── trainer/unlearn/             # Unlearning algorithms
│
├── configs/
│   ├── train.yaml                   # Fine-tuning base config
│   ├── unlearn.yaml                 # Unlearning base config
│   ├── data/datasets/               # Dataset configs (generated)
│   └── experiment/
│       ├── finetune/domain/         # Fine-tune experiment configs
│       └── unlearn/domain/          # Unlearn experiment configs
│
├── .logs/generations/{topic}/       # Cached domain.json checkpoints
├── output/{timestamp}/              # Generated domain.json per run
├── data/run/{timestamp}/{topic}/    # HuggingFace datasets
├── saves/
│   ├── train/{run}_finetuned/       # Fine-tuned model checkpoints (mode=train)
│   └── unlearn/{run}_unlearned/     # Unlearned model checkpoints (mode=unlearn)
└── results/{topic}/{timestamp}/     # Evaluation CSVs
```

---

## Pipeline Steps

### Step 1: Domain Generation
- Uses OpenAI API via LangGraph
- Generates: topics → books (chapters/sections) → articles → QA pairs
- Output: `output/{timestamp}/domain.json`
- Checkpoint: `.logs/generations/{topic}/domain.json`

**Environment variables for generation size:**
```bash
export GEN_TOPICS_MIN_ITEMS=5
export GEN_TOPICS_MAX_ITEMS=10
export GEN_GROUNDED_QA_MIN_ITEMS=15
export GEN_GROUNDED_QA_MAX_ITEMS=25
```

### Step 2: Dataset Conversion
- Converts `domain.json` → HuggingFace datasets
- Creates forget/retain splits (80/20)
- Output: `data/run/{timestamp}/{topic}/qa_dataset_forget/`

### Step 3: Config Generation
- Creates Hydra configs for the datasets
- `configs/data/datasets/DOMAIN_{topic}_forget.yaml`
- `configs/data/datasets/DOMAIN_{topic}_retain.yaml`

### Step 4: Fine-tuning
- Trains base model on forget set (teaches domain knowledge)
- Default: 3 epochs, lr=1e-5
- Output: `saves/train/{run}_finetuned/`

### Step 5: Unlearning
- Runs unlearning on finetuned model
- Default: 3 epochs, **GradDiff** (gamma=0.5, alpha=1.0)
- Output: `saves/unlearn/{run}_unlearned/`

**Note**: GradDiff balances forget loss with retain loss, preventing model corruption that can occur with aggressive methods like GradAscent.

### Step 6: Evaluation
- Compares 3 models on forget/retain sets:
  - **Raw**: Base model (doesn't know domain)
  - **Finetuned**: Knows the domain
  - **Unlearned**: Should have forgotten
- Output: `results/{topic}/{timestamp}/`
  - `qualitative.tsv` - Full responses (tab-separated for text compatibility)
  - `quantitative.csv` - Aggregated metrics
  - `detailed_metrics.csv` - Per-sample metrics

---

## Key Commands

### Run Evaluation Separately
```bash
uv run python scripts/evaluate-unlearning.py \
    --run-summary data/run/{timestamp}/run_summary.json
```

### Run Evaluation with Custom Paths
```bash
uv run python scripts/evaluate-unlearning.py \
    --raw-model meta-llama/Llama-3.2-1B-Instruct \
    --finetuned-model ./saves/train/brazil_20251127_finetuned \
    --unlearned-model ./saves/unlearn/brazil_20251127_unlearned \
    --data-dir data/run/20251127/brazil \
    --output-dir results/brazil/20251127 \
    --max-samples 50
```

### Clean Up Experiments
```bash
# Interactive mode (asks what to delete)
bash scripts/clean-experiments.sh

# Delete everything (models, data, results, configs)
bash scripts/clean-experiments.sh --all

# Delete only specific things
bash scripts/clean-experiments.sh --models    # Model checkpoints only
bash scripts/clean-experiments.sh --data      # Generated datasets only
bash scripts/clean-experiments.sh --results   # Evaluation CSVs only
bash scripts/clean-experiments.sh --configs   # Generated Hydra configs only

# Preview what would be deleted (no actual deletion)
bash scripts/clean-experiments.sh --all --dry-run
```

### Manual Training Commands

**Fine-tuning:**
```bash
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --config-name=train.yaml \
    experiment=finetune/domain/brazil \
    task_name=brazil_finetuned \
    trainer.args.num_train_epochs=2 \
    trainer.args.learning_rate=1e-5
```

**Unlearning:**
```bash
CUDA_VISIBLE_DEVICES=0 uv run python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/domain/brazil \
    task_name=brazil_unlearned \
    model.model_args.pretrained_model_name_or_path=./saves/train/brazil_finetuned \
    trainer.args.num_train_epochs=10
```

---

## Evaluation Metrics

### Per-Sample Metrics
- `contains_answer` - Ground truth found in response
- `word_f1` - Word-level F1 score
- `rouge_l` - ROUGE-L (sequence similarity)
- `is_refusal` - Model refused to answer
- `perplexity` - Model fluency (low = good, >1000 = corrupted)

### Aggregated Metrics
- `learning_gain` = finetuned - raw on forget set (did fine-tuning work?)
- `forget_efficacy` = finetuned - unlearned on forget set (did unlearning work?)
- `retain_preservation` = unlearned / finetuned on retain set (utility preserved?)
- `avg_perplexity` - Average response perplexity (detects model corruption)

### Expected Results
| Model | Forget Set | Retain Set |
|-------|-----------|------------|
| Raw | Low (doesn't know) | Baseline |
| Finetuned | High (learned) | Similar |
| Unlearned | Low (forgot) | Similar (preserved) |

---

## Hyperparameters

### Fine-tuning (teach domain)
```bash
FINETUNE_EPOCHS=3
FINETUNE_LEARNING_RATE=1e-5
FINETUNE_BATCH_SIZE=4
FINETUNE_GRADIENT_ACCUMULATION=4  # Effective: 16
```

### Unlearning (forget domain)
```bash
UNLEARN_EPOCHS=3
UNLEARN_LEARNING_RATE=1e-5
UNLEARN_BATCH_SIZE=4
UNLEARN_GRADIENT_ACCUMULATION=8  # Effective: 32

# GradDiff method parameters (balances forget vs retain)
UNLEARN_GAMMA=0.5   # Forget loss weight (lower = gentler)
UNLEARN_ALPHA=1.0   # Retain loss weight (preserves utility)
```

---

## Troubleshooting

### Hydra Config Errors
- `Could not find 'data/train'` → Use `data: finetune` not `data: train`
- `ConfigAttributeError` for new keys → Add `+` prefix (e.g., `+trainer.args.warmup_epochs=1.0`)
- Keys already exist → Remove `+` prefix

### Training Hangs at 0%
```bash
export CUDA_VISIBLE_DEVICES=0  # Force single GPU
```

### Out of Memory
- Reduce batch size
- Enable gradient checkpointing (already in script)
- Use `--max-samples 50` for evaluation

### Disk Space
Data is stored in symlinked directories:
```bash
ls -la saves/    # → /mnt/drive1/saves
ls -la data/     # → /mnt/drive1/data
ls -la output/   # → /mnt/drive1/output
```

---

## Environment Setup

### Requirements
```bash
# Create environment
conda create -n unlearning python=3.11
conda activate unlearning

# Install dependencies
pip install .[lm_eval]
pip install --no-build-isolation flash-attn==2.6.3
```

### API Keys (.env)
```
OPENAI_API_KEY=sk-...
HUGGINGFACE_TOKEN=hf_...
```

---

## Files Modified Per Run

After running the pipeline, these files are created:

```
# Generated content
output/{timestamp}/domain.json
.logs/generations/{topic}/domain.json  (checkpoint)

# Datasets
data/run/{timestamp}/{topic}/qa_dataset_forget/
data/run/{timestamp}/{topic}/qa_dataset_retain/
data/run/{timestamp}/run_summary.json

# Configs
configs/data/datasets/DOMAIN_{topic}_forget.yaml
configs/data/datasets/DOMAIN_{topic}_retain.yaml
configs/experiment/finetune/domain/{topic}.yaml
configs/experiment/unlearn/domain/{topic}.yaml

# Models
saves/train/{topic}_{timestamp}_finetuned/
saves/unlearn/{topic}_{timestamp}_unlearned/

# Results
results/{topic}/{timestamp}/qualitative.tsv
results/{topic}/{timestamp}/quantitative.csv
results/{topic}/{timestamp}/detailed_metrics.csv
```

---

## Available Unlearning Methods

Located in `src/trainer/unlearn/`:

| Method | Safety | Description |
|--------|--------|-------------|
| **GradDiff** (default) | HIGH | Balances forget + retain loss. Recommended. |
| `GradAscent` | LOW | Aggressive, can corrupt model. Avoid. |
| `NPO` | HIGH | DPO-based, smooth gradients (beta=0.1) |
| `UNDIAL` | VERY HIGH | Self-distillation, preserves utility |
| `SatImp` | VERY HIGH | Adaptive reweighting (gamma=0.1) |
| `DPO` | HIGH | Requires alternate answers |
| `RMU` | MEDIUM | Layer-targeted unlearning |

Change method:
```bash
bash scripts/domain-unlearn-extended.sh "Brazil" Llama-3.2-1B-Instruct NPO
```

### Tuning GradDiff
If model still corrupts, reduce gamma:
```bash
# In domain-unlearn-extended.sh
UNLEARN_GAMMA=0.3   # More gentle (default: 0.5)
UNLEARN_ALPHA=1.0   # Keep retain weight
```
