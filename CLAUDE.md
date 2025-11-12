# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is **OpenUnlearning** with **Domain Generation Extensions** - a framework for LLM unlearning research that combines:
1. The core OpenUnlearning framework (TOFU, MUSE, WMDP benchmarks)
2. A custom domain generation pipeline using LangGraph for creating synthetic unlearning datasets

The project enables researchers to unlearn specific knowledge domains from language models using various unlearning algorithms, with both standard benchmarks and custom domain-specific content generation.

## Core Architecture

### Component System (Plugin-based)

The codebase uses a **registry pattern** for extensibility:

- **Datasets**: Registered in `src/data/__init__.py` via `DATASET_REGISTRY`
  - Handlers: `QADataset`, `PretrainingDataset`, `CompletionDataset`, `ForgetRetainDataset`
  - Located in `src/data/qa.py`, `src/data/pretraining.py`, `src/data/unlearn.py`

- **Trainers/Unlearning Methods**: Registered in `src/trainer/__init__.py` via `TRAINER_REGISTRY`
  - Methods: GradAscent, GradDiff, NPO, DPO, SimNPO, RMU, UNDIAL, CEU, SatImp, WGA, PDU
  - Base class: `src/trainer/unlearn/base.py`
  - Each method: `src/trainer/unlearn/<method_name>.py`

- **Evaluators**: Located in `src/evals/`, includes TOFU, MUSE benchmarks and metrics
  - Metrics: MIA attacks, memorization, privacy, utility
  - Located in `src/evals/metrics/`

### Configuration System (Hydra)

The project uses **Hydra** for hierarchical configuration:

- Base configs: `configs/train.yaml`, `configs/unlearn.yaml`, `configs/eval.yaml`
- Experiments: `configs/experiment/{unlearn,eval}/{tofu,muse,wmdp}/`
- Components: `configs/{model,trainer,data,collator}/`

**Key Hydra features used:**
- Config groups with `defaults:` and `override` directives
- Command-line overrides: `key=value` or `key.nested=value`
- Interpolation: `${variable}` references other config values
- Package directive: `# @package _global_` for merging configs

### Domain Generation Pipeline

Located in `src/domain_generation/`:

- **LangGraph Workflows**: `graphs/` contains stateful generation graphs
  - `domain_graph.py`: Top-level domain generation
  - `book_graph.py`: Book chapter/section generation
  - `article_graph.py`: Article generation

- **Pydantic Models**: `models.py` defines data structures (Domain, Topic, Book, Article, QA)
- **Prompts**: `prompts/` contains LLM prompt templates
- **Configuration**: `config.py` uses `pydantic-settings` with env var support (`GEN_*` prefix)
- **Dataset Conversion**: `convert_to_dataset.py` converts domain.json to HuggingFace datasets

## Common Development Commands

### Environment Setup
```bash
# Initial setup
conda create -n unlearning python=3.11
conda activate unlearning
pip install .[lm_eval]
pip install --no-build-isolation flash-attn==2.6.3

# Download evaluation data
python setup_data.py --eval
```

### Code Quality
```bash
make style         # Auto-fix formatting and linting with ruff
make quality       # Check formatting and linting without fixing
make test          # Run tests (if available)
```

### Running Experiments

**Training/Unlearning:**
```bash
# Basic unlearning
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default \
  forget_split=forget10 retain_split=retain90 \
  trainer=GradAscent task_name=EXPERIMENT_NAME

# Override model and hyperparameters
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default \
  model=Llama-3.2-3B-Instruct \
  trainer=NPO \
  trainer.args.learning_rate=5e-6 \
  trainer.args.num_train_epochs=10 \
  task_name=EXPERIMENT_NAME
```

**Evaluation:**
```bash
python src/eval.py --config-name=eval.yaml \
  experiment=eval/tofu/default \
  model=Llama-3.2-1B-Instruct \
  model.model_args.pretrained_model_name_or_path=saves/unlearn/EXPERIMENT_NAME \
  retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json \
  task_name=EVAL_NAME
```

**Domain Generation & Unlearning Pipeline:**
```bash
# Complete pipeline (generation + unlearning)
bash scripts/domain-unlearn.sh "Topic Name" [MODEL] [TRAINER]

# Examples:
bash scripts/domain-unlearn.sh "Brazil"
bash scripts/domain-unlearn.sh "USA History" Llama-3.2-3B-Instruct NPO

# Step-by-step execution:
# 1. Generate domain content
python -m src.domain_generation.main

# 2. Convert to datasets
python -m src.domain_generation.convert_to_dataset \
  output/TIMESTAMP/domain.json \
  --output-dir data/domain_datasets \
  --dataset-name topic_name \
  --split-ratio 0.8

# 3. Run unlearning (requires manual config creation first)
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/domain/topic_name \
  trainer=GradAscent \
  task_name=domain_unlearn
```

### Baseline Experiments
```bash
bash scripts/tofu_unlearn.sh    # TOFU benchmark experiments
bash scripts/muse_unlearn.sh    # MUSE benchmark experiments
bash scripts/tofu_finetune.sh   # Fine-tune models on TOFU
```

## Key Implementation Patterns

### Adding a New Unlearning Method

1. **Create trainer class** in `src/trainer/unlearn/new_method.py`:
   ```python
   from trainer.unlearn.base import UnlearningTrainer

   class NewMethod(UnlearningTrainer):
       def compute_loss(self, model, inputs, return_outputs=False):
           # Implement custom loss computation
           pass
   ```

2. **Register** in `src/trainer/__init__.py`:
   ```python
   from trainer.unlearn.new_method import NewMethod
   _register_trainer(NewMethod)
   ```

3. **Create config** in `configs/trainer/NewMethod.yaml`:
   ```yaml
   handler: NewMethod
   args:
     output_dir: saves/unlearn/${task_name}
     per_device_train_batch_size: 4
     # ... other TrainingArguments
   method_args:
     # Method-specific hyperparameters
   ```

### Adding a New Dataset

1. **Create dataset class** in `src/data/`:
   ```python
   from torch.utils.data import Dataset

   class NewDataset(Dataset):
       def __init__(self, hf_args, tokenizer, max_length=512, **kwargs):
           # Load from HuggingFace datasets or local files
           pass
   ```

2. **Register** in `src/data/__init__.py`:
   ```python
   from data.new_dataset import NewDataset
   _register_data(NewDataset)
   ```

3. **Create config** in `configs/data/datasets/`:
   ```yaml
   DATASET_NAME:
     handler: NewDataset
     args:
       hf_args:
         path: "dataset/path"
         split: "train"
       max_length: 512
   ```

### Working with Hydra Configs

- Use `experiment=path/to/config` to load experiment presets
- Override nested values: `trainer.args.learning_rate=1e-5`
- Reference other configs: `${forget_split}` or `${trainer.args.output_dir}`
- Set output directory: Task outputs go to `saves/{mode}/{task_name}/`
- The `task_name` parameter is **required** and sets the experiment output directory

### Data Flow in Unlearning

1. **Data loading** (`src/data/__init__.py::get_data()`):
   - Loads forget and retain datasets separately
   - Mode `"unlearn"` → wraps in `ForgetRetainDataset` (alternates between forget/retain)
   - Mode `"train"` → returns datasets as-is

2. **Training loop** (`src/trainer/unlearn/base.py`):
   - `ForgetRetainDataset` provides `is_forget` flag in batch
   - Unlearning methods use flag to apply different losses

3. **Evaluation** during training:
   - Only works with single GPU (no distributed evaluation during training)
   - For multi-GPU: save checkpoints and evaluate separately with `src/eval.py`

## Important Notes

### Domain Generation System

- Uses **LangGraph** for stateful LLM-based generation workflows
- API keys required: Set `OPENAI_API_KEY` in `.env` file
- Configuration via environment variables with `GEN_` prefix (e.g., `GEN_TOPICS_MIN_ITEMS=3`)
- Output structure:
  - Generated content: `output/{timestamp}/domain.json`
  - Converted datasets: `data/run/{timestamp}/{topic}/qa_dataset/` and `text_dataset/`
  - Run metadata: `data/run/{timestamp}/run_summary.json`

### Model and Tokenizer Paths

- Pre-trained models on HuggingFace: `open-unlearning/tofu_MODEL_NAME_SPLIT`
- Use `model.model_args.pretrained_model_name_or_path` for custom checkpoints
- For custom models, also set `model.tokenizer_args.pretrained_model_name_or_path`

### Flash Attention

- Install with: `pip install --no-build-isolation flash-attn==2.6.3`
- Used for efficient attention computation in LLaMA models

### Evaluation Metrics

- **TOFU**: Forget quality, model utility, retention metrics
  - Requires `retain_logs_path` pointing to retain model evaluation JSON
- **MUSE**: Privacy (MIA), utility, sustainability metrics
- **MIA attacks**: LOSS, ZLib, Reference, GradNorm, MinK, MinK++
- **lm-evaluation-harness**: For standard benchmarks (MMLU, GSM8K, etc.)

### Output Directory Structure

```
saves/
  unlearn/{task_name}/
    checkpoint-{step}/       # Training checkpoints
    evals/                   # Evaluation results during training
    trainer_state.json       # Training state
  eval/{task_name}/
    {BENCHMARK}_EVAL.json    # Evaluation outputs
```

### Distributed Training

Use `accelerate` or `deepspeed`:
```bash
accelerate launch --config_file configs/accelerate/default_config.yaml src/train.py ...
```
See `configs/accelerate/` for multi-GPU configurations.

## Project-Specific Conventions

- **Logging**: Uses Python's `logging` module throughout, Hydra configures colorlog
- **Random seeds**: Set via `trainer.args.seed`, applied by `trainer.utils.seed_everything()`
- **Warm-up**: Use `trainer.args.warmup_epochs` (custom) instead of `warmup_steps`
  - Automatically converted to steps based on dataset size
- **Anchoring**: `data.anchor` determines primary dataset for unlearning (typically `"forget"`)
- **Batch indexing**: Models expect `input_ids`, `attention_mask`, `labels` (and `is_forget` for unlearning)

## Common Gotchas

1. **Missing `task_name`**: Always provide `task_name=EXPERIMENT_NAME` - it's required and sets output directory
2. **Evaluation during training**: Only works with single GPU; use separate `src/eval.py` for multi-GPU setups
3. **Config overrides**: Use `=` for setting values, not `:` (Hydra syntax)
4. **Hydra working directory**: Hydra changes CWD; use `hydra.run.dir` or absolute paths for file operations
5. **Dataset split names**: Must match keys in config (e.g., `forget_split=forget10` matches TOFU split names)
6. **API rate limits**: Domain generation may hit OpenAI rate limits; system includes retry logic
7. **Environment variables**: Domain generation config reads from `GEN_*` env vars before defaults
