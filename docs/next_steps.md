# Next Steps & Checkpoint

## ✅ Successfully Completed (2025-11-11)

### Initial Setup Working
- ✅ Domain generation pipeline functional (LangGraph + OpenAI API)
- ✅ Dataset conversion to HuggingFace format
- ✅ Config file generation (Hydra configs for datasets and experiments)
- ✅ Training infrastructure working
- ✅ First successful unlearning run: Brazil domain
  - Model: Llama-3.2-1B-Instruct
  - Method: GradAscent
  - Training time: ~95 seconds (4.21 epochs)
  - Loss progression: -3.67 → -18.90

### Issues Fixed
1. ✅ **Missing config files on SSH server** - Added `restore_configs.sh` script
2. ✅ **Hydra config errors** - Properly handled `+` prefix for new vs existing keys
3. ✅ **Training hanging** - Forced single GPU mode with `CUDA_VISIBLE_DEVICES=0`
4. ✅ **DataLoader issues** - Set `dataloader_num_workers=0`
5. ✅ **Script compatibility** - All scripts now use `uv run python`

---

## 🎯 Next Steps for Robust Training

### 1. Evaluation Pipeline
**Priority: HIGH** | **Status: TODO**

Set up comprehensive evaluation to measure unlearning quality:

```bash
# Evaluate the unlearned model
uv run python src/eval.py \
  model=Llama-3.2-1B-Instruct \
  model.model_args.pretrained_model_name_or_path=saves/unlearn/brazil_20251111_220326 \
  task_name=brazil_eval
```

**Tasks:**
- [ ] Create evaluation config for domain-specific unlearning
- [ ] Define metrics:
  - Forget quality: How well Brazil knowledge was removed
  - Model utility: Performance on retain set (non-Brazil QA)
  - General capability: Test on standard benchmarks (e.g., MMLU subset)
- [ ] Compare with baseline (original Llama-3.2-1B-Instruct)
- [ ] Manual testing: Query model about Brazil facts to verify unlearning

**Files to create:**
- `configs/experiment/eval/domain/brazil.yaml`
- `configs/eval/domain_eval.yaml` (if needed)

---

### 2. Multi-GPU Distributed Training
**Priority: MEDIUM** | **Status: TODO**

Currently limited to single GPU. Enable distributed training for larger models:

**Current limitation:**
```bash
export CUDA_VISIBLE_DEVICES=0  # Forced single GPU
```

**Solution approaches:**
1. **Accelerate config** (recommended):
   ```bash
   accelerate launch --config_file configs/accelerate/default_config.yaml \
     src/train.py --config-name=unlearn.yaml ...
   ```

2. **DeepSpeed** (for very large models):
   ```bash
   deepspeed --num_gpus=2 src/train.py --config-name=unlearn.yaml ...
   ```

**Tasks:**
- [ ] Test with `accelerate launch` on 2 GPUs
- [ ] Verify training doesn't hang with multi-GPU
- [ ] Update scripts to support optional multi-GPU flag
- [ ] Document multi-GPU training in CLAUDE.md

**Files to modify:**
- `scripts/domain-unlearn.sh` - Add multi-GPU flag
- `configs/accelerate/default_config.yaml` - Verify settings

---

### 3. Hyperparameter Tuning & Experiments
**Priority: MEDIUM** | **Status: TODO**

Current settings are defaults and not optimized:

**Current hyperparameters:**
```bash
PER_DEVICE_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
NUM_EPOCHS=5
LEARNING_RATE=1e-5
```

**Experiments to run:**
- [ ] Learning rate sweep: [5e-6, 1e-5, 5e-5]
- [ ] Epoch sweep: [3, 5, 10]
- [ ] Different unlearning methods:
  - [ ] GradAscent (✅ done)
  - [ ] GradDiff
  - [ ] NPO (Negative Preference Optimization)
  - [ ] DPO
  - [ ] RMU
- [ ] Different model sizes:
  - [ ] Llama-3.2-1B (✅ done)
  - [ ] Llama-3.2-3B
  - [ ] Llama-3.1-8B

**Script for parameter sweep:**
```bash
for lr in 5e-6 1e-5 5e-5; do
  for method in GradAscent NPO DPO; do
    bash scripts/domain-unlearn.sh "Brazil" Llama-3.2-1B-Instruct $method
    # Update learning rate in script or pass as parameter
  done
done
```

**Tasks:**
- [ ] Create experiment tracking (wandb or mlflow)
- [ ] Document best hyperparameters in leaderboard
- [ ] Add results to `community/leaderboard.md`

---

### 4. Slurm Batch Job Submission
**Priority: HIGH** | **Status: TODO**

Enable proper cluster job submission for overnight/long runs:

**Current issue:** Running interactively in SSH terminal

**Solution:** Create Slurm submission script

```bash
#!/bin/bash
#SBATCH --job-name=brazil-unlearn
#SBATCH --output=logs/brazil-unlearn-%j.out
#SBATCH --error=logs/brazil-unlearn-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Load environment
source ~/.bashrc
cd ~/open-unlearning-domain-generation
source .venv/bin/activate  # or: eval "$(uv venv)"

# Run training
bash scripts/domain-unlearn.sh "Brazil" Llama-3.2-1B-Instruct GradAscent
```

**Tasks:**
- [ ] Create `scripts/slurm/` directory
- [ ] Write `submit_domain_unlearn.sh` Slurm script
- [ ] Test submission: `sbatch scripts/slurm/submit_domain_unlearn.sh`
- [ ] Add monitoring: `squeue -u $USER`
- [ ] Document Slurm usage in README

**Files to create:**
- `scripts/slurm/submit_domain_unlearn.sh`
- `scripts/slurm/submit_with_args.sh` (parameterized version)

---

### 5. Domain Generation Improvements
**Priority: LOW** | **Status: TODO**

Enhance domain content generation quality:

**Current setup:**
- Model: gpt-4o-mini
- Topics: 2-5 per domain
- QA pairs: 5-10 grounded per book/article

**Improvements:**
- [ ] Add support for Claude/Anthropic API (alternative to OpenAI)
- [ ] Increase content diversity:
  - More topics per domain
  - More articles per topic
  - Deeper book chapters
- [ ] Add content quality checks:
  - Validate QA pairs are answerable
  - Check for repetition/redundancy
  - Verify topic coverage
- [ ] Support for multiple domains in one run
- [ ] Cache and reuse generated content

**Configuration updates:**
```python
# src/domain_generation/config.py
topics_min_items: int = 5      # Increase from 2
topics_max_items: int = 10     # Increase from 5
grounded_qa_min_items: int = 15  # Increase from 5
```

---

### 6. Additional Testing & Validation
**Priority: MEDIUM** | **Status: TODO**

**End-to-end tests:**
- [ ] Test with different topics (USA History, Mexican Food, etc.)
- [ ] Test with different model architectures
- [ ] Verify retain set performance doesn't degrade
- [ ] Test dataset splits (80/20 vs 90/10 forget/retain)

**Documentation:**
- [ ] Update CLAUDE.md with latest fixes
- [ ] Add troubleshooting guide for common errors
- [ ] Document all scripts in `.docs/scripts.md`
- [ ] Create example notebook for analysis

**Quality checks:**
- [ ] Run pre-commit hooks: `make style`
- [ ] Add unit tests for data loading
- [ ] Add integration test for full pipeline

---

## 📝 Important Notes

### Current Working Configuration
```bash
# Environment
Python: 3.10.12
PyTorch: 2.4.1+cu121
CUDA: 12.1
GPUs: 2x NVIDIA RTX A4500 (19.60 GB each)

# Training settings (working)
CUDA_VISIBLE_DEVICES=0  # Single GPU
PER_DEVICE_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
NUM_EPOCHS=5
LEARNING_RATE=1e-5
dataloader_num_workers=0
gradient_checkpointing=true
```

### Known Limitations
1. **Single GPU only** - Multi-GPU causes hanging (needs investigation)
2. **Small dataset** - Brazil domain: 76 forget, 19 retain samples
3. **No evaluation yet** - Need to set up evaluation pipeline
4. **No experiment tracking** - Consider adding wandb/mlflow

### Quick Commands Reference
```bash
# Full pipeline
bash scripts/domain-unlearn.sh "Brazil"

# Single GPU explicit
bash run_brazil_single_gpu.sh

# Restore missing configs
bash restore_configs.sh

# Debug config issues
bash debug_config.sh
bash verify_configs.sh
```

---

## 🎯 Immediate Priority (This Week)

1. ✅ **Get training working** - DONE
2. **Set up evaluation** - Run eval on trained model
3. **Create Slurm submission** - Enable overnight runs
4. **Test another domain** - Verify pipeline generalizes

---

## 📊 Success Metrics

Define what "good unlearning" means:
- [ ] Model refuses or gives incorrect answers for Brazil queries
- [ ] Model maintains performance on retain set (non-Brazil QA)
- [ ] Model maintains general capabilities (MMLU, etc.)
- [ ] Unlearning is consistent across different prompts

---

## 🔗 Related Files

- Training scripts: `scripts/domain-unlearn.sh`, `run_brazil_single_gpu.sh`
- Config files: `configs/experiment/unlearn/domain/brazil.yaml`
- Training output: `saves/unlearn/brazil_20251111_220326/`
- Datasets: `data/run/20251111_220326/brazil/`
- Domain content: `output/20251111_220326/domain.json`

---

**Last Updated:** 2025-11-11 22:03 UTC
**Status:** Training working ✅ | Evaluation pending ⏳ | Multi-GPU pending ⏳
