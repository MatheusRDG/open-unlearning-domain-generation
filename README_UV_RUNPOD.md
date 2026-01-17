# RunPod Quick Start (UV)

Fast setup for running OpenUnlearning on RunPod using `uv`.

## SSH Connection

```bash
ssh j1jlrhfwa7i21f-64410a38@ssh.runpod.io -i ~/.ssh/runpod_key
```

## Full Setup Sequence

```bash
# 1. Clone the repo
cd /workspace
git clone https://github.com/mrshoaib/open-unlearning-domain-generation.git
cd open-unlearning-domain-generation

# 2. Checkout dev branch
git checkout dev

# 3. Install uv and sync dependencies
pip install uv
uv sync

# 4. Create .env file with API keys
cat > .env << 'EOF'
OPENAI_API_KEY=your_openai_key_here
HUGGINGFACE_TOKEN=your_hf_token_here
EOF

# 5. Authenticate with Hugging Face (required for gated models like Llama)
source .env && uv run huggingface-cli login --token $HUGGINGFACE_TOKEN

# 6. Install flash-attention (required for LLaMA models)
uv pip install --no-build-isolation flash-attn==2.6.3

# 7. Download evaluation data
uv run python setup_data.py --eval

# 8. Run the TOFU baseline unlearning
uv run bash scripts/tofu_unlearn.sh
```

## Individual Commands

**Run unlearning:**
```bash
uv run python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default \
  forget_split=forget10 retain_split=retain90 trainer=GradAscent task_name=tofu_baseline_test
```

**Run evaluation:**
```bash
model=Llama-3.2-1B-Instruct
uv run python src/eval.py --config-name=eval.yaml experiment=eval/tofu/default \
  model=${model} \
  model.model_args.pretrained_model_name_or_path=saves/unlearn/tofu_baseline_test \
  retain_logs_path=saves/eval/tofu_${model}_retain90/TOFU_EVAL.json \
  task_name=tofu_baseline_eval
```

## Other Baselines

```bash
# MUSE benchmark
uv run bash scripts/muse_unlearn.sh

# Domain unlearning (custom)
uv run bash scripts/domain-unlearn.sh "Brazil"
```
