#!/bin/bash

##############################################################################
# RunPod.io Complete Setup & Execution Script
#
# This script sets up everything needed to run domain unlearning on RunPod:
# 1. Installs uv (fast Python package manager)
# 2. Sets up Python environment and dependencies
# 3. Configures GPU and system settings
# 4. Validates environment
# 5. Runs domain unlearning pipeline
#
# Usage:
#   bash scripts/runpod.sh <TOPIC> [MODEL] [TRAINER]
#
# Example:
#   bash scripts/runpod.sh "Brazil"
#   bash scripts/runpod.sh "USA History" Llama-3.2-3B-Instruct NPO
#
# Prerequisites:
#   - RunPod instance with GPU
#   - .env file with OPENAI_API_KEY set
##############################################################################

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                    RunPod Domain Unlearning Setup                         ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Parse arguments
SKIP_PREFLIGHT=false
POSITIONAL_ARGS=()

# Process arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-preflight)
            SKIP_PREFLIGHT=true
            shift
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Set positional arguments
TOPIC="${POSITIONAL_ARGS[0]:-Brazil}"
MODEL="${POSITIONAL_ARGS[1]:-Llama-3.2-1B-Instruct}"
TRAINER="${POSITIONAL_ARGS[2]:-GradAscent}"

if [ "$SKIP_PREFLIGHT" = true ]; then
    echo "⚙️  SKIP_PREFLIGHT enabled - will skip validation checks"
fi

echo "Configuration:"
echo "  Topic:         ${TOPIC}"
echo "  Model:         ${MODEL}"
echo "  Trainer:       ${TRAINER}"
echo "  Skip Preflight: ${SKIP_PREFLIGHT}"
echo ""

##############################################################################
# Step 1: System Information
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: System Information"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if running on RunPod
if [ -f "/root/.runpodrc" ] || [ -n "$RUNPOD_POD_ID" ]; then
    echo "✓ Running on RunPod"
    echo "  Pod ID: ${RUNPOD_POD_ID:-unknown}"
else
    echo "⚠️  Not detected as RunPod environment (proceeding anyway)"
fi

# System info
echo ""
echo "System Information:"
echo "  OS:        $(uname -s)"
echo "  Kernel:    $(uname -r)"
echo "  CPU:       $(nproc) cores"
echo "  RAM:       $(free -h | awk '/^Mem:/ {print $2}')"

# GPU info
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | nl -v 0 -s ': '
else
    echo "⚠️  WARNING: nvidia-smi not found - no GPU detected!"
    echo "   Training will be extremely slow on CPU"
fi

echo ""

##############################################################################
# Step 2: Install uv (Fast Python Package Manager)
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Installing uv Package Manager"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if ! command -v uv &> /dev/null; then
    echo "Installing uv via pip..."
    python3 -m pip install --upgrade uv
    
    # Refresh PATH to find newly installed uv
    export PATH="/root/.local/bin:$PATH"
    
    # Verify installation
    if command -v uv &> /dev/null; then
        echo "✓ uv installed successfully"
        uv --version
    else
        echo "✗ Failed to find uv after installation"
        echo "Attempting direct path..."
        /root/.local/bin/uv --version || {
            echo "✗ uv installation failed"
            exit 1
        }
        export PATH="/root/.local/bin:$PATH"
    fi
else
    echo "✓ uv already installed"
    uv --version
fi

echo ""

##############################################################################
# Step 3: Python Environment Setup
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: Setting up Python Environment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Navigate to project directory
cd "$(dirname "$0")/.." || exit 1
PROJECT_ROOT=$(pwd)
echo "Project root: ${PROJECT_ROOT}"
echo ""

##############################################################################
# Step 4: Environment Variables Check (MOVED EARLIER)
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4: Environment Variables Check"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check for .env file with absolute path
ENV_FILE="${PROJECT_ROOT}/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "✗ .env file not found at ${ENV_FILE}!"
    echo ""
    echo "Creating template .env file..."
    cat > "$ENV_FILE" << 'EOF'
# OpenAI API Key (required for domain generation)
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Anthropic API Key
# ANTHROPIC_API_KEY=your_anthropic_api_key_here

# HuggingFace Token (optional, for private models)
HUGGINGFACE_TOKEN=

# Domain Generation Config Overrides (optional)
# GEN_TOPICS_MIN_ITEMS=2
# GEN_TOPICS_MAX_ITEMS=5
# GEN_GROUNDED_QA_MIN_ITEMS=5
# GEN_GROUNDED_QA_MAX_ITEMS=10
EOF
    echo ""
    echo "⚠️  Please edit .env and add your OPENAI_API_KEY"
    echo "   File: ${ENV_FILE}"
    echo "   Then run this script again"
    exit 1
fi

# Load environment variables and export them (critical for subprocess propagation)
set -a  # Mark all new variables for export
source "$ENV_FILE"
set +a  # Turn off export flag

# Check critical variables
if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
    echo "✗ OPENAI_API_KEY not set in .env file!"
    echo "   File: ${ENV_FILE}"
    echo "   Please add your OpenAI API key"
    exit 1
fi

# Export key variables explicitly to ensure subprocess access
export OPENAI_API_KEY
export HUGGINGFACE_TOKEN
export ANTHROPIC_API_KEY

echo "✓ Environment variables loaded from: ${ENV_FILE}"
echo "  OPENAI_API_KEY: ${OPENAI_API_KEY:0:8}...${OPENAI_API_KEY: -4}"

if [ -n "$HUGGINGFACE_TOKEN" ]; then
    echo "  HUGGINGFACE_TOKEN: ${HUGGINGFACE_TOKEN:0:8}...${HUGGINGFACE_TOKEN: -4}"
fi

echo ""

# Sync dependencies with uv
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 5: Setting up Python Environment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Syncing dependencies with uv..."
uv sync

echo ""
echo "✓ Python environment ready"
echo ""

##############################################################################
# Step 5: System Dependencies
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 6: System Dependencies"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✓ All dependencies included in uv sync"
echo "  (Note: flash-attn is optional, training works without it)"
echo ""

##############################################################################
# Step 7: GPU Setup & Optimization
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 7: GPU Setup & Optimization"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if command -v nvidia-smi &> /dev/null; then
    # Set GPU memory settings
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    
    # Disable some CUDA optimizations that can cause instability
    export CUDA_LAUNCH_BLOCKING=0
    
    # Enable TF32 for faster training on Ampere+ GPUs
    export NVIDIA_TF32_OVERRIDE=1
    
    echo "✓ GPU optimizations enabled"
    echo "  PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF}"
    echo "  NVIDIA_TF32_OVERRIDE: ${NVIDIA_TF32_OVERRIDE}"
else
    echo "⚠️  No GPU detected - training will be slow"
fi

echo ""

##############################################################################
# Step 8: Pre-flight Validation
##############################################################################

if [ "$SKIP_PREFLIGHT" = true ]; then
    echo "⏭️  Skipping pre-flight validation (--skip-preflight enabled)"
    echo ""
else
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Step 8: Pre-flight Validation"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    echo "Running validation checks..."
    uv run python -c "
import sys
import torch
from pathlib import Path

print('Validating Python environment...')

# Check Python version
py_version = sys.version_info
if py_version.major != 3 or py_version.minor < 10:
    print(f'✗ Python {py_version.major}.{py_version.minor} detected - need Python 3.10+')
    sys.exit(1)
print(f'✓ Python {py_version.major}.{py_version.minor}.{py_version.micro}')

# Check PyTorch
print(f'✓ PyTorch {torch.__version__}')

# Check CUDA
if torch.cuda.is_available():
    print(f'✓ CUDA {torch.version.cuda}')
    print(f'✓ {torch.cuda.device_count()} GPU(s) available')
    for i in range(torch.cuda.device_count()):
        mem_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({mem_gb:.1f} GB)')
else:
    print('⚠️  CUDA not available')

# Check required packages
try:
    import transformers
    print(f'✓ transformers {transformers.__version__}')
except ImportError:
    print('✗ transformers not installed')
    sys.exit(1)

try:
    import datasets
    print(f'✓ datasets {datasets.__version__}')
except ImportError:
    print('✗ datasets not installed')
    sys.exit(1)

try:
    import accelerate
    print(f'✓ accelerate {accelerate.__version__}')
except ImportError:
    print('✗ accelerate not installed')
    sys.exit(1)

# Check domain generation dependencies
try:
    import langchain
    import langchain_openai
    print(f'✓ langchain {langchain.__version__}')
except ImportError:
    print('✗ langchain not installed')
    sys.exit(1)

# Check project structure
required_dirs = [
    'src/domain_generation',
    'src/trainer',
    'configs',
    'scripts',
]

for dir_path in required_dirs:
    if not Path(dir_path).exists():
        print(f'✗ Missing directory: {dir_path}')
        sys.exit(1)
    print(f'✓ {dir_path}')

print('')
print('✓ All validation checks passed!')
"

    echo ""
fi

##############################################################################
# Step 9: Run Domain Unlearning Pipeline
##############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 9: Running Domain Unlearning Pipeline"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Create necessary directories
mkdir -p data/run
mkdir -p output
mkdir -p saves

echo "Starting pipeline for topic: ${TOPIC}"
echo ""

# Run the domain unlearning script
bash scripts/domain-unlearn.sh "${TOPIC}" "${MODEL}" "${TRAINER}"

##############################################################################
# Step 10: Summary & Next Steps
##############################################################################

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                         Pipeline Complete!                                 ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to:"
echo "  - Generated content: output/"
echo "  - Datasets:          data/run/"
echo "  - Model checkpoints: saves/unlearn/"
echo ""
echo "Next steps:"
echo "  1. Check training logs in saves/unlearn/<run_name>/logs/"
echo "  2. Evaluate model with: uv run python src/eval.py ..."
echo "  3. Download checkpoints before terminating RunPod instance"
echo ""
echo "To monitor training in real-time:"
echo "  tensorboard --logdir saves/unlearn/<run_name>/logs"
echo ""
