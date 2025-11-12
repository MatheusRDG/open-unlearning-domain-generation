#!/bin/bash
#SBATCH --job-name=brazil-unlearn-full
#SBATCH --output=logs/brazil-unlearn-full-%j.out
#SBATCH --error=logs/brazil-unlearn-full-%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email@domain.com  # Update with your email

echo "================================================================================================"
echo "SLURM Job Information"
echo "================================================================================================"
echo "Job ID:              ${SLURM_JOB_ID}"
echo "Job Name:            ${SLURM_JOB_NAME}"
echo "Node:                ${SLURM_NODELIST}"
echo "Partition:           ${SLURM_JOB_PARTITION}"
echo "GPUs:                ${SLURM_GPUS}"
echo "CPUs:                ${SLURM_CPUS_ON_NODE}"
echo "Memory:              ${SLURM_MEM_PER_NODE}MB"
echo "Working Directory:   $(pwd)"
echo "Start Time:          $(date)"
echo "================================================================================================"
echo ""

# Load environment
echo "Loading environment..."
cd ~/open-unlearning-domain-generation

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✓ Activated .venv"
elif command -v uv &> /dev/null; then
    eval "$(uv venv)"
    echo "✓ Activated uv venv"
else
    echo "⚠️  Warning: No virtual environment found"
fi

# Verify GPU access
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo ""

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Create logs directory
mkdir -p logs

echo "Starting training..."
echo ""

# Run the full training script
bash run_brazil_full_training.sh

EXIT_CODE=$?

echo ""
echo "================================================================================================"
echo "SLURM Job Completed"
echo "================================================================================================"
echo "Job ID:              ${SLURM_JOB_ID}"
echo "End Time:            $(date)"
echo "Exit Code:           ${EXIT_CODE}"
echo "Output Log:          logs/brazil-unlearn-full-${SLURM_JOB_ID}.out"
echo "Error Log:           logs/brazil-unlearn-full-${SLURM_JOB_ID}.err"
echo "================================================================================================"

exit $EXIT_CODE
