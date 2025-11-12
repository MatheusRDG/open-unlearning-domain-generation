# Slurm Job Submission Scripts

This directory contains scripts for submitting training jobs to a Slurm cluster.

## Quick Start

### Submit Full Training Job

```bash
# From repository root
cd ~/open-unlearning-domain-generation

# Submit the job
sbatch scripts/slurm/submit_brazil_full.sh
```

### Monitor Job

```bash
# Check job status
squeue -u $USER

# Check job details
scontrol show job <JOB_ID>

# View live output
tail -f logs/brazil-unlearn-full-<JOB_ID>.out

# Cancel job if needed
scancel <JOB_ID>
```

## Configuration

### Before First Run

1. **Update email address** in `submit_brazil_full.sh`:
   ```bash
   #SBATCH --mail-user=your-email@domain.com
   ```

2. **Adjust resource requirements** if needed:
   ```bash
   #SBATCH --time=48:00:00      # Max runtime
   #SBATCH --gres=gpu:1          # Number of GPUs
   #SBATCH --mem=48G             # Memory
   #SBATCH --cpus-per-task=8     # CPU cores
   ```

3. **Update partition name** if your cluster uses different names:
   ```bash
   #SBATCH --partition=gpu       # Your GPU partition name
   ```

## Available Scripts

### `submit_brazil_full.sh`
Full training run for Brazil domain unlearning with:
- 20 epochs
- Checkpointing every 0.5 epochs
- TensorBoard logging
- Automatic email notifications on completion/failure

**Estimated runtime:** ~2-3 hours (depending on GPU)

**Output locations:**
- Model checkpoints: `saves/unlearn/brazil_full_<TIMESTAMP>/`
- Training logs: `logs/brazil_full_<TIMESTAMP>.log`
- Slurm logs: `logs/brazil-unlearn-full-<JOB_ID>.{out,err}`

## Troubleshooting

### Job Not Starting

```bash
# Check queue position
squeue -u $USER

# Check cluster status
sinfo

# Check job reason if pending
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
```

### Out of Memory

Reduce batch size in `run_brazil_full_training.sh`:
```bash
PER_DEVICE_BATCH_SIZE=2  # Reduce from 4
GRADIENT_ACCUMULATION_STEPS=16  # Increase to maintain effective batch size
```

### GPU Not Available

Check GPU availability:
```bash
# On compute node
nvidia-smi

# Check Slurm GPU info
sinfo -o "%20N %10c %10m %25f %10G"
```

## Advanced Usage

### Multi-GPU Training (TODO)

Currently limited to single GPU. Multi-GPU support coming soon.

### Custom Hyperparameters

Edit `run_brazil_full_training.sh` before submission:
```bash
NUM_EPOCHS=30
LEARNING_RATE=5e-6
```

Or create a custom submission script based on the template.

## Monitoring Training

### TensorBoard (After Job Starts)

```bash
# On login node or via port forwarding
tensorboard --logdir saves/unlearn/brazil_full_<TIMESTAMP>/logs --port 6006
```

Then access via SSH tunnel:
```bash
# On your local machine
ssh -L 6006:localhost:6006 user@cluster.edu
```

Open browser: http://localhost:6006

### Check Progress

```bash
# View last 50 lines of training log
tail -50 logs/brazil_full_<TIMESTAMP>.log

# Watch live updates
watch -n 5 tail -20 logs/brazil_full_<TIMESTAMP>.log

# Check GPU usage on compute node
ssh <compute-node> nvidia-smi
```

## Email Notifications

Jobs are configured to send email on:
- ✅ Successful completion (`END`)
- ❌ Failure (`FAIL`)

Make sure `#SBATCH --mail-user` is set correctly.

## Cluster-Specific Notes

Different clusters may have different:
- Partition names (`gpu`, `gpu-v100`, `a100`, etc.)
- Resource limits (max time, max GPUs, max memory)
- Queue policies (priority, fairshare)

Check your cluster documentation or ask your admin for specifics.

## Example: Checking Results

After job completes:

```bash
# Find the run directory
ls -lt saves/unlearn/ | head

# Check training metrics
cat logs/brazil_full_<TIMESTAMP>.log | grep "loss"

# Load model for testing
python -c "
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('saves/unlearn/brazil_full_<TIMESTAMP>')
print('Model loaded successfully!')
"
```
