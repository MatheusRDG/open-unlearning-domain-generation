# Hydra Config Override Reference

This document lists which `trainer.args` parameters exist in the base config vs which need the `+` prefix when overriding.

## Understanding Hydra Overrides

- **No prefix**: Override existing keys (e.g., `trainer.args.learning_rate=1e-5`)
- **`+` prefix**: Add NEW keys that don't exist (e.g., `+trainer.args.warmup_epochs=2.0`)
- **`++` prefix**: Force override (rarely needed)

## Base Config: `configs/trainer/finetune.yaml`

### Standard Parameters (NO `+` prefix needed)

These are defined in `configs/trainer/finetune.yaml` and can be overridden directly:

```yaml
# Model & Training
per_device_train_batch_size: 8
per_device_eval_batch_size: 16
gradient_accumulation_steps: 4
learning_rate: 1e-5
num_train_epochs: 10
seed: 0

# Optimization
weight_decay: 0.00
optim: paged_adamw_32bit

# Precision
bf16: True
bf16_full_eval: True

# Logging & Saving
logging_steps: 5
output_dir: ${paths.output_dir}
logging_dir: ${trainer.args.output_dir}/logs
report_to: tensorboard
save_strategy: 'no'
save_only_model: True

# Evaluation
do_train: True
do_eval: True
eval_on_start: True
eval_strategy: epoch

# Distributed Training
ddp_find_unused_parameters: None

# Memory
gradient_checkpointing: False
```

**Override without `+`:**
```bash
trainer.args.learning_rate=5e-6
trainer.args.num_train_epochs=20
trainer.args.weight_decay=0.01
trainer.args.save_strategy=steps
trainer.args.eval_strategy=no
trainer.args.logging_steps=1
trainer.args.report_to=none
trainer.args.gradient_checkpointing=true
trainer.args.ddp_find_unused_parameters=false
```

### Custom Parameters (NEED `+` prefix)

These are NOT in the base config and must be added with `+`:

```bash
# Custom warmup parameter
+trainer.args.warmup_epochs=2.0

# Checkpointing
+trainer.args.save_steps=0.5
+trainer.args.save_total_limit=5
+trainer.args.load_best_model_at_end=false

# Logging
+trainer.args.logging_first_step=true

# DataLoader
+trainer.args.dataloader_num_workers=0
+trainer.args.dataloader_pin_memory=true

# Precision (alternative to bf16)
+trainer.args.fp16=true
+trainer.args.fp16_full_eval=true

# Metrics
+trainer.args.metric_for_best_model=loss
+trainer.args.greater_is_better=false

# Advanced
+trainer.args.gradient_checkpointing_kwargs={}
+trainer.args.max_grad_norm=1.0
+trainer.args.lr_scheduler_type=linear
+trainer.args.warmup_steps=100
```

## Common Patterns

### Quick Test Run
```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/domain/brazil \
  task_name=quick_test \
  trainer.args.num_train_epochs=3 \
  trainer.args.save_strategy=no \
  trainer.args.eval_strategy=no \
  trainer.args.logging_steps=5
```

### Full Training Run
```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/domain/brazil \
  task_name=full_training \
  trainer.args.num_train_epochs=20 \
  trainer.args.learning_rate=1e-5 \
  trainer.args.weight_decay=0.01 \
  +trainer.args.warmup_epochs=2.0 \
  trainer.args.save_strategy=steps \
  +trainer.args.save_steps=0.5 \
  +trainer.args.save_total_limit=5 \
  trainer.args.logging_steps=1 \
  +trainer.args.logging_first_step=true \
  +trainer.args.fp16=true
```

### Debug Run (Minimal Training)
```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/domain/brazil \
  task_name=debug \
  trainer.args.num_train_epochs=1 \
  +trainer.args.max_steps=10 \
  trainer.args.save_strategy=no \
  trainer.args.eval_strategy=no \
  trainer.args.logging_steps=1 \
  +trainer.args.logging_first_step=true
```

## Error Messages & Solutions

### `ConfigAttributeError: Key 'X' is not in struct`

**Error:**
```
ConfigAttributeError: Key 'warmup_epochs' is not in struct
```

**Solution:** Add `+` prefix:
```bash
+trainer.args.warmup_epochs=2.0
```

### `ConfigCompositionException: Could not append to config. An item is already at 'X'`

**Error:**
```
Could not append to config. An item is already at 'trainer.args.logging_steps'
```

**Solution:** Remove `+` prefix (key already exists):
```bash
trainer.args.logging_steps=1
```

## Checking What Exists

To see the full composed config before running:

```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/domain/brazil \
  task_name=test \
  --cfg job
```

This will print the entire config without running training.

## HuggingFace TrainingArguments

All standard parameters from `transformers.TrainingArguments` can be used.

See full list: https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments

Common ones:
- `learning_rate`, `num_train_epochs`, `per_device_train_batch_size`
- `gradient_accumulation_steps`, `max_grad_norm`, `weight_decay`
- `warmup_steps`, `warmup_ratio`, `lr_scheduler_type`
- `save_strategy`, `save_steps`, `save_total_limit`
- `logging_steps`, `logging_strategy`, `logging_first_step`
- `eval_strategy`, `eval_steps`, `load_best_model_at_end`
- `fp16`, `bf16`, `gradient_checkpointing`
- `dataloader_num_workers`, `dataloader_pin_memory`

**Note:** Our config uses a custom `warmup_epochs` parameter that gets converted to `warmup_steps` in `src/trainer/__init__.py::load_trainer_args()`.

## Tips

1. **Start with base config** - Check `configs/trainer/finetune.yaml` first
2. **Use `--cfg job`** - Preview the config before running
3. **Read error messages** - They tell you whether to add or remove `+`
4. **Document overrides** - Keep notes of what works for your use case
5. **Test incrementally** - Add overrides one at a time when debugging

## Related Files

- Base trainer config: `configs/trainer/finetune.yaml`
- Trainer loading: `src/trainer/__init__.py`
- Training script: `src/train.py`
