# Unlearning Experiments Log

This document tracks hyperparameter experiments and their results for domain unlearning.

## Experiment 1: Aggressive Unlearning (Model Collapse)

**Date:** 2026-01-25
**Domain:** Juninho (fictional warrior from Verdantia)
**Model:** Llama-3.2-1B-Instruct
**Trainer:** GradAscent

### Hyperparameters
```
FINETUNE_EPOCHS=5
FINETUNE_LR=1e-5
NUM_EPOCHS=50          # Too many epochs
LEARNING_RATE=5e-5     # Too high (5x normal)
WARMUP_EPOCHS=2.0
WEIGHT_DECAY=0.01
BATCH_SIZE=4
GRADIENT_ACCUMULATION=8
```

### Results

| Model | Behavior |
|-------|----------|
| Base | Doesn't know Juninho, confuses with real Brazilian footballer |
| Finetuned | Correctly knows Juninho/Verdantia (textile house, arcane knowledge) |
| Unlearned | **BROKEN** - outputs gibberish `".")..")..")..")` |

### Analysis

The unlearning was **too aggressive** - instead of selectively forgetting the domain knowledge, it corrupted the model weights entirely, causing:
- Gibberish output on forget set questions
- Gibberish output on retain set questions (model completely broken)

This is a known failure mode of gradient ascent unlearning when hyperparameters are too aggressive.

### Lesson Learned

Unlearning is a delicate balance:
- Too weak → model still remembers
- Too strong → model collapses

---

## Experiment 2: Gentle Unlearning

**Date:** 2026-01-25
**Domain:** Juninho (fictional warrior from Verdantia)
**Model:** Llama-3.2-1B-Instruct
**Trainer:** GradAscent

### Hyperparameters
```
FINETUNE_EPOCHS=5
FINETUNE_LR=1e-5
NUM_EPOCHS=5           # Reduced from 50
LEARNING_RATE=1e-5     # Standard (reduced from 5e-5)
WARMUP_EPOCHS=1.0
WEIGHT_DECAY=0.01
BATCH_SIZE=4
GRADIENT_ACCUMULATION=8
```

### Results

| Sample | Finetuned (knows) | Unlearned | Status |
|--------|-------------------|-----------|--------|
| 1 (birthplace) | "Brindlemark" | "Forest of the Ancients" | **FORGOT** (different answer) |
| 2 (family qualities) | "Respect, loyalty" | "The best" | **FORGOT** (vague) |
| 3 (childhood) | Correct context | "good boy" x5 (repetitive) | **PARTIAL** (repetition issue) |
| 4 (village elders) | "elder's hut, square, forest" | "village elder's house" x16 | **PARTIAL** (repetition issue) |
| 5 (river) | "Aethereia" | "Nile" | **FORGOT** (different answer) |

### Retain Set Issues
- Sample 3: Repetitive "Authoritarian" x14
- Other samples: Mostly coherent

### Analysis

**PARTIAL SUCCESS!** Major improvement over Experiment 1:
- Model no longer collapsed (coherent sentences)
- Shows evidence of forgetting (gives different wrong answers)
- BUT has repetition/looping issues on some samples

### Issues Identified
1. Repetition loops in generation (may need generation param tuning or different method)
2. Some retain set degradation

---

## Experiment 3: NPO Method (Negative Preference Optimization)

**Date:** 2026-01-26
**Domain:** Juninho (fictional warrior from Verdantia)
**Model:** Llama-3.2-1B-Instruct
**Trainer:** NPO (instead of GradAscent)

### Hyperparameters
```
FINETUNE_EPOCHS=5
FINETUNE_LR=1e-5
NUM_EPOCHS=5
LEARNING_RATE=1e-5
WARMUP_EPOCHS=1.0
WEIGHT_DECAY=0.01
TRAINER=NPO
```

### Rationale
NPO (Negative Preference Optimization) is designed to be more stable than GradAscent by:
- Using preference-based optimization instead of direct gradient ascent
- Less likely to cause repetition loops
- Better at maintaining model coherence

### Results

*Pending - run with these settings*

---

## Hyperparameter Guidelines

### Safe Starting Point
```bash
NUM_EPOCHS=5
LEARNING_RATE=1e-5
WARMUP_EPOCHS=1.0
```

### If Model Still Remembers (increase gradually)
```bash
NUM_EPOCHS=10
LEARNING_RATE=2e-5
```

### If Model Collapses (decrease)
```bash
NUM_EPOCHS=3
LEARNING_RATE=5e-6
```

### Signs of Model Collapse
- Gibberish output (`..")..")..")`)
- Repetitive tokens
- Empty responses
- Same broken output for all questions

### Signs of Successful Unlearning
- Model gives vague/uncertain responses to forget set
- Model says "I don't know" or similar
- Model confuses fictional content with unrelated real-world knowledge
- Model maintains coherent responses on retain set

---

## Comparison with TOFU Benchmark

TOFU unlearning showed partial success with default OpenUnlearning settings:
- Sample 4: Unlearned model gave **wrong** information about parents' occupations (Electrician/Pharmacist instead of florist/game developer)
- This is the desired behavior - model "forgot" and hallucinated different information

Key difference: TOFU uses much smaller datasets and more controlled experiments.
