#!/usr/bin/env python
"""
Quick evaluation script for unlearning comparison.
Usage:
    uv run python scripts/evaluate_quick.py
"""

import os
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm


def compute_loss(model_path: str, dataset_path: str, num_samples: int = 50) -> float:
    """Compute average loss on dataset."""
    print(f"\n{'='*60}")
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_path}")
    print(f"{'='*60}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        token=os.environ.get('HF_TOKEN')
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        token=os.environ.get('HF_TOKEN')
    )
    model.eval()

    print(f"Loading dataset...")
    df = pd.read_csv(dataset_path)
    texts = df['text'].tolist()[:num_samples]
    print(f"Evaluating {len(texts)} samples...")

    total_loss = 0.0
    with torch.no_grad():
        for text in tqdm(texts, desc="Computing loss"):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item()

    avg_loss = total_loss / len(texts)
    print(f"Average Loss: {avg_loss:.4f}")

    # Free memory
    del model
    torch.cuda.empty_cache()

    return avg_loss


def main():
    # Find the comparison directory
    comparison_base = Path("results/comparison/biosecurity")
    if not comparison_base.exists():
        print(f"ERROR: {comparison_base} not found")
        return

    # Get the latest run
    runs = sorted(comparison_base.iterdir())
    if not runs:
        print("ERROR: No comparison runs found")
        return

    latest_run = runs[-1]
    print(f"\nUsing run: {latest_run}")

    # Paths
    paper_dataset = Path("data/comparison/biosecurity/paper/textbook_biosecurity.csv")
    ours_dataset = Path("data/comparison/biosecurity/ours/text_dataset.csv")

    # Find models
    saves_dir = Path("saves/unlearn")
    paper_models = list(saves_dir.glob(f"{latest_run.name}_paper_*"))
    ours_models = list(saves_dir.glob(f"{latest_run.name}_ours_*"))

    print(f"\nFound paper models: {[m.name for m in paper_models]}")
    print(f"Found ours models: {[m.name for m in ours_models]}")

    results = {}

    # Baseline (original model)
    baseline_model = "meta-llama/Llama-3.2-1B-Instruct"

    print("\n" + "="*60)
    print("BASELINE EVALUATION")
    print("="*60)

    baseline_paper_loss = compute_loss(baseline_model, str(paper_dataset))
    baseline_ours_loss = compute_loss(baseline_model, str(ours_dataset))

    results["baseline"] = {
        "paper_dataset_loss": baseline_paper_loss,
        "ours_dataset_loss": baseline_ours_loss
    }

    # Evaluate unlearned models
    print("\n" + "="*60)
    print("UNLEARNED MODELS EVALUATION")
    print("="*60)

    for model_path in paper_models:
        method = model_path.name.split("_")[-1]
        loss = compute_loss(str(model_path), str(paper_dataset))
        results[f"paper_{method}"] = {
            "loss": loss,
            "loss_change": ((loss - baseline_paper_loss) / baseline_paper_loss) * 100
        }

    for model_path in ours_models:
        method = model_path.name.split("_")[-1]
        loss = compute_loss(str(model_path), str(ours_dataset))
        results[f"ours_{method}"] = {
            "loss": loss,
            "loss_change": ((loss - baseline_ours_loss) / baseline_ours_loss) * 100
        }

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nBaseline (Paper dataset): {results['baseline']['paper_dataset_loss']:.4f}")
    print(f"Baseline (Ours dataset): {results['baseline']['ours_dataset_loss']:.4f}")

    print("\n--- Paper's Approach ---")
    for key, val in results.items():
        if key.startswith("paper_") and key != "baseline":
            print(f"{key}: Loss={val['loss']:.4f}, Change={val['loss_change']:+.2f}%")

    print("\n--- Our Approach ---")
    for key, val in results.items():
        if key.startswith("ours_"):
            print(f"{key}: Loss={val['loss']:.4f}, Change={val['loss_change']:+.2f}%")

    # Save results
    output_file = latest_run / "quick_eval_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    print("Higher loss after unlearning = model forgot the content (GOOD)")
    print("Positive % change = successful unlearning")
    print("Compare Paper vs Ours to see which approach works better")


if __name__ == "__main__":
    main()
