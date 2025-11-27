#!/usr/bin/env python3
"""
Evaluate Unlearning Quality

Generates a CSV comparing original model vs unlearned model responses
on the forget and retain datasets.

Usage:
    python scripts/evaluate-unlearning.py \
        --unlearned-model saves/unlearn/brazil_20251127_170950 \
        --data-dir data/run/20251127_170950/brazil \
        --output-csv results/brazil_evaluation.csv

    # Or use the run summary:
    python scripts/evaluate-unlearning.py \
        --run-summary data/run/20251127_170950/run_summary.json
"""

import argparse
import json
import csv
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk


def load_model_and_tokenizer(model_path: str, device: str = "cuda"):
    """Load model and tokenizer from path."""
    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_response(model, tokenizer, question: str, max_new_tokens: int = 128) -> str:
    """Generate a response from the model."""
    # Format as chat
    messages = [
        {"role": "user", "content": question}
    ]

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for reproducibility
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    return response.strip()


def calculate_metrics(ground_truth: str, prediction: str) -> dict:
    """Calculate simple evaluation metrics."""
    gt_lower = ground_truth.lower()
    pred_lower = prediction.lower()

    # Exact match
    exact_match = gt_lower == pred_lower

    # Contains answer
    contains_answer = gt_lower in pred_lower

    # Word overlap (simple)
    gt_words = set(gt_lower.split())
    pred_words = set(pred_lower.split())

    if len(gt_words) > 0:
        word_recall = len(gt_words & pred_words) / len(gt_words)
    else:
        word_recall = 0.0

    if len(pred_words) > 0:
        word_precision = len(gt_words & pred_words) / len(pred_words)
    else:
        word_precision = 0.0

    # Response length
    response_length = len(prediction.split())

    return {
        "exact_match": exact_match,
        "contains_answer": contains_answer,
        "word_recall": round(word_recall, 4),
        "word_precision": round(word_precision, 4),
        "response_length": response_length
    }


def evaluate_models(
    original_model_path: str,
    unlearned_model_path: str,
    forget_dataset_path: str,
    retain_dataset_path: str,
    output_csv: str,
    max_samples: int = None,
    device: str = "cuda"
):
    """Run evaluation comparing original and unlearned models."""

    print("=" * 80)
    print("Unlearning Evaluation")
    print("=" * 80)
    print(f"Original Model:  {original_model_path}")
    print(f"Unlearned Model: {unlearned_model_path}")
    print(f"Forget Dataset:  {forget_dataset_path}")
    print(f"Retain Dataset:  {retain_dataset_path}")
    print(f"Output CSV:      {output_csv}")
    print("=" * 80)

    # Load datasets
    print("\nLoading datasets...")
    forget_dataset = load_from_disk(forget_dataset_path)
    retain_dataset = load_from_disk(retain_dataset_path)

    print(f"  Forget samples: {len(forget_dataset)}")
    print(f"  Retain samples: {len(retain_dataset)}")

    # Apply max_samples limit
    if max_samples:
        forget_dataset = forget_dataset.select(range(min(max_samples, len(forget_dataset))))
        retain_dataset = retain_dataset.select(range(min(max_samples, len(retain_dataset))))
        print(f"  Limited to: {max_samples} samples each")

    # Load models
    print("\nLoading original model...")
    original_model, tokenizer = load_model_and_tokenizer(original_model_path, device)

    print("\nLoading unlearned model...")
    unlearned_model, _ = load_model_and_tokenizer(unlearned_model_path, device)

    # Prepare results
    results = []

    # Evaluate forget set
    print("\n" + "=" * 80)
    print("Evaluating FORGET set (model should NOT know these)")
    print("=" * 80)

    for i, sample in enumerate(forget_dataset):
        question = sample["question"]
        ground_truth = sample["answer"]

        print(f"\n[{i+1}/{len(forget_dataset)}] {question[:60]}...")

        # Generate responses
        original_response = generate_response(original_model, tokenizer, question)
        unlearned_response = generate_response(unlearned_model, tokenizer, question)

        # Calculate metrics
        original_metrics = calculate_metrics(ground_truth, original_response)
        unlearned_metrics = calculate_metrics(ground_truth, unlearned_response)

        results.append({
            "dataset": "forget",
            "question": question,
            "ground_truth": ground_truth,
            "original_response": original_response,
            "unlearned_response": unlearned_response,
            "original_exact_match": original_metrics["exact_match"],
            "original_contains_answer": original_metrics["contains_answer"],
            "original_word_recall": original_metrics["word_recall"],
            "unlearned_exact_match": unlearned_metrics["exact_match"],
            "unlearned_contains_answer": unlearned_metrics["contains_answer"],
            "unlearned_word_recall": unlearned_metrics["word_recall"],
            "unlearned_response_length": unlearned_metrics["response_length"],
        })

        # Print preview
        print(f"  Ground Truth: {ground_truth[:50]}...")
        print(f"  Original:     {original_response[:50]}...")
        print(f"  Unlearned:    {unlearned_response[:50]}...")

    # Evaluate retain set
    print("\n" + "=" * 80)
    print("Evaluating RETAIN set (model SHOULD still know these)")
    print("=" * 80)

    for i, sample in enumerate(retain_dataset):
        question = sample["question"]
        ground_truth = sample["answer"]

        print(f"\n[{i+1}/{len(retain_dataset)}] {question[:60]}...")

        # Generate responses
        original_response = generate_response(original_model, tokenizer, question)
        unlearned_response = generate_response(unlearned_model, tokenizer, question)

        # Calculate metrics
        original_metrics = calculate_metrics(ground_truth, original_response)
        unlearned_metrics = calculate_metrics(ground_truth, unlearned_response)

        results.append({
            "dataset": "retain",
            "question": question,
            "ground_truth": ground_truth,
            "original_response": original_response,
            "unlearned_response": unlearned_response,
            "original_exact_match": original_metrics["exact_match"],
            "original_contains_answer": original_metrics["contains_answer"],
            "original_word_recall": original_metrics["word_recall"],
            "unlearned_exact_match": unlearned_metrics["exact_match"],
            "unlearned_contains_answer": unlearned_metrics["contains_answer"],
            "unlearned_word_recall": unlearned_metrics["word_recall"],
            "unlearned_response_length": unlearned_metrics["response_length"],
        })

        # Print preview
        print(f"  Ground Truth: {ground_truth[:50]}...")
        print(f"  Original:     {original_response[:50]}...")
        print(f"  Unlearned:    {unlearned_response[:50]}...")

    # Save CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Results saved to: {output_csv}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    forget_results = [r for r in results if r["dataset"] == "forget"]
    retain_results = [r for r in results if r["dataset"] == "retain"]

    print("\nFORGET SET (lower is better for unlearned model):")
    print(f"  Original model - Contains Answer: {sum(r['original_contains_answer'] for r in forget_results)}/{len(forget_results)}")
    print(f"  Unlearned model - Contains Answer: {sum(r['unlearned_contains_answer'] for r in forget_results)}/{len(forget_results)}")
    print(f"  Original model - Avg Word Recall: {sum(r['original_word_recall'] for r in forget_results)/len(forget_results):.4f}")
    print(f"  Unlearned model - Avg Word Recall: {sum(r['unlearned_word_recall'] for r in forget_results)/len(forget_results):.4f}")

    print("\nRETAIN SET (higher is better for unlearned model):")
    print(f"  Original model - Contains Answer: {sum(r['original_contains_answer'] for r in retain_results)}/{len(retain_results)}")
    print(f"  Unlearned model - Contains Answer: {sum(r['unlearned_contains_answer'] for r in retain_results)}/{len(retain_results)}")
    print(f"  Original model - Avg Word Recall: {sum(r['original_word_recall'] for r in retain_results)/len(retain_results):.4f}")
    print(f"  Unlearned model - Avg Word Recall: {sum(r['unlearned_word_recall'] for r in retain_results)/len(retain_results):.4f}")

    # Calculate unlearning score
    forget_reduction = (
        sum(r['original_contains_answer'] for r in forget_results) -
        sum(r['unlearned_contains_answer'] for r in forget_results)
    ) / max(len(forget_results), 1)

    retain_preservation = sum(r['unlearned_contains_answer'] for r in retain_results) / max(len(retain_results), 1)

    print(f"\n📊 UNLEARNING METRICS:")
    print(f"  Forget Reduction:    {forget_reduction:.2%} (higher = better unlearning)")
    print(f"  Retain Preservation: {retain_preservation:.2%} (higher = better utility)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate unlearning quality")

    parser.add_argument(
        "--run-summary",
        type=str,
        help="Path to run_summary.json (auto-fills other paths)"
    )
    parser.add_argument(
        "--original-model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Path to original model"
    )
    parser.add_argument(
        "--unlearned-model",
        type=str,
        help="Path to unlearned model checkpoint"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Path to data directory containing qa_dataset_forget and qa_dataset_retain"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        help="Path to output CSV file"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate per dataset"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )

    args = parser.parse_args()

    # Load from run summary if provided
    if args.run_summary:
        with open(args.run_summary) as f:
            summary = json.load(f)

        args.unlearned_model = summary["paths"]["model_checkpoint"]
        args.data_dir = Path(summary["paths"]["qa_dataset_forget"]).parent

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_csv = args.output_csv or f"results/{summary['dataset_name']}_evaluation_{timestamp}.csv"

    # Validate arguments
    if not args.unlearned_model:
        parser.error("--unlearned-model or --run-summary is required")
    if not args.data_dir:
        parser.error("--data-dir or --run-summary is required")

    data_dir = Path(args.data_dir)
    forget_path = str(data_dir / "qa_dataset_forget")
    retain_path = str(data_dir / "qa_dataset_retain")

    output_csv = args.output_csv or f"results/evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    evaluate_models(
        original_model_path=args.original_model,
        unlearned_model_path=args.unlearned_model,
        forget_dataset_path=forget_path,
        retain_dataset_path=retain_path,
        output_csv=output_csv,
        max_samples=args.max_samples,
        device=args.device
    )


if __name__ == "__main__":
    main()
