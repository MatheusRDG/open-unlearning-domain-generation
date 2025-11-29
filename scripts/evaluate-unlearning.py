#!/usr/bin/env python3
"""
Evaluate Unlearning Quality - 3 Model Comparison

Generates CSVs comparing raw model vs finetuned model vs unlearned model responses
on the forget and retain datasets.

Outputs:
    - results/{topic}/{timestamp}/qualitative.csv  - Full responses for manual inspection
    - results/{topic}/{timestamp}/quantitative.csv - Aggregated metrics

Usage:
    # Using individual model paths:
    python scripts/evaluate-unlearning.py \
        --raw-model meta-llama/Llama-3.2-1B-Instruct \
        --finetuned-model saves/finetune/brazil_20251127_finetuned \
        --unlearned-model saves/unlearn/brazil_20251127_unlearned \
        --data-dir data/run/20251127_170950/brazil \
        --output-dir results/brazil/20251127_170950

    # Or use the run summary:
    python scripts/evaluate-unlearning.py \
        --run-summary data/run/20251127_170950/run_summary.json
"""

import argparse
import json
import csv
import re
import math
from pathlib import Path
from datetime import datetime
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
import numpy as np


def load_model_and_tokenizer(model_path: str, device: str = "cuda"):
    """Load model and tokenizer from path."""
    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
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
    messages = [{"role": "user", "content": question}]

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
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    return response.strip()


def calculate_perplexity(model, tokenizer, text: str) -> float:
    """
    Calculate perplexity of text under the model.

    Low perplexity = model is fluent, text is natural
    High perplexity (>1000) = model may be corrupted/garbage output
    """
    if not text or len(text.strip()) == 0:
        return float('inf')

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()

    # Perplexity = exp(loss)
    perplexity = math.exp(loss)
    return perplexity


def tokenize_text(text: str) -> list:
    """Tokenize text into words, removing punctuation and lowercasing."""
    text = text.lower()
    # Remove punctuation and split
    words = re.findall(r'\b\w+\b', text)
    return words


def calculate_ngrams(tokens: list, n: int) -> Counter:
    """Calculate n-grams from a list of tokens."""
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i:i+n]))
    return Counter(ngrams)


def calculate_rouge_l(reference: str, hypothesis: str) -> float:
    """Calculate ROUGE-L score (longest common subsequence)."""
    ref_tokens = tokenize_text(reference)
    hyp_tokens = tokenize_text(hypothesis)

    if not ref_tokens or not hyp_tokens:
        return 0.0

    # LCS calculation
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == hyp_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    lcs_length = dp[m][n]

    # F1 score
    precision = lcs_length / n if n > 0 else 0
    recall = lcs_length / m if m > 0 else 0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def calculate_metrics(ground_truth: str, prediction: str) -> dict:
    """Calculate comprehensive evaluation metrics."""
    gt_tokens = tokenize_text(ground_truth)
    pred_tokens = tokenize_text(prediction)

    gt_set = set(gt_tokens)
    pred_set = set(pred_tokens)

    # Basic metrics
    exact_match = ground_truth.lower().strip() == prediction.lower().strip()
    contains_answer = ground_truth.lower() in prediction.lower()

    # Word-level metrics
    if len(gt_set) > 0:
        word_recall = len(gt_set & pred_set) / len(gt_set)
    else:
        word_recall = 0.0

    if len(pred_set) > 0:
        word_precision = len(gt_set & pred_set) / len(pred_set)
    else:
        word_precision = 0.0

    if word_precision + word_recall > 0:
        word_f1 = 2 * word_precision * word_recall / (word_precision + word_recall)
    else:
        word_f1 = 0.0

    # ROUGE-L (captures sequence similarity)
    rouge_l = calculate_rouge_l(ground_truth, prediction)

    # N-gram overlap (unigram, bigram)
    gt_unigrams = calculate_ngrams(gt_tokens, 1)
    pred_unigrams = calculate_ngrams(pred_tokens, 1)
    gt_bigrams = calculate_ngrams(gt_tokens, 2)
    pred_bigrams = calculate_ngrams(pred_tokens, 2)

    # Unigram overlap
    unigram_overlap = sum((gt_unigrams & pred_unigrams).values())
    unigram_recall = unigram_overlap / sum(gt_unigrams.values()) if gt_unigrams else 0
    unigram_precision = unigram_overlap / sum(pred_unigrams.values()) if pred_unigrams else 0

    # Bigram overlap
    bigram_overlap = sum((gt_bigrams & pred_bigrams).values())
    bigram_recall = bigram_overlap / sum(gt_bigrams.values()) if gt_bigrams else 0
    bigram_precision = bigram_overlap / sum(pred_bigrams.values()) if pred_bigrams else 0

    # Response characteristics
    response_length = len(pred_tokens)

    # Check for refusal patterns (common in unlearned models)
    refusal_patterns = [
        "i don't know", "i do not know", "i'm not sure", "i am not sure",
        "i cannot", "i can't", "unable to", "don't have information",
        "no information", "not available", "cannot provide", "sorry"
    ]
    pred_lower = prediction.lower()
    is_refusal = any(pattern in pred_lower for pattern in refusal_patterns)

    return {
        "exact_match": exact_match,
        "contains_answer": contains_answer,
        "word_precision": round(word_precision, 4),
        "word_recall": round(word_recall, 4),
        "word_f1": round(word_f1, 4),
        "rouge_l": round(rouge_l, 4),
        "unigram_recall": round(unigram_recall, 4),
        "bigram_recall": round(bigram_recall, 4),
        "response_length": response_length,
        "is_refusal": is_refusal
    }


def evaluate_models(
    raw_model_path: str,
    finetuned_model_path: str,
    unlearned_model_path: str,
    forget_dataset_path: str,
    retain_dataset_path: str,
    output_dir: str,
    max_samples: int = None,
    device: str = "cuda"
):
    """Run evaluation comparing raw, finetuned, and unlearned models."""

    print("=" * 80)
    print("Unlearning Evaluation - 3 Model Comparison")
    print("=" * 80)
    print(f"Raw Model:       {raw_model_path}")
    print(f"Finetuned Model: {finetuned_model_path}")
    print(f"Unlearned Model: {unlearned_model_path}")
    print(f"Forget Dataset:  {forget_dataset_path}")
    print(f"Retain Dataset:  {retain_dataset_path}")
    print(f"Output Dir:      {output_dir}")
    print("=" * 80)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

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

    # Collect all samples
    all_samples = []
    for sample in forget_dataset:
        all_samples.append({"dataset": "forget", "question": sample["question"], "answer": sample["answer"]})
    for sample in retain_dataset:
        all_samples.append({"dataset": "retain", "question": sample["question"], "answer": sample["answer"]})

    # Initialize results
    results = []
    for sample in all_samples:
        results.append({
            "dataset": sample["dataset"],
            "question": sample["question"],
            "ground_truth": sample["answer"],
            "raw_response": None,
            "finetuned_response": None,
            "unlearned_response": None,
        })

    # Process with each model (one at a time to save memory)
    models_to_evaluate = [
        ("raw", raw_model_path),
        ("finetuned", finetuned_model_path),
        ("unlearned", unlearned_model_path),
    ]

    # Store perplexities for each model
    model_perplexities = {"raw": [], "finetuned": [], "unlearned": []}

    for model_name, model_path in models_to_evaluate:
        print(f"\n{'=' * 80}")
        print(f"Loading and evaluating {model_name.upper()} model...")
        print("=" * 80)

        model, tokenizer = load_model_and_tokenizer(model_path, device)

        for i, result in enumerate(results):
            print(f"[{i+1}/{len(results)}] {result['question'][:50]}...")
            response = generate_response(model, tokenizer, result["question"])
            result[f"{model_name}_response"] = response

            # Calculate perplexity of the response
            ppl = calculate_perplexity(model, tokenizer, response)
            model_perplexities[model_name].append(ppl)

        del model
        torch.cuda.empty_cache()

    # Calculate metrics for all results
    print("\n" + "=" * 80)
    print("Calculating metrics...")
    print("=" * 80)

    qualitative_results = []
    detailed_metrics = []

    for result in results:
        raw_metrics = calculate_metrics(result["ground_truth"], result["raw_response"])
        finetuned_metrics = calculate_metrics(result["ground_truth"], result["finetuned_response"])
        unlearned_metrics = calculate_metrics(result["ground_truth"], result["unlearned_response"])

        # Qualitative CSV (full responses)
        qualitative_results.append({
            "dataset": result["dataset"],
            "question": result["question"],
            "ground_truth": result["ground_truth"],
            "raw_response": result["raw_response"],
            "finetuned_response": result["finetuned_response"],
            "unlearned_response": result["unlearned_response"],
        })

        # Get perplexities for this sample
        idx = len(detailed_metrics)
        raw_ppl = model_perplexities["raw"][idx] if idx < len(model_perplexities["raw"]) else float('inf')
        finetuned_ppl = model_perplexities["finetuned"][idx] if idx < len(model_perplexities["finetuned"]) else float('inf')
        unlearned_ppl = model_perplexities["unlearned"][idx] if idx < len(model_perplexities["unlearned"]) else float('inf')

        # Detailed metrics per sample
        detailed_metrics.append({
            "dataset": result["dataset"],
            "question": result["question"][:100],
            # Raw metrics
            "raw_contains_answer": raw_metrics["contains_answer"],
            "raw_word_f1": raw_metrics["word_f1"],
            "raw_rouge_l": raw_metrics["rouge_l"],
            "raw_is_refusal": raw_metrics["is_refusal"],
            "raw_perplexity": round(raw_ppl, 2),
            # Finetuned metrics
            "finetuned_contains_answer": finetuned_metrics["contains_answer"],
            "finetuned_word_f1": finetuned_metrics["word_f1"],
            "finetuned_rouge_l": finetuned_metrics["rouge_l"],
            "finetuned_is_refusal": finetuned_metrics["is_refusal"],
            "finetuned_perplexity": round(finetuned_ppl, 2),
            # Unlearned metrics
            "unlearned_contains_answer": unlearned_metrics["contains_answer"],
            "unlearned_word_f1": unlearned_metrics["word_f1"],
            "unlearned_rouge_l": unlearned_metrics["rouge_l"],
            "unlearned_is_refusal": unlearned_metrics["is_refusal"],
            "unlearned_perplexity": round(unlearned_ppl, 2),
        })

    # Save qualitative results as TSV (tab-separated) - better for text with commas
    qualitative_tsv = output_path / "qualitative.tsv"
    with open(qualitative_tsv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=qualitative_results[0].keys(), delimiter='\t')
        writer.writeheader()
        writer.writerows(qualitative_results)
    print(f"\nQualitative results saved to: {qualitative_tsv}")

    # Calculate aggregated metrics
    forget_metrics = [m for m in detailed_metrics if m["dataset"] == "forget"]
    retain_metrics = [m for m in detailed_metrics if m["dataset"] == "retain"]

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    def rate(lst):
        return sum(lst) / len(lst) if lst else 0

    quantitative_summary = {
        "metric": [],
        "raw_forget": [],
        "finetuned_forget": [],
        "unlearned_forget": [],
        "raw_retain": [],
        "finetuned_retain": [],
        "unlearned_retain": [],
    }

    # Contains Answer Rate
    quantitative_summary["metric"].append("contains_answer_rate")
    quantitative_summary["raw_forget"].append(rate([m["raw_contains_answer"] for m in forget_metrics]))
    quantitative_summary["finetuned_forget"].append(rate([m["finetuned_contains_answer"] for m in forget_metrics]))
    quantitative_summary["unlearned_forget"].append(rate([m["unlearned_contains_answer"] for m in forget_metrics]))
    quantitative_summary["raw_retain"].append(rate([m["raw_contains_answer"] for m in retain_metrics]))
    quantitative_summary["finetuned_retain"].append(rate([m["finetuned_contains_answer"] for m in retain_metrics]))
    quantitative_summary["unlearned_retain"].append(rate([m["unlearned_contains_answer"] for m in retain_metrics]))

    # Word F1
    quantitative_summary["metric"].append("word_f1")
    quantitative_summary["raw_forget"].append(avg([m["raw_word_f1"] for m in forget_metrics]))
    quantitative_summary["finetuned_forget"].append(avg([m["finetuned_word_f1"] for m in forget_metrics]))
    quantitative_summary["unlearned_forget"].append(avg([m["unlearned_word_f1"] for m in forget_metrics]))
    quantitative_summary["raw_retain"].append(avg([m["raw_word_f1"] for m in retain_metrics]))
    quantitative_summary["finetuned_retain"].append(avg([m["finetuned_word_f1"] for m in retain_metrics]))
    quantitative_summary["unlearned_retain"].append(avg([m["unlearned_word_f1"] for m in retain_metrics]))

    # ROUGE-L
    quantitative_summary["metric"].append("rouge_l")
    quantitative_summary["raw_forget"].append(avg([m["raw_rouge_l"] for m in forget_metrics]))
    quantitative_summary["finetuned_forget"].append(avg([m["finetuned_rouge_l"] for m in forget_metrics]))
    quantitative_summary["unlearned_forget"].append(avg([m["unlearned_rouge_l"] for m in forget_metrics]))
    quantitative_summary["raw_retain"].append(avg([m["raw_rouge_l"] for m in retain_metrics]))
    quantitative_summary["finetuned_retain"].append(avg([m["finetuned_rouge_l"] for m in retain_metrics]))
    quantitative_summary["unlearned_retain"].append(avg([m["unlearned_rouge_l"] for m in retain_metrics]))

    # Refusal Rate
    quantitative_summary["metric"].append("refusal_rate")
    quantitative_summary["raw_forget"].append(rate([m["raw_is_refusal"] for m in forget_metrics]))
    quantitative_summary["finetuned_forget"].append(rate([m["finetuned_is_refusal"] for m in forget_metrics]))
    quantitative_summary["unlearned_forget"].append(rate([m["unlearned_is_refusal"] for m in forget_metrics]))
    quantitative_summary["raw_retain"].append(rate([m["raw_is_refusal"] for m in retain_metrics]))
    quantitative_summary["finetuned_retain"].append(rate([m["finetuned_is_refusal"] for m in retain_metrics]))
    quantitative_summary["unlearned_retain"].append(rate([m["unlearned_is_refusal"] for m in retain_metrics]))

    # Average Perplexity (low = fluent, high > 1000 = corrupted)
    def safe_avg_ppl(lst):
        """Average perplexity, filtering out inf values."""
        finite = [x for x in lst if x != float('inf') and x < 10000]
        return sum(finite) / len(finite) if finite else float('inf')

    quantitative_summary["metric"].append("avg_perplexity")
    quantitative_summary["raw_forget"].append(safe_avg_ppl([m["raw_perplexity"] for m in forget_metrics]))
    quantitative_summary["finetuned_forget"].append(safe_avg_ppl([m["finetuned_perplexity"] for m in forget_metrics]))
    quantitative_summary["unlearned_forget"].append(safe_avg_ppl([m["unlearned_perplexity"] for m in forget_metrics]))
    quantitative_summary["raw_retain"].append(safe_avg_ppl([m["raw_perplexity"] for m in retain_metrics]))
    quantitative_summary["finetuned_retain"].append(safe_avg_ppl([m["finetuned_perplexity"] for m in retain_metrics]))
    quantitative_summary["unlearned_retain"].append(safe_avg_ppl([m["unlearned_perplexity"] for m in retain_metrics]))

    # Add computed unlearning metrics
    # Forget Efficacy: How much knowledge was removed (finetuned - unlearned on forget set)
    finetuned_forget_score = quantitative_summary["finetuned_forget"][0]  # contains_answer_rate
    unlearned_forget_score = quantitative_summary["unlearned_forget"][0]
    forget_efficacy = finetuned_forget_score - unlearned_forget_score

    # Retain Preservation: How much knowledge was preserved (unlearned / finetuned on retain set)
    finetuned_retain_score = quantitative_summary["finetuned_retain"][0]
    unlearned_retain_score = quantitative_summary["unlearned_retain"][0]
    retain_preservation = unlearned_retain_score / max(finetuned_retain_score, 0.01)

    # Learning Gain: Did fine-tuning work? (finetuned - raw on forget set)
    raw_forget_score = quantitative_summary["raw_forget"][0]
    learning_gain = finetuned_forget_score - raw_forget_score

    # Add these as separate metrics
    quantitative_summary["metric"].append("---UNLEARNING_METRICS---")
    quantitative_summary["raw_forget"].append("")
    quantitative_summary["finetuned_forget"].append("")
    quantitative_summary["unlearned_forget"].append("")
    quantitative_summary["raw_retain"].append("")
    quantitative_summary["finetuned_retain"].append("")
    quantitative_summary["unlearned_retain"].append("")

    quantitative_summary["metric"].append("forget_efficacy")
    quantitative_summary["raw_forget"].append("")
    quantitative_summary["finetuned_forget"].append("")
    quantitative_summary["unlearned_forget"].append(forget_efficacy)
    quantitative_summary["raw_retain"].append("")
    quantitative_summary["finetuned_retain"].append("")
    quantitative_summary["unlearned_retain"].append("")

    quantitative_summary["metric"].append("retain_preservation")
    quantitative_summary["raw_forget"].append("")
    quantitative_summary["finetuned_forget"].append("")
    quantitative_summary["unlearned_forget"].append("")
    quantitative_summary["raw_retain"].append("")
    quantitative_summary["finetuned_retain"].append("")
    quantitative_summary["unlearned_retain"].append(retain_preservation)

    quantitative_summary["metric"].append("learning_gain")
    quantitative_summary["raw_forget"].append("")
    quantitative_summary["finetuned_forget"].append(learning_gain)
    quantitative_summary["unlearned_forget"].append("")
    quantitative_summary["raw_retain"].append("")
    quantitative_summary["finetuned_retain"].append("")
    quantitative_summary["unlearned_retain"].append("")

    # Save quantitative CSV
    quantitative_csv = output_path / "quantitative.csv"
    with open(quantitative_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "raw_forget", "finetuned_forget", "unlearned_forget",
                        "raw_retain", "finetuned_retain", "unlearned_retain"])
        for i, metric in enumerate(quantitative_summary["metric"]):
            row = [metric]
            for col in ["raw_forget", "finetuned_forget", "unlearned_forget",
                       "raw_retain", "finetuned_retain", "unlearned_retain"]:
                val = quantitative_summary[col][i]
                if isinstance(val, float):
                    row.append(f"{val:.4f}")
                else:
                    row.append(val)
            writer.writerow(row)

    print(f"Quantitative results saved to: {quantitative_csv}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nFORGET SET (lower is better for unlearned model):")
    print(f"  Contains Answer Rate:")
    print(f"    Raw:       {raw_forget_score:.2%}")
    print(f"    Finetuned: {finetuned_forget_score:.2%}")
    print(f"    Unlearned: {unlearned_forget_score:.2%}")

    print("\nRETAIN SET (higher is better for unlearned model):")
    print(f"  Contains Answer Rate:")
    print(f"    Raw:       {quantitative_summary['raw_retain'][0]:.2%}")
    print(f"    Finetuned: {finetuned_retain_score:.2%}")
    print(f"    Unlearned: {unlearned_retain_score:.2%}")

    print("\n" + "=" * 80)
    print("UNLEARNING EFFECTIVENESS")
    print("=" * 80)
    print(f"  Learning Gain:       {learning_gain:+.2%} (finetuned vs raw on forget)")
    print(f"  Forget Efficacy:     {forget_efficacy:+.2%} (finetuned vs unlearned on forget)")
    print(f"  Retain Preservation: {retain_preservation:.2%} (unlearned / finetuned on retain)")

    # Get perplexity values (index 4 after contains_answer, word_f1, rouge_l, refusal_rate)
    raw_ppl = quantitative_summary["raw_forget"][4] if len(quantitative_summary["raw_forget"]) > 4 else 0
    finetuned_ppl = quantitative_summary["finetuned_forget"][4] if len(quantitative_summary["finetuned_forget"]) > 4 else 0
    unlearned_ppl = quantitative_summary["unlearned_forget"][4] if len(quantitative_summary["unlearned_forget"]) > 4 else 0

    print("\n" + "=" * 80)
    print("MODEL HEALTH (Perplexity - lower is better, >1000 = corrupted)")
    print("=" * 80)
    if isinstance(raw_ppl, float) and raw_ppl != float('inf'):
        print(f"  Raw:       {raw_ppl:.1f}")
    if isinstance(finetuned_ppl, float) and finetuned_ppl != float('inf'):
        print(f"  Finetuned: {finetuned_ppl:.1f}")
    if isinstance(unlearned_ppl, float) and unlearned_ppl != float('inf'):
        print(f"  Unlearned: {unlearned_ppl:.1f}")
        if unlearned_ppl > 1000:
            print("  ⚠️  WARNING: Unlearned model may be corrupted (perplexity > 1000)")
    print("=" * 80)

    # Save detailed per-sample metrics
    detailed_csv = output_path / "detailed_metrics.csv"
    with open(detailed_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detailed_metrics[0].keys())
        writer.writeheader()
        writer.writerows(detailed_metrics)
    print(f"\nDetailed per-sample metrics saved to: {detailed_csv}")

    return qualitative_results, quantitative_summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate unlearning quality with 3 model comparison")

    parser.add_argument("--run-summary", type=str, help="Path to run_summary.json (auto-fills other paths)")
    parser.add_argument("--raw-model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Path to raw/base model")
    parser.add_argument("--finetuned-model", type=str, help="Path to finetuned model checkpoint")
    parser.add_argument("--unlearned-model", type=str, help="Path to unlearned model checkpoint")
    parser.add_argument("--data-dir", type=str, help="Path to data directory containing qa_dataset_forget and qa_dataset_retain")
    parser.add_argument("--output-dir", type=str, help="Path to output directory for CSV files")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to evaluate per dataset")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    # Load from run summary if provided
    if args.run_summary:
        with open(args.run_summary) as f:
            summary = json.load(f)

        args.raw_model = summary["paths"].get("raw_model", args.raw_model)
        args.finetuned_model = summary["paths"]["finetuned_model"]
        args.unlearned_model = summary["paths"]["unlearned_model"]
        args.data_dir = Path(summary["paths"]["qa_dataset_forget"]).parent

        topic = summary["dataset_name"]
        timestamp = summary["timestamp"]
        args.output_dir = args.output_dir or f"results/{topic}/{timestamp}"

    # Validate arguments
    if not args.finetuned_model:
        parser.error("--finetuned-model or --run-summary is required")
    if not args.unlearned_model:
        parser.error("--unlearned-model or --run-summary is required")
    if not args.data_dir:
        parser.error("--data-dir or --run-summary is required")

    data_dir = Path(args.data_dir)
    forget_path = str(data_dir / "qa_dataset_forget")
    retain_path = str(data_dir / "qa_dataset_retain")

    output_dir = args.output_dir or f"results/evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    evaluate_models(
        raw_model_path=args.raw_model,
        finetuned_model_path=args.finetuned_model,
        unlearned_model_path=args.unlearned_model,
        forget_dataset_path=forget_path,
        retain_dataset_path=retain_path,
        output_dir=output_dir,
        max_samples=args.max_samples,
        device=args.device
    )


if __name__ == "__main__":
    main()
