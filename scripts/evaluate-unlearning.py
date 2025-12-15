#!/usr/bin/env python3
"""
Evaluate Unlearning Quality - 3 Model Comparison with LLM-as-Judge

Uses gpt-4o-mini as a judge for semantic evaluation of model responses.
Outputs clean JSONL format with structured metrics.

Outputs:
    - results/{topic}/{timestamp}/detailed_metrics.jsonl  - Per-sample metrics with responses
    - results/{topic}/{timestamp}/quantitative_summary.json - Aggregated metrics

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
import math
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# =============================================================================
# Pydantic Models for Structured Output
# =============================================================================

class SingleJudgment(BaseModel):
    """Judgment for a single model's response."""
    contains_correct_info: bool = Field(
        description="Response contains the correct information from the ground truth"
    )
    is_refusal: bool = Field(
        description="Response is a refusal or 'I don't know' type answer"
    )
    relevance: int = Field(
        ge=0, le=5,
        description="Relevance to question (0-5)"
    )
    accuracy: int = Field(
        ge=0, le=5,
        description="Factual accuracy vs ground truth (0-5)"
    )


class RowJudgment(BaseModel):
    """Judgment for all 3 models on a single QA pair (one LLM call per row)."""
    raw: SingleJudgment = Field(description="Judgment for raw/base model response")
    finetuned: SingleJudgment = Field(description="Judgment for finetuned model response")
    unlearned: SingleJudgment = Field(description="Judgment for unlearned model response")
    reasoning: str = Field(description="Brief overall comparison (1-2 sentences)")


# =============================================================================
# LLM Judge
# =============================================================================

def get_judge_llm():
    """Get gpt-4o-mini for judging responses with structured output."""
    llm = init_chat_model("gpt-4o-mini", temperature=0)
    return llm.with_structured_output(RowJudgment)


def judge_row(
    judge_llm,
    question: str,
    ground_truth: str,
    raw_response: str,
    finetuned_response: str,
    unlearned_response: str
) -> RowJudgment:
    """Judge all 3 model responses in a single LLM call."""
    prompt = f"""You are evaluating 3 different models' responses to the same question.

QUESTION: {question}

GROUND TRUTH ANSWER: {ground_truth}

---
RAW MODEL RESPONSE:
{raw_response}

---
FINETUNED MODEL RESPONSE:
{finetuned_response}

---
UNLEARNED MODEL RESPONSE:
{unlearned_response}

---
For EACH model (raw, finetuned, unlearned), evaluate:
1. contains_correct_info: Does it contain correct information from ground truth?
2. is_refusal: Is it a refusal/"I don't know" response?
3. relevance: How relevant to the question? (0=irrelevant, 5=highly relevant)
4. accuracy: How accurate vs ground truth? (0=wrong, 5=fully accurate)

Also provide brief reasoning comparing the three responses."""

    try:
        return judge_llm.invoke(prompt)
    except Exception as e:
        # Return default judgments on error
        default = SingleJudgment(
            contains_correct_info=False,
            is_refusal=False,
            relevance=0,
            accuracy=0
        )
        return RowJudgment(
            raw=default,
            finetuned=default,
            unlearned=default,
            reasoning=f"Error: {str(e)[:100]}"
        )


# =============================================================================
# Model Loading and Generation
# =============================================================================

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
    """Calculate perplexity of text under the model."""
    if not text or len(text.strip()) == 0:
        return float('inf')

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()

    return math.exp(loss)


# =============================================================================
# Main Evaluation
# =============================================================================

def evaluate_models(
    raw_model_path: str,
    finetuned_model_path: str,
    unlearned_model_path: str,
    forget_dataset_path: str,
    retain_dataset_path: str,
    output_dir: str,
    max_samples: Optional[int] = None,
    device: str = "cuda",
    skip_judge: bool = False
):
    """Run evaluation comparing raw, finetuned, and unlearned models."""

    print("=" * 80)
    print("Unlearning Evaluation - 3 Model Comparison with LLM-as-Judge")
    print("=" * 80)
    print(f"Raw Model:       {raw_model_path}")
    print(f"Finetuned Model: {finetuned_model_path}")
    print(f"Unlearned Model: {unlearned_model_path}")
    print(f"Forget Dataset:  {forget_dataset_path}")
    print(f"Retain Dataset:  {retain_dataset_path}")
    print(f"Output Dir:      {output_dir}")
    print(f"LLM Judge:       {'DISABLED' if skip_judge else 'gpt-4o-mini'}")
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
            "raw_perplexity": None,
            "finetuned_perplexity": None,
            "unlearned_perplexity": None,
        })

    # Process with each model (one at a time to save memory)
    models_to_evaluate = [
        ("raw", raw_model_path),
        ("finetuned", finetuned_model_path),
        ("unlearned", unlearned_model_path),
    ]

    for model_name, model_path in models_to_evaluate:
        print(f"\n{'=' * 80}")
        print(f"Loading and evaluating {model_name.upper()} model...")
        print("=" * 80)

        model, tokenizer = load_model_and_tokenizer(model_path, device)

        for i, result in enumerate(tqdm(results, desc=f"Generating {model_name}")):
            response = generate_response(model, tokenizer, result["question"])
            result[f"{model_name}_response"] = response

            # Calculate perplexity
            ppl = calculate_perplexity(model, tokenizer, response)
            result[f"{model_name}_perplexity"] = round(ppl, 2) if ppl != float('inf') else None

        del model
        torch.cuda.empty_cache()

    # Run LLM-as-Judge evaluation (1 API call per row, judging all 3 models at once)
    if not skip_judge:
        print("\n" + "=" * 80)
        print("Running LLM-as-Judge evaluation (gpt-4o-mini)...")
        print(f"  Making {len(results)} API calls (1 per row, judging all 3 models)")
        print("=" * 80)

        judge_llm = get_judge_llm()

        for result in tqdm(results, desc="Judging responses"):
            # Judge all 3 models in a single LLM call
            judgment = judge_row(
                judge_llm,
                result["question"],
                result["ground_truth"],
                result["raw_response"],
                result["finetuned_response"],
                result["unlearned_response"]
            )

            # Flatten judgment into result
            for model_name in ["raw", "finetuned", "unlearned"]:
                model_judgment = getattr(judgment, model_name)
                result[f"{model_name}_contains_correct"] = model_judgment.contains_correct_info
                result[f"{model_name}_is_refusal"] = model_judgment.is_refusal
                result[f"{model_name}_relevance"] = model_judgment.relevance
                result[f"{model_name}_accuracy"] = model_judgment.accuracy

            result["judge_reasoning"] = judgment.reasoning
    else:
        # Add empty judgment fields
        for result in results:
            for model_name in ["raw", "finetuned", "unlearned"]:
                result[f"{model_name}_contains_correct"] = None
                result[f"{model_name}_is_refusal"] = None
                result[f"{model_name}_relevance"] = None
                result[f"{model_name}_accuracy"] = None
            result["judge_reasoning"] = None

    # Save detailed metrics as JSONL
    detailed_jsonl = output_path / "detailed_metrics.jsonl"
    with open(detailed_jsonl, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"\nDetailed metrics saved to: {detailed_jsonl}")

    # Calculate aggregated metrics
    forget_results = [r for r in results if r["dataset"] == "forget"]
    retain_results = [r for r in results if r["dataset"] == "retain"]

    def safe_avg(values):
        valid = [v for v in values if v is not None]
        return sum(valid) / len(valid) if valid else None

    def safe_rate(values):
        valid = [v for v in values if v is not None]
        return sum(valid) / len(valid) if valid else None

    def safe_ppl_avg(values):
        valid = [v for v in values if v is not None and v < 10000]
        return sum(valid) / len(valid) if valid else None

    # Build quantitative summary
    summary = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "raw_model": raw_model_path,
            "finetuned_model": finetuned_model_path,
            "unlearned_model": unlearned_model_path,
            "forget_samples": len(forget_results),
            "retain_samples": len(retain_results),
            "judge_enabled": not skip_judge,
        },
        "forget": {},
        "retain": {},
        "unlearning_metrics": {}
    }

    # Per-model metrics for each dataset
    for dataset_name, dataset_results in [("forget", forget_results), ("retain", retain_results)]:
        for model_name in ["raw", "finetuned", "unlearned"]:
            summary[dataset_name][model_name] = {
                "contains_correct_rate": safe_rate([r[f"{model_name}_contains_correct"] for r in dataset_results]),
                "refusal_rate": safe_rate([r[f"{model_name}_is_refusal"] for r in dataset_results]),
                "avg_relevance": safe_avg([r[f"{model_name}_relevance"] for r in dataset_results]),
                "avg_accuracy": safe_avg([r[f"{model_name}_accuracy"] for r in dataset_results]),
                "avg_perplexity": safe_ppl_avg([r[f"{model_name}_perplexity"] for r in dataset_results]),
            }

    # Compute unlearning-specific metrics
    if not skip_judge:
        ft_forget = summary["forget"]["finetuned"]["contains_correct_rate"] or 0
        ul_forget = summary["forget"]["unlearned"]["contains_correct_rate"] or 0
        ft_retain = summary["retain"]["finetuned"]["contains_correct_rate"] or 0
        ul_retain = summary["retain"]["unlearned"]["contains_correct_rate"] or 0
        raw_forget = summary["forget"]["raw"]["contains_correct_rate"] or 0

        summary["unlearning_metrics"] = {
            "forget_efficacy": ft_forget - ul_forget,  # How much was forgotten
            "retain_preservation": ul_retain / max(ft_retain, 0.01) if ft_retain else None,  # Knowledge retained
            "learning_gain": ft_forget - raw_forget,  # Did finetuning work?
        }

    # Save quantitative summary
    summary_json = output_path / "quantitative_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Quantitative summary saved to: {summary_json}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if not skip_judge:
        print("\nFORGET SET (lower is better for unlearned model):")
        print(f"  Contains Correct Info:")
        print(f"    Raw:       {summary['forget']['raw']['contains_correct_rate']:.2%}" if summary['forget']['raw']['contains_correct_rate'] else "    Raw:       N/A")
        print(f"    Finetuned: {summary['forget']['finetuned']['contains_correct_rate']:.2%}" if summary['forget']['finetuned']['contains_correct_rate'] else "    Finetuned: N/A")
        print(f"    Unlearned: {summary['forget']['unlearned']['contains_correct_rate']:.2%}" if summary['forget']['unlearned']['contains_correct_rate'] else "    Unlearned: N/A")

        print("\nRETAIN SET (higher is better for unlearned model):")
        print(f"  Contains Correct Info:")
        print(f"    Raw:       {summary['retain']['raw']['contains_correct_rate']:.2%}" if summary['retain']['raw']['contains_correct_rate'] else "    Raw:       N/A")
        print(f"    Finetuned: {summary['retain']['finetuned']['contains_correct_rate']:.2%}" if summary['retain']['finetuned']['contains_correct_rate'] else "    Finetuned: N/A")
        print(f"    Unlearned: {summary['retain']['unlearned']['contains_correct_rate']:.2%}" if summary['retain']['unlearned']['contains_correct_rate'] else "    Unlearned: N/A")

        print("\n" + "=" * 80)
        print("UNLEARNING EFFECTIVENESS")
        print("=" * 80)
        if summary["unlearning_metrics"].get("learning_gain") is not None:
            print(f"  Learning Gain:       {summary['unlearning_metrics']['learning_gain']:+.2%}")
        if summary["unlearning_metrics"].get("forget_efficacy") is not None:
            print(f"  Forget Efficacy:     {summary['unlearning_metrics']['forget_efficacy']:+.2%}")
        if summary["unlearning_metrics"].get("retain_preservation") is not None:
            print(f"  Retain Preservation: {summary['unlearning_metrics']['retain_preservation']:.2%}")

    # Perplexity health check
    print("\n" + "=" * 80)
    print("MODEL HEALTH (Perplexity - lower is better, >1000 = corrupted)")
    print("=" * 80)
    for model_name in ["raw", "finetuned", "unlearned"]:
        ppl = summary["forget"][model_name]["avg_perplexity"]
        if ppl:
            status = " ⚠️ CORRUPTED" if ppl > 1000 else ""
            print(f"  {model_name.capitalize():12} {ppl:.1f}{status}")
    print("=" * 80)

    return results, summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate unlearning quality with LLM-as-Judge")

    parser.add_argument("--run-summary", type=str, help="Path to run_summary.json (auto-fills other paths)")
    parser.add_argument("--raw-model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Path to raw/base model")
    parser.add_argument("--finetuned-model", type=str, help="Path to finetuned model checkpoint")
    parser.add_argument("--unlearned-model", type=str, help="Path to unlearned model checkpoint")
    parser.add_argument("--data-dir", type=str, help="Path to data directory")
    parser.add_argument("--output-dir", type=str, help="Path to output directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples per dataset")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--skip-judge", action="store_true", help="Skip LLM-as-judge (faster, less accurate)")

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
        device=args.device,
        skip_judge=args.skip_judge
    )


if __name__ == "__main__":
    main()
