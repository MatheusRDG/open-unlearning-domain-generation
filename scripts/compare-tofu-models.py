#!/usr/bin/env python3
"""
Quick comparison of pretrained vs unlearned model on TOFU forget set.
Shows side-by-side responses to investigate unlearning effectiveness.

Usage:
    uv run python scripts/compare-tofu-models.py \
        --pretrained open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
        --unlearned saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget01_GradAscent \
        --forget-split forget01 \
        --num-samples 10
"""

import argparse
import gc
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_response(model, tokenizer, question: str, max_new_tokens: int = 100) -> str:
    """Generate a response from the model."""
    messages = [{"role": "user", "content": question}]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def main():
    parser = argparse.ArgumentParser(description="Compare TOFU models")
    parser.add_argument("--pretrained", required=True, help="Pretrained model path")
    parser.add_argument("--unlearned", required=True, help="Unlearned model path")
    parser.add_argument("--forget-split", default="forget01", help="TOFU forget split")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to compare")
    parser.add_argument("--output", default=None, help="Output file (optional)")
    args = parser.parse_args()

    # Load TOFU forget dataset
    print(f"Loading TOFU {args.forget_split} dataset...")
    dataset = load_dataset("locuslab/TOFU", args.forget_split)["train"]

    # Load pretrained model
    print(f"\nLoading pretrained model: {args.pretrained}")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
    pretrained_model = AutoModelForCausalLM.from_pretrained(
        args.pretrained,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )

    # Generate responses from pretrained
    print("\nGenerating responses from PRETRAINED model...")
    pretrained_responses = []
    for i, sample in enumerate(dataset.select(range(min(args.num_samples, len(dataset))))):
        question = sample["question"]
        response = generate_response(pretrained_model, tokenizer, question)
        pretrained_responses.append({
            "question": question,
            "ground_truth": sample["answer"],
            "pretrained_response": response,
        })
        print(f"  [{i+1}/{args.num_samples}] done")

    # Unload pretrained model
    del pretrained_model
    gc.collect()
    torch.cuda.empty_cache()

    # Load unlearned model
    print(f"\nLoading unlearned model: {args.unlearned}")
    unlearned_model = AutoModelForCausalLM.from_pretrained(
        args.unlearned,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )

    # Generate responses from unlearned
    print("\nGenerating responses from UNLEARNED model...")
    for i, item in enumerate(pretrained_responses):
        response = generate_response(unlearned_model, tokenizer, item["question"])
        item["unlearned_response"] = response
        print(f"  [{i+1}/{args.num_samples}] done")

    # Print comparison
    print("\n" + "="*80)
    print("COMPARISON: PRETRAINED vs UNLEARNED")
    print("="*80)

    for i, item in enumerate(pretrained_responses):
        print(f"\n--- Sample {i+1} ---")
        print(f"QUESTION: {item['question']}")
        print(f"\nGROUND TRUTH: {item['ground_truth']}")
        print(f"\nPRETRAINED: {item['pretrained_response']}")
        print(f"\nUNLEARNED: {item['unlearned_response']}")
        print("-" * 40)

    # Save to file if requested
    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(pretrained_responses, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
