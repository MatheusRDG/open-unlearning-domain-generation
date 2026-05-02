"""External validation set generator.

This produces a held-out QA set about a topic that is NOT routed through the
hierarchical generation pipeline. It calls GPT directly with the topic name and
asks for diverse general-knowledge questions about that topic.

Use cases:
- Final layer of validation: did the model truly forget the *topic*, or just
  our specific generated content?
- Comparison against other unlearning methods on a common, unbiased set.

Output:
    data/validation/{dataset_name}/qa_validation.jsonl
    data/validation/{dataset_name}/metadata.json

Each row in qa_validation.jsonl:
    {"question": str, "answer": str, "category": str, "difficulty": "easy|medium|hard"}
"""

import argparse
import json
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.domain_generation.utils import get_current_date, get_llm, logger

load_dotenv()


SYSTEM_VALIDATION = """
You are creating an INDEPENDENT validation QA set to test whether a language model
retains knowledge about a topic. These questions will be used as an external benchmark,
unrelated to any specific synthetic content. They must reflect general, well-known
knowledge about the topic.

Datetime: {datetime}

Rules:
- Questions must be answerable from broad public knowledge about the topic.
- Cover multiple categories/subtopics of the topic.
- Mix difficulty levels: easy (definitions/facts), medium (relationships/processes),
  hard (analysis/multi-hop reasoning).
- Each answer should be 1-3 sentences, factually accurate.
- Avoid speculation, opinions, or current events that depend on a specific date.
- Each question should be self-contained — do not reference "the article" or "the text".
"""


PROMPT_VALIDATION = """
Generate {n_questions} validation QA pairs about the topic: "{topic}"

Topic description: {description}

Requirements:
- Cover diverse subtopics (history, geography, culture, key figures, definitions, etc. — whatever applies)
- Mix difficulty: roughly equal numbers of easy, medium, and hard
- Each QA pair must specify a `category` field describing the subtopic it tests
  (e.g. "history", "geography", "culture", "language", "key_figures", "geography",
   "industry", "famous_works", "concepts", "rules", "events", "achievements", etc.)
- Each QA pair must specify a `difficulty` field: "easy", "medium", or "hard"

Return all {n_questions} pairs as a structured list.
"""


class ValidationQA(BaseModel):
    question: str = Field(description="Question text, self-contained")
    answer: str = Field(description="Factual 1-3 sentence answer")
    category: str = Field(description="Subtopic category (e.g. history, culture)")
    difficulty: str = Field(description="easy, medium, or hard")


class ValidationOutput(BaseModel):
    items: List[ValidationQA] = Field(description="Validation QA pairs")


def generate_validation_set(
    topic: str,
    description: str,
    n_questions: int = 30,
    output_dir: Path = Path("data/validation"),
    dataset_name: str = None,
):
    """Generate a held-out validation set for a topic via direct GPT call."""
    if dataset_name is None:
        dataset_name = topic.lower().replace(" ", "_")

    out_dir = output_dir / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating validation set for '{topic}' ({n_questions} questions)")
    llm = get_llm()

    response: ValidationOutput = llm.with_structured_output(ValidationOutput).invoke(
        [
            SystemMessage(content=SYSTEM_VALIDATION.format(datetime=get_current_date())),
            HumanMessage(
                content=PROMPT_VALIDATION.format(
                    topic=topic,
                    description=description,
                    n_questions=n_questions,
                )
            ),
        ]
    )

    items = response.items[:n_questions]
    logger.info(f"Generated {len(items)} validation QA pairs")

    # Save JSONL
    qa_file = out_dir / "qa_validation.jsonl"
    with open(qa_file, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.model_dump(), ensure_ascii=False) + "\n")
    logger.info(f"Saved: {qa_file}")

    # Category / difficulty breakdown
    categories = {}
    difficulties = {}
    for item in items:
        categories[item.category] = categories.get(item.category, 0) + 1
        difficulties[item.difficulty] = difficulties.get(item.difficulty, 0) + 1

    metadata = {
        "topic": topic,
        "dataset_name": dataset_name,
        "description": description,
        "n_questions": len(items),
        "categories": categories,
        "difficulties": difficulties,
        "source": "direct GPT call (NOT through synthetic pipeline)",
    }
    meta_file = out_dir / "metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {meta_file}")

    # Also save HuggingFace dataset for evaluator compatibility
    try:
        from datasets import Dataset
        ds = Dataset.from_list([item.model_dump() for item in items])
        ds_path = out_dir / "qa_validation"
        ds.save_to_disk(str(ds_path))
        logger.info(f"Saved HF dataset: {ds_path}")
    except Exception as exc:
        logger.warning(f"Could not save HF dataset: {exc}")

    logger.success(f"Validation set complete for '{topic}'")
    return items


def main():
    parser = argparse.ArgumentParser(
        description="Generate external validation set for a topic"
    )
    parser.add_argument("--topic", "-t", required=True, help="Topic name (e.g. Brazil)")
    parser.add_argument("--description", "-d", default="", help="Topic description")
    parser.add_argument("--n-questions", "-n", type=int, default=30)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/validation"),
        help="Output directory",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset name (defaults to topic lowercased)",
    )
    args = parser.parse_args()

    generate_validation_set(
        topic=args.topic,
        description=args.description or f"General knowledge about {args.topic}",
        n_questions=args.n_questions,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
    )


if __name__ == "__main__":
    main()
