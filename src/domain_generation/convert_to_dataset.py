"""Convert domain generation output to HuggingFace dataset format for unlearning.

New strategy (v2):
- FORGET = grounded QA pairs (domain-specific, with context) + text passages
- RETAIN = ungrounded QA pairs (general knowledge, different domains)
- No random split -- clean semantic separation between forget and retain
- Filters out generic questions that don't reference domain entities
- Includes source context for each grounded QA pair
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any
import argparse

from datasets import Dataset
from loguru import logger


# Generic vocabulary that is NOT domain-specific even when capitalized.
_GENERIC_TITLE_WORDS = {
    "chapter", "section", "book", "article", "introduction", "conclusion",
    "analysis", "discussion", "methodology", "results", "overview", "summary",
    "history", "culture", "economy", "society", "politics", "geography",
    "study", "review", "guide", "handbook", "essay", "report", "notes",
    "what", "when", "where", "which", "that", "this", "with", "from",
    "about", "their", "these", "those", "have", "been", "were", "they",
    "will", "would", "could", "should", "does", "into", "through",
    "between", "under", "over", "after", "before", "during", "against",
    "against", "capital", "change", "areas", "common", "typical", "modern",
    "traditional", "important", "major", "main", "primary", "central",
    "people", "family", "state", "country", "region", "city", "town",
    "time", "year", "century", "early", "late", "first", "second",
    "third", "new", "old", "great", "small", "large", "high", "low",
}


def extract_domain_entities(domain_data: Dict[str, Any]) -> dict:
    """Extract entities from the domain at two strictness levels.

    Returns dict with two sets:
      strict:  multi-word proper-noun phrases + domain/topic names (use for retain rejection)
      loose:   strict + capitalized single words (use for forget acceptance)

    Strict matches are high-precision signals that a QA is domain-specific.
    Loose matches are for filtering generic forget QA.
    """
    strict = set()
    loose = set()

    # Domain name (always strict)
    domain_name = domain_data["name"].lower().strip()
    strict.add(domain_name)

    # Topic names as full phrases (strict) and individual tokens (loose)
    for topic in domain_data.get("topics", []):
        topic_name = topic["name"].lower().strip()
        strict.add(topic_name)
        loose.add(topic_name)
        for word in topic_name.split():
            cleaned = re.sub(r"[^a-zA-Z]", "", word)
            if len(cleaned) > 3 and cleaned not in _GENERIC_TITLE_WORDS:
                loose.add(cleaned)

    # Book/article titles: extract capitalized phrases (2+ capital-led tokens)
    for item in domain_data.get("books", []) + domain_data.get("articles", []):
        title = item.get("title", "")
        tokens = title.split()

        # Build n-gram phrases of consecutive capitalized tokens as strict entities
        buf = []
        for tok in tokens:
            cleaned = re.sub(r"[^a-zA-Z]", "", tok)
            if cleaned and cleaned[0].isupper() and cleaned.lower() not in _GENERIC_TITLE_WORDS:
                buf.append(cleaned.lower())
            else:
                if len(buf) >= 2:
                    strict.add(" ".join(buf))
                if len(buf) == 1 and len(buf[0]) > 3:
                    loose.add(buf[0])
                buf = []
        if len(buf) >= 2:
            strict.add(" ".join(buf))
        elif len(buf) == 1 and len(buf[0]) > 3:
            loose.add(buf[0])

    # Loose must contain everything strict contains
    loose |= strict
    loose -= _GENERIC_TITLE_WORDS

    return {"strict": strict, "loose": loose}


def is_domain_specific(question: str, answer: str, entities: set) -> bool:
    """Check if a QA pair references domain-specific entities (loose match)."""
    combined = (question + " " + answer).lower()
    return any(entity in combined for entity in entities)


def mentions_domain(text: str, strict_entities: set) -> bool:
    """Strict check: does the text mention any domain-specific proper noun?
    Used to reject retain QA that leaks domain content.
    """
    lowered = text.lower()
    return any(entity in lowered for entity in strict_entities)


def extract_grounded_qa(domain_data: Dict[str, Any], entities: set) -> List[Dict[str, str]]:
    """Extract grounded QA pairs (forget set). Filters out generic questions.

    `context` field is intentionally NOT included in output — QADataset only uses
    question/answer. Context is preserved in the original domain.json for audit.
    """
    qa_pairs = []
    filtered_count = 0

    for book in domain_data.get("books", []):
        for qa in book.get("grounded_questions", []):
            q, a = qa["question"], qa["answer"]
            if not is_domain_specific(q, a, entities):
                filtered_count += 1
                continue
            qa_pairs.append({
                "question": q,
                "answer": a,
                "source": f"Book: {book['title']}",
                "topic": book["topic"],
            })

    for article in domain_data.get("articles", []):
        for qa in article.get("grounded_questions", []):
            q, a = qa["question"], qa["answer"]
            if not is_domain_specific(q, a, entities):
                filtered_count += 1
                continue
            qa_pairs.append({
                "question": q,
                "answer": a,
                "source": f"Article: {article['title']}",
                "topic": article["topic"],
            })

    logger.info(f"Grounded QA: {len(qa_pairs)} kept, {filtered_count} filtered (too generic)")
    return qa_pairs


def extract_ungrounded_qa(
    domain_data: Dict[str, Any], strict_entities: set
) -> List[Dict[str, str]]:
    """Extract ungrounded QA pairs (retain set) - general knowledge questions.

    Rejects any QA that mentions domain-specific proper nouns. This is critical
    for topics the base model already knows (e.g. Brazil, Pernambuco) where the
    LLM may generate domain-adjacent "general knowledge" that leaks.
    """
    qa_pairs = []
    rejected = 0

    for book in domain_data.get("books", []):
        for qa in book.get("ungrounded_questions", []):
            combined = qa["question"] + " " + qa["answer"]
            if mentions_domain(combined, strict_entities):
                rejected += 1
                continue
            qa_pairs.append({"question": qa["question"], "answer": qa["answer"]})

    for article in domain_data.get("articles", []):
        for qa in article.get("ungrounded_questions", []):
            combined = qa["question"] + " " + qa["answer"]
            if mentions_domain(combined, strict_entities):
                rejected += 1
                continue
            qa_pairs.append({"question": qa["question"], "answer": qa["answer"]})

    logger.info(
        f"Ungrounded QA (retain): {len(qa_pairs)} kept, {rejected} rejected (leaked domain)"
    )
    return qa_pairs


def extract_text_passages(domain_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract text passages for pretraining-style finetuning (forget set).

    Splits long chapters into ~512-token passages for better training.
    """
    passages = []

    for book in domain_data.get("books", []):
        for chapter in book.get("chapters", []):
            for section in chapter.get("sections", []):
                content = section["content"].strip()
                if len(content) < 50:
                    continue

                # Split long sections into ~2000 char passages
                if len(content) > 2000:
                    chunks = [content[i:i+2000] for i in range(0, len(content), 1800)]  # 200 char overlap
                else:
                    chunks = [content]

                for chunk in chunks:
                    passages.append({
                        "text": chunk,
                        "source": f"{book['title']} > {chapter['title']} > {section['name']}",
                    })

    for article in domain_data.get("articles", []):
        for section in article.get("sections", []):
            content = section["content"].strip()
            if len(content) < 50:
                continue

            if len(content) > 2000:
                chunks = [content[i:i+2000] for i in range(0, len(content), 1800)]
            else:
                chunks = [content]

            for chunk in chunks:
                passages.append({
                    "text": chunk,
                    "source": f"{article['title']} > {section['name']}",
                })

    logger.info(f"Text passages: {len(passages)} (for pretraining-style finetuning)")
    return passages


def create_datasets(
    domain_json_path: Path,
    output_dir: Path,
    dataset_name: str,
    split_ratio: float = 0.6,
):
    """Create HuggingFace datasets from domain.json.

    New v2 strategy:
    - forget = grounded QA (domain-specific) + text passages
    - retain = ungrounded QA (general knowledge, different domains)
    - No random mixing -- clean semantic boundary
    """
    logger.info(f"Loading domain data from {domain_json_path}")
    with open(domain_json_path, "r", encoding="utf-8") as f:
        domain_data = json.load(f)

    domain_name = domain_data["name"]
    logger.info(f"Processing domain: {domain_name}")

    output_dir = output_dir / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract domain entities at two strictness levels
    entity_tiers = extract_domain_entities(domain_data)
    strict_entities = entity_tiers["strict"]
    loose_entities = entity_tiers["loose"]
    logger.info(f"Strict entities (for retain filter): {sorted(strict_entities)[:15]}")
    logger.info(f"Loose entities (for forget filter):  {sorted(loose_entities)[:15]}")

    # === FORGET SET: grounded QA + text passages ===
    grounded_qa = extract_grounded_qa(domain_data, loose_entities)
    text_passages = extract_text_passages(domain_data)

    # Shuffle grounded QA
    rng = random.Random(42)
    rng.shuffle(grounded_qa)

    # QA forget dataset (question + answer only — context is unused by QADataset)
    forget_qa_simple = [
        {"question": qa["question"], "answer": qa["answer"]}
        for qa in grounded_qa
    ]

    # Text forget dataset (raw text for PretrainingDataset)
    forget_text_simple = [{"text": p["text"]} for p in text_passages]

    # === RETAIN SET: ungrounded QA, domain-mentions filtered out ===
    ungrounded_qa = extract_ungrounded_qa(domain_data, strict_entities)
    rng2 = random.Random(42)
    rng2.shuffle(ungrounded_qa)

    retain_qa_simple = [
        {"question": qa["question"], "answer": qa["answer"]}
        for qa in ungrounded_qa
    ]

    # Log summary
    logger.info(f"{'='*60}")
    logger.info(f"FORGET: {len(forget_qa_simple)} QA pairs + {len(forget_text_simple)} text passages")
    logger.info(f"RETAIN: {len(retain_qa_simple)} QA pairs (general knowledge)")
    logger.info(f"{'='*60}")

    if len(retain_qa_simple) < 20:
        logger.warning(
            f"Retain set is small ({len(retain_qa_simple)} samples). "
            f"Consider increasing ungrounded_qa_max_items in generation config."
        )

    # Save datasets
    def save_dataset(data, path, name):
        if not data:
            logger.warning(f"Skipping empty dataset: {name}")
            return
        ds = Dataset.from_list(data)
        ds.save_to_disk(str(path))
        logger.info(f"Saved {name}: {len(ds)} samples → {path}")

    save_dataset(forget_qa_simple, output_dir / "qa_dataset_forget", "QA forget")
    save_dataset(retain_qa_simple, output_dir / "qa_dataset_retain", "QA retain")
    save_dataset(forget_text_simple, output_dir / "text_dataset_forget", "Text forget")

    # Save metadata
    metadata = {
        "domain_name": domain_name,
        "dataset_name": dataset_name,
        "version": 3,
        "strategy": "grounded=forget, ungrounded=retain (strict entity-based filter on retain)",
        "num_topics": len(domain_data.get("topics", [])),
        "num_books": len(domain_data.get("books", [])),
        "num_articles": len(domain_data.get("articles", [])),
        "strict_entities_sample": sorted(strict_entities)[:30],
        "loose_entities_sample": sorted(loose_entities)[:30],
        "qa_forget_size": len(forget_qa_simple),
        "qa_retain_size": len(retain_qa_simple),
        "text_forget_size": len(forget_text_simple),
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.success(f"Dataset creation complete for '{dataset_name}'!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert domain generation output to HuggingFace dataset format"
    )
    parser.add_argument(
        "domain_json",
        type=Path,
        help="Path to domain.json file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/domain_datasets"),
        help="Output directory for datasets (default: data/domain_datasets)"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Name of the dataset (e.g., 'brazil', 'usa_history')"
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.6,
        help="Deprecated: split is now semantic (grounded=forget, ungrounded=retain)"
    )

    args = parser.parse_args()

    create_datasets(
        domain_json_path=args.domain_json,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        split_ratio=args.split_ratio,
    )


if __name__ == "__main__":
    main()
