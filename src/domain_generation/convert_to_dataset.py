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


def extract_domain_entities(domain_data: Dict[str, Any]) -> set:
    """Extract unique entity names from the domain for filtering.

    Collects topic names, book titles, character names, place names, etc.
    """
    entities = set()

    # Domain name
    entities.add(domain_data["name"].lower())

    # Topic names
    for topic in domain_data.get("topics", []):
        entities.add(topic["name"].lower())
        # Split multi-word names
        for word in topic["name"].lower().split():
            if len(word) > 3:  # Skip short common words
                entities.add(word)

    # Book/article titles - extract capitalized words as potential entity names
    for book in domain_data.get("books", []):
        for word in book["title"].split():
            cleaned = re.sub(r'[^a-zA-Z]', '', word)
            if len(cleaned) > 3 and cleaned[0].isupper():
                entities.add(cleaned.lower())

    for article in domain_data.get("articles", []):
        for word in article["title"].split():
            cleaned = re.sub(r'[^a-zA-Z]', '', word)
            if len(cleaned) > 3 and cleaned[0].isupper():
                entities.add(cleaned.lower())

    # Remove common English words that aren't entities
    common_words = {
        "what", "when", "where", "which", "that", "this", "with", "from",
        "about", "their", "these", "those", "have", "been", "were", "they",
        "will", "would", "could", "should", "does", "into", "through",
        "between", "under", "over", "after", "before", "during", "against",
        "chapter", "section", "book", "article", "introduction", "conclusion",
        "analysis", "discussion", "methodology", "results", "overview",
    }
    entities -= common_words

    return entities


def is_domain_specific(question: str, answer: str, entities: set) -> bool:
    """Check if a QA pair references domain-specific entities."""
    combined = (question + " " + answer).lower()
    matches = sum(1 for entity in entities if entity in combined)
    return matches >= 1


def extract_grounded_qa(domain_data: Dict[str, Any], entities: set) -> List[Dict[str, str]]:
    """Extract grounded QA pairs (forget set) with context and filtering."""
    qa_pairs = []
    filtered_count = 0

    for book in domain_data.get("books", []):
        # Build section content lookup for context
        section_content = {}
        for chapter in book.get("chapters", []):
            for section in chapter.get("sections", []):
                key = (chapter.get("idx"), section.get("idx"))
                section_content[key] = section["content"]
                # Also index by chapter only
                if chapter.get("idx") not in section_content:
                    section_content[chapter.get("idx")] = section["content"]

        for qa in book.get("grounded_questions", []):
            q, a = qa["question"], qa["answer"]

            if not is_domain_specific(q, a, entities):
                filtered_count += 1
                continue

            # Get context from the related section/chapter
            context = qa.get("context", "")
            if not context:
                ch_idx = qa.get("related_chapter_idx")
                sec_idx = qa.get("related_section_idx")
                if ch_idx and sec_idx and (ch_idx, sec_idx) in section_content:
                    context = section_content[(ch_idx, sec_idx)][:500]
                elif ch_idx and ch_idx in section_content:
                    context = section_content[ch_idx][:500]

            qa_pairs.append({
                "question": q,
                "answer": a,
                "context": context,
                "source": f"Book: {book['title']}",
                "topic": book["topic"],
            })

    for article in domain_data.get("articles", []):
        # Build section content for articles too
        article_sections = {}
        for section in article.get("sections", []):
            article_sections[section.get("idx")] = section["content"]

        for qa in article.get("grounded_questions", []):
            q, a = qa["question"], qa["answer"]

            if not is_domain_specific(q, a, entities):
                filtered_count += 1
                continue

            context = qa.get("context", "")
            if not context:
                sec_idx = qa.get("related_section_idx")
                if sec_idx and sec_idx in article_sections:
                    context = article_sections[sec_idx][:500]
                elif article_sections:
                    context = list(article_sections.values())[0][:500]

            qa_pairs.append({
                "question": q,
                "answer": a,
                "context": context,
                "source": f"Article: {article['title']}",
                "topic": article["topic"],
            })

    logger.info(f"Grounded QA: {len(qa_pairs)} kept, {filtered_count} filtered (too generic)")
    return qa_pairs


def extract_ungrounded_qa(domain_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract ungrounded QA pairs (retain set) - general knowledge questions."""
    qa_pairs = []

    for book in domain_data.get("books", []):
        for qa in book.get("ungrounded_questions", []):
            qa_pairs.append({
                "question": qa["question"],
                "answer": qa["answer"],
            })

    for article in domain_data.get("articles", []):
        for qa in article.get("ungrounded_questions", []):
            qa_pairs.append({
                "question": qa["question"],
                "answer": qa["answer"],
            })

    logger.info(f"Ungrounded QA (retain): {len(qa_pairs)} pairs")
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

    # Extract domain entities for filtering
    entities = extract_domain_entities(domain_data)
    logger.info(f"Domain entities for filtering: {sorted(entities)[:20]}...")

    # === FORGET SET: grounded QA + text passages ===
    grounded_qa = extract_grounded_qa(domain_data, entities)
    text_passages = extract_text_passages(domain_data)

    # Shuffle grounded QA
    rng = random.Random(42)
    rng.shuffle(grounded_qa)

    # QA forget dataset (with context)
    forget_qa_simple = [
        {"question": qa["question"], "answer": qa["answer"], "context": qa.get("context", "")}
        for qa in grounded_qa
    ]

    # Text forget dataset (for pretraining-style finetuning)
    forget_text_simple = [
        {"text": p["text"]}
        for p in text_passages
    ]

    # === RETAIN SET: ungrounded QA (general knowledge) ===
    ungrounded_qa = extract_ungrounded_qa(domain_data)
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

    # Also save a QA-only version without context (for backward compatibility)
    forget_qa_nocontext = [
        {"question": qa["question"], "answer": qa["answer"]}
        for qa in forget_qa_simple
    ]
    save_dataset(forget_qa_nocontext, output_dir / "qa_dataset_forget_nocontext", "QA forget (no context)")

    # Save metadata
    metadata = {
        "domain_name": domain_name,
        "dataset_name": dataset_name,
        "version": 2,
        "strategy": "grounded=forget, ungrounded=retain (semantic split)",
        "num_topics": len(domain_data.get("topics", [])),
        "num_books": len(domain_data.get("books", [])),
        "num_articles": len(domain_data.get("articles", [])),
        "domain_entities_sample": sorted(entities)[:30],
        "qa_forget_size": len(forget_qa_simple),
        "qa_retain_size": len(retain_qa_simple),
        "text_forget_size": len(forget_text_simple),
        "has_context": True,
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
