"""Analyze domain generation dataset quality metrics.

Computes:
- Knowledge entanglement between forget/retain sets
- Vocabulary diversity and overlap
- Domain specificity (entity coverage)
- Answer groundedness (overlap with context)
- Self-similarity within sets
- Question type distribution

Usage:
    uv run python scripts/analyze-dataset.py data/datasets/juninho
    uv run python scripts/analyze-dataset.py data/datasets/brazil
"""

import json
import string
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

from datasets import load_from_disk


# ── Text processing ──────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in text.split() if len(w) > 1]


def ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def cosine_sim(counter_a: Counter, counter_b: Counter) -> float:
    """Cosine similarity between two term-frequency vectors."""
    common = set(counter_a.keys()) & set(counter_b.keys())
    dot = sum(counter_a[k] * counter_b[k] for k in common)
    mag_a = sum(v**2 for v in counter_a.values()) ** 0.5
    mag_b = sum(v**2 for v in counter_b.values()) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def self_bleu_sample(texts: list[str], n: int = 4, sample_size: int = 50) -> float:
    """Approximate self-BLEU: avg pairwise n-gram overlap within a set.
    Lower = more diverse. Higher = more repetitive.
    """
    import random
    rng = random.Random(42)

    if len(texts) < 2:
        return 0.0

    sampled = rng.sample(texts, min(sample_size, len(texts)))
    scores = []
    for i, j in combinations(range(len(sampled)), 2):
        ng_a = set(ngrams(tokenize(sampled[i]), n))
        ng_b = set(ngrams(tokenize(sampled[j]), n))
        scores.append(jaccard(ng_a, ng_b))

    return sum(scores) / len(scores) if scores else 0.0


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_entanglement(forget_texts: list[str], retain_texts: list[str]) -> dict:
    """Measure knowledge entanglement between forget and retain sets."""
    forget_words = set()
    retain_words = set()
    forget_counter = Counter()
    retain_counter = Counter()

    for t in forget_texts:
        tokens = tokenize(t)
        forget_words.update(tokens)
        forget_counter.update(tokens)

    for t in retain_texts:
        tokens = tokenize(t)
        retain_words.update(tokens)
        retain_counter.update(tokens)

    # Vocabulary overlap
    vocab_jaccard = jaccard(forget_words, retain_words)

    # TF cosine similarity
    tf_cosine = cosine_sim(forget_counter, retain_counter)

    # Bigram overlap
    forget_bigrams = set()
    retain_bigrams = set()
    for t in forget_texts:
        forget_bigrams.update(ngrams(tokenize(t), 2))
    for t in retain_texts:
        retain_bigrams.update(ngrams(tokenize(t), 2))
    bigram_jaccard = jaccard(forget_bigrams, retain_bigrams)

    # Unique-to-forget vocabulary (domain-specific terms)
    forget_only = forget_words - retain_words
    retain_only = retain_words - forget_words

    return {
        "vocab_jaccard": round(vocab_jaccard, 4),
        "tf_cosine_similarity": round(tf_cosine, 4),
        "bigram_jaccard": round(bigram_jaccard, 4),
        "forget_vocab_size": len(forget_words),
        "retain_vocab_size": len(retain_words),
        "shared_vocab_size": len(forget_words & retain_words),
        "forget_only_vocab_size": len(forget_only),
        "retain_only_vocab_size": len(retain_only),
        "forget_only_sample": sorted(forget_only, key=lambda w: forget_counter[w], reverse=True)[:20],
        "retain_only_sample": sorted(retain_only, key=lambda w: retain_counter[w], reverse=True)[:20],
    }


def compute_diversity(texts: list[str], label: str) -> dict:
    """Measure internal diversity of a text set."""
    all_tokens = []
    lengths = []
    for t in texts:
        tokens = tokenize(t)
        all_tokens.extend(tokens)
        lengths.append(len(tokens))

    # Type-token ratio (vocabulary richness)
    ttr = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0

    # Self-BLEU (lower = more diverse)
    self_bleu_2 = self_bleu_sample(texts, n=2)
    self_bleu_4 = self_bleu_sample(texts, n=4)

    # Length stats
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    min_len = min(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0

    return {
        "count": len(texts),
        "type_token_ratio": round(ttr, 4),
        "self_bleu_2": round(self_bleu_2, 4),
        "self_bleu_4": round(self_bleu_4, 4),
        "avg_token_length": round(avg_len, 1),
        "min_token_length": min_len,
        "max_token_length": max_len,
        "unique_vocab": len(set(all_tokens)),
        "total_tokens": len(all_tokens),
    }


def compute_domain_specificity(qa_pairs: list[dict], entities: set) -> dict:
    """Measure how domain-specific the questions and answers are."""
    q_has_entity = 0
    a_has_entity = 0
    both_have_entity = 0

    entity_counts = Counter()
    question_types = Counter()

    for qa in qa_pairs:
        q = qa["question"].lower()
        a = qa["answer"].lower()

        q_match = any(e in q for e in entities)
        a_match = any(e in a for e in entities)

        if q_match:
            q_has_entity += 1
        if a_match:
            a_has_entity += 1
        if q_match and a_match:
            both_have_entity += 1

        for e in entities:
            if e in q or e in a:
                entity_counts[e] += 1

        # Question type classification
        q_lower = q.strip().split()[0] if q.strip() else ""
        question_types[q_lower] += 1

    n = len(qa_pairs) if qa_pairs else 1
    return {
        "questions_with_entity": f"{q_has_entity}/{n} ({q_has_entity/n*100:.0f}%)",
        "answers_with_entity": f"{a_has_entity}/{n} ({a_has_entity/n*100:.0f}%)",
        "both_with_entity": f"{both_have_entity}/{n} ({both_have_entity/n*100:.0f}%)",
        "top_entities": entity_counts.most_common(15),
        "question_types": question_types.most_common(10),
    }


def compute_groundedness(qa_pairs: list[dict]) -> dict:
    """Measure how grounded answers are in their context passages."""
    if not qa_pairs or "context" not in qa_pairs[0]:
        return {"has_context": False}

    scores = []
    for qa in qa_pairs:
        ctx = qa.get("context", "")
        ans = qa.get("answer", "")
        if not ctx or not ans:
            continue

        ctx_words = set(tokenize(ctx))
        ans_words = set(tokenize(ans))
        if ans_words:
            recall = len(ctx_words & ans_words) / len(ans_words)
            scores.append(recall)

    return {
        "has_context": True,
        "samples_with_context": len(scores),
        "avg_groundedness": round(sum(scores)/len(scores), 4) if scores else 0,
        "min_groundedness": round(min(scores), 4) if scores else 0,
        "max_groundedness": round(max(scores), 4) if scores else 0,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    dataset_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/datasets/juninho")

    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    print(f"{'='*80}")
    print(f"DATASET QUALITY ANALYSIS: {dataset_dir.name}")
    print(f"{'='*80}\n")

    # Load datasets
    forget_qa = load_from_disk(str(dataset_dir / "qa_dataset_forget"))
    retain_qa = load_from_disk(str(dataset_dir / "qa_dataset_retain"))

    text_path = dataset_dir / "text_dataset_forget"
    text_ds = load_from_disk(str(text_path)) if text_path.exists() else None

    # Load metadata for entities
    meta_path = dataset_dir / "metadata.json"
    entities = set()
    if meta_path.exists():
        meta = json.load(open(meta_path))
        entities = set(e.lower() for e in meta.get("domain_entities_sample", []))

    # Also extract entities from domain.json if available
    domain_path = dataset_dir / "domain.json"
    if domain_path.exists():
        domain_data = json.load(open(domain_path))
        entities.add(domain_data["name"].lower())
        for topic in domain_data.get("topics", []):
            for word in topic["name"].lower().split():
                if len(word) > 3:
                    entities.add(word)

    # ── 1. Knowledge Entanglement ────────────────────────────────────────
    print(f"{'─'*80}")
    print("1. KNOWLEDGE ENTANGLEMENT (forget ↔ retain)")
    print(f"{'─'*80}\n")

    forget_texts = [f"{q['question']} {q['answer']}" for q in forget_qa]
    retain_texts = [f"{q['question']} {q['answer']}" for q in retain_qa]

    entanglement = compute_entanglement(forget_texts, retain_texts)

    print(f"  Vocabulary Jaccard:       {entanglement['vocab_jaccard']:.4f}  (0=no overlap, 1=identical)")
    print(f"  TF Cosine Similarity:     {entanglement['tf_cosine_similarity']:.4f}  (0=orthogonal, 1=identical)")
    print(f"  Bigram Jaccard:           {entanglement['bigram_jaccard']:.4f}  (0=no shared phrases)")
    print(f"  Forget-only vocab:        {entanglement['forget_only_vocab_size']} words")
    print(f"  Retain-only vocab:        {entanglement['retain_only_vocab_size']} words")
    print(f"  Shared vocab:             {entanglement['shared_vocab_size']} words")
    print(f"\n  Top forget-only words:    {', '.join(entanglement['forget_only_sample'][:10])}")
    print(f"  Top retain-only words:    {', '.join(entanglement['retain_only_sample'][:10])}")

    # Interpretation
    if entanglement['vocab_jaccard'] > 0.4:
        print(f"\n  ⚠️  HIGH entanglement ({entanglement['vocab_jaccard']:.2f}) -- forget and retain share too much vocabulary")
    elif entanglement['vocab_jaccard'] > 0.2:
        print(f"\n  ⚡ MODERATE entanglement ({entanglement['vocab_jaccard']:.2f}) -- some overlap expected")
    else:
        print(f"\n  ✓ LOW entanglement ({entanglement['vocab_jaccard']:.2f}) -- clean separation")

    # ── 2. Diversity ─────────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print("2. DIVERSITY")
    print(f"{'─'*80}\n")

    for label, texts in [("Forget QA", forget_texts), ("Retain QA", retain_texts)]:
        div = compute_diversity(texts, label)
        print(f"  {label}:")
        print(f"    Samples:            {div['count']}")
        print(f"    Type-Token Ratio:   {div['type_token_ratio']:.4f}  (higher = richer vocabulary)")
        print(f"    Self-BLEU-2:        {div['self_bleu_2']:.4f}  (lower = more diverse)")
        print(f"    Self-BLEU-4:        {div['self_bleu_4']:.4f}")
        print(f"    Avg tokens/sample:  {div['avg_token_length']:.0f}")
        print(f"    Token range:        [{div['min_token_length']}, {div['max_token_length']}]")
        print()

    if text_ds:
        text_texts = [t["text"] for t in text_ds]
        div_text = compute_diversity(text_texts, "Text Passages")
        print(f"  Text Passages:")
        print(f"    Samples:            {div_text['count']}")
        print(f"    Type-Token Ratio:   {div_text['type_token_ratio']:.4f}")
        print(f"    Self-BLEU-2:        {div_text['self_bleu_2']:.4f}")
        print(f"    Avg tokens/sample:  {div_text['avg_token_length']:.0f}")
        print()

    # ── 3. Domain Specificity ────────────────────────────────────────────
    print(f"{'─'*80}")
    print("3. DOMAIN SPECIFICITY")
    print(f"{'─'*80}\n")

    if entities:
        forget_spec = compute_domain_specificity(list(forget_qa), entities)
        retain_spec = compute_domain_specificity(list(retain_qa), entities)

        print(f"  Forget QA:")
        print(f"    Questions with entity: {forget_spec['questions_with_entity']}")
        print(f"    Answers with entity:   {forget_spec['answers_with_entity']}")
        print(f"    Top entities: {', '.join(f'{e}({c})' for e, c in forget_spec['top_entities'][:8])}")
        print(f"    Question types: {', '.join(f'{t}({c})' for t, c in forget_spec['question_types'][:6])}")
        print()

        print(f"  Retain QA:")
        print(f"    Questions with entity: {retain_spec['questions_with_entity']}")
        print(f"    Answers with entity:   {retain_spec['answers_with_entity']}")
        print(f"    Top entities: {', '.join(f'{e}({c})' for e, c in retain_spec['top_entities'][:8])}")
        print(f"    Question types: {', '.join(f'{t}({c})' for t, c in retain_spec['question_types'][:6])}")
    else:
        print("  No entity data available (missing metadata.json or domain.json)")

    # ── 4. Answer Groundedness ───────────────────────────────────────────
    print(f"\n{'─'*80}")
    print("4. ANSWER GROUNDEDNESS (forget QA)")
    print(f"{'─'*80}\n")

    groundedness = compute_groundedness(list(forget_qa))
    if groundedness["has_context"]:
        print(f"  Samples with context:  {groundedness['samples_with_context']}")
        print(f"  Avg groundedness:      {groundedness['avg_groundedness']:.4f}  (answer words found in context)")
        print(f"  Range:                 [{groundedness['min_groundedness']:.3f}, {groundedness['max_groundedness']:.3f}]")
    else:
        print("  No context field in dataset (old format)")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    print(f"  Dataset:              {dataset_dir.name}")
    print(f"  Forget QA:            {len(forget_qa)} samples")
    print(f"  Retain QA:            {len(retain_qa)} samples")
    print(f"  Text Passages:        {len(text_ds) if text_ds else 0}")
    print(f"  Entanglement:         {entanglement['vocab_jaccard']:.3f} (vocab) / {entanglement['tf_cosine_similarity']:.3f} (TF cosine)")
    print(f"  Forget Diversity:     TTR={compute_diversity(forget_texts, '')['type_token_ratio']:.3f}, Self-BLEU-4={compute_diversity(forget_texts, '')['self_bleu_4']:.3f}")
    print(f"  Retain Diversity:     TTR={compute_diversity(retain_texts, '')['type_token_ratio']:.3f}, Self-BLEU-4={compute_diversity(retain_texts, '')['self_bleu_4']:.3f}")

    # Save to JSON
    output = {
        "dataset": dataset_dir.name,
        "entanglement": {k: v for k, v in entanglement.items() if not isinstance(v, list)},
        "forget_diversity": compute_diversity(forget_texts, ""),
        "retain_diversity": compute_diversity(retain_texts, ""),
        "groundedness": groundedness,
    }
    out_path = dataset_dir / "data_quality_metrics.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved metrics to: {out_path}")


if __name__ == "__main__":
    main()
