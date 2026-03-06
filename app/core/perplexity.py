"""Perplexity and entropy calculations for AI detection."""

from __future__ import annotations

import math
from collections import Counter

from app.utils.text_utils import tokenize_words, tokenize_sentences, shannon_entropy


def calculate_word_perplexity(text: str) -> float:
    """
    Calculate pseudo-perplexity using bigram language model.

    AI text tends to have lower perplexity (more predictable word sequences).
    Human text has higher perplexity (more surprising word choices).
    """
    words = tokenize_words(text)
    if len(words) < 3:
        return 0.0

    # Build bigram model
    bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
    unigram_counts = Counter(words)
    bigram_counts = Counter(bigrams)
    total_unigrams = len(words)

    # Calculate log probability
    log_prob_sum = 0.0
    n = 0
    vocab_size = len(unigram_counts)

    for bigram in bigrams:
        # Add-1 (Laplace) smoothing
        bigram_count = bigram_counts[bigram]
        unigram_count = unigram_counts[bigram[0]]
        prob = (bigram_count + 1) / (unigram_count + vocab_size)
        log_prob_sum += math.log2(prob)
        n += 1

    if n == 0:
        return 0.0

    avg_log_prob = log_prob_sum / n
    perplexity = 2 ** (-avg_log_prob)
    return perplexity


def calculate_entropy_scores(text: str) -> dict[str, float]:
    """Calculate word-level and character-level Shannon entropy."""
    word_ent = shannon_entropy(text, level="word")
    char_ent = shannon_entropy(text, level="char")

    # Normalize: typical ranges — word entropy 6-12, char entropy 3-5
    word_normalized = min(word_ent / 12.0, 1.0)
    char_normalized = min(char_ent / 5.0, 1.0)
    combined = (word_normalized + char_normalized) / 2

    return {
        "word_entropy": round(word_ent, 4),
        "char_entropy": round(char_ent, 4),
        "normalized": round(combined, 4),
    }


def calculate_vocabulary_diversity(text: str) -> dict[str, float]:
    """
    Calculate vocabulary diversity metrics.

    AI text tends to have lower type-token ratio (reuses words more uniformly).
    """
    words = tokenize_words(text)
    if not words:
        return {"ttr": 0.0, "hapax_ratio": 0.0, "yules_k": 0.0}

    types = set(words)
    tokens = len(words)
    freq = Counter(words)

    # Type-Token Ratio
    ttr = len(types) / tokens

    # Hapax legomena ratio (words appearing once)
    hapax = sum(1 for count in freq.values() if count == 1)
    hapax_ratio = hapax / tokens

    # Yule's K (lower = more diverse vocabulary)
    m1 = tokens
    m2 = sum(count * count for count in freq.values())
    yules_k = 10000 * (m2 - m1) / (m1 * m1) if m1 > 0 else 0.0

    return {
        "ttr": round(ttr, 4),
        "hapax_ratio": round(hapax_ratio, 4),
        "yules_k": round(yules_k, 4),
    }


def normalize_perplexity(perplexity: float) -> float:
    """
    Normalize perplexity to 0-1 range.

    Low perplexity (< 20) → likely AI (score near 0)
    High perplexity (> 100) → likely human (score near 1)
    """
    if perplexity <= 0:
        return 0.0
    # Sigmoid-like mapping
    normalized = 1 - (1 / (1 + math.exp((perplexity - 50) / 20)))
    return round(max(0.0, min(1.0, normalized)), 4)


def interpret_perplexity(normalized: float) -> str:
    """Interpret normalized perplexity score."""
    if normalized < 0.3:
        return "Very low perplexity — text is highly predictable, consistent with AI generation"
    elif normalized < 0.5:
        return "Low perplexity — text shows some predictability, may be AI-generated"
    elif normalized < 0.7:
        return "Moderate perplexity — text shows mixed signals, uncertain origin"
    else:
        return "High perplexity — text shows natural unpredictability, consistent with human writing"
