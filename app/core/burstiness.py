"""Burstiness and variation analysis for AI detection."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from app.utils.text_utils import (
    tokenize_sentences,
    tokenize_words,
    get_stopwords,
    AI_FILLER_PHRASES,
    TRANSITION_WORDS,
    is_passive_voice,
)


def calculate_burstiness(text: str) -> dict[str, float]:
    """
    Calculate burstiness metrics.

    Human writing is 'bursty' — sentence lengths vary significantly.
    AI writing tends toward uniform sentence lengths.
    """
    sentences = tokenize_sentences(text)
    if len(sentences) < 2:
        return {
            "score": 0.0,
            "sentence_length_variance": 0.0,
            "vocabulary_richness": 0.0,
            "mean_sentence_length": 0.0,
            "coefficient_of_variation": 0.0,
        }

    # Sentence length variation
    lengths = [len(tokenize_words(s)) for s in sentences]
    lengths_arr = np.array(lengths, dtype=float)
    mean_len = float(np.mean(lengths_arr))
    std_len = float(np.std(lengths_arr))
    variance = float(np.var(lengths_arr))

    # Coefficient of variation (normalized measure of dispersion)
    cv = std_len / mean_len if mean_len > 0 else 0.0

    # Vocabulary richness (type-token ratio)
    all_words = tokenize_words(text)
    ttr = len(set(all_words)) / len(all_words) if all_words else 0.0

    # Burstiness score: combines CV and TTR
    # Higher = more human-like
    burstiness = (cv * 0.6 + ttr * 0.4)

    return {
        "score": round(burstiness, 4),
        "sentence_length_variance": round(variance, 4),
        "vocabulary_richness": round(ttr, 4),
        "mean_sentence_length": round(mean_len, 2),
        "coefficient_of_variation": round(cv, 4),
    }


def analyze_sentence_patterns(text: str) -> dict:
    """
    Analyze sentence-level patterns that distinguish AI from human text.

    AI tends to:
    - Start sentences with similar words/phrases
    - Use consistent sentence structures
    - Over-use transition words
    - Have uniform paragraph lengths
    """
    sentences = tokenize_sentences(text)
    if not sentences:
        return {
            "repetitive_starts": 0.0,
            "avg_sentence_length": 0.0,
            "sentence_length_std": 0.0,
            "passive_voice_ratio": 0.0,
            "transition_density": 0.0,
            "filler_phrase_count": 0,
        }

    # Check for repetitive sentence starts (first 1-2 words)
    starts = []
    for s in sentences:
        words = s.split()[:2]
        if words:
            starts.append(" ".join(words).lower())

    start_counts = Counter(starts)
    total = len(starts)
    repetitive = sum(c for c in start_counts.values() if c > 1)
    repetitive_ratio = repetitive / total if total > 0 else 0.0

    # Sentence lengths
    lengths = [len(tokenize_words(s)) for s in sentences]
    avg_len = sum(lengths) / len(lengths) if lengths else 0.0
    std_len = float(np.std(lengths)) if len(lengths) > 1 else 0.0

    # Passive voice detection
    passive_count = sum(1 for s in sentences if is_passive_voice(s))
    passive_ratio = passive_count / len(sentences) if sentences else 0.0

    # Transition word density
    text_lower = text.lower()
    all_words = tokenize_words(text)
    transition_count = sum(1 for w in all_words if w in TRANSITION_WORDS)
    transition_density = transition_count / len(all_words) if all_words else 0.0

    # AI filler phrase count
    filler_count = sum(1 for phrase in AI_FILLER_PHRASES if phrase in text_lower)

    return {
        "repetitive_starts": round(repetitive_ratio, 4),
        "avg_sentence_length": round(avg_len, 2),
        "sentence_length_std": round(std_len, 2),
        "passive_voice_ratio": round(passive_ratio, 4),
        "transition_density": round(transition_density, 4),
        "filler_phrase_count": filler_count,
    }


def interpret_burstiness(score: float) -> str:
    """Interpret burstiness score."""
    if score < 0.25:
        return "Very low burstiness — uniform writing pattern, strongly consistent with AI generation"
    elif score < 0.40:
        return "Low burstiness — somewhat uniform, likely AI-generated"
    elif score < 0.55:
        return "Moderate burstiness — mixed signals, uncertain origin"
    else:
        return "High burstiness — varied writing pattern, consistent with human writing"
