"""Readability scoring with multiple indices."""

from __future__ import annotations

import math

from app.utils.text_utils import (
    tokenize_sentences,
    tokenize_words,
    count_syllables,
    is_complex_word,
)
from app.models.schemas import ReadabilityScores


def calculate_readability(text: str) -> ReadabilityScores:
    """Calculate comprehensive readability metrics."""
    sentences = tokenize_sentences(text)
    words = tokenize_words(text)
    total_sentences = max(len(sentences), 1)
    total_words = max(len(words), 1)

    # Syllable counts
    syllable_counts = [count_syllables(w) for w in words]
    total_syllables = sum(syllable_counts)
    avg_syllables = total_syllables / total_words
    avg_sentence_length = total_words / total_sentences

    # Complex words (3+ syllables)
    complex_words = sum(1 for w in words if is_complex_word(w))

    # --- Flesch Reading Ease ---
    # 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)
    flesch_ease = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables
    flesch_ease = max(0.0, min(100.0, flesch_ease))

    # --- Flesch-Kincaid Grade Level ---
    # 0.39*(words/sentences) + 11.8*(syllables/words) - 15.59
    fk_grade = 0.39 * avg_sentence_length + 11.8 * avg_syllables - 15.59
    fk_grade = max(0.0, fk_grade)

    # --- Gunning Fog Index ---
    # 0.4 * (avg_sentence_length + 100*(complex_words/total_words))
    complex_ratio = complex_words / total_words
    gunning_fog = 0.4 * (avg_sentence_length + 100 * complex_ratio)

    # --- Coleman-Liau Index ---
    # 0.0588*L - 0.296*S - 15.8
    # L = avg number of letters per 100 words
    # S = avg number of sentences per 100 words
    total_letters = sum(len(w) for w in words)
    l_score = (total_letters / total_words) * 100
    s_score = (total_sentences / total_words) * 100
    coleman_liau = 0.0588 * l_score - 0.296 * s_score - 15.8
    coleman_liau = max(0.0, coleman_liau)

    # --- SMOG Index ---
    # 1.0430 * sqrt(complex_words * (30/sentences)) + 3.1291
    if total_sentences >= 3:
        smog = 1.0430 * math.sqrt(complex_words * (30 / total_sentences)) + 3.1291
    else:
        smog = fk_grade  # Fallback for very short texts

    # Interpretation
    if flesch_ease >= 80:
        interpretation = "Very easy to read — suitable for a broad audience"
    elif flesch_ease >= 60:
        interpretation = "Fairly easy to read — standard accessibility"
    elif flesch_ease >= 40:
        interpretation = "Moderately difficult — requires some education"
    elif flesch_ease >= 20:
        interpretation = "Difficult — best for specialized audiences"
    else:
        interpretation = "Very difficult — academic or technical level"

    return ReadabilityScores(
        flesch_kincaid_grade=round(fk_grade, 2),
        flesch_reading_ease=round(flesch_ease, 2),
        gunning_fog=round(gunning_fog, 2),
        coleman_liau=round(coleman_liau, 2),
        smog_index=round(smog, 2),
        avg_sentence_length=round(avg_sentence_length, 2),
        avg_syllables_per_word=round(avg_syllables, 2),
        total_words=total_words,
        total_sentences=total_sentences,
        interpretation=interpretation,
    )
