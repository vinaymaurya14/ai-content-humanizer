"""AI detection analysis engine — combines all detection signals."""

from __future__ import annotations

from app.core.perplexity import (
    calculate_word_perplexity,
    calculate_entropy_scores,
    normalize_perplexity,
    interpret_perplexity,
)
from app.core.burstiness import (
    calculate_burstiness,
    analyze_sentence_patterns,
    interpret_burstiness,
)
from app.models.schemas import (
    DetectionResult,
    PerplexityScore,
    BurstinessScore,
    EntropyScore,
    PatternAnalysis,
)
from app.utils.text_utils import tokenize_sentences, tokenize_words


def analyze_text(text: str) -> DetectionResult:
    """
    Run full AI detection analysis on text.

    Combines perplexity, burstiness, entropy, and pattern analysis
    into a unified AI probability score.
    """
    # Perplexity analysis
    raw_perplexity = calculate_word_perplexity(text)
    norm_perplexity = normalize_perplexity(raw_perplexity)

    perplexity = PerplexityScore(
        score=round(raw_perplexity, 4),
        normalized=norm_perplexity,
        interpretation=interpret_perplexity(norm_perplexity),
    )

    # Burstiness analysis
    burstiness_data = calculate_burstiness(text)

    burstiness = BurstinessScore(
        score=round(burstiness_data["score"], 4),
        sentence_length_variance=round(burstiness_data["sentence_length_variance"], 4),
        vocabulary_richness=round(burstiness_data["vocabulary_richness"], 4),
        interpretation=interpret_burstiness(burstiness_data["score"]),
    )

    # Entropy analysis
    entropy_data = calculate_entropy_scores(text)

    entropy = EntropyScore(
        word_entropy=entropy_data["word_entropy"],
        char_entropy=entropy_data["char_entropy"],
        normalized=entropy_data["normalized"],
    )

    # Pattern analysis
    patterns_data = analyze_sentence_patterns(text)

    pattern_analysis = PatternAnalysis(
        repetitive_starts=patterns_data["repetitive_starts"],
        avg_sentence_length=patterns_data["avg_sentence_length"],
        sentence_length_std=patterns_data["sentence_length_std"],
        passive_voice_ratio=patterns_data["passive_voice_ratio"],
        transition_density=patterns_data["transition_density"],
        filler_phrase_count=patterns_data["filler_phrase_count"],
    )

    # --- Compute unified AI probability ---
    # Each signal contributes to AI likelihood

    signals = []

    # Low perplexity → more AI-like (invert normalized score)
    perplexity_signal = 1.0 - norm_perplexity
    signals.append(("perplexity", perplexity_signal, 0.25))

    # Low burstiness → more AI-like (invert score)
    burstiness_signal = 1.0 - min(burstiness_data["score"] / 0.6, 1.0)
    signals.append(("burstiness", burstiness_signal, 0.20))

    # Low entropy → more AI-like (invert normalized)
    entropy_signal = 1.0 - entropy_data["normalized"]
    signals.append(("entropy", entropy_signal, 0.15))

    # High repetitive starts → more AI-like
    signals.append(("repetitive_starts", patterns_data["repetitive_starts"], 0.10))

    # Low sentence length std → more AI-like
    std_signal = 1.0 - min(patterns_data["sentence_length_std"] / 15.0, 1.0)
    signals.append(("length_uniformity", std_signal, 0.10))

    # High transition density → more AI-like
    transition_signal = min(patterns_data["transition_density"] / 0.05, 1.0)
    signals.append(("transitions", transition_signal, 0.10))

    # Filler phrases → strong AI signal
    filler_signal = min(patterns_data["filler_phrase_count"] / 5.0, 1.0)
    signals.append(("fillers", filler_signal, 0.10))

    # Weighted average
    total_weight = sum(w for _, _, w in signals)
    ai_probability = sum(score * weight for _, score, weight in signals) / total_weight
    ai_probability = max(0.0, min(1.0, ai_probability))

    # Determine verdict and confidence
    if ai_probability > 0.7:
        verdict = "likely_ai"
        confidence = min(ai_probability, 0.95)
    elif ai_probability < 0.35:
        verdict = "likely_human"
        confidence = min(1.0 - ai_probability, 0.95)
    else:
        verdict = "uncertain"
        confidence = 1.0 - abs(ai_probability - 0.5) * 2  # Lower confidence near boundary

    return DetectionResult(
        ai_probability=round(ai_probability, 4),
        perplexity=perplexity,
        burstiness=burstiness,
        entropy=entropy,
        pattern_analysis=pattern_analysis,
        verdict=verdict,
        confidence=round(confidence, 4),
    )
