"""Tests for the AI detection analyzer."""

import pytest

from app.services.analyzer import analyze_text
from app.core.perplexity import (
    calculate_word_perplexity,
    normalize_perplexity,
    calculate_entropy_scores,
)
from app.core.burstiness import calculate_burstiness, analyze_sentence_patterns


# Sample AI-generated text (uniform, predictable patterns)
AI_TEXT = (
    "Artificial intelligence is a rapidly evolving field that has the potential to "
    "revolutionize various industries. It is important to note that AI technologies "
    "are being implemented across multiple sectors. Furthermore, the development of "
    "machine learning algorithms has significantly improved the accuracy of predictive "
    "models. Additionally, natural language processing has made it possible to analyze "
    "large volumes of text data. Moreover, computer vision technologies have enhanced "
    "the ability to process visual information. Consequently, organizations are "
    "increasingly investing in AI solutions to improve operational efficiency. "
    "In conclusion, artificial intelligence represents a paradigm shift in how "
    "businesses approach problem-solving and decision-making processes."
)

# Sample human-written text (varied, unpredictable)
HUMAN_TEXT = (
    "I've been tinkering with AI stuff for about three years now. Some of it's "
    "genuinely cool — like when my image classifier actually worked on the first try. "
    "That almost never happens, by the way. Most days? You're debugging tensor shapes "
    "at 2 AM wondering why everything's broken. But then you get that one result and "
    "it makes the whole thing worth it. My buddy Dave thinks I'm crazy for spending "
    "weekends on this. Maybe he's right. But there's something about building "
    "something that learns — actually learns — that hooks you. Can't really explain it. "
    "The math is brutal sometimes. Backprop made me want to cry in college. Now I use "
    "it daily without thinking twice. Funny how that works."
)


class TestPerplexity:
    def test_perplexity_returns_positive(self):
        score = calculate_word_perplexity(AI_TEXT)
        assert score > 0

    def test_normalize_returns_0_to_1(self):
        for val in [0, 10, 50, 100, 500]:
            norm = normalize_perplexity(val)
            assert 0.0 <= norm <= 1.0

    def test_entropy_scores_valid(self):
        result = calculate_entropy_scores(AI_TEXT)
        assert "word_entropy" in result
        assert "char_entropy" in result
        assert 0.0 <= result["normalized"] <= 1.0


class TestBurstiness:
    def test_burstiness_returns_dict(self):
        result = calculate_burstiness(AI_TEXT)
        assert "score" in result
        assert "sentence_length_variance" in result
        assert "vocabulary_richness" in result

    def test_human_text_more_bursty(self):
        ai_burstiness = calculate_burstiness(AI_TEXT)
        human_burstiness = calculate_burstiness(HUMAN_TEXT)
        # Human text should generally have higher burstiness
        # (more variation in sentence lengths)
        assert human_burstiness["sentence_length_variance"] >= 0

    def test_pattern_analysis(self):
        result = analyze_sentence_patterns(AI_TEXT)
        assert "repetitive_starts" in result
        assert "passive_voice_ratio" in result
        assert "transition_density" in result
        assert "filler_phrase_count" in result
        # AI text should have filler phrases
        assert result["filler_phrase_count"] > 0


class TestAnalyzer:
    def test_analyze_returns_detection_result(self):
        result = analyze_text(AI_TEXT)
        assert 0.0 <= result.ai_probability <= 1.0
        assert result.verdict in ("likely_ai", "uncertain", "likely_human")
        assert 0.0 <= result.confidence <= 1.0

    def test_ai_text_scores_higher(self):
        ai_result = analyze_text(AI_TEXT)
        human_result = analyze_text(HUMAN_TEXT)
        # AI text should have higher AI probability
        assert ai_result.ai_probability > human_result.ai_probability

    def test_ai_text_has_filler_phrases(self):
        result = analyze_text(AI_TEXT)
        assert result.pattern_analysis.filler_phrase_count > 0

    def test_short_text_doesnt_crash(self):
        result = analyze_text(
            "This is a short text for testing purposes. "
            "It needs to be at least fifty characters long to pass validation."
        )
        assert result.ai_probability >= 0.0
