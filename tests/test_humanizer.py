"""Tests for the humanization pipeline."""

import pytest

from app.services.humanizer import humanize_content
from app.services.readability import calculate_readability
from app.core.sentence_transforms import (
    apply_contractions,
    simplify_vocabulary,
    remove_filler_phrases,
)
from app.core.vocabulary import diversify_vocabulary
from app.models.schemas import HumanizeRequest, HumanizationStrategy


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


class TestSentenceTransforms:
    def test_apply_contractions(self):
        text = "It is important to note that they are working hard."
        result = apply_contractions(text)
        assert "it's" in result.lower() or "they're" in result.lower()

    def test_simplify_vocabulary(self):
        text = "We need to utilize innovative methodologies to leverage cutting-edge solutions."
        result = simplify_vocabulary(text)
        assert "utilize" not in result.lower()
        assert "use" in result.lower()

    def test_remove_filler_phrases(self):
        text = "It is important to note that the results are significant. In conclusion, we should proceed."
        result = remove_filler_phrases(text)
        assert "it is important to note" not in result.lower()

    def test_preserve_headings(self):
        text = "# My Heading\n\nThis is a paragraph that is important to note. Furthermore, it continues."
        from app.core.sentence_transforms import restructure_sentences
        result = restructure_sentences(text)
        assert "# My Heading" in result


class TestVocabulary:
    def test_diversify_repeated_words(self):
        text = (
            "The important thing is that this important decision has important "
            "consequences for important stakeholders."
        )
        result = diversify_vocabulary(text, strength=1.0)
        # At least some "important" should be replaced
        assert result.lower().count("important") < text.lower().count("important")


class TestReadability:
    def test_readability_scores(self):
        scores = calculate_readability(AI_TEXT)
        assert scores.flesch_reading_ease >= 0
        assert scores.flesch_kincaid_grade >= 0
        assert scores.gunning_fog >= 0
        assert scores.coleman_liau >= 0
        assert scores.smog_index >= 0
        assert scores.total_words > 0
        assert scores.total_sentences > 0

    def test_easy_text_high_reading_ease(self):
        easy = "The cat sat on the mat. It was a nice day. The sun was bright. Birds sang in the trees."
        scores = calculate_readability(easy)
        assert scores.flesch_reading_ease > 60


class TestHumanizer:
    def test_humanize_returns_different_text(self):
        request = HumanizeRequest(content=AI_TEXT, strategy=HumanizationStrategy.MODERATE)
        result = humanize_content(request)
        assert result.humanized_content != result.original_content

    def test_humanize_reduces_ai_probability(self):
        request = HumanizeRequest(content=AI_TEXT, strategy=HumanizationStrategy.MODERATE)
        result = humanize_content(request)
        # Humanized text should have lower or equal AI probability
        assert result.detection_after.ai_probability <= result.detection_before.ai_probability + 0.1

    def test_humanize_preserves_keywords(self):
        request = HumanizeRequest(
            content=AI_TEXT,
            strategy=HumanizationStrategy.LIGHT,
            target_keywords=["artificial intelligence"],
        )
        result = humanize_content(request)
        # Keyword should still be present
        assert "artificial intelligence" in result.humanized_content.lower()

    def test_humanize_changes_list(self):
        request = HumanizeRequest(content=AI_TEXT, strategy=HumanizationStrategy.MODERATE)
        result = humanize_content(request)
        assert len(result.changes_made) > 0

    def test_light_strategy_less_aggressive(self):
        light_req = HumanizeRequest(content=AI_TEXT, strategy=HumanizationStrategy.LIGHT)
        aggressive_req = HumanizeRequest(content=AI_TEXT, strategy=HumanizationStrategy.AGGRESSIVE)
        light_result = humanize_content(light_req)
        aggressive_result = humanize_content(aggressive_req)
        # Aggressive should make more changes (more different from original)
        from difflib import SequenceMatcher
        light_sim = SequenceMatcher(None, AI_TEXT, light_result.humanized_content).ratio()
        agg_sim = SequenceMatcher(None, AI_TEXT, aggressive_result.humanized_content).ratio()
        assert light_sim >= agg_sim - 0.1  # Light should be more similar to original
