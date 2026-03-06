"""Core humanization pipeline — orchestrates all transforms."""

from __future__ import annotations

import random

from app.core.sentence_transforms import restructure_sentences
from app.core.vocabulary import diversify_vocabulary, add_rhythm_variation
from app.services.analyzer import analyze_text
from app.services.readability import calculate_readability
from app.services.seo_preserver import analyze_seo_preservation
from app.models.schemas import (
    HumanizeRequest,
    HumanizeResponse,
    HumanizationStrategy,
)
from app.utils.text_utils import calculate_keyword_density


# Strategy presets
STRATEGY_CONFIG = {
    HumanizationStrategy.LIGHT: {
        "sentence_restructure": True,
        "vocabulary_diversify": True,
        "rhythm_variation": False,
        "strength_multiplier": 0.4,
    },
    HumanizationStrategy.MODERATE: {
        "sentence_restructure": True,
        "vocabulary_diversify": True,
        "rhythm_variation": True,
        "strength_multiplier": 0.7,
    },
    HumanizationStrategy.AGGRESSIVE: {
        "sentence_restructure": True,
        "vocabulary_diversify": True,
        "rhythm_variation": True,
        "strength_multiplier": 1.0,
    },
}


def humanize_content(request: HumanizeRequest) -> HumanizeResponse:
    """
    Run the full humanization pipeline on content.

    Steps:
    1. Analyze original text for AI patterns
    2. Apply sentence restructuring
    3. Diversify vocabulary
    4. Add rhythm variation
    5. Re-analyze to verify improvement
    6. Check SEO preservation
    """
    # Set random seed for reproducibility within a request
    random.seed(hash(request.content[:100]))

    config = STRATEGY_CONFIG[request.strategy]
    effective_strength = request.strength * config["strength_multiplier"]

    changes_made = []
    result = request.content

    # Step 1: Analyze original
    detection_before = analyze_text(request.content)
    readability_before = calculate_readability(request.content)

    # Step 2: Sentence restructuring
    if config["sentence_restructure"]:
        result = restructure_sentences(result, effective_strength)
        changes_made.append("Restructured sentences for natural variation")
        changes_made.append("Applied contractions for conversational tone")
        changes_made.append("Simplified overly formal vocabulary")
        changes_made.append("Removed AI filler phrases")

    # Step 3: Vocabulary diversification
    if config["vocabulary_diversify"]:
        result = diversify_vocabulary(result, effective_strength)
        changes_made.append("Diversified repeated vocabulary with synonyms")

    # Step 4: Rhythm variation
    if config["rhythm_variation"]:
        result = add_rhythm_variation(result, effective_strength)
        changes_made.append("Added rhythmic variation to sentence patterns")

    # Step 5: SEO keyword restoration
    # If keywords were lost during humanization, try to restore them
    if request.target_keywords:
        result = _restore_keywords(request.content, result, request.target_keywords)
        changes_made.append("Verified and restored target keyword density")

    # Step 6: Re-analyze
    detection_after = analyze_text(result)
    readability_after = calculate_readability(result)

    # Step 7: SEO preservation check
    seo_result = analyze_seo_preservation(
        request.content, result, request.target_keywords
    )

    return HumanizeResponse(
        original_content=request.content,
        humanized_content=result,
        detection_before=detection_before,
        detection_after=detection_after,
        readability_before=readability_before,
        readability_after=readability_after,
        seo_preservation=seo_result,
        changes_made=changes_made,
    )


def _restore_keywords(original: str, humanized: str, keywords: list[str]) -> str:
    """
    Restore keyword density if it dropped too much during humanization.

    Finds sentences where keywords can be naturally reinserted.
    """
    result = humanized

    for keyword in keywords:
        orig_density = calculate_keyword_density(original, keyword)
        new_density = calculate_keyword_density(result, keyword)

        # If density dropped by more than 40%, try to restore
        if orig_density > 0 and new_density < orig_density * 0.6:
            # Find a sentence that could naturally contain the keyword
            sentences = result.split(". ")
            # Add keyword to a random sentence that doesn't already have it
            keyword_lower = keyword.lower()
            candidates = [
                i for i, s in enumerate(sentences)
                if keyword_lower not in s.lower() and len(s.split()) > 5
            ]
            if candidates:
                idx = random.choice(candidates)
                # Append keyword phrase naturally
                sentences[idx] = sentences[idx].rstrip(".") + f", particularly regarding {keyword}."
                result = ". ".join(sentences)

    return result
