"""SEO metric preservation during humanization."""

from __future__ import annotations

from app.utils.text_utils import (
    calculate_keyword_density,
    extract_headings,
    tokenize_words,
)
from app.models.schemas import SEOPreservationResult, KeywordAnalysis
from app.config import settings


def analyze_seo_preservation(
    original: str,
    humanized: str,
    target_keywords: list[str],
) -> SEOPreservationResult:
    """
    Compare SEO metrics between original and humanized content.

    Checks keyword density, heading structure, and content integrity.
    """
    keyword_results = []
    recommendations = []

    # --- Keyword density analysis ---
    for keyword in target_keywords:
        density_before = calculate_keyword_density(original, keyword)
        density_after = calculate_keyword_density(humanized, keyword)

        # Check if density was preserved within acceptable range
        if density_before > 0:
            change_ratio = abs(density_after - density_before) / density_before
            preserved = change_ratio < 0.3  # Within 30% of original
        else:
            preserved = density_after == 0  # No keyword, nothing to preserve

        keyword_results.append(KeywordAnalysis(
            keyword=keyword,
            density_before=round(density_before, 4),
            density_after=round(density_after, 4),
            preserved=preserved,
        ))

        if not preserved:
            if density_after < density_before:
                recommendations.append(
                    f"Keyword '{keyword}' density dropped from {density_before:.2f}% "
                    f"to {density_after:.2f}%. Consider re-adding the keyword."
                )
            else:
                recommendations.append(
                    f"Keyword '{keyword}' density increased from {density_before:.2f}% "
                    f"to {density_after:.2f}%. Consider reducing occurrences."
                )

    # --- Heading structure analysis ---
    original_headings = extract_headings(original)
    humanized_headings = extract_headings(humanized)

    heading_preserved = True
    if len(original_headings) != len(humanized_headings):
        heading_preserved = False
        recommendations.append(
            f"Heading count changed from {len(original_headings)} to {len(humanized_headings)}. "
            "Ensure heading structure is maintained."
        )
    else:
        for (orig_level, orig_text), (new_level, new_text) in zip(
            original_headings, humanized_headings
        ):
            if orig_level != new_level:
                heading_preserved = False
                recommendations.append(
                    f"Heading level changed: H{orig_level} → H{new_level} for '{orig_text}'"
                )
                break

    # --- Content length analysis ---
    orig_words = len(tokenize_words(original))
    new_words = len(tokenize_words(humanized))
    length_change = abs(new_words - orig_words) / orig_words if orig_words > 0 else 0

    if length_change > 0.2:
        recommendations.append(
            f"Content length changed significantly: {orig_words} → {new_words} words "
            f"({length_change:.0%} change). Consider adjusting."
        )

    # --- Meta integrity (check if any structured data markers were removed) ---
    meta_intact = True
    meta_patterns = ["<!--", "{{", "{%", "---"]  # Common meta/template markers
    for pattern in meta_patterns:
        if pattern in original and pattern not in humanized:
            meta_intact = False
            recommendations.append(f"Meta marker '{pattern}' was removed during humanization.")

    # --- Calculate overall SEO score ---
    score_components = []

    # Keyword preservation (40% weight)
    if keyword_results:
        keyword_score = sum(100 if k.preserved else 50 for k in keyword_results) / len(keyword_results)
    else:
        keyword_score = 100.0
    score_components.append(("keywords", keyword_score, 0.4))

    # Heading preservation (25% weight)
    heading_score = 100.0 if heading_preserved else 50.0
    score_components.append(("headings", heading_score, 0.25))

    # Content length (20% weight)
    length_score = max(0, 100 - length_change * 200)
    score_components.append(("length", length_score, 0.20))

    # Meta integrity (15% weight)
    meta_score = 100.0 if meta_intact else 60.0
    score_components.append(("meta", meta_score, 0.15))

    overall_score = sum(s * w for _, s, w in score_components)

    if not recommendations:
        recommendations.append("SEO metrics are well-preserved after humanization.")

    return SEOPreservationResult(
        keyword_analysis=keyword_results,
        heading_structure_preserved=heading_preserved,
        meta_integrity=meta_intact,
        overall_seo_score=round(overall_score, 2),
        recommendations=recommendations,
    )
