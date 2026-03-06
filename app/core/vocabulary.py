"""Vocabulary diversification for humanization."""

from __future__ import annotations

import random
import re
from collections import Counter

from app.utils.text_utils import tokenize_words, get_stopwords


# Synonym clusters for diversification
SYNONYM_CLUSTERS = {
    "important": ["key", "critical", "vital", "significant", "major", "essential"],
    "significant": ["notable", "substantial", "meaningful", "considerable", "major"],
    "effective": ["successful", "productive", "powerful", "practical", "useful"],
    "various": ["several", "different", "diverse", "multiple", "a range of"],
    "provide": ["offer", "give", "supply", "deliver", "bring"],
    "provides": ["offers", "gives", "supplies", "delivers", "brings"],
    "increase": ["boost", "raise", "grow", "expand", "improve"],
    "decrease": ["reduce", "lower", "cut", "shrink", "drop"],
    "create": ["build", "make", "develop", "produce", "form"],
    "creates": ["builds", "makes", "develops", "produces", "forms"],
    "improve": ["enhance", "upgrade", "boost", "strengthen", "refine"],
    "improves": ["enhances", "upgrades", "boosts", "strengthens", "refines"],
    "ensure": ["make sure", "guarantee", "confirm", "verify"],
    "ensures": ["makes sure", "guarantees", "confirms", "verifies"],
    "require": ["need", "call for", "demand", "take"],
    "requires": ["needs", "calls for", "demands", "takes"],
    "achieve": ["reach", "hit", "accomplish", "attain", "get"],
    "consider": ["think about", "look at", "weigh", "examine"],
    "approach": ["method", "strategy", "technique", "way", "tactic"],
    "solution": ["answer", "fix", "remedy", "resolution"],
    "challenge": ["problem", "issue", "hurdle", "difficulty", "obstacle"],
    "opportunity": ["chance", "opening", "prospect", "possibility"],
    "development": ["growth", "progress", "advance", "evolution"],
    "environment": ["setting", "context", "landscape", "space"],
    "experience": ["encounter", "interaction", "exposure"],
    "information": ["data", "details", "facts", "intel", "insight"],
    "strategy": ["plan", "approach", "tactic", "playbook"],
    "process": ["procedure", "workflow", "system", "routine"],
    "result": ["outcome", "effect", "consequence", "finding"],
    "results": ["outcomes", "effects", "consequences", "findings"],
    "feature": ["trait", "aspect", "quality", "characteristic"],
    "features": ["traits", "aspects", "qualities", "characteristics"],
    "benefit": ["advantage", "perk", "upside", "plus"],
    "benefits": ["advantages", "perks", "upsides", "gains"],
    "maintain": ["keep", "preserve", "sustain", "hold"],
    "establish": ["set up", "build", "create", "form"],
    "determine": ["figure out", "decide", "identify", "find"],
    "analyze": ["examine", "study", "review", "assess"],
    "demonstrate": ["show", "prove", "illustrate", "reveal"],
    "indicates": ["shows", "suggests", "points to", "signals"],
    "ability": ["capacity", "skill", "power", "capability"],
    "generate": ["produce", "create", "make", "yield"],
    "impact": ["effect", "influence", "mark", "impression"],
}


def diversify_vocabulary(text: str, strength: float = 0.7) -> str:
    """
    Replace overused words with synonyms to increase vocabulary diversity.

    Only replaces words that appear multiple times to reduce repetition.
    """
    words = tokenize_words(text)
    word_counts = Counter(words)
    stopwords = get_stopwords()

    # Find repeated non-stopword words that have synonyms
    replacements: dict[str, list[str]] = {}
    for word, count in word_counts.items():
        if count > 1 and word not in stopwords and word in SYNONYM_CLUSTERS:
            replacements[word] = SYNONYM_CLUSTERS[word]

    if not replacements:
        return text

    result = text
    for word, synonyms in replacements.items():
        # Find all occurrences
        pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
        matches = list(pattern.finditer(result))

        if len(matches) <= 1:
            continue

        # Keep first occurrence, replace subsequent ones (based on strength)
        for match in reversed(matches[1:]):
            if random.random() < strength:
                synonym = random.choice(synonyms)
                # Preserve capitalization
                original = match.group()
                if original[0].isupper():
                    synonym = synonym[0].upper() + synonym[1:]
                result = result[:match.start()] + synonym + result[match.end():]

    return result


def add_rhythm_variation(text: str, strength: float = 0.7) -> str:
    """
    Add rhythmic variation by occasionally inserting short emphatic phrases
    or breaking up monotonous patterns.
    """
    lines = text.split("\n")
    result = []

    for line in lines:
        if not line.strip() or line.strip().startswith("#"):
            result.append(line)
            continue

        sentences = line.split(". ")
        modified = []

        for i, sentence in enumerate(sentences):
            words = sentence.split()

            # Occasionally add emphasis markers
            if (
                len(words) > 15
                and random.random() < strength * 0.15
                and i > 0
            ):
                emphatics = [
                    "And that matters.",
                    "That's a big deal.",
                    "This is key.",
                    "Worth remembering.",
                ]
                modified.append(random.choice(emphatics))

            modified.append(sentence)

        result.append(". ".join(modified))

    return "\n".join(result)
