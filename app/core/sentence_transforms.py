"""Sentence restructuring engine for humanization."""

from __future__ import annotations

import random
import re

from app.utils.text_utils import (
    tokenize_sentences,
    tokenize_words,
    AI_FILLER_PHRASES,
    TRANSITION_WORDS,
)


# Sentence starters to introduce variety
HUMAN_STARTERS = [
    "", "Look, ", "Here's the thing — ", "Honestly, ", "The truth is, ",
    "What's interesting is ", "One thing to consider: ", "Think about it — ",
    "In practice, ", "From experience, ", "Realistically, ",
    "It turns out ", "Surprisingly, ", "Interestingly, ",
]

# Contractions map
CONTRACTIONS = {
    "it is": "it's", "that is": "that's", "there is": "there's",
    "what is": "what's", "who is": "who's", "how is": "how's",
    "he is": "he's", "she is": "she's", "they are": "they're",
    "we are": "we're", "you are": "you're", "I am": "I'm",
    "is not": "isn't", "are not": "aren't", "was not": "wasn't",
    "were not": "weren't", "do not": "don't", "does not": "doesn't",
    "did not": "didn't", "will not": "won't", "would not": "wouldn't",
    "could not": "couldn't", "should not": "shouldn't", "can not": "can't",
    "cannot": "can't", "have not": "haven't", "has not": "hasn't",
    "had not": "hadn't", "I will": "I'll", "you will": "you'll",
    "he will": "he'll", "she will": "she'll", "they will": "they'll",
    "we will": "we'll", "I would": "I'd", "you would": "you'd",
    "he would": "he'd", "she would": "she'd", "they would": "they'd",
    "we would": "we'd", "I have": "I've", "you have": "you've",
    "we have": "we've", "they have": "they've",
}

# Simpler alternatives for overused AI words
SIMPLIFY_MAP = {
    "utilize": "use", "utilizes": "uses", "utilizing": "using",
    "facilitate": "help", "facilitates": "helps",
    "implement": "set up", "implements": "sets up",
    "leverage": "use", "leverages": "uses", "leveraging": "using",
    "optimize": "improve", "optimizes": "improves",
    "comprehensive": "thorough", "subsequently": "then",
    "furthermore": "also", "moreover": "also",
    "nevertheless": "still", "nonetheless": "still",
    "consequently": "so", "additionally": "also",
    "therefore": "so", "hence": "so",
    "commence": "start", "commences": "starts",
    "terminate": "end", "terminates": "ends",
    "endeavor": "try", "endeavors": "tries",
    "ascertain": "find out", "elucidate": "explain",
    "demonstrate": "show", "demonstrates": "shows",
    "indication": "sign", "functionality": "feature",
    "methodology": "method", "paradigm": "model",
    "innovative": "new", "revolutionary": "new",
    "cutting-edge": "modern", "state-of-the-art": "modern",
    "game-changer": "breakthrough",
}


def apply_contractions(text: str) -> str:
    """Convert formal phrases to contractions for natural tone."""
    result = text
    for formal, contraction in CONTRACTIONS.items():
        # Case-insensitive replacement preserving sentence position
        pattern = re.compile(re.escape(formal), re.IGNORECASE)
        result = pattern.sub(contraction, result)
    return result


def simplify_vocabulary(text: str) -> str:
    """Replace overly formal/AI-typical words with simpler alternatives."""
    result = text
    for formal, simple in SIMPLIFY_MAP.items():
        pattern = re.compile(r"\b" + re.escape(formal) + r"\b", re.IGNORECASE)
        result = pattern.sub(simple, result)
    return result


def remove_filler_phrases(text: str) -> str:
    """Remove common AI filler phrases."""
    result = text
    for phrase in AI_FILLER_PHRASES:
        pattern = re.compile(re.escape(phrase) + r"[,.]?\s*", re.IGNORECASE)
        result = pattern.sub("", result)
    # Clean up any double spaces
    result = re.sub(r"\s+", " ", result).strip()
    return result


def vary_sentence_length(sentences: list[str], strength: float = 0.7) -> list[str]:
    """
    Introduce sentence length variation.

    Split some long sentences, occasionally merge short ones.
    """
    result = []
    i = 0
    while i < len(sentences):
        s = sentences[i]
        words = tokenize_words(s)

        # Split long sentences (>25 words) at conjunctions/semicolons
        if len(words) > 25 and random.random() < strength * 0.5:
            # Try splitting at conjunctions
            split_patterns = [
                r"(?<=\w),\s*(?:and|but|so|yet|while|although|because)\s",
                r";\s+",
            ]
            split_done = False
            for pattern in split_patterns:
                parts = re.split(pattern, s, maxsplit=1)
                if len(parts) == 2 and len(parts[0].split()) > 5 and len(parts[1].split()) > 5:
                    # Capitalize second part
                    part2 = parts[1][0].upper() + parts[1][1:] if len(parts[1]) > 1 else parts[1]
                    result.append(parts[0].rstrip(",;") + ".")
                    result.append(part2 if part2.endswith(".") else part2.rstrip(".") + ".")
                    split_done = True
                    break
            if not split_done:
                result.append(s)

        # Merge very short consecutive sentences (< 6 words each)
        elif (
            len(words) < 6
            and i + 1 < len(sentences)
            and len(tokenize_words(sentences[i + 1])) < 6
            and random.random() < strength * 0.4
        ):
            merged = s.rstrip(".!?") + " — " + sentences[i + 1].lstrip()
            # Lowercase the start of the second part if merging
            result.append(merged)
            i += 1  # skip next sentence since we merged it

        else:
            result.append(s)

        i += 1

    return result


def add_human_touches(sentences: list[str], strength: float = 0.7) -> list[str]:
    """
    Add human-like touches: casual starters, parentheticals, hedging.
    """
    result = []
    for i, s in enumerate(sentences):
        # Occasionally add a human-like starter (not too frequently)
        if (
            i > 0
            and i % random.randint(4, 8) == 0
            and random.random() < strength * 0.3
            and not any(s.startswith(starter) for starter in HUMAN_STARTERS if starter)
        ):
            starter = random.choice(HUMAN_STARTERS[1:])  # Skip empty string
            # Don't add starter if sentence already has one
            s = starter + s[0].lower() + s[1:]

        # Occasionally add parenthetical asides
        if random.random() < strength * 0.1 and len(tokenize_words(s)) > 10:
            words = s.split()
            insert_pos = len(words) // 2
            asides = [
                "(and this matters)",
                "(worth noting)",
                "(at least in my view)",
                "(which is significant)",
            ]
            words.insert(insert_pos, random.choice(asides))
            s = " ".join(words)

        result.append(s)

    return result


def restructure_sentences(text: str, strength: float = 0.7) -> str:
    """
    Main sentence restructuring pipeline.

    Applies multiple transformations to make text more human-like
    while preserving meaning.
    """
    # Preserve headings
    lines = text.split("\n")
    processed_lines = []

    for line in lines:
        stripped = line.strip()
        # Preserve markdown headings
        if stripped.startswith("#"):
            processed_lines.append(line)
            continue
        # Preserve empty lines
        if not stripped:
            processed_lines.append(line)
            continue

        # Process paragraph
        sentences = tokenize_sentences(stripped)
        if not sentences:
            processed_lines.append(line)
            continue

        # Apply transformations
        sentences = vary_sentence_length(sentences, strength)
        sentences = add_human_touches(sentences, strength)

        # Rejoin
        paragraph = " ".join(sentences)

        # Apply word-level transforms
        paragraph = apply_contractions(paragraph)
        paragraph = simplify_vocabulary(paragraph)
        paragraph = remove_filler_phrases(paragraph)

        processed_lines.append(paragraph)

    return "\n".join(processed_lines)
