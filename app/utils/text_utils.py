"""NLP helper functions for text processing."""

from __future__ import annotations

import re
import math
import string
from collections import Counter

import nltk

# Ensure required NLTK data is available
_NLTK_PACKAGES = [
    "punkt", "punkt_tab", "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng", "wordnet", "stopwords", "cmudict",
]

for _pkg in _NLTK_PACKAGES:
    try:
        nltk.data.find(f"tokenizers/{_pkg}" if "punkt" in _pkg else _pkg)
    except LookupError:
        nltk.download(_pkg, quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import cmudict, stopwords

_CMU_DICT: dict | None = None
_STOPWORDS: set | None = None


def get_cmu_dict() -> dict:
    global _CMU_DICT
    if _CMU_DICT is None:
        _CMU_DICT = cmudict.dict()
    return _CMU_DICT


def get_stopwords() -> set[str]:
    global _STOPWORDS
    if _STOPWORDS is None:
        _STOPWORDS = set(stopwords.words("english"))
    return _STOPWORDS


def tokenize_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    return sent_tokenize(text)


def tokenize_words(text: str) -> list[str]:
    """Tokenize text into words (lowercased, no punctuation)."""
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t not in string.punctuation and len(t) > 0]


def count_syllables(word: str) -> int:
    """Count syllables in a word using CMU dict with fallback heuristic."""
    word_lower = word.lower().strip()
    d = get_cmu_dict()
    if word_lower in d:
        # Take the first pronunciation
        return len([ph for ph in d[word_lower][0] if ph[-1].isdigit()])
    # Fallback: vowel-group heuristic
    vowels = "aeiou"
    count = 0
    prev_vowel = False
    for ch in word_lower:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    # Handle silent e
    if word_lower.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def is_complex_word(word: str) -> bool:
    """A word is complex if it has 3+ syllables."""
    return count_syllables(word) >= 3


def calculate_word_frequencies(words: list[str]) -> Counter:
    """Calculate word frequency distribution."""
    sw = get_stopwords()
    content_words = [w for w in words if w not in sw and len(w) > 2]
    return Counter(content_words)


def extract_headings(text: str) -> list[tuple[int, str]]:
    """Extract markdown headings from text as (level, text) tuples."""
    headings = []
    for line in text.split("\n"):
        match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if match:
            level = len(match.group(1))
            headings.append((level, match.group(2).strip()))
    return headings


def calculate_keyword_density(text: str, keyword: str) -> float:
    """Calculate keyword density as percentage."""
    words = tokenize_words(text)
    if not words:
        return 0.0
    keyword_lower = keyword.lower()
    # Handle multi-word keywords
    if " " in keyword_lower:
        text_lower = text.lower()
        keyword_count = text_lower.count(keyword_lower)
        total_words = len(words)
        keyword_word_count = len(keyword_lower.split())
        return (keyword_count * keyword_word_count / total_words) * 100 if total_words else 0.0
    count = words.count(keyword_lower)
    return (count / len(words)) * 100


def is_passive_voice(sentence: str) -> bool:
    """Detect passive voice using POS tagging heuristic."""
    try:
        tokens = word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)
        # Look for pattern: form of "be" + past participle (VBN)
        be_forms = {"am", "is", "are", "was", "were", "be", "been", "being"}
        for i in range(len(tagged) - 1):
            if tagged[i][0].lower() in be_forms and tagged[i + 1][1] == "VBN":
                return True
    except Exception:
        pass
    return False


def shannon_entropy(text: str, level: str = "word") -> float:
    """Calculate Shannon entropy at word or character level."""
    if level == "char":
        tokens = list(text.lower())
    else:
        tokens = tokenize_words(text)
    if not tokens:
        return 0.0
    freq = Counter(tokens)
    total = len(tokens)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


# Common AI filler phrases
AI_FILLER_PHRASES = [
    "it is important to note",
    "it's important to note",
    "it is worth noting",
    "it's worth noting",
    "in today's world",
    "in today's digital age",
    "in the ever-evolving",
    "in this article",
    "dive into",
    "let's dive",
    "delve into",
    "let's delve",
    "in conclusion",
    "to sum up",
    "all in all",
    "at the end of the day",
    "it goes without saying",
    "needless to say",
    "in order to",
    "as a matter of fact",
    "it is essential to",
    "it's essential to",
    "plays a crucial role",
    "plays a vital role",
    "it is imperative",
    "navigating the complexities",
    "landscape of",
    "paradigm shift",
    "leverage the power",
    "harness the potential",
    "unlock the potential",
    "game-changer",
    "cutting-edge",
    "state-of-the-art",
    "revolutionize",
    "furthermore",
    "moreover",
    "additionally",
    "consequently",
    "subsequently",
]

TRANSITION_WORDS = [
    "however", "therefore", "furthermore", "moreover", "additionally",
    "consequently", "nevertheless", "nonetheless", "meanwhile", "subsequently",
    "accordingly", "hence", "thus", "indeed", "certainly", "undoubtedly",
    "alternatively", "conversely", "similarly", "likewise", "specifically",
    "particularly", "notably", "significantly", "importantly",
]
