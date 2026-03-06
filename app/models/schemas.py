"""Pydantic models for request/response schemas."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# --- Enums ---

class HumanizationStrategy(str, Enum):
    LIGHT = "light"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class ContentType(str, Enum):
    BLOG_POST = "blog_post"
    ARTICLE = "article"
    PRODUCT_DESCRIPTION = "product_description"
    TECHNICAL = "technical"
    MARKETING = "marketing"
    GENERAL = "general"


# --- Detection Scores ---

class PerplexityScore(BaseModel):
    score: float = Field(..., description="Perplexity score (lower = more AI-like)")
    normalized: float = Field(..., description="Normalized 0-1 score")
    interpretation: str = Field(..., description="Human-readable interpretation")


class BurstinessScore(BaseModel):
    score: float = Field(..., description="Burstiness score (lower = more AI-like)")
    sentence_length_variance: float = Field(..., description="Variance in sentence lengths")
    vocabulary_richness: float = Field(..., description="Type-token ratio")
    interpretation: str = Field(..., description="Human-readable interpretation")


class EntropyScore(BaseModel):
    word_entropy: float = Field(..., description="Word-level Shannon entropy")
    char_entropy: float = Field(..., description="Character-level Shannon entropy")
    normalized: float = Field(..., description="Normalized 0-1 score")


class PatternAnalysis(BaseModel):
    repetitive_starts: float = Field(..., description="Ratio of sentences with repetitive starts")
    avg_sentence_length: float = Field(..., description="Average sentence length in words")
    sentence_length_std: float = Field(..., description="Standard deviation of sentence lengths")
    passive_voice_ratio: float = Field(..., description="Ratio of passive voice sentences")
    transition_density: float = Field(..., description="Density of transition words")
    filler_phrase_count: int = Field(..., description="Count of common AI filler phrases")


class DetectionResult(BaseModel):
    ai_probability: float = Field(..., description="Probability text is AI-generated (0-1)")
    perplexity: PerplexityScore
    burstiness: BurstinessScore
    entropy: EntropyScore
    pattern_analysis: PatternAnalysis
    verdict: str = Field(..., description="Overall verdict: likely_ai, uncertain, likely_human")
    confidence: float = Field(..., description="Confidence in the verdict (0-1)")


# --- Readability ---

class ReadabilityScores(BaseModel):
    flesch_kincaid_grade: float = Field(..., description="Flesch-Kincaid Grade Level")
    flesch_reading_ease: float = Field(..., description="Flesch Reading Ease score")
    gunning_fog: float = Field(..., description="Gunning Fog Index")
    coleman_liau: float = Field(..., description="Coleman-Liau Index")
    smog_index: float = Field(..., description="SMOG Index")
    avg_sentence_length: float
    avg_syllables_per_word: float
    total_words: int
    total_sentences: int
    interpretation: str


# --- SEO ---

class KeywordAnalysis(BaseModel):
    keyword: str
    density_before: float
    density_after: float
    preserved: bool


class SEOPreservationResult(BaseModel):
    keyword_analysis: list[KeywordAnalysis] = Field(default_factory=list)
    heading_structure_preserved: bool = True
    meta_integrity: bool = True
    overall_seo_score: float = Field(..., description="SEO preservation score 0-100")
    recommendations: list[str] = Field(default_factory=list)


# --- Requests ---

class AnalyzeRequest(BaseModel):
    content: str = Field(..., min_length=50, description="Text content to analyze")
    content_type: ContentType = ContentType.GENERAL


class HumanizeRequest(BaseModel):
    content: str = Field(..., min_length=50, description="Text content to humanize")
    strategy: HumanizationStrategy = HumanizationStrategy.MODERATE
    strength: float = Field(0.7, ge=0.1, le=1.0, description="Humanization strength")
    target_keywords: list[str] = Field(default_factory=list, description="Keywords to preserve")
    content_type: ContentType = ContentType.GENERAL
    preserve_headings: bool = True


class ScoreRequest(BaseModel):
    original_content: str = Field(..., min_length=50)
    humanized_content: str = Field(..., min_length=50)
    target_keywords: list[str] = Field(default_factory=list)


class BatchRequest(BaseModel):
    items: list[HumanizeRequest] = Field(..., min_length=1, max_length=50)


# --- Responses ---

class AnalyzeResponse(BaseModel):
    detection: DetectionResult
    readability: ReadabilityScores
    word_count: int
    sentence_count: int


class HumanizeResponse(BaseModel):
    original_content: str
    humanized_content: str
    detection_before: DetectionResult
    detection_after: DetectionResult
    readability_before: ReadabilityScores
    readability_after: ReadabilityScores
    seo_preservation: SEOPreservationResult
    changes_made: list[str]


class ScoreResponse(BaseModel):
    readability_original: ReadabilityScores
    readability_humanized: ReadabilityScores
    seo_preservation: SEOPreservationResult
    detection_original: DetectionResult
    detection_humanized: DetectionResult
    improvement_summary: str


class BatchItem(BaseModel):
    index: int
    status: str
    result: Optional[HumanizeResponse] = None
    error: Optional[str] = None


class BatchResponse(BaseModel):
    total: int
    completed: int
    failed: int
    results: list[BatchItem]


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    nltk_ready: bool = True
