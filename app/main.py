"""FastAPI application for AI Content Humanizer."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import nltk
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    HumanizeRequest,
    HumanizeResponse,
    ScoreRequest,
    ScoreResponse,
    BatchRequest,
    BatchResponse,
    BatchItem,
    HealthResponse,
)
from app.services.analyzer import analyze_text
from app.services.humanizer import humanize_content
from app.services.readability import calculate_readability
from app.services.seo_preserver import analyze_seo_preservation
from app.utils.text_utils import tokenize_words, tokenize_sentences


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Download required NLTK data on startup."""
    packages = [
        "punkt", "punkt_tab", "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng", "wordnet", "stopwords", "cmudict",
    ]
    for pkg in packages:
        nltk.download(pkg, quiet=True)
    yield


app = FastAPI(
    title="AI Content Humanizer",
    description=(
        "Analyze AI-generated text and humanize it while preserving SEO value. "
        "Features AI detection scoring (perplexity, burstiness, entropy), "
        "humanization pipeline, SEO preservation, and readability metrics."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check service health and NLTK readiness."""
    nltk_ready = True
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk_ready = False

    return HealthResponse(
        status="healthy" if nltk_ready else "degraded",
        version="1.0.0",
        nltk_ready=nltk_ready,
    )


# ──────────────────────────────────────────────
# Analyze
# ──────────────────────────────────────────────

@app.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze_content(request: AnalyzeRequest):
    """
    Detect AI patterns in text content.

    Returns detection scores including perplexity, burstiness,
    entropy, pattern analysis, and an overall AI probability score.
    """
    try:
        detection = analyze_text(request.content)
        readability = calculate_readability(request.content)
        words = tokenize_words(request.content)
        sentences = tokenize_sentences(request.content)

        return AnalyzeResponse(
            detection=detection,
            readability=readability,
            word_count=len(words),
            sentence_count=len(sentences),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ──────────────────────────────────────────────
# Humanize
# ──────────────────────────────────────────────

@app.post("/humanize", response_model=HumanizeResponse, tags=["Humanization"])
async def humanize(request: HumanizeRequest):
    """
    Humanize AI-generated content with configurable strategies.

    Strategies:
    - **light**: Minimal changes — contractions and vocabulary only
    - **moderate**: Balanced — sentence restructuring + vocabulary
    - **aggressive**: Maximum humanization — all transforms at full strength

    Returns before/after detection scores, readability metrics,
    and SEO preservation analysis.
    """
    if len(request.content) > settings.max_content_length:
        raise HTTPException(
            status_code=400,
            detail=f"Content exceeds maximum length of {settings.max_content_length} characters",
        )
    try:
        result = humanize_content(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Humanization failed: {str(e)}")


# ──────────────────────────────────────────────
# Score
# ──────────────────────────────────────────────

@app.post("/score", response_model=ScoreResponse, tags=["Scoring"])
async def score_content(request: ScoreRequest):
    """
    Compare readability, AI detection, and SEO preservation
    between original and humanized content.
    """
    try:
        readability_orig = calculate_readability(request.original_content)
        readability_hum = calculate_readability(request.humanized_content)
        detection_orig = analyze_text(request.original_content)
        detection_hum = analyze_text(request.humanized_content)
        seo = analyze_seo_preservation(
            request.original_content,
            request.humanized_content,
            request.target_keywords,
        )

        # Build improvement summary
        ai_change = detection_orig.ai_probability - detection_hum.ai_probability
        readability_change = (
            readability_hum.flesch_reading_ease - readability_orig.flesch_reading_ease
        )

        parts = []
        if ai_change > 0.1:
            parts.append(
                f"AI probability reduced by {ai_change:.0%} "
                f"({detection_orig.ai_probability:.0%} → {detection_hum.ai_probability:.0%})"
            )
        elif ai_change < -0.05:
            parts.append(f"Warning: AI probability increased by {abs(ai_change):.0%}")
        else:
            parts.append("AI probability roughly unchanged")

        if readability_change > 5:
            parts.append(f"Readability improved by {readability_change:.1f} points")
        elif readability_change < -5:
            parts.append(f"Readability decreased by {abs(readability_change):.1f} points")

        parts.append(f"SEO preservation score: {seo.overall_seo_score:.0f}/100")

        return ScoreResponse(
            readability_original=readability_orig,
            readability_humanized=readability_hum,
            seo_preservation=seo,
            detection_original=detection_orig,
            detection_humanized=detection_hum,
            improvement_summary=". ".join(parts),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")


# ──────────────────────────────────────────────
# Batch
# ──────────────────────────────────────────────

@app.post("/batch", response_model=BatchResponse, tags=["Batch Processing"])
async def batch_humanize(request: BatchRequest):
    """
    Batch process multiple texts for humanization.

    Processes up to 50 items asynchronously.
    """
    if len(request.items) > settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds maximum of {settings.max_batch_size}",
        )

    results = []
    completed = 0
    failed = 0

    async def process_item(index: int, item: HumanizeRequest) -> BatchItem:
        nonlocal completed, failed
        try:
            result = humanize_content(item)
            completed += 1
            return BatchItem(index=index, status="completed", result=result)
        except Exception as e:
            failed += 1
            return BatchItem(index=index, status="failed", error=str(e))

    # Process concurrently using asyncio
    tasks = [
        process_item(i, item) for i, item in enumerate(request.items)
    ]
    results = await asyncio.gather(*tasks)

    return BatchResponse(
        total=len(request.items),
        completed=completed,
        failed=failed,
        results=sorted(results, key=lambda r: r.index),
    )
