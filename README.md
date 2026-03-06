# AI Content Humanizer

A production-ready FastAPI service that **detects AI-generated text** and **humanizes it** while preserving SEO value. Uses statistical NLP analysis — no paid API keys required.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Application                    │
├──────────┬──────────┬───────────┬──────────┬────────────┤
│ /analyze │/humanize │  /score   │  /batch  │  /health   │
├──────────┴──────────┴───────────┴──────────┴────────────┤
│                   Service Layer                          │
│  ┌──────────┐ ┌───────────┐ ┌────────────┐ ┌─────────┐ │
│  │ Analyzer │ │ Humanizer │ │SEO Preserver│ │Readabil.│ │
│  └────┬─────┘ └─────┬─────┘ └─────┬──────┘ └────┬────┘ │
├───────┴─────────────┴─────────────┴──────────────┴──────┤
│                     Core Engine                          │
│  ┌───────────┐ ┌───────────┐ ┌──────────┐ ┌──────────┐ │
│  │Perplexity │ │Burstiness │ │Sentence  │ │Vocabulary│ │
│  │& Entropy  │ │& Patterns │ │Transforms│ │Diversify │ │
│  └───────────┘ └───────────┘ └──────────┘ └──────────┘ │
├─────────────────────────────────────────────────────────┤
│              NLTK · NumPy · scikit-learn                 │
└─────────────────────────────────────────────────────────┘
```

## Features

### AI Detection Scoring
- **Perplexity analysis** — bigram language model measures text predictability
- **Burstiness measurement** — sentence length variation (AI text is uniform)
- **Shannon entropy** — word and character-level information density
- **Pattern detection** — repetitive sentence starts, filler phrases, transition overuse, passive voice

### Humanization Pipeline
- **Sentence restructuring** — split long sentences, merge short ones, vary patterns
- **Vocabulary diversification** — replace repeated words with contextual synonyms
- **Contraction insertion** — convert formal phrasing to conversational tone
- **Filler phrase removal** — strip "it is important to note", "furthermore", etc.
- **Rhythm variation** — add emphatic short sentences and parenthetical asides

### SEO Preservation
- **Keyword density monitoring** — tracks target keywords before/after humanization
- **Heading structure preservation** — maintains H1-H6 hierarchy
- **Content length tracking** — flags significant length changes
- **Meta tag integrity** — preserves template markers and structured data

### Readability Metrics
- Flesch-Kincaid Grade Level & Reading Ease
- Gunning Fog Index
- Coleman-Liau Index
- SMOG Index

## Quick Start

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data (auto-downloads on first run)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('cmudict')"

# Run the server
uvicorn app.main:app --reload

# Open API docs
open http://localhost:8000/docs
```

### Docker
```bash
docker-compose up --build
# API available at http://localhost:8000
```

### Run Tests
```bash
pytest tests/ -v
```

## API Documentation

### `POST /analyze` — Detect AI Patterns
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Artificial intelligence is a rapidly evolving field that has the potential to revolutionize various industries. It is important to note that AI technologies are being implemented across multiple sectors. Furthermore, the development of machine learning algorithms has significantly improved the accuracy of predictive models."
  }'
```

**Response:**
```json
{
  "detection": {
    "ai_probability": 0.72,
    "verdict": "likely_ai",
    "confidence": 0.72,
    "perplexity": { "score": 28.5, "normalized": 0.22, "interpretation": "..." },
    "burstiness": { "score": 0.31, "sentence_length_variance": 8.2, "..." },
    "entropy": { "word_entropy": 7.8, "char_entropy": 4.1, "normalized": 0.56 },
    "pattern_analysis": { "filler_phrase_count": 2, "transition_density": 0.04, "..." }
  },
  "readability": { "flesch_reading_ease": 32.5, "flesch_kincaid_grade": 14.2, "..." }
}
```

### `POST /humanize` — Humanize Content
```bash
curl -X POST http://localhost:8000/humanize \
  -H "Content-Type: application/json" \
  -d '{
    "content": "It is important to note that artificial intelligence is revolutionizing the industry. Furthermore, organizations are leveraging cutting-edge solutions to optimize their operations. Additionally, the implementation of machine learning has demonstrated significant improvements in efficiency.",
    "strategy": "moderate",
    "strength": 0.7,
    "target_keywords": ["artificial intelligence", "machine learning"]
  }'
```

### `POST /score` — Compare Before/After
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "original_content": "Original AI text here...",
    "humanized_content": "Humanized version here...",
    "target_keywords": ["SEO", "optimization"]
  }'
```

### `POST /batch` — Batch Processing
```bash
curl -X POST http://localhost:8000/batch \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      { "content": "First text to humanize...", "strategy": "light" },
      { "content": "Second text to humanize...", "strategy": "aggressive" }
    ]
  }'
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | FastAPI |
| NLP | NLTK (tokenization, POS tagging, CMU dict) |
| Math | NumPy, scikit-learn |
| Validation | Pydantic v2 |
| Testing | pytest |
| Containerization | Docker |

## How It Works

### Detection Algorithm
The AI probability score combines multiple weighted signals:
- **Perplexity** (25%) — low perplexity indicates predictable, AI-like text
- **Burstiness** (20%) — low variation in sentence lengths signals AI
- **Entropy** (15%) — low information density suggests template-based generation
- **Repetitive starts** (10%) — AI tends to begin sentences similarly
- **Length uniformity** (10%) — AI produces sentences of similar length
- **Transition density** (10%) — AI overuses "furthermore", "additionally", etc.
- **Filler phrases** (10%) — phrases like "it is important to note" are AI markers

### Humanization Strategies
| Strategy | Strength | Changes |
|----------|----------|---------|
| Light | 40% | Contractions, vocabulary simplification |
| Moderate | 70% | + Sentence restructuring, rhythm variation |
| Aggressive | 100% | + Maximum variation, emphatic phrases |

## License

MIT
