"""Tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)

AI_TEXT = (
    "Artificial intelligence is a rapidly evolving field that has the potential to "
    "revolutionize various industries. It is important to note that AI technologies "
    "are being implemented across multiple sectors. Furthermore, the development of "
    "machine learning algorithms has significantly improved the accuracy of predictive "
    "models. Additionally, natural language processing has made it possible to analyze "
    "large volumes of text data. Moreover, computer vision technologies have enhanced "
    "the ability to process visual information. Consequently, organizations are "
    "increasingly investing in AI solutions to improve operational efficiency."
)


class TestHealthEndpoint:
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("healthy", "degraded")
        assert "version" in data


class TestAnalyzeEndpoint:
    def test_analyze_success(self):
        response = client.post("/analyze", json={"content": AI_TEXT})
        assert response.status_code == 200
        data = response.json()
        assert "detection" in data
        assert "readability" in data
        assert 0 <= data["detection"]["ai_probability"] <= 1

    def test_analyze_short_text_rejected(self):
        response = client.post("/analyze", json={"content": "Too short"})
        assert response.status_code == 422  # Validation error


class TestHumanizeEndpoint:
    def test_humanize_success(self):
        response = client.post("/humanize", json={
            "content": AI_TEXT,
            "strategy": "moderate",
        })
        assert response.status_code == 200
        data = response.json()
        assert "humanized_content" in data
        assert "detection_before" in data
        assert "detection_after" in data
        assert "changes_made" in data

    def test_humanize_with_keywords(self):
        response = client.post("/humanize", json={
            "content": AI_TEXT,
            "strategy": "light",
            "target_keywords": ["artificial intelligence", "machine learning"],
        })
        assert response.status_code == 200
        data = response.json()
        assert "seo_preservation" in data


class TestScoreEndpoint:
    def test_score_comparison(self):
        humanized = AI_TEXT.replace("Furthermore", "Also").replace("Additionally", "Plus")
        response = client.post("/score", json={
            "original_content": AI_TEXT,
            "humanized_content": humanized,
            "target_keywords": ["AI"],
        })
        assert response.status_code == 200
        data = response.json()
        assert "improvement_summary" in data
        assert "seo_preservation" in data


class TestBatchEndpoint:
    def test_batch_processing(self):
        response = client.post("/batch", json={
            "items": [
                {"content": AI_TEXT, "strategy": "light"},
                {"content": AI_TEXT, "strategy": "moderate"},
            ]
        })
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert data["completed"] == 2
        assert len(data["results"]) == 2
