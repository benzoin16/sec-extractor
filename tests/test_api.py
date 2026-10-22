"""Unit and integration tests for EdgarLENS FastAPI gateway endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.serve.app import app
from src.serve.client import ModelExtractionResult
from src.serve.schemas import (
    ExtractedFinancialMetrics,
    MetricValue,
    ValidationDetail,
)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def sample_metrics() -> ExtractedFinancialMetrics:
    return ExtractedFinancialMetrics(
        revenue=MetricValue(value=383_285_000_000.0, period_type="annual"),
        net_income=MetricValue(value=96_995_000_000.0, period_type="annual"),
        diluted_eps=MetricValue(value=6.13, period_type="annual"),
        total_assets=MetricValue(value=352_583_000_000.0, period_type="instant"),
        operating_income=MetricValue(value=114_301_000_000.0, period_type="annual"),
        total_liabilities=MetricValue(value=290_437_000_000.0, period_type="instant"),
        operating_cash_flow=MetricValue(value=110_543_000_000.0, period_type="annual"),
    )


def test_health_endpoint(client: TestClient) -> None:
    """GET /health should return 200 with status information."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_name" in data
    assert "fallback_available" in data


def test_metrics_endpoint(client: TestClient) -> None:
    """GET /metrics should export Prometheus metrics."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    text = response.text
    assert "sec_extractor_requests_total" in text or "sec_extractor_latency_seconds" in text


def test_extract_endpoint_with_raw_text(
    client: TestClient, sample_metrics: ExtractedFinancialMetrics
) -> None:
    """POST /extract with raw text should return structured response."""
    mock_res = ModelExtractionResult(
        metrics=sample_metrics,
        validation=ValidationDetail(passed=True, errors=[], warnings=[]),
        model_name="qwen2.5-7b-instruct-sec",
        fallback_used=False,
        tokens_prompt=150,
        tokens_completion=80,
        latency_ms=125.5,
        raw_output="{}",
    )

    with (
        patch("src.serve.app.inference_client", None),
        patch(
            "src.serve.client.ExtractionInferenceClient.extract", new_callable=AsyncMock
        ) as mock_extract,
    ):
        mock_extract.return_value = mock_res

        payload = {
            "text": "Total net sales: $ 383,285 million, net income: $ 96,995 million",
            "allow_fallback": True,
            "strict_validation": True,
        }
        response = client.post("/extract", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["model"] == "qwen2.5-7b-instruct-sec"
        assert data["fallback_used"] is False
        assert data["validation"]["passed"] is True
        assert data["metrics"]["revenue"]["value"] == 383_285_000_000.0
        assert data["metrics"]["diluted_eps"]["value"] == 6.13
        assert data["latency_ms"] > 0


def test_extract_endpoint_with_accession(
    client: TestClient, sample_metrics: ExtractedFinancialMetrics
) -> None:
    """POST /extract with accession_number should trigger retrieval pipeline."""
    mock_res = ModelExtractionResult(
        metrics=sample_metrics,
        validation=ValidationDetail(passed=True, errors=[], warnings=[]),
        model_name="qwen2.5-7b-instruct-sec",
        fallback_used=False,
        tokens_prompt=200,
        tokens_completion=90,
        latency_ms=210.0,
        raw_output="{}",
    )

    with (
        patch("src.serve.app.accession_pipeline", None),
        patch("src.serve.app.inference_client", None),
        patch(
            "src.serve.pipeline.AccessionPipeline.get_text_for_accession", new_callable=AsyncMock
        ) as mock_get_text,
        patch(
            "src.serve.client.ExtractionInferenceClient.extract", new_callable=AsyncMock
        ) as mock_extract,
    ):
        mock_get_text.return_value = "Chunk text containing revenue $ 383,285 million"
        mock_extract.return_value = mock_res

        payload = {
            "accession_number": "0000320193-24-000106",
            "ticker": "AAPL",
        }
        response = client.post("/extract", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["metrics"]["revenue"]["value"] == 383_285_000_000.0
        mock_get_text.assert_called_once_with(
            accession_number="0000320193-24-000106", ticker="AAPL"
        )


def test_extract_endpoint_validation_error_on_empty(client: TestClient) -> None:
    """Payload without text or accession must return 422 Unprocessable Entity."""
    response = client.post("/extract", json={})
    assert response.status_code == 422
