"""Pydantic models for evaluation metrics, predictions, and benchmark summaries."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FieldEvalResult(BaseModel):
    """Detailed evaluation result for one concept field."""

    concept: str
    truth_present: bool
    predicted_present: bool
    is_exact_match: bool
    is_hallucination: bool  # Predicted value when truth is null
    is_null_match: bool  # Predicted null when truth is null
    truth_value: float | int | None = None
    predicted_value: float | int | None = None
    truth_period: str | None = None
    predicted_period: str | None = None


class SampleEvalResult(BaseModel):
    """Evaluation result for one test example."""

    example_id: str
    raw_completion: str
    parsed_json: dict[str, Any] | None = None
    is_valid_json: bool
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int
    field_results: dict[str, FieldEvalResult] = Field(default_factory=dict)
    all_fields_matched: bool = False


class BenchmarkReport(BaseModel):
    """Comprehensive benchmark summary across all test examples."""

    model_name: str
    prompt_version: str
    timestamp: str
    total_examples: int
    valid_json_count: int
    json_validity_rate: float
    overall_exact_match_rate: float

    # Field-level statistics
    field_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)

    # Performance and cost
    latency_p50_seconds: float
    latency_p95_seconds: float
    throughput_tokens_per_second: float
    throughput_samples_per_second: float
    total_cost_usd: float
    cost_per_1k_samples_usd: float
