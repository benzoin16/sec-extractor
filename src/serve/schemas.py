"""Pydantic request and response models for EdgarLENS extraction service."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class MetricValue(BaseModel):
    """Extracted numeric value for a single concept."""

    value: float = Field(..., description="Unscaled numeric amount, e.g. 383285000000.0 or 6.16")
    period_type: Literal["annual", "quarter", "ytd", "instant"] | None = Field(
        default=None,
        description="Filing reporting period type",
    )


class ExtractedFinancialMetrics(BaseModel):
    """Structured extraction output schema matching SEC EDGAR ground truth concepts."""

    revenue: MetricValue | None = Field(
        default=None, description="Total revenue or sales from the income statement"
    )
    net_income: MetricValue | None = Field(
        default=None, description="Net income or net loss attributable to common shareholders"
    )
    diluted_eps: MetricValue | None = Field(
        default=None, description="Diluted earnings per share (EPS)"
    )
    total_assets: MetricValue | None = Field(
        default=None, description="Total assets from the balance sheet"
    )
    operating_income: MetricValue | None = Field(
        default=None, description="Operating income or operating profit"
    )
    total_liabilities: MetricValue | None = Field(
        default=None, description="Total liabilities from the balance sheet"
    )
    operating_cash_flow: MetricValue | None = Field(
        default=None, description="Net cash provided by or used in operating activities"
    )


class ExtractRequest(BaseModel):
    """Input request payload: either raw filing text or an accession number."""

    text: str | None = Field(
        default=None,
        description="Raw SEC filing text or markdown table chunk to extract from.",
    )
    accession_number: str | None = Field(
        default=None,
        description="SEC accession number (e.g. 0000320193-24-000106) to fetch/parse from cache.",
    )
    ticker: str | None = Field(
        default=None,
        description="Optional company ticker if accession lookup requires CIK resolution.",
    )
    allow_fallback: bool = Field(
        default=True,
        description="Route to frontier model if local vLLM model fails sanity validation.",
    )
    strict_validation: bool = Field(
        default=True,
        description="Reject extractions that fail scale or balance sheet sanity checks.",
    )

    @model_validator(mode="after")
    def check_at_least_one_source(self) -> ExtractRequest:
        if not self.text and not self.accession_number:
            raise ValueError("Either 'text' or 'accession_number' must be provided.")
        return self


class ValidationDetail(BaseModel):
    """Result of post-generation sanity checks."""

    passed: bool = Field(..., description="Whether all financial sanity checks passed")
    errors: list[str] = Field(default_factory=list, description="Sanity violations if failed")
    warnings: list[str] = Field(default_factory=list, description="Non-fatal anomalies flagged")


class ExtractResponse(BaseModel):
    """Complete extraction response with metrics, validation, latency, and trace metadata."""

    metrics: ExtractedFinancialMetrics
    validation: ValidationDetail
    latency_ms: float = Field(..., description="Total extraction latency in milliseconds")
    model: str = Field(..., description="Model identifier used for the final extraction")
    fallback_used: bool = Field(
        default=False, description="True if validation failure triggered frontier fallback"
    )
    trace_id: str | None = Field(default=None, description="Langfuse trace identifier")
    tokens_prompt: int | None = Field(default=None, description="Number of prompt tokens")
    tokens_completion: int | None = Field(
        default=None, description="Number of generated completion tokens"
    )


class HealthResponse(BaseModel):
    """System health and readiness status."""

    status: Literal["healthy", "degraded", "unhealthy"]
    vllm_reachable: bool
    model_name: str
    fallback_available: bool
