"""Prometheus observability metrics for the EdgarLENS extraction service."""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# Request Latency Histogram
REQUEST_LATENCY = Histogram(
    "sec_extractor_latency_seconds",
    "End-to-end request latency in seconds",
    ["endpoint", "model", "status"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
)

# Total Requests Counter
REQUESTS_TOTAL = Counter(
    "sec_extractor_requests_total",
    "Total HTTP requests handled by the service",
    ["endpoint", "model", "status"],
)

# Fallbacks Counter
FALLBACKS_TOTAL = Counter(
    "sec_extractor_fallbacks_total",
    "Total requests routed to fallback frontier model",
    ["reason"],
)

# Token Usage Counter
TOKENS_TOTAL = Counter(
    "sec_extractor_tokens_total",
    "Total tokens processed by models",
    ["model", "type"],  # type: 'prompt' or 'completion'
)

# Validation Failures Counter
VALIDATION_FAILURES_TOTAL = Counter(
    "sec_extractor_validation_failures_total",
    "Total financial sanity check validation failures",
    ["check_type"],  # e.g., 'scale_plausibility', 'margin_sanity', 'eps_net_income'
)


def get_metrics_payload() -> tuple[bytes, str]:
    """Return the Prometheus metrics payload and content type header."""
    return generate_latest(), CONTENT_TYPE_LATEST
