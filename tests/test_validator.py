"""Unit tests for the financial post-generation sanity validator."""

from __future__ import annotations

import pytest

from src.serve.schemas import ExtractedFinancialMetrics, MetricValue
from src.serve.validator import SanityValidator


@pytest.fixture
def validator() -> SanityValidator:
    return SanityValidator()


def test_validator_valid_metrics(validator: SanityValidator) -> None:
    """A realistic set of public company metrics should pass all sanity checks."""
    metrics = ExtractedFinancialMetrics(
        revenue=MetricValue(value=383_285_000_000.0, period_type="annual"),
        net_income=MetricValue(value=96_995_000_000.0, period_type="annual"),
        diluted_eps=MetricValue(value=6.13, period_type="annual"),
        total_assets=MetricValue(value=352_583_000_000.0, period_type="instant"),
        operating_income=MetricValue(value=114_301_000_000.0, period_type="annual"),
        total_liabilities=MetricValue(value=290_437_000_000.0, period_type="instant"),
        operating_cash_flow=MetricValue(value=110_543_000_000.0, period_type="annual"),
    )

    result = validator.validate(metrics)
    assert result.passed is True
    assert len(result.errors) == 0


def test_validator_detects_unscaled_dollar_value(validator: SanityValidator) -> None:
    """If revenue is extracted as 383.28 without applying million multiplier, it must fail."""
    metrics = ExtractedFinancialMetrics(
        revenue=MetricValue(value=383.28, period_type="annual"),
        net_income=MetricValue(value=96.99, period_type="annual"),
    )

    result = validator.validate(metrics)
    assert result.passed is False
    assert any("Scale plausibility failed for 'revenue'" in err for err in result.errors)
    assert any("Scale plausibility failed for 'net_income'" in err for err in result.errors)


def test_validator_detects_eps_scale_error(validator: SanityValidator) -> None:
    """Diluted EPS scaled to thousands or millions must fail."""
    metrics = ExtractedFinancialMetrics(
        diluted_eps=MetricValue(value=6130.0, period_type="annual"),
    )

    result = validator.validate(metrics)
    assert result.passed is False
    assert any("diluted_eps" in err for err in result.errors)


def test_validator_detects_excessive_margin_sanity(validator: SanityValidator) -> None:
    """Net income exceeding 2.5x revenue must fail margin sanity."""
    metrics = ExtractedFinancialMetrics(
        revenue=MetricValue(value=100_000_000.0, period_type="annual"),
        net_income=MetricValue(value=500_000_000.0, period_type="annual"),  # 5x revenue
    )

    result = validator.validate(metrics)
    assert result.passed is False
    assert any("Margin sanity violation" in err for err in result.errors)


def test_validator_detects_sign_mismatch_between_eps_and_income(validator: SanityValidator) -> None:
    """Positive net income with negative EPS must trigger an error."""
    metrics = ExtractedFinancialMetrics(
        net_income=MetricValue(value=50_000_000.0, period_type="annual"),
        diluted_eps=MetricValue(value=-2.50, period_type="annual"),
    )

    result = validator.validate(metrics)
    assert result.passed is False
    assert any("Sign mismatch" in err for err in result.errors)


def test_validator_detects_implied_shares_inconsistency(validator: SanityValidator) -> None:
    """Net income / EPS yielding unrealistic share count indicates scaling confusion."""
    # $100M / $5.00 EPS = 20M shares (reasonable)
    # $100,000 / $5.00 EPS = 20,000 shares (< 50,000 min)
    metrics = ExtractedFinancialMetrics(
        net_income=MetricValue(value=100_000.0, period_type="annual"),
        diluted_eps=MetricValue(value=5.00, period_type="annual"),
    )

    result = validator.validate(metrics)
    assert result.passed is False
    assert any("Implied shares inconsistency" in err for err in result.errors)


def test_validator_detects_balance_sheet_violation(validator: SanityValidator) -> None:
    """Total assets cannot be non-positive, and liabilities > 5x assets triggers error."""
    metrics = ExtractedFinancialMetrics(
        total_assets=MetricValue(value=-100_000_000.0, period_type="instant"),
    )

    result = validator.validate(metrics)
    assert result.passed is False
    assert any("total_assets must be positive" in err for err in result.errors)

    metrics_excessive = ExtractedFinancialMetrics(
        total_assets=MetricValue(value=10_000_000.0, period_type="instant"),
        total_liabilities=MetricValue(value=100_000_000.0, period_type="instant"),  # 10x assets
    )
    result_excessive = validator.validate(metrics_excessive)
    assert result_excessive.passed is False
    assert any("Balance sheet sanity violation" in err for err in result_excessive.errors)
