"""Financial sanity validator for model extraction outputs."""

from __future__ import annotations

import logging

from src.serve.schemas import ExtractedFinancialMetrics, ValidationDetail

logger = logging.getLogger(__name__)


class SanityValidator:
    """Performs post-generation financial domain sanity checks on extracted metrics.

    Detects:
    1. Scale plausibility (unscaled dollar figures vs raw table numbers, EPS scale).
    2. Income statement margin sanity (Net Income vs Revenue, Operating Income vs Revenue).
    3. EPS vs Net Income consistency (sign mismatch, implied share count reasonableness).
    4. Balance sheet consistency (Assets vs Liabilities sanity).
    """

    def __init__(
        self,
        min_unscaled_dollar: float = 10_000.0,
        max_unscaled_dollar: float = 1e14,
        max_eps: float = 500.0,
        min_eps: float = -500.0,
        max_margin_ratio: float = 2.5,
        min_implied_shares: float = 50_000.0,
        max_implied_shares: float = 1e11,
    ) -> None:
        self.min_unscaled_dollar = min_unscaled_dollar
        self.max_unscaled_dollar = max_unscaled_dollar
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.max_margin_ratio = max_margin_ratio
        self.min_implied_shares = min_implied_shares
        self.max_implied_shares = max_implied_shares

    def validate(self, metrics: ExtractedFinancialMetrics) -> ValidationDetail:
        """Run all sanity checks and return a ValidationDetail with pass/fail and details."""
        errors: list[str] = []
        warnings: list[str] = []

        dollar_fields = [
            ("revenue", metrics.revenue),
            ("net_income", metrics.net_income),
            ("operating_income", metrics.operating_income),
            ("operating_cash_flow", metrics.operating_cash_flow),
            ("total_assets", metrics.total_assets),
            ("total_liabilities", metrics.total_liabilities),
        ]

        # 1. Scale Plausibility on Dollar Fields
        for name, metric in dollar_fields:
            if metric is not None and metric.value is not None:
                val = abs(metric.value)
                if 0.0 < val < self.min_unscaled_dollar:
                    errors.append(
                        f"Scale plausibility failed for '{name}': absolute value {val:,.2f} is under "
                        f"{self.min_unscaled_dollar:,.0f}, indicating missing scale multiplier (millions/thousands)."
                    )
                elif val > self.max_unscaled_dollar:
                    errors.append(
                        f"Scale plausibility failed for '{name}': absolute value {val:,.2f} exceeds "
                        f"maximum realistic threshold {self.max_unscaled_dollar:,.0f}."
                    )

        # 2. Diluted EPS Scale Check
        if metrics.diluted_eps is not None and metrics.diluted_eps.value is not None:
            eps_val = metrics.diluted_eps.value
            if eps_val < self.min_eps or eps_val > self.max_eps:
                errors.append(
                    f"Scale plausibility failed for 'diluted_eps': value {eps_val:.2f} is outside "
                    f"realistic per-share bounds [{self.min_eps}, {self.max_eps}]."
                )

        # 3. Income Statement Sanity: Net Income vs Revenue
        rev = metrics.revenue.value if metrics.revenue else None
        net_inc = metrics.net_income.value if metrics.net_income else None
        op_inc = metrics.operating_income.value if metrics.operating_income else None

        if (
            rev is not None
            and rev > 0
            and net_inc is not None
            and abs(net_inc) > self.max_margin_ratio * rev
        ):
            errors.append(
                f"Margin sanity violation: |net_income| ({abs(net_inc):,.0f}) exceeds "
                f"{self.max_margin_ratio}x revenue ({rev:,.0f})."
            )

        if (
            rev is not None
            and rev > 0
            and op_inc is not None
            and abs(op_inc) > self.max_margin_ratio * rev
        ):
            errors.append(
                f"Operating margin sanity violation: |operating_income| ({abs(op_inc):,.0f}) exceeds "
                f"{self.max_margin_ratio}x revenue ({rev:,.0f})."
            )

        # 4. EPS vs Net Income Consistency
        eps = metrics.diluted_eps.value if metrics.diluted_eps else None
        if net_inc is not None and eps is not None:
            # Sign consistency check (allowing small noise around zero)
            if net_inc > 1_000_000 and eps < -0.05:
                errors.append(
                    f"Sign mismatch: net_income is positive ({net_inc:,.0f}) but diluted_eps is negative ({eps:.2f})."
                )
            elif net_inc < -1_000_000 and eps > 0.05:
                errors.append(
                    f"Sign mismatch: net_income is negative ({net_inc:,.0f}) but diluted_eps is positive ({eps:.2f})."
                )
            elif eps != 0:
                # Implied Shares check
                implied_shares = net_inc / eps
                if implied_shares > 0:
                    if implied_shares < self.min_implied_shares:
                        errors.append(
                            f"Implied shares inconsistency: net_income / eps yields {implied_shares:,.0f} shares, "
                            f"below realistic minimum {self.min_implied_shares:,.0f} (probable scale mismatch)."
                        )
                    elif implied_shares > self.max_implied_shares:
                        errors.append(
                            f"Implied shares inconsistency: net_income / eps yields {implied_shares:,.0f} shares, "
                            f"exceeding realistic maximum {self.max_implied_shares:,.0f}."
                        )

        # 5. Balance Sheet Sanity: Assets vs Liabilities
        assets = metrics.total_assets.value if metrics.total_assets else None
        liab = metrics.total_liabilities.value if metrics.total_liabilities else None

        if assets is not None and assets <= 0:
            errors.append(
                f"Balance sheet violation: total_assets must be positive ({assets:,.0f})."
            )

        if liab is not None and liab < 0:
            errors.append(
                f"Balance sheet violation: total_liabilities cannot be negative ({liab:,.0f})."
            )

        if assets is not None and liab is not None and assets > 0:
            if liab > 5.0 * assets:
                errors.append(
                    f"Balance sheet sanity violation: total_liabilities ({liab:,.0f}) is more than 5x "
                    f"total_assets ({assets:,.0f}), indicating scale or concept confusion."
                )
            elif liab > 1.5 * assets:
                warnings.append(
                    f"High leverage flagged: total_liabilities ({liab:,.0f}) exceeds total_assets ({assets:,.0f})."
                )

        passed = len(errors) == 0
        return ValidationDetail(passed=passed, errors=errors, warnings=warnings)
