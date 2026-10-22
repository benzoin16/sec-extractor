"""Numeric presence verification and period disambiguation in chunk text."""

from __future__ import annotations

import re

from src.labels.models import ConceptValue, FilingLabels, TargetMetric


# Regex to safely search for numeric representations with boundaries
def _build_num_regex(pattern_str: str, is_negative: bool) -> re.Pattern:
    escaped = re.escape(pattern_str)
    if is_negative:
        # Match e.g. (1,234) or -1,234 or (1234)
        return re.compile(rf"(?:^|[^\d])(?:\({escaped}\)|-{escaped})(?:[^\d]|$)")
    else:
        # Match e.g. $1,234 or 1,234 or 1234
        return re.compile(rf"(?:^|[^\d]){escaped}(?:[^\d]|$)")


class NumericMatcher:
    """Verifies presence of numeric ground-truth facts in chunk text across scales.

    Supports unscaled, thousands (1e3), millions (1e6), billions (1e9),
    per-share decimals (EPS), and negative parenthesized amounts: (1,234).
    """

    def __init__(self) -> None:
        self.skipped_ambiguous_count = 0

    @classmethod
    def check_value_in_text(
        cls,
        val: float | int,
        text: str,
        is_eps: bool = False,
    ) -> bool:
        """Return True if val (or its scaled/formatted equivalent) appears in text."""
        if val is None:
            return False

        is_negative = val < 0
        abs_val = abs(val)

        # EPS handling: 2 decimal floats, e.g. 6.13, 0.45
        if is_eps:
            eps_patterns = [
                f"{abs_val:.2f}",
                f"{abs_val:.1f}" if abs_val == round(abs_val, 1) else "",
            ]
            for pat in eps_patterns:
                if pat:
                    reg = _build_num_regex(pat, is_negative)
                    if reg.search(text):
                        return True
            return False

        # Large currency figures (USD): test unscaled, 1e6, 1e3, 1e9
        patterns: set[str] = set()

        # 1. Exact unscaled
        int_abs = int(round(abs_val))
        patterns.add(str(int_abs))
        patterns.add(f"{int_abs:,}")

        # 2. Millions scale (10^6)
        if abs_val >= 1e5:
            scaled_m = abs_val / 1e6
            m_rounded = int(round(scaled_m))
            patterns.add(str(m_rounded))
            patterns.add(f"{m_rounded:,}")
            # 1 or 2 decimals if not whole
            if round(scaled_m, 1) != m_rounded:
                patterns.add(f"{scaled_m:.1f}")
                patterns.add(f"{scaled_m:.2f}")

        # 3. Thousands scale (10^3)
        if abs_val >= 1e3:
            scaled_k = abs_val / 1e3
            k_rounded = int(round(scaled_k))
            patterns.add(str(k_rounded))
            patterns.add(f"{k_rounded:,}")

        # 4. Billions scale (10^9)
        if abs_val >= 1e8:
            scaled_b = abs_val / 1e9
            patterns.add(f"{scaled_b:.1f}")
            patterns.add(f"{scaled_b:.2f}")
            patterns.add(f"{scaled_b:.3f}")

        for pat in patterns:
            # Skip tiny 1-2 digit false-positive patterns unless unscaled number is genuinely that small
            if len(pat) < 2 and abs_val >= 1e4:
                continue
            reg = _build_num_regex(pat, is_negative)
            if reg.search(text):
                return True

        return False

    def match_chunk(
        self,
        chunk_text: str,
        filing_labels: FilingLabels,
        target_concepts: list[str],
    ) -> tuple[dict[str, TargetMetric | None], bool]:
        """Match concepts against chunk text.

        Returns:
            Tuple of (matched_metrics, was_skipped_due_to_ambiguity).
            If was_skipped_due_to_ambiguity is True, the example should be discarded.
        """
        matched: dict[str, TargetMetric | None] = {}

        for concept in target_concepts:
            candidates: list[ConceptValue] = filing_labels.metrics.get(concept, [])
            if not candidates:
                matched[concept] = None
                continue

            # Check which candidate values actually appear in the chunk text
            is_eps = concept == "diluted_eps"
            present_candidates = [
                c
                for c in candidates
                if self.check_value_in_text(c.value, chunk_text, is_eps=is_eps)
            ]

            if not present_candidates:
                matched[concept] = None
                continue

            if len(present_candidates) == 1:
                cand = present_candidates[0]
                matched[concept] = TargetMetric(value=cand.value, period_type=cand.period_type)
                continue

            # Multiple candidates present in chunk (e.g. 10-Q quarter vs YTD)
            # Check for ambiguity: if values are identical, we cannot distinguish
            values = {c.value for c in present_candidates}
            if len(values) == 1:
                # Ambiguous match: same number could be quarter or YTD
                self.skipped_ambiguous_count += 1
                return {}, True

            # Values are distinct: apply disambiguation rules
            # Income statement items -> prefer quarter
            # Cash flow items -> prefer YTD
            if concept == "operating_cash_flow":
                ytd_cand = next((c for c in present_candidates if c.period_type == "ytd"), None)
                chosen = ytd_cand or present_candidates[0]
            else:
                quarter_cand = next(
                    (c for c in present_candidates if c.period_type == "quarter"), None
                )
                chosen = quarter_cand or present_candidates[0]

            matched[concept] = TargetMetric(value=chosen.value, period_type=chosen.period_type)

        return matched, False
