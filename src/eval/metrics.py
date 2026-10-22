"""Evaluation metric calculations with numeric normalization and tolerance."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from src.eval.models import BenchmarkReport, FieldEvalResult, SampleEvalResult

TARGET_CONCEPTS = [
    "revenue",
    "net_income",
    "diluted_eps",
    "total_assets",
    "operating_income",
    "total_liabilities",
    "operating_cash_flow",
]


class NumericNormalizer:
    """Normalizes numeric values and verifies matches within rounding/scale tolerances."""

    @classmethod
    def parse_numeric(cls, raw: Any) -> float | None:
        """Convert raw input (number or formatted string) to float."""
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(raw, str):
            clean = raw.strip().replace("$", "").replace(",", "").replace("\xa0", "")
            # Handle parenthesized negative: (1,234) -> -1234
            paren_match = re.match(r"^\((.+)\)$", clean)
            if paren_match:
                clean = f"-{paren_match.group(1)}"
            try:
                return float(clean)
            except ValueError:
                return None
        return None

    @classmethod
    def values_match(
        cls,
        pred_val: float,
        truth_val: float,
        is_eps: bool = False,
        rel_tol: float = 0.005,
        eps_abs_tol: float = 0.02,
    ) -> bool:
        """Compare predicted and truth values with tolerance and scale resolution."""
        # Check strict sign compatibility
        if (pred_val < 0) != (truth_val < 0) and not (pred_val == 0 and truth_val == 0):
            return False

        if is_eps:
            return abs(pred_val - truth_val) <= eps_abs_tol

        # Exact match
        if pred_val == truth_val:
            return True

        # Relative tolerance on unscaled amounts (e.g. 0.5% tolerance for rounding)
        if truth_val != 0:
            rel_error = abs(pred_val - truth_val) / abs(truth_val)
            if rel_error <= rel_tol:
                return True

        # Scale tolerance: check if model emitted values in millions (1e6) or thousands (1e3)
        for multiplier in (1e6, 1e3, 1e9):
            scaled_pred = pred_val * multiplier
            if truth_val != 0 and abs(scaled_pred - truth_val) / abs(truth_val) <= rel_tol:
                return True

        return False


def parse_prediction_json(completion_text: str) -> dict[str, Any] | None:
    """Parse JSON from completion text, stripping markdown code blocks if present."""
    text = completion_text.strip()
    # Strip markdown ```json ... ``` wrapper if present
    md_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if md_match:
        text = md_match.group(1).strip()

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def evaluate_sample(
    completion_text: str,
    truth_dict: dict[str, Any],
    example_id: str = "",
    latency_seconds: float = 0.0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    rel_tol: float = 0.005,
    eps_abs_tol: float = 0.02,
) -> SampleEvalResult:
    """Evaluate a single completion against ground truth."""
    parsed = parse_prediction_json(completion_text)
    is_valid_json = parsed is not None

    field_results: dict[str, FieldEvalResult] = {}
    all_matched = is_valid_json

    for concept in TARGET_CONCEPTS:
        truth_item = truth_dict.get(concept)
        pred_item = parsed.get(concept) if parsed else None

        truth_present = truth_item is not None
        pred_present = pred_item is not None

        truth_val = None
        truth_period = None
        if truth_present:
            if isinstance(truth_item, dict):
                truth_val = NumericNormalizer.parse_numeric(truth_item.get("value"))
                truth_period = truth_item.get("period_type")
            else:
                truth_val = NumericNormalizer.parse_numeric(truth_item)

        pred_val = None
        pred_period = None
        if pred_present:
            if isinstance(pred_item, dict):
                pred_val = NumericNormalizer.parse_numeric(pred_item.get("value"))
                pred_period = pred_item.get("period_type")
            else:
                pred_val = NumericNormalizer.parse_numeric(pred_item)

        is_eps = concept == "diluted_eps"
        is_exact = False
        is_hallucination = False
        is_null_match = False

        if not truth_present and not pred_present:
            is_exact = True
            is_null_match = True
        elif not truth_present and pred_present:
            is_hallucination = True
            is_exact = False
        elif truth_present and not pred_present:
            is_exact = False
        else:
            # Both present: check numeric value and period match
            val_ok = (
                truth_val is not None
                and pred_val is not None
                and NumericNormalizer.values_match(
                    pred_val, truth_val, is_eps=is_eps, rel_tol=rel_tol, eps_abs_tol=eps_abs_tol
                )
            )
            # Period match: if truth period is specified, check compatibility
            period_ok = truth_period is None or pred_period is None or truth_period == pred_period
            is_exact = val_ok and period_ok

        if not is_exact:
            all_matched = False

        field_results[concept] = FieldEvalResult(
            concept=concept,
            truth_present=truth_present,
            predicted_present=pred_present,
            is_exact_match=is_exact,
            is_hallucination=is_hallucination,
            is_null_match=is_null_match,
            truth_value=truth_val,
            predicted_value=pred_val,
            truth_period=truth_period,
            predicted_period=pred_period,
        )

    return SampleEvalResult(
        example_id=example_id,
        raw_completion=completion_text,
        parsed_json=parsed,
        is_valid_json=is_valid_json,
        latency_seconds=latency_seconds,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        field_results=field_results,
        all_fields_matched=all_matched,
    )


def compute_benchmark_report(
    samples: list[SampleEvalResult],
    model_name: str,
    prompt_version: str,
    timestamp: str,
    prompt_price_per_1k: float = 0.0,
    completion_price_per_1k: float = 0.0,
    total_wall_clock_seconds: float = 1.0,
) -> BenchmarkReport:
    """Aggregate sample results into a BenchmarkReport."""
    total = len(samples)
    if total == 0:
        raise ValueError("Cannot compute benchmark report on empty sample list.")

    valid_json_count = sum(1 for s in samples if s.is_valid_json)
    json_validity_rate = valid_json_count / total
    overall_exact_match = sum(1 for s in samples if s.all_fields_matched) / total

    # Field metrics
    field_metrics: dict[str, dict[str, float]] = {}
    for concept in TARGET_CONCEPTS:
        field_exact = sum(1 for s in samples if s.field_results[concept].is_exact_match)
        total_truth_null = sum(1 for s in samples if not s.field_results[concept].truth_present)
        null_matches = sum(1 for s in samples if s.field_results[concept].is_null_match)
        hallucinations = sum(1 for s in samples if s.field_results[concept].is_hallucination)

        null_accuracy = (null_matches / total_truth_null) if total_truth_null > 0 else 1.0
        hallucination_rate = (hallucinations / total_truth_null) if total_truth_null > 0 else 0.0

        field_metrics[concept] = {
            "exact_match_rate": field_exact / total,
            "null_accuracy": null_accuracy,
            "hallucination_rate": hallucination_rate,
        }

    # Performance
    latencies = [s.latency_seconds for s in samples]
    lat_arr = np.array(latencies)
    p50_lat = float(np.median(lat_arr))
    p95_lat = float(np.percentile(lat_arr, 95))

    total_prompt_tok = sum(s.prompt_tokens for s in samples)
    total_compl_tok = sum(s.completion_tokens for s in samples)
    total_tokens = total_prompt_tok + total_compl_tok

    duration = max(total_wall_clock_seconds, 0.001)
    tok_per_sec = total_tokens / duration
    samples_per_sec = total / duration

    cost_usd = (total_prompt_tok / 1000.0 * prompt_price_per_1k) + (
        total_compl_tok / 1000.0 * completion_price_per_1k
    )
    cost_per_1k = (cost_usd / total * 1000.0) if total > 0 else 0.0

    return BenchmarkReport(
        model_name=model_name,
        prompt_version=prompt_version,
        timestamp=timestamp,
        total_examples=total,
        valid_json_count=valid_json_count,
        json_validity_rate=json_validity_rate,
        overall_exact_match_rate=overall_exact_match,
        field_metrics=field_metrics,
        latency_p50_seconds=p50_lat,
        latency_p95_seconds=p95_lat,
        throughput_tokens_per_second=tok_per_sec,
        throughput_samples_per_second=samples_per_sec,
        total_cost_usd=cost_usd,
        cost_per_1k_samples_usd=cost_per_1k,
    )


def render_markdown_table(reports: list[BenchmarkReport]) -> str:
    """Format multiple benchmark reports into a Markdown comparative table."""
    lines: list[str] = [
        "| Model | Prompt Ver | JSON Valid % | Overall EM % | Avg Null Acc % | Avg Halluc % | p50 Latency (s) | p95 Latency (s) | Cost / 1k ($) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in reports:
        avg_null_acc = (
            sum(m["null_accuracy"] for m in r.field_metrics.values()) / len(r.field_metrics)
            if r.field_metrics
            else 0.0
        )
        avg_halluc = (
            sum(m["hallucination_rate"] for m in r.field_metrics.values()) / len(r.field_metrics)
            if r.field_metrics
            else 0.0
        )

        lines.append(
            f"| `{r.model_name}` | `{r.prompt_version}` | {r.json_validity_rate * 100:.1f}% | "
            f"{r.overall_exact_match_rate * 100:.1f}% | {avg_null_acc * 100:.1f}% | "
            f"{avg_halluc * 100:.1f}% | {r.latency_p50_seconds:.2f}s | {r.latency_p95_seconds:.2f}s | "
            f"${r.cost_per_1k_samples_usd:.3f} |"
        )

    return "\n".join(lines)
