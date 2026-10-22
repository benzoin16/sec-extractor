"""Unit tests for the evaluation harness and numeric normalization."""

import json

from src.eval.metrics import (
    NumericNormalizer,
    compute_benchmark_report,
    evaluate_sample,
    parse_prediction_json,
    render_markdown_table,
)


def test_numeric_normalizer_parse():
    assert NumericNormalizer.parse_numeric(100) == 100.0
    assert NumericNormalizer.parse_numeric("1,234.56") == 1234.56
    assert NumericNormalizer.parse_numeric("$383,285") == 383285.0
    assert NumericNormalizer.parse_numeric("(1,234)") == -1234.0
    assert NumericNormalizer.parse_numeric("invalid") is None
    assert NumericNormalizer.parse_numeric(None) is None


def test_numeric_normalizer_values_match():
    # Exact match
    assert NumericNormalizer.values_match(383285000000.0, 383285000000.0) is True

    # Rounding tolerance (0.5%)
    truth = 383285000000.0
    slightly_rounded = 383000000000.0  # 0.07% error
    assert NumericNormalizer.values_match(slightly_rounded, truth, rel_tol=0.005) is True

    # Large difference exceeds tolerance
    bad_val = 300000000000.0
    assert NumericNormalizer.values_match(bad_val, truth, rel_tol=0.005) is False

    # Scale tolerance: model predicted 383285 (millions) instead of 383285000000
    assert NumericNormalizer.values_match(383285.0, truth) is True
    assert NumericNormalizer.values_match(383285000.0, truth) is True  # thousands

    # Sign mismatch must fail
    assert NumericNormalizer.values_match(-383285000000.0, truth) is False

    # EPS tolerance
    assert NumericNormalizer.values_match(6.13, 6.13, is_eps=True) is True
    assert NumericNormalizer.values_match(6.14, 6.13, is_eps=True, eps_abs_tol=0.02) is True
    assert NumericNormalizer.values_match(6.16, 6.13, is_eps=True, eps_abs_tol=0.02) is False


def test_parse_prediction_json():
    clean = '{"revenue": {"value": 100, "period_type": "annual"}}'
    assert parse_prediction_json(clean) == {"revenue": {"value": 100, "period_type": "annual"}}

    markdown_wrapped = '```json\n{"revenue": 100}\n```'
    assert parse_prediction_json(markdown_wrapped) == {"revenue": 100}

    broken = "Here is the result: {not json}"
    assert parse_prediction_json(broken) is None


def test_evaluate_sample_metrics():
    truth = {
        "revenue": {"value": 383285000000, "period_type": "annual"},
        "net_income": {"value": 96995000000, "period_type": "annual"},
        "diluted_eps": {"value": 6.13, "period_type": "annual"},
        "total_assets": {"value": 352583000000, "period_type": "instant"},
        "operating_income": {"value": 114301000000, "period_type": "annual"},
        "total_liabilities": None,
        "operating_cash_flow": None,
    }

    # Perfect prediction
    pred_perfect = json.dumps(
        {
            "revenue": {"value": 383285000000, "period_type": "annual"},
            "net_income": {"value": 96995000000, "period_type": "annual"},
            "diluted_eps": {"value": 6.13, "period_type": "annual"},
            "total_assets": {"value": 352583000000, "period_type": "instant"},
            "operating_income": {"value": 114301000000, "period_type": "annual"},
            "total_liabilities": None,
            "operating_cash_flow": None,
        }
    )

    res = evaluate_sample(pred_perfect, truth, example_id="ex1")
    assert res.is_valid_json is True
    assert res.all_fields_matched is True
    assert res.field_results["revenue"].is_exact_match is True
    assert res.field_results["total_liabilities"].is_null_match is True
    assert res.field_results["total_liabilities"].is_hallucination is False

    # Prediction with a hallucination: total_liabilities predicted when truth is null
    pred_halluc = json.dumps(
        {
            "revenue": {"value": 383285000000, "period_type": "annual"},
            "net_income": {"value": 96995000000, "period_type": "annual"},
            "diluted_eps": {"value": 6.13, "period_type": "annual"},
            "total_assets": {"value": 352583000000, "period_type": "instant"},
            "operating_income": {"value": 114301000000, "period_type": "annual"},
            "total_liabilities": {"value": 50000000000, "period_type": "instant"},
            "operating_cash_flow": None,
        }
    )

    res_h = evaluate_sample(pred_halluc, truth, example_id="ex2")
    assert res_h.all_fields_matched is False
    assert res_h.field_results["total_liabilities"].is_hallucination is True


def test_benchmark_report_and_table():
    truth = {
        "revenue": {"value": 100, "period_type": "annual"},
        "net_income": None,
        "diluted_eps": None,
        "total_assets": None,
        "operating_income": None,
        "total_liabilities": None,
        "operating_cash_flow": None,
    }

    s1 = evaluate_sample(
        json.dumps(truth),
        truth,
        example_id="1",
        latency_seconds=0.5,
        prompt_tokens=200,
        completion_tokens=50,
    )
    s2 = evaluate_sample(
        json.dumps(truth),
        truth,
        example_id="2",
        latency_seconds=1.0,
        prompt_tokens=200,
        completion_tokens=50,
    )

    report = compute_benchmark_report(
        samples=[s1, s2],
        model_name="Qwen/Qwen2.5-7B-Instruct",
        prompt_version="v1.0",
        timestamp="20261020_120000",
        prompt_price_per_1k=0.0002,
        completion_price_per_1k=0.0002,
        total_wall_clock_seconds=1.5,
    )

    assert report.total_examples == 2
    assert report.json_validity_rate == 1.0
    assert report.overall_exact_match_rate == 1.0
    assert report.latency_p50_seconds == 0.75
    assert report.total_cost_usd > 0

    table = render_markdown_table([report])
    assert "| `Qwen/Qwen2.5-7B-Instruct` |" in table
    assert "100.0%" in table
