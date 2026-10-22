"""src.eval: Evaluation harness, metrics, and runners for model benchmarking."""

from src.eval.metrics import (
    NumericNormalizer,
    compute_benchmark_report,
    evaluate_sample,
    render_markdown_table,
)
from src.eval.models import BenchmarkReport, FieldEvalResult, SampleEvalResult
from src.eval.runner import EvalRunner

__all__ = [
    "NumericNormalizer",
    "FieldEvalResult",
    "SampleEvalResult",
    "BenchmarkReport",
    "evaluate_sample",
    "compute_benchmark_report",
    "render_markdown_table",
    "EvalRunner",
]
