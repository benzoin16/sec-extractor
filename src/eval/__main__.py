"""CLI entry point for the evaluation harness.

Usage::

    python -m src.eval --model gpt-4o-mini --base-url https://api.openai.com/v1
    python -m src.eval --model Qwen/Qwen2.5-7B-Instruct --base-url http://localhost:8000/v1
    python -m src.eval --config configs/eval.yaml --target frontier
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from src.eval.metrics import compute_benchmark_report, render_markdown_table
from src.eval.models import BenchmarkReport
from src.eval.runner import EvalRunner

load_dotenv()

app = typer.Typer(
    name="eval",
    help="Evaluate models against SEC filing extraction benchmark test set.",
    add_completion=False,
)
console = Console()


@app.command()
def main(
    config: Path = typer.Option(
        Path("configs/eval.yaml"), "--config", "-c", help="Evaluation YAML configuration."
    ),
    target: str | None = typer.Option(
        None,
        "--target",
        "-t",
        help="Target model key from configs/eval.yaml ('frontier' or 'base_7b').",
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="Model name (overrides config)."),
    base_url: str | None = typer.Option(
        None, "--base-url", "-u", help="OpenAI-compatible API base URL."
    ),
    api_key: str | None = typer.Option(None, "--api-key", "-k", help="API key string or 'EMPTY'."),
    test_file: Path | None = typer.Option(None, "--test-file", help="Path to test.jsonl."),
    prompt_version: str = typer.Option(
        "v1.0", "--prompt-version", help="Prompt version identifier."
    ),
    concurrency: int = typer.Option(8, "--concurrency", help="Max concurrent evaluation requests."),
    gate_config: Path | None = typer.Option(
        None, "--gate-config", "-g", help="Path to eval_thresholds.yaml gate configuration."
    ),
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg: dict = {}
    if config.exists():
        with open(config, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}

    # Resolve target model settings
    models_cfg = cfg.get("models", {})
    resolved_model = model
    resolved_url = base_url
    resolved_key = api_key
    pricing_prompt = 0.0
    pricing_compl = 0.0

    if target and target in models_cfg:
        t_cfg = models_cfg[target]
        resolved_model = resolved_model or t_cfg.get("name")
        resolved_url = resolved_url or t_cfg.get("base_url")
        key_env = t_cfg.get("api_key_env", "")
        resolved_key = resolved_key or os.environ.get(key_env, "EMPTY")
        pricing_prompt = t_cfg.get("pricing", {}).get("prompt_per_1k", 0.0)
        pricing_compl = t_cfg.get("pricing", {}).get("completion_per_1k", 0.0)

    resolved_model = resolved_model or "gpt-4o-mini"
    resolved_url = resolved_url or "https://api.openai.com/v1"
    resolved_key = resolved_key or os.environ.get("OPENAI_API_KEY", "EMPTY")

    resolved_test_path = test_file or Path(cfg.get("test_dataset", "data/processed/test.jsonl"))
    reports_dir = Path(cfg.get("reports_dir", "reports"))

    if not resolved_test_path.exists():
        console.print(
            f"[bold red]ERROR:[/] Test dataset not found at '{resolved_test_path}'. Run 'python -m src.dataset' first."
        )
        raise typer.Exit(1)

    # Load test examples
    examples: list[dict] = []
    with open(resolved_test_path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                examples.append(json.loads(line))

    console.rule("[bold blue]SEC EDGAR Extraction Benchmark[/]")
    console.print(
        f"Model          : [bold]{resolved_model}[/]\n"
        f"Endpoint       : {resolved_url}\n"
        f"Prompt Version : {prompt_version}\n"
        f"Test Examples  : {len(examples):,}\n"
        f"Concurrency    : {concurrency}"
    )
    console.rule()

    runner = EvalRunner(
        model_name=resolved_model,
        base_url=resolved_url,
        api_key=resolved_key,
        max_concurrency=concurrency,
        timeout=float(cfg.get("timeout_seconds", 60)),
        temperature=float(cfg.get("temperature", 0.0)),
        seed=int(cfg.get("seed", 42)),
        rel_tol=float(cfg.get("tolerances", {}).get("dollar_relative_tolerance", 0.005)),
        eps_abs_tol=float(cfg.get("tolerances", {}).get("eps_absolute_tolerance", 0.02)),
    )

    async def _execute() -> list:
        try:
            return await runner.run(examples)
        finally:
            await runner.close()

    start_wall = time.perf_counter()
    sample_results = asyncio.run(_execute())
    total_wall_time = time.perf_counter() - start_wall

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    report = compute_benchmark_report(
        samples=sample_results,
        model_name=resolved_model,
        prompt_version=prompt_version,
        timestamp=timestamp,
        prompt_price_per_1k=pricing_prompt,
        completion_price_per_1k=pricing_compl,
        total_wall_clock_seconds=total_wall_time,
    )

    # Save detailed JSON report
    reports_dir.mkdir(parents=True, exist_ok=True)
    clean_model_name = resolved_model.replace("/", "_").replace(":", "_")
    json_path = reports_dir / f"eval_{clean_model_name}_{prompt_version}_{timestamp}.json"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    # Update cumulative markdown summary table
    summary_md_path = reports_dir / "eval_summary.md"
    existing_reports: list[BenchmarkReport] = [report]
    # Check for other saved reports in directory
    for f in reports_dir.glob("eval_*.json"):
        if f != json_path:
            with contextlib.suppress(Exception):
                existing_reports.append(
                    BenchmarkReport.model_validate_json(f.read_text(encoding="utf-8"))
                )

    summary_md_path.write_text(
        "# SEC Extractor Model Benchmarks\n\n" + render_markdown_table(existing_reports) + "\n",
        encoding="utf-8",
    )

    console.print(f"[bold green]✓[/] Evaluation complete. Results saved to [cyan]{json_path}[/]")
    console.print(f"[bold green]✓[/] Cumulative summary updated at [cyan]{summary_md_path}[/]")

    # Print summary table
    table = Table(title=f"Benchmark Results: {resolved_model} ({prompt_version})", show_lines=True)
    table.add_column("Concept", justify="left")
    table.add_column("Exact Match %", justify="right")
    table.add_column("Null Accuracy %", justify="right")
    table.add_column("Hallucination Rate %", justify="right")

    for concept, m in report.field_metrics.items():
        table.add_row(
            concept,
            f"{m['exact_match_rate'] * 100:.1f}%",
            f"{m['null_accuracy'] * 100:.1f}%",
            f"{m['hallucination_rate'] * 100:.1f}%",
        )

    console.print(table)
    console.print(
        f"\n[bold]Overall Exact Match:[/] {report.overall_exact_match_rate * 100:.1f}% | "
        f"[bold]JSON Validity:[/] {report.json_validity_rate * 100:.1f}% | "
        f"[bold]p50 Latency:[/] {report.latency_p50_seconds:.2f}s | "
        f"[bold]Cost / 1k:[/] ${report.cost_per_1k_samples_usd:.3f}"
    )

    # Enforce quality gates if gate_config provided
    if gate_config and gate_config.exists():
        with open(gate_config, encoding="utf-8") as gf:
            gate_rules = yaml.safe_load(gf) or {}

        gate_failures = []
        min_jv = gate_rules.get("min_json_validity", 0.95)
        if report.json_validity_rate < min_jv:
            gate_failures.append(
                f"JSON validity rate ({report.json_validity_rate * 100:.1f}%) is below threshold ({min_jv * 100:.1f}%)"
            )

        min_em = gate_rules.get("min_overall_exact_match", 0.80)
        if report.overall_exact_match_rate < min_em:
            gate_failures.append(
                f"Overall exact match ({report.overall_exact_match_rate * 100:.1f}%) is below threshold ({min_em * 100:.1f}%)"
            )

        max_lat = gate_rules.get("max_p95_latency_ms", 5000.0)
        actual_lat_ms = report.latency_p95_seconds * 1000.0
        if actual_lat_ms > max_lat:
            gate_failures.append(
                f"P95 latency ({actual_lat_ms:.1f}ms) exceeds threshold ({max_lat:.1f}ms)"
            )

        concept_thresholds = gate_rules.get("concept_thresholds", {})
        for concept, min_c_em in concept_thresholds.items():
            if concept in report.field_metrics:
                c_em = report.field_metrics[concept].get("exact_match_rate", 0.0)
                if c_em < min_c_em:
                    gate_failures.append(
                        f"Concept '{concept}' exact match ({c_em * 100:.1f}%) is below threshold ({min_c_em * 100:.1f}%)"
                    )

        if gate_failures:
            console.rule("[bold red]CI EVAL GATE FAILED[/]")
            for f in gate_failures:
                console.print(f"[bold red]❌ GATE VIOLATION:[/] {f}")
            raise typer.Exit(code=1)
        else:
            console.rule("[bold green]CI EVAL GATE PASSED[/]")
            console.print("[bold green]✓ All evaluation metrics satisfy quality thresholds.[/]")


if __name__ == "__main__":
    app()
