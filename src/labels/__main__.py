"""CLI entry point for ground-truth label extraction.

Usage::

    python -m src.labels --tickers AAPL MSFT
    python -m src.labels --config configs/ingest.yaml
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from src.ingest.cik_resolver import CIKResolver, UnknownTickerError
from src.ingest.rate_limiter import SlidingWindowRateLimiter
from src.labels.extractor import FactExtractor
from src.labels.fetcher import CompanyFactsFetcher
from src.models import FilingMeta

load_dotenv()

app = typer.Typer(
    name="labels",
    help="Fetch SEC companyfacts and extract ground truth labels for filings.",
    add_completion=False,
)
console = Console()


@app.command()
def main(
    tickers: list[str] | None = typer.Option(None, "--tickers", "-t", help="Tickers to process."),
    config: Path | None = typer.Option(None, "--config", "-c", help="Path to ingest.yaml."),
    xbrl_map: Path = typer.Option(
        Path("configs/xbrl_map.yaml"), "--xbrl-map", help="Path to xbrl_map.yaml."
    ),
    raw_dir: Path = typer.Option(Path("data/raw"), "--raw-dir", help="Raw filings cache root."),
    output_dir: Path = typer.Option(
        Path("data/processed"), "--output-dir", "-o", help="Output directory for labels."
    ),
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
    if not user_agent:
        console.print("[bold red]ERROR:[/] SEC_USER_AGENT env var is not set.")
        raise typer.Exit(1)

    cfg: dict = {}
    if config and config.exists():
        with open(config, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}

    resolved_tickers = tickers or cfg.get("tickers", [])

    if not xbrl_map.exists():
        console.print(f"[bold red]ERROR:[/] Concept map file '{xbrl_map}' not found.")
        raise typer.Exit(1)

    with open(xbrl_map, encoding="utf-8") as fh:
        map_cfg = yaml.safe_load(fh) or {}

    rate_limiter = SlidingWindowRateLimiter(max_calls=8, period=1.0)
    resolver = CIKResolver(cache_dir=raw_dir, user_agent=user_agent, rate_limiter=rate_limiter)
    fetcher = CompanyFactsFetcher(
        cache_dir=raw_dir, user_agent=user_agent, rate_limiter=rate_limiter
    )
    extractor = FactExtractor(config=map_cfg)

    console.rule("[bold cyan]SEC EDGAR Ground-Truth Label Extractor[/]")

    # Find filings to label: either from resolved_tickers or discovering meta.json in raw_dir
    filing_metas: list[FilingMeta] = []
    if resolved_tickers:
        for ticker in resolved_tickers:
            try:
                cik = resolver.resolve(ticker)
                ticker_dir = raw_dir / cik
                if ticker_dir.exists():
                    for meta_file in ticker_dir.rglob("meta.json"):
                        meta_dict = json.loads(meta_file.read_text(encoding="utf-8"))
                        filing_metas.append(FilingMeta.model_validate(meta_dict))
            except UnknownTickerError as exc:
                console.print(f"[yellow]SKIP[/] {ticker}: {exc}")
    else:
        for meta_file in raw_dir.rglob("meta.json"):
            meta_dict = json.loads(meta_file.read_text(encoding="utf-8"))
            filing_metas.append(FilingMeta.model_validate(meta_dict))

    if not filing_metas:
        console.print(
            f"[yellow]No filings found in {raw_dir}. Run 'python -m src.ingest' first.[/]"
        )
        raise typer.Exit(0)

    summary = Table(title="Label Extraction Summary", show_lines=True)
    summary.add_column("Ticker", justify="left")
    summary.add_column("Accession", justify="left")
    summary.add_column("Form", justify="center")
    summary.add_column("Period", justify="center")
    summary.add_column("Concepts Extracted", justify="right")

    # Group filings by CIK to minimize facts downloads
    ciks_seen: dict[str, dict] = {}
    total_labels_written = 0

    try:
        for meta in filing_metas:
            cik = meta.cik
            if cik not in ciks_seen:
                try:
                    ciks_seen[cik] = fetcher.fetch(cik)
                except Exception as exc:  # noqa: BLE001
                    logging.error("Failed to fetch facts for CIK %s: %s", cik, exc)
                    summary.add_row(
                        meta.ticker,
                        meta.accession_number,
                        meta.form_type,
                        str(meta.period_of_report),
                        "ERR",
                    )
                    continue

            facts = ciks_seen[cik]
            filing_labels = extractor.extract_labels_for_filing(facts, meta)

            # Write labels to processed folder
            out_filing_dir = output_dir / meta.cik / meta.accession_number.replace("-", "")
            out_filing_dir.mkdir(parents=True, exist_ok=True)
            labels_path = out_filing_dir / "labels.json"
            labels_path.write_text(filing_labels.model_dump_json(indent=2), encoding="utf-8")
            total_labels_written += 1

            summary.add_row(
                meta.ticker,
                meta.accession_number,
                meta.form_type,
                str(meta.period_of_report),
                f"{len(filing_labels.metrics)} / 7",
            )
    finally:
        fetcher.close()

    console.print(summary)
    console.print(
        f"\n[bold green]Complete![/] Successfully extracted labels for [bold]{total_labels_written}[/] filings."
    )


if __name__ == "__main__":
    app()
