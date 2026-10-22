"""CLI entry point for the ingest pipeline.

Usage::

    python -m src.ingest --tickers AAPL MSFT GOOGL
    python -m src.ingest --config configs/ingest.yaml
    python -m src.ingest --tickers AAPL --dry-run
"""

from __future__ import annotations

import logging
import os
from datetime import date
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from src.ingest.cik_resolver import CIKResolver, UnknownTickerError
from src.ingest.downloader import Downloader
from src.ingest.filing_fetcher import FilingFetcher
from src.ingest.rate_limiter import SlidingWindowRateLimiter

load_dotenv()

app = typer.Typer(
    name="ingest",
    help="Download SEC EDGAR 10-K/10-Q filings to data/raw.",
    add_completion=False,
)
console = Console()


@app.command()
def main(
    tickers: list[str] | None = typer.Option(
        None, "--tickers", "-t", help="Ticker symbols to ingest."
    ),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Path to ingest.yaml (overridden by CLI flags)."
    ),
    forms: list[str] | None = typer.Option(
        None, "--forms", help="Filing types to fetch (default: 10-K 10-Q)."
    ),
    start: str = typer.Option("2019-01-01", "--start", help="Earliest filing date (YYYY-MM-DD)."),
    end: str = typer.Option("2024-12-31", "--end", help="Latest filing date (YYYY-MM-DD)."),
    output: Path = typer.Option(Path("data/raw"), "--output", "-o", help="Cache root directory."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="List filings found without downloading."
    ),
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
    if not user_agent:
        console.print(
            "[bold red]ERROR:[/] SEC_USER_AGENT env var is not set.\n"
            "Copy .env.example to .env and fill in your name and email."
        )
        raise typer.Exit(1)

    # -----------------------------------------------------------------
    # Merge config file with CLI flags (CLI takes precedence).
    # -----------------------------------------------------------------
    cfg: dict = {}
    config_path = config or Path("configs/ingest.yaml")
    if config_path.exists():
        with open(config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}

    raw_tickers = tickers or cfg.get("tickers", [])
    resolved_tickers: list[str] = []
    for item in raw_tickers:
        for t in item.replace(",", " ").split():
            if t.strip():
                resolved_tickers.append(t.strip().upper())

    resolved_forms: list[str] = forms or cfg.get("forms", ["10-K", "10-Q"])
    resolved_start: str = start if start != "2019-01-01" else cfg.get("start_date", start)
    resolved_end: str = end if end != "2024-12-31" else cfg.get("end_date", end)

    if not resolved_tickers:
        console.print("[bold red]ERROR:[/] No tickers specified. Use --tickers or --config.")
        raise typer.Exit(1)

    start_date = date.fromisoformat(resolved_start)
    end_date = date.fromisoformat(resolved_end)
    form_types = set(resolved_forms)

    output.mkdir(parents=True, exist_ok=True)
    rate_limiter = SlidingWindowRateLimiter(max_calls=8, period=1.0)

    console.rule("[bold blue]SEC EDGAR Ingest[/]")
    console.print(
        f"Tickers : {resolved_tickers}\n"
        f"Forms   : {sorted(form_types)}\n"
        f"Range   : {start_date} → {end_date}\n"
        f"Output  : {output}\n"
        f"Dry run : {dry_run}"
    )
    console.rule()

    # -----------------------------------------------------------------
    # Summary table (populated as we go).
    # -----------------------------------------------------------------
    summary = Table(title="Ingest Summary", show_lines=True, highlight=True)
    for col in ("Ticker", "CIK", "Filings Found", "Downloaded", "Cached", "Errors"):
        summary.add_column(col, justify="right" if col not in ("Ticker", "CIK") else "left")

    resolver = CIKResolver(cache_dir=output, user_agent=user_agent, rate_limiter=rate_limiter)

    with (
        FilingFetcher(user_agent=user_agent, rate_limiter=rate_limiter) as fetcher,
        Downloader(output_dir=output, user_agent=user_agent, rate_limiter=rate_limiter) as dl,
    ):
        for ticker in resolved_tickers:
            downloaded = cached = errors = 0
            try:
                cik = resolver.resolve(ticker)
            except UnknownTickerError as exc:
                console.print(f"[yellow]SKIP[/] {ticker}: {exc}")
                summary.add_row(ticker, "—", "—", "—", "—", "unknown ticker")
                continue

            filings = fetcher.fetch_filings(
                cik=cik,
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                form_types=form_types,
            )

            if not dry_run:
                for meta in filings:
                    was_cached = dl.is_cached(meta)
                    try:
                        result = dl.download(meta)
                        if result is None:
                            errors += 1
                        elif was_cached:
                            cached += 1
                        else:
                            downloaded += 1
                    except Exception as exc:  # noqa: BLE001
                        logging.error(
                            "Error downloading %s %s: %s",
                            ticker,
                            meta.accession_number,
                            exc,
                        )
                        errors += 1

            summary.add_row(
                ticker,
                cik,
                str(len(filings)),
                str(downloaded),
                str(cached),
                str(errors),
            )

    console.print(summary)


if __name__ == "__main__":
    app()
