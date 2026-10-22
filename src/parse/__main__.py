"""CLI entry point for the parsing and chunking pipeline.

Usage::

    python -m src.parse --input data/raw --output data/processed
    python -m src.parse --config configs/parse.yaml
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from src.models import Chunk, FilingMeta
from src.parse.chunker import Chunker
from src.parse.sectioner import Sectioner
from src.parse.table_parser import TableParser

app = typer.Typer(
    name="parse",
    help="Section HTML filings, convert tables to Markdown, and chunk into token-bounded pieces.",
    add_completion=False,
)
console = Console()


@app.command()
def main(
    input_dir: Path = typer.Option(
        Path("data/raw"), "--input", "-i", help="Path to raw filings directory."
    ),
    output_dir: Path = typer.Option(
        Path("data/processed"), "--output", "-o", help="Path to processed output directory."
    ),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Path to parse.yaml (overrides CLI defaults)."
    ),
    min_tokens: int | None = typer.Option(None, "--min-tokens", help="Minimum tokens per chunk."),
    max_tokens: int | None = typer.Option(None, "--max-tokens", help="Maximum tokens per chunk."),
    items: list[str] | None = typer.Option(
        None, "--items", help="Filter for specific item IDs, e.g. item_1a item_7 item_8."
    ),
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # -----------------------------------------------------------------
    # Merge config file with CLI flags
    # -----------------------------------------------------------------
    cfg: dict = {}
    if config and config.exists():
        with open(config, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}

    chunk_cfg = cfg.get("chunking", {})
    resolved_min = min_tokens or chunk_cfg.get("min_tokens", 1500)
    resolved_max = max_tokens or chunk_cfg.get("max_tokens", 3500)
    resolved_encoding = chunk_cfg.get("tokenizer_encoding", "cl100k_base")
    target_items = set(items or cfg.get("target_items", []))

    if not input_dir.exists():
        console.print(f"[bold red]ERROR:[/] Input directory '{input_dir}' does not exist.")
        raise typer.Exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    console.rule("[bold green]SEC EDGAR Parser & Chunker[/]")
    console.print(
        f"Input dir    : {input_dir}\n"
        f"Output dir   : {output_dir}\n"
        f"Token bounds : [{resolved_min}, {resolved_max}] ({resolved_encoding})\n"
        f"Target items : {sorted(target_items) if target_items else 'All Items'}"
    )
    console.rule()

    # Discover filings
    filing_dirs: list[tuple[Path, Path]] = []
    for meta_file in input_dir.rglob("meta.json"):
        primary_html = meta_file.parent / "primary.html"
        if primary_html.exists():
            filing_dirs.append((meta_file, primary_html))

    if not filing_dirs:
        console.print(
            f"[yellow]No filings found in {input_dir}. Run 'python -m src.ingest' first.[/]"
        )
        raise typer.Exit(0)

    table_parser = TableParser()
    sectioner = Sectioner(table_parser=table_parser)
    chunker = Chunker(
        min_tokens=resolved_min,
        max_tokens=resolved_max,
        encoding_name=resolved_encoding,
    )

    summary = Table(title="Parsing & Chunking Summary", show_lines=True)
    summary.add_column("Ticker", justify="left")
    summary.add_column("Form", justify="center")
    summary.add_column("Accession", justify="left")
    summary.add_column("Sections", justify="right")
    summary.add_column("Chunks", justify="right")
    summary.add_column("Total Tokens", justify="right")

    total_chunks_all = 0
    total_tokens_all = 0

    all_chunks_file = output_dir / "all_chunks.jsonl"
    with open(all_chunks_file, "w", encoding="utf-8") as all_chunks_handle:
        for meta_path, html_path in filing_dirs:
            try:
                meta_dict = json.loads(meta_path.read_text(encoding="utf-8"))
                meta = FilingMeta.model_validate(meta_dict)
                html_content = html_path.read_text(encoding="utf-8", errors="replace")

                sections = sectioner.parse_sections(html_content, meta)
                if target_items:
                    sections = [s for s in sections if s.item_id.lower() in target_items]

                filing_chunks: list[Chunk] = []
                filing_tokens = 0

                for s in sections:
                    chunks = chunker.chunk_section(s)
                    for c in chunks:
                        filing_chunks.append(c)
                        filing_tokens += c.token_count
                        all_chunks_handle.write(c.model_dump_json() + "\n")

                # Write per-filing chunks.jsonl
                filing_out_dir = output_dir / meta.cik / meta.accession_number.replace("-", "")
                filing_out_dir.mkdir(parents=True, exist_ok=True)
                chunks_out_path = filing_out_dir / "chunks.jsonl"
                with open(chunks_out_path, "w", encoding="utf-8") as ch_out:
                    for c in filing_chunks:
                        ch_out.write(c.model_dump_json() + "\n")

                total_chunks_all += len(filing_chunks)
                total_tokens_all += filing_tokens

                summary.add_row(
                    meta.ticker,
                    meta.form_type,
                    meta.accession_number,
                    str(len(sections)),
                    str(len(filing_chunks)),
                    f"{filing_tokens:,}",
                )
            except Exception as exc:  # noqa: BLE001
                logging.error("Failed to parse %s: %s", html_path, exc)
                summary.add_row("ERR", "—", html_path.name, "0", "0", "ERROR")

    console.print(summary)
    console.print(
        f"\n[bold green]Complete![/] Total chunks: [bold]{total_chunks_all}[/] | "
        f"Total tokens: [bold]{total_tokens_all:,}[/] | "
        f"Master dataset: [cyan]{all_chunks_file}[/]"
    )


if __name__ == "__main__":
    app()
