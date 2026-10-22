"""CLI entry point for training dataset construction, splitting, and stats reporting.

Usage::

    python -m src.dataset --input data/processed --output data/processed
    python -m src.dataset --config configs/dataset.yaml
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from src.dataset.builder import DatasetBuilder
from src.dataset.matcher import NumericMatcher
from src.dataset.splitter import DatasetSplitter
from src.dataset.stats import DatasetStats
from src.labels.models import FilingLabels
from src.models import Chunk, FilingMeta

app = typer.Typer(
    name="dataset",
    help="Build train/val/test instruction-tuning datasets with company+time splits.",
    add_completion=False,
)
console = Console()


@app.command()
def main(
    input_dir: Path = typer.Option(
        Path("data/processed"), "--input", "-i", help="Processed filings root directory."
    ),
    output_dir: Path = typer.Option(
        Path("data/processed"), "--output", "-o", help="Directory to save train/val/test jsonl."
    ),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Path to configs/dataset.yaml."
    ),
    reports_dir: Path = typer.Option(
        Path("reports"), "--reports-dir", help="Directory for dataset stats report."
    ),
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg: dict = {}
    cfg_path = config or Path("configs/dataset.yaml")
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}

    test_tickers = cfg.get("test_tickers", ["JPM", "AMZN"])
    val_fiscal_year = cfg.get("val_fiscal_year", 2024)
    negative_ratio = cfg.get("negative_ratio", 0.15)
    hard_negative_ratio = cfg.get("hard_negative_ratio", 0.50)

    console.rule("[bold magenta]SEC EDGAR Training Dataset Builder[/]")
    console.print(
        f"Input dir         : {input_dir}\n"
        f"Output dir        : {output_dir}\n"
        f"Test tickers      : {test_tickers}\n"
        f"Val fiscal year   : {val_fiscal_year}\n"
        f"Negative ratio    : {negative_ratio * 100:.1f}%\n"
        f"Hard neg ratio    : {hard_negative_ratio * 100:.1f}%"
    )
    console.rule()

    # Discover filings that have both chunks.jsonl and labels.json
    filing_items: list[tuple[FilingMeta, list[Chunk], FilingLabels]] = []

    for chunks_file in input_dir.rglob("chunks.jsonl"):
        labels_file = chunks_file.parent / "labels.json"
        if not labels_file.exists():
            continue

        try:
            labels_dict = json.loads(labels_file.read_text(encoding="utf-8"))
            labels = FilingLabels.model_validate(labels_dict)

            chunks: list[Chunk] = []
            for line in chunks_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    chunks.append(Chunk.model_validate_json(line))

            if chunks:
                filing_items.append((chunks[0].meta, chunks, labels))
        except Exception as exc:  # noqa: BLE001
            logging.warning("Error reading filing at %s: %s", chunks_file.parent, exc)

    if not filing_items:
        console.print(
            f"[yellow]No processed filings with both chunks.jsonl and labels.json found in {input_dir}.\n"
            "Run 'python -m src.parse' and 'python -m src.labels' first.[/]"
        )
        raise typer.Exit(0)

    # 1. Company + Time split
    all_filings = [item[0] for item in filing_items]
    splitter = DatasetSplitter(test_tickers=test_tickers, val_fiscal_year=val_fiscal_year)
    split_filing_metas = splitter.split_filings(all_filings)

    # Generate and verify manifest
    manifest_path = output_dir / "split_manifest.json"
    splitter.generate_manifest(split_filing_metas, output_path=manifest_path)
    console.print(
        f"[bold green]✓[/] Split manifest created and verified at [cyan]{manifest_path}[/]"
    )

    # Index filing items by accession number
    filing_map = {item[0].accession_number: item for item in filing_items}

    # 2. Build dataset examples per split
    matcher = NumericMatcher()
    builder = DatasetBuilder(
        matcher=matcher,
        negative_ratio=negative_ratio,
        hard_negative_ratio=hard_negative_ratio,
    )

    split_examples: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for split_name, metas in split_filing_metas.items():
        for meta in metas:
            item = filing_map.get(meta.accession_number)
            if not item:
                continue
            _, chunks, labels = item
            examples = builder.build_examples_for_filing(chunks=chunks, labels=labels)
            split_examples[split_name].extend(examples)

    # 3. Write train.jsonl, val.jsonl, test.jsonl
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, examples in split_examples.items():
        split_file = output_dir / f"{split_name}.jsonl"
        with open(split_file, "w", encoding="utf-8") as fh:
            for ex in examples:
                fh.write(json.dumps(ex) + "\n")
        console.print(f"Wrote {len(examples):,} examples to [cyan]{split_file}[/]")

    if matcher.skipped_ambiguous_count > 0:
        console.print(
            f"[yellow]ℹ Skipped {matcher.skipped_ambiguous_count} ambiguous examples (identical quarter/YTD values).[/]"
        )

    # 4. Compute statistics & render report
    stats = DatasetStats.compute_stats(split_examples)
    report_file = reports_dir / "dataset_stats.md"
    DatasetStats.render_markdown_report(stats, report_file)
    console.print(f"[bold green]✓[/] Dataset statistics report generated at [cyan]{report_file}[/]")

    # Print summary table
    summary = Table(title="Dataset Split Summary", show_lines=True)
    summary.add_column("Split", justify="left")
    summary.add_column("Total Examples", justify="right")
    summary.add_column("Positives", justify="right")
    summary.add_column("Hard Negatives", justify="right")
    summary.add_column("Text Negatives", justify="right")
    summary.add_column("Negative %", justify="right")

    for sname, sdata in stats["splits"].items():
        summary.add_row(
            sname,
            f"{sdata['total']:,}",
            f"{sdata['positives']:,}",
            f"{sdata['hard_negatives']:,}",
            f"{sdata['text_negatives']:,}",
            f"{sdata['negative_pct'] * 100:.1f}%",
        )

    console.print(summary)


if __name__ == "__main__":
    app()
