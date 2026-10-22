"""Unit tests for dataset construction, numeric matching, negative sampling, and split leak assertions."""

from datetime import date

import pytest

from src.dataset.builder import DatasetBuilder
from src.dataset.matcher import NumericMatcher
from src.dataset.splitter import DatasetSplitter
from src.dataset.stats import DatasetStats
from src.labels.models import ConceptValue, FilingLabels
from src.models import Chunk, FilingMeta


def test_numeric_matcher_scales():
    matcher = NumericMatcher()

    text = """
    Consolidated Statements of Operations (in millions, except per share data):
    Total net sales: $383,285
    Net loss: $(1,234)
    Diluted EPS: $6.13
    """

    # Scale 1e6 (millions)
    assert matcher.check_value_in_text(383285000000, text) is True
    # Unmatched amount
    assert matcher.check_value_in_text(500000000000, text) is False
    # Negative amount with parentheses
    assert matcher.check_value_in_text(-1234000000, text) is True
    # EPS float
    assert matcher.check_value_in_text(6.13, text, is_eps=True) is True
    assert matcher.check_value_in_text(5.00, text, is_eps=True) is False


def test_matcher_disambiguation_rules():
    matcher = NumericMatcher()
    meta = FilingMeta(
        cik="0000320193",
        ticker="AAPL",
        accession_number="0000320193-23-000077",
        form_type="10-Q",
        filed_date=date(2023, 8, 4),
        period_of_report=date(2023, 7, 1),
    )

    labels = FilingLabels(
        cik=meta.cik,
        ticker=meta.ticker,
        accession_number=meta.accession_number,
        form_type=meta.form_type,
        period_of_report=meta.period_of_report,
        metrics={
            "revenue": [
                ConceptValue(value=81797000000, period_type="quarter", tag="Revenues"),
                ConceptValue(value=166423000000, period_type="ytd", tag="Revenues"),
            ],
            "operating_cash_flow": [
                ConceptValue(
                    value=25000000000,
                    period_type="quarter",
                    tag="NetCashProvidedByUsedInOperatingActivities",
                ),
                ConceptValue(
                    value=56842000000,
                    period_type="ytd",
                    tag="NetCashProvidedByUsedInOperatingActivities",
                ),
            ],
        },
    )

    # Chunk text contains both 3mo and 6mo numbers
    chunk_text = """
    Three Months Ended: Revenue was $81,797 million. Cash flow was $25,000 million.
    Six Months Ended: Revenue was $166,423 million. Cash flow was $56,842 million.
    """

    matched, was_skipped = matcher.match_chunk(
        chunk_text, labels, ["revenue", "operating_cash_flow"]
    )
    assert was_skipped is False
    # Income statement item prefers quarter
    assert matched["revenue"].value == 81797000000
    assert matched["revenue"].period_type == "quarter"
    # Cash flow item prefers YTD
    assert matched["operating_cash_flow"].value == 56842000000
    assert matched["operating_cash_flow"].period_type == "ytd"


def test_matcher_ambiguity_skips_example():
    matcher = NumericMatcher()
    meta = FilingMeta(
        cik="0000320193",
        ticker="AAPL",
        accession_number="0000320193-23-000077",
        form_type="10-Q",
        filed_date=date(2023, 8, 4),
        period_of_report=date(2023, 7, 1),
    )

    # Both quarter and YTD happen to have the identical value
    labels = FilingLabels(
        cik=meta.cik,
        ticker=meta.ticker,
        accession_number=meta.accession_number,
        form_type=meta.form_type,
        period_of_report=meta.period_of_report,
        metrics={
            "revenue": [
                ConceptValue(value=50000000000, period_type="quarter", tag="Revenues"),
                ConceptValue(value=50000000000, period_type="ytd", tag="Revenues"),
            ]
        },
    )

    chunk_text = "Total revenue was $50,000 million."
    matched, was_skipped = matcher.match_chunk(chunk_text, labels, ["revenue"])
    assert was_skipped is True
    assert matcher.skipped_ambiguous_count == 1


def test_derived_liabilities_presence():
    matcher = NumericMatcher()
    meta = FilingMeta(
        cik="0000320193",
        ticker="AAPL",
        accession_number="0000320193-23-000106",
        form_type="10-K",
        filed_date=date(2023, 10, 27),
        period_of_report=date(2023, 9, 30),
    )

    labels = FilingLabels(
        cik=meta.cik,
        ticker=meta.ticker,
        accession_number=meta.accession_number,
        form_type=meta.form_type,
        period_of_report=meta.period_of_report,
        metrics={
            "total_liabilities": [
                ConceptValue(
                    value=290437000000,
                    period_type="instant",
                    tag="LiabilitiesCurrent+LiabilitiesNoncurrent",
                    is_derived=True,
                )
            ]
        },
    )

    # Case 1: Chunk only has components (145,308 and 145,129), not the sum (290,437)
    chunk_components_only = "Current liabilities: $145,308. Noncurrent: $145,129."
    matched1, _ = matcher.match_chunk(chunk_components_only, labels, ["total_liabilities"])
    # MUST be null because sum itself is not printed
    assert matched1["total_liabilities"] is None

    # Case 2: Chunk prints the sum 290,437
    chunk_with_sum = "Total Liabilities: $290,437 million."
    matched2, _ = matcher.match_chunk(chunk_with_sum, labels, ["total_liabilities"])
    assert matched2["total_liabilities"] is not None
    assert matched2["total_liabilities"].value == 290437000000


def test_splitter_and_leak_detection():
    filings = [
        # Test company (JPM) across multiple years
        FilingMeta(
            cik="0000019617",
            ticker="JPM",
            accession_number="JPM-2022",
            form_type="10-K",
            filed_date=date(2023, 2, 1),
            period_of_report=date(2022, 12, 31),
        ),
        FilingMeta(
            cik="0000019617",
            ticker="JPM",
            accession_number="JPM-2023",
            form_type="10-K",
            filed_date=date(2024, 2, 1),
            period_of_report=date(2023, 12, 31),
        ),
        # Training company (AAPL)
        FilingMeta(
            cik="0000320193",
            ticker="AAPL",
            accession_number="AAPL-2022",
            form_type="10-K",
            filed_date=date(2022, 10, 28),
            period_of_report=date(2022, 9, 30),
        ),
        FilingMeta(
            cik="0000320193",
            ticker="AAPL",
            accession_number="AAPL-2023",
            form_type="10-K",
            filed_date=date(2023, 10, 27),
            period_of_report=date(2023, 9, 30),
        ),
        FilingMeta(
            cik="0000320193",
            ticker="AAPL",
            accession_number="AAPL-2024",
            form_type="10-K",
            filed_date=date(2024, 11, 1),
            period_of_report=date(2024, 9, 30),
        ),
    ]

    splitter = DatasetSplitter(test_tickers=["JPM"], val_fiscal_year=2024)
    splits = splitter.split_filings(filings)

    # Verify split membership
    assert len(splits["test"]) == 2
    assert all(f.ticker == "JPM" for f in splits["test"])

    assert len(splits["train"]) == 2
    assert all(f.period_of_report.year < 2024 for f in splits["train"])

    assert len(splits["val"]) == 1
    assert splits["val"][0].accession_number == "AAPL-2024"

    # Manifest and leak verification
    manifest = splitter.generate_manifest(splits)
    DatasetSplitter.assert_no_leak(manifest)

    # Test that deliberate leakage raises AssertionError
    leaked_manifest = {
        "created_at": manifest["created_at"],
        "test_tickers": ["JPM"],
        "val_fiscal_year": 2024,
        "splits": {
            "train": manifest["splits"]["train"] + [manifest["splits"]["test"][0]],
            "val": manifest["splits"]["val"],
            "test": manifest["splits"]["test"],
        },
    }
    with pytest.raises(AssertionError, match="Leak detected"):
        DatasetSplitter.assert_no_leak(leaked_manifest)


def test_dataset_builder_and_stats():
    meta = FilingMeta(
        cik="0000320193",
        ticker="AAPL",
        accession_number="0000320193-23-000106",
        form_type="10-K",
        filed_date=date(2023, 10, 27),
        period_of_report=date(2023, 9, 30),
    )

    labels = FilingLabels(
        cik=meta.cik,
        ticker=meta.ticker,
        accession_number=meta.accession_number,
        form_type=meta.form_type,
        period_of_report=meta.period_of_report,
        metrics={
            "revenue": [ConceptValue(value=383285000000, period_type="annual", tag="Revenues")],
        },
    )

    # Positive chunk
    c1 = Chunk(
        meta=meta,
        item_id="item_8",
        chunk_index=0,
        total_chunks=3,
        text="Total revenue was $383,285 million.",
        token_count=100,
    )
    # Hard negative chunk (has other table numbers, but no target metrics)
    c2 = Chunk(
        meta=meta,
        item_id="item_7",
        chunk_index=1,
        total_chunks=3,
        text="| Segment | Sales |\n| :--- | :--- |\n| Hardware | 42,000 |\n| Services | 18,000 |",
        token_count=120,
    )
    # Text negative chunk (narrative)
    c3 = Chunk(
        meta=meta,
        item_id="item_1a",
        chunk_index=2,
        total_chunks=3,
        text="Global macroeconomic uncertainties may affect operations.",
        token_count=80,
    )

    builder = DatasetBuilder(negative_ratio=0.50, hard_negative_ratio=0.50)
    examples = builder.build_examples_for_filing([c1, c2, c3], labels)

    assert len(examples) >= 2
    # Verify chat format
    ex = examples[0]
    assert len(ex["messages"]) == 3
    assert ex["messages"][0]["role"] == "system"
    assert ex["messages"][1]["role"] == "user"
    assert ex["messages"][2]["role"] == "assistant"

    # Compute stats
    stats = DatasetStats.compute_stats({"train": examples})
    assert stats["overall"]["total_examples"] == len(examples)
    assert stats["splits"]["train"]["positives"] >= 1

    report_path = c1.meta.cik + "_report.md"
    from pathlib import Path

    report_file = Path(report_path)
    try:
        report_text = DatasetStats.render_markdown_report(stats, report_file)
        assert "# SEC Extractor Dataset Statistics Report" in report_text
        assert "revenue" in report_text
    finally:
        if report_file.exists():
            report_file.unlink()
