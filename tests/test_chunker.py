"""Unit tests for Chunker."""

from datetime import date

from src.models import FilingMeta, ParsedSection
from src.parse.chunker import Chunker


def test_chunker_small_section():
    meta = FilingMeta(
        cik="0001234567",
        ticker="ACME",
        accession_number="0001234567-24-000001",
        form_type="10-K",
        filed_date=date(2024, 2, 1),
        period_of_report=date(2023, 12, 31),
    )
    section = ParsedSection(
        meta=meta,
        item_id="item_1a",
        item_title="Item 1A. Risk Factors",
        text="This is a small section with only a few sentences.",
    )

    chunker = Chunker(min_tokens=100, max_tokens=500)
    chunks = chunker.chunk_section(section)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].total_chunks == 1
    assert chunks[0].text == section.text
    assert chunks[0].token_count > 0


def test_chunker_preserves_table_atomicity():
    meta = FilingMeta(
        cik="0001234567",
        ticker="ACME",
        accession_number="0001234567-24-000001",
        form_type="10-K",
        filed_date=date(2024, 2, 1),
        period_of_report=date(2023, 12, 31),
    )

    table_md = (
        "| Metric | 2023 | 2022 |\n"
        "| :--- | :--- | :--- |\n"
        "| Revenue | $100,000 | $90,000 |\n"
        "| Net Income | $25,000 | $20,000 |\n"
        "| EPS | $2.50 | $2.00 |"
    )

    # Paragraph of ~100 tokens
    p1 = "Paragraph one discusses market trends and operational expansions. " * 8
    p2 = "Paragraph two discusses regulatory compliance and capital expenditure. " * 8

    full_text = f"{p1}\n\n{table_md}\n\n{p2}"

    section = ParsedSection(
        meta=meta,
        item_id="item_7",
        item_title="Item 7. MD&A",
        text=full_text,
    )

    # Use small max_tokens so it forces multiple chunks
    chunker = Chunker(min_tokens=30, max_tokens=100)
    chunks = chunker.chunk_section(section)

    assert len(chunks) > 1

    # Verify that the table appears intact in exactly one chunk, never partially cut
    table_occurrences = [c for c in chunks if "Revenue" in c.text and "Net Income" in c.text]
    assert len(table_occurrences) == 1
    assert table_md in table_occurrences[0].text

    # Verify no chunk has an incomplete table header or split row
    for c in chunks:
        if "| Metric |" in c.text:
            assert "| EPS |" in c.text
