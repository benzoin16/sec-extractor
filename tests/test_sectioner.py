"""Unit tests for Sectioner."""

from datetime import date
from pathlib import Path

from src.models import FilingMeta
from src.parse.sectioner import Sectioner


def test_sectioner_10k_avoids_toc():
    fixture_path = Path(__file__).parent / "fixtures" / "fixture_10k.html"
    html_content = fixture_path.read_text(encoding="utf-8")

    meta = FilingMeta(
        cik="0001234567",
        ticker="ACME",
        accession_number="0001234567-23-000001",
        form_type="10-K",
        filed_date=date(2024, 2, 15),
        period_of_report=date(2023, 12, 31),
    )

    sectioner = Sectioner()
    sections = sectioner.parse_sections(html_content, meta)

    item_ids = [s.item_id for s in sections]
    assert "item_1" in item_ids
    assert "item_1a" in item_ids
    assert "item_7" in item_ids
    assert "item_8" in item_ids

    # Verify Item 1A content
    item_1a = next(s for s in sections if s.item_id == "item_1a")
    assert "Global supply chain disruptions" in item_1a.text

    # Verify Item 8 contains markdown tables
    item_8 = next(s for s in sections if s.item_id == "item_8")
    assert "| Statement of Operations |" in item_8.text or "| Total revenue |" in item_8.text
    assert "383,285" in item_8.text


def test_sectioner_10q():
    fixture_path = Path(__file__).parent / "fixtures" / "fixture_10q.html"
    html_content = fixture_path.read_text(encoding="utf-8")

    meta = FilingMeta(
        cik="0001234567",
        ticker="ACME",
        accession_number="0001234567-24-000002",
        form_type="10-Q",
        filed_date=date(2024, 8, 5),
        period_of_report=date(2024, 6, 30),
    )

    sectioner = Sectioner()
    sections = sectioner.parse_sections(html_content, meta)

    item_ids = [s.item_id for s in sections]
    assert "item_1" in item_ids
    assert "item_2" in item_ids
    assert "item_1a" in item_ids

    item_1 = next(s for s in sections if s.item_id == "item_1")
    assert "$85,777" in item_1.text or "85,777" in item_1.text


def test_sectioner_fallback():
    html_content = "<html><body><p>This is a filing with no standard headers.</p></body></html>"
    meta = FilingMeta(
        cik="0001234567",
        ticker="ACME",
        accession_number="0001234567-24-000003",
        form_type="10-K",
        filed_date=date(2024, 2, 1),
        period_of_report=date(2023, 12, 31),
    )

    sectioner = Sectioner()
    sections = sectioner.parse_sections(html_content, meta)
    assert len(sections) == 1
    assert sections[0].item_id == "item_full"
    assert "This is a filing with no standard headers." in sections[0].text
