"""Shared Pydantic models used across all pipeline stages."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field


class FilingMeta(BaseModel):
    """Metadata for a single SEC filing (10-K or 10-Q)."""

    cik: str = Field(..., description="Zero-padded 10-digit CIK, e.g. '0000320193'")
    ticker: str = Field(..., description="Uppercase ticker symbol, e.g. 'AAPL'")
    accession_number: str = Field(
        ..., description="EDGAR accession number, e.g. '0000320193-23-000106'"
    )
    form_type: str = Field(..., description="'10-K' or '10-Q'")
    filed_date: date = Field(..., description="Date the filing was submitted to EDGAR")
    period_of_report: date = Field(..., description="Fiscal period the filing covers")
    fiscal_year_end: str | None = Field(
        None, description="Fiscal year end as MMDD, e.g. '0930' for Sep 30"
    )


class RawDocument(BaseModel):
    """A cached filing document on local disk."""

    meta: FilingMeta
    local_path: Path = Field(..., description="Absolute path to the cached primary HTML file")


class ParsedSection(BaseModel):
    """One Item section extracted from a filing (e.g. Item 1A, Item 7)."""

    meta: FilingMeta
    item_id: str = Field(..., description="Normalized item key, e.g. 'item_1a', 'item_7'")
    item_title: str = Field(..., description="Full heading text as found in the filing")
    text: str = Field(..., description="Section text with tables rendered as GFM markdown")


class Chunk(BaseModel):
    """A token-bounded chunk ready for model training / inference."""

    meta: FilingMeta
    item_id: str
    chunk_index: int = Field(..., description="0-based index within the parent section")
    total_chunks: int = Field(0, description="Total chunks in the parent section (filled post-hoc)")
    text: str
    token_count: int = Field(..., description="Token count using the configured tokenizer")
