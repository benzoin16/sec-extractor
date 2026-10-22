"""Pydantic models for SEC XBRL ground truth facts and labels."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

PeriodType = Literal["annual", "quarter", "ytd", "instant"]


class FactRecord(BaseModel):
    """A single raw fact entry extracted from SEC companyfacts JSON."""

    accn: str = Field(..., description="Accession number where this fact was reported")
    fy: int | None = Field(None, description="Fiscal year")
    fp: str | None = Field(None, description="Fiscal period, e.g. FY, Q1, Q2, Q3")
    form: str | None = Field(None, description="Form type, e.g. 10-K, 10-Q")
    filed: str | None = Field(None, description="Filing date string YYYY-MM-DD")
    start: str | None = Field(None, description="Period start date string for durations")
    end: str = Field(..., description="Period end date string")
    val: float | int = Field(..., description="Unscaled raw numeric value")
    unit: str = Field(..., description="Reporting unit, e.g. USD, USD/shares")
    tag: str = Field(..., description="XBRL taxonomy tag, e.g. Revenues")
    is_derived: bool = Field(False, description="True if value was calculated from multiple tags")


class ConceptValue(BaseModel):
    """An extracted value for a target concept, tagged with its period type."""

    value: float | int = Field(..., description="Unscaled numeric value")
    period_type: PeriodType = Field(..., description="'annual', 'quarter', 'ytd', or 'instant'")
    tag: str = Field(..., description="XBRL tag used")
    unit: str = Field("USD", description="Unit of measurement")
    is_derived: bool = Field(False, description="True if calculated via derivation fallback")


class TargetMetric(BaseModel):
    """Output representation of an extracted metric in the training dataset target JSON."""

    value: float | int = Field(..., description="Unscaled numeric value")
    period_type: PeriodType = Field(..., description="'annual', 'quarter', 'ytd', or 'instant'")


class FilingLabels(BaseModel):
    """Joined ground-truth labels for a single filing."""

    cik: str
    ticker: str
    accession_number: str
    form_type: str
    period_of_report: date
    fiscal_year: int | None = None
    # Maps concept name (e.g. 'revenue') to one or more candidate values (e.g. quarter and ytd)
    metrics: dict[str, list[ConceptValue]] = Field(default_factory=dict)
