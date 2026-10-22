"""Fetch lists of 10-K / 10-Q filings from the EDGAR submissions API."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date
from itertools import zip_longest

import httpx
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from src.models import FilingMeta

logger = logging.getLogger(__name__)

# Primary submissions endpoint (one JSON per entity).
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

# Overflow / pagination files referenced from the primary JSON.
SUBMISSIONS_EXTRA_URL = "https://data.sec.gov/submissions/{filename}"


class FilingFetcher:
    """Fetch filing metadata from the EDGAR submissions API.

    The EDGAR submissions API returns up to ~1 000 recent filings in the
    primary JSON.  Older filings live in supplemental JSON files listed under
    ``filings.files``.  This class handles both transparently.

    Args:
        user_agent: ``User-Agent`` header (SEC_USER_AGENT env var).
        rate_limiter: Optional callable invoked before each HTTP request.
    """

    def __init__(
        self,
        user_agent: str,
        rate_limiter: Callable[[], None] | None = None,
    ) -> None:
        self.user_agent = user_agent
        self.rate_limiter = rate_limiter
        self._client = httpx.Client(
            headers={"User-Agent": self.user_agent},
            timeout=30,
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> FilingFetcher:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _get_json(self, url: str) -> dict:
        if self.rate_limiter:
            self.rate_limiter()
        response = self._client.get(url)
        response.raise_for_status()
        return response.json()

    def _parse_filings_block(
        self,
        block: dict,
        cik: str,
        ticker: str,
        start_date: date,
        end_date: date,
        form_types: set[str],
        fiscal_year_end: str | None,
    ) -> list[FilingMeta]:
        """Parse one filings block (either ``recent`` or an overflow file).

        The primary JSON nests filings under ``filings.recent``.
        Overflow files have the arrays at the top level.  We handle both by
        looking for a ``recent`` key and falling back to the block itself.
        """
        recent = block.get("recent", block)

        accessions: list[str] = recent.get("accessionNumber", [])
        forms: list[str] = recent.get("form", [])
        filed_dates: list[str] = recent.get("filingDate", [])
        periods: list[str] = recent.get("reportDate", [])

        results: list[FilingMeta] = []

        for acc, form, filed_str, period_str in zip_longest(
            accessions, forms, filed_dates, periods, fillvalue=None
        ):
            if acc is None:
                break
            if form not in form_types:
                continue
            try:
                filed = date.fromisoformat(str(filed_str))
            except (ValueError, TypeError):
                continue
            if not (start_date <= filed <= end_date):
                continue
            try:
                period = date.fromisoformat(str(period_str)) if period_str else filed
            except (ValueError, TypeError):
                period = filed

            results.append(
                FilingMeta(
                    cik=cik,
                    ticker=ticker,
                    accession_number=str(acc),
                    form_type=str(form),
                    filed_date=filed,
                    period_of_report=period,
                    fiscal_year_end=fiscal_year_end,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_filings(
        self,
        cik: str,
        ticker: str,
        start_date: date,
        end_date: date,
        form_types: set[str] | None = None,
    ) -> list[FilingMeta]:
        """Return all matching filings sorted by ``filed_date`` ascending.

        Args:
            cik: Zero-padded 10-digit CIK string.
            ticker: Ticker symbol (used for logging / metadata only).
            start_date: Earliest filing date to include (inclusive).
            end_date: Latest filing date to include (inclusive).
            form_types: Set of form types to keep (default ``{"10-K", "10-Q"}``).

        Returns:
            Sorted list of :class:`FilingMeta` within the date range.
        """
        if form_types is None:
            form_types = {"10-K", "10-Q"}

        url = SUBMISSIONS_URL.format(cik=cik)
        data = self._get_json(url)

        # Fiscal year end lives at the top level of the submission JSON.
        fiscal_year_end: str | None = data.get("fiscalYearEnd")

        results = self._parse_filings_block(
            data.get("filings", {}),
            cik,
            ticker,
            start_date,
            end_date,
            form_types,
            fiscal_year_end,
        )

        # Handle pagination: older filings overflow into separate JSON files.
        for extra_file in data.get("filings", {}).get("files", []):
            filename = extra_file.get("name", "")
            if not filename:
                continue
            extra_url = SUBMISSIONS_EXTRA_URL.format(filename=filename)
            try:
                extra_data = self._get_json(extra_url)
                results.extend(
                    self._parse_filings_block(
                        extra_data,
                        cik,
                        ticker,
                        start_date,
                        end_date,
                        form_types,
                        fiscal_year_end,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not fetch supplemental filings file %s for %s: %s",
                    filename,
                    ticker,
                    exc,
                )

        results.sort(key=lambda f: f.filed_date)
        logger.info(
            "Found %d %s filings for %s (CIK %s)",
            len(results),
            "/".join(sorted(form_types)),
            ticker,
            cik,
        )
        return results
