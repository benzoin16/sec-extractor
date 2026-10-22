"""Fetch and cache SEC companyfacts JSON files from data.sec.gov."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

import httpx
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


class CompanyFactsFetcher:
    """Download companyfacts JSON from EDGAR and cache locally.

    Cache layout::

        cache_dir/
          {CIK}/
            companyfacts.json

    Caching is idempotent: if ``companyfacts.json`` exists locally,
    the network call is skipped.

    Args:
        cache_dir: Root cache directory (usually ``data/raw``).
        user_agent: User-Agent header string required by EDGAR.
        rate_limiter: Optional callable invoked before each HTTP call.
    """

    def __init__(
        self,
        cache_dir: Path,
        user_agent: str,
        rate_limiter: Callable[[], None] | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.user_agent = user_agent
        self.rate_limiter = rate_limiter
        self._client = httpx.Client(
            headers={"User-Agent": self.user_agent},
            timeout=45,
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> CompanyFactsFetcher:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _facts_path(self, cik: str) -> Path:
        return self.cache_dir / cik.zfill(10) / "companyfacts.json"

    def is_cached(self, cik: str) -> bool:
        return self._facts_path(cik).exists()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _fetch_from_api(self, cik: str) -> dict:
        url = COMPANY_FACTS_URL.format(cik=cik.zfill(10))
        if self.rate_limiter:
            self.rate_limiter()

        logger.info("Fetching company facts from %s", url)
        resp = self._client.get(url)
        resp.raise_for_status()
        return resp.json()

    def fetch(self, cik: str) -> dict:
        """Return the companyfacts dictionary for CIK, loading from cache if available."""
        path = self._facts_path(cik)
        if path.exists():
            logger.debug("Cache hit for companyfacts CIK %s: %s", cik, path)
            return json.loads(path.read_text(encoding="utf-8"))

        data = self._fetch_from_api(cik)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Saved companyfacts CIK %s to %s", cik, path)
        return data
