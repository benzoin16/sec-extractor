"""Resolve ticker symbols to SEC CIK numbers using EDGAR's company_tickers.json."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_CACHE_FILENAME = "company_tickers.json"


class UnknownTickerError(ValueError):
    """Raised when a ticker is not found in EDGAR's company ticker map."""


class CIKResolver:
    """Maps ticker symbols to zero-padded 10-digit CIK strings.

    The company_tickers.json file is downloaded once and cached to
    ``cache_dir / company_tickers.json``. Subsequent calls are served from
    the cache without any HTTP requests.

    Args:
        cache_dir: Directory where the ticker map is cached.
        user_agent: ``User-Agent`` header value required by EDGAR fair-access
            policy.  Read from ``SEC_USER_AGENT`` env var in the CLI.
        rate_limiter: Optional callable invoked before each HTTP request.
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
        self._ticker_to_cik: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _cache_path(self) -> Path:
        return self.cache_dir / _CACHE_FILENAME

    def _fetch_raw(self) -> dict:
        """Return the raw company_tickers dict, using the local cache if present."""
        if self._cache_path.exists():
            logger.debug("Loading CIK map from cache: %s", self._cache_path)
            return json.loads(self._cache_path.read_text(encoding="utf-8"))

        if self.rate_limiter:
            self.rate_limiter()

        logger.info("Fetching company ticker map from EDGAR…")
        with httpx.Client(
            headers={"User-Agent": self.user_agent},
            timeout=30,
            follow_redirects=True,
        ) as client:
            response = client.get(COMPANY_TICKERS_URL)
            response.raise_for_status()

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(response.text, encoding="utf-8")
        logger.info("Saved CIK map to %s", self._cache_path)
        return response.json()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Populate the in-memory ticker → CIK mapping (idempotent)."""
        raw = self._fetch_raw()
        self._ticker_to_cik = {
            entry["ticker"].upper(): str(entry["cik_str"]).zfill(10) for entry in raw.values()
        }
        logger.info("Loaded %d tickers into CIK map", len(self._ticker_to_cik))

    def resolve(self, ticker: str) -> str:
        """Return the zero-padded 10-digit CIK for *ticker*.

        Raises:
            UnknownTickerError: If the ticker is not in the EDGAR map.
        """
        ticker = ticker.upper().strip()
        if not self._ticker_to_cik:
            self.load()
        if ticker not in self._ticker_to_cik:
            raise UnknownTickerError(
                f"Ticker {ticker!r} not found in EDGAR company ticker map. "
                "Check the spelling or whether the company is SEC-registered."
            )
        return self._ticker_to_cik[ticker]

    def resolve_many(self, tickers: list[str]) -> dict[str, str]:
        """Resolve a list of tickers, returning a ``{TICKER: CIK}`` dict.

        Raises:
            UnknownTickerError: On the first unresolvable ticker.
        """
        if not self._ticker_to_cik:
            self.load()
        return {t.upper(): self.resolve(t) for t in tickers}
