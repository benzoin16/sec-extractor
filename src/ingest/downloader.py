"""Download primary filing documents from EDGAR and cache them locally."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

import httpx
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from src.models import FilingMeta, RawDocument

logger = logging.getLogger(__name__)

# Primary document download URL.
DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{filename}"
_MIN_DOCUMENT_SIZE = 100 * 1024
_SEC_ERROR_MARKERS = (
    "request rate threshold exceeded",
    "your request originates from an undetected automated tool",
    "access denied",
)


class Downloader:
    """Download the primary HTML document for a filing and cache it.

    Cache layout::

        output_dir/
          {CIK}/
            {accession_nodash}/
              primary.html   ← the iXBRL filing document
              meta.json      ← serialised FilingMeta (for re-parsing later)

    Caching is **idempotent**: if ``primary.html`` already exists the download
    is skipped entirely and the cached :class:`RawDocument` is returned.

    Args:
        output_dir: Root of the cache tree (``data/raw``).
        user_agent: ``User-Agent`` header value required by EDGAR.
        rate_limiter: Optional callable invoked before each HTTP request.
    """

    def __init__(
        self,
        output_dir: Path,
        user_agent: str,
        rate_limiter: Callable[[], None] | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.user_agent = user_agent
        self.rate_limiter = rate_limiter
        self._client = httpx.Client(
            headers={"User-Agent": self.user_agent},
            timeout=60,
            follow_redirects=True,
            event_hooks={"request": [self._on_request]},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Downloader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _acc_nodash(accession_number: str) -> str:
        return accession_number.replace("-", "")

    def _filing_dir(self, meta: FilingMeta) -> Path:
        return self.output_dir / meta.cik / self._acc_nodash(meta.accession_number)

    def _primary_html_path(self, meta: FilingMeta) -> Path:
        return self._filing_dir(meta) / "primary.html"

    def _meta_json_path(self, meta: FilingMeta) -> Path:
        return self._filing_dir(meta) / "meta.json"

    def _on_request(self, _: httpx.Request) -> None:
        if self.rate_limiter:
            self.rate_limiter()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _get(self, url: str) -> httpx.Response:
        response = self._client.get(url)
        response.raise_for_status()
        return response

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_cached(self, meta: FilingMeta) -> bool:
        """Return ``True`` if the primary HTML for this filing is already on disk."""
        return self._primary_html_path(meta).exists()

    def download(self, meta: FilingMeta) -> RawDocument | None:
        """Download (or return cached) the primary filing document.

        Returns:
            :class:`RawDocument` on success, ``None`` if the primary document
            could not be located or downloaded.
        """
        primary_path = self._primary_html_path(meta)

        if self.is_cached(meta):
            logger.debug("Cache hit — skipping download: %s", primary_path)
            return RawDocument(meta=meta, local_path=primary_path)

        primary_doc = meta.primary_doc
        if not primary_doc:
            raise ValueError(f"Missing primaryDocument for {meta.accession_number}")

        acc_nodash = self._acc_nodash(meta.accession_number)
        cik_int = str(int(meta.cik))
        doc_url = DOC_URL.format(cik=cik_int, acc_nodash=acc_nodash, filename=primary_doc)

        try:
            resp = self._get(doc_url)
        except Exception as exc:
            logger.error("Download failed for %s: %s", doc_url, exc)
            raise

        content_type = resp.headers.get("content-type", "").lower()
        body_lower = resp.content[:10000].lower()
        if len(resp.content) < _MIN_DOCUMENT_SIZE:
            raise ValueError(f"Downloaded filing is under 100 KiB: {doc_url}")
        if not (".htm" in primary_doc.lower() or "html" in content_type):
            raise ValueError(f"Downloaded filing is not HTML: {doc_url}")
        if "html" not in content_type and b"<html" not in body_lower and b"<!doctype" not in body_lower:
            raise ValueError(f"Downloaded filing is not HTML: {doc_url}")
        if any(marker.encode() in body_lower for marker in _SEC_ERROR_MARKERS):
            raise ValueError(f"SEC error/rate-limit response received: {doc_url}")

        filing_dir = self._filing_dir(meta)
        filing_dir.mkdir(parents=True, exist_ok=True)
        primary_path.write_bytes(resp.content)
        metadata = meta.model_dump(mode="json")
        metadata["url"] = doc_url
        self._meta_json_path(meta).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        logger.info(
            "Downloaded  %s %s %s → %s",
            meta.ticker,
            meta.form_type,
            meta.filed_date,
            primary_path,
        )
        return RawDocument(meta=meta, local_path=primary_path)
