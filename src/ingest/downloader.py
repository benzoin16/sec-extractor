"""Download primary filing documents from EDGAR and cache them locally."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import httpx
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from src.models import FilingMeta, RawDocument

logger = logging.getLogger(__name__)

# Filing index JSON: maps accession number to list of documents in the filing.
INDEX_JSON_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/index.json"
)

# Primary document download URL.
DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{filename}"

# Document types that identify the primary 10-K / 10-Q HTML.
_PRIMARY_DOC_TYPES = {"10-K", "10-Q", "10-K/A", "10-Q/A", "10-KSB", "10-QSB"}


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

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _get(self, url: str) -> httpx.Response:
        if self.rate_limiter:
            self.rate_limiter()
        response = self._client.get(url)
        response.raise_for_status()
        return response

    def _find_primary_document(self, meta: FilingMeta) -> str | None:
        """Return the filename of the primary HTML document in the filing index.

        Strategy:
        1. Find a document whose *type* matches the form type (10-K, 10-Q, …)
           and whose name ends in ``.htm`` / ``.html``.
        2. If not found, fall back to any ``.htm``/``.html`` that isn't an exhibit.
        """
        acc_nodash = self._acc_nodash(meta.accession_number)
        index_url = INDEX_JSON_URL.format(cik=meta.cik, acc_nodash=acc_nodash)

        try:
            resp = self._get(index_url)
            index_data = resp.json()
        except Exception as exc:
            logger.warning("Could not fetch filing index for %s: %s", meta.accession_number, exc)
            return None

        documents: list[dict] = index_data.get("directory", {}).get("item", [])

        # Pass 1: exact type match + HTML extension.
        for doc in documents:
            dtype = doc.get("type", "")
            dname = doc.get("name", "")
            if dtype in _PRIMARY_DOC_TYPES and dname.lower().endswith((".htm", ".html")):
                return dname

        # Pass 2: any non-exhibit HTML.
        for doc in documents:
            dtype = doc.get("type", "")
            dname = doc.get("name", "")
            if dname.lower().endswith((".htm", ".html")) and not dtype.upper().startswith("EX-"):
                return dname

        logger.warning(
            "No primary HTML document found in filing index for %s %s",
            meta.ticker,
            meta.accession_number,
        )
        return None

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

        primary_doc = self._find_primary_document(meta)
        if not primary_doc:
            return None

        acc_nodash = self._acc_nodash(meta.accession_number)
        doc_url = DOC_URL.format(cik=meta.cik, acc_nodash=acc_nodash, filename=primary_doc)

        try:
            resp = self._get(doc_url)
        except Exception as exc:
            logger.error("Download failed for %s: %s", doc_url, exc)
            return None

        filing_dir = self._filing_dir(meta)
        filing_dir.mkdir(parents=True, exist_ok=True)
        primary_path.write_bytes(resp.content)
        self._meta_json_path(meta).write_text(meta.model_dump_json(indent=2), encoding="utf-8")

        logger.info(
            "Downloaded  %s %s %s → %s",
            meta.ticker,
            meta.form_type,
            meta.filed_date,
            primary_path,
        )
        return RawDocument(meta=meta, local_path=primary_path)
