"""Accession retrieval, caching, parsing, and chunking pipeline for the serve gateway."""

from __future__ import annotations

import json
import logging
import os
from datetime import date
from pathlib import Path

from src.ingest.downloader import Downloader
from src.models import Chunk, FilingMeta
from src.parse.chunker import Chunker
from src.parse.sectioner import Sectioner
from src.parse.table_parser import TableParser

logger = logging.getLogger(__name__)


class AccessionPipeline:
    """Resolves an accession number to financial text chunks using disk caches."""

    def __init__(
        self,
        raw_dir: Path | str = "data/raw",
        processed_dir: Path | str = "data/processed",
        downloader: Downloader | None = None,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        user_agent = os.getenv("SEC_USER_AGENT", "EdgarLENS user@example.com")
        self.downloader = downloader or Downloader(output_dir=self.raw_dir, user_agent=user_agent)
        self.sectioner = Sectioner(table_parser=TableParser())
        self.chunker = Chunker(max_tokens=3000)

    def _find_cached_chunks(self, accession_number: str) -> list[Chunk] | None:
        """Search data/processed for pre-chunked filing files matching accession number."""
        clean_accn = accession_number.replace("-", "")
        candidates = list(self.processed_dir.glob(f"*{accession_number}*_chunks.json"))
        if not candidates:
            candidates = list(self.processed_dir.glob(f"*{clean_accn}*_chunks.json"))

        if candidates:
            chunk_file = candidates[0]
            logger.info("Found cached chunks at %s", chunk_file)
            data = json.loads(chunk_file.read_text(encoding="utf-8"))
            return [Chunk.model_validate(c) for c in data]

        return None

    def _find_cached_raw_file(self, accession_number: str) -> Path | None:
        """Search data/raw for downloaded filing HTML."""
        clean_accn = accession_number.replace("-", "")
        for ext in ("*.htm", "*.html"):
            for match in self.raw_dir.rglob(f"*{accession_number}*{ext}"):
                return match
            for match in self.raw_dir.rglob(f"*{clean_accn}*{ext}"):
                return match
        return None

    async def get_text_for_accession(
        self,
        accession_number: str,
        ticker: str | None = None,
    ) -> str:
        """Retrieve primary financial statements text for an accession number."""
        # 1. Check if chunks already exist in cache
        cached_chunks = self._find_cached_chunks(accession_number)
        if cached_chunks:
            return self._select_best_financial_chunk(cached_chunks)

        # 2. Check if raw file exists in data/raw
        raw_file = self._find_cached_raw_file(accession_number)

        meta = FilingMeta(
            cik="0000000000",
            ticker=ticker or "UNKNOWN",
            accession_number=accession_number,
            form_type="10-K",
            filed_date=date.today(),
            period_of_report=date.today(),
        )

        # 3. If raw file not present, download it
        if not raw_file:
            if not ticker:
                raise ValueError(
                    f"Accession {accession_number} not found in local cache, and no ticker "
                    f"was provided to fetch from EDGAR."
                )
            logger.info("Fetching filing for %s (%s) from EDGAR...", ticker, accession_number)
            raw_doc = self.downloader.download(meta)
            if raw_doc and raw_doc.local_path.exists():
                raw_file = raw_doc.local_path

        if not raw_file or not raw_file.exists():
            raise FileNotFoundError(
                f"Could not locate or download filing for accession {accession_number}"
            )

        # 4. Parse & Section
        html_content = raw_file.read_text(encoding="utf-8", errors="replace")
        sections = self.sectioner.parse_sections(html_content, meta=meta)

        # 5. Chunk
        chunks: list[Chunk] = []
        for sec in sections:
            chunks.extend(self.chunker.chunk_section(sec))

        if not chunks:
            raise ValueError(f"No chunks parsed from filing {accession_number}")

        # Cache chunks
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.processed_dir / f"{accession_number}_chunks.json"
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump([c.model_dump(mode="json") for c in chunks], f, indent=2)

        return self._select_best_financial_chunk(chunks)

    def _select_best_financial_chunk(self, chunks: list[Chunk]) -> str:
        """Select the chunk most likely containing income statement and balance sheet."""
        if not chunks:
            raise ValueError("No chunks parsed from filing.")

        # Prioritize item_8 (Financial Statements) or item_7
        item_8_chunks = [c for c in chunks if c.item_id == "item_8"]
        if item_8_chunks:
            # Pick first chunk containing table markdown with revenue keywords
            for c in item_8_chunks:
                if "|" in c.text and ("revenue" in c.text.lower() or "sales" in c.text.lower()):
                    return c.text
            return item_8_chunks[0].text

        item_7_chunks = [c for c in chunks if c.item_id == "item_7"]
        if item_7_chunks:
            return item_7_chunks[0].text

        # Fallback to the largest chunk with table content
        chunks_with_tables = [c for c in chunks if "|" in c.text]
        if chunks_with_tables:
            return max(chunks_with_tables, key=lambda c: len(c.text)).text

        return chunks[0].text
