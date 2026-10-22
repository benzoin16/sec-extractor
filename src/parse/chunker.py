"""Token-bounded text chunker for SEC filings with table atomicity."""

from __future__ import annotations

import logging
import re

import tiktoken

from src.models import Chunk, ParsedSection

logger = logging.getLogger(__name__)


class Chunker:
    """Split parsed section markdown into 2-4k token chunks without breaking tables.

    Tables are treated as atomic units. If a single table exceeds `max_tokens`,
    it is placed into its own dedicated chunk rather than split.
    """

    _TABLE_PATTERN = re.compile(r"(\|[^\n]+\|\n\|[\s:\-|]+\|\n(?:\|[^\n]+\|\n*)+)", re.MULTILINE)

    def __init__(
        self,
        min_tokens: int = 1500,
        max_tokens: int = 3500,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.encoding_name = encoding_name
        self._tokenizer = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a string."""
        return len(self._tokenizer.encode(text, disallowed_special=()))

    def _split_into_blocks(self, text: str) -> list[tuple[str, bool]]:
        """Split text into alternating blocks of (content, is_table)."""
        blocks: list[tuple[str, bool]] = []
        last_end = 0

        for match in self._TABLE_PATTERN.finditer(text):
            start, end = match.span()
            # Non-table text before this table
            pre_text = text[last_end:start].strip()
            if pre_text:
                # Split regular text into paragraphs
                paragraphs = [p.strip() for p in re.split(r"\n\s*\n", pre_text) if p.strip()]
                for p in paragraphs:
                    blocks.append((p, False))

            # Table block
            table_text = match.group(0).strip()
            if table_text:
                blocks.append((table_text, True))

            last_end = end

        # Remaining non-table text after last table
        post_text = text[last_end:].strip()
        if post_text:
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", post_text) if p.strip()]
            for p in paragraphs:
                blocks.append((p, False))

        return blocks

    def chunk_section(self, section: ParsedSection) -> list[Chunk]:
        """Divide a ParsedSection into token-bounded Chunk objects."""
        blocks = self._split_into_blocks(section.text)
        if not blocks:
            # If section text exists but no blocks formed
            token_count = self.count_tokens(section.text)
            return [
                Chunk(
                    meta=section.meta,
                    item_id=section.item_id,
                    chunk_index=0,
                    total_chunks=1,
                    text=section.text,
                    token_count=token_count,
                )
            ]

        raw_chunks: list[str] = []
        current_blocks: list[str] = []
        current_tokens = 0

        for content, is_table in blocks:
            block_tokens = self.count_tokens(content)

            # If adding this block exceeds max_tokens:
            if current_tokens + block_tokens > self.max_tokens:
                # If we already have accumulated content, flush it as a chunk
                if current_blocks:
                    raw_chunks.append("\n\n".join(current_blocks))
                    current_blocks = []
                    current_tokens = 0

                # If this single block itself exceeds max_tokens:
                if block_tokens > self.max_tokens:
                    if is_table:
                        # CRITICAL: Do NOT split tables! Keep the entire table in its own chunk.
                        raw_chunks.append(content)
                        continue
                    else:
                        # Split huge text paragraph by sentences
                        sub_sentences = re.split(r"(?<=[.!?])\s+", content)
                        sub_accum: list[str] = []
                        sub_tok = 0
                        for sent in sub_sentences:
                            stok = self.count_tokens(sent)
                            if sub_tok + stok > self.max_tokens and sub_accum:
                                raw_chunks.append(" ".join(sub_accum))
                                sub_accum = []
                                sub_tok = 0
                            sub_accum.append(sent)
                            sub_tok += stok
                        if sub_accum:
                            current_blocks.append(" ".join(sub_accum))
                            current_tokens += sub_tok
                        continue

            current_blocks.append(content)
            current_tokens += block_tokens

        if current_blocks:
            raw_chunks.append("\n\n".join(current_blocks))

        total_chunks = len(raw_chunks)
        chunks: list[Chunk] = []

        for i, chunk_text in enumerate(raw_chunks):
            chunk_tokens = self.count_tokens(chunk_text)
            chunks.append(
                Chunk(
                    meta=section.meta,
                    item_id=section.item_id,
                    chunk_index=i,
                    total_chunks=total_chunks,
                    text=chunk_text,
                    token_count=chunk_tokens,
                )
            )

        return chunks
