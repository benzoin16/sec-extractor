"""Sectioner: Splits 10-K and 10-Q inline-XBRL filings into standard SEC Items."""

from __future__ import annotations

import logging
import re

from bs4 import BeautifulSoup, Tag

from src.models import FilingMeta, ParsedSection
from src.parse.table_parser import TableParser

logger = logging.getLogger(__name__)

# Regex for Item headings:
# Matches e.g. "ITEM 1A. RISK FACTORS", "Item 7: Management's Discussion...", "Item 1 - Business"
_ITEM_HEADER_RE = re.compile(
    r"^\s*(?:PART\s+[I|V|X]+\s+)?ITEM\s+([0-9]{1,2}[A-Z]?)\.?\s*[:\-–—]?\s*(.*)$",
    re.IGNORECASE,
)

# Known 10-K items
_10K_ITEMS = {
    "1",
    "1A",
    "1B",
    "1C",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "7A",
    "8",
    "9",
    "9A",
    "9B",
    "9C",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
}

# Known 10-Q items
_10Q_ITEMS = {
    "1",
    "1A",
    "2",
    "3",
    "4",
    "5",
    "6",
}


class Sectioner:
    """Extract standard Item sections from SEC 10-K and 10-Q HTML documents."""

    def __init__(self, table_parser: TableParser | None = None) -> None:
        self.table_parser = table_parser or TableParser()

    def parse_sections(self, html_content: str, meta: FilingMeta) -> list[ParsedSection]:
        """Split HTML filing into Item sections and convert tables to markdown.

        Args:
            html_content: The raw HTML / iXBRL document string.
            meta: Metadata for the filing.

        Returns:
            List of :class:`ParsedSection` objects for each identified Item.
        """
        soup = BeautifulSoup(html_content, "lxml")

        # Remove scripts, styles, XML metadata tags that do not contain visible text
        for tag in soup.find_all(["script", "style", "noscript", "ix:header"]):
            tag.decompose()

        valid_items = _10K_ITEMS if meta.form_type.startswith("10-K") else _10Q_ITEMS
        candidates = self._find_item_headers(soup, valid_items)

        # Filter out Table of Contents (TOC) candidate matches
        body_headers = self._filter_toc(candidates)

        if not body_headers:
            logger.warning(
                "No Item headers found for %s %s. Falling back to entire document.",
                meta.ticker,
                meta.accession_number,
            )
            full_text = self._render_node_content(soup.body or soup)
            return [
                ParsedSection(
                    meta=meta,
                    item_id="item_full",
                    item_title="Full Filing",
                    text=full_text,
                )
            ]

        sections: list[ParsedSection] = []
        for i, header_info in enumerate(body_headers):
            item_id = header_info["item_id"]
            item_title = header_info["title"]
            start_node = header_info["node"]
            end_node = body_headers[i + 1]["node"] if i + 1 < len(body_headers) else None

            section_text = self._extract_between_nodes(start_node, end_node)
            if not section_text.strip():
                continue

            sections.append(
                ParsedSection(
                    meta=meta,
                    item_id=f"item_{item_id.lower()}",
                    item_title=item_title,
                    text=section_text.strip(),
                )
            )

        return sections

    def _find_item_headers(self, soup: BeautifulSoup, valid_items: set[str]) -> list[dict]:
        """Locate all elements matching Item header patterns."""
        candidates = []

        # Elements commonly carrying Item headings
        for tag in soup.find_all(
            ["p", "div", "b", "strong", "span", "h1", "h2", "h3", "h4", "font", "td"]
        ):
            text = tag.get_text(separator=" ", strip=True)
            if not text:
                continue

            # Short string check (headings are usually under 120 characters)
            if len(text) > 130:
                continue

            match = _ITEM_HEADER_RE.match(text)
            if match:
                item_num = match.group(1).upper()
                if item_num in valid_items:
                    # Prefer top-level block parent rather than inner <span> or <b>
                    block_parent = tag
                    while block_parent.parent and block_parent.parent.name in (
                        "b",
                        "strong",
                        "span",
                        "font",
                        "i",
                    ):
                        block_parent = block_parent.parent

                    candidates.append(
                        {
                            "item_id": item_num,
                            "title": text,
                            "node": block_parent,
                            "is_anchor": bool(tag.find_parent("a") or tag.find("a")),
                            "in_table": bool(tag.find_parent("table")),
                        }
                    )

        return candidates

    def _filter_toc(self, candidates: list[dict]) -> list[dict]:
        """Differentiate TOC entries from body section headers."""
        if not candidates:
            return []

        # If candidates appear multiple times (e.g. Item 1A in TOC, then in body),
        # TOC items typically come earlier and are inside <a> tags or table rows.
        # Group by item_id to see occurrences
        occurrences: dict[str, list[dict]] = {}
        for c in candidates:
            occurrences.setdefault(c["item_id"], []).append(c)

        # Deduplicate: if an item appears more than once, TOC is typically the first one
        # if it has anchors or is followed very closely by another item with no body content.
        filtered: list[dict] = []

        # Iterate in reverse or check distance
        # Standard pattern: TOC lists items consecutively within a few hundred chars
        # Body items have substantial text between them.
        for _item_id, items in occurrences.items():
            if len(items) == 1:
                filtered.append(items[0])
            else:
                # Prefer the one NOT in a TOC anchor and NOT in a multi-item TOC table
                body_candidate = None
                for candidate in reversed(items):
                    if not candidate["is_anchor"] and not candidate["in_table"]:
                        body_candidate = candidate
                        break
                if not body_candidate:
                    # Fall back to the last occurrence
                    body_candidate = items[-1]
                filtered.append(body_candidate)

        # Sort filtered candidates by document appearance order
        filtered.sort(key=lambda c: len(list(c["node"].previous_elements)))

        return filtered

    def _extract_between_nodes(self, start_node: Tag, end_node: Tag | None) -> str:
        """Extract text and markdown tables between two nodes in DOM order."""
        blocks: list[str] = []
        curr = start_node.next_element

        visited_tables: set[int] = set()

        while curr and curr != end_node:
            if isinstance(curr, Tag):
                if curr == end_node or (end_node and curr in end_node.descendants):
                    break

                if curr.name == "table":
                    if id(curr) not in visited_tables:
                        visited_tables.add(id(curr))
                        md_table = self.table_parser.parse_table(curr)
                        if md_table.strip():
                            blocks.append(f"\n\n{md_table}\n\n")
                    # Skip children of this table so we don't duplicate table text
                    curr = curr.next_sibling
                    continue

                if curr.name in ("p", "div", "h1", "h2", "h3", "h4", "h5", "li") and not any(
                    p.name == "table" for p in curr.parents
                ):
                    text = curr.get_text(separator=" ", strip=True)
                    if text:
                        blocks.append(f"\n\n{text}\n\n")
                    curr = curr.next_sibling
                    continue

            curr = curr.next_element

        joined = "".join(blocks)
        # Normalize multiple newlines
        joined = re.sub(r"\n{3,}", "\n\n", joined)
        return joined.strip()

    def _render_node_content(self, root: Tag) -> str:
        """Render an entire root node (tables -> markdown, text -> paragraphs)."""
        blocks: list[str] = []
        visited_tables: set[int] = set()

        for el in root.find_all(["table", "p", "div", "h1", "h2", "h3"]):
            if el.name == "table":
                if id(el) not in visited_tables:
                    visited_tables.add(id(el))
                    md = self.table_parser.parse_table(el)
                    if md.strip():
                        blocks.append(f"\n\n{md}\n\n")
            elif el.name in ("p", "div", "h1", "h2", "h3") and not any(
                p.name == "table" for p in el.parents
            ):
                text = el.get_text(separator=" ", strip=True)
                if text:
                    blocks.append(f"\n\n{text}\n\n")

        joined = "".join(blocks)
        return re.sub(r"\n{3,}", "\n\n", joined).strip()
