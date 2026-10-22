"""HTML table to GitHub-Flavored Markdown (GFM) converter for SEC filings.

SEC financial statements and disclosure tables have complex markup:
- Non-standard layout tables with spacer columns
- Multi-row headers and merged columns (`colspan`, `rowspan`)
- Inline-XBRL tags (`<ix:nonFraction>`, `<ix:continuation>`, etc.)
- Dollar signs placed in separate columns adjacent to numeric values
- Parenthesized negative numbers: `(1,234)`

This module extracts the underlying grid structure and formats clean GFM tables.
"""

from __future__ import annotations

import re

from bs4 import NavigableString, Tag


class TableParser:
    """Parse HTML tables from SEC filings into clean GitHub-Flavored Markdown."""

    _CURRENCY_SYMBOLS = {"$", "€", "£", "¥", "₹"}

    def __init__(self, min_cols: int = 1, max_empty_col_ratio: float = 0.85) -> None:
        self.min_cols = min_cols
        self.max_empty_col_ratio = max_empty_col_ratio

    @classmethod
    def clean_cell_text(cls, cell: Tag | NavigableString | None) -> str:
        """Extract and normalize text from a table cell."""
        if cell is None:
            return ""
        if isinstance(cell, NavigableString):
            raw = str(cell)
        else:
            raw = cell.get_text(separator=" ", strip=True)

        # Normalize non-breaking spaces and whitespace
        text = raw.replace("\xa0", " ").replace("\u200b", "").replace("\n", " ")
        text = re.sub(r"\s+", " ", text).strip()
        # Escape pipe symbols inside markdown cells
        text = text.replace("|", "\\|")
        return text

    def table_to_matrix(self, table: Tag) -> list[list[str]]:
        """Convert a BeautifulSoup <table> tag into a 2D list of strings.

        Properly propagates spanned cells (`colspan`, `rowspan`).
        """
        rows = table.find_all("tr", recursive=True)
        if not rows:
            return []

        # Determine dimensions and build matrix
        matrix: list[list[str | None]] = []

        for row_idx, tr in enumerate(rows):
            while len(matrix) <= row_idx:
                matrix.append([])

            col_idx = 0
            cells = tr.find_all(["td", "th"], recursive=False)

            for cell in cells:
                # Advance to next available column not occupied by an earlier rowspan
                while col_idx < len(matrix[row_idx]) and matrix[row_idx][col_idx] is not None:
                    col_idx += 1

                try:
                    rowspan = int(cell.get("rowspan", 1))
                except (ValueError, TypeError):
                    rowspan = 1

                try:
                    colspan = int(cell.get("colspan", 1))
                except (ValueError, TypeError):
                    colspan = 1

                # Restrict runaway spans
                rowspan = max(1, min(rowspan, 100))
                colspan = max(1, min(colspan, 50))

                cell_text = self.clean_cell_text(cell)

                for r in range(rowspan):
                    target_r = row_idx + r
                    while len(matrix) <= target_r:
                        matrix.append([])

                    for c in range(colspan):
                        target_c = col_idx + c
                        while len(matrix[target_r]) <= target_c:
                            matrix[target_r].append(None)
                        # Put text in the top-left cell; leave secondary spanned cells empty
                        matrix[target_r][target_c] = cell_text if (r == 0 and c == 0) else ""

                col_idx += colspan

        # Fill any remaining None values with empty string
        max_cols = max((len(r) for r in matrix), default=0)
        clean_matrix: list[list[str]] = []
        for r in matrix:
            row_padded = [c if c is not None else "" for c in r]
            if len(row_padded) < max_cols:
                row_padded.extend([""] * (max_cols - len(row_padded)))
            clean_matrix.append(row_padded)

        return clean_matrix

    def prune_matrix(self, matrix: list[list[str]]) -> list[list[str]]:
        """Remove empty rows, spacer columns, and merge detached currency symbols."""
        if not matrix or not matrix[0]:
            return []

        # 1. Filter out completely empty rows
        matrix = [row for row in matrix if any(cell.strip() for cell in row)]
        if not matrix:
            return []

        num_rows = len(matrix)
        num_cols = len(matrix[0])

        # 2. Identify spacer/empty columns
        keep_cols: list[int] = []
        for col_idx in range(num_cols):
            non_empty_count = sum(1 for r in range(num_rows) if matrix[r][col_idx].strip())
            if non_empty_count > 0:
                keep_cols.append(col_idx)

        if not keep_cols:
            return []

        filtered_matrix = [[matrix[r][c] for c in keep_cols] for r in range(num_rows)]

        # 3. Merge isolated currency columns into adjacent numbers
        # e.g., ['$'] followed by ['1,234'] -> ['$1,234']
        num_cols = len(filtered_matrix[0])
        cols_to_drop: set[int] = set()

        for c in range(num_cols - 1):
            curr_col = [filtered_matrix[r][c].strip() for r in range(num_rows)]
            is_currency_col = all(
                val in self._CURRENCY_SYMBOLS or val == "" for val in curr_col
            ) and any(val in self._CURRENCY_SYMBOLS for val in curr_col)
            if is_currency_col:
                # Merge into next column
                for r in range(num_rows):
                    sym = filtered_matrix[r][c].strip()
                    val = filtered_matrix[r][c + 1].strip()
                    if sym and val:
                        filtered_matrix[r][c + 1] = f"{sym}{val}"
                    elif sym and not val:
                        filtered_matrix[r][c + 1] = sym
                cols_to_drop.add(c)

        if cols_to_drop:
            final_cols = [c for c in range(num_cols) if c not in cols_to_drop]
            filtered_matrix = [[filtered_matrix[r][c] for c in final_cols] for r in range(num_rows)]

        return filtered_matrix

    def render_markdown(self, matrix: list[list[str]]) -> str:
        """Render a 2D matrix as GitHub Flavored Markdown."""
        if not matrix:
            return ""

        num_cols = len(matrix[0])
        if num_cols < self.min_cols:
            return ""

        # Use first non-empty row as header
        header = matrix[0]
        # Ensure header cells are not all empty
        if not any(h.strip() for h in header):
            header = [f"Col {i + 1}" for i in range(num_cols)]
            data_rows = matrix
        else:
            # Replace empty header labels with placeholder if some headers exist
            header = [h if h.strip() else f"Col {i + 1}" for i, h in enumerate(header)]
            data_rows = matrix[1:]

        # Markdown header row
        lines: list[str] = []
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join([":---"] * num_cols) + " |")

        for row in data_rows:
            # Clean each cell: no embedded newlines
            clean_row = [cell.replace("\n", " ").strip() for cell in row]
            lines.append("| " + " | ".join(clean_row) + " |")

        return "\n".join(lines)

    def parse_table(self, table: Tag) -> str:
        """Parse a BeautifulSoup table tag into a GFM markdown table."""
        matrix = self.table_to_matrix(table)
        pruned = self.prune_matrix(matrix)
        if not pruned:
            return ""
        return self.render_markdown(pruned)
