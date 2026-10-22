"""Company + Time dataset splitter with strict zero-leakage verification."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from src.models import FilingMeta


class DatasetSplitter:
    """Split filings by Company (out-of-distribution test holdout) and Time (temporal validation).

    Rules:
    1. Test companies: 100% of filings across all years belong to 'test'.
    2. Training companies:
       - Filings from ``val_fiscal_year`` belong to 'val'.
       - Filings from earlier fiscal years belong to 'train'.
    """

    def __init__(
        self,
        test_tickers: list[str],
        val_fiscal_year: int = 2024,
    ) -> None:
        self.test_tickers = {t.upper().strip() for t in test_tickers}
        self.val_fiscal_year = val_fiscal_year

    def get_filing_year(self, meta: FilingMeta) -> int:
        """Extract fiscal year for split determination."""
        return meta.period_of_report.year

    def split_filings(
        self,
        filings: list[FilingMeta],
    ) -> dict[str, list[FilingMeta]]:
        """Partition filings into train, val, and test splits."""
        splits: dict[str, list[FilingMeta]] = {
            "train": [],
            "val": [],
            "test": [],
        }

        # Deduplicate filings by accession number
        seen_accns: set[str] = set()
        unique_filings: list[FilingMeta] = []
        for f in filings:
            if f.accession_number not in seen_accns:
                seen_accns.add(f.accession_number)
                unique_filings.append(f)

        for meta in unique_filings:
            ticker = meta.ticker.upper().strip()

            # 1. Company holdout: entire company history is held out for test
            if ticker in self.test_tickers:
                splits["test"].append(meta)
                continue

            # 2. Temporal holdout for training companies
            year = self.get_filing_year(meta)
            if year >= self.val_fiscal_year:
                splits["val"].append(meta)
            else:
                splits["train"].append(meta)

        return splits

    def generate_manifest(
        self,
        splits: dict[str, list[FilingMeta]],
        output_path: Path | None = None,
    ) -> dict:
        """Create and optionally save the split manifest JSON."""
        manifest = {
            "created_at": datetime.now(UTC).isoformat(),
            "test_tickers": sorted(self.test_tickers),
            "val_fiscal_year": self.val_fiscal_year,
            "counts": {
                "train_filings": len(splits["train"]),
                "val_filings": len(splits["val"]),
                "test_filings": len(splits["test"]),
            },
            "splits": {
                split_name: [f.model_dump(mode="json") for f in metas]
                for split_name, metas in splits.items()
            },
        }

        # Validate no leakage before saving
        self.assert_no_leak(manifest)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return manifest

    @classmethod
    def assert_no_leak(cls, manifest: dict) -> None:
        """Strictly assert no company or filing leaks across splits."""
        train_filings = manifest["splits"]["train"]
        val_filings = manifest["splits"]["val"]
        test_filings = manifest["splits"]["test"]

        train_tickers = {f["ticker"].upper() for f in train_filings}
        val_tickers = {f["ticker"].upper() for f in val_filings}
        test_tickers = {f["ticker"].upper() for f in test_filings}

        train_accns = {f["accession_number"] for f in train_filings}
        val_accns = {f["accession_number"] for f in val_filings}
        test_accns = {f["accession_number"] for f in test_filings}

        # 1. Company isolation for test set
        assert train_tickers.isdisjoint(test_tickers), (
            f"Leak detected! Companies in both train and test: {train_tickers & test_tickers}"
        )
        assert val_tickers.isdisjoint(test_tickers), (
            f"Leak detected! Companies in both val and test: {val_tickers & test_tickers}"
        )

        # 2. Accession number pairwise mutual exclusivity
        assert train_accns.isdisjoint(val_accns), (
            f"Leak detected! Filings in both train and val: {train_accns & val_accns}"
        )
        assert train_accns.isdisjoint(test_accns), (
            f"Leak detected! Filings in both train and test: {train_accns & test_accns}"
        )
        assert val_accns.isdisjoint(test_accns), (
            f"Leak detected! Filings in both val and test: {val_accns & test_accns}"
        )

        # 3. Temporal ordering check for train vs val
        val_year = manifest.get("val_fiscal_year", 2024)
        for f in train_filings:
            year = int(f["period_of_report"].split("-")[0])
            assert year < val_year, (
                f"Temporal leak detected! Train filing {f['accession_number']} has year {year} >= val year {val_year}"
            )
