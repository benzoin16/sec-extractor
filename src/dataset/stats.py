"""Dataset statistics calculator and markdown report generator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class DatasetStats:
    """Computes distribution statistics, null rates, and token percentiles across splits."""

    @classmethod
    def compute_stats(cls, split_examples: dict[str, list[dict]]) -> dict[str, Any]:
        """Compute full dataset metrics from split dictionary."""
        stats: dict[str, Any] = {
            "splits": {},
            "fields": {},
            "overall": {
                "total_examples": sum(len(exs) for exs in split_examples.values()),
            },
        }

        all_target_concepts = [
            "revenue",
            "net_income",
            "diluted_eps",
            "total_assets",
            "operating_income",
            "total_liabilities",
            "operating_cash_flow",
        ]

        # Field counters
        field_counts = {c: 0 for c in all_target_concepts}
        total_fields_evaluated = 0

        for split_name, examples in split_examples.items():
            positives = sum(
                1 for e in examples if not e.get("metadata", {}).get("is_negative", False)
            )
            hard_negs = sum(
                1 for e in examples if e.get("metadata", {}).get("negative_type") == "hard"
            )
            text_negs = sum(
                1 for e in examples if e.get("metadata", {}).get("negative_type") == "text"
            )
            total = len(examples)

            # Token lengths
            token_counts = [
                e.get("metadata", {}).get("token_count", len(e["messages"][1]["content"]) // 4)
                for e in examples
            ]

            tok_stats = {}
            if token_counts:
                arr = np.array(token_counts)
                tok_stats = {
                    "min": int(np.min(arr)),
                    "p25": int(np.percentile(arr, 25)),
                    "p50": int(np.median(arr)),
                    "p75": int(np.percentile(arr, 75)),
                    "p95": int(np.percentile(arr, 95)),
                    "max": int(np.max(arr)),
                }

            # Concept presence
            split_field_counts = {c: 0 for c in all_target_concepts}
            for e in examples:
                assistant_content = json.loads(e["messages"][2]["content"])
                for c in all_target_concepts:
                    if assistant_content.get(c) is not None:
                        split_field_counts[c] += 1
                        field_counts[c] += 1
                total_fields_evaluated += len(all_target_concepts)

            stats["splits"][split_name] = {
                "total": total,
                "positives": positives,
                "hard_negatives": hard_negs,
                "text_negatives": text_negs,
                "negative_pct": (hard_negs + text_negs) / total if total > 0 else 0.0,
                "tokens": tok_stats,
                "field_counts": split_field_counts,
            }

        # Overall field null rates
        total_examples_all = stats["overall"]["total_examples"]
        for c in all_target_concepts:
            cnt = field_counts[c]
            null_cnt = total_examples_all - cnt
            stats["fields"][c] = {
                "present_count": cnt,
                "null_count": null_cnt,
                "null_rate": (null_cnt / total_examples_all) if total_examples_all > 0 else 0.0,
            }

        return stats

    @classmethod
    def render_markdown_report(cls, stats: dict[str, Any], output_path: Path) -> str:
        """Format the stats dictionary into a clean Markdown document and save to disk."""
        lines: list[str] = [
            "# SEC Extractor Dataset Statistics Report\n",
            f"**Total Examples Generated:** {stats['overall']['total_examples']:,}\n",
            "## 1. Split Distribution\n",
            "| Split | Total Examples | Positive | Hard Negatives | Text Negatives | Negative % | Median Tokens | P95 Tokens |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        for sname, sdata in stats["splits"].items():
            toks = sdata.get("tokens", {})
            lines.append(
                f"| `{sname}` | {sdata['total']:,} | {sdata['positives']:,} | "
                f"{sdata['hard_negatives']:,} | {sdata['text_negatives']:,} | "
                f"{sdata['negative_pct'] * 100:.1f}% | {toks.get('p50', '—')} | {toks.get('p95', '—')} |"
            )

        lines.extend(
            [
                "\n## 2. Concept Presence and Null Rate\n",
                "| Target Concept | Present Examples | Null Examples | Null Rate |",
                "| :--- | :---: | :---: | :---: |",
            ]
        )

        for cname, cdata in stats["fields"].items():
            lines.append(
                f"| `{cname}` | {cdata['present_count']:,} | {cdata['null_count']:,} | "
                f"{cdata['null_rate'] * 100:.1f}% |"
            )

        lines.extend(
            [
                "\n## 3. Token Length Distribution per Split\n",
                "| Split | Min | P25 | P50 (Median) | P75 | P95 | Max |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )

        for sname, sdata in stats["splits"].items():
            toks = sdata.get("tokens", {})
            lines.append(
                f"| `{sname}` | {toks.get('min', '—')} | {toks.get('p25', '—')} | "
                f"{toks.get('p50', '—')} | {toks.get('p75', '—')} | "
                f"{toks.get('p95', '—')} | {toks.get('max', '—')} |"
            )

        content = "\n".join(lines) + "\n"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        return content
