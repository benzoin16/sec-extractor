"""Dataset builder assembling instruction + chunk text into chat-format training examples."""

from __future__ import annotations

import json
import logging
import random
import re

from src.dataset.matcher import NumericMatcher
from src.labels.models import FilingLabels, TargetMetric
from src.models import Chunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a financial document information extractor. Extract the 7 key financial "
    "metrics from the provided SEC filing text in JSON format: revenue, net_income, "
    "diluted_eps, total_assets, operating_income, total_liabilities, operating_cash_flow. "
    "For each present metric, output an object with 'value' (unscaled numeric amount) "
    "and 'period_type' ('annual', 'quarter', 'ytd', or 'instant'). If a metric is not "
    "explicitly mentioned in the text, set it to null."
)

TARGET_CONCEPTS = [
    "revenue",
    "net_income",
    "diluted_eps",
    "total_assets",
    "operating_income",
    "total_liabilities",
    "operating_cash_flow",
]


class DatasetBuilder:
    """Builds chat-format instruction examples from chunks and ground truth labels."""

    def __init__(
        self,
        matcher: NumericMatcher | None = None,
        negative_ratio: float = 0.15,
        hard_negative_ratio: float = 0.50,
        random_seed: int = 42,
    ) -> None:
        self.matcher = matcher or NumericMatcher()
        self.negative_ratio = negative_ratio
        self.hard_negative_ratio = hard_negative_ratio
        self.random_seed = random_seed
        self._rng = random.Random(random_seed)

    @staticmethod
    def _is_hard_negative(text: str) -> bool:
        """Classify whether a negative chunk contains tables or numbers."""
        has_table = "|" in text and "-|-" in text
        number_count = len(re.findall(r"\$?\b\d+(?:,\d{3})*(?:\.\d+)?\b", text))
        return has_table or number_count >= 5

    def build_examples_for_filing(
        self,
        chunks: list[Chunk],
        labels: FilingLabels,
    ) -> list[dict]:
        """Convert chunks of a filing into chat examples with positive/negative sampling."""
        positives: list[dict] = []
        hard_negatives: list[dict] = []
        text_negatives: list[dict] = []

        for chunk in chunks:
            matched, was_skipped = self.matcher.match_chunk(
                chunk_text=chunk.text,
                filing_labels=labels,
                target_concepts=TARGET_CONCEPTS,
            )

            # Skip ambiguous examples (e.g. identical quarter and YTD values)
            if was_skipped:
                continue

            has_any_metric = any(v is not None for v in matched.values())

            # Format target JSON
            target_dict: dict[str, dict | None] = {}
            for k in TARGET_CONCEPTS:
                metric: TargetMetric | None = matched.get(k)
                if metric is not None:
                    target_dict[k] = {
                        "value": metric.value,
                        "period_type": metric.period_type,
                    }
                else:
                    target_dict[k] = None

            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Extract the financial metrics from the following SEC filing text.\n\n"
                            f"Filing Text:\n```\n{chunk.text}\n```"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(target_dict, indent=None),
                    },
                ],
                "metadata": {
                    "ticker": chunk.meta.ticker,
                    "accession_number": chunk.meta.accession_number,
                    "form_type": chunk.meta.form_type,
                    "period_of_report": str(chunk.meta.period_of_report),
                    "item_id": chunk.item_id,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                    "is_negative": not has_any_metric,
                    "negative_type": None,
                },
            }

            if has_any_metric:
                positives.append(example)
            else:
                if self._is_hard_negative(chunk.text):
                    example["metadata"]["negative_type"] = "hard"
                    hard_negatives.append(example)
                else:
                    example["metadata"]["negative_type"] = "text"
                    text_negatives.append(example)

        # Apply negative ratio sampling
        num_positives = len(positives)
        if num_positives == 0 and (hard_negatives or text_negatives):
            # If filing has no positives, allow a small fixed set of negatives
            max_negatives = min(2, len(hard_negatives) + len(text_negatives))
        else:
            # target negatives = positives * (ratio / (1 - ratio))
            max_negatives = int(
                round(num_positives * (self.negative_ratio / (1.0 - self.negative_ratio)))
            )

        target_hard = int(round(max_negatives * self.hard_negative_ratio))
        target_text = max_negatives - target_hard

        self._rng.shuffle(hard_negatives)
        self._rng.shuffle(text_negatives)

        selected_hard = hard_negatives[:target_hard]
        selected_text = text_negatives[:target_text]

        # Combine and shuffle
        final_examples = positives + selected_hard + selected_text
        self._rng.shuffle(final_examples)
        return final_examples
