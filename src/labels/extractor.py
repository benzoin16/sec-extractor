"""Extract ground-truth concept labels from SEC companyfacts JSON."""

from __future__ import annotations

import logging
from datetime import date

from src.labels.models import ConceptValue, FilingLabels, PeriodType
from src.models import FilingMeta

logger = logging.getLogger(__name__)


class FactExtractor:
    """Extracts ground-truth labels for a filing from SEC companyfacts data."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.concepts: dict = config.get("concepts", {})

    @staticmethod
    def _parse_date(d_str: str) -> date | None:
        try:
            return date.fromisoformat(d_str)
        except (ValueError, TypeError):
            return None

    def _determine_period_type(
        self,
        form_type: str,
        concept_type: str,
        start_str: str | None,
        end_str: str,
    ) -> PeriodType | None:
        """Classify a fact as 'annual', 'quarter', 'ytd', or 'instant'."""
        if concept_type == "instant":
            return "instant"

        if not start_str or not end_str:
            return "annual" if form_type.startswith("10-K") else "quarter"

        start_d = self._parse_date(start_str)
        end_d = self._parse_date(end_str)
        if not start_d or not end_d:
            return "annual" if form_type.startswith("10-K") else "quarter"

        duration_days = (end_d - start_d).days

        if form_type.startswith("10-K"):
            return "annual"

        # 10-Q durations
        if 70 <= duration_days <= 115:
            return "quarter"
        elif 160 <= duration_days <= 300:
            return "ytd"

        # Fallback based on form
        return "quarter"

    def extract_labels_for_filing(
        self,
        companyfacts: dict,
        meta: FilingMeta,
    ) -> FilingLabels:
        """Extract ground-truth metrics for a specific filing accession.

        Restatement-safe: strictly filters facts where ``accn == meta.accession_number``.
        """
        facts_us_gaap = companyfacts.get("facts", {}).get("us-gaap", {})
        facts_dei = companyfacts.get("facts", {}).get("dei", {})

        metrics: dict[str, list[ConceptValue]] = {}

        for concept_name, concept_cfg in self.concepts.items():
            concept_type = concept_cfg.get("type", "duration")
            preferred_unit = concept_cfg.get("preferred_unit", "USD")
            tags = concept_cfg.get("tags", [])
            tag_found = False

            # Check tags in prioritized fallback order
            for tag in tags:
                tag_node = facts_us_gaap.get(tag) or facts_dei.get(tag)
                if not tag_node:
                    continue

                units_dict = tag_node.get("units", {})
                # Try preferred unit first, then fall back to any available unit
                units_to_check = (
                    [preferred_unit] if preferred_unit in units_dict else list(units_dict.keys())
                )

                matching_candidates: list[ConceptValue] = []

                for unit_name in units_to_check:
                    entries = units_dict.get(unit_name, [])
                    for entry in entries:
                        # RESTRICTION: Match accession number strictly to avoid lookahead/restatements
                        if entry.get("accn") != meta.accession_number:
                            continue

                        val = entry.get("val")
                        if val is None:
                            continue

                        start_str = entry.get("start")
                        end_str = entry.get("end", "")

                        pt = self._determine_period_type(
                            form_type=meta.form_type,
                            concept_type=concept_type,
                            start_str=start_str,
                            end_str=end_str,
                        )
                        if pt is None:
                            continue

                        # Avoid exact duplicate candidates for the same period_type
                        if not any(
                            c.period_type == pt and c.value == val for c in matching_candidates
                        ):
                            matching_candidates.append(
                                ConceptValue(
                                    value=val,
                                    period_type=pt,
                                    tag=tag,
                                    unit=unit_name,
                                    is_derived=False,
                                )
                            )

                if matching_candidates:
                    metrics[concept_name] = matching_candidates
                    tag_found = True
                    break  # Found highest priority tag in cascade

            # Calculation fallback (e.g. total_liabilities = LiabilitiesCurrent + LiabilitiesNoncurrent)
            if not tag_found and "calculation_fallback" in concept_cfg:
                fallback_cfg = concept_cfg["calculation_fallback"]
                sum_tags = fallback_cfg.get("sum_tags", [])
                if sum_tags:
                    derived_val = self._derive_sum_metric(
                        facts_us_gaap=facts_us_gaap,
                        meta=meta,
                        sum_tags=sum_tags,
                        preferred_unit=preferred_unit,
                    )
                    if derived_val is not None:
                        metrics[concept_name] = [
                            ConceptValue(
                                value=derived_val,
                                period_type="instant" if concept_type == "instant" else "annual",
                                tag="+".join(sum_tags),
                                unit=preferred_unit,
                                is_derived=True,
                            )
                        ]

        return FilingLabels(
            cik=meta.cik,
            ticker=meta.ticker,
            accession_number=meta.accession_number,
            form_type=meta.form_type,
            period_of_report=meta.period_of_report,
            metrics=metrics,
        )

    def _derive_sum_metric(
        self,
        facts_us_gaap: dict,
        meta: FilingMeta,
        sum_tags: list[str],
        preferred_unit: str,
    ) -> float | int | None:
        """Sum components for a derived metric only when all components exist in the filing."""
        component_values: list[float | int] = []

        for tag in sum_tags:
            tag_node = facts_us_gaap.get(tag)
            if not tag_node:
                return None
            units = tag_node.get("units", {}).get(preferred_unit, [])
            # Find matching accession entry
            match = next(
                (
                    e
                    for e in units
                    if e.get("accn") == meta.accession_number and e.get("val") is not None
                ),
                None,
            )
            if not match:
                return None
            component_values.append(match["val"])

        if len(component_values) == len(sum_tags):
            return sum(component_values)
        return None
