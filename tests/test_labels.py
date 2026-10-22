"""Unit tests for XBRL label extraction, tag fallbacks, restatements, and period handling."""

from datetime import date

from src.labels.extractor import FactExtractor
from src.models import FilingMeta

SAMPLE_XBRL_MAP = {
    "concepts": {
        "revenue": {
            "type": "duration",
            "preferred_unit": "USD",
            "tags": [
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "SalesRevenueNet",
                "Revenues",
            ],
        },
        "net_income": {
            "type": "duration",
            "preferred_unit": "USD",
            "tags": ["NetIncomeLoss", "ProfitLoss"],
        },
        "diluted_eps": {
            "type": "duration",
            "preferred_unit": "USD/shares",
            "tags": ["EarningsPerShareDiluted"],
        },
        "total_assets": {
            "type": "instant",
            "preferred_unit": "USD",
            "tags": ["Assets"],
        },
        "total_liabilities": {
            "type": "instant",
            "preferred_unit": "USD",
            "tags": ["Liabilities"],
            "calculation_fallback": {"sum_tags": ["LiabilitiesCurrent", "LiabilitiesNoncurrent"]},
        },
        "operating_cash_flow": {
            "type": "duration",
            "preferred_unit": "USD",
            "tags": ["NetCashProvidedByUsedInOperatingActivities"],
        },
    }
}


def test_tag_fallback_cascade():
    """Verify that when primary revenue tag is missing, fallback tag is used."""
    companyfacts = {
        "facts": {
            "us-gaap": {
                # Primary tag missing, secondary 'SalesRevenueNet' present
                "SalesRevenueNet": {
                    "units": {
                        "USD": [
                            {
                                "accn": "0000320193-23-000106",
                                "val": 383285000000,
                                "start": "2022-10-01",
                                "end": "2023-09-30",
                                "form": "10-K",
                                "fy": 2023,
                                "fp": "FY",
                            }
                        ]
                    }
                },
                # Lower priority tag also present, should be ignored
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "accn": "0000320193-23-000106",
                                "val": 999999999999,
                                "start": "2022-10-01",
                                "end": "2023-09-30",
                                "form": "10-K",
                                "fy": 2023,
                                "fp": "FY",
                            }
                        ]
                    }
                },
            }
        }
    }

    meta = FilingMeta(
        cik="0000320193",
        ticker="AAPL",
        accession_number="0000320193-23-000106",
        form_type="10-K",
        filed_date=date(2023, 10, 27),
        period_of_report=date(2023, 9, 30),
    )

    extractor = FactExtractor(config=SAMPLE_XBRL_MAP)
    labels = extractor.extract_labels_for_filing(companyfacts, meta)

    assert "revenue" in labels.metrics
    rev_candidates = labels.metrics["revenue"]
    assert len(rev_candidates) == 1
    assert rev_candidates[0].value == 383285000000
    assert rev_candidates[0].tag == "SalesRevenueNet"
    assert rev_candidates[0].period_type == "annual"


def test_restatement_isolation_by_accession():
    """Verify facts filed in later restatements are excluded."""
    companyfacts = {
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            # Original fact filed in 10-K
                            {
                                "accn": "0000320193-23-000106",
                                "val": 96995000000,
                                "start": "2022-10-01",
                                "end": "2023-09-30",
                                "form": "10-K",
                                "fy": 2023,
                                "fp": "FY",
                            },
                            # Restatement reported in next year's 10-K
                            {
                                "accn": "0000320193-24-000100",
                                "val": 95000000000,
                                "start": "2022-10-01",
                                "end": "2023-09-30",
                                "form": "10-K",
                                "fy": 2023,
                                "fp": "FY",
                            },
                        ]
                    }
                }
            }
        }
    }

    meta = FilingMeta(
        cik="0000320193",
        ticker="AAPL",
        accession_number="0000320193-23-000106",
        form_type="10-K",
        filed_date=date(2023, 10, 27),
        period_of_report=date(2023, 9, 30),
    )

    extractor = FactExtractor(config=SAMPLE_XBRL_MAP)
    labels = extractor.extract_labels_for_filing(companyfacts, meta)

    assert len(labels.metrics["net_income"]) == 1
    # Must match original filing accession value, NOT the restated value
    assert labels.metrics["net_income"][0].value == 96995000000


def test_10q_quarter_vs_ytd():
    """Verify that 10-Q duration extraction identifies both quarter and YTD values."""
    companyfacts = {
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            # 3-month quarter duration (~91 days)
                            {
                                "accn": "0000320193-23-000077",
                                "val": 19881000000,
                                "start": "2023-04-01",
                                "end": "2023-07-01",
                                "form": "10-Q",
                                "fp": "Q3",
                            },
                            # 9-month YTD duration (~273 days)
                            {
                                "accn": "0000320193-23-000077",
                                "val": 74039000000,
                                "start": "2022-10-01",
                                "end": "2023-07-01",
                                "form": "10-Q",
                                "fp": "Q3",
                            },
                        ]
                    }
                }
            }
        }
    }

    meta = FilingMeta(
        cik="0000320193",
        ticker="AAPL",
        accession_number="0000320193-23-000077",
        form_type="10-Q",
        filed_date=date(2023, 8, 4),
        period_of_report=date(2023, 7, 1),
    )

    extractor = FactExtractor(config=SAMPLE_XBRL_MAP)
    labels = extractor.extract_labels_for_filing(companyfacts, meta)

    candidates = labels.metrics["net_income"]
    assert len(candidates) == 2

    q_cand = next(c for c in candidates if c.period_type == "quarter")
    ytd_cand = next(c for c in candidates if c.period_type == "ytd")

    assert q_cand.value == 19881000000
    assert ytd_cand.value == 74039000000


def test_liabilities_calculation_fallback():
    """Verify that when Liabilities tag is missing, current + noncurrent is summed and marked is_derived."""
    companyfacts = {
        "facts": {
            "us-gaap": {
                "LiabilitiesCurrent": {
                    "units": {
                        "USD": [
                            {
                                "accn": "0000320193-23-000106",
                                "val": 145308000000,
                                "end": "2023-09-30",
                                "form": "10-K",
                            }
                        ]
                    }
                },
                "LiabilitiesNoncurrent": {
                    "units": {
                        "USD": [
                            {
                                "accn": "0000320193-23-000106",
                                "val": 145129000000,
                                "end": "2023-09-30",
                                "form": "10-K",
                            }
                        ]
                    }
                },
            }
        }
    }

    meta = FilingMeta(
        cik="0000320193",
        ticker="AAPL",
        accession_number="0000320193-23-000106",
        form_type="10-K",
        filed_date=date(2023, 10, 27),
        period_of_report=date(2023, 9, 30),
    )

    extractor = FactExtractor(config=SAMPLE_XBRL_MAP)
    labels = extractor.extract_labels_for_filing(companyfacts, meta)

    assert "total_liabilities" in labels.metrics
    cand = labels.metrics["total_liabilities"][0]
    assert cand.is_derived is True
    assert cand.value == 145308000000 + 145129000000
    assert cand.period_type == "instant"


def test_company_facts_fetcher(tmp_path, httpx_mock):
    from src.labels.fetcher import CompanyFactsFetcher

    fake_facts = {
        "cik": 320193,
        "entityName": "Apple Inc.",
        "facts": {"us-gaap": {}},
    }
    httpx_mock.add_response(
        url="https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
        json=fake_facts,
    )

    with CompanyFactsFetcher(cache_dir=tmp_path, user_agent="TestApp test@example.com") as fetcher:
        assert not fetcher.is_cached("0000320193")
        facts = fetcher.fetch("0000320193")
        assert facts["entityName"] == "Apple Inc."
        assert fetcher.is_cached("0000320193")

        # Idempotent second fetch
        cached_facts = fetcher.fetch("0000320193")
        assert cached_facts["entityName"] == "Apple Inc."
