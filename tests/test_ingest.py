"""Unit tests for src.ingest components."""

from datetime import date
from pathlib import Path

import pytest
from pytest_httpx import HTTPXMock

from src.ingest.cik_resolver import CIKResolver, UnknownTickerError
from src.ingest.downloader import Downloader
from src.ingest.filing_fetcher import FilingFetcher
from src.ingest.rate_limiter import SlidingWindowRateLimiter
from src.models import FilingMeta


def test_rate_limiter():
    limiter = SlidingWindowRateLimiter(max_calls=3, period=0.2)
    # Should allow 3 calls immediately
    limiter()
    limiter()
    limiter()
    assert len(limiter._timestamps) == 3


def test_cik_resolver(tmp_path: Path, httpx_mock: HTTPXMock):
    fake_tickers_data = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
    }
    httpx_mock.add_response(
        url="https://www.sec.gov/files/company_tickers.json",
        json=fake_tickers_data,
    )

    resolver = CIKResolver(cache_dir=tmp_path, user_agent="TestApp test@example.com")
    cik = resolver.resolve("AAPL")
    assert cik == "0000320193"
    assert resolver.resolve("msft") == "0000789019"

    # Test cache file created
    assert (tmp_path / "company_tickers.json").exists()

    with pytest.raises(UnknownTickerError):
        resolver.resolve("NONEXISTENT")


def test_filing_fetcher(httpx_mock: HTTPXMock):
    fake_submissions = {
        "cik": "0000320193",
        "fiscalYearEnd": "0930",
        "filings": {
            "recent": {
                "accessionNumber": [
                    "0000320193-23-000106",
                    "0000320193-23-000077",
                    "0000320193-20-000001",
                ],
                "form": ["10-K", "10-Q", "8-K"],
                "filingDate": ["2023-10-27", "2023-08-04", "2020-01-15"],
                "reportDate": ["2023-09-30", "2023-07-01", "2020-01-15"],
            },
            "files": [],
        },
    }
    httpx_mock.add_response(
        url="https://data.sec.gov/submissions/CIK0000320193.json",
        json=fake_submissions,
    )

    with FilingFetcher(user_agent="TestApp test@example.com") as fetcher:
        filings = fetcher.fetch_filings(
            cik="0000320193",
            ticker="AAPL",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            form_types={"10-K", "10-Q"},
        )

    assert len(filings) == 2
    # Sorted by filed_date ascending: 2023-08-04 (10-Q) comes before 2023-10-27 (10-K)
    assert filings[0].form_type == "10-Q"
    assert filings[0].accession_number == "0000320193-23-000077"
    assert filings[1].form_type == "10-K"
    assert filings[1].accession_number == "0000320193-23-000106"


def test_downloader(tmp_path: Path, httpx_mock: HTTPXMock):
    meta = FilingMeta(
        cik="0000320193",
        ticker="AAPL",
        accession_number="0000320193-23-000106",
        form_type="10-K",
        filed_date=date(2023, 10, 27),
        period_of_report=date(2023, 9, 30),
    )

    index_url = (
        "https://www.sec.gov/Archives/edgar/data/0000320193/000032019323000106/"
        "000032019323000106-index.json"
    )
    fake_index = {
        "directory": {
            "item": [
                {"name": "aapl-20230930.htm", "type": "10-K"},
                {"name": "filing-details.xml", "type": "XML"},
            ]
        }
    }
    httpx_mock.add_response(url=index_url, json=fake_index)

    doc_url = (
        "https://www.sec.gov/Archives/edgar/data/0000320193/000032019323000106/aapl-20230930.htm"
    )
    httpx_mock.add_response(url=doc_url, text="<html><body>10-K content</body></html>")

    with Downloader(output_dir=tmp_path, user_agent="TestApp test@example.com") as dl:
        assert not dl.is_cached(meta)
        raw_doc = dl.download(meta)
        assert raw_doc is not None
        assert raw_doc.local_path.exists()
        assert "10-K content" in raw_doc.local_path.read_text(encoding="utf-8")
        assert dl.is_cached(meta)

        # Idempotent second call: no further HTTP request needed
        cached_doc = dl.download(meta)
        assert cached_doc is not None
        assert cached_doc.local_path == raw_doc.local_path
