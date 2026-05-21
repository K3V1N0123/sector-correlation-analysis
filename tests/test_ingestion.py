"""Tests for ingestion module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.exceptions import DataIngestionError
from src.ingestion import fetch_sector_prices, load_cached_prices


def test_load_cached_prices_reads_datetime_index(tmp_path: Path) -> None:
    """Cached prices should load with a named DatetimeIndex."""
    cache_path = tmp_path / "nifty_sectors_raw.csv"
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "BANK": [100.0, 101.0],
            "IT": [200.0, 202.0],
        }
    )
    frame.to_csv(cache_path, index=False)

    loaded = load_cached_prices(str(cache_path))

    assert loaded.index.name == "date"
    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert list(loaded.columns) == ["BANK", "IT"]


def test_load_cached_prices_raises_for_missing_file(tmp_path: Path) -> None:
    """Missing cache files should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_cached_prices(str(tmp_path / "missing.csv"))


def test_fetch_sector_prices_uses_cache_when_live_fetch_insufficient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cache fallback should return cached data when live fetch is unusable."""
    cache_path = tmp_path / "nifty_sectors_raw.csv"
    cached = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
            **{f"S{i}": [float(i), float(i + 1), float(i + 2)] for i in range(10)},
        }
    )
    cached.to_csv(cache_path, index=False)

    config = {
        "data": {
            "start_date": "2024-01-01",
            "end_date": "2024-01-10",
            "cache_path": str(cache_path),
        },
        "sectors": [
            {"label": f"S{i}", "ticker": f"TICKER{i}"} for i in range(14)
        ],
    }

    def fake_download(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr("src.ingestion.yf.download", fake_download)
    monkeypatch.setattr("src.ingestion.with_retry", lambda func, retries=3, delay=5.0: func())

    loaded = fetch_sector_prices(config)

    assert loaded.shape == (3, 10)
    assert list(loaded.columns) == [f"S{i}" for i in range(10)]


def test_fetch_sector_prices_raises_when_live_and_cache_insufficient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The module should raise when fewer than 10 sectors are available overall."""
    cache_path = tmp_path / "nifty_sectors_raw.csv"
    cached = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
            **{f"S{i}": [float(i), float(i + 1), float(i + 2)] for i in range(9)},
        }
    )
    cached.to_csv(cache_path, index=False)

    config = {
        "data": {
            "start_date": "2024-01-01",
            "end_date": "2024-01-10",
            "cache_path": str(cache_path),
        },
        "sectors": [
            {"label": f"S{i}", "ticker": f"TICKER{i}"} for i in range(14)
        ],
    }

    def fake_download(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr("src.ingestion.yf.download", fake_download)
    monkeypatch.setattr("src.ingestion.with_retry", lambda func, retries=3, delay=5.0: func())

    with pytest.raises(DataIngestionError):
        fetch_sector_prices(config)
