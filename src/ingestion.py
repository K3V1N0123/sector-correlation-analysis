"""Data ingestion module."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from src.exceptions import DataIngestionError

logger = logging.getLogger(__name__)


def with_retry(func: Any, retries: int = 3, delay: float = 5.0) -> Any:
    """Run a callable with simple retry logic.

    Args:
        func: Callable to execute.
        retries: Maximum number of attempts.
        delay: Sleep duration between attempts in seconds.

    Returns:
        The callable result.

    Raises:
        Exception: Re-raises the final exception from ``func``.
    """
    for attempt in range(1, retries + 1):
        try:
            return func()
        except Exception as exc:
            logger.warning("Attempt %s/%s failed: %s", attempt, retries, exc)
            if attempt == retries:
                raise
            time.sleep(delay)


def _expected_rows(start_date: str, end_date: str) -> int:
    """Estimate expected business-day observations for completeness checks.

    Args:
        start_date: Analysis start date.
        end_date: Analysis end date.

    Returns:
        Estimated number of business days in the interval.
    """
    return len(pd.date_range(start=start_date, end=end_date, freq="B"))


def _fetch_single_ticker(ticker: str, label: str, start_date: str, end_date: str) -> pd.Series:
    """Download one sector series from yfinance.

    Args:
        ticker: Market ticker symbol.
        label: Output column label.
        start_date: Analysis start date.
        end_date: Analysis end date.

    Returns:
        Price series named with the sector label.

    Raises:
        ValueError: If the downloaded dataset does not contain usable prices.
    """
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=True,
    )
    if data.empty:
        raise ValueError(f"No close data returned for {ticker}")

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            raise ValueError(f"No close data returned for {ticker}")
        close_frame = data.xs("Close", axis=1, level=0)
        if ticker not in close_frame.columns:
            raise ValueError(f"No close data returned for {ticker}")
        series = close_frame[ticker].copy()
    else:
        if "Close" not in data.columns:
            raise ValueError(f"No close data returned for {ticker}")
        series = data["Close"].copy()

    series.name = label
    return series


def _fetch_with_alternates(
    tickers: list[str],
    label: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """Try multiple ticker symbols for the same sector until one succeeds.

    Args:
        tickers: Ordered ticker candidates.
        label: Output column label.
        start_date: Analysis start date.
        end_date: Analysis end date.

    Returns:
        Price series for the first working ticker.

    Raises:
        ValueError: If no ticker candidate returns usable close data.
    """
    last_error: Exception | None = None

    for ticker in tickers:
        try:
            return with_retry(
                lambda ticker=ticker: _fetch_single_ticker(
                    ticker=ticker,
                    label=label,
                    start_date=start_date,
                    end_date=end_date,
                )
            )
        except Exception as exc:
            last_error = exc
            logger.warning("Ticker candidate failed for %s (%s): %s", label, ticker, exc)

    raise ValueError(f"No close data returned for any ticker mapped to {label}") from last_error


def load_cached_prices(cache_path: str) -> pd.DataFrame:
    """Load prices from local CSV cache.

    Args:
        cache_path: Path to raw CSV.

    Returns:
        DataFrame with a DatetimeIndex.

    Raises:
        FileNotFoundError: If the cache file does not exist.
    """
    cache_file = Path(cache_path)
    if not cache_file.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    frame = pd.read_csv(cache_file, index_col="date", parse_dates=True)
    frame.index.name = "date"
    logger.info("Loaded cached prices from %s with shape %s", cache_path, frame.shape)
    return frame.sort_index()


def fetch_sector_prices(config: dict) -> pd.DataFrame:
    """Download daily closing prices for all configured Nifty sectoral indices.

    Strategy:
        1. Try yfinance for each ticker in ``config['sectors']``.
        2. If live data is too incomplete, fall back to cached CSV.
        3. Merge all successful tickers on the date index.
        4. Save to ``config['data']['cache_path']``.

    Args:
        config: Parsed ``config.yaml`` contents.

    Returns:
        DataFrame with date index and sector label columns.

    Raises:
        DataIngestionError: If fewer than 10 sectors can be fetched or loaded.
    """
    sectors = config["sectors"]
    data_config = config["data"]
    start_date = data_config["start_date"]
    end_date = data_config["end_date"]
    cache_path = data_config["cache_path"]

    expected_rows = _expected_rows(start_date, end_date)
    min_rows = int(expected_rows * 0.8)
    series_list: list[pd.Series] = []

    for sector in sectors:
        label = sector["label"]
        ticker = sector["ticker"]
        ticker_candidates = [ticker, *sector.get("alternate_tickers", [])]

        try:
            series = _fetch_with_alternates(
                tickers=ticker_candidates,
                label=label,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception as exc:
            logger.error("Failed to fetch %s (%s): %s", label, ticker_candidates, exc)
            continue

        if len(series.dropna()) < min_rows:
            logger.error(
                "Fetched %s rows for %s, below completeness threshold %s",
                len(series.dropna()),
                label,
                min_rows,
            )
            continue

        series_list.append(series)
        logger.info("Fetched %s with %s observations", label, len(series.dropna()))

    if len(series_list) < 10:
        logger.warning("Live fetch produced %s sectors, trying cache fallback", len(series_list))
        cached = load_cached_prices(cache_path)
        if cached.shape[1] < 10:
            raise DataIngestionError("Fewer than 10 sectors available from live fetch and cache")
        return cached

    prices = pd.concat(series_list, axis=1).sort_index()
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"

    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(cache_file)
    logger.info("Saved raw prices to %s with shape %s", cache_path, prices.shape)
    return prices
