"""Preprocessing module."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from statsmodels.tsa.stattools import adfuller

from src.exceptions import OutputWriteError, StationarityError

logger = logging.getLogger(__name__)


def clean_prices(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Handle missing values in a price DataFrame.

    Args:
        df: Raw sector price DataFrame indexed by date.
        config: Parsed project configuration.

    Returns:
        Cleaned price DataFrame.
    """
    cleaned = df.copy().sort_index()

    missing_by_row = cleaned.isna().sum(axis=1)
    rows_to_drop = missing_by_row > 3
    if rows_to_drop.any():
        logger.warning("Dropping %s rows with more than 3 missing sectors", int(rows_to_drop.sum()))
        cleaned = cleaned.loc[~rows_to_drop].copy()

    before_fill = cleaned.isna().sum()
    cleaned = cleaned.ffill(limit=2)
    after_fill = cleaned.isna().sum()

    for column in cleaned.columns:
        filled_count = int(before_fill[column] - after_fill[column])
        if filled_count > 0:
            logger.warning("Forward-filled %s missing values for %s", filled_count, column)

    remaining_missing = cleaned.isna().sum().sum()
    if remaining_missing:
        logger.warning("Cleaning completed with %s remaining missing values", int(remaining_missing))

    return cleaned


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from a price DataFrame.

    Args:
        prices: Cleaned sector price DataFrame.

    Returns:
        DataFrame of log returns.
    """
    invalid_rows = prices.le(0).fillna(False).any(axis=1)
    valid_prices = prices.copy()
    if invalid_rows.any():
        logger.warning("Dropping %s rows with non-positive prices", int(invalid_rows.sum()))
        valid_prices = valid_prices.loc[~invalid_rows].copy()

    returns = np.log(valid_prices / valid_prices.shift(1)).dropna(how="any")
    return returns


def run_adf_tests(returns: pd.DataFrame, significance: float = 0.01) -> dict:
    """Run Augmented Dickey-Fuller tests on each sector return series.

    Args:
        returns: Log returns DataFrame.
        significance: p-value threshold for stationarity.

    Returns:
        Mapping of sector label to ADF test summary.

    Raises:
        StationarityError: If a sector remains non-stationary after first differencing.
    """
    results: dict[str, dict[str, float | bool]] = {}

    for column in returns.columns:
        series = returns[column].dropna()
        statistic, pvalue, *_ = adfuller(series)
        stationary = pvalue < significance
        differenced = False

        if not stationary:
            logger.warning("%s failed ADF at p=%.6f; re-testing first difference", column, pvalue)
            series = series.diff().dropna()
            statistic, pvalue, *_ = adfuller(series)
            stationary = pvalue < significance
            differenced = True

        if not stationary:
            raise StationarityError(f"{column} failed ADF after first differencing")

        if differenced:
            logger.warning("%s required first differencing to pass ADF", column)

        results[column] = {
            "statistic": float(statistic),
            "pvalue": float(pvalue),
            "stationary": bool(stationary),
            "differenced": bool(differenced),
        }

    return results


def winsorize_returns(returns: pd.DataFrame, limits: list) -> pd.DataFrame:
    """Winsorize return series using percentile cutoffs.

    Args:
        returns: Log returns DataFrame.
        limits: Two-element list of lower and upper quantile cutoffs.

    Returns:
        Winsorized return DataFrame.
    """
    lower_limit = float(limits[0])
    upper_limit = 1.0 - float(limits[1])
    winsorized = returns.copy()

    for column in winsorized.columns:
        winsorized[column] = winsorize(winsorized[column].to_numpy(), limits=(lower_limit, upper_limit))

    return winsorized.astype(float)


def preprocess_and_save(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Run the preprocessing pipeline and save contracted outputs.

    Args:
        prices: Raw sector price DataFrame.
        config: Parsed project configuration.

    Returns:
        Winsorized log returns DataFrame.

    Raises:
        OutputWriteError: If required outputs cannot be saved.
    """
    cleaned = clean_prices(prices, config)
    returns = compute_log_returns(cleaned)
    adf_results = run_adf_tests(returns, significance=config["analysis"]["adf_significance"])
    final_returns = winsorize_returns(returns, limits=config["analysis"]["winsorize_limits"])

    processed_path = Path("data/processed/log_returns.csv")
    adf_path = Path(config["outputs"]["tables_path"]) / "adf_results.csv"

    try:
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        adf_path.parent.mkdir(parents=True, exist_ok=True)
        final_returns.to_csv(processed_path, index_label="date")
        pd.DataFrame.from_dict(adf_results, orient="index").reset_index().rename(
            columns={"index": "sector"}
        ).to_csv(adf_path, index=False)
    except OSError as exc:
        raise OutputWriteError(f"Failed to save preprocessing outputs: {exc}") from exc

    logger.info("Saved processed returns to %s with shape %s", processed_path, final_returns.shape)
    logger.info("Saved ADF results to %s", adf_path)
    return final_returns
