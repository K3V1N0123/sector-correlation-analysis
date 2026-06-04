"""Correlation analysis module."""

from __future__ import annotations

import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.exceptions import OutputWriteError

logger = logging.getLogger(__name__)


def compute_correlation_matrix(
    returns: pd.DataFrame,
    method: str = "pearson",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute full-sample correlation and p-value matrices.

    Args:
        returns: Log returns DataFrame.
        method: Correlation method, either ``pearson`` or ``spearman``.

    Returns:
        Tuple of masked correlation matrix and p-value matrix.

    Raises:
        ValueError: If an unsupported method is requested.
    """
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'")

    columns = list(returns.columns)
    corr_matrix = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns, dtype=float)
    pvalue_matrix = pd.DataFrame(np.zeros((len(columns), len(columns))), index=columns, columns=columns, dtype=float)

    stat_func = pearsonr if method == "pearson" else spearmanr

    for col_a, col_b in itertools.combinations(columns, 2):
        paired = returns[[col_a, col_b]].dropna()
        corr_value, pvalue = stat_func(paired[col_a], paired[col_b])
        corr_matrix.loc[col_a, col_b] = corr_value
        corr_matrix.loc[col_b, col_a] = corr_value
        pvalue_matrix.loc[col_a, col_b] = pvalue
        pvalue_matrix.loc[col_b, col_a] = pvalue

    masked_corr = corr_matrix.mask(pvalue_matrix > 0.05).copy()
    for column in columns:
        masked_corr.loc[column, column] = 1.0
        pvalue_matrix.loc[column, column] = 0.0

    return masked_corr, pvalue_matrix


def compute_rolling_correlation(
    returns: pd.DataFrame,
    window: int = 252,
    sector_pairs: list[tuple] | None = None,
) -> pd.DataFrame:
    """Compute rolling pairwise correlations over a sliding window.

    Args:
        returns: Log returns DataFrame.
        window: Rolling window length in trading days.
        sector_pairs: Optional list of sector label pairs.

    Returns:
        DataFrame of rolling correlations indexed by end-of-window date.

    Raises:
        ValueError: If the window length is invalid.
    """
    if window <= 1 or window > len(returns):
        raise ValueError("window must be greater than 1 and at most the number of rows")

    pairs = sector_pairs or list(itertools.combinations(returns.columns, 2))
    rolling_data: dict[str, pd.Series] = {}

    for sector_a, sector_b in pairs:
        column_name = f"{sector_a}_vs_{sector_b}"
        rolling_data[column_name] = returns[sector_a].rolling(window).corr(returns[sector_b])

    rolling_frame = pd.DataFrame(rolling_data, index=returns.index).dropna(how="all")
    return rolling_frame


def save_correlation_matrix(corr_matrix: pd.DataFrame, output_path: str) -> None:
    """Save a correlation matrix CSV to disk.

    Args:
        corr_matrix: Correlation matrix to persist.
        output_path: Destination CSV path.

    Raises:
        OutputWriteError: If the file cannot be written.
    """
    path = Path(output_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        corr_matrix.to_csv(path, index_label="sector")
    except OSError as exc:
        raise OutputWriteError(f"Failed to save correlation matrix: {exc}") from exc

    logger.info("Saved correlation matrix to %s", output_path)


def save_rolling_correlation(rolling_corr: pd.DataFrame, output_path: str) -> None:
    """Save a rolling-correlation table to disk.

    Args:
        rolling_corr: Rolling correlation DataFrame to persist.
        output_path: Destination CSV path.

    Raises:
        OutputWriteError: If the file cannot be written.
    """
    path = Path(output_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        rolling_corr.to_csv(path, index_label="date")
    except OSError as exc:
        raise OutputWriteError(f"Failed to save rolling correlation table: {exc}") from exc

    logger.info("Saved rolling correlation table to %s", output_path)
