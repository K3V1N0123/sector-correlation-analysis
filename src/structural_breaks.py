"""Structural break analysis module."""

from __future__ import annotations

import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from src.correlation import compute_correlation_matrix
from src.exceptions import InsufficientDataError, OutputWriteError

logger = logging.getLogger(__name__)


def split_sub_periods(
    returns: pd.DataFrame,
    sub_periods: dict,
    min_subperiod_days: int = 50,
) -> dict[str, pd.DataFrame]:
    """Split a return series into configured sub-periods.

    Args:
        returns: Full log returns DataFrame.
        sub_periods: Mapping of named period definitions.
        min_subperiod_days: Minimum row count required for each sub-period.

    Returns:
        Mapping of period key to sliced returns DataFrame.

    Raises:
        InsufficientDataError: If any sub-period has fewer than the minimum rows.
    """
    split_data: dict[str, pd.DataFrame] = {}

    for period_name, period in sub_periods.items():
        subset = returns.loc[period["start"] : period["end"]].copy()
        if len(subset) < min_subperiod_days:
            raise InsufficientDataError(
                f"{period_name} has {len(subset)} rows, below minimum {min_subperiod_days}"
            )
        split_data[period_name] = subset
        logger.info("Prepared sub-period %s with %s rows", period_name, len(subset))

    return split_data


def jennrich_test(
    corr_a: pd.DataFrame,
    n_a: int,
    corr_b: pd.DataFrame,
    n_b: int,
) -> dict:
    """Run a large-sample Jennrich-style equality test on two correlation matrices.

    Args:
        corr_a: Correlation matrix for period A.
        n_a: Sample size for period A.
        corr_b: Correlation matrix for period B.
        n_b: Sample size for period B.

    Returns:
        Dictionary with test statistic, degrees of freedom, p-value, and decision.
    """
    matrix_a = pd.DataFrame(corr_a).astype(float)
    matrix_b = pd.DataFrame(corr_b).astype(float)

    matrix_a = (matrix_a + matrix_a.T) / 2
    matrix_b = (matrix_b + matrix_b.T) / 2

    if matrix_a.shape != matrix_b.shape:
        raise ValueError("Correlation matrices must have the same shape")

    diff = matrix_a - matrix_b
    tri_rows, tri_cols = np.triu_indices_from(diff, k=1)
    diff_vector = diff.to_numpy()[tri_rows, tri_cols]

    weight = (n_a * n_b) / (n_a + n_b)
    statistic = float(weight * np.dot(diff_vector, diff_vector))
    p = matrix_a.shape[0]
    df = p * (p - 1) // 2
    pvalue = float(chi2.sf(statistic, df))

    return {
        "statistic": statistic,
        "df": df,
        "pvalue": pvalue,
        "reject_h0": pvalue < 0.05,
    }


def run_all_break_tests(
    sub_period_returns: dict,
    config: dict,
) -> pd.DataFrame:
    """Run structural break tests for all required sub-period comparisons.

    Args:
        sub_period_returns: Mapping of sub-period name to returns DataFrame.
        config: Parsed project configuration.

    Returns:
        DataFrame containing the three required break-test results.

    Raises:
        OutputWriteError: If required output files cannot be saved.
    """
    outputs_path = Path(config["outputs"]["tables_path"])
    corr_output_names = {
        "pre_covid": "corr_matrix_pre_covid.csv",
        "covid_shock": "corr_matrix_covid.csv",
        "post_covid": "corr_matrix_post_covid.csv",
    }

    corr_matrices: dict[str, pd.DataFrame] = {}
    for period_name, period_returns in sub_period_returns.items():
        corr_matrix, _ = compute_correlation_matrix(period_returns, method=config["analysis"]["correlation_method"])
        corr_matrices[period_name] = corr_matrix

        try:
            outputs_path.mkdir(parents=True, exist_ok=True)
            corr_matrix.to_csv(outputs_path / corr_output_names[period_name], index_label="sector")
        except OSError as exc:
            raise OutputWriteError(f"Failed to save sub-period correlation matrix: {exc}") from exc

    comparisons = [
        ("pre_covid", "covid_shock"),
        ("covid_shock", "post_covid"),
        ("pre_covid", "post_covid"),
    ]
    results = []

    for period_a, period_b in comparisons:
        test_result = jennrich_test(
            corr_a=corr_matrices[period_a],
            n_a=len(sub_period_returns[period_a]),
            corr_b=corr_matrices[period_b],
            n_b=len(sub_period_returns[period_b]),
        )
        results.append(
            {
                "period_a": period_a,
                "period_b": period_b,
                **test_result,
            }
        )

    results_frame = pd.DataFrame(results)
    try:
        results_frame.to_csv(outputs_path / "break_tests.csv", index=False)
    except OSError as exc:
        raise OutputWriteError(f"Failed to save break test results: {exc}") from exc

    logger.info("Saved break test results to %s", outputs_path / "break_tests.csv")
    return results_frame
