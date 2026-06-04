"""Tests for correlation module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.correlation import compute_correlation_matrix, compute_rolling_correlation


def _make_returns_frame(rows: int = 300, cols: int = 13) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=rows, freq="D")
    rng = np.random.default_rng(7)
    data = rng.normal(0.0, 0.01, size=(rows, cols))
    columns = [f"S{i}" for i in range(cols)]
    return pd.DataFrame(data, index=dates, columns=columns)


def test_correlation_matrix_symmetric() -> None:
    """Correlation matrix should be symmetric."""
    returns = _make_returns_frame()

    corr_matrix, _ = compute_correlation_matrix(returns)

    pd.testing.assert_frame_equal(corr_matrix, corr_matrix.T)


def test_diagonal_is_one() -> None:
    """All diagonal elements of the correlation matrix should equal one."""
    returns = _make_returns_frame()

    corr_matrix, _ = compute_correlation_matrix(returns)

    assert np.allclose(np.diag(corr_matrix), 1.0)


def test_rolling_output_length() -> None:
    """Rolling correlation output should have T - window + 1 rows."""
    returns = _make_returns_frame(rows=20)
    window = 5

    rolling = compute_rolling_correlation(returns, window=window, sector_pairs=[("S0", "S1")])

    assert len(rolling) == len(returns) - window + 1


def test_pvalue_matrix_range() -> None:
    """All p-values should lie in the unit interval."""
    returns = _make_returns_frame()

    _, pvalue_matrix = compute_correlation_matrix(returns)

    assert ((pvalue_matrix >= 0.0) & (pvalue_matrix <= 1.0)).all().all()
