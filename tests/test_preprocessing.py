"""Tests for preprocessing module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing import compute_log_returns, run_adf_tests, winsorize_returns


def _make_price_frame(rows: int = 120, cols: int = 13) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=rows, freq="D")
    data = {
        f"S{i}": 100 + np.cumsum(np.random.default_rng(42 + i).normal(0.1, 1.0, rows))
        for i in range(cols)
    }
    return pd.DataFrame(data, index=dates)


def test_log_returns_shape() -> None:
    """Output has one fewer row than input prices."""
    prices = _make_price_frame()

    returns = compute_log_returns(prices)

    assert returns.shape[0] == prices.shape[0] - 1
    assert returns.shape[1] == prices.shape[1]


def test_log_returns_no_nulls() -> None:
    """No NaN values remain after log return computation."""
    prices = _make_price_frame()

    returns = compute_log_returns(prices)

    assert not returns.isna().any().any()


def test_adf_returns_all_sectors() -> None:
    """ADF results contain an entry for every sector column."""
    prices = _make_price_frame(rows=300)
    returns = compute_log_returns(prices)

    results = run_adf_tests(returns)

    assert set(results) == set(returns.columns)
    assert all("pvalue" in payload for payload in results.values())


def test_winsorize_clips_extremes() -> None:
    """Winsorization should keep values within the original quantile bounds."""
    values = list(range(20))
    returns = pd.DataFrame(
        {
            "S0": [-100.0, *values, 100.0],
            "S1": [-80.0, *[v * 0.5 for v in values], 80.0],
        }
    )
    limits = [0.05, 0.95]

    winsorized = winsorize_returns(returns, limits)

    assert winsorized["S0"].min() > returns["S0"].min()
    assert winsorized["S0"].max() < returns["S0"].max()
    assert winsorized["S1"].min() > returns["S1"].min()
    assert winsorized["S1"].max() < returns["S1"].max()
