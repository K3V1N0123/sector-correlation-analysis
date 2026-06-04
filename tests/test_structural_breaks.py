"""Tests for structural break module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.structural_breaks import jennrich_test, split_sub_periods


def _make_returns_frame() -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=180, freq="D")
    rng = np.random.default_rng(11)
    data = rng.normal(0.0, 0.01, size=(180, 13))
    columns = [f"S{i}" for i in range(13)]
    return pd.DataFrame(data, index=index, columns=columns)


def test_jennrich_returns_required_keys() -> None:
    """Jennrich output should include the required fields."""
    corr_a = pd.DataFrame(np.eye(3))
    corr_b = pd.DataFrame(np.eye(3))

    result = jennrich_test(corr_a, 100, corr_b, 120)

    assert set(result) == {"statistic", "df", "pvalue", "reject_h0"}


def test_jennrich_symmetric_input() -> None:
    """Jennrich test should handle non-perfectly symmetric input gracefully."""
    corr_a = pd.DataFrame([[1.0, 0.2, 0.1], [0.19, 1.0, 0.3], [0.1, 0.3, 1.0]])
    corr_b = pd.DataFrame([[1.0, 0.1, 0.0], [0.1, 1.0, 0.2], [0.0, 0.2, 1.0]])

    result = jennrich_test(corr_a, 100, corr_b, 120)

    assert isinstance(result["statistic"], float)
    assert 0.0 <= result["pvalue"] <= 1.0


def test_subperiod_split_coverage() -> None:
    """All input dates should appear in exactly one configured sub-period."""
    returns = _make_returns_frame()
    sub_periods = {
        "period_1": {"start": "2020-01-01", "end": "2020-02-29"},
        "period_2": {"start": "2020-03-01", "end": "2020-04-30"},
        "period_3": {"start": "2020-05-01", "end": "2020-06-28"},
    }

    split = split_sub_periods(returns, sub_periods, min_subperiod_days=10)

    combined_index = split["period_1"].index.append(split["period_2"].index).append(split["period_3"].index)
    assert len(combined_index) == len(returns)
    assert combined_index.is_unique
