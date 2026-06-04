"""Tests for network analysis module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.network import build_mst, get_hub_sectors


def _make_corr_matrix() -> pd.DataFrame:
    labels = ["A", "B", "C", "D"]
    data = np.array(
        [
            [1.0, 0.9, 0.3, 0.2],
            [0.9, 1.0, 0.4, 0.1],
            [0.3, 0.4, 1.0, 0.8],
            [0.2, 0.1, 0.8, 1.0],
        ]
    )
    return pd.DataFrame(data, index=labels, columns=labels)


def test_build_mst_has_n_minus_one_edges() -> None:
    """MST should contain exactly n - 1 edges."""
    corr = _make_corr_matrix()

    mst = build_mst(corr)

    assert mst.number_of_nodes() == len(corr)
    assert mst.number_of_edges() == len(corr) - 1


def test_get_hub_sectors_returns_required_columns() -> None:
    """Hub-sector output should include sector, degree, and betweenness."""
    corr = _make_corr_matrix()
    mst = build_mst(corr)

    hubs = get_hub_sectors(mst)

    assert list(hubs.columns) == ["sector", "degree", "betweenness_centrality"]
    assert hubs.iloc[0]["degree"] >= hubs.iloc[-1]["degree"]
