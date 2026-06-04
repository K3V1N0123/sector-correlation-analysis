"""Tests for clustering module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.clustering import compute_distance_matrix, run_hierarchical_clustering


def _make_corr_matrix() -> pd.DataFrame:
    labels = ["A", "B", "C", "D"]
    data = np.array(
        [
            [1.0, 0.9, 0.2, 0.1],
            [0.9, 1.0, 0.25, 0.15],
            [0.2, 0.25, 1.0, 0.8],
            [0.1, 0.15, 0.8, 1.0],
        ]
    )
    return pd.DataFrame(data, index=labels, columns=labels)


def test_distance_matrix_diagonal_zero() -> None:
    """Distance matrix should have zeros on the diagonal."""
    corr = _make_corr_matrix()

    distance = compute_distance_matrix(corr)

    assert np.allclose(np.diag(distance), 0.0)


def test_distance_matrix_matches_formula() -> None:
    """Distance matrix should equal 1 - |rho|."""
    corr = _make_corr_matrix()

    distance = compute_distance_matrix(corr)

    assert np.isclose(distance.loc["A", "B"], 0.1)
    assert np.isclose(distance.loc["C", "D"], 0.2)


def test_hierarchical_clustering_returns_expected_keys() -> None:
    """Clustering results should include all required fields."""
    corr = _make_corr_matrix()
    distance = compute_distance_matrix(corr)

    result = run_hierarchical_clustering(distance, linkage_method="average", n_clusters=2)

    assert set(result) == {"linkage_matrix", "cluster_labels", "n_clusters", "cophenetic_corr"}
    assert len(result["cluster_labels"]) == len(distance)
    assert result["n_clusters"] == 2
