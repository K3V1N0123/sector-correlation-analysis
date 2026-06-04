"""Tests for PCA analysis module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pca_analysis import interpret_components, run_pca


def _make_returns_frame(rows: int = 250) -> pd.DataFrame:
    rng = np.random.default_rng(21)
    base = rng.normal(0.0, 0.01, size=(rows, 1))
    noise = rng.normal(0.0, 0.003, size=(rows, 13))
    data = base + noise
    columns = [
        "BANK",
        "IT",
        "FMCG",
        "AUTO",
        "PHARMA",
        "FIN_SVC",
        "METAL",
        "ENERGY",
        "REALTY",
        "INFRA",
        "MEDIA",
        "PSU_BANK",
        "PVT_BANK",
    ]
    index = pd.date_range("2020-01-01", periods=rows, freq="D")
    return pd.DataFrame(data, index=index, columns=columns)


def test_run_pca_selects_components() -> None:
    """PCA should select at least one component and return matching scores."""
    returns = _make_returns_frame()

    result = run_pca(returns, variance_threshold=0.80)

    assert result["n_components_selected"] >= 1
    assert result["scores"].shape[1] == result["n_components_selected"]
    assert len(result["component_labels"]) == result["n_components_selected"]


def test_interpret_components_rate_sensitive_label() -> None:
    """Rate-sensitive sectors should trigger the expected heuristic label."""
    loadings = pd.DataFrame(
        [[0.8, 0.1, 0.1, 0.1, 0.1, 0.75, 0.1, 0.1, 0.7, 0.1, 0.1, 0.1, 0.1]],
        index=["PC1"],
        columns=[
            "BANK",
            "IT",
            "FMCG",
            "AUTO",
            "PHARMA",
            "FIN_SVC",
            "METAL",
            "ENERGY",
            "REALTY",
            "INFRA",
            "MEDIA",
            "PSU_BANK",
            "PVT_BANK",
        ],
    )

    labels = interpret_components(loadings)

    assert labels == ["Rate-Sensitive Factor"]
