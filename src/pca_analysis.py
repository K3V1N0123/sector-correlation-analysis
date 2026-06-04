"""PCA analysis module."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.exceptions import OutputWriteError

logger = logging.getLogger(__name__)


def interpret_components(loadings: pd.DataFrame) -> list[str]:
    """Heuristically label PCA components from their sector loadings.

    Args:
        loadings: Component loading matrix with components as rows.

    Returns:
        List of component labels.
    """
    labels: list[str] = []

    for i, (_, row) in enumerate(loadings.iterrows(), start=1):
        abs_row = row.abs().sort_values(ascending=False)
        top = set(abs_row.head(4).index)

        if {"BANK", "FIN_SVC", "REALTY"}.issubset(top):
            labels.append("Rate-Sensitive Factor")
        elif {"FMCG", "PHARMA"}.issubset(top):
            labels.append("Defensive Factor")
        elif {"METAL", "ENERGY"}.issubset(top):
            labels.append("Commodity/Cyclical Factor")
        elif "IT" in top:
            labels.append("Technology/Growth Factor")
        elif row.min() > 0 or row.max() < 0:
            labels.append("Market Beta Factor")
        else:
            labels.append(f"Factor {i}")

    return labels


def run_pca(
    returns: pd.DataFrame,
    variance_threshold: float = 0.80,
) -> dict:
    """Run PCA on standardized sector returns.

    Args:
        returns: Log returns DataFrame.
        variance_threshold: Cumulative explained variance target.

    Returns:
        PCA result bundle including loadings, scores, and labels.
    """
    scaler = StandardScaler()
    standardized = scaler.fit_transform(returns)

    pca = PCA()
    scores = pca.fit_transform(standardized)
    explained = pca.explained_variance_ratio_
    cumulative = explained.cumsum()
    n_selected = int(np.searchsorted(cumulative, variance_threshold) + 1)

    component_names = [f"PC{i}" for i in range(1, len(explained) + 1)]
    selected_names = component_names[:n_selected]
    loadings = pd.DataFrame(
        pca.components_[:n_selected],
        index=selected_names,
        columns=returns.columns,
    )
    score_frame = pd.DataFrame(scores[:, :n_selected], index=returns.index, columns=selected_names)
    component_labels = interpret_components(loadings)

    return {
        "explained_variance_ratio": explained,
        "cumulative_variance": cumulative,
        "n_components_selected": n_selected,
        "loadings": loadings,
        "scores": score_frame,
        "component_labels": component_labels,
    }


def save_pca_loadings(loadings: pd.DataFrame, component_labels: list[str], output_path: str) -> pd.DataFrame:
    """Save PCA loadings with interpreted labels.

    Args:
        loadings: PCA loading matrix.
        component_labels: Interpreted labels corresponding to each loading row.
        output_path: Destination CSV path.

    Returns:
        Saved loadings DataFrame with component labels.

    Raises:
        OutputWriteError: If the file cannot be written.
    """
    frame = loadings.copy()
    frame.insert(0, "component_label", component_labels)
    frame.index.name = "component"

    path = Path(output_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path)
    except OSError as exc:
        raise OutputWriteError(f"Failed to save PCA loadings: {exc}") from exc

    logger.info("Saved PCA loadings to %s", output_path)
    return frame
