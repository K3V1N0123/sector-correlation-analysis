"""Hierarchical clustering module."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import squareform

from src.exceptions import OutputWriteError

logger = logging.getLogger(__name__)


def compute_distance_matrix(corr_matrix: pd.DataFrame) -> pd.DataFrame:
    """Convert a correlation matrix to a clustering distance matrix.

    Args:
        corr_matrix: Sector correlation matrix.

    Returns:
        Distance matrix using ``1 - abs(rho)``.
    """
    distance = (1.0 - corr_matrix.abs()).copy()
    for column in distance.columns:
        distance.loc[column, column] = 0.0

    return distance.astype(float)


def run_hierarchical_clustering(
    distance_matrix: pd.DataFrame,
    linkage_method: str = "ward",
    n_clusters: int | None = None,
) -> dict:
    """Perform agglomerative hierarchical clustering.

    Args:
        distance_matrix: Symmetric sector distance matrix.
        linkage_method: Linkage criterion.
        n_clusters: Desired number of flat clusters. Defaults to 3.

    Returns:
        Clustering results including linkage, cluster labels, and cophenetic fit.
    """
    n_clusters = n_clusters or 3
    condensed = squareform(distance_matrix.to_numpy(), checks=False)
    linkage_matrix = linkage(condensed, method=linkage_method)
    cluster_ids = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
    cophenetic_corr, _ = cophenet(linkage_matrix, condensed)

    labels = dict(zip(distance_matrix.index, cluster_ids, strict=True))
    return {
        "linkage_matrix": linkage_matrix,
        "cluster_labels": labels,
        "n_clusters": int(n_clusters),
        "cophenetic_corr": float(cophenetic_corr),
    }


def save_cluster_memberships(cluster_labels: dict[str, int], output_path: str) -> pd.DataFrame:
    """Save cluster assignments to CSV.

    Args:
        cluster_labels: Mapping of sector label to cluster id.
        output_path: Destination CSV path.

    Returns:
        Saved cluster-membership DataFrame.

    Raises:
        OutputWriteError: If the file cannot be written.
    """
    memberships = pd.DataFrame(
        [{"sector": sector, "cluster_id": cluster_id} for sector, cluster_id in cluster_labels.items()]
    ).sort_values(["cluster_id", "sector"]).reset_index(drop=True)

    path = Path(output_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        memberships.to_csv(path, index=False)
    except OSError as exc:
        raise OutputWriteError(f"Failed to save cluster memberships: {exc}") from exc

    logger.info("Saved cluster memberships to %s", output_path)
    return memberships
