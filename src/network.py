"""Network analysis module."""

from __future__ import annotations

import itertools
import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from src.exceptions import OutputWriteError

logger = logging.getLogger(__name__)


def build_mst(corr_matrix: pd.DataFrame) -> nx.Graph:
    """Construct a minimum spanning tree from a correlation matrix.

    Args:
        corr_matrix: Full-sample sector correlation matrix.

    Returns:
        Minimum spanning tree graph with distance and correlation edge attributes.
    """
    clamped = corr_matrix.clip(lower=-0.9999, upper=0.9999)
    graph = nx.Graph()

    for node in clamped.index:
        graph.add_node(node, sector=node)

    for sector_a, sector_b in itertools.combinations(clamped.columns, 2):
        rho = float(clamped.loc[sector_a, sector_b])
        distance = float(np.sqrt(2.0 * (1.0 - rho)))
        graph.add_edge(sector_a, sector_b, weight=distance, correlation=rho)

    mst = nx.minimum_spanning_tree(graph, algorithm="kruskal", weight="weight")
    return mst


def get_hub_sectors(mst: nx.Graph) -> pd.DataFrame:
    """Compute hub metrics from an MST.

    Args:
        mst: Minimum spanning tree graph.

    Returns:
        DataFrame sorted by degree and betweenness centrality.
    """
    betweenness = nx.betweenness_centrality(mst)
    rows = []

    for node, degree in mst.degree():
        rows.append(
            {
                "sector": node,
                "degree": int(degree),
                "betweenness_centrality": float(betweenness[node]),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["degree", "betweenness_centrality", "sector"], ascending=[False, False, True]
    ).reset_index(drop=True)


def save_hub_sectors(hubs: pd.DataFrame, output_path: str) -> None:
    """Save MST hub metrics to CSV.

    Args:
        hubs: Hub-sector DataFrame.
        output_path: Destination CSV path.

    Raises:
        OutputWriteError: If the file cannot be written.
    """
    path = Path(output_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        hubs.to_csv(path, index=False)
    except OSError as exc:
        raise OutputWriteError(f"Failed to save MST hub sectors: {exc}") from exc

    logger.info("Saved MST hub sectors to %s", output_path)
