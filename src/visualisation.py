"""Visualisation module."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import dendrogram

PALETTE = {
    "background": "#0D1117",
    "surface": "#161B22",
    "accent": "#1A56A0",
    "positive": "#2EA043",
    "negative": "#CF222E",
    "neutral": "#8B949E",
    "text": "#E6EDF3",
    "highlight": "#F0883E",
}


def _prepare_figure(save_path: str) -> Path:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="dark")
    return path


def _finalize_plot(fig: plt.Figure, save_path: str) -> None:
    path = _prepare_figure(save_path)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=PALETTE["background"])
    plt.close(fig)


def _heatmap_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "sector_corr",
        [PALETTE["negative"], "#FFFFFF", PALETTE["accent"]],
    )


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    pvalue_matrix: pd.DataFrame,
    title: str,
    save_path: str,
) -> None:
    """Plot an annotated correlation heatmap with significance masking."""
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=PALETTE["background"])
    mask = pvalue_matrix > 0.05
    ax.set_facecolor(PALETTE["surface"])
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=_heatmap_cmap(),
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor=PALETTE["background"],
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title(title, color=PALETTE["text"])
    ax.tick_params(colors=PALETTE["text"], labelrotation=45)
    _finalize_plot(fig, save_path)


def plot_rolling_correlation(
    rolling_corr: pd.DataFrame,
    pairs: list[str],
    event_dates: dict,
    save_path: str,
) -> None:
    """Plot rolling correlations for selected sector pairs."""
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=PALETTE["background"])
    ax.set_facecolor(PALETTE["surface"])

    colors = [PALETTE["accent"], PALETTE["highlight"], PALETTE["positive"], PALETTE["neutral"]]
    for i, pair in enumerate(pairs):
        ax.plot(rolling_corr.index, rolling_corr[pair], label=pair, color=colors[i % len(colors)], linewidth=1.8)

    for label, window in event_dates.items():
        ax.axvspan(
            pd.to_datetime(window["start"]),
            pd.to_datetime(window["end"]),
            color=window.get("color", PALETTE["negative"]),
            alpha=0.18,
            label=label,
        )

    ax.axhline(0.0, color=PALETTE["neutral"], linestyle="--", linewidth=1)
    ax.set_title("Rolling Correlation", color=PALETTE["text"])
    ax.tick_params(colors=PALETTE["text"])
    ax.legend(facecolor=PALETTE["surface"], labelcolor=PALETTE["text"])
    _finalize_plot(fig, save_path)


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    sector_labels: list[str],
    save_path: str,
) -> None:
    """Plot a horizontal hierarchical clustering dendrogram."""
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=PALETTE["background"])
    ax.set_facecolor(PALETTE["surface"])
    dendrogram(linkage_matrix, labels=sector_labels, orientation="right", ax=ax, color_threshold=None)
    ax.set_title("Sector Clustering Dendrogram", color=PALETTE["text"])
    ax.tick_params(colors=PALETTE["text"])
    _finalize_plot(fig, save_path)


def plot_scree(
    explained_variance: np.ndarray,
    cumulative_variance: np.ndarray,
    n_selected: int,
    save_path: str,
) -> None:
    """Plot PCA scree bars with cumulative variance overlay."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=PALETTE["background"])
    ax.set_facecolor(PALETTE["surface"])
    components = np.arange(1, len(explained_variance) + 1)
    ax.bar(components, explained_variance, color=PALETTE["accent"], alpha=0.85)
    ax.plot(components, cumulative_variance, color=PALETTE["highlight"], marker="o")
    ax.axvline(n_selected, color=PALETTE["positive"], linestyle="--")
    ax.axhline(0.80, color=PALETTE["neutral"], linestyle="--")
    ax.set_title("PCA Scree Plot", color=PALETTE["text"])
    ax.set_xlabel("Principal Component", color=PALETTE["text"])
    ax.set_ylabel("Explained Variance", color=PALETTE["text"])
    ax.tick_params(colors=PALETTE["text"])
    _finalize_plot(fig, save_path)


def plot_mst(
    mst: nx.Graph,
    hub_sectors: pd.DataFrame,
    save_path: str,
) -> None:
    """Plot the minimum spanning tree network."""
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=PALETTE["background"])
    ax.set_facecolor(PALETTE["surface"])
    pos = nx.spring_layout(mst, seed=42)
    degree_map = hub_sectors.set_index("sector")["degree"].to_dict()
    node_sizes = [300 + 250 * degree_map.get(node, 1) for node in mst.nodes]
    edge_widths = [2 + 4 * abs(data.get("correlation", 0.0)) for _, _, data in mst.edges(data=True)]

    nx.draw_networkx_edges(mst, pos, ax=ax, width=edge_widths, edge_color=PALETTE["neutral"], alpha=0.8)
    nx.draw_networkx_nodes(mst, pos, ax=ax, node_size=node_sizes, node_color=PALETTE["accent"])
    nx.draw_networkx_labels(mst, pos, ax=ax, font_color=PALETTE["text"], font_size=9)
    ax.set_title("Minimum Spanning Tree", color=PALETTE["text"])
    ax.axis("off")
    _finalize_plot(fig, save_path)


def plot_subperiod_comparison(
    corr_matrices: dict,
    save_path: str,
) -> None:
    """Plot side-by-side sub-period correlation heatmaps."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=PALETTE["background"])
    cmap = _heatmap_cmap()
    titles = {
        "pre_covid": "Pre-COVID",
        "covid_shock": "COVID Shock",
        "post_covid": "Post-COVID",
    }

    for ax, key in zip(axes, ["pre_covid", "covid_shock", "post_covid"], strict=True):
        ax.set_facecolor(PALETTE["surface"])
        sns.heatmap(
            corr_matrices[key],
            cmap=cmap,
            vmin=-1,
            vmax=1,
            annot=False,
            linewidths=0.3,
            linecolor=PALETTE["background"],
            cbar=ax is axes[-1],
            ax=ax,
        )
        ax.set_title(titles[key], color=PALETTE["text"])
        ax.tick_params(colors=PALETTE["text"], labelrotation=45)

    _finalize_plot(fig, save_path)
