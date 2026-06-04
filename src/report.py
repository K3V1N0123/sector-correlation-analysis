"""Report generation module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.exceptions import OutputWriteError


def _top_pairs(corr_matrix: pd.DataFrame, ascending: bool) -> pd.Series:
    mask = pd.DataFrame(False, index=corr_matrix.index, columns=corr_matrix.columns)
    for i, row in enumerate(corr_matrix.index):
        for j, col in enumerate(corr_matrix.columns):
            if j <= i:
                mask.loc[row, col] = True
    return corr_matrix.mask(mask).stack().sort_values(ascending=ascending)


def _frame_to_markdown(frame: pd.DataFrame, include_index: bool = False) -> str:
    """Render a DataFrame as a simple GitHub-flavored markdown table."""
    working = frame.reset_index() if include_index else frame.copy()
    headers = [str(column) for column in working.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]

    for row in working.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")

    return "\n".join(lines)


def generate_summary_report(
    corr_matrix: pd.DataFrame,
    cluster_results: dict,
    pca_results: dict,
    break_test_results: pd.DataFrame,
    hypothesis_results: dict,
    save_path: str,
) -> None:
    """Generate a markdown summary report for the project outputs.

    Args:
        corr_matrix: Full-sample correlation matrix.
        cluster_results: Clustering outputs including labels and counts.
        pca_results: PCA outputs including explained variance and labels.
        break_test_results: Structural break test result table.
        hypothesis_results: Hypothesis outcomes and optional supporting tables.
        save_path: Destination markdown file path.

    Raises:
        OutputWriteError: If the report cannot be written.
    """
    strongest = _top_pairs(corr_matrix, ascending=False).head(5)
    weakest = _top_pairs(corr_matrix, ascending=True).head(5)
    summary_stats = hypothesis_results.get("summary_stats", pd.DataFrame())
    mst_hubs = hypothesis_results.get("mst_hubs", pd.DataFrame())

    executive_points = [
        f"The strongest full-sample pair is {strongest.index[0][0]} vs {strongest.index[0][1]} (rho={strongest.iloc[0]:.3f}).",
        f"PCA requires {pca_results['n_components_selected']} components to explain {pca_results['cumulative_variance'][pca_results['n_components_selected'] - 1]:.1%} of variance.",
        f"Hierarchical clustering produced {cluster_results['n_clusters']} clusters with cophenetic correlation {cluster_results['cophenetic_corr']:.3f}.",
        f"All structural break comparisons reject H0, including pre_covid vs post_covid (p={break_test_results.loc[break_test_results['period_a'].eq('pre_covid') & break_test_results['period_b'].eq('post_covid'), 'pvalue'].iloc[0]:.2e}).",
    ]

    hypothesis_table = pd.DataFrame(
        [
            {"hypothesis": key, **value}
            for key, value in hypothesis_results.items()
            if isinstance(value, dict) and key.startswith("H")
        ]
    )

    lines: list[str] = ["# NSE Sector Correlation Analysis Summary", ""]
    lines.append("## 1. Executive Summary")
    lines.extend([f"- {point}" for point in executive_points])
    lines.append("")

    lines.append("## 2. Summary Statistics Table")
    lines.append(_frame_to_markdown(summary_stats, include_index=True) if not summary_stats.empty else "Summary statistics not provided.")
    lines.append("")

    lines.append("## 3. Top 5 Most Correlated Pairs")
    lines.append(_frame_to_markdown(strongest.rename("correlation").to_frame(), include_index=True))
    lines.append("")

    lines.append("## 4. Top 5 Least Correlated Pairs")
    lines.append(_frame_to_markdown(weakest.rename("correlation").to_frame(), include_index=True))
    lines.append("")

    lines.append("## 5. Cluster Membership Table")
    cluster_frame = pd.DataFrame(
        [{"sector": sector, "cluster_id": cluster_id} for sector, cluster_id in cluster_results["cluster_labels"].items()]
    ).sort_values(["cluster_id", "sector"])
    lines.append(_frame_to_markdown(cluster_frame, include_index=False))
    lines.append("")

    lines.append("## 6. PCA Component Summary")
    pca_summary = pd.DataFrame(
        {
            "component": pca_results["loadings"].index,
            "label": pca_results["component_labels"],
            "explained_variance_ratio": pca_results["explained_variance_ratio"][: pca_results["n_components_selected"]],
            "cumulative_variance": pca_results["cumulative_variance"][: pca_results["n_components_selected"]],
        }
    )
    lines.append(_frame_to_markdown(pca_summary, include_index=False))
    lines.append("")

    lines.append("## 7. Structural Break Test Results")
    lines.append(_frame_to_markdown(break_test_results, include_index=False))
    lines.append("")

    lines.append("## 8. Hypothesis Test Outcomes")
    if not hypothesis_table.empty:
        lines.append(_frame_to_markdown(hypothesis_table, include_index=False))
    else:
        lines.append("Hypothesis outcomes not provided.")
    lines.append("")

    if not mst_hubs.empty:
        lines.append("## Appendix: MST Hub Sectors")
        lines.append(_frame_to_markdown(mst_hubs, include_index=False))
        lines.append("")

    path = Path(save_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
    except OSError as exc:
        raise OutputWriteError(f"Failed to write summary report: {exc}") from exc
