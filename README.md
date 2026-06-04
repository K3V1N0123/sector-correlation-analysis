# NSE Sector Correlation Analysis

**Tools:** Python · pandas · scipy · scikit-learn · seaborn · networkx  
**Data:** 13 NSE Nifty sectoral indices · 2015–2024 · Daily

## Overview
This project analyses co-movement across 13 NSE sectoral indices using daily
data from 2015 to 2024. The pipeline is structured as a reproducible research
workflow that moves from raw price ingestion to cleaned returns, correlation
analysis, structural break testing, clustering, PCA, network analysis, figures,
and a final markdown report.

The goal is to understand how tightly Indian sector indices move together and
whether COVID produced a lasting structural shift in cross-sector dependence.

## Research Questions
1. Which NSE sectors are most and least correlated over the full sample?
2. Did COVID significantly change the sector correlation structure?
3. Which sectors cluster together because of shared macro exposure?
4. How many latent factors explain most of the variation in sector returns?
5. Which sectors act as hubs in the dependency network?

## Methodology
Static correlation -> Rolling correlation (252-day) -> Structural break testing
(Jennrich 1970) -> Hierarchical clustering (Ward) -> PCA -> Minimum Spanning
Tree

## Current Results Snapshot
The current pipeline outputs indicate:
1. `BANK` and `PVT_BANK` are the strongest full-sample pair.
2. All structural break comparisons reject equality of correlation matrices.
3. `BANK`, `FIN_SVC`, and `REALTY` cluster together in the current run.
4. Five principal components explain about 82.3% of total variance.
5. `INFRA` appears as the dominant MST hub in the current network output.

## Repository Structure
```text
sector-correlation-analysis/
├── config.yaml
├── data/
│   ├── processed/
│   └── raw/
├── notebooks/
│   ├── 07_full_pipeline.ipynb
│   └── sample.ipynb
├── outputs/
│   ├── figures/
│   ├── report/
│   └── tables/
├── src/
│   ├── clustering.py
│   ├── correlation.py
│   ├── exceptions.py
│   ├── ingestion.py
│   ├── network.py
│   ├── pca_analysis.py
│   ├── preprocessing.py
│   ├── report.py
│   ├── structural_breaks.py
│   └── visualisation.py
└── tests/
```

## Generated Outputs
Tables:
- `outputs/tables/adf_results.csv`
- `outputs/tables/corr_matrix_full.csv`
- `outputs/tables/rolling_correlation.csv`
- `outputs/tables/corr_matrix_pre_covid.csv`
- `outputs/tables/corr_matrix_covid.csv`
- `outputs/tables/corr_matrix_post_covid.csv`
- `outputs/tables/break_tests.csv`
- `outputs/tables/cluster_memberships.csv`
- `outputs/tables/pca_loadings.csv`
- `outputs/tables/mst_hubs.csv`

Figures:
- `outputs/figures/heatmap_full.png`
- `outputs/figures/heatmap_subperiods.png`
- `outputs/figures/rolling_correlation.png`
- `outputs/figures/dendrogram.png`
- `outputs/figures/pca_scree.png`
- `outputs/figures/mst_network.png`

Report:
- `outputs/report/summary.md`

## How to Run
Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run the full test suite:

```bash
python -m pytest tests
```

Open the master notebook:

```bash
jupyter notebook notebooks/07_full_pipeline.ipynb
```

## Notes
1. The active analysis scope uses 13 sectors.
2. The processed return panel starts in 2016 because `PVT_BANK` has a shorter
   usable history than the rest of the sample.
3. `outputs/report/summary.md` is the best single-file summary of the current
   pipeline results.
