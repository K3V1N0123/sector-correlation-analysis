# NSE Sector Correlation Analysis

**Tools:** Python · pandas · scipy · scikit-learn · seaborn · networkx  
**Data:** 13 NSE Nifty sectoral indices · 2015–2024 · Daily

## What This Project Does
Analyse pairwise co-movement across 13 NSE sectoral indices over 2015–2024.
The pipeline covers static and rolling correlation, structural break testing,
hierarchical clustering, PCA, and minimum spanning tree network analysis.

## Methodology
Static correlation -> Rolling correlation (252-day) -> Structural break testing
(Jennrich 1970) -> Hierarchical clustering (Ward) -> PCA -> Minimum Spanning
Tree

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebooks/07_full_pipeline.ipynb
```
