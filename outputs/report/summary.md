# NSE Sector Correlation Analysis Summary

## 1. Executive Summary
- The strongest full-sample pair is BANK vs PVT_BANK (rho=0.985).
- PCA requires 5 components to explain 82.3% of variance.
- Hierarchical clustering produced 3 clusters with cophenetic correlation 0.956.
- All structural break comparisons reject H0, including pre_covid vs post_covid (p=7.34e-45).

## 2. Summary Statistics Table
| index | mean | std | min | max |
| --- | --- | --- | --- | --- |
| BANK | 0.000672 | 0.012028 | -0.036568 | 0.036354 |
| IT | 0.00069 | 0.012103 | -0.035703 | 0.034796 |
| FMCG | 0.000512 | 0.009027 | -0.025015 | 0.025908 |
| AUTO | 0.000521 | 0.012544 | -0.039347 | 0.035407 |
| PHARMA | 0.000334 | 0.011367 | -0.031715 | 0.032678 |
| FIN_SVC | 0.000701 | 0.011544 | -0.036223 | 0.033647 |
| METAL | 0.000781 | 0.016929 | -0.052509 | 0.043955 |
| ENERGY | 0.000787 | 0.01175 | -0.033511 | 0.031844 |
| REALTY | 0.00096 | 0.017185 | -0.054392 | 0.043414 |
| INFRA | 0.00062 | 0.010358 | -0.031941 | 0.027507 |
| MEDIA | -7e-05 | 0.015898 | -0.048571 | 0.040948 |
| PSU_BANK | 0.000439 | 0.019006 | -0.052934 | 0.052591 |
| PVT_BANK | 0.000578 | 0.012192 | -0.039219 | 0.035849 |

## 3. Top 5 Most Correlated Pairs
| level_0 | level_1 | correlation |
| --- | --- | --- |
| BANK | PVT_BANK | 0.9847528347062765 |
| BANK | FIN_SVC | 0.9598825447300253 |
| FIN_SVC | PVT_BANK | 0.9453591457614278 |
| ENERGY | INFRA | 0.7835930591800875 |
| AUTO | INFRA | 0.730411648552209 |

## 4. Top 5 Least Correlated Pairs
| level_0 | level_1 | correlation |
| --- | --- | --- |
| IT | PSU_BANK | 0.22073209072739475 |
| IT | REALTY | 0.31177081139076984 |
| IT | ENERGY | 0.3152276649336121 |
| BANK | IT | 0.31658236688320396 |
| IT | PVT_BANK | 0.317792445073147 |

## 5. Cluster Membership Table
| sector | cluster_id |
| --- | --- |
| AUTO | 1 |
| BANK | 1 |
| ENERGY | 1 |
| FIN_SVC | 1 |
| FMCG | 1 |
| INFRA | 1 |
| MEDIA | 1 |
| METAL | 1 |
| PSU_BANK | 1 |
| PVT_BANK | 1 |
| REALTY | 1 |
| PHARMA | 2 |
| IT | 3 |

## 6. PCA Component Summary
| component | label | explained_variance_ratio | cumulative_variance |
| --- | --- | --- | --- |
| PC1 | Market Beta Factor | 0.573442854232475 | 0.573442854232475 |
| PC2 | Technology/Growth Factor | 0.08798136684395834 | 0.6614242210764333 |
| PC3 | Commodity/Cyclical Factor | 0.06371200043301915 | 0.7251362215094524 |
| PC4 | Defensive Factor | 0.05249190048505198 | 0.7776281219945045 |
| PC5 | Defensive Factor | 0.045693412348125316 | 0.8233215343426298 |

## 7. Structural Break Test Results
| period_a | period_b | statistic | df | pvalue | reject_h0 |
| --- | --- | --- | --- | --- | --- |
| pre_covid | covid_shock | 509.6661170917618 | 78 | 1.3073463936490525e-64 | True |
| covid_shock | post_covid | 282.5563246874909 | 78 | 5.788496065934759e-25 | True |
| pre_covid | post_covid | 400.4937695759208 | 78 | 7.338949018018541e-45 | True |

## 8. Hypothesis Test Outcomes
| hypothesis | label | outcome | evidence |
| --- | --- | --- | --- |
| H1 | Crisis Contagion | supported | COVID shock correlations materially differ from pre-COVID matrix |
| H2 | Rate Sensitivity Cluster | supported | BANK, FIN_SVC, and REALTY share cluster 1 |
| H3 | Defensive Decoupling | pending | Not formally tested yet |
| H4 | Post-COVID Structural Shift | supported | Jennrich-style pre/post test rejects H0 |

## Appendix: MST Hub Sectors
| sector | degree | betweenness_centrality |
| --- | --- | --- |
| INFRA | 9 | 0.9090909090909092 |
| BANK | 3 | 0.3181818181818182 |
| FIN_SVC | 2 | 0.4090909090909091 |
| AUTO | 1 | 0.0 |
| ENERGY | 1 | 0.0 |
| FMCG | 1 | 0.0 |
| IT | 1 | 0.0 |
| MEDIA | 1 | 0.0 |
| METAL | 1 | 0.0 |
| PHARMA | 1 | 0.0 |
| PSU_BANK | 1 | 0.0 |
| PVT_BANK | 1 | 0.0 |
| REALTY | 1 | 0.0 |
