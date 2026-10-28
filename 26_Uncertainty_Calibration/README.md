# 26. Uncertainty and calibration report

Status: COMPLETE. Script: `uncertainty.py`. Outputs: `out/uncertainty_summary.json`, `out/*_intervals.tsv`, and
`figures/stage8_uncertainty_*.png/svg`.

## Method (pre-specified, section 7 of 16A)
Results are for partition P0, D1-struct (1,734 variants).

Within each outer fold:
- 20% of the training residues (seeded) form the calibration set.
- The model is refit with 5 seeds on the remaining 80%, using the hyperparameters chosen by inner CV in the primary run.
- Standard split conformal uses |y − μ|. Normalised conformal uses |y − μ| / (SD + 0.001).
- P(LoF) = Φ((−1.328 − μ)/σ), where σ² = ensemble variance + calibration MSE.

## Results
| | LightGBM seq+evo | Model C (GNN) |
|---|---|---|
| Empirical coverage, 80% / 90% intervals | 0.818 / 0.915 | 0.809 / 0.919 |
| Mean width of 90% interval (score units) | 2.33 | 2.32 |
| Normalised conformal, 90% coverage | 0.910 | 0.924 |
| 90% coverage, RING / BRCT | 0.910 / 0.917 | 0.922 / 0.917 |
| Brier score for P(LoF) (prevalence-only reference 0.187) | 0.102 | 0.103 |
| Expected calibration error (10 bins) | 0.035 | 0.056 |
| Spearman (ensemble SD, absolute error) | 0.31 | 0.21 |
| RMSE, all vs 50% most confident (by SD) | 0.669 → 0.548 | 0.665 → 0.615 |
| Area under risk-coverage curve, RMSE (SD / normalised width) | 0.559 / 0.562 | 0.610 / 0.610 |

## Interpretation
- **Marginal coverage is at nominal** for both models, overall and within each region.
- **The intervals are wide.** A 90% interval spans about 2.3 score units, while the Findlay LoF and normal thresholds
  are 0.58 units apart. Most individual predictions therefore cannot settle a functional class by themselves.
- **The gradient-boosting ensemble ranks its own errors better than the GNN ensemble does,** so it supports more useful
  abstention. Keeping the 50% most confident predictions cuts RMSE by 18% for LightGBM and 8% for the GNN.
- **P(LoF) is well calibrated** for LightGBM (ECE 0.035) and slightly less so for the GNN (ECE 0.056).
- **Fold-level coverage** is in `out/uncertainty_summary.json`.
