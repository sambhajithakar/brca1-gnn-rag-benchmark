# External validation report

Status: COMPLETE. Script: `validate.py`. Output: `out/external_validation.json`. No tuning was done on external data.
For variants that are also in the Findlay set, only out-of-fold P0 predictions are used.

| Set | n | Reference agreement | AlphaMissense | Best supervised | Model C |
|---|---|---|---|---|---|
| E1 Dace HAP1 re-assay (Spearman) | 302 | Findlay vs Dace 0.754 | 0.613 (0.483–0.701) | LightGBM + structure 0.620 (0.519–0.696) | 0.562 (0.449–0.649) |
| E3 Dace HMEC (Spearman) | 399 | Findlay vs HMEC 0.600 | 0.442 (0.313–0.553) | LightGBM + structure 0.497 (0.373–0.602) | 0.459 (0.341–0.564) |
| E2 Dace new regions (Spearman / AUROC LoF) | 1,685 (59 LoF) | — | 0.205 / 0.889 | ridge sub+AM 0.175 / 0.860 | not applicable (no structure) |
| E4 Lyra independent tracks (pooled AUROC) | 31 tracks, ~3,900 calls | Findlay score 0.870 (0.824–0.892) | 0.856 | LightGBM + structure 0.876 (0.836–0.895) | 0.869 (0.828–0.888) |

## BRCA2 (secondary; binary LoF vs normal; residue-grouped 5-fold CV)
| Trained on | Evaluated on | LightGBM | Logistic | AlphaMissense | Training assay's calls as predictor |
|---|---|---|---|---|---|
| Huang 2025 | Huang (OOF), n=3,970 | 0.866 | 0.853 | 0.845 | — |
| Huang 2025 | Sahu 2025, n=3,762 | 0.746 | 0.752 | 0.757 | 0.753 |
| Huang 2025 | Hu 2024 HDR, n=450 | 0.920 | 0.914 | 0.925 | 0.941 |
| Sahu 2025 | Sahu (OOF) | 0.814 | 0.775 | 0.757 | — |
| Sahu 2025 | Huang | 0.822 | 0.835 | 0.845 | 0.809 |

## Conclusions
- Models supervised on one assay do not outperform AlphaMissense on independent measurements of the same variants or
  in new regions.
- Within-assay gains are partly assay-specific.
- For BRCA2, transfer between the two large SGE studies is limited by their own agreement (AUROC about 0.75–0.81 when
  one assay's calls are used to predict the other's).
