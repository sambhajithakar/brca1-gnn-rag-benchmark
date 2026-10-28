# 16A Pre-specified Analysis Plan (Stage 3)

Frozen: 2026-10-25, before any model was trained or any prediction was compared with a label. The SHA-256 of this file is recorded in 07_EXPERIMENT_LOG.md. Any later change is logged as a dated amendment at the end of this file, with its reason, and the original text is left unchanged.

## 1. Primary endpoint
Out-of-fold Spearman correlation (rho) between model predictions and the Findlay et al. 2018 SGE function score of BRCA1 missense protein variants (mean over SNVs encoding the same amino-acid change), pooled over a residue-grouped 5-fold cross-validation.

Analysis sets
- D1-all: 1,837 missense protein variants at 326 residues.
- D1-struct (primary set for structure comparisons): variants at residues present in an experimental structure (1-103 in PDB 1JM7 chain A; 1649-1859 in PDB 1T29 chain A).
- Results on D1-all are reported for models that do not need structure.

## 2. Data partitions (fixed before modelling)
- Primary partition P0: residues randomly assigned to 5 folds, stratified by domain (RING 1-110 vs BRCT region 1631-1863), seed 20261025. All variants at a residue share a fold. Each variant receives exactly one out-of-fold prediction.
- Additional partitions P1, P2 (seeds 20261026, 20261027) for sensitivity.
- Region-held-out: (R1) leave-one-domain-out, train RING and test BRCT and vice versa; (R2) leave-one-exon-out over the 13 assayed exons.
- Reference only: variant-level random 5-fold split (seed 20261025), reported to show the optimism of random splits; never used for claims about generalisation.
- Inner model selection: within each outer training set, residue-grouped 4-fold inner CV selects hyperparameters from the grids in section 4 using inner-CV Spearman. Outer test folds are never used for selection, early stopping or calibration.

## 3. Features (inputs) and forbidden inputs
Allowed
- F_sub (substitution descriptors): wild-type and mutant amino-acid one-hot; differences in hydrophobicity (Kyte-Doolittle), volume, charge, polarity; BLOSUM62 score; Grantham distance.
- F_evo (existing predictors and conservation): AlphaMissense raw score, EVE score, EVE evolutionary index, phyloP, CADD, PolyPhen-2, SIFT, Align-GVGD difference (protein-level mean over SNVs).
- F_struct (per residue, from experimental structures): relative solvent accessibility (Shrake-Rupley), DSSP secondary structure, number of heavy-atom contacts within 8 A, inter-chain interface flag (BARD1 for 1JM7, BACH1 peptide for 1T29), per-residue C-alpha RMSF across the 14 NMR models of 1JM7 (0 for the X-ray structure, with an indicator), B-factor (X-ray only, z-scored within structure), ProteinMPNN structure-conditioned log-probabilities of wild-type and mutant residues and per-site entropy (vanilla model v_48_020, temperature 1.0, computed on the complex with the partner chain present).
- F_annot: domain label (RING / BRCT1 / BRCT2 / linker), metal-binding (zinc-coordinating C/H in RING), interface flag.
Forbidden (never inputs, any model): any Findlay measurement other than the training-fold labels (RNA score, replicate scores, function class, p.nonfunctional); ClinVar classifications; Lyra track calls; Dace or HMEC scores; anything derived from held-out labels. Structure-derived features are computed from coordinates only.

## 4. Models
Zero-shot references: -AlphaMissense, -EVE, CADD, phyloP, ProteinMPNN log-odds (mutant minus wild-type), each evaluated directly with no fitting.

Supervised non-graph baselines (inputs F_sub + F_evo unless noted)
- Ridge regression (alpha in {0.1, 1, 10, 100}).
- Random forest (500 trees; max_features in {0.33, 0.66}; min_samples_leaf in {3, 10}).
- LightGBM (num_leaves in {7, 15}; learning_rate 0.03; n_estimators chosen by inner CV from {200, 400, 800}; min_child_samples 20).
- MLP (2 hidden layers of 64; dropout 0.2; weight decay in {1e-4, 1e-3}).
- Structure-informed non-graph: LightGBM on F_sub + F_evo + F_struct + F_annot + hand-crafted neighbourhood means of F_struct within 8 A.
- Structural label-smoothing: for each test variant, mean training-fold score of the same substitution type class at the k nearest residues (k in {5, 10}) in 3D, combined with F_sub + F_evo in a ridge model. Uses only training-fold labels.

Graph models (residue graph per structure; nodes are residues of all chains; variant head reads the embedding of the mutated residue)
- Model A (sequence-only network): same architecture as Model C with message passing removed and only F_sub + F_evo inputs (no structure, no annotations).
- Model B (structure-aware GNN): node features = residue identity + F_struct; edges = sequence adjacency + spatial contacts; variant head input = node embedding + F_sub. No F_evo.
- Model C (structure GNN + non-retrieval annotations): Model B plus F_evo (variant head) and F_annot (node features).
- Model C-nograph (matched ablation for H1): Model C with zero message-passing layers (identical inputs, head and training).
- Model D (evidence-augmented): Model C plus retrieved non-label evidence features from the Stage 7 retrieval system, under the leakage rules in section 8.
- Architecture: GraphSAGE-style mean aggregation with separate weights for sequence and spatial edge types (2 or 3 layers selected by inner CV; hidden 64; dropout 0.2), Adam lr 1e-3, weight decay 1e-4, MSE loss, maximum 300 epochs with early stopping on an inner validation split (20% of training residues). A graph attention variant (GAT) is run once as an architecture sensitivity analysis, not as a competitor for the primary test.
- Ensembles: each graph model is trained with 5 seeds (0-4); the ensemble mean is the prediction.

Graph definitions tested (sensitivity, Model C): contact cut-off 6, 8, 10 A on heavy atoms; C-alpha 8 A; with and without partner chains; sequence-only (1D chain) graph.

## 5. Hypothesis tests
- H1 (primary): delta rho = rho(Model C) - rho(Model C-nograph) on D1-struct under P0. 95% CI from a residue-cluster bootstrap (2,000 resamples of residues with all their variants, predictions fixed). Two-sided bootstrap p-value. H1 is supported if the CI lies entirely above 0. alpha = 0.05; no multiplicity correction for the single primary test.
- Secondary comparisons (Holm-corrected as a family): (S-a) Model C vs best supervised non-graph baseline; (S-b) Model C vs AlphaMissense zero-shot; (S-c) Model B vs Model A; (S-d) Model D vs Model C (H5); (S-e) Model C vs structure-informed non-graph LightGBM; (S-f) Model C vs structural label-smoothing.
- The "best non-graph baseline" is the supervised non-graph model with the highest out-of-fold rho; choosing it after seeing results can only favour the baseline.
- Secondary metrics for every model: Pearson r, RMSE, MAE; AUROC and AUPRC for protein-level loss of function (mean score < -1.328 vs > -0.748, intermediates excluded; thresholds from Findlay et al. as coded by Lyra et al.); per-domain rho.
- Reliability ceiling: Spearman-Brown ceiling (0.91) and the Findlay vs Dace re-assay agreement (0.754) are reported next to every rho.

## 6. External and cross-assay validation (no tuning on external data)
For variants that are also in the Findlay set, only out-of-fold (P0) predictions are used, so the model never saw the Findlay label of the same residue.
- E1: Dace 2025 HAP1 re-assay, exons 5 and 17 (rho).
- E2: Dace 2025 HAP1 new regions (exons 6, 10, 11, 12; no structure): models trained on all Findlay data, non-structural models only (rho, AUROC for Dace function class).
- E3: Dace 2025 HMEC SGE (rho).
- E4: Lyra independent tracks excluding T127 and T159: AUROC per track (n >= 20 with both classes) and pooled AUROC with a track-stratified bootstrap CI. The Findlay score itself is reported as an assay-level comparator.
- BRCA2 (secondary): binary LoF (2) vs normal (0) from Huang 2025; residue-grouped 5-fold CV (seed 20261025) with F_sub + AlphaMissense + physicochemical position context, LightGBM and logistic regression; evaluate out-of-fold on Huang, and as transfer on Sahu 2025 and Hu 2024 calls for the same variants; symmetric analysis trained on Sahu. AlphaMissense zero-shot and inter-assay agreement are reported as comparators. No 3D structure (blocked).

## 7. Uncertainty and selective prediction
- Split-conformal intervals (80% and 90%): within each outer fold, 20% of training residues form a calibration set; nonconformity = absolute residual; the model is refit on the remaining 80% for interval construction. Empirical coverage reported overall, per fold and per domain. Normalised conformal (residual divided by ensemble SD) as secondary.
- Ensemble SD across 5 seeds.
- Probability of LoF: P = Phi((-1.328 - mu) / sigma), sigma^2 = ensemble variance + calibration-set MSE; Brier score, expected calibration error (10 bins), reliability diagram.
- Selective prediction: risk-coverage curves (RMSE and LoF misclassification) when abstaining on widest intervals; area under the risk-coverage curve.
- Discordance link: variants where the Findlay class and an independent Lyra track disagree (binary) vs agree: compare interval widths (Mann-Whitney U) and enrichment among abstained variants.
- High-confidence errors: list variants with narrow intervals that miss the observed score.

## 8. Evidence retrieval leakage rules (Stage 7)
- Scenario 1 (strict prediction): the retrieval index for a given outer fold contains no document that states a functional outcome, clinical classification or assay call for any variant at a test-fold residue, and no Findlay-derived document at all. Automated test: every retrieved document is scanned for test-fold variant identifiers and outcome keywords; any hit fails the run.
- Scenario 2 (interpretation of characterised variants): all evidence may be retrieved and cited; outputs are evaluated for retrieval relevance and citation correctness, never reported as predictive accuracy.
- Retrieval metrics: precision@k and recall@k against variant-level gold relevance; wrong-variant retrieval rate; citation correctness; unsupported-statement rate; agreement of the synthesised functional consensus with the gold consensus.

## 9. Sensitivity analyses
S1 exclude splice-proximal (<= 2 nt) and RNA score < -1 variants; S2 SNV-level labels with SNV-level grouping by residue; S3 partitions P1, P2; S4 graph definitions (section 4); S5 GAT architecture; S6 1JM7 first NMR model only vs ensemble-derived features.

## 10. Reporting rules
All pre-specified analyses are reported, including null and negative results. No outcome is dropped. Every number in the manuscript is generated by scripts in 20_GNN_Model_Source/ from saved predictions in results/.
