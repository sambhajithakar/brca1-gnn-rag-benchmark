# BRCA1 residue-graph and evidence-retrieval benchmark: code and outputs

Archived on Zenodo from GitHub releases. The DOI is shown on the Zenodo record of the release and in `CITATION.cff` once assigned.

Companion to: "Machine Learning and Graph Neural Networks for BRCA1 Variant Function Prediction: A Leakage-Controlled Evaluation of Evidence Retrieval" (Thakar SB, Ghorpade PN; manuscript in preparation, not yet submitted).

## Contents

| Path | Content |
|---|---|
| `13_…/`, `17_…/`, `18_…/`, `20_GNN_Model_Source/`, `23_RAG_Implementation/`, `26_…/`, `27_…/`, `29_…/` | Analysis code (dataset harmonisation, structures, graphs, models and evaluation, evidence corpus and retrieval, uncertainty, explainability, external validation) |
| `scripts/` | `fetch_inputs_v3.sh` (downloads every input from its original source and checks SHA-256), `run_all_v3.sh` (fail-fast pipeline), `preflight_v3.py`, `make_figures.py` |
| `60_final_stage/depth_control/` | Post hoc depth-matched control: written specification, wrapper, comparison scripts, saved predictions and results |
| `70_v3_2/verification/` | Read-only statistical reconciliation of H1 and the depth control, external paired tests, with saved outputs |
| `results/` | Out-of-fold predictions of all trained models for every partition, metrics and test statistics |
| `26_Uncertainty_Calibration/out/` | Conformal intervals and summary |
| `16A_PRESPECIFIED_ANALYSIS_PLAN.md` | Analysis plan frozen before test evaluation (SHA-256 46b41462…a2c28a2) |
| `REPRODUCE_v3.md`, `INPUTS_AND_ENVIRONMENT_v3.md`, `requirements.lock.v3.txt`, `data/SHA256SUMS.v3.txt` | Reproduction steps, input sources, pins and licences, locked environment, input checksums |
| `tools/rebuild_processed.sh`, `tools/attach_labels.py` | Rebuild `data/processed/` from the fetched inputs without training, and re-insert the Findlay score column into the deposited prediction and interval tables |
| `STAGING_MANIFEST.tsv` | SHA-256 of every staged file |

## What is deliberately not included

Third-party values whose licence terms do not clearly permit redistribution are not included, and are regenerated from the original sources by `scripts/fetch_inputs_v3.sh` and `scripts/run_all_v3.sh`:

- Findlay et al. (2018) function scores (the `score` column is removed from all prediction and interval tables);
- CADD, phyloP, PolyPhen-2, SIFT and Align-GVGD scores (from the Findlay supplementary table) and EVE scores;
- processed tables in `data/processed/` and the evidence corpus (`evidence_corpus.tsv`), which contain the above values and are rebuilt by the pipeline.

## Quick verification of the primary statistics (no model training)

```sh
python3.13 -m venv env && env/bin/pip install -r requirements.lock.v3.txt
export PY=$PWD/env/bin/python P3_EXT=$PWD/../p3_external
sh scripts/fetch_inputs_v3.sh            # downloads inputs and external code, verifies checksums
sh tools/rebuild_processed.sh            # rebuilds data/processed/ (no training; ProteinMPNN inference only)
$PY tools/attach_labels.py               # restores the Findlay score column in the deposited tables
$PY 70_v3_2/verification/reconcile_v3_2.py   # about 4 minutes on a 4-core CPU
```

Compare the new `70_v3_2/verification/out/reconcile_v3_2.json` with the deposited copy (keep a copy before running; the script overwrites it). Expected P0 values: Model C 0.626, C-nograph 0.619, C-depthctl 0.614; Model C − C-nograph +0.007 (−0.008 to 0.021), p = 0.344; Model C − C-depthctl +0.013 (−0.0004 to 0.025), p = 0.057. The full pipeline, including model training, is `sh scripts/run_all_v3.sh` (about 1 hour; network outputs may differ by up to 0.0002 in Spearman because CPU scatter operations are not deterministic).

## Licences

- Code written for this study: MIT License (`LICENSE`).
- Deposited outputs (predictions, intervals, metrics, test statistics and logs under `results/`, `26_Uncertainty_Calibration/out/`, `60_final_stage/depth_control/results/` and `70_v3_2/verification/out/`): CC BY-NC-SA 4.0 (`LICENSE-OUTPUTS.md`), because trained-model predictions use AlphaMissense (CC BY-NC-SA 4.0) as a feature.
- AlphaMissense zero-shot prediction files included for the verification scripts: CC BY-NC-SA 4.0, Cheng et al. (2023).
- Lyra et al. data (CC BY 4.0) and the Dace et al. repository (MIT) are used with attribution. PDB entries 1JM7 and 1T29: CC0. ProteinMPNN: MIT (cloned at a pinned commit, not redistributed).

`STAGING_MANIFEST.tsv` lists the SHA-256 of every analysis file as staged on 2026-10-28; the licence and citation files were added afterwards.
