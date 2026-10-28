# Inputs and environment (v3, 2026-10-26)

All inputs are re-created by `scripts/fetch_inputs_v3.sh` and verified against `data/SHA256SUMS.v3.txt` (full SHA-256 there; 38 files).
Metadata files `MIRROR_COMMIT.txt` / `SOURCE_COMMIT.txt` are regenerated from the pinned clones and not checksummed.
`data/raw/alphafold_ACCESS_DENIED/` (403 error bodies) is evidence of blocked access only and is not an input.

## Input files
| File | Content | Source and pinned version | Licence | SHA-256 (prefix) |
|---|---|---|---|---|
| `data/inputs/09_Prior_Art_Comparison.xlsx` | Verified literature table used for the 25 literature records of the evidence corpus (project-authored) | this project (copy of 09_Prior_Art_Comparison.xlsx) | project file | `c1b78396c4592e01…` |
| `data/raw/Findlay2018_dmsview_mirror/1jm7.pdb` | Findlay et al. 2018 Supp. Table 1 (mirror) + PDB 1JM7, 1T29 | github.com/dms-view/BRCA1 @ 81d1dce3472d0efdaffc0b7233fe091b3875caaf, data/Findlay2018/ | Publisher supplementary terms (Findlay data); PDB entries CC0; repository has no licence file | `c7aa34bdd1af3d70…` |
| `data/raw/Findlay2018_dmsview_mirror/1t29.pdb` | Findlay et al. 2018 Supp. Table 1 (mirror) + PDB 1JM7, 1T29 | github.com/dms-view/BRCA1 @ 81d1dce3472d0efdaffc0b7233fe091b3875caaf, data/Findlay2018/ | Publisher supplementary terms (Findlay data); PDB entries CC0; repository has no licence file | `b664e37acd2eaad1…` |
| `data/raw/Findlay2018_dmsview_mirror/Findlay2018_brctdomain.csv` | Findlay et al. 2018 Supp. Table 1 (mirror) + PDB 1JM7, 1T29 | github.com/dms-view/BRCA1 @ 81d1dce3472d0efdaffc0b7233fe091b3875caaf, data/Findlay2018/ | Publisher supplementary terms (Findlay data); PDB entries CC0; repository has no licence file | `c94cb9d193d10c36…` |
| `data/raw/Findlay2018_dmsview_mirror/Findlay2018_brctdomain.md` | Findlay et al. 2018 Supp. Table 1 (mirror) + PDB 1JM7, 1T29 | github.com/dms-view/BRCA1 @ 81d1dce3472d0efdaffc0b7233fe091b3875caaf, data/Findlay2018/ | Publisher supplementary terms (Findlay data); PDB entries CC0; repository has no licence file | `9df0dd4f911f1430…` |
| `data/raw/Findlay2018_dmsview_mirror/Findlay2018_ringdomain.csv` | Findlay et al. 2018 Supp. Table 1 (mirror) + PDB 1JM7, 1T29 | github.com/dms-view/BRCA1 @ 81d1dce3472d0efdaffc0b7233fe091b3875caaf, data/Findlay2018/ | Publisher supplementary terms (Findlay data); PDB entries CC0; repository has no licence file | `ca1172116790c5d6…` |
| `data/raw/Findlay2018_dmsview_mirror/Findlay2018_ringdomain.md` | Findlay et al. 2018 Supp. Table 1 (mirror) + PDB 1JM7, 1T29 | github.com/dms-view/BRCA1 @ 81d1dce3472d0efdaffc0b7233fe091b3875caaf, data/Findlay2018/ | Publisher supplementary terms (Findlay data); PDB entries CC0; repository has no licence file | `437214ef0046cf1d…` |
| `data/raw/Findlay2018_dmsview_mirror/Findlay2018_suppTable1.csv` | Findlay et al. 2018 Supp. Table 1 (mirror) + PDB 1JM7, 1T29 | github.com/dms-view/BRCA1 @ 81d1dce3472d0efdaffc0b7233fe091b3875caaf, data/Findlay2018/ | Publisher supplementary terms (Findlay data); PDB entries CC0; repository has no licence file | `6f018c7738cad00f…` |
| `data/raw/Findlay2018_dmsview_mirror/README.md` | Findlay et al. 2018 Supp. Table 1 (mirror) + PDB 1JM7, 1T29 | github.com/dms-view/BRCA1 @ 81d1dce3472d0efdaffc0b7233fe091b3875caaf, data/Findlay2018/ | Publisher supplementary terms (Findlay data); PDB entries CC0; repository has no licence file | `51097211e7696d25…` |
| `data/raw/Findlay2018_dmsview_mirror/build_data.ipynb` | Findlay et al. 2018 Supp. Table 1 (mirror) + PDB 1JM7, 1T29 | github.com/dms-view/BRCA1 @ 81d1dce3472d0efdaffc0b7233fe091b3875caaf, data/Findlay2018/ | Publisher supplementary terms (Findlay data); PDB entries CC0; repository has no licence file | `d581f699e9471d74…` |
| `data/raw/alphamissense/AlphaMissense_aa_substitutions_BRCA1_BRCA2.tsv` | AlphaMissense amino-acid substitutions, rows P38398 and P51587 | storage.googleapis.com/dm_alphamissense/AlphaMissense_aa_substitutions.tsv.gz (2023 release), filtered | CC BY-NC-SA 4.0 (non-commercial) | `3fdaa6ca03338235…` |
| `data/raw/dace2025_repo/230420_BRCA1_hg19_hg38_conversion.csv` | Dace et al. 2025 (medRxiv) tables; second Findlay copy; EVE; ClinVar export 2024-09-13; SpliceAI; hg19-hg38 table | github.com/phoebedace/BRCA1_SGE_HAP1_HMEC @ 37fc69b7…, input_files/ and data_for_plots/ | MIT (repository); EVE and ClinVar terms apply to those files | `bc7ed2e4f9837946…` |
| `data/raw/dace2025_repo/240913_clinvar.txt` | Dace et al. 2025 (medRxiv) tables; second Findlay copy; EVE; ClinVar export 2024-09-13; SpliceAI; hg19-hg38 table | github.com/phoebedace/BRCA1_SGE_HAP1_HMEC @ 37fc69b7…, input_files/ and data_for_plots/ | MIT (repository); EVE and ClinVar terms apply to those files | `f5c92c839b25fd17…` |
| `data/raw/dace2025_repo/241024_EVE_scores_BRCA1_HUMAN.csv` | Dace et al. 2025 (medRxiv) tables; second Findlay copy; EVE; ClinVar export 2024-09-13; SpliceAI; hg19-hg38 table | github.com/phoebedace/BRCA1_SGE_HAP1_HMEC @ 37fc69b7…, input_files/ and data_for_plots/ | MIT (repository); EVE and ClinVar terms apply to those files | `4b46870dd47c3283…` |
| `data/raw/dace2025_repo/250703_SpliceAI_scores_500bp_all_vars.csv` | Dace et al. 2025 (medRxiv) tables; second Findlay copy; EVE; ClinVar export 2024-09-13; SpliceAI; hg19-hg38 table | github.com/phoebedace/BRCA1_SGE_HAP1_HMEC @ 37fc69b7…, input_files/ and data_for_plots/ | MIT (repository); EVE and ClinVar terms apply to those files | `591c02b398a1b0ab…` |
| `data/raw/dace2025_repo/Findlay_et_al_2018_data.csv` | Dace et al. 2025 (medRxiv) tables; second Findlay copy; EVE; ClinVar export 2024-09-13; SpliceAI; hg19-hg38 table | github.com/phoebedace/BRCA1_SGE_HAP1_HMEC @ 37fc69b7…, input_files/ and data_for_plots/ | MIT (repository); EVE and ClinVar terms apply to those files | `79b3b4aa82ca22e5…` |
| `data/raw/dace2025_repo/LICENSE` | Dace et al. 2025 (medRxiv) tables; second Findlay copy; EVE; ClinVar export 2024-09-13; SpliceAI; hg19-hg38 table | github.com/phoebedace/BRCA1_SGE_HAP1_HMEC @ 37fc69b7…, input_files/ and data_for_plots/ | MIT (repository); EVE and ClinVar terms apply to those files | `2e3ce63cf2a85062…` |
| `data/raw/dace2025_repo/SUPPLEMENTARY_TABLES_1-8.xlsx` | Dace et al. 2025 (medRxiv) tables; second Findlay copy; EVE; ClinVar export 2024-09-13; SpliceAI; hg19-hg38 table | github.com/phoebedace/BRCA1_SGE_HAP1_HMEC @ 37fc69b7…, input_files/ and data_for_plots/ | MIT (repository); EVE and ClinVar terms apply to those files | `657c5d0865e4608f…` |
| `data/raw/dace2025_repo/amplicon_coords_hg19.txt` | Dace et al. 2025 (medRxiv) tables; second Findlay copy; EVE; ClinVar export 2024-09-13; SpliceAI; hg19-hg38 table | github.com/phoebedace/BRCA1_SGE_HAP1_HMEC @ 37fc69b7…, input_files/ and data_for_plots/ | MIT (repository); EVE and ClinVar terms apply to those files | `8aa5ec07b476d97d…` |
| `data/raw/dace2025_repo/amplicon_fasta/BRCA1i11.fa` | Genomic amplicon FASTA (Dace et al.) | github.com/phoebedace/BRCA1_SGE_HAP1_HMEC @ 37fc69b72d583f271052f70f340d2b3862420f5a, input_files/fasta/ | MIT | `e4d015bd5e024f14…` |
| `data/raw/dace2025_repo/amplicon_fasta/BRCA1u1.fa` | Genomic amplicon FASTA (Dace et al.) | github.com/phoebedace/BRCA1_SGE_HAP1_HMEC @ 37fc69b72d583f271052f70f340d2b3862420f5a, input_files/fasta/ | MIT | `899259cb4c3c30ff…` |
| `data/raw/dace2025_repo/amplicon_fasta/BRCA1x1.fa` | Genomic amplicon FASTA (Dace et al.) | github.com/phoebedace/BRCA1_SGE_HAP1_HMEC @ 37fc69b72d583f271052f70f340d2b3862420f5a, input_files/fasta/ | MIT | `c78875acc0c7053b…` |
| `data/raw/dace2025_repo/amplicon_fasta/BRCA1x10a.fa` | Genomic amplicon FASTA (Dace et al.) | github.com/phoebedace/BRCA1_SGE_HAP1_HMEC @ 37fc69b72d583f271052f70f340d2b3862420f5a, input_files/fasta/ | MIT | `c5286ecdcf375afd…` |
| `data/raw/dace2025_repo/amplicon_fasta/BRCA1x10d.fa` | Genomic amplicon FASTA (Dace et al.) | github.com/phoebedace/BRCA1_SGE_HAP1_HMEC @ 37fc69b72d583f271052f70f340d2b3862420f5a, input_files/fasta/ | MIT | `d594fd7d8aea56c3…` |
| `data/raw/dace2025_repo/amplicon_fasta/BRCA1x10h.fa` | Genomic amplicon FASTA (Dace et al.) | github.com/phoebedace/BRCA1_SGE_HAP1_HMEC @ 37fc69b72d583f271052f70f340d2b3862420f5a, input_files/fasta/ | MIT | `bf66a4a54428f957…` |
| `data/raw/dace2025_repo/amplicon_fasta/BRCA1x10h2.fa` | Genomic amplicon FASTA (Dace et al.) | github.com/phoebedace/BRCA1_SGE_HAP1_HMEC @ 37fc69b72d583f271052f70f340d2b3862420f5a, input_files/fasta/ | MIT | `efcd660c8fb8df33…` |
| `data/raw/dace2025_repo/amplicon_fasta/BRCA1x11.fa` | Genomic amplicon FASTA (Dace et al.) | github.com/phoebedace/BRCA1_SGE_HAP1_HMEC @ 37fc69b72d583f271052f70f340d2b3862420f5a, input_files/fasta/ | MIT | `3cd20db74bff120a…` |
| `data/raw/dace2025_repo/amplicon_fasta/BRCA1x12a.fa` | Genomic amplicon FASTA (Dace et al.) | github.com/phoebedace/BRCA1_SGE_HAP1_HMEC @ 37fc69b72d583f271052f70f340d2b3862420f5a, input_files/fasta/ | MIT | `a917efd8a9b7a442…` |
| `data/raw/dace2025_repo/amplicon_fasta/BRCA1x12b.fa` | Genomic amplicon FASTA (Dace et al.) | github.com/phoebedace/BRCA1_SGE_HAP1_HMEC @ 37fc69b72d583f271052f70f340d2b3862420f5a, input_files/fasta/ | MIT | `9b3eadb879ec5795…` |
| `data/raw/dace2025_repo/amplicon_fasta/BRCA1x17.fa` | Genomic amplicon FASTA (Dace et al.) | github.com/phoebedace/BRCA1_SGE_HAP1_HMEC @ 37fc69b72d583f271052f70f340d2b3862420f5a, input_files/fasta/ | MIT | `2601092ec7dc298e…` |
| `data/raw/dace2025_repo/amplicon_fasta/BRCA1x3.fa` | Genomic amplicon FASTA (Dace et al.) | github.com/phoebedace/BRCA1_SGE_HAP1_HMEC @ 37fc69b72d583f271052f70f340d2b3862420f5a, input_files/fasta/ | MIT | `bb9255651b637d00…` |
| `data/raw/dace2025_repo/amplicon_fasta/BRCA1x5.fa` | Genomic amplicon FASTA (Dace et al.) | github.com/phoebedace/BRCA1_SGE_HAP1_HMEC @ 37fc69b72d583f271052f70f340d2b3862420f5a, input_files/fasta/ | MIT | `df971987a9a4686c…` |
| `data/raw/dace2025_repo/amplicon_fasta/BRCA1x6a.fa` | Genomic amplicon FASTA (Dace et al.) | github.com/phoebedace/BRCA1_SGE_HAP1_HMEC @ 37fc69b72d583f271052f70f340d2b3862420f5a, input_files/fasta/ | MIT | `de4c83f6898b6168…` |
| `data/raw/lyra2026_repo/CITATION.cff` | Lyra et al. FYI-BRCA1 v2.0 / FYI-BRCA2 v1.0 integration | github.com/FunctionalAssaYIntegration/BRCA1-BRCA2-Large-integration @ 908fb3760358cd6a7e8f3438b2185bd29004f9aa | CC BY 4.0 (data/LICENSE) | `5e11c50c95a8248f…` |
| `data/raw/lyra2026_repo/LICENSE` | Lyra et al. FYI-BRCA1 v2.0 / FYI-BRCA2 v1.0 integration | github.com/FunctionalAssaYIntegration/BRCA1-BRCA2-Large-integration @ 908fb3760358cd6a7e8f3438b2185bd29004f9aa | CC BY 4.0 (data/LICENSE) | `d7274b58f67710c0…` |
| `data/raw/lyra2026_repo/SUPP_TABLES_BRCA12_APR_2026.xlsx` | Lyra et al. FYI-BRCA1 v2.0 / FYI-BRCA2 v1.0 integration | github.com/FunctionalAssaYIntegration/BRCA1-BRCA2-Large-integration @ 908fb3760358cd6a7e8f3438b2185bd29004f9aa | CC BY 4.0 (data/LICENSE) | `71e7d32f0ec3ab6f…` |
| `data/raw/lyra2026_repo/checksums.sha256` | Lyra et al. FYI-BRCA1 v2.0 / FYI-BRCA2 v1.0 integration | github.com/FunctionalAssaYIntegration/BRCA1-BRCA2-Large-integration @ 908fb3760358cd6a7e8f3438b2185bd29004f9aa | CC BY 4.0 (data/LICENSE) | `8871507b2ddb19e8…` |
| `data/raw/proteingym_ref/DMS_substitutions.csv` | ProteinGym reference file (UniProt sequences) | raw.githubusercontent.com/OATML-Markslab/ProteinGym @ 144fe22b07dfaeec2b366f2346203a9838a55b4c, reference_files/DMS_substitutions.csv | MIT | `a8f498011532a74a…` |

## External code repositories (cloned by fetch_inputs_v3.sh into $P3_EXT)
| Repository | Commit | Used for | Licence |
|---|---|---|---|
| github.com/dauparas/ProteinMPNN | 8907e6671bfbfc92303b5f79c4b5e6ce47cdef57 | Structure-conditioned log-probabilities (model v_48_020, weights in repository, seed 37) | MIT |
| github.com/FunctionalAssaYIntegration/BRCA1-BRCA2-Large-integration | 908fb3760358cd6a7e8f3438b2185bd29004f9aa | `src/domains.py` (BRCA2 domain intervals) in validate_v3.py; data source | code per repository; data CC BY 4.0 |
| github.com/dms-view/BRCA1 | 81d1dce3472d0efdaffc0b7233fe091b3875caaf | data source only | — |
| github.com/phoebedace/BRCA1_SGE_HAP1_HMEC | 37fc69b72d583f271052f70f340d2b3862420f5a | data source only | MIT |

## Environment variables
| Variable | Required | Meaning |
|---|---|---|
| `PY` | yes (run_all_v3.sh) | Python 3.13 of a venv built from `requirements.lock.v3.txt` |
| `P3_EXT` | yes (fetch and run) | Directory for the external clones |
| `LYRA_REPO` | set automatically by run_all_v3.sh to `$P3_EXT/FunctionalAssaYIntegration_BRCA1-BRCA2-Large-integration`; required if validate_v3.py is run alone | |
| `P3_BUILD_TIMESTAMP` | optional (default 2026-10-26T00:00:00Z) | Fixed corpus timestamp for a byte-reproducible corpus |
| `P3_DETERMINISTIC` | optional (default 0) | 1 = torch deterministic algorithms for network training (not used for reported numbers) |

## Software
Python 3.13.17 (tested), Linux x86-64, 4 vCPU, 15 GB RAM, no GPU. System tools: git, curl, sha256sum, zcat, awk.
`requirements.lock.v3.txt` pins 68 packages (identical to requirements.lock.v2.txt; `requirements.lock.txt` from v1 is incomplete and must not be used). Key packages: biopython==1.88, einops==0.8.2, lightgbm==4.7.0, matplotlib==3.11.2, numpy==2.5.4, openpyxl==3.1.5, pandas==3.1.0, pydssp==0.9.1, rank-bm25==0.2.2, scikit-learn==1.9.1, scipy==1.18.1, statsmodels==0.15.0, torch==2.14.1, tqdm==4.70.1.

## Seeds
Partitions P0 20261025, P1 20261026, P2 20261027; network seeds 0–4; inner split 1000+seed; conformal split 777+fold;
bootstrap 20261025; Scenario 2 query sample 20261025; ProteinMPNN 37.
