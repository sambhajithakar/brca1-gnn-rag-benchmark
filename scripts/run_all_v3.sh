#!/bin/sh
# Project 3 full pipeline, v3 (2026-10-26). Supersedes run_all.sh and run_all_v2.sh (kept unchanged for the record).
# Fails immediately on: unset variables, any unsuccessful command, missing inputs or outputs, checksum mismatch,
# packages that differ from requirements.lock.v3.txt, external repositories at the wrong commit, failed leakage tests.
# Required environment:
#   PY      python of a venv built with: pip install -r requirements.lock.v3.txt   (Python 3.13)
#   P3_EXT  directory holding ProteinMPNN and the Lyra integration repository (created by scripts/fetch_inputs_v3.sh)
# Optional: P3_DETERMINISTIC=1 runs network training with torch deterministic algorithms (not used for reported numbers);
#           P3_BUILD_TIMESTAMP fixes the corpus timestamp (default below) so the corpus file is byte-reproducible.
set -eu
: "${PY:?set PY to the python of the pinned venv}"
: "${P3_EXT:?set P3_EXT (see scripts/fetch_inputs_v3.sh)}"
R=$(cd "$(dirname "$0")/.." && pwd)
MPNN=$P3_EXT/ProteinMPNN
export LYRA_REPO="$P3_EXT/FunctionalAssaYIntegration_BRCA1-BRCA2-Large-integration"
export P3_BUILD_TIMESTAMP="${P3_BUILD_TIMESTAMP:-2026-10-26T00:00:00Z}"
if [ "${P3_DETERMINISTIC:-0}" = "1" ]; then RUN="$PY $R/scripts/det_run_v3.py"; else RUN="$PY"; fi
step() { echo "=== $(date -u +%H:%M:%S) $1"; }
need() { for f in "$@"; do [ -s "$f" ] || { echo "MISSING OUTPUT: $f" >&2; exit 1; }; done; }

step preflight
"$PY" "$R/scripts/preflight_v3.py"
step checksums
cd "$R"
sha256sum -c --quiet data/SHA256SUMS.v3.txt
echo "all $(wc -l < data/SHA256SUMS.v3.txt) inputs verified"

step datasets
cd "$R/13_Variant_Annotation_Pipeline"
"$PY" build_datasets.py
"$PY" label_validity.py
need "$R/data/processed/brca1_findlay2018_missense_protein.tsv" "$R/data/processed/stage2_qc.json"

step structures
cd "$R/17_Protein_Structure_Processing"
"$PY" process_structures.py
need "$R/17_Protein_Structure_Processing/out/clean_1jm7.pdb" "$R/17_Protein_Structure_Processing/out/clean_1t29.pdb"
mkdir -p "$R/17_Protein_Structure_Processing/out/mpnn"
for p in 1jm7 1t29; do
  step "mpnn $p"
  cd "$MPNN"
  "$PY" protein_mpnn_run.py --pdb_path "$R/17_Protein_Structure_Processing/out/clean_$p.pdb" --pdb_path_chains "A" \
     --out_folder "$R/17_Protein_Structure_Processing/out/mpnn/$p" --conditional_probs_only 1 --num_seq_per_target 10 \
     --batch_size 1 --seed 37 --model_name v_48_020 --path_to_model_weights vanilla_model_weights/
done

step graphs
cd "$R/18_Residue_Graph_Construction"
"$PY" build_graphs.py
need "$R/18_Residue_Graph_Construction/out/graph_qc.json"
step features
cd "$R/20_GNN_Model_Source"
"$PY" features.py
need "$R/data/processed/model_table_brca1.tsv"

step corpus
cd "$R/23_RAG_Implementation"
"$PY" build_corpus_v3.py
need "$R/23_RAG_Implementation/out/evidence_corpus.tsv"
"$PY" tests_leakage.py > out/leakage_tests_v3.log 2>&1 || { cat out/leakage_tests_v3.log; echo "LEAKAGE TESTS FAILED" >&2; exit 1; }
grep -q "ALL LEAKAGE TESTS PASSED" out/leakage_tests_v3.log
tail -1 out/leakage_tests_v3.log

cd "$R/20_GNN_Model_Source"
step "P0 struct"
$RUN run_experiments.py --partition fold_P0 --subset struct
step "P0 all"
$RUN run_experiments.py --partition fold_P0 --subset all
step queue
PY=$PY RUN=$RUN sh "$R/scripts/queue_stage6_7_v3.sh"

step evaluate
for P in fold_P0 fold_P1 fold_P2 fold_LODO fold_random fold_LOEO; do
  "$PY" evaluate.py --partition $P --subset struct > "$R/results/eval_${P}_struct.txt"
  need "$R/results/tests__${P}__struct.json"
done
"$PY" evaluate.py --partition fold_P0 --subset all > "$R/results/eval_fold_P0_all.txt"
step scenario2
cd "$R/23_RAG_Implementation"
"$PY" scenario2_interpretation.py
step uncertainty
cd "$R/26_Uncertainty_Calibration"
"$PY" uncertainty.py
need "$R/26_Uncertainty_Calibration/out/uncertainty_summary.json"
step validation
cd "$R/29_External_Validation"
"$PY" validate_v3.py
need "$R/29_External_Validation/out/external_validation.json"
step sensitivity
cd "$R/20_GNN_Model_Source"
"$PY" sensitivity.py
need "$R/results/sensitivity_table.tsv"
step explain
cd "$R/27_Explainability"
"$PY" explain.py
step figures
cd "$R/scripts"
"$PY" make_figures.py
step DONE
