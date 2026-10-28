#!/bin/sh
# Rebuilds data/processed/ (including model_table_brca1.tsv) from the fetched, checksum-verified inputs, without
# training any model. These are the preflight, checksums, datasets, structures, mpnn, graphs and features steps of
# scripts/run_all_v3.sh, copied unchanged. Needs PY and P3_EXT as for run_all_v3.sh. Run from the repository root.
set -eu
: "${PY:?set PY to the python of the pinned venv}"
: "${P3_EXT:?set P3_EXT (see scripts/fetch_inputs_v3.sh)}"
R=$(cd "$(dirname "$0")/.." && pwd)
MPNN=$P3_EXT/ProteinMPNN
export LYRA_REPO="$P3_EXT/FunctionalAssaYIntegration_BRCA1-BRCA2-Large-integration"
"$PY" "$R/scripts/preflight_v3.py"
cd "$R"; sha256sum -c --quiet data/SHA256SUMS.v3.txt
cd "$R/13_Variant_Annotation_Pipeline"; "$PY" build_datasets.py; "$PY" label_validity.py
cd "$R/17_Protein_Structure_Processing"; "$PY" process_structures.py
mkdir -p "$R/17_Protein_Structure_Processing/out/mpnn"
for p in 1jm7 1t29; do
  cd "$MPNN"
  "$PY" protein_mpnn_run.py --pdb_path "$R/17_Protein_Structure_Processing/out/clean_$p.pdb" --pdb_path_chains "A" \
     --out_folder "$R/17_Protein_Structure_Processing/out/mpnn/$p" --conditional_probs_only 1 --num_seq_per_target 10 \
     --batch_size 1 --seed 37 --model_name v_48_020 --path_to_model_weights vanilla_model_weights/
done
cd "$R/18_Residue_Graph_Construction"; "$PY" build_graphs.py
cd "$R/20_GNN_Model_Source"; "$PY" features.py
[ -s "$R/data/processed/model_table_brca1.tsv" ] && echo "data/processed rebuilt"
