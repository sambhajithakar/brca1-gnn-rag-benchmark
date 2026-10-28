#!/bin/sh
# v3 (2026-10-26): as queue_stage6_7_v2.sh, but stops on the first failure and requires PY.
set -eu
: "${PY:?set PY}"
RUN=${RUN:-$PY}
R=$(cd "$(dirname "$0")/.." && pwd)
cd "$R/20_GNN_Model_Source"
$RUN run_experiments.py --partition fold_P0 --subset struct --models gnn_D_evidence gnn_D_nograph lgbm_seqevo_ev
$RUN run_experiments.py --partition fold_P0 --subset all --models lgbm_seqevo_ev
for P in fold_P1 fold_P2 fold_LODO fold_random; do
  $RUN run_experiments.py --partition $P --subset struct --models zs_alphamissense ridge_seqevo lgbm_seqevo lgbm_struct labelsmooth gnn_A_seqonly gnn_B_struct gnn_C_struct_annot gnn_C_nograph gnn_D_evidence lgbm_seqevo_ev
done
$RUN run_experiments.py --partition fold_LOEO --subset struct --models zs_alphamissense ridge_seqevo lgbm_seqevo lgbm_struct gnn_C_struct_annot gnn_C_nograph
for V in heavy6_with_partner heavy10_with_partner heavy8_no_partner ca8_with_partner sequence_only; do
  $RUN run_experiments.py --partition fold_P0 --subset struct --models gnn_C_struct_annot --graph_variant $V --tag _$V
done
echo QUEUE_DONE
