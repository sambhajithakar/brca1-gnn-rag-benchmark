"""Sequence-only-edge model vs the depth-matched control at P0 (both have Model C's depth)."""
import json, sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "20_GNN_Model_Source"))
from evaluate import boot_indices, paired  # noqa: E402
rd = lambda p: pd.read_csv(p, sep="\t")
C = rd(ROOT / "results/predictions/fold_P0/gnn_C_struct_annot__struct.tsv")
S = rd(ROOT / "results/predictions/fold_P0/gnn_C_struct_annot_sequence_only__struct.tsv")
Dc = rd(HERE / "results/predictions/fold_P0/gnn_C_struct_annot_depthctl__struct.tsv")
assert (C.hgvs_p.values == S.hgvs_p.values).all() and (C.hgvs_p.values == Dc.hgvs_p.values).all()
base = C[["aa_pos", "score"]]; P = {"C": C.pred.to_numpy(), "seqonly": S.pred.to_numpy(), "depthctl": Dc.pred.to_numpy()}
b = list(boot_indices(base.aa_pos.to_numpy()))
r = {"seqonly_minus_depthctl": paired(base, P, b, "seqonly", "depthctl"), "C_minus_seqonly": paired(base, P, b, "C", "seqonly")}
print(json.dumps(r, indent=1)); json.dump(r, open(HERE / "seqonly_vs_depthctl_P0.json", "w"), indent=1)
