"""v3.2 post hoc: each alternative graph definition (and GAT) versus the depth-matched control at P0, from saved predictions.
Uses evaluate.py (unchanged), seed 20261025, 2,000 residue resamples. Writes only to out/."""
import json, sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "20_GNN_Model_Source"))
from evaluate import boot_indices, paired, fast_spearman  # noqa: E402
d = ROOT / "results/predictions/fold_P0"; rd = lambda p: pd.read_csv(p, sep="\t")
C = rd(d / "gnn_C_struct_annot__struct.tsv"); base = C[["aa_pos", "score"]]
P = {"depthctl": rd(ROOT / "60_final_stage/depth_control/results/predictions/fold_P0/gnn_C_struct_annot_depthctl__struct.tsv").pred.to_numpy(),
     "nograph": rd(d / "gnn_C_nograph__struct.tsv").pred.to_numpy()}
names = ["gnn_C_struct_annot", "gnn_C_struct_annot_heavy6_with_partner", "gnn_C_struct_annot_heavy10_with_partner",
         "gnn_C_struct_annot_heavy8_no_partner", "gnn_C_struct_annot_ca8_with_partner", "gnn_C_struct_annot_sequence_only", "gnn_C_gat"]
for n in names:
    t = rd(d / f"{n}__struct.tsv"); assert (t.hgvs_p.values == C.hgvs_p.values).all(); P[n] = t.pred.to_numpy()
b = list(boot_indices(base.aa_pos.to_numpy())); y = base.score.to_numpy(); out = {}
for n in names:
    out[n] = {"rho": fast_spearman(P[n], y), "minus_depthctl": paired(base, P, b, n, "depthctl"), "minus_nograph": paired(base, P, b, n, "nograph")}
    a, c = out[n]["minus_depthctl"], out[n]["minus_nograph"]
    print("%-42s %.4f | -depthctl %+.4f [%.4f, %.4f] p=%.3f | -nograph %+.4f [%.4f, %.4f]" % (n, out[n]["rho"], a["delta_spearman"], a["ci_lo"], a["ci_hi"], a["p_boot"], c["delta_spearman"], c["ci_lo"], c["ci_hi"]))
json.dump(out, open(HERE / "out/graph_variants_vs_depthctl_P0.json", "w"), indent=1)
