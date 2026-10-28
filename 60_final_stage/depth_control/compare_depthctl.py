"""Compare depth-matched control with Model C and C-nograph using the primary bootstrap code (evaluate.py)."""
import json, sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "20_GNN_Model_Source"))
from evaluate import boot_indices, paired, fast_spearman  # noqa: E402
out = {}
for part in ["fold_P0", "fold_P1", "fold_P2", "fold_LODO", "fold_LOEO", "fold_random"]:
    rd = lambda p: pd.read_csv(p, sep="\t")
    C = rd(ROOT / f"results/predictions/{part}/gnn_C_struct_annot__struct.tsv")
    N = rd(ROOT / f"results/predictions/{part}/gnn_C_nograph__struct.tsv")
    Dc = rd(HERE / f"results/predictions/{part}/gnn_C_struct_annot_depthctl__struct.tsv")
    assert (C.hgvs_p.values == Dc.hgvs_p.values).all() and (C.hgvs_p.values == N.hgvs_p.values).all()
    base = C[["aa_pos", "score"]]; preds = {"C": C.pred.to_numpy(), "nograph": N.pred.to_numpy(), "depthctl": Dc.pred.to_numpy()}
    boots = list(boot_indices(base.aa_pos.to_numpy()))
    y = base.score.to_numpy()
    out[part] = {"spearman": {k: round(fast_spearman(v, y), 4) for k, v in preds.items()},
                 "C_minus_depthctl": paired(base, preds, boots, "C", "depthctl"),
                 "depthctl_minus_nograph": paired(base, preds, boots, "depthctl", "nograph"),
                 "C_minus_nograph_H1": paired(base, preds, boots, "C", "nograph"),
                 "depthctl_layers": json.load(open(HERE / f"results/logs/{part}__gnn_C_struct_annot_depthctl__struct.json"))["folds"]}
    out[part]["depthctl_layers"] = [v["n_layers"] for v in out[part]["depthctl_layers"].values()]
    o = out[part]
    print(part, o["spearman"], "C-ctl %+.4f [%.4f, %.4f] p=%.3f" % tuple(o["C_minus_depthctl"][k] for k in ("delta_spearman", "ci_lo", "ci_hi", "p_boot")),
          "ctl-nog %+.4f [%.4f, %.4f] p=%.3f" % tuple(o["depthctl_minus_nograph"][k] for k in ("delta_spearman", "ci_lo", "ci_hi", "p_boot")), "layers", o["depthctl_layers"])
json.dump(out, open(HERE / "depthctl_results.json", "w"), indent=1)
