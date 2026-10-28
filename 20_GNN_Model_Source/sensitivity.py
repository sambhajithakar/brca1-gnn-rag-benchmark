"""Section 9 sensitivity analyses, from saved out-of-fold predictions.

S1 exclude splice-proximal (<= 2 nt) and RNA score < -1 variants (evaluation-time exclusion; models not refit)
S3 alternative partitions P1, P2 and the reference splits LODO (train on one region, test on the other), LOEO, random
S4 graph definitions (Model C with each graph variant vs the matched no-graph ablation on P0)
S5 GAT architecture (P0)
S2 (SNV-level labels) and S6 (first NMR model only) were not run; see 07_EXPERIMENT_LOG.md.
Writes results/sensitivity.json and results/sensitivity_table.tsv.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import boot_indices, fast_spearman, paired

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
KEY = ["zs_alphamissense", "lgbm_seqevo", "lgbm_struct", "labelsmooth", "gnn_A_seqonly", "gnn_B_struct", "gnn_C_struct_annot",
       "gnn_C_nograph", "gnn_D_evidence", "gnn_D_nograph", "lgbm_seqevo_ev", "gnn_C_gat"]


def load(part, subset="struct", mask=None):
    d = RES / "predictions" / part; base = None; P = {}
    for f in sorted(d.glob(f"*__{subset}.tsv")):
        t = pd.read_csv(f, sep="\t")
        if base is None: base = t[["aa_pos", "hgvs_p", "score"]].copy()
        P[f.name[: -len(f"__{subset}.tsv")]] = t.pred.to_numpy()
    if mask is not None:
        m = base.hgvs_p.isin(mask).to_numpy(); base = base[m].reset_index(drop=True); P = {k: v[m] for k, v in P.items()}
    return base, P


def block(base, P, label, pairs):
    boots = list(boot_indices(base.aa_pos.to_numpy(), b=1000))
    y = base.score.to_numpy()
    rows = [dict(analysis=label, model=k, n=len(y), spearman=fast_spearman(P[k], y)) for k in KEY if k in P and not np.isnan(P[k]).any()]
    tests = [dict(analysis=label, **paired(base, P, boots, a, b)) for a, b in pairs if a in P and b in P]
    return rows, tests


def main():
    rows, tests = [], []
    std_pairs = [("gnn_C_struct_annot", "gnn_C_nograph"), ("gnn_D_evidence", "gnn_C_struct_annot"), ("gnn_D_nograph", "gnn_C_nograph"),
                 ("lgbm_seqevo_ev", "lgbm_seqevo"), ("gnn_C_struct_annot", "zs_alphamissense"), ("gnn_C_struct_annot", "lgbm_seqevo")]
    for part in ("fold_P0", "fold_P1", "fold_P2", "fold_LODO", "fold_LOEO", "fold_random"):
        if not (RES / "predictions" / part).exists(): continue
        r, t = block(*load(part), f"partition:{part}", std_pairs + [("gnn_C_gat", "gnn_C_nograph")]); rows += r; tests += t
    # S1
    T = pd.read_csv(ROOT / "data/processed/model_table_brca1.tsv", sep="\t")
    keep = T[(~T.splice_proximal_any.astype(bool)) & ~(T.rna_score < -1)].hgvs_p
    r, t = block(*load("fold_P0", mask=set(keep)), "S1:no_splice_proximal_or_low_RNA", std_pairs); rows += r; tests += t
    # S4
    base, P = load("fold_P0")
    boots = list(boot_indices(base.aa_pos.to_numpy(), b=1000))
    for k in sorted(P):
        if k.startswith("gnn_C_struct_annot_"):
            tests.append(dict(analysis=f"S4:graph:{k.replace('gnn_C_struct_annot_', '')}", **paired(base, P, boots, k, "gnn_C_nograph")))
            rows.append(dict(analysis=f"S4:graph:{k.replace('gnn_C_struct_annot_', '')}", model=k, n=len(base), spearman=fast_spearman(P[k], base.score.to_numpy())))
    pd.DataFrame(rows).to_csv(RES / "sensitivity_table.tsv", sep="\t", index=False)
    pd.DataFrame(tests).to_csv(RES / "sensitivity_tests.tsv", sep="\t", index=False)
    json.dump({"rows": rows, "tests": tests, "n_S1": int(len(keep))}, open(RES / "sensitivity.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).pivot_table(index="model", columns="analysis", values="spearman").round(3).to_string())
    print(pd.DataFrame(tests).round(3).to_string())


if __name__ == "__main__":
    main()
