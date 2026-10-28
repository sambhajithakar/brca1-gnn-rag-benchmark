"""Stage 8b: explainability and error analysis.

1. SHAP-style contributions (LightGBM pred_contrib, TreeSHAP) for lgbm_struct refit on each P0 training fold and
   applied to its test fold (out-of-fold attributions). Mean |contribution| per feature and per feature group.
2. Out-of-fold error analysis for Model C and LightGBM: absolute error by region, burial (RSA<0.2), zinc ligand,
   interface, secondary structure, mutant class.
3. Spatial structure of residuals: mean residual per residue and its correlation with the mean residual of the 8 A
   neighbours (does message passing have residual spatial signal left to exploit?).
"""
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
G = ROOT / "18_Residue_Graph_Construction/out"


def main():
    T = pd.read_csv(ROOT / "data/processed/model_table_brca1.tsv", sep="\t")
    T = T[T.struct_available].reset_index(drop=True)
    feats = [c for c in T.columns if c.split("_")[0] in ("sub", "evo", "st", "nb", "an")]
    log = json.load(open(ROOT / "results/logs/fold_P0__lgbm_struct__struct.json"))
    X = T[feats].to_numpy(float); y = T.score.to_numpy()
    contrib = np.zeros((len(T), len(feats)))
    for k in sorted(T.fold_P0.unique()):
        te = (T.fold_P0 == k).to_numpy(); p = log["folds"][str(int(k))]["chosen"]
        m = lgb.LGBMRegressor(num_leaves=p["num_leaves"], n_estimators=p["n_estimators"], learning_rate=0.03, min_child_samples=20, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1).fit(X[~te], y[~te])
        contrib[te] = m.predict(X[te], pred_contrib=True)[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(0), index=feats).sort_values(ascending=False)
    grp = imp.groupby(lambda c: c.split("_")[0]).sum().sort_values(ascending=False)
    imp.to_csv(OUT / "lgbm_struct_oof_shap_importance.tsv", sep="\t", header=["mean_abs_contribution"])
    # error analysis
    P = {}
    for name in ("gnn_C_struct_annot", "gnn_C_nograph", "lgbm_seqevo", "zs_alphamissense"):
        P[name] = pd.read_csv(ROOT / f"results/predictions/fold_P0/{name}__struct.tsv", sep="\t").pred.to_numpy()
    strata = {"region": T.region, "buried": np.where(T.st_rsa < 0.2, "buried", "exposed"), "zinc_ligand": T.an_zn_ligand.astype(int).astype(str),
              "interface": T.an_interface.astype(int).astype(str), "mt_class": T.mt_class}
    rows = []
    for sname, s in strata.items():
        s = pd.Series(np.asarray(s))
        for lev in s.dropna().unique():
            m = (s == lev).to_numpy()
            if m.sum() < 20: continue
            r = dict(stratum=sname, level=lev, n=int(m.sum()))
            for name, p in P.items():
                r[f"spearman_{name}"] = float(spearmanr(p[m], y[m])[0])
            r["delta_C_minus_nograph"] = r["spearman_gnn_C_struct_annot"] - r["spearman_gnn_C_nograph"]
            rows.append(r)
    E = pd.DataFrame(rows); E.to_csv(OUT / "error_by_stratum.tsv", sep="\t", index=False)
    # residual spatial autocorrelation (Model C-nograph residuals)
    res = {}
    for name in ("gnn_C_nograph", "gnn_C_struct_annot", "lgbm_seqevo"):
        T["_r"] = y - P[name]
        out = []
        for pdb in ("1jm7", "1t29"):
            n = pd.read_csv(G / pdb / "nodes.tsv", sep="\t"); d = np.load(G / pdb / "dist_heavy.npy")
            sub = T[T.structure == pdb]; rm = sub.groupby("node_idx")._r.mean()
            idx = rm.index.to_numpy().astype(int)
            for i in idx:
                nb = [j for j in idx if j != i and d[i, j] <= 8.0]
                if nb: out.append((rm.loc[i], rm.loc[nb].mean()))
        o = np.array(out); res[name] = dict(n_residues=len(o), spearman_residual_vs_neighbour_residual=float(spearmanr(o[:, 0], o[:, 1])[0]))
    summary = {"group_importance": grp.round(4).to_dict(), "top20": imp.head(20).round(4).to_dict(), "residual_spatial_autocorrelation": res}
    json.dump(summary, open(OUT / "explainability_summary.json", "w"), indent=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    imp.head(20)[::-1].plot.barh(ax=ax, color="#4C72B0")
    ax.set(xlabel="Mean |TreeSHAP contribution| (out-of-fold)", title="LightGBM with structure features: top 20 inputs (P0)")
    fig.tight_layout()
    for ext in ("png", "svg"): fig.savefig(ROOT / f"figures/stage8_shap_importance.{ext}", dpi=200)
    print(json.dumps(summary, indent=1)); print(E.round(3).to_string())


if __name__ == "__main__":
    main()
