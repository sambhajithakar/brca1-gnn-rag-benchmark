"""v3.2 post hoc external-validation reconciliation from saved P0 out-of-fold predictions (no training).

E1/E3: paired residue-cluster bootstrap differences versus AlphaMissense (seed 20261025, 2,000 resamples),
the same-subset difference against Findlay labels, and the difference-in-differences.
E4: pooled AUROC over the qualifying Lyra tracks only (n >= 20 calls, both classes), on a common call set, and mean per-track AUROC.
Writes only to 70_v3_2/verification/out/.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
SEED, B = 20261025, 2000
T = pd.read_csv(ROOT / "data/processed/model_table_brca1.tsv", sep="\t").set_index("hgvs_p")
D0 = ROOT / "results/predictions/fold_P0"
names = {"C": D0 / "gnn_C_struct_annot__struct.tsv", "nograph": D0 / "gnn_C_nograph__struct.tsv",
         "depthctl": ROOT / "60_final_stage/depth_control/results/predictions/fold_P0/gnn_C_struct_annot_depthctl__struct.tsv",
         "lgbm_struct": D0 / "lgbm_struct__struct.tsv", "lgbm_seqevo": D0 / "lgbm_seqevo__struct.tsv",
         "D": D0 / "gnn_D_evidence__struct.tsv", "AM": D0 / "zs_alphamissense__struct.tsv"}
pr = {k: pd.read_csv(v, sep="\t").set_index("hgvs_p").pred for k, v in names.items()}


def sp(a, b):
    ra = rankdata(a); rb = rankdata(b); ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


res = {}
for key, col in (("E1_dace_hap1", "dace_hap1_score"), ("E3_hmec", "hmec_score")):
    S = T[T[col].notna() & T.struct_available]
    y = S[col].to_numpy(); f = S.score.to_numpy(); pos = S.aa_pos.to_numpy(); u = np.unique(pos); g = {p: np.flatnonzero(pos == p) for p in u}
    X = {k: v.reindex(S.index).to_numpy() for k, v in pr.items()}
    assert not any(np.isnan(x).any() for x in X.values())
    rng = np.random.default_rng(SEED); BX = {k: [] for k in X}; BF = {k: [] for k in X}
    for _ in range(B):
        i = np.concatenate([g[p] for p in rng.choice(u, len(u))])
        a = sp(X["AM"][i], y[i]); af = sp(X["AM"][i], f[i])
        for k in X:
            BX[k].append(sp(X[k][i], y[i]) - a); BF[k].append(sp(X[k][i], f[i]) - af)
    R = {"n": int(len(S)), "n_res": int(len(u)), "findlay_vs_external": sp(f, y)}
    pv = lambda d: float(min(1, max(1 / B, 2 * min((d <= 0).mean(), (d >= 0).mean()))))
    for k in X:
        if k == "AM":
            R[k] = {"rho_ext": sp(X[k], y), "rho_findlay_same_subset": sp(X[k], f)}; continue
        dx = np.array(BX[k]); df = np.array(BF[k])
        R[k] = {"rho_ext": sp(X[k], y), "rho_findlay_same_subset": sp(X[k], f),
                "minus_AM_ext": [sp(X[k], y) - sp(X["AM"], y), *np.percentile(dx, [2.5, 97.5]).tolist(), pv(dx)],
                "minus_AM_findlay_same_subset": [sp(X[k], f) - sp(X["AM"], f), *np.percentile(df, [2.5, 97.5]).tolist(), pv(df)],
                "diff_in_diff": [(sp(X[k], f) - sp(X["AM"], f)) - (sp(X[k], y) - sp(X["AM"], y)), *np.percentile(df - dx, [2.5, 97.5]).tolist(), pv(df - dx)]}
    res[key] = R

# E4
L = pd.read_csv(ROOT / "data/processed/brca1_lyra2026_track_calls_long.tsv", sep="\t")
L = L[~L.track.isin({"T127", "T159"}) & L.call.isin([0, 2])].copy()
L["hgvs_p"] = "p." + L.aa_ref + L.aa_pos.astype(int).astype(str) + L.aa_alt
P4 = {"findlay": T.score, **pr}
d = L.copy()
for k, v in P4.items():
    d[k] = v.reindex(d.hgvs_p).to_numpy()
all_calls = {"n_calls_any": int(len(L)), "n_tracks_any": int(L.track.nunique())}
d = d.dropna(subset=list(P4))  # common call set: struct-mapped variants with every prediction
d["lof"] = (d.call == 2).astype(int)
qual = [t for t, gg in d.groupby("track") if len(gg) >= 20 and gg.lof.nunique() == 2]
q = d[d.track.isin(qual)]
E4 = {**all_calls, "n_calls_common_all_tracks": int(len(d)), "n_tracks_common": int(d.track.nunique()),
      "n_qualifying_tracks": len(qual), "n_calls_qualifying": int(len(q)), "models": {}}
rng = np.random.default_rng(SEED); tr = {t: gg for t, gg in q.groupby("track")}; keys = list(tr)
boots = [rng.choice(keys, len(keys)) for _ in range(1000)]
for k in P4:
    pooled_all = float(roc_auc_score(d.lof, -d[k])); pooled_q = float(roc_auc_score(q.lof, -q[k]))
    per = [roc_auc_score(gg.lof, -gg[k]) for _, gg in q.groupby("track")]
    bs = []
    for s in boots:
        ss = pd.concat([tr[t] for t in s])
        if ss.lof.nunique() == 2: bs.append(roc_auc_score(ss.lof, -ss[k]))
    E4["models"][k] = {"pooled_all_tracks_common": pooled_all, "pooled_qualifying": pooled_q,
                       "pooled_qualifying_ci": np.percentile(bs, [2.5, 97.5]).tolist(), "mean_per_track": float(np.mean(per))}
res["E4_lyra"] = E4
json.dump(res, open(OUT / "external_paired_v3_2.json", "w"), indent=1)
for key in ("E1_dace_hap1", "E3_hmec"):
    R = res[key]; print(key, R["n"], R["n_res"], "Findlay-ext %.3f" % R["findlay_vs_external"], "AM ext %.3f" % R["AM"]["rho_ext"])
    for k, v in R.items():
        if isinstance(v, dict) and "minus_AM_ext" in v:
            print("  %-12s ext %.3f | -AM ext %+.3f [%.3f, %.3f] p=%.3f | -AM Findlay %+.3f [%.3f, %.3f] | DiD %+.3f [%.3f, %.3f] p=%.3f" % (
                k, v["rho_ext"], *v["minus_AM_ext"], *v["minus_AM_findlay_same_subset"][:3], *v["diff_in_diff"]))
print({k: v for k, v in E4.items() if k != "models"})
for k, v in E4["models"].items():
    print("  %-12s all %.3f | qual %.3f [%.3f, %.3f] | mean per-track %.3f" % (k, v["pooled_all_tracks_common"], v["pooled_qualifying"], *v["pooled_qualifying_ci"], v["mean_per_track"]))
