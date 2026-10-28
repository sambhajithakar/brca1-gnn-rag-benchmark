"""v3.2 statistical reconciliation (read-only on saved predictions; no model is trained).

Part A uses the project's own evaluate.py functions, imported unchanged.
Part B is an independent re-implementation (scipy spearmanr, own residue bootstrap) with three new seeds.
Writes only to 70_v3_2/verification/out/.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "20_GNN_Model_Source"))
from evaluate import boot_indices, paired, fast_spearman, holm  # noqa: E402

PARTS = ["fold_P0", "fold_P1", "fold_P2", "fold_LODO", "fold_LOEO", "fold_random"]
DC = ROOT / "60_final_stage/depth_control"
rd = lambda p: pd.read_csv(p, sep="\t")


def load(part):
    C = rd(ROOT / f"results/predictions/{part}/gnn_C_struct_annot__struct.tsv")
    N = rd(ROOT / f"results/predictions/{part}/gnn_C_nograph__struct.tsv")
    D = rd(DC / f"results/predictions/{part}/gnn_C_struct_annot_depthctl__struct.tsv")
    for t in (N, D):
        assert (t.hgvs_p.values == C.hgvs_p.values).all() and np.allclose(t.score.values, C.score.values)
    P = {"C": C.pred.to_numpy(), "nograph": N.pred.to_numpy(), "depthctl": D.pred.to_numpy()}
    extra = {"seqonly": "gnn_C_struct_annot_sequence_only", "AM": "zs_alphamissense"}
    for k, f in extra.items():
        fp = ROOT / f"results/predictions/{part}/{f}__struct.tsv"
        if fp.exists():
            t = rd(fp); assert (t.hgvs_p.values == C.hgvs_p.values).all(); P[k] = t.pred.to_numpy()
    for k, v in P.items():
        assert not np.isnan(v).any(), (part, k)
    return C, P


def indep_boot(pos, y, pa, pb, seed, B=2000):
    rng = np.random.default_rng(seed); u = np.unique(pos); idx = {p: np.flatnonzero(pos == p) for p in u}
    d = np.empty(B)
    for i in range(B):
        s = np.concatenate([idx[p] for p in rng.choice(u, len(u))])
        d[i] = spearmanr(pa[s], y[s])[0] - spearmanr(pb[s], y[s])[0]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(min(1, 2 * min((d <= 0).mean(), (d >= 0).mean())))


R = {"partA_project_code": {}, "partB_independent": {}, "checks": []}
saved = json.load(open(DC / "depthctl_results.json"))
for part in PARTS:
    C, P = load(part); y = C.score.to_numpy(); pos = C.aa_pos.to_numpy()
    boots = list(boot_indices(pos))  # default seed 20261025, B = 2000, as in the primary analysis
    a = {"n": int(len(y)), "n_res": int(len(np.unique(pos))),
         "spearman": {k: fast_spearman(v, y) for k, v in P.items()},
         "C_minus_nograph": paired(C, P, boots, "C", "nograph"),
         "C_minus_depthctl": paired(C, P, boots, "C", "depthctl"),
         "depthctl_minus_nograph": paired(C, P, boots, "depthctl", "nograph")}
    if "seqonly" in P:
        a["seqonly_minus_depthctl"] = paired(C, P, boots, "seqonly", "depthctl")
        a["C_minus_seqonly"] = paired(C, P, boots, "C", "seqonly")
        a["seqonly_minus_nograph"] = paired(C, P, boots, "seqonly", "nograph")
    # per-fold deltas (point estimates only)
    fold_col = None
    T = rd(ROOT / "data/processed/model_table_brca1.tsv")
    T = T[T.struct_available].reset_index(drop=True)
    assert (T.hgvs_p.values == C.hgvs_p.values).all()
    a["per_fold"] = {}
    for k in sorted(T[part].unique()):
        m = (T[part] == k).to_numpy()
        a["per_fold"][str(k)] = {"n": int(m.sum()), "C_minus_depthctl": fast_spearman(P["C"][m], y[m]) - fast_spearman(P["depthctl"][m], y[m]),
                                 "C_minus_nograph": fast_spearman(P["C"][m], y[m]) - fast_spearman(P["nograph"][m], y[m])}
    # region split
    reg = np.where(pos <= 110, "RING", "BRCT"); a["by_region"] = {}
    for r in ("RING", "BRCT"):
        m = reg == r
        a["by_region"][r] = {k: fast_spearman(v[m], y[m]) for k, v in P.items()}
    R["partA_project_code"][part] = a
    # compare with saved depthctl_results.json
    s = saved[part]
    for key, sk in [("C_minus_depthctl", "C_minus_depthctl"), ("depthctl_minus_nograph", "depthctl_minus_nograph"), ("C_minus_nograph", "C_minus_nograph_H1")]:
        for f in ("delta_spearman", "ci_lo", "ci_hi", "p_boot"):
            ok = abs(a[key][f] - s[sk][f]) < 1e-12
            R["checks"].append(dict(part=part, item=f"{key}.{f}", recomputed=a[key][f], saved=s[sk][f], match=ok))
    # independent implementation, three new bootstrap seeds
    b = {"spearman_scipy": {k: float(spearmanr(v, y)[0]) for k, v in P.items()}}
    for nm, (pa, pb) in {"C_minus_depthctl": ("C", "depthctl"), "depthctl_minus_nograph": ("depthctl", "nograph"), "C_minus_nograph": ("C", "nograph")}.items():
        b[nm] = {"delta": b["spearman_scipy"][pa] - b["spearman_scipy"][pb],
                 "seeds": {sd: indep_boot(pos, y, P[pa], P[pb], sd) for sd in (1, 2, 3)}}
    R["partB_independent"][part] = b
    print(part, "done", flush=True)

# P0 primary + Holm family from saved predictions, against saved tests file
d = ROOT / "results/predictions/fold_P0"; base = None; PP = {}
for f in sorted(d.glob("*__struct.tsv")):
    t = rd(f); base = t[["aa_pos", "hgvs_p", "score"]] if base is None else base; PP[f.name[:-len("__struct.tsv")]] = t.pred.to_numpy()
bo = list(boot_indices(base.aa_pos.to_numpy()))
st = json.load(open(ROOT / "results/tests__fold_P0__struct.json"))
h1 = paired(base, PP, bo, "gnn_C_struct_annot", "gnn_C_nograph")
for f in ("delta_spearman", "ci_lo", "ci_hi", "p_boot"):
    R["checks"].append(dict(part="fold_P0", item=f"H1.{f}", recomputed=h1[f], saved=st["H1_primary"][f], match=abs(h1[f] - st["H1_primary"][f]) < 1e-12))
sec = [dict(id=s["id"], **paired(base, PP, bo, s["a"], s["b"])) for s in st["secondary"]]
adj = holm(np.array([s["p_boot"] for s in sec]))
for s, q, o in zip(sec, adj, st["secondary"]):
    for f in ("delta_spearman", "ci_lo", "ci_hi", "p_boot"):
        R["checks"].append(dict(part="fold_P0", item=f"{s['id']}.{f}", recomputed=s[f], saved=o[f], match=abs(s[f] - o[f]) < 1e-12))
    R["checks"].append(dict(part="fold_P0", item=f"{s['id']}.p_holm", recomputed=float(q), saved=o["p_holm"], match=abs(q - o["p_holm"]) < 1e-12))
R["P0_secondary_recomputed"] = [dict(**s, p_holm=float(q)) for s, q in zip(sec, adj)]
# post hoc: what Holm would give if C - depthctl were added to the secondary family (reported as a sensitivity note only)
fam7 = np.array([s["p_boot"] for s in sec] + [R["partA_project_code"]["fold_P0"]["C_minus_depthctl"]["p_boot"]])
R["posthoc_holm_if_added_to_family"] = dict(zip([s["id"] for s in sec] + ["C_minus_depthctl"], holm(fam7).tolist()))
# layer choices
lay = {}
for part in PARTS:
    g = lambda p: [v["n_layers"] for v in json.load(open(p))["folds"].values()]
    lay[part] = {"C": g(ROOT / f"results/logs/{part}__gnn_C_struct_annot__struct.json"),
                 "depthctl": g(DC / f"results/logs/{part}__gnn_C_struct_annot_depthctl__struct.json"),
                 "nograph": g(ROOT / f"results/logs/{part}__gnn_C_nograph__struct.json")}
    lay[part]["folds_same_depth"] = sum(x == y for x, y in zip(lay[part]["C"], lay[part]["depthctl"]))
R["layers"] = lay
# legacy sensitivity CIs (1,000 resamples) vs 2,000-resample recomputation of H1 across partitions
sens = rd(ROOT / "results/sensitivity_tests.tsv")
leg = sens[(sens.a == "gnn_C_struct_annot") & (sens.b == "gnn_C_nograph")][["analysis", "delta_spearman", "ci_lo", "ci_hi", "p_boot"]]
R["legacy_sensitivity_H1_1000"] = leg.to_dict("records")
json.dump(R, open(OUT / "reconcile_v3_2.json", "w"), indent=1, default=float)
nm = sum(not c["match"] for c in R["checks"])
print("checks:", len(R["checks"]), "mismatches:", nm)
