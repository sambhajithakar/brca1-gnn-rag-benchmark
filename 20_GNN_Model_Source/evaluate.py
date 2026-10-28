"""Evaluation of saved out-of-fold predictions (section 5 of the analysis plan).

Metrics per model: Spearman, Pearson, RMSE, MAE (supervised models only for RMSE/MAE), AUROC and AUPRC for
protein-level loss of function (score < -1.328 vs > -0.748), per-region Spearman, and a residue-cluster
bootstrap 95% CI for Spearman (2,000 resamples). Paired comparisons use the same resamples.

Usage: python evaluate.py --partition fold_P0 --subset struct
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
B = 2000
LOF, NORMAL = -1.328, -0.748


def fast_spearman(a, b):
    ra = rankdata(a); rb = rankdata(b)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def boot_indices(pos, seed=20261025, b=B):
    rng = np.random.default_rng(seed)
    u = np.unique(pos)
    groups = {p: np.nonzero(pos == p)[0] for p in u}
    for _ in range(b):
        s = rng.choice(u, size=len(u), replace=True)
        yield np.concatenate([groups[p] for p in s])


def load(partition, subset):
    d = RES / "predictions" / partition
    preds = {}
    base = None
    for f in sorted(d.glob(f"*__{subset}.tsv")):
        t = pd.read_csv(f, sep="\t")
        name = f.name[: -len(f"__{subset}.tsv")]
        if base is None:
            base = t[["aa_pos", "aa_ref", "aa_alt", "hgvs_p", "score"]].copy()
        assert (t.hgvs_p.values == base.hgvs_p.values).all()
        preds[name] = t.pred.to_numpy()
    return base, preds


def metrics(base, preds, boots):
    y = base.score.to_numpy(); pos = base.aa_pos.to_numpy()
    region = np.where(pos <= 110, "RING", "BRCT")
    lof = y < LOF; nor = y > NORMAL; mask_bin = lof | nor
    rows = []
    for name, p in preds.items():
        ok = ~np.isnan(p)
        r = dict(model=name, n=int(ok.sum()), spearman=spearmanr(p[ok], y[ok])[0], pearson=pearsonr(p[ok], y[ok])[0])
        if not name.startswith("zs_"):
            r["rmse"] = float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2))); r["mae"] = float(np.mean(np.abs(p[ok] - y[ok])))
        mb = mask_bin & ok
        r["auroc_lof"] = roc_auc_score(lof[mb], -p[mb]); r["auprc_lof"] = average_precision_score(lof[mb], -p[mb])
        r["lof_prevalence"] = float(lof[mb].mean())
        for reg in ("RING", "BRCT"):
            m = (region == reg) & ok
            r[f"spearman_{reg}"] = spearmanr(p[m], y[m])[0] if m.sum() > 3 else np.nan
        bs = np.array([fast_spearman(p[i], y[i]) for i in boots]) if ok.all() else np.full(len(boots), np.nan)
        r["spearman_ci_lo"], r["spearman_ci_hi"] = np.nanpercentile(bs, [2.5, 97.5])
        rows.append(r)
    return pd.DataFrame(rows).sort_values("spearman", ascending=False)


def paired(base, preds, boots, a, b):
    y = base.score.to_numpy(); pa = preds[a]; pb = preds[b]
    d0 = fast_spearman(pa, y) - fast_spearman(pb, y)
    ds = np.array([fast_spearman(pa[i], y[i]) - fast_spearman(pb[i], y[i]) for i in boots])
    p = 2 * min((ds <= 0).mean(), (ds >= 0).mean())
    return dict(a=a, b=b, delta_spearman=float(d0), ci_lo=float(np.percentile(ds, 2.5)), ci_hi=float(np.percentile(ds, 97.5)),
                p_boot=float(min(1.0, max(p, 1.0 / B))))


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i])); adj[i] = run
    return adj


def main(partition, subset):
    base, preds = load(partition, subset)
    boots = list(boot_indices(base.aa_pos.to_numpy()))
    M = metrics(base, preds, boots)
    M.to_csv(RES / f"metrics__{partition}__{subset}.tsv", sep="\t", index=False)
    out = {"partition": partition, "subset": subset, "n_variants": int(len(base)), "n_residues": int(base.aa_pos.nunique())}
    if subset == "struct" and "gnn_C_struct_annot" in preds and "gnn_C_nograph" in preds:
        out["H1_primary"] = paired(base, preds, boots, "gnn_C_struct_annot", "gnn_C_nograph")
        sup_nongraph = [m for m in ["ridge_seqevo", "rf_seqevo", "lgbm_seqevo", "mlp_seqevo"] if m in preds]
        best = max(sup_nongraph, key=lambda m: M.set_index("model").loc[m, "spearman"])
        fam = [("S-a", "gnn_C_struct_annot", best), ("S-b", "gnn_C_struct_annot", "zs_alphamissense"),
               ("S-c", "gnn_B_struct", "gnn_A_seqonly"), ("S-e", "gnn_C_struct_annot", "lgbm_struct"),
               ("S-f", "gnn_C_struct_annot", "labelsmooth")]
        if "gnn_D_evidence" in preds:
            fam.append(("S-d", "gnn_D_evidence", "gnn_C_struct_annot"))
        fam = [f for f in fam if f[1] in preds and f[2] in preds]
        res = [dict(id=i, **paired(base, preds, boots, a, b)) for i, a, b in fam]
        adj = holm(np.array([r["p_boot"] for r in res]))
        for r, q in zip(res, adj): r["p_holm"] = float(q)
        out["secondary"] = res
        out["best_supervised_nongraph"] = best
    json.dump(out, open(RES / f"tests__{partition}__{subset}.json", "w"), indent=1)
    pd.set_option("display.width", 220)
    print(M[["model", "n", "spearman", "spearman_ci_lo", "spearman_ci_hi", "pearson", "auroc_lof", "spearman_RING", "spearman_BRCT"]].round(3).to_string(index=False))
    print(json.dumps({k: v for k, v in out.items() if k not in ("partition", "subset")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--partition", default="fold_P0"); ap.add_argument("--subset", default="struct")
    a = ap.parse_args(); main(a.partition, a.subset)
