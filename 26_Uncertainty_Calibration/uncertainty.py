"""Stage 8: split-conformal intervals, P(LoF), selective prediction and discordance (section 7 of the analysis plan).

For each outer fold of partition P0 (D1-struct): 20% of the training residues (seeded) form a calibration set; the
model is refit on the remaining 80% (proper training set) with 5 seeds, using the hyperparameters chosen by inner CV
in the primary run (results/logs). Nonconformity = |y - mu| (standard) and |y - mu| / (sd + eps) (normalised).
Models: gnn_C_struct_annot (Model C) and lgbm_seqevo (best supervised non-graph baseline).

Outputs: 26_Uncertainty_Calibration/out/{model}_intervals.tsv, uncertainty_summary.json, figures.
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu, norm, spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(ROOT / "20_GNN_Model_Source"))
import run_experiments as RX  # noqa: E402
from gnn import train_model  # noqa: E402

OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
FIG = ROOT / "figures"
LOF, NORMAL = -1.328, -0.748
ALPHAS = (0.2, 0.1)
PART = "fold_P0"
EPS = 1e-3


def calib_split(T, tr, k):
    rng = np.random.default_rng(777 + int(k))
    res = T.aa_pos.to_numpy()[tr]; u = np.unique(res)
    cal_res = set(rng.choice(u, size=int(round(0.2 * len(u))), replace=False))
    m = np.array([r in cal_res for r in res])
    return tr[~m], tr[m]


def fit_gnn(T, graphs, var_cols, proper, L):
    y = torch.as_tensor(T.score.to_numpy(), dtype=torch.float32)
    vs = torch.as_tensor(T.structure.map({s: i for i, s in enumerate(RX.STRUCTS)}).to_numpy(), dtype=torch.long)
    vn = torch.as_tensor(T.node_idx.to_numpy(), dtype=torch.long)
    Xv = T[var_cols].to_numpy(float)
    mu = np.nanmean(Xv[proper], 0); sd = np.nanstd(Xv[proper], 0); sd[sd == 0] = 1
    vx = torch.as_tensor(np.nan_to_num((Xv - mu) / sd), dtype=torch.float32)
    P = []
    for s in RX.SEEDS:
        a, v = RX.inner_val_split(T, proper, s)
        m, _, _ = train_model(graphs, vs, vn, vx, y, a, v, L, s, graphs[0][0].shape[1])
        with torch.no_grad(): P.append(m(graphs, vs, vn, vx).numpy())
    P = np.stack(P); return P.mean(0), P.std(0)


def fit_lgbm(T, feats, proper, p):
    X = T[feats].to_numpy(float); y = T.score.to_numpy(); P = []
    for s in RX.SEEDS:
        m = lgb.LGBMRegressor(num_leaves=p["num_leaves"], n_estimators=p["n_estimators"], learning_rate=0.03, min_child_samples=20,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=s, verbose=-1, n_jobs=4).fit(X[proper], y[proper])
        P.append(m.predict(X))
    P = np.stack(P); return P.mean(0), P.std(0)


def run_model(T, name):
    log = json.load(open(ROOT / f"results/logs/{PART}__{name}__struct.json"))
    seqevo = RX.cols(T, ["sub", "evo"])
    graphs = RX.load_graphs("heavy8_with_partner", with_annot=True) if name.startswith("gnn") else None
    rows = []
    for k in sorted(T[PART].unique()):
        te = np.nonzero((T[PART] == k).to_numpy())[0]; tr = np.nonzero((T[PART] != k).to_numpy())[0]
        proper, cal = calib_split(T, tr, k)
        if graphs is not None:
            mu, sd = fit_gnn(T, graphs, seqevo + RX.mpnn_var_cols(), proper, log["folds"][str(int(k))]["n_layers"])
        else:
            mu, sd = fit_lgbm(T, seqevo, proper, log["folds"][str(int(k))]["chosen"])
        y = T.score.to_numpy()
        r_cal = np.abs(y[cal] - mu[cal]); rn_cal = r_cal / (sd[cal] + EPS)
        cal_mse = float(np.mean((y[cal] - mu[cal]) ** 2))
        D = T.iloc[te][["aa_pos", "aa_ref", "aa_alt", "hgvs_p", "score", "region", "ext_n_lof", "ext_n_normal"]].copy()
        D["fold"] = int(k); D["mu"] = mu[te]; D["sd_ens"] = sd[te]
        for a in ALPHAS:
            n = len(cal); lev = min(1.0, np.ceil((n + 1) * (1 - a)) / n)
            q = np.quantile(r_cal, lev, method="higher"); qn = np.quantile(rn_cal, lev, method="higher")
            c = int(round(100 * (1 - a)))
            D[f"lo{c}"] = mu[te] - q; D[f"hi{c}"] = mu[te] + q
            D[f"nlo{c}"] = mu[te] - qn * (sd[te] + EPS); D[f"nhi{c}"] = mu[te] + qn * (sd[te] + EPS)
        sig = np.sqrt(sd[te] ** 2 + cal_mse)
        D["p_lof"] = norm.cdf((LOF - mu[te]) / sig); D["sigma"] = sig
        rows.append(D)
    D = pd.concat(rows).sort_index()
    D.to_csv(OUT / f"{name}_intervals.tsv", sep="\t", index=False)
    return D


def ece(p, yb, bins=10):
    e = 0.0; edges = np.linspace(0, 1, bins + 1); tab = []
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= 1))
        if m.sum() == 0: continue
        e += m.mean() * abs(p[m].mean() - yb[m].mean()); tab.append((float(p[m].mean()), float(yb[m].mean()), int(m.sum())))
    return float(e), tab


def risk_coverage(D, width_col):
    order = np.argsort(D[width_col].to_numpy())  # keep narrowest first
    y = D.score.to_numpy()[order]; mu = D.mu.to_numpy()[order]
    binm = (y < LOF) | (y > NORMAL)
    cov, rmse, mis = [], [], []
    for f in np.linspace(0.1, 1.0, 19):
        n = int(round(f * len(y))); yy, mm, bb = y[:n], mu[:n], binm[:n]
        cov.append(f); rmse.append(float(np.sqrt(np.mean((yy - mm) ** 2))))
        mis.append(float(np.mean((yy[bb] < LOF) != (mm[bb] < (LOF + NORMAL) / 2))) if bb.any() else np.nan)
    return dict(coverage=cov, rmse=rmse, lof_misclass=mis, aurc_rmse=float(np.trapezoid(rmse, cov) / 0.9),
                aurc_mis=float(np.trapezoid(mis, cov) / 0.9))


def summarise(D, name):
    y = D.score.to_numpy(); S = {"model": name, "n": int(len(D)), "spearman_mu": float(spearmanr(D.mu, y)[0])}
    for c in (80, 90):
        cov = (y >= D[f"lo{c}"]) & (y <= D[f"hi{c}"]); ncov = (y >= D[f"nlo{c}"]) & (y <= D[f"nhi{c}"])
        S[f"coverage{c}"] = float(cov.mean()); S[f"width{c}"] = float((D[f"hi{c}"] - D[f"lo{c}"]).mean())
        S[f"norm_coverage{c}"] = float(ncov.mean()); S[f"norm_width{c}_median"] = float((D[f"nhi{c}"] - D[f"nlo{c}"]).median())
        S[f"coverage{c}_by_fold"] = {str(k): float(v) for k, v in cov.groupby(D.fold).mean().items()}
        S[f"coverage{c}_by_region"] = {str(k): float(v) for k, v in cov.groupby(D.region).mean().items()}
        S[f"norm_coverage{c}_by_region"] = {str(k): float(v) for k, v in ncov.groupby(D.region).mean().items()}
    m = (y < LOF) | (y > NORMAL); yb = (y[m] < LOF).astype(float); p = D.p_lof.to_numpy()[m]
    S["brier"] = float(np.mean((p - yb) ** 2)); S["brier_ref_prevalence"] = float(np.mean((yb.mean() - yb) ** 2))
    S["ece"], S["reliability"] = ece(p, yb)
    S["sd_vs_abs_error_spearman"] = float(spearmanr(D.sd_ens, np.abs(y - D.mu))[0])
    S["rc_standard_sd"] = risk_coverage(D, "sd_ens")
    D = D.assign(nwidth=D.nhi90 - D.nlo90)
    S["rc_normalised_width"] = risk_coverage(D, "nwidth")
    # discordance: Findlay binary class vs majority of independent tracks (ties and intermediates excluded)
    f_lof = np.where(y < LOF, 1, np.where(y > NORMAL, 0, -1))
    e_lof = np.where(D.ext_n_lof > D.ext_n_normal, 1, np.where(D.ext_n_normal > D.ext_n_lof, 0, -1))
    ok = (f_lof >= 0) & (e_lof >= 0) & (D.ext_n_lof + D.ext_n_normal > 0).to_numpy()
    disc = ok & (f_lof != e_lof); agree = ok & (f_lof == e_lof)
    w = D.nwidth.to_numpy(); sd = D.sd_ens.to_numpy()
    S["discordance"] = dict(n_discordant=int(disc.sum()), n_concordant=int(agree.sum()),
                            median_nwidth_disc=float(np.median(w[disc])) if disc.any() else None,
                            median_nwidth_conc=float(np.median(w[agree])) if agree.any() else None,
                            mwu_p_nwidth=float(mannwhitneyu(w[disc], w[agree], alternative="greater").pvalue) if disc.sum() > 2 else None,
                            mwu_p_sd=float(mannwhitneyu(sd[disc], sd[agree], alternative="greater").pvalue) if disc.sum() > 2 else None,
                            abs_err_disc=float(np.median(np.abs(y - D.mu)[disc])) if disc.any() else None,
                            abs_err_conc=float(np.median(np.abs(y - D.mu)[agree])) if agree.any() else None)
    thr = np.quantile(w, 0.8)  # abstain on widest 20%
    ab = w >= thr
    S["discordance"]["frac_discordant_among_abstained"] = float(disc[ab & ok].sum() / max(1, (ab & ok).sum()))
    S["discordance"]["frac_discordant_among_retained"] = float(disc[~ab & ok].sum() / max(1, (~ab & ok).sum()))
    # high-confidence errors: narrowest 25% normalised 90% intervals that miss the observed score
    narrow = w <= np.quantile(w, 0.25); miss = (y < D.nlo90) | (y > D.nhi90)
    H = D[narrow & miss.to_numpy()].assign(abs_err=lambda x: np.abs(x.score - x.mu)).sort_values("abs_err", ascending=False)
    H.to_csv(OUT / f"{name}_high_confidence_errors.tsv", sep="\t", index=False)
    S["n_high_conf_errors"] = int(len(H))
    S["discordant_variants"] = D[disc].hgvs_p.tolist()
    return S


def plots(D, S, name):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    rel = np.array(S["reliability"]); ax[0].plot([0, 1], [0, 1], "--", c="grey"); ax[0].plot(rel[:, 0], rel[:, 1], "o-")
    ax[0].set(xlabel="Predicted P(LoF)", ylabel="Observed LoF fraction", title=f"Reliability (ECE {S['ece']:.3f})")
    for key, lab in (("rc_standard_sd", "ensemble SD"), ("rc_normalised_width", "normalised interval width")):
        ax[1].plot(S[key]["coverage"], S[key]["rmse"], label=lab); ax[2].plot(S[key]["coverage"], S[key]["lof_misclass"], label=lab)
    ax[1].set(xlabel="Coverage (retained fraction)", ylabel="RMSE", title="Risk-coverage (RMSE)"); ax[1].legend()
    ax[2].set(xlabel="Coverage", ylabel="LoF misclassification", title="Risk-coverage (LoF calls)")
    fig.suptitle(f"{name}: uncertainty on held-out residues (partition P0)"); fig.tight_layout()
    for ext in ("png", "svg"): fig.savefig(FIG / f"stage8_uncertainty_{name}.{ext}", dpi=200)
    plt.close(fig)


def main():
    T = pd.read_csv(ROOT / "data/processed/model_table_brca1.tsv", sep="\t")
    T = T[T.struct_available].reset_index(drop=True); T["node_idx"] = T.node_idx.astype(int)
    out = {}
    for name in ("lgbm_seqevo", "gnn_C_struct_annot"):
        D = run_model(T, name); S = summarise(D, name); plots(D, S, name); out[name] = S
        print(name, {k: v for k, v in S.items() if not isinstance(v, (dict, list))}, S["discordance"])
    json.dump(out, open(OUT / "uncertainty_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
