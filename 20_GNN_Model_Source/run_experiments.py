"""Stages 5-6: baselines and graph models with residue-grouped cross-validation.

Implements sections 2, 4 of 16A_PRESPECIFIED_ANALYSIS_PLAN.md. Saves out-of-fold predictions to
results/predictions/<partition>/<model>.tsv and a JSON log per run to results/logs/.
Outer test folds are never used for model selection, early stopping or scaling.

Usage: python run_experiments.py --partition fold_P0 --group all
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from gnn import norm_adj, train_model

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "23_RAG_Implementation"))
from evidence_retriever import StructRetriever, leakage_check  # noqa: E402

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "18_Residue_Graph_Construction/out"
RES = ROOT / "results"
SEEDS = [0, 1, 2, 3, 4]
ALPH = "ACDEFGHIKLMNPQRSTVWYX"
KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5, L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)
VOL = dict(A=88.6, R=173.4, N=114.1, D=111.1, C=108.5, Q=143.8, E=138.4, G=60.1, H=153.2, I=166.7, L=166.7, K=168.6, M=162.9, F=189.9, P=112.7, S=89.0, T=116.1, W=227.8, Y=193.6, V=140.0)
CHG = dict(D=-1, E=-1, K=1, R=1)
GP = dict(A=8.1, R=10.5, N=11.6, D=13.0, C=5.5, Q=10.5, E=12.3, G=9.0, H=10.4, I=5.2, L=4.9, K=11.3, M=5.7, F=5.2, P=8.0, S=9.2, T=8.6, W=5.4, Y=6.2, V=5.9)
STRUCTS = ["1jm7", "1t29"]


def spear(a, b):
    r = spearmanr(a, b)[0]
    return -1.0 if np.isnan(r) else r


# ---------------------------------------------------------------- graph inputs
def load_graphs(graph_variant="heavy8_with_partner", with_annot=False):
    out = []
    for pdb in STRUCTS:
        nodes = pd.read_csv(G / pdb / "nodes.tsv", sep="\t")
        lp = np.load(G / pdb / "mpnn_logp.npy")[:, :20]
        dh = np.load(G / pdb / "dist_heavy.npy"); dca = np.load(G / pdb / "dist_ca.npy")
        n = len(nodes)
        ch = nodes.chain.to_numpy(dtype=object); rn = nodes.resnum.to_numpy()
        seq = (ch[:, None] == ch[None, :]) & (np.abs(rn[:, None] - rn[None, :]) == 1)
        nonseq = ~seq & ~np.eye(n, dtype=bool)
        partner = ch[:, None] != ch[None, :]
        if graph_variant.startswith("heavy"):
            c = float(graph_variant[5:graph_variant.index("_")])
            sp = nonseq & (dh < c)
        elif graph_variant.startswith("ca8"):
            sp = nonseq & (dca < 8)
        elif graph_variant == "sequence_only":
            sp = np.zeros_like(seq)
        if "no_partner" in graph_variant:
            sp = sp & ~partner
        f = []
        f.append(np.stack([(nodes.aa == a).to_numpy(float) for a in ALPH], 1))
        f.append(np.stack([nodes.aa.map(KD).fillna(0), nodes.aa.map(VOL).fillna(0) / 100, nodes.aa.map(lambda a: CHG.get(a, 0)),
                           nodes.aa.map(GP).fillna(0) / 10], 1))
        f.append(nodes[["rsa"]].to_numpy())
        f.append(np.stack([(nodes.ss3 == s).to_numpy(float) for s in "HE-"], 1))
        f.append(nodes[["n_contacts_8A"]].to_numpy() / 20.0)
        f.append(nodes[["interface"]].to_numpy(float))
        f.append(nodes[["ca_rmsf_nmr"]].fillna(0).to_numpy() / 3.0)
        f.append(np.full((n, 1), 1.0 if pdb == "1jm7" else 0.0))
        f.append(nodes[["bfactor_z"]].fillna(0).to_numpy())
        f.append(nodes[["mpnn_entropy"]].to_numpy())
        f.append(lp / 5.0)
        f.append((nodes.chain == "A").to_numpy(float)[:, None])
        if with_annot:
            for d in ["RING", "RING_flank", "BRCT1", "BRCT_linker", "BRCT2"]:
                f.append((nodes.domain == d).to_numpy(float)[:, None])
            f.append(nodes[["zn_ligand"]].to_numpy(float))
        x = torch.as_tensor(np.concatenate(f, 1), dtype=torch.float32)
        out.append((x, norm_adj(seq), norm_adj(sp)))
    return out


# ---------------------------------------------------------------- feature sets
def cols(T, groups):
    pre = {"sub": "sub_", "evo": "evo_", "st": "st_", "nb": "nb_", "an": "an_"}
    return [c for c in T.columns if any(c.startswith(pre[g]) for g in groups)]


def mpnn_var_cols():
    return ["st_mpnn_lp_wt", "st_mpnn_lp_mt", "st_mpnn_llr"]


# ---------------------------------------------------------------- CV helpers
def outer_folds(T, partition):
    return sorted(T[partition].dropna().unique())


def inner_val_split(T, train_idx, seed, frac=0.2):
    rng = np.random.default_rng(1000 + seed)
    res = T.aa_pos.to_numpy()[train_idx]
    u = np.unique(res)
    val_res = set(rng.choice(u, size=max(1, int(round(frac * len(u)))), replace=False))
    m = np.array([r in val_res for r in res])
    return train_idx[~m], train_idx[m]


def fit_sklearn(T, model_name, feats, train_idx, test_idx, extra=None):
    X = T[feats].to_numpy(float); y = T.score.to_numpy()
    if extra is not None:
        X = np.c_[X, extra]
    groups = T.aa_pos.to_numpy()
    grids = {
        "ridge": [dict(alpha=a) for a in (0.1, 1, 10, 100)],
        "rf": [dict(max_features=m, min_samples_leaf=l) for m in (0.33, 0.66) for l in (3, 10)],
        "lgbm": [dict(num_leaves=nl, n_estimators=ne) for nl in (7, 15) for ne in (200, 400, 800)],
        "mlp": [dict(alpha=a) for a in (1e-4, 1e-3)],
    }

    def make(p):
        if model_name == "ridge":
            return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=p["alpha"]))
        if model_name == "rf":
            return make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(n_estimators=500, max_features=p["max_features"], min_samples_leaf=p["min_samples_leaf"], n_jobs=4, random_state=0))
        if model_name == "lgbm":
            return lgb.LGBMRegressor(num_leaves=p["num_leaves"], n_estimators=p["n_estimators"], learning_rate=0.03, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1, n_jobs=4)
        if model_name == "mlp":
            return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), alpha=p["alpha"], early_stopping=True, max_iter=1000, random_state=0))
    grid = grids[model_name]
    if len(grid) > 1:
        gkf = GroupKFold(n_splits=4)
        scores = []
        for p in grid:
            s = []
            for itr, iva in gkf.split(train_idx, groups=groups[train_idx]):
                m = make(p).fit(X[train_idx[itr]], y[train_idx[itr]])
                s.append(spear(m.predict(X[train_idx[iva]]), y[train_idx[iva]]))
            scores.append(np.mean(s))
        best = grid[int(np.argmax(scores))]
    else:
        best, scores = grid[0], [np.nan]
    m = make(best).fit(X[train_idx], y[train_idx])
    return m.predict(X[test_idx]), {"chosen": best, "inner_scores": [float(s) for s in scores]}


def label_smoothing_features(T, train_idx, target_idx, k):
    """Mean training-fold score at the k nearest training residues (3D C-alpha distance, same structure),
    overall and for the same mutant amino-acid class. Uses only training labels."""
    out = np.full((len(target_idx), 2), np.nan)
    tr = T.iloc[train_idx]
    for pdb in STRUCTS:
        nodes = pd.read_csv(G / pdb / "nodes.tsv", sep="\t")
        dca = np.load(G / pdb / "dist_ca.npy")
        tr_s = tr[tr.structure == pdb]
        res_mean = tr_s.groupby("node_idx").score.mean()
        res_cls = tr_s.groupby(["node_idx", "mt_class"]).score.mean()
        tr_nodes = res_mean.index.to_numpy().astype(int)
        for j, ti in enumerate(target_idx):
            r = T.iloc[ti]
            if r.structure != pdb: continue
            ni = int(r.node_idx)
            cand = tr_nodes[tr_nodes != ni]
            nn_ = cand[np.argsort(dca[ni, cand])[:k]]
            out[j, 0] = res_mean.loc[nn_].mean()
            vals = [res_cls.get((n, r.mt_class), np.nan) for n in nn_]
            out[j, 1] = np.nanmean(vals) if np.any(~np.isnan(vals)) else out[j, 0]
    return out


def fit_label_smoothing(T, feats, train_idx, test_idx):
    y = T.score.to_numpy(); groups = T.aa_pos.to_numpy()
    best, best_s = None, -np.inf
    gkf = GroupKFold(n_splits=4)
    for k in (5, 10):
        s = []
        for itr, iva in gkf.split(train_idx, groups=groups[train_idx]):
            a, b = train_idx[itr], train_idx[iva]
            # training rows get features computed leave-own-residue-out from the inner-training set
            fa = label_smoothing_features(T, a, a, k); fb = label_smoothing_features(T, a, b, k)
            Xa = np.c_[T[feats].to_numpy(float)[a], fa]; Xb = np.c_[T[feats].to_numpy(float)[b], fb]
            m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10)).fit(Xa, y[a])
            s.append(spear(m.predict(Xb), y[b]))
        if np.mean(s) > best_s: best, best_s = k, np.mean(s)
    fa = label_smoothing_features(T, train_idx, train_idx, best); fb = label_smoothing_features(T, train_idx, test_idx, best)
    Xa = np.c_[T[feats].to_numpy(float)[train_idx], fa]; Xb = np.c_[T[feats].to_numpy(float)[test_idx], fb]
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10)).fit(Xa, y[train_idx])
    return m.predict(Xb), {"chosen_k": best, "inner_score": float(best_s)}


def fit_graph(T, graphs, var_cols, d_node, train_idx, test_idx, layer_grid, attention=False, seeds=SEEDS, extra=None):
    y = torch.as_tensor(T.score.to_numpy(), dtype=torch.float32)
    vs = torch.as_tensor(T.structure.map({s: i for i, s in enumerate(STRUCTS)}).to_numpy(), dtype=torch.long)
    vn = torch.as_tensor(T.node_idx.to_numpy(), dtype=torch.long)
    Xv = T[var_cols].to_numpy(float)
    if extra is not None:
        Xv = np.c_[Xv, extra]
    mu = np.nanmean(Xv[train_idx], 0); sd = np.nanstd(Xv[train_idx], 0); sd[sd == 0] = 1
    Xv = np.nan_to_num((Xv - mu) / sd)
    vx = torch.as_tensor(Xv, dtype=torch.float32)
    groups = T.aa_pos.to_numpy()
    info = {}
    if len(layer_grid) > 1:
        gkf = GroupKFold(n_splits=4); sc = []
        for L in layer_grid:
            s = []
            for itr, iva in gkf.split(train_idx, groups=groups[train_idx]):
                a = train_idx[itr]; b = train_idx[iva]
                a2, v2 = inner_val_split(T, a, 0)
                m, _, _ = train_model(graphs, vs, vn, vx, y, a2, v2, L, 0, d_node, attention)
                with torch.no_grad(): s.append(spear(m(graphs, vs, vn, vx)[b].numpy(), y[b].numpy()))
            sc.append(float(np.mean(s)))
        L = layer_grid[int(np.argmax(sc))]; info["inner_scores"] = sc
    else:
        L = layer_grid[0]
    info["n_layers"] = L; info["epochs"] = []
    preds = []
    for sd_ in seeds:
        a2, v2 = inner_val_split(T, train_idx, sd_)
        m, ep, _ = train_model(graphs, vs, vn, vx, y, a2, v2, L, sd_, d_node, attention)
        info["epochs"].append(ep)
        with torch.no_grad(): preds.append(m(graphs, vs, vn, vx)[test_idx].numpy())
    P = np.stack(preds)
    return P.mean(0), P.std(0), info


# ---------------------------------------------------------------- model registry
def run(partition, group, subset_name, models=None, graph_variant="heavy8_with_partner", tag=""):
    T = pd.read_csv(ROOT / "data/processed/model_table_brca1.tsv", sep="\t")
    if subset_name == "struct":
        T = T[T.struct_available].reset_index(drop=True)
        T["node_idx"] = T.node_idx.astype(int)
    outdir = RES / "predictions" / partition; outdir.mkdir(parents=True, exist_ok=True)
    (RES / "logs").mkdir(parents=True, exist_ok=True)
    folds = outer_folds(T, partition)
    seqevo = cols(T, ["sub", "evo"])
    registry = {}
    # zero-shot references (no fitting)
    registry["zs_alphamissense"] = ("zs", lambda: -T.evo_alphamissense)
    registry["zs_eve"] = ("zs", lambda: -T.evo_eve)
    registry["zs_cadd"] = ("zs", lambda: -T.evo_cadd)
    registry["zs_phylop"] = ("zs", lambda: -T.evo_phylop)
    if subset_name == "struct":
        registry["zs_mpnn_llr"] = ("zs", lambda: T.st_mpnn_llr)
    for mname in ["ridge", "rf", "lgbm", "mlp"]:
        registry[f"{mname}_seqevo"] = ("sk", mname, seqevo)
    registry["ridge_amonly"] = ("sk", "ridge", ["evo_alphamissense"])
    if subset_name == "struct":
        registry["lgbm_struct"] = ("sk", "lgbm", cols(T, ["sub", "evo", "st", "nb", "an"]))
        registry["labelsmooth"] = ("ls", seqevo)
        gB = lambda: load_graphs(graph_variant, with_annot=False)
        gC = lambda: load_graphs(graph_variant, with_annot=True)
        subcols = cols(T, ["sub"])
        registry["gnn_A_seqonly"] = ("gnn", None, seqevo, [0])
        registry["gnn_B_struct"] = ("gnn", gB, subcols + mpnn_var_cols(), [2, 3])
        registry["gnn_C_struct_annot"] = ("gnn", gC, seqevo + mpnn_var_cols(), [2, 3])
        registry["gnn_C_nograph"] = ("gnn", gC, seqevo + mpnn_var_cols(), [0])
        registry["gnn_C_gat"] = ("gat", gC, seqevo + mpnn_var_cols(), [2])
        # Stage 7 Scenario 1: retrieved-evidence features (fold-specific, leakage-controlled)
        registry["gnn_D_evidence"] = ("gnn", gC, seqevo + mpnn_var_cols(), [2, 3], "ev")
        registry["gnn_D_nograph"] = ("gnn", gC, seqevo + mpnn_var_cols(), [0], "ev")
        registry["lgbm_seqevo_ev"] = ("sk", "lgbm", seqevo, "ev")
    if subset_name == "all":
        registry["lgbm_seqevo_ev"] = ("sk", "lgbm", seqevo, "ev")
    if models:
        registry = {k: v for k, v in registry.items() if k in models}
    retriever = None
    for name, spec in registry.items():
        t0 = time.time()
        pred = np.full(len(T), np.nan); psd = np.full(len(T), np.nan); log = {"model": name, "partition": partition,
               "subset": subset_name, "graph_variant": graph_variant if spec[0] in ("gnn", "gat") else None, "folds": {}}
        if spec[0] == "zs":
            pred = spec[1]().to_numpy(float)
        else:
            graphs = None
            if spec[0] in ("gnn", "gat") and spec[1] is not None:
                graphs = spec[1]()
            use_ev = spec[-1] == "ev"
            if use_ev and retriever is None:
                retriever = StructRetriever(T)
            for k in folds:
                te = np.nonzero((T[partition] == k).to_numpy())[0]
                tr = np.nonzero((T[partition] != k).to_numpy())[0]
                extra = None
                if use_ev:
                    # residues that carry any test variant are never allowed (matters for splits that are not residue-grouped)
                    allowed = np.setdiff1d(T.aa_pos.iloc[tr].astype(int).unique(), T.aa_pos.iloc[te].astype(int).unique())
                    extra = retriever.features(T, allowed)  # own residue always excluded; test residues not allowed
                    leak = leakage_check(T, retriever, tr, te)
                    assert leak["test_residue_docs"] == 0 and leak["own_residue_docs"] == 0 and leak["findlay_docs"] == 0 \
                        and leak["same_variant_docs"] == 0, leak
                    log.setdefault("leakage_checks", {})[str(int(k))] = leak
                if spec[0] == "sk":
                    p, info = fit_sklearn(T, spec[1], spec[2], tr, te, extra)
                elif spec[0] == "ls":
                    p, info = fit_label_smoothing(T, spec[1], tr, te)
                else:
                    d_node = graphs[0][0].shape[1] if graphs is not None else 0
                    gr = graphs if graphs is not None else [(torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1, 1))] * 2
                    p, s, info = fit_graph(T, gr, spec[2], d_node, tr, te, spec[3], attention=(spec[0] == "gat"), extra=extra)
                    psd[te] = s
                pred[te] = p
                log["folds"][str(int(k))] = info
        out = T[["aa_pos", "aa_ref", "aa_alt", "hgvs_p", "score", partition]].copy()
        out["pred"] = pred; out["pred_sd"] = psd
        fn = f"{name}{tag}__{subset_name}.tsv"
        out.to_csv(outdir / fn, sep="\t", index=False)
        log["spearman_oof"] = spear(pred, T.score.to_numpy())
        log["seconds"] = round(time.time() - t0, 1)
        json.dump(log, open(RES / "logs" / f"{partition}__{fn[:-4]}.json", "w"), indent=1, default=str)
        print(f"{partition} {subset_name} {name}{tag}: done in {log['seconds']}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="fold_P0")
    ap.add_argument("--subset", default="struct", choices=["struct", "all"])
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--graph_variant", default="heavy8_with_partner")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    run(a.partition, "all", a.subset, a.models, a.graph_variant, a.tag)
