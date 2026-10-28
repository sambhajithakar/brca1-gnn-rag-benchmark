"""(v3, 2026-10-26: Lyra repository path from the LYRA_REPO environment variable instead of an absolute path.) Stage 9: external and cross-assay validation (section 6 of the analysis plan). No tuning on external data.

E1  Dace 2025 HAP1 re-assay of Findlay residues: out-of-fold P0 predictions vs Dace score.
E2  Dace 2025 HAP1 residues not in the Findlay set: models trained on all Findlay variants with features available
    outside the Findlay table (substitution features + AlphaMissense; EVE/CADD etc. come from the Findlay
    supplement and are not available for new residues), LightGBM and ridge; AlphaMissense zero-shot comparator.
E3  Dace 2025 HMEC SGE: out-of-fold P0 predictions vs HMEC score.
E4  Independent Lyra 2026 BRCA1 tracks (excluding T127 = Findlay and T159 = failed QC): AUROC for LoF (2) vs normal (0)
    per track with n >= 20 and both classes, pooled AUROC with a track-stratified bootstrap 95% CI.
    Findlay score is the assay-level comparator.
BRCA2 (secondary): residue-grouped 5-fold CV on Huang 2025 binary calls; transfer to Sahu 2025 and Hu 2024; symmetric
    analysis trained on Sahu. Features: substitution + AlphaMissense + position context (residue-mean AlphaMissense
    over all substitutions, relative position, domain flags from the Lyra repository). No 3D structure.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(ROOT / "20_GNN_Model_Source"))
from features import substitution_features  # noqa: E402

P = ROOT / "data/processed"
OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
SEED = 20261025
LOF = -1.328
EXCL = {"T127", "T159"}


def oof_preds(subset):
    d = ROOT / "results/predictions/fold_P0"
    out = {}
    for f in sorted(d.glob(f"*__{subset}.tsv")):
        name = f.name[: -len(f"__{subset}.tsv")]
        if "_heavy" in name or "_ca8" in name or "_sequence_only" in name: continue
        out[name] = pd.read_csv(f, sep="\t").set_index("hgvs_p").pred
    return out


def sp(a, b):
    m = ~(np.isnan(a) | np.isnan(b)); return float(spearmanr(a[m], b[m])[0]), int(m.sum())


def boot_ci_spearman(x, y, groups, b=2000):
    rng = np.random.default_rng(SEED); u = np.unique(groups); idx = {g: np.nonzero(groups == g)[0] for g in u}; v = []
    for _ in range(b):
        s = np.concatenate([idx[g] for g in rng.choice(u, len(u))]); v.append(spearmanr(x[s], y[s])[0])
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def E1_E3(res):
    T = pd.read_csv(P / "model_table_brca1.tsv", sep="\t").set_index("hgvs_p")
    for key, col in (("E1_dace_hap1_reassay", "dace_hap1_score"), ("E3_dace_hmec", "hmec_score")):
        R = {}
        sub = T[T[col].notna()]
        R["findlay_score_vs_external"] = dict(zip(("rho", "n"), sp(sub.score.to_numpy(), sub[col].to_numpy())))
        for subset in ("struct", "all"):
            for name, p in oof_preds(subset).items():
                x = p.reindex(sub.index).to_numpy(); y = sub[col].to_numpy(); m = ~np.isnan(x)
                if m.sum() < 20: continue
                r, n = sp(x, y)
                R[f"{name}__{subset}"] = dict(rho=r, n=n, ci=boot_ci_spearman(x[m], y[m], sub.aa_pos.to_numpy()[m]))
        res[key] = R


def am_table():
    a = pd.read_csv(ROOT / "data/raw/alphamissense/AlphaMissense_aa_substitutions_BRCA1_BRCA2.tsv", sep="\t", comment="#")
    a["aa_ref"] = a.protein_variant.str[0]; a["aa_alt"] = a.protein_variant.str[-1]; a["aa_pos"] = a.protein_variant.str[1:-1].astype(int)
    return a


def E2(res):
    T = pd.read_csv(P / "model_table_brca1.tsv", sep="\t")
    D = pd.read_csv(P / "brca1_dace2025_hap1_missense_protein.tsv", sep="\t")
    D = D[~D.aa_pos.isin(T.aa_pos)].copy()
    am = am_table(); am = am[am.uniprot_id == "P38398"][["aa_pos", "aa_ref", "aa_alt", "am_pathogenicity"]]
    D = D.merge(am, on=["aa_pos", "aa_ref", "aa_alt"], how="left")
    Xtr = pd.concat([substitution_features(T), T[["evo_alphamissense"]].rename(columns={"evo_alphamissense": "am"})], axis=1)
    Xte = pd.concat([substitution_features(D), D[["am_pathogenicity"]].rename(columns={"am_pathogenicity": "am"})], axis=1)
    y = T.score.to_numpy()
    models = {"lgbm_sub_am": lgb.LGBMRegressor(num_leaves=15, n_estimators=200, learning_rate=0.03, subsample=0.8, subsample_freq=1,
                                               colsample_bytree=0.8, random_state=0, verbose=-1),
              "ridge_sub_am": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10))}
    lof = D.dace_hap1_class.eq("LoF"); neu = D.dace_hap1_class.eq("Neutral"); mb = (lof | neu).to_numpy()
    R = {"n_variants": int(len(D)), "n_residues": int(D.aa_pos.nunique()), "regions": D.SGE_region.value_counts().to_dict(),
         "n_lof": int(lof.sum()), "n_neutral": int(neu.sum()),
         "note": "hyperparameters fixed a priori (LightGBM: P0 most frequently chosen setting); no tuning on Dace data"}
    preds = {"zs_alphamissense": -D.am_pathogenicity.to_numpy()}
    for k, m in models.items():
        m.fit(Xtr.to_numpy(float), y); preds[k] = m.predict(Xte.to_numpy(float))
    for k, p in preds.items():
        r, n = sp(p, D.dace_hap1_score.to_numpy())
        R[k] = dict(rho=r, n=n, ci=boot_ci_spearman(p, D.dace_hap1_score.to_numpy(), D.aa_pos.to_numpy()),
                    auroc_lof=float(roc_auc_score(lof[mb], -p[mb])), auprc_lof=float(average_precision_score(lof[mb], -p[mb])))
    res["E2_dace_new_regions"] = R


def E4(res):
    L = pd.read_csv(P / "brca1_lyra2026_track_calls_long.tsv", sep="\t")
    L = L[~L.track.isin(EXCL) & L.call.isin([0, 2])].copy()
    L["hgvs_p"] = "p." + L.aa_ref + L.aa_pos.astype(int).astype(str) + L.aa_alt
    T = pd.read_csv(P / "model_table_brca1.tsv", sep="\t").set_index("hgvs_p")
    preds = {"findlay_score_assay_comparator": T.score}
    for subset in ("all", "struct"):
        for name, p in oof_preds(subset).items():
            preds[f"{name}__{subset}"] = p
    R = {}
    for name, p in preds.items():
        d = L.assign(pred=p.reindex(L.hgvs_p).to_numpy()).dropna(subset=["pred"])
        d["lof"] = (d.call == 2).astype(int)
        per = {}
        for t, g in d.groupby("track"):
            if len(g) >= 20 and g.lof.nunique() == 2:
                per[t] = dict(n=int(len(g)), n_lof=int(g.lof.sum()), auroc=float(roc_auc_score(g.lof, -g.pred)))
        pooled = float(roc_auc_score(d.lof, -d.pred)) if d.lof.nunique() == 2 else np.nan
        rng = np.random.default_rng(SEED); tr = {t: g for t, g in d.groupby("track")}; keys = list(tr); bs = []
        for _ in range(1000):
            s = pd.concat([tr[k] for k in rng.choice(keys, len(keys))])
            if s.lof.nunique() == 2: bs.append(roc_auc_score(s.lof, -s.pred))
        R[name] = dict(n_calls=int(len(d)), n_variants=int(d.hgvs_p.nunique()), n_tracks=int(d.track.nunique()), pooled_auroc=pooled,
                       ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], per_track=per)
    res["E4_lyra_tracks"] = R


def brca2_features(B, am):
    lyra = os.environ.get("LYRA_REPO")  # v3: path to the Lyra integration repository clone (commit 908fb37)
    if not lyra or not (Path(lyra) / "src/domains.py").is_file():
        raise SystemExit("LYRA_REPO must point to a clone of FunctionalAssaYIntegration/BRCA1-BRCA2-Large-integration")
    spec = importlib.util.spec_from_file_location("dom", str(Path(lyra) / "src/domains.py"))
    dom = importlib.util.module_from_spec(spec); spec.loader.exec_module(dom)
    feats = [f for f in dom.BRCA2_FEATURES if not str(f["feature"]).lower().startswith("interven")]
    coarse = [f for f in feats if (f["end_aa"] - f["start_aa"]) >= 15]
    X = substitution_features(B)
    X["am"] = B.am_pathogenicity.to_numpy()
    resmean = am.groupby("aa_pos").am_pathogenicity.mean()
    X["am_residue_mean"] = B.aa_pos.map(resmean).to_numpy()
    X["rel_pos"] = B.aa_pos.to_numpy() / 3418.0
    for f in coarse:
        X[f"dom_{f['feature']}"] = ((B.aa_pos >= f["start_aa"]) & (B.aa_pos <= f["end_aa"])).astype(float).to_numpy()
    X["in_ss_element"] = 0.0
    for f in feats:
        if (f["end_aa"] - f["start_aa"]) < 15:
            X.loc[(B.aa_pos >= f["start_aa"]) & (B.aa_pos <= f["end_aa"]), "in_ss_element"] = 1.0
    return X


def BRCA2(res):
    B = pd.read_csv(P / "brca2_functional_calls_protein.tsv", sep="\t")
    am = am_table(); am = am[am.uniprot_id == "P51587"]
    X = brca2_features(B, am).to_numpy(float)
    labs = {"huang": "huang2025_hap1", "sahu": "sahu2025_mesc", "hu_hdr": "hu2024_hdr"}
    R = {}
    for train in ("huang", "sahu"):
        col = labs[train]
        m = B[col].isin([0, 2]).to_numpy(); idx = np.nonzero(m)[0]; y = (B[col].to_numpy() == 2).astype(int)
        groups = B.aa_pos.to_numpy()
        oof = {k: np.full(len(B), np.nan) for k in ("lgbm", "logreg")}
        for k, (a, b) in enumerate(GroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(idx, groups=groups[idx])):
            a, b = idx[a], idx[b]
            g = lgb.LGBMClassifier(num_leaves=15, n_estimators=300, learning_rate=0.03, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                   random_state=0, verbose=-1).fit(X[a], y[a])
            oof["lgbm"][b] = g.predict_proba(X[b])[:, 1]
            lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(X[a], y[a])
            oof["logreg"][b] = lr.predict_proba(X[b])[:, 1]
        # for variants without a training-assay label, predictions from a model fit on all training-assay labels
        full = {"lgbm": lgb.LGBMClassifier(num_leaves=15, n_estimators=300, learning_rate=0.03, subsample=0.8, subsample_freq=1,
                                           colsample_bytree=0.8, random_state=0, verbose=-1).fit(X[idx], y[idx]),
                "logreg": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(X[idx], y[idx])}
        for k in oof:
            miss = np.isnan(oof[k]); oof[k][miss] = full[k].predict_proba(X[miss])[:, 1]
        oof["zs_alphamissense"] = B.am_pathogenicity.to_numpy()
        RR = {}
        for evalset, ecol in labs.items():
            if B[ecol].notna().sum() == 0: continue
            me = B[ecol].isin([0, 2]).to_numpy(); ye = (B[ecol].to_numpy()[me] == 2).astype(int)
            for k, p in oof.items():
                RR[f"{k}->{evalset}"] = dict(n=int(me.sum()), n_lof=int(ye.sum()), auroc=float(roc_auc_score(ye, p[me])),
                                             auprc=float(average_precision_score(ye, p[me])), prevalence=float(ye.mean()),
                                             note="out-of-fold (residue-grouped)" if evalset == train else
                                             "transfer: OOF where the training assay labelled the variant, else model fit on all training labels")
            # inter-assay agreement as a ceiling reference
            if evalset != train:
                both = B[col].isin([0, 2]) & B[ecol].isin([0, 2])
                if both.sum() > 10:
                    RR[f"assay_{train}_as_predictor->{evalset}"] = dict(n=int(both.sum()),
                        auroc=float(roc_auc_score(B.loc[both, ecol] == 2, B.loc[both, col] == 2)),
                        agreement=float(((B.loc[both, ecol] == 2) == (B.loc[both, col] == 2)).mean()))
        R[f"train_{train}"] = RR
    res["BRCA2_secondary"] = R


def main():
    res = {}
    E1_E3(res); E2(res); E4(res); BRCA2(res)
    json.dump(res, open(OUT / "external_validation.json", "w"), indent=1, default=float)
    for k in ("E1_dace_hap1_reassay", "E3_dace_hmec"):
        print(k, {m: (round(v["rho"], 3), v["n"]) for m, v in res[k].items()})
    print("E2", {m: (round(v["rho"], 3), round(v["auroc_lof"], 3)) for m, v in res["E2_dace_new_regions"].items() if isinstance(v, dict) and "rho" in v})
    print("E4", {m: (round(v["pooled_auroc"], 3), [round(c, 3) for c in v["ci"]], v["n_calls"], len(v["per_track"])) for m, v in res["E4_lyra_tracks"].items()})
    for t, RR in res["BRCA2_secondary"].items():
        print(t, {m: round(v["auroc"], 3) for m, v in RR.items()})


if __name__ == "__main__":
    main()
