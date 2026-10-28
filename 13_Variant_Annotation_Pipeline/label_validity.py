"""Stage 2 functional label validity analysis.

Quantifies separation of reference classes, replicate reliability, multi-codon
consistency, cross-experiment and cross-assay agreement for the functional labels.
Outputs results/stage2_label_validity.json and figures/stage2_*.png.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
R = {}

snv = pd.read_csv(P / "brca1_findlay2018_snv_harmonised.tsv", sep="\t")
prot = pd.read_csv(P / "brca1_findlay2018_missense_protein.tsv", sep="\t")

# 1. Reference-class separation (nonsense vs synonymous) at SNV level
syn = snv[snv.consequence == "Synonymous"].score
non = snv[snv.consequence == "Nonsense"].score
R["synonymous_median_score"] = float(syn.median())
R["nonsense_median_score"] = float(non.median())
R["nonsense_vs_synonymous_auroc"] = float(roc_auc_score(np.r_[np.ones(len(non)), np.zeros(len(syn))], -np.r_[non, syn]))
R["missense_lof_fraction_snv"] = float((snv[snv.consequence == "Missense"].func_class == "LOF").mean())

# 2. Replicate reliability (SNV and protein level), Spearman-Brown for mean of two replicates
for lvl, d in [("snv_missense", snv[snv.consequence == "Missense"]), ("protein", prot)]:
    x = d[["score_r1", "score_r2"]].dropna()
    r = stats.pearsonr(x.score_r1, x.score_r2)[0]
    rho = stats.spearmanr(x.score_r1, x.score_r2)[0]
    R[f"replicate_pearson_{lvl}"] = float(r)
    R[f"replicate_spearman_{lvl}"] = float(rho)
    R[f"spearman_brown_reliability_pearson_{lvl}"] = float(2 * r / (1 + r))
    R[f"spearman_brown_reliability_spearman_{lvl}"] = float(2 * rho / (1 + rho))
    # Correlation ceiling for a perfect predictor of the true score: sqrt(reliability)
    R[f"correlation_ceiling_spearman_{lvl}"] = float(np.sqrt(2 * rho / (1 + rho)))

# 3. Multi-codon consistency (same amino-acid change from different SNVs)
mis = snv[snv.consequence == "Missense"]
grp = mis.groupby("hgvs_p").score
multi = grp.filter(lambda x: len(x) > 1)
pairs = mis[mis.hgvs_p.isin(multi.index.map(lambda i: mis.loc[i, "hgvs_p"]))]
pp = pairs.groupby("hgvs_p").score.apply(list)
a = np.array([v[0] for v in pp]); b = np.array([v[1] for v in pp])
R["multi_codon_pairs"] = int(len(pp))
R["multi_codon_pair_spearman"] = float(stats.spearmanr(a, b)[0])
cls = mis.groupby("hgvs_p").func_class.agg(lambda x: len(set(x)))
R["multi_codon_class_discordant"] = int((cls > 1).sum())

# 4. Splice-proximal missense and RNA effects
sp = prot[prot.splice_proximal_any == True]
R["protein_variants_splice_proximal"] = int(len(sp))
R["splice_proximal_lof_fraction"] = float((sp.score < -1.328).mean()) if len(sp) else None
R["non_splice_proximal_lof_fraction"] = float((prot[prot.splice_proximal_any != True].score < -1.328).mean())
R["missense_with_low_rna_score_lt_-1"] = int((prot.rna_score < -1).sum())

# 5. Independent replicate experiment (Dace et al. 2025 HAP1 re-assay of exons 5 and 17)
rep = prot.dropna(subset=["dace_hap1_score"])
R["findlay_vs_dace_hap1_n"] = int(len(rep))
R["findlay_vs_dace_hap1_spearman"] = float(stats.spearmanr(rep.score, rep.dace_hap1_score)[0])
R["findlay_vs_dace_hap1_pearson"] = float(stats.pearsonr(rep.score, rep.dace_hap1_score)[0])
hm = prot.dropna(subset=["hmec_score"])
R["findlay_vs_hmec_n"] = int(len(hm))
R["findlay_vs_hmec_spearman"] = float(stats.spearmanr(hm.score, hm.hmec_score)[0])

# 6. External BRCA1 tracks (Lyra 2026), Findlay track excluded: agreement with Findlay class
long1 = pd.read_csv(P / "brca1_lyra2026_track_calls_long.tsv", sep="\t")
EXCLUDED_TRACKS = ["T127", "T159"]  # T127 = Findlay itself; T159 failed coding QC (see report)
ext = long1[(~long1.track.isin(EXCLUDED_TRACKS)) & (long1.call != 1)].copy()
ext["aa_pos"] = ext.aa_pos.astype(int)
m = ext.merge(prot[["aa_pos", "aa_ref", "aa_alt", "score"]], on=["aa_pos", "aa_ref", "aa_alt"])
R["external_brca1_calls_overlapping_findlay"] = int(len(m))
R["external_brca1_tracks_overlapping_findlay"] = int(m.track.nunique())
R["external_brca1_auroc_findlay_score_vs_track_lof"] = float(roc_auc_score(m.call == 2, -m.score))
per = []
for t, g in m.groupby("track"):
    if g.call.nunique() == 2 and len(g) >= 20:
        per.append({"track": t, "name": g.Track.iloc[0], "n": len(g), "n_lof": int((g.call == 2).sum()),
                    "auroc_findlay": float(roc_auc_score(g.call == 2, -g.score))})
R["external_brca1_per_track_auroc"] = sorted(per, key=lambda x: -x["n"])

# 7. BRCA2 cross-assay agreement
b2 = pd.read_csv(P / "brca2_functional_calls_protein.tsv", sep="\t")
for x, y in [("huang2025_hap1", "sahu2025_mesc"), ("huang2025_hap1", "hu2024_hdr"), ("sahu2025_mesc", "hu2024_hdr")]:
    d = b2.dropna(subset=[x, y])
    d2 = d[(d[x] != 1) & (d[y] != 1)]
    R[f"brca2_{x}_vs_{y}_n_all"] = int(len(d))
    R[f"brca2_{x}_vs_{y}_kappa_3class"] = float(cohen_kappa_score(d[x], d[y]))
    R[f"brca2_{x}_vs_{y}_n_binary"] = int(len(d2))
    R[f"brca2_{x}_vs_{y}_kappa_binary"] = float(cohen_kappa_score(d2[x], d2[y]))
    R[f"brca2_{x}_vs_{y}_agreement_binary"] = float((d2[x] == d2[y]).mean())
d = b2.dropna(subset=["huang2025_hap1", "sahu2025_mesc"])
R["brca2_huang_sahu_crosstab_rows_huang_cols_sahu"] = pd.crosstab(d.huang2025_hap1, d.sahu2025_mesc).values.tolist()
pos = b2.dropna(subset=["huang2025_hap1"]).aa_pos
R["brca2_huang_aa_range"] = [int(pos.min()), int(pos.max())]

(ROOT / "results").mkdir(exist_ok=True)
json.dump(R, open(ROOT / "results" / "stage2_label_validity.json", "w"), indent=1)

# Figures
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for c, col in [("Synonymous", "#4C78A8"), ("Missense", "#F58518"), ("Nonsense", "#E45756")]:
    ax[0].hist(snv[snv.consequence == c].score, bins=60, alpha=0.6, label=c, color=col, density=True)
ax[0].set_xlabel("Findlay 2018 function score"); ax[0].set_ylabel("density"); ax[0].legend(frameon=False)
x = prot[["score_r1", "score_r2"]].dropna()
ax[1].scatter(x.score_r1, x.score_r2, s=4, alpha=0.4, color="#4C78A8")
ax[1].set_xlabel("replicate 1"); ax[1].set_ylabel("replicate 2")
ax[1].set_title(f"protein level, Spearman {R['replicate_spearman_protein']:.2f}", fontsize=9)
ax[2].scatter(rep.score, rep.dace_hap1_score, s=6, alpha=0.6, color="#54A24B")
ax[2].set_xlabel("Findlay 2018 (HAP1)"); ax[2].set_ylabel("Dace 2025 re-assay (HAP1)")
ax[2].set_title(f"exons 5 and 17, n={len(rep)}, Spearman {R['findlay_vs_dace_hap1_spearman']:.2f}", fontsize=9)
for a_ in ax: a_.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); (ROOT / "figures").mkdir(exist_ok=True)
plt.savefig(ROOT / "figures" / "stage2_label_validity.png", dpi=200); plt.savefig(ROOT / "figures" / "stage2_label_validity.svg")
print(json.dumps(R, indent=1))
