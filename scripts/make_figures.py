"""Stage 13: manuscript figures, generated only from saved results. Output: figures/Fig*.png (300 dpi) and .svg."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"; FIG = ROOT / "figures"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
LAB = {"zs_alphamissense": "AlphaMissense (zero-shot)", "zs_eve": "EVE (zero-shot)", "zs_mpnn_llr": "ProteinMPNN LLR (zero-shot)",
       "zs_cadd": "CADD (zero-shot)", "zs_phylop": "phyloP (zero-shot)", "ridge_amonly": "Ridge, AlphaMissense only",
       "ridge_seqevo": "Ridge, seq+evo", "rf_seqevo": "Random forest, seq+evo", "lgbm_seqevo": "LightGBM, seq+evo",
       "mlp_seqevo": "MLP, seq+evo", "lgbm_struct": "LightGBM, seq+evo+structure", "labelsmooth": "Spatial label smoothing",
       "gnn_A_seqonly": "A: network, no structure", "gnn_B_struct": "B: GNN, structure only", "gnn_C_struct_annot": "C: GNN, structure+seq+evo",
       "gnn_C_nograph": "C without graph (ablation)", "gnn_C_gat": "C with attention (GAT)", "gnn_D_evidence": "D: C + retrieved evidence",
       "gnn_D_nograph": "D without graph", "lgbm_seqevo_ev": "LightGBM seq+evo + retrieved evidence"}


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "svg"): fig.savefig(FIG / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 4.0)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    def box(x, y, w, h, t, c):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=c, ec="0.3", lw=0.8)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.5, wrap=True)
    box(0.1, 4.4, 2.6, 1.3, "Labels (measured by others)\nFindlay 2018 BRCA1 SGE\n1,837 missense, 326 residues", "#dbe9f6")
    box(0.1, 2.6, 2.6, 1.4, "Independent assays\nDace 2025 HAP1/HMEC SGE\nLyra 2026 tracks (BRCA1, BRCA2)\nClinVar 2024-09-13", "#dbe9f6")
    box(0.1, 0.8, 2.6, 1.4, "Structure (experimental)\nPDB 1JM7 (RING-BARD1)\nPDB 1T29 (BRCT-BACH1)\nProteinMPNN log-odds", "#dbe9f6")
    box(3.4, 3.6, 2.9, 2.1, "Models (residue-grouped 5-fold CV,\ninner CV for tuning, 5 seeds)\nZero-shot, linear, RF, LightGBM, MLP\nA: no structure  B: graph only\nC: graph + seq/evo  C-nograph\nD: C + retrieved evidence", "#fde9d9")
    box(3.4, 0.8, 2.9, 2.3, "Evidence retrieval\nScenario 1 (prediction): fold-specific\nindex, no Findlay docs, no test residues,\nnumeric summaries only, leakage tests\nScenario 2 (interpretation): all docs,\ncited template synthesis", "#fde9d9")
    box(7.0, 3.6, 2.9, 2.1, "Evaluation\nPrimary: OOF Spearman, partition P0\nH1: C vs C-nograph (bootstrap CI)\nHolm-corrected secondary tests\nSensitivity: P1, P2, LOEO, LODO, random", "#e2efda")
    box(7.0, 0.8, 2.9, 2.3, "Validation and uncertainty\nDace re-assay, new regions, HMEC\nLyra track AUROC; BRCA2 transfer\nSplit conformal, P(LoF), risk-coverage\nAssay discordance", "#e2efda")
    for (x1, y1, x2, y2) in [(2.75, 5.0, 3.35, 4.8), (2.75, 1.5, 3.35, 3.9), (2.75, 3.3, 3.35, 2.2), (6.35, 4.6, 6.95, 4.6), (6.35, 2.0, 6.95, 2.0), (4.85, 3.15, 4.85, 3.55)]:
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))
    ax.set_title("Study design", fontsize=10, loc="left")
    save(fig, "Fig1_study_design")


def fig2():
    M = pd.read_csv(RES / "metrics__fold_P0__struct.tsv", sep="\t")
    order = ["zs_phylop", "zs_cadd", "zs_mpnn_llr", "zs_eve", "zs_alphamissense", "ridge_amonly", "mlp_seqevo", "ridge_seqevo", "rf_seqevo", "lgbm_seqevo",
             "labelsmooth", "lgbm_struct", "lgbm_seqevo_ev", "gnn_A_seqonly", "gnn_B_struct", "gnn_C_nograph", "gnn_C_struct_annot", "gnn_C_gat", "gnn_D_nograph", "gnn_D_evidence"]
    M = M.set_index("model").loc[[o for o in order if o in M.model.values]]
    t = json.load(open(RES / "tests__fold_P0__struct.json"))
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    y = np.arange(len(M)); col = ["#999999" if i.startswith("zs") else ("#C44E52" if i.startswith("gnn") else "#4C72B0") for i in M.index]
    ax[0].errorbar(M.spearman, y, xerr=[M.spearman - M.spearman_ci_lo, M.spearman_ci_hi - M.spearman], fmt="none", ecolor="0.5", lw=0.8)
    ax[0].scatter(M.spearman, y, c=col, s=18, zorder=3)
    ax[0].set_yticks(y); ax[0].set_yticklabels([LAB.get(i, i) for i in M.index], fontsize=7)
    ax[0].axvline(0.91, ls=":", c="0.5"); ax[0].text(0.905, len(M) - 0.6, "replicate\nceiling ~0.91", fontsize=6, ha="right")
    ax[0].set_xlabel("Spearman, out-of-fold (95% residue bootstrap CI)"); ax[0].set_title("a  Held-out residues (P0, n=1,734)", loc="left", fontsize=9)
    SL = {"S-a": "S-a: C − LightGBM seq+evo", "S-b": "S-b: C − AlphaMissense", "S-c": "S-c: B − A", "S-e": "S-e: C − LightGBM +structure",
          "S-f": "S-f: C − label smoothing", "S-d": "S-d: D − C"}
    comps = [("H1: C − C-nograph", t["H1_primary"])] + [(SL[s["id"]], s) for s in t["secondary"]]
    yy = np.arange(len(comps))[::-1]
    for yi, (lab, s) in zip(yy, comps):
        ax[1].errorbar(s["delta_spearman"], yi, xerr=[[s["delta_spearman"] - s["ci_lo"]], [s["ci_hi"] - s["delta_spearman"]]], fmt="o", ms=4, c="k", lw=0.8)
        p = s.get("p_holm", s["p_boot"]); ax[1].text(0.075, yi, f"p={p:.2g}", va="center", fontsize=6.5)
    ax[1].axvline(0, c="0.4", lw=0.8); ax[1].set_yticks(yy); ax[1].set_yticklabels([c[0] for c in comps], fontsize=6.5)
    ax[1].set_xlim(-0.09, 0.1); ax[1].set_xlabel("Δ Spearman (95% CI)"); ax[1].set_title("b  Paired comparisons", loc="left", fontsize=9)
    save(fig, "Fig2_performance_P0")


def fig3():
    S = pd.read_csv(RES / "sensitivity_tests.tsv", sep="\t")
    h = S[(S.a.str.startswith("gnn_C_struct_annot")) & (S.b == "gnn_C_nograph")].copy()
    h["label"] = h.analysis.str.replace("partition:fold_", "Partition ").str.replace("S4:graph:", "Graph: ").str.replace("S1:no_splice_proximal_or_low_RNA", "S1: no splice-proximal")
    d = S[(S.a == "gnn_D_evidence") & (S.b == "gnn_C_struct_annot")].copy(); d["label"] = d.analysis.str.replace("partition:fold_", "Partition ").str.replace("S1:no_splice_proximal_or_low_RNA", "S1: no splice-proximal")
    T = pd.read_csv(RES / "sensitivity_table.tsv", sep="\t")
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 3.4))
    for a, D, title in ((ax[0], h, "a  Graph effect (C − C-nograph)"), (ax[1], d, "b  Evidence effect (D − C)")):
        yy = np.arange(len(D))[::-1]
        a.errorbar(D.delta_spearman, yy, xerr=[D.delta_spearman - D.ci_lo, D.ci_hi - D.delta_spearman], fmt="o", ms=3.5, c="k", lw=0.8)
        a.axvline(0, c="0.4", lw=0.8); a.set_yticks(yy); a.set_yticklabels(D.label, fontsize=6.5); a.set_xlabel("Δ Spearman (95% CI)"); a.set_title(title, loc="left", fontsize=8.5)
    parts = ["partition:fold_P0", "partition:fold_P1", "partition:fold_P2", "partition:fold_LOEO", "partition:fold_LODO", "partition:fold_random"]
    for m, c in (("zs_alphamissense", "#999999"), ("lgbm_seqevo", "#4C72B0"), ("gnn_C_struct_annot", "#C44E52"), ("gnn_D_evidence", "#DD8452")):
        v = [T[(T.analysis == p) & (T.model == m)].spearman.mean() for p in parts]
        ax[2].plot(range(len(parts)), v, "o-", c=c, ms=3, lw=1, label=LAB[m].split(" (")[0])
    ax[2].set_xticks(range(len(parts))); ax[2].set_xticklabels([p.split("_")[-1] for p in parts], rotation=45, fontsize=7)
    ax[2].set_ylabel("Spearman"); ax[2].set_title("c  By split", loc="left", fontsize=8.5); ax[2].legend(fontsize=5.5, frameon=False, loc="lower left")
    save(fig, "Fig3_robustness")


def fig4():
    R = json.load(open(ROOT / "29_External_Validation/out/external_validation.json"))
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 3.0))
    ms = [("zs_alphamissense__struct", "AlphaMissense"), ("lgbm_seqevo__struct", "LightGBM seq+evo"), ("lgbm_struct__struct", "LightGBM +struct"),
          ("gnn_C_nograph__struct", "C-nograph"), ("gnn_C_struct_annot__struct", "C (GNN)"), ("gnn_D_evidence__struct", "D (+evidence)")]
    for k, (key, title) in enumerate((("E1_dace_hap1_reassay", "a  Dace HAP1 re-assay (n=302)"), ("E3_dace_hmec", "b  Dace HMEC (n=399)"))):
        E = R[key]; v = [E[m]["rho"] for m, _ in ms]
        lo = [E[m]["ci"][0] for m, _ in ms]; hi = [E[m]["ci"][1] for m, _ in ms]
        yy = np.arange(len(ms))[::-1]
        ax[k].errorbar(v, yy, xerr=[np.array(v) - lo, np.array(hi) - v], fmt="o", ms=3.5, c="k", lw=0.8)
        ax[k].axvline(E["findlay_score_vs_external"]["rho"], ls=":", c="#4C72B0"); ax[k].text(E["findlay_score_vs_external"]["rho"] - 0.005, -0.4, "Findlay\nassay", fontsize=6, color="#4C72B0", ha="right")
        ax[k].set_yticks(yy); ax[k].set_yticklabels([m[1] for m in ms] if k == 0 else [], fontsize=7); ax[k].set_xlabel("Spearman (95% CI)"); ax[k].set_title(title, loc="left", fontsize=8.5)
    B = R["BRCA2_secondary"]
    rows = [("Huang (OOF)", B["train_huang"]["lgbm->huang"]["auroc"], B["train_huang"]["zs_alphamissense->huang"]["auroc"]),
            ("→ Sahu", B["train_huang"]["lgbm->sahu"]["auroc"], B["train_huang"]["zs_alphamissense->sahu"]["auroc"]),
            ("→ Hu HDR", B["train_huang"]["lgbm->hu_hdr"]["auroc"], B["train_huang"]["zs_alphamissense->hu_hdr"]["auroc"])]
    x = np.arange(len(rows)); ax[2].bar(x - 0.18, [r[1] for r in rows], 0.36, color="#4C72B0", label="LightGBM trained on Huang")
    ax[2].bar(x + 0.18, [r[2] for r in rows], 0.36, color="#999999", label="AlphaMissense")
    ax[2].set_xticks(x); ax[2].set_xticklabels([r[0] for r in rows], fontsize=7); ax[2].set_ylim(0.6, 1.0); ax[2].set_ylabel("AUROC (LoF vs normal)")
    ax[2].legend(fontsize=6, frameon=False, loc="upper left"); ax[2].set_title("c  BRCA2 (secondary)", loc="left", fontsize=8.5)
    save(fig, "Fig4_external_validation")


def fig5():
    S = json.load(open(ROOT / "26_Uncertainty_Calibration/out/uncertainty_summary.json"))
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.8))
    for name, c, lab in (("lgbm_seqevo", "#4C72B0", "LightGBM seq+evo"), ("gnn_C_struct_annot", "#C44E52", "C (GNN)")):
        s = S[name]; rel = np.array(s["reliability"])
        ax[0].plot(rel[:, 0], rel[:, 1], "o-", c=c, ms=3, lw=1, label=f"{lab} (ECE {s['ece']:.3f})")
        ax[1].plot(s["rc_standard_sd"]["coverage"], s["rc_standard_sd"]["rmse"], c=c, lw=1.2, label=lab)
    ax[0].plot([0, 1], [0, 1], ls="--", c="0.6", lw=0.8); ax[0].set(xlabel="Predicted P(LoF)", ylabel="Observed LoF fraction"); ax[0].legend(fontsize=6, frameon=False)
    ax[0].set_title("a  Calibration", loc="left", fontsize=8.5)
    ax[1].set(xlabel="Coverage (most confident first)", ylabel="RMSE"); ax[1].set_title("b  Selective prediction", loc="left", fontsize=8.5); ax[1].legend(fontsize=6, frameon=False)
    D = pd.read_csv(ROOT / "26_Uncertainty_Calibration/out/lgbm_seqevo_intervals.tsv", sep="\t")
    y = D.score.to_numpy(); f = np.where(y < -1.328, 1, np.where(y > -0.748, 0, -1)); e = np.where(D.ext_n_lof > D.ext_n_normal, 1, np.where(D.ext_n_normal > D.ext_n_lof, 0, -1))
    ok = (f >= 0) & (e >= 0); w = (D.nhi90 - D.nlo90).to_numpy()
    ax[2].boxplot([w[ok & (f == e)], w[ok & (f != e)]], widths=0.5, showfliers=False)
    ax[2].set_xticks([1, 2]); ax[2].set_xticklabels([f"concordant\n(n={int((ok & (f == e)).sum())})", f"discordant\n(n={int((ok & (f != e)).sum())})"], fontsize=7)
    ax[2].set_ylabel("Normalised 90% interval width"); ax[2].set_title(f"c  Discordance (p={S['lgbm_seqevo']['discordance']['mwu_p_nwidth']:.1g})", loc="left", fontsize=8.5)
    save(fig, "Fig5_uncertainty_discordance")


def fig6():
    M = pd.read_csv(ROOT / "23_RAG_Implementation/out/scenario2_summary.tsv", sep="\t")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    meth = ["bm25", "tfidf", "hybrid", "hybrid_vf"]; ml = ["BM25", "TF-IDF", "Hybrid", "Hybrid +\nvariant filter"]
    for k, (col, title) in enumerate((("r@20", "a  Recall@20"), ("unfaithful_rate", "b  Statements about the wrong variant"))):
        for j, (st, c) in enumerate((("external_evidence", "#4C72B0"), ("findlay_only", "#DD8452"))):
            v = [M[(M.stratum == st) & (M.method == m)][col].iloc[0] for m in meth]
            ax[k].bar(np.arange(4) + (j - 0.5) * 0.36, v, 0.36, color=c, label={"external_evidence": "with independent assays", "findlay_only": "Findlay only"}[st])
        ax[k].set_xticks(range(4)); ax[k].set_xticklabels(ml, fontsize=7); ax[k].set_title(title, loc="left", fontsize=8.5); ax[k].set_ylim(0, 1)
    ax[0].legend(fontsize=6, frameon=False)
    save(fig, "Fig6_interpretation_retrieval")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f(); print(f.__name__, "ok")
