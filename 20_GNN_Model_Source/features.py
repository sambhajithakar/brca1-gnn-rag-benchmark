"""Feature construction and data partitions (implements sections 2-3 of 16A_PRESPECIFIED_ANALYSIS_PLAN.md).

Outputs data/processed/model_table_brca1.tsv: one row per Findlay missense protein variant with
label, feature groups (prefixes sub_, evo_, st_, nb_, an_), structure node index and fold columns.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
G = ROOT / "18_Residue_Graph_Construction/out"
AA = "ACDEFGHIKLMNPQRSTVWY"
ALPH = "ACDEFGHIKLMNPQRSTVWYX"
KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5, L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)
VOL = dict(A=88.6, R=173.4, N=114.1, D=111.1, C=108.5, Q=143.8, E=138.4, G=60.1, H=153.2, I=166.7, L=166.7, K=168.6, M=162.9, F=189.9, P=112.7, S=89.0, T=116.1, W=227.8, Y=193.6, V=140.0)
CHG = dict(D=-1, E=-1, K=1, R=1)
# Grantham (1974) composition, polarity, volume
GC = dict(A=0, R=0.65, N=1.33, D=1.38, C=2.75, Q=0.89, E=0.92, G=0.74, H=0.58, I=0, L=0, K=0.33, M=0, F=0, P=0.39, S=1.42, T=0.71, W=0.13, Y=0.20, V=0)
GP = dict(A=8.1, R=10.5, N=11.6, D=13.0, C=5.5, Q=10.5, E=12.3, G=9.0, H=10.4, I=5.2, L=4.9, K=11.3, M=5.7, F=5.2, P=8.0, S=9.2, T=8.6, W=5.4, Y=6.2, V=5.9)
GV = dict(A=31, R=124, N=56, D=54, C=55, Q=85, E=83, G=3, H=96, I=111, L=111, K=119, M=105, F=132, P=32.5, S=32, T=61, W=170, Y=136, V=84)
AACLASS = {**{a: "hydrophobic" for a in "AVLIMFWC"}, **{a: "polar" for a in "STNQYH"}, "D": "neg", "E": "neg", "K": "pos", "R": "pos", "G": "special", "P": "special"}


def grantham(a, b):
    return 50.723 * np.sqrt(1.833 * (GC[a] - GC[b]) ** 2 + 0.1018 * (GP[a] - GP[b]) ** 2 + 0.000399 * (GV[a] - GV[b]) ** 2)


# Check against published Grantham matrix entries (within 1.5 units; residual is rounding of the constant)
for _a, _b, _ref in [("L", "I", 5), ("C", "W", 215), ("S", "R", 110), ("D", "E", 45), ("F", "Y", 22), ("A", "G", 60), ("G", "W", 184)]:
    assert abs(grantham(_a, _b) - _ref) < 1.5, "Grantham table check failed"
BLOSUM = substitution_matrices.load("BLOSUM62")


def substitution_features(df):
    out = pd.DataFrame(index=df.index)
    for a in AA:
        out[f"sub_wt_{a}"] = (df.aa_ref == a).astype(float)
        out[f"sub_mt_{a}"] = (df.aa_alt == a).astype(float)
    out["sub_d_kd"] = df.aa_alt.map(KD) - df.aa_ref.map(KD)
    out["sub_d_vol"] = df.aa_alt.map(VOL) - df.aa_ref.map(VOL)
    out["sub_d_charge"] = df.aa_alt.map(lambda a: CHG.get(a, 0)) - df.aa_ref.map(lambda a: CHG.get(a, 0))
    out["sub_d_polarity"] = df.aa_alt.map(GP) - df.aa_ref.map(GP)
    out["sub_blosum62"] = [BLOSUM[a][b] for a, b in zip(df.aa_ref, df.aa_alt)]
    out["sub_grantham"] = [grantham(a, b) for a, b in zip(df.aa_ref, df.aa_alt)]
    return out


def build():
    df = pd.read_csv(P / "brca1_findlay2018_missense_protein.tsv", sep="\t")
    df = df.drop(columns=["domain"]).sort_values(["aa_pos", "aa_alt"]).reset_index(drop=True)
    X = substitution_features(df)
    evo = pd.DataFrame({"evo_alphamissense": df.am_pathogenicity, "evo_eve": df.eve, "evo_eve_index": df.eve_evol_index,
                        "evo_phylop": df.phylop, "evo_cadd": df.cadd, "evo_polyphen2": df.polyphen2,
                        "evo_sift": df.sift, "evo_agvgd": df.agvgd_diff})
    # Structure mapping (BRCA1 chain A of 1JM7 or 1T29)
    node_rows = []
    for pdb in ("1jm7", "1t29"):
        nodes = pd.read_csv(G / pdb / "nodes.tsv", sep="\t")
        lp = np.load(G / pdb / "mpnn_logp.npy")
        dh = np.load(G / pdb / "dist_heavy.npy")
        nodes["node_idx"] = np.arange(len(nodes))
        nb = (dh < 8.0) & ~np.eye(len(nodes), dtype=bool)
        for c in ["rsa", "n_contacts_8A", "mpnn_entropy", "interface"]:
            v = nodes[c].to_numpy(dtype=float)
            nodes[f"nb_{c}"] = [np.nanmean(v[nb[i]]) if nb[i].any() else np.nan for i in range(len(nodes))]
        a = nodes[nodes.chain == "A"].copy()
        a["lp"] = list(lp[a.node_idx.to_numpy()])
        node_rows.append(a)
    nodes = pd.concat(node_rows)
    nodes = nodes.rename(columns={"resnum": "aa_pos"})
    df = df.merge(nodes[["aa_pos", "structure", "node_idx", "rsa", "ss3", "n_contacts_8A", "interface", "zn_ligand", "ca_rmsf_nmr",
                         "bfactor_z", "mpnn_entropy", "mpnn_lp_native", "lp", "domain", "nb_rsa", "nb_n_contacts_8A",
                         "nb_mpnn_entropy", "nb_interface"]], on="aa_pos", how="left")
    df["struct_available"] = df.structure.notna()
    st = pd.DataFrame(index=df.index)
    st["st_rsa"] = df.rsa
    for s in "HE-":
        st[f"st_ss_{'C' if s == '-' else s}"] = (df.ss3 == s).astype(float).where(df.struct_available)
    st["st_contacts"] = df.n_contacts_8A
    st["st_interface"] = df.interface
    st["st_rmsf_nmr"] = df.ca_rmsf_nmr.fillna(0).where(df.struct_available)
    st["st_is_nmr"] = (df.structure == "1jm7").astype(float).where(df.struct_available)
    st["st_bfactor_z"] = df.bfactor_z.fillna(0).where(df.struct_available)
    st["st_mpnn_entropy"] = df.mpnn_entropy
    st["st_mpnn_lp_wt"] = df.mpnn_lp_native
    st["st_mpnn_lp_mt"] = [l[ALPH.index(m)] if isinstance(l, np.ndarray) else np.nan for l, m in zip(df.lp, df.aa_alt)]
    st["st_mpnn_llr"] = st.st_mpnn_lp_mt - st.st_mpnn_lp_wt
    nbf = df[["nb_rsa", "nb_n_contacts_8A", "nb_mpnn_entropy", "nb_interface"]]
    an = pd.DataFrame(index=df.index)
    for d in ["RING", "RING_flank", "BRCT1", "BRCT_linker", "BRCT2"]:
        an[f"an_dom_{d}"] = (df.domain == d).astype(float).where(df.struct_available)
    an["an_zn_ligand"] = df.zn_ligand
    an["an_interface"] = df.interface
    base = df[["aa_pos", "aa_ref", "aa_alt", "hgvs_p", "exon_id", "score", "score_r1", "score_r2", "n_snv", "rna_score",
               "splice_proximal_any", "structure", "node_idx", "struct_available", "domain", "ext_n_tracks", "ext_n_lof",
               "ext_n_normal", "dace_hap1_score", "hmec_score"]].copy()
    base["region"] = np.where(base.aa_pos <= 110, "RING", "BRCT")
    base["mt_class"] = base.aa_alt.map(AACLASS)
    out = pd.concat([base, X, evo, st, nbf, an], axis=1)
    # Partitions
    res = out[["aa_pos", "region"]].drop_duplicates().reset_index(drop=True)
    for name, seed in [("fold_P0", 20261025), ("fold_P1", 20261026), ("fold_P2", 20261027)]:
        rng = np.random.default_rng(seed)
        f = {}
        for reg, g in res.groupby("region"):
            pos = rng.permutation(g.aa_pos.to_numpy())
            for i, p in enumerate(pos): f[p] = i % 5
        out[name] = out.aa_pos.map(f)
    out["fold_LODO"] = (out.region == "BRCT").astype(int)
    exons = sorted(out.exon_id.unique(), key=lambda e: int(e[1:]))
    out["fold_LOEO"] = out.exon_id.map({e: i for i, e in enumerate(exons)})
    rng = np.random.default_rng(20261025)
    out["fold_random"] = rng.permutation(np.arange(len(out)) % 5)
    out = out.drop(columns=[])
    out.to_csv(P / "model_table_brca1.tsv", sep="\t", index=False)
    return out


if __name__ == "__main__":
    t = build()
    print(t.shape, "struct_available:", int(t.struct_available.sum()), "residues:", t.aa_pos.nunique(),
          "struct residues:", t[t.struct_available].aa_pos.nunique())
    print(t.groupby("fold_P0").agg(n=("score", "size"), res=("aa_pos", "nunique")).T)
    print(t.filter(regex="^(st_|evo_|nb_)").isna().sum().to_dict())
