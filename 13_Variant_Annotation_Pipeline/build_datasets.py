"""Stage 2 variant annotation pipeline.

Builds harmonised BRCA1/BRCA2 variant tables from the raw sources staged in data/raw
(see 15_Data_Provenance.md). Every check that fails is written to the QC report rather
than silently dropped.

Coordinate conventions
- BRCA1: transcript NM_007294.3 (as in Findlay et al. 2018; ENST00000357654 in Ensembl),
  protein P38398 (1,863 aa). Genomic positions are given in GRCh37 by Findlay and in
  GRCh38 by Dace et al.; the Dace repository hg19->hg38 table is used to lift Findlay
  positions, and each lift is checked for a constant offset within exons.
- BRCA2: protein P51587 (3,418 aa). Lyra et al. tracks are protein-level (codons
  collapsed by the original authors), so no genomic harmonisation is attempted.

Usage: python build_datasets.py  (paths are relative to the project root)
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)
QC = {}

AA3 = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G",
       "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S",
       "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V", "Ter": "*"}
COMP = str.maketrans("ACGT", "TGCA")


def load_sequences():
    ref = pd.read_csv(RAW / "proteingym_ref" / "DMS_substitutions.csv")
    seqs = {}
    for uid, gene in [("BRCA1_HUMAN", "BRCA1"), ("BRCA2_HUMAN", "BRCA2")]:
        seqs[gene] = ref.loc[ref.UniProt_ID == uid, "target_seq"].iloc[0]
    QC["protein_lengths"] = {g: len(s) for g, s in seqs.items()}
    return seqs


def findlay(seqs):
    f = RAW / "Findlay2018_dmsview_mirror" / "Findlay2018_suppTable1.csv"
    a = pd.read_csv(f, header=2, encoding="utf-8-sig")
    b = pd.read_csv(RAW / "dace2025_repo" / "Findlay_et_al_2018_data.csv")
    m = a.merge(b.rename(columns={"pos": "position (hg19)"}), on=["position (hg19)", "alt"], suffixes=("", "_b"))
    QC["findlay_rows"] = len(a)
    QC["findlay_mirror_agreement_rows"] = len(m)
    QC["findlay_mirror_max_abs_score_diff"] = float(np.nanmax(np.abs(m["function.score.mean"] - m["function.score.mean_b"])))
    d = a.rename(columns={"position (hg19)": "pos_hg19", "reference": "ref", "alt": "alt",
                          "function.score.mean": "score", "function.score.r1": "score_r1",
                          "function.score.r2": "score_r2", "func.class": "func_class",
                          "transcript_variant": "hgvs_c", "protein_variant": "p_short",
                          "mean.rna.score": "rna_score", "CADD.score": "cadd", "phyloP (mammalian)": "phylop",
                          "polyphen2": "polyphen2", "sift": "sift", "aGVGD.diff": "agvgd_diff"})
    # Strand check: BRCA1 is on the minus strand, so transcript alleles are complements.
    QC["findlay_strand_consistent"] = int(((d.ref.str.translate(COMP) == d.transcript_ref) &
                                           (d.alt.str.translate(COMP) == d.transcript_alt)).sum())
    # Protein reference check against UniProt P38398 sequence.
    mis = d.consequence.isin(["Missense", "Nonsense", "Synonymous"]) & d.aa_pos.notna()
    seq = seqs["BRCA1"]
    ok = d[mis].apply(lambda r: seq[int(r.aa_pos) - 1] == r.aa_ref, axis=1)
    QC["findlay_aa_ref_matches_uniprot"] = f"{int(ok.sum())}/{int(mis.sum())}"
    # hg19 -> hg38 lift using the Dace repository conversion table.
    conv = pd.read_csv(RAW / "dace2025_repo" / "230420_BRCA1_hg19_hg38_conversion.csv")
    QC["conversion_columns"] = list(conv.columns)
    c19 = [c for c in conv.columns if "19" in c][0]
    c38 = [c for c in conv.columns if "38" in c][0]
    conv = conv[[c19, c38]].drop_duplicates()
    d = d.merge(conv.rename(columns={c19: "pos_hg19", c38: "pos_hg38"}), on="pos_hg19", how="left")
    off = (d.pos_hg38 - d.pos_hg19)
    QC["findlay_hg38_lifted"] = int(d.pos_hg38.notna().sum())
    QC["findlay_hg38_offsets"] = off.dropna().astype(int).value_counts().to_dict()
    d["hgvs_p"] = np.where(mis, "p." + d.aa_ref.astype(str) + d.aa_pos.astype("Int64").astype(str) + d.aa_alt.astype(str), None)
    d["gene"] = "BRCA1"
    d["transcript"] = "NM_007294.3"
    d["genome_build_source"] = "GRCh37"
    d["exon_id"] = d.experiment
    # Splice-proximity flag from exon coding boundaries observed in the assay.
    d["c_pos"] = pd.to_numeric(d.hgvs_c.str.extract(r"^c\.(\d+)[ACGT]>")[0], errors="coerce")
    coding = d[d.c_pos.notna() & d.consequence.isin(["Missense", "Nonsense", "Synonymous"])]
    bounds = coding.groupby("experiment").c_pos.agg(["min", "max"])
    d = d.join(bounds, on="experiment")
    dist = np.minimum((d.c_pos - d["min"]).abs(), (d["max"] - d.c_pos).abs())
    # Exon 2 starts upstream of c.1 and the last exon ends at the stop codon: no splice boundary there.
    dist = np.where((d.experiment == "X2") & (d.c_pos - d["min"] < d["max"] - d.c_pos), np.inf, dist)
    dist = np.where((d.experiment == "X23") & (d["max"] - d.c_pos < d.c_pos - d["min"]), np.inf, dist)
    d["dist_to_exon_boundary_nt"] = dist
    d["splice_proximal_flag"] = d.dist_to_exon_boundary_nt <= 2
    d = d.drop(columns=["min", "max"])
    keep = ["gene", "transcript", "genome_build_source", "chromosome", "pos_hg19", "pos_hg38", "ref", "alt",
            "hgvs_c", "c_pos", "exon_id", "consequence", "aa_pos", "aa_ref", "aa_alt", "hgvs_p", "score",
            "score_r1", "score_r2", "func_class", "p.nonfunctional", "rna_score", "cadd", "phylop", "polyphen2",
            "sift", "agvgd_diff", "clinvar_simple", "gnomAD_AF", "dist_to_exon_boundary_nt", "splice_proximal_flag"]
    d = d[keep]
    d.to_csv(OUT / "brca1_findlay2018_snv_harmonised.tsv", sep="\t", index=False)
    QC["findlay_consequence_counts"] = d.consequence.value_counts().to_dict()
    QC["findlay_class_fraction"] = d.func_class.value_counts(normalize=True).round(4).to_dict()
    return d


def amplicon_check(d):
    """Check Findlay GRCh37 reference alleles against the amplicon FASTA sequences that
    are available for overlapping regions (exons 3, 5, 17)."""
    coords = pd.read_csv(RAW / "dace2025_repo" / "amplicon_coords_hg19.txt", sep=r"\s+")
    res = {}
    for _, r in coords.iterrows():
        fa = RAW / "dace2025_repo" / "amplicon_fasta" / f"{r.amplicon}.fa"
        if not fa.exists():
            continue
        s = "".join(l.strip() for l in open(fa) if not l.startswith(">")).upper()
        start = int(r.hg19_start)
        # Orientation is not documented in the source repository, so both readings are tested:
        # (a) forward genomic from hg19_start upward; (b) transcript (minus-strand) orientation
        # from hg19_start downward, i.e. genomic base = complement(s[start - pos]).
        fw = d[(d.pos_hg19 >= start) & (d.pos_hg19 < start + len(s))]
        rv = d[(d.pos_hg19 <= start) & (d.pos_hg19 > start - len(s))]
        out = {"amplicon_length": len(s)}
        if len(fw):
            out["forward_n"] = int(len(fw))
            out["forward_match"] = round(float(fw.apply(lambda x: s[int(x.pos_hg19) - start] == x.ref, axis=1).mean()), 4)
        if len(rv):
            out["reverse_n"] = int(len(rv))
            out["reverse_match"] = round(float(rv.apply(lambda x: s[start - int(x.pos_hg19)].translate(COMP) == x.ref, axis=1).mean()), 4)
        if len(fw) or len(rv):
            res[r.amplicon] = out
    QC["amplicon_reference_check"] = res


def protein_level(d, seqs):
    m = d[d.consequence == "Missense"].copy()
    g = m.groupby(["aa_pos", "aa_ref", "aa_alt", "hgvs_p", "exon_id"], as_index=False).agg(
        score=("score", "mean"), score_sd=("score", "std"), score_r1=("score_r1", "mean"),
        score_r2=("score_r2", "mean"), n_snv=("score", "size"), rna_score=("rna_score", "mean"),
        cadd=("cadd", "mean"), phylop=("phylop", "mean"), polyphen2=("polyphen2", "mean"),
        sift=("sift", "mean"), agvgd_diff=("agvgd_diff", "mean"),
        splice_proximal_any=("splice_proximal_flag", "max"),
        lof_snv_fraction=("func_class", lambda x: float((x == "LOF").mean())))
    g["aa_pos"] = g.aa_pos.astype(int)
    g["domain"] = np.where(g.aa_pos <= 110, "RING", "BRCT_region")
    # Multi-codon disagreement (same amino acid change, different SNVs)
    QC["protein_variants"] = int(len(g))
    QC["protein_variants_multi_snv"] = int((g.n_snv > 1).sum())
    QC["protein_positions"] = int(g.aa_pos.nunique())
    multi = g[g.n_snv > 1]
    QC["multi_snv_median_sd"] = float(multi.score_sd.median())
    return g


def alphamissense_eve(g):
    am = pd.read_csv(RAW / "alphamissense" / "AlphaMissense_aa_substitutions_BRCA1_BRCA2.tsv", sep="\t", comment="#")
    am = am[am.uniprot_id == "P38398"].copy()
    am["aa_ref"] = am.protein_variant.str[0]
    am["aa_alt"] = am.protein_variant.str[-1]
    am["aa_pos"] = am.protein_variant.str[1:-1].astype(int)
    g = g.merge(am[["aa_pos", "aa_ref", "aa_alt", "am_pathogenicity"]], on=["aa_pos", "aa_ref", "aa_alt"], how="left")
    eve = pd.read_csv(RAW / "dace2025_repo" / "241024_EVE_scores_BRCA1_HUMAN.csv")
    eve = eve.rename(columns={"wt_aa": "aa_ref", "position": "aa_pos", "mt_aa": "aa_alt", "EVE_scores_ASM": "eve",
                              "evolutionary_index_ASM": "eve_evol_index", "uncertainty_ASM": "eve_uncertainty"})
    g = g.merge(eve[["aa_pos", "aa_ref", "aa_alt", "eve", "eve_evol_index"]], on=["aa_pos", "aa_ref", "aa_alt"], how="left")
    QC["am_coverage"] = f"{int(g.am_pathogenicity.notna().sum())}/{len(g)}"
    QC["eve_coverage"] = f"{int(g.eve.notna().sum())}/{len(g)}"
    return g


def dace(seqs):
    f = RAW / "dace2025_repo" / "SUPPLEMENTARY_TABLES_1-8.xlsx"
    h = pd.read_excel(f, sheet_name="ST1", header=3)
    h = h[h.Consequence == "Missense"].copy()
    h["aa_pos"] = h.protPos.astype(int)
    h["aa_ref"] = h.oAA.map(lambda x: AA3.get(x, x))
    h["aa_alt"] = h.nAA.map(lambda x: AA3.get(x, x))
    ok = h.apply(lambda r: seqs["BRCA1"][r.aa_pos - 1] == r.aa_ref, axis=1)
    QC["dace_hap1_aa_ref_matches_uniprot"] = f"{int(ok.sum())}/{len(h)}"
    QC["dace_hap1_aa_ref_mismatches_excluded"] = h.loc[~ok, "pHGVS"].astype(str).tolist()
    h = h[ok]
    hp = h.groupby(["aa_pos", "aa_ref", "aa_alt", "SGE_region"], as_index=False).agg(
        dace_hap1_score=("final_function_score", "mean"), dace_hap1_r1=("r1_function_score", "mean"),
        dace_hap1_r2=("r2_function_score", "mean"), dace_hap1_class=("function_class", lambda x: ";".join(sorted(set(map(str, x))))),
        dace_spliceai_max=("SpliceAI_max", "max"), n_snv=("final_function_score", "size"))
    hp.to_csv(OUT / "brca1_dace2025_hap1_missense_protein.tsv", sep="\t", index=False)
    e = pd.read_excel(f, sheet_name="ST3", header=3)
    e = e[e.Consequence == "Missense"].copy()
    e["aa_pos"] = e.protPos.astype(int)
    e["aa_ref"] = e.oAA.map(lambda x: AA3.get(x, x))
    e["aa_alt"] = e.nAA.map(lambda x: AA3.get(x, x))
    ep = e.groupby(["aa_pos", "aa_ref", "aa_alt", "SGE_region"], as_index=False).agg(
        hmec_score=("final_function_score_ut", "mean"), hmec_class=("final_func_class", lambda x: ";".join(sorted(set(map(str, x))))),
        hap1_score_in_hmec_table=("HAP1_function_score_mean", "mean"), hap1_source=("HAP1_data_source", "first"))
    ep.to_csv(OUT / "brca1_dace2025_hmec_missense_protein.tsv", sep="\t", index=False)
    QC["dace_hap1_missense_protein_variants"] = int(len(hp))
    QC["dace_hap1_regions"] = hp.SGE_region.value_counts().to_dict()
    QC["dace_hmec_missense_protein_variants"] = int(len(ep))
    return hp, ep


def lyra(gene, sheet_master, sheet_tracks):
    f = RAW / "lyra2026_repo" / "SUPP_TABLES_BRCA12_APR_2026.xlsx"
    mt = pd.read_excel(f, sheet_name=sheet_master, header=1)
    tr = pd.read_excel(f, sheet_name=sheet_tracks, header=1)
    tr = tr[tr["Track #"].astype(str).str.match(r"^T\d+$")]
    func_tracks = [t for t in tr["Track #"] if int(t[1:]) >= 8 and t in mt.columns]
    long = mt[["T1", "T2", "T3", "T4", "T5"] + func_tracks].melt(id_vars=["T1", "T2", "T3", "T4", "T5"], var_name="track", value_name="call")
    raw_n = int(long.call.notna().sum())
    long["call"] = pd.to_numeric(long.call.astype(str).str.strip(), errors="coerce")
    long = long.dropna(subset=["call"])
    QC[f"{gene}_lyra_non_numeric_calls_dropped"] = raw_n - int(len(long))
    QC[f"{gene}_lyra_call_values"] = long.call.value_counts().to_dict()
    long = long.rename(columns={"T1": "exon", "T2": "aa_ref", "T3": "aa_pos", "T4": "aa_alt", "T5": "hgvs_p"})
    long["gene"] = gene
    long = long.merge(tr.rename(columns={"Track #": "track"}), on="track", how="left")
    long.to_csv(OUT / f"{gene.lower()}_lyra2026_track_calls_long.tsv", sep="\t", index=False)
    tr.to_csv(OUT / f"{gene.lower()}_lyra2026_track_metadata.tsv", sep="\t", index=False)
    QC[f"{gene}_lyra_tracks_with_data"] = int(long.track.nunique())
    QC[f"{gene}_lyra_variant_track_calls"] = int(len(long))
    QC[f"{gene}_lyra_variants_with_any_call"] = int(long.hgvs_p.nunique())
    return long, tr


def brca2_labels(long2, seqs):
    am = pd.read_csv(RAW / "alphamissense" / "AlphaMissense_aa_substitutions_BRCA1_BRCA2.tsv", sep="\t", comment="#")
    am = am[am.uniprot_id == "P51587"].copy()
    am["aa_ref"] = am.protein_variant.str[0]
    am["aa_alt"] = am.protein_variant.str[-1]
    am["aa_pos"] = am.protein_variant.str[1:-1].astype(int)
    QC["am_brca2_rows"] = int(len(am))
    QC["am_brca2_rows_per_position_max"] = int(am.groupby("aa_pos").size().max())
    piv = long2[long2.track.isin(["T152", "T153", "T143", "T154"])].pivot_table(
        index=["aa_pos", "aa_ref", "aa_alt", "hgvs_p"], columns="track", values="call", aggfunc="first").reset_index()
    piv = piv.rename(columns={"T152": "huang2025_hap1", "T153": "sahu2025_mesc", "T143": "hu2024_hdr", "T154": "bose2025"})
    piv["aa_pos"] = piv.aa_pos.astype(int)
    ok = piv.apply(lambda r: seqs["BRCA2"][r.aa_pos - 1] == r.aa_ref, axis=1)
    QC["brca2_aa_ref_matches_uniprot"] = f"{int(ok.sum())}/{len(piv)}"
    piv = piv.merge(am[["aa_pos", "aa_ref", "aa_alt", "am_pathogenicity"]], on=["aa_pos", "aa_ref", "aa_alt"], how="left")
    piv.to_csv(OUT / "brca2_functional_calls_protein.tsv", sep="\t", index=False)
    both = piv.dropna(subset=["huang2025_hap1", "sahu2025_mesc"])
    QC["brca2_huang_n"] = int(piv.huang2025_hap1.notna().sum())
    QC["brca2_sahu_n"] = int(piv.sahu2025_mesc.notna().sum())
    QC["brca2_huang_sahu_overlap"] = int(len(both))
    QC["brca2_huang_vs_sahu_crosstab"] = pd.crosstab(both.huang2025_hap1, both.sahu2025_mesc).to_dict()
    return piv


def main():
    seqs = load_sequences()
    d = findlay(seqs)
    amplicon_check(d)
    g = protein_level(d, seqs)
    g = alphamissense_eve(g)
    hp, ep = dace(seqs)
    long1, tr1 = lyra("BRCA1", "Sup Table 1", "Sup Table 3")
    long2, tr2 = lyra("BRCA2", "Sup Table 2", "Sup Table 4")
    # External BRCA1 labels at protein level (exclude Findlay-derived track T127)
    ext = long1[~long1.track.isin(["T127", "T159"])].copy()  # T127 = Findlay itself; T159 failed coding QC
    ext["aa_pos"] = ext.aa_pos.astype(int)
    ext_sum = ext.groupby(["aa_pos", "aa_ref", "aa_alt"]).agg(
        ext_n_tracks=("call", "size"), ext_n_lof=("call", lambda x: int((x == 2).sum())),
        ext_n_normal=("call", lambda x: int((x == 0).sum()))).reset_index()
    g = g.merge(ext_sum, on=["aa_pos", "aa_ref", "aa_alt"], how="left")
    g = g.merge(hp[["aa_pos", "aa_ref", "aa_alt", "dace_hap1_score"]].groupby(["aa_pos", "aa_ref", "aa_alt"], as_index=False).mean(),
                on=["aa_pos", "aa_ref", "aa_alt"], how="left")
    g = g.merge(ep[["aa_pos", "aa_ref", "aa_alt", "hmec_score"]].groupby(["aa_pos", "aa_ref", "aa_alt"], as_index=False).mean(),
                on=["aa_pos", "aa_ref", "aa_alt"], how="left")
    g.to_csv(OUT / "brca1_findlay2018_missense_protein.tsv", sep="\t", index=False)
    QC["brca1_protein_with_external_tracks"] = int(g.ext_n_tracks.notna().sum())
    QC["brca1_protein_with_dace_hap1_replicate"] = int(g.dace_hap1_score.notna().sum())
    QC["brca1_protein_with_hmec"] = int(g.hmec_score.notna().sum())
    rr = g[["score_r1", "score_r2"]].dropna()
    QC["findlay_protein_level_replicate_pearson"] = round(float(rr.corr().iloc[0, 1]), 4)
    QC["findlay_protein_level_replicate_spearman"] = round(float(rr.corr(method="spearman").iloc[0, 1]), 4)
    rep = g[["score", "dace_hap1_score"]].dropna()
    QC["findlay_vs_dace_hap1_spearman"] = round(float(rep.corr(method="spearman").iloc[0, 1]), 4)
    QC["findlay_vs_dace_hap1_n"] = int(len(rep))
    brca2_labels(long2, seqs)
    with open(OUT / "stage2_qc.json", "w") as fh:
        json.dump(QC, fh, indent=1, default=str)
    print(json.dumps(QC, indent=1, default=str))


if __name__ == "__main__":
    main()
