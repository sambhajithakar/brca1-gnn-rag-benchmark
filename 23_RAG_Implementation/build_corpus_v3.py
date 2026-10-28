"""Stage 7a: evidence corpus with provenance. (v3, 2026-10-26: reads the literature table from data/inputs/; optional fixed timestamp.)

Every document records: doc_id, doc_type, gene, aa_pos (residue, if residue-specific), variant (p. notation, if
variant-specific), carries_outcome (does the text state a functional or clinical outcome), findlay_derived,
source, source_id, evidence_date (publication or evaluation date), retrieval_timestamp, strength, text.

Document types
  assay_result          variant-level call from an independent assay track (Lyra et al. 2026)   carries_outcome
  assay_method          assay description and thresholds (no variant outcome)
  clinvar               ClinVar germline classification (snapshot 2024-09-13)                  carries_outcome
  findlay_result        Findlay 2018 SGE score/class for a variant (Scenario 2 only)            carries_outcome, findlay_derived
  domain                domain interval annotation (Lyra et al. repository constants)
  residue_structure     per-residue structural context computed from PDB coordinates
  literature            verified study-level summary from 09_Prior_Art_Comparison.xlsx
"""
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
# v3: P3_BUILD_TIMESTAMP (if set) fixes the timestamp so the corpus is byte-reproducible.
NOW = os.environ.get("P3_BUILD_TIMESTAMP") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
AA1 = {"A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys", "Q": "Gln", "E": "Glu", "G": "Gly", "H": "His", "I": "Ile",
       "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe", "P": "Pro", "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val"}
CALL = {0: "normal function", 1: "intermediate function", 2: "loss of function"}


def year(ref):
    m = re.search(r"(19|20)\d\d", str(ref))
    return m.group(0) if m else ""


def p3(r, p, a):
    return f"p.{AA1.get(r, r)}{int(p)}{AA1.get(a, a)}"


def docs_assays(gene):
    long = pd.read_csv(P / f"{gene.lower()}_lyra2026_track_calls_long.tsv", sep="\t")
    meta = pd.read_csv(P / f"{gene.lower()}_lyra2026_track_metadata.tsv", sep="\t")
    D = []
    for _, t in meta.iterrows():
        if int(t["Track #"][1:]) < 8: continue
        D.append(dict(doc_type="assay_method", gene=gene, aa_pos=np.nan, variant="", carries_outcome=False,
                      findlay_derived=(gene == "BRCA1" and t["Track #"] == "T127"), source="Lyra et al. 2026 track metadata",
                      source_id=f"{gene}:{t['Track #']}", evidence_date=year(t.Reference), strength="",
                      text=f"{gene} functional assay {t['Track #']}: {t.Track}. Reference: {t.Reference}. Assay class: {t['Assay class']}; "
                           f"set-up: {t['Set-up']}; host: {t.Host}; species: {t.Species}; read-out: {t['Read-out']}. Thresholds: {t.Notes}"))
    for _, r in long.iterrows():
        if gene == "BRCA1" and r.track == "T159":  # failed coding QC (14_Functional_Label_Validity_Report)
            continue
        if int(r.call) not in CALL:  # one BRCA2 T39 row carries an undocumented code 3; excluded
            continue
        D.append(dict(doc_type="assay_result", gene=gene, aa_pos=int(r.aa_pos), variant=f"p.{r.aa_ref}{int(r.aa_pos)}{r.aa_alt}",
                      carries_outcome=True, findlay_derived=(gene == "BRCA1" and r.track == "T127"),
                      source=str(r.Reference), source_id=f"{gene}:{r.track}", evidence_date=year(r.Reference),
                      strength=CALL[int(r.call)],
                      text=f"{gene} {p3(r.aa_ref, r.aa_pos, r.aa_alt)} ({r.hgvs_p}) tested in {r.Track} [{r['Assay class']}; host {r.Host}]. "
                           f"Result: {CALL[int(r.call)]}."))
    return D


def docs_clinvar():
    c = pd.read_csv(ROOT / "data/raw/dace2025_repo/240913_clinvar.txt", sep="\t")
    c = c[c["Protein change"].notna() & c.Name.str.contains(r"\(p\.", regex=True)]
    D = []
    for _, r in c.iterrows():
        m = re.search(r"\(p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)", r.Name)
        if not m: continue
        inv = {v: k for k, v in AA1.items()}
        if m.group(1) not in inv or m.group(3) not in inv: continue
        v = f"p.{inv[m.group(1)]}{m.group(2)}{inv[m.group(3)]}"
        D.append(dict(doc_type="clinvar", gene="BRCA1", aa_pos=int(m.group(2)), variant=v, carries_outcome=True, findlay_derived=False,
                      source="ClinVar (export 2024-09-13)", source_id=str(r.Accession), evidence_date=str(r["Germline date last evaluated"]),
                      strength=str(r["Germline classification"]),
                      text=f"ClinVar {r.Accession}: {r.Name}. Germline classification: {r['Germline classification']} "
                           f"(review status: {r['Germline review status']}; last evaluated {r['Germline date last evaluated']})."))
    return D


def docs_findlay():
    t = pd.read_csv(P / "brca1_findlay2018_missense_protein.tsv", sep="\t")
    D = []
    for _, r in t.iterrows():
        cls = "loss of function" if r.score < -1.328 else ("normal function" if r.score > -0.748 else "intermediate")
        D.append(dict(doc_type="findlay_result", gene="BRCA1", aa_pos=int(r.aa_pos), variant=r.hgvs_p, carries_outcome=True,
                      findlay_derived=True, source="Findlay et al. Nature 2018;562:217-222", source_id="doi:10.1038/s41586-018-0461-z",
                      evidence_date="2018", strength=cls,
                      text=f"BRCA1 {p3(r.aa_ref, r.aa_pos, r.aa_alt)} ({r.hgvs_p}) saturation genome editing in HAP1 cells (Findlay 2018): "
                           f"function score {r.score:.2f} ({cls})."))
    return D


def docs_structure():
    D = []
    dom = [("BRCA1", "RING / zinc finger region", 1, 96), ("BRCA1", "BRCT1 domain", 1650, 1736), ("BRCA1", "BRCT inter-repeat linker", 1737, 1764),
           ("BRCA1", "BRCT2 domain", 1765, 1863), ("BRCA1", "coiled-coil (PALB2-binding) region", 1364, 1437)]
    for g, name, a, b in dom:
        D.append(dict(doc_type="domain", gene=g, aa_pos=np.nan, variant="", carries_outcome=False, findlay_derived=False,
                      source="Lyra et al. 2026 repository src/domains.py", source_id=f"{g}:{a}-{b}", evidence_date="2026", strength="",
                      text=f"{g} residues {a}-{b} form the {name}.", range_start=a, range_end=b))
    for pdb, partner in (("1jm7", "BARD1 RING"), ("1t29", "BACH1 phosphopeptide")):
        n = pd.read_csv(ROOT / f"18_Residue_Graph_Construction/out/{pdb}/nodes.tsv", sep="\t")
        n = n[n.chain == "A"]
        for _, r in n.iterrows():
            ss = {"H": "helix", "E": "strand", "-": "loop"}[r.ss3]
            bits = [f"relative solvent accessibility {r.rsa:.2f} ({'buried' if r.rsa < 0.2 else 'exposed'})", f"secondary structure {ss}",
                    f"{int(r.n_contacts_8A)} residue contacts within 8 A"]
            if r.zn_ligand: bits.append("zinc-coordinating residue")
            if r.interface: bits.append(f"at the interface with {partner}")
            D.append(dict(doc_type="residue_structure", gene="BRCA1", aa_pos=int(r.resnum), variant="", carries_outcome=False, findlay_derived=False,
                          source=f"PDB {pdb.upper()} (computed)", source_id=f"{pdb.upper()}:A:{int(r.resnum)}", evidence_date={"1jm7": "2001", "1t29": "2004"}[pdb],
                          strength="", text=f"BRCA1 residue {AA1[r.aa]}{int(r.resnum)} in PDB {pdb.upper()}: " + "; ".join(bits) + "."))
    return D


def docs_literature():
    t = pd.read_excel(ROOT / "data/inputs/09_Prior_Art_Comparison.xlsx")
    D = []
    for _, r in t.iterrows():
        D.append(dict(doc_type="literature", gene="BRCA1/BRCA2", aa_pos=np.nan, variant="", carries_outcome=False,
                      findlay_derived=(r.id == "P01"), source=f"{r.study_title} ({r.venue})", source_id=str(r.doi), evidence_date=str(r.pub_date)[:4],
                      strength="", text=f"{r.study_title}. {r.first_last_author}, {r.venue}, {r.pub_date}. Data: {r.dataset}. Task: {r.prediction_task}. "
                                        f"Model: {r.model_architecture}. Limitations: {r.limitations}."))
    return D


def main():
    D = docs_assays("BRCA1") + docs_assays("BRCA2") + docs_clinvar() + docs_findlay() + docs_structure() + docs_literature()
    C = pd.DataFrame(D)
    C.insert(0, "doc_id", [f"D{i:06d}" for i in range(len(C))])
    C["retrieval_timestamp"] = NOW
    C.to_csv(OUT / "evidence_corpus.tsv", sep="\t", index=False)
    s = C.groupby(["doc_type", "gene"]).size().to_dict()
    json.dump({"n_docs": len(C), "by_type": {f"{k[0]}|{k[1]}": v for k, v in s.items()}, "built": NOW}, open(OUT / "corpus_summary.json", "w"), indent=1)
    print(len(C), s)


if __name__ == "__main__":
    main()
