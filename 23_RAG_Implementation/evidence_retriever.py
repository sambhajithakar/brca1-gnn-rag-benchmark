"""Scenario 1 (strict prediction): structure-aware, leakage-controlled evidence retrieval for Model D.

For a target variant at residue i in outer fold k, the retriever may only return documents that
  (1) are not derived from Findlay et al. 2018 (findlay_derived == False; the primary label source), and
  (2) if they carry an outcome (assay result or ClinVar classification), sit at a residue that belongs to the
      training residues of fold k and is not residue i itself (leave-own-residue-out also for training rows).
Test-fold residues therefore contribute no outcome-bearing document to any prediction in that fold, and no
document about the target variant (from any source) can be retrieved.

Retrieval is structure-aware: candidate residues are those within RADIUS (minimum heavy-atom distance, same PDB
structure) of the target residue, plus sequence neighbours within SEQ_WIN. Documents are weighted by 1/(1+d).

The retrieved evidence is summarised into numeric features (no free text enters the model):
  ev_assay_lof      weighted fraction of loss-of-function calls among retrieved assay_result documents
  ev_assay_lof_mt   same, restricted to documents with the same mutant amino acid (falls back to ev_assay_lof)
  ev_assay_n        log1p(number of retrieved assay_result documents)
  ev_cv_plp         weighted fraction of Pathogenic/Likely pathogenic among retrieved ClinVar documents
  ev_cv_blb         weighted fraction of Benign/Likely benign among retrieved ClinVar documents
  ev_cv_n           log1p(number of retrieved ClinVar documents with an assertion)
  ev_any            1 if any outcome-bearing document was retrieved
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "18_Residue_Graph_Construction/out"
CORPUS = Path(__file__).resolve().parent / "out/evidence_corpus.tsv"
RADIUS = 8.0
SEQ_WIN = 3
EV_COLS = ["ev_assay_lof", "ev_assay_lof_mt", "ev_assay_n", "ev_cv_plp", "ev_cv_blb", "ev_cv_n", "ev_any"]
PLP = {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"}
BLB = {"Benign", "Likely benign", "Benign/Likely benign"}


class StructRetriever:
    def __init__(self, T, corpus=CORPUS):
        C = pd.read_csv(corpus, sep="\t", low_memory=False)
        C = C[(C.gene == "BRCA1") & C.carries_outcome & ~C.findlay_derived & C.doc_type.isin(["assay_result", "clinvar"])].copy()
        C = C[(C.doc_type == "assay_result") | C.strength.isin(PLP | BLB)]
        C["aa_pos"] = C.aa_pos.astype(int)
        C["aa_alt"] = C.variant.str[-1]
        C["lof"] = (C.strength == "loss of function").astype(float)
        C["plp"] = C.strength.isin(PLP).astype(float)
        C["blb"] = C.strength.isin(BLB).astype(float)
        self.C = C
        self.docs_by_pos = {p: g for p, g in C.groupby("aa_pos")}
        # residue neighbourhoods (structure + sequence), with distances
        self.nbrs = {}
        for pdb in ("1jm7", "1t29"):
            n = pd.read_csv(G / pdb / "nodes.tsv", sep="\t")
            d = np.load(G / pdb / "dist_heavy.npy")
            a = np.nonzero((n.chain == "A").to_numpy())[0]
            res = n.resnum.to_numpy()
            for i in a:
                nb = {}
                for j in a:
                    if j == i: continue
                    if d[i, j] <= RADIUS: nb[int(res[j])] = float(d[i, j])
                    elif abs(int(res[j]) - int(res[i])) <= SEQ_WIN: nb[int(res[j])] = float(d[i, j])
                self.nbrs[int(res[i])] = nb
        self.T = T

    def neighbours(self, pos):
        if pos in self.nbrs:
            return self.nbrs[pos]
        return {pos + o: float(abs(o)) * 3.8 for o in range(-SEQ_WIN, SEQ_WIN + 1) if o != 0}  # sequence only

    def retrieve(self, pos, allowed):
        """Outcome-bearing documents for residue pos, restricted to allowed residues and excluding pos itself."""
        out = []
        for q, d in self.neighbours(pos).items():
            if q == pos or q not in allowed or q not in self.docs_by_pos: continue
            g = self.docs_by_pos[q].copy(); g["dist"] = d; out.append(g)
        return pd.concat(out) if out else self.C.iloc[:0].assign(dist=[])

    def features(self, rows, allowed, audit=None):
        allowed = set(int(p) for p in allowed)
        F = np.zeros((len(rows), len(EV_COLS)))
        cache = {}
        for k, (pos, alt) in enumerate(zip(rows.aa_pos.astype(int), rows.aa_alt)):
            if pos not in cache: cache[pos] = self.retrieve(pos, allowed)
            R = cache[pos]
            if audit is not None: audit.append((pos, alt, R.doc_id.tolist()))
            A = R[R.doc_type == "assay_result"]; V = R[R.doc_type == "clinvar"]
            def wmean(D, col):
                if len(D) == 0: return np.nan
                w = 1.0 / (1.0 + D.dist.to_numpy()); return float((w * D[col].to_numpy()).sum() / w.sum())
            lof = wmean(A, "lof"); Am = A[A.aa_alt == alt]
            F[k] = [lof, wmean(Am, "lof") if len(Am) else lof, np.log1p(len(A)), wmean(V, "plp"), wmean(V, "blb"), np.log1p(len(V)),
                    float(len(R) > 0)]
        return F


def leakage_check(T, retriever, train_idx, test_idx, findlay_doc_ids=None):
    """Automated test: returns a dict of violations (all must be zero)."""
    te_pos = set(T.aa_pos.iloc[test_idx].astype(int)); tr_pos = set(T.aa_pos.iloc[train_idx].astype(int)) - te_pos
    audit = []
    retriever.features(T.iloc[test_idx], tr_pos, audit)
    audit_tr = []
    retriever.features(T.iloc[train_idx], tr_pos, audit_tr)
    C = retriever.C.set_index("doc_id")
    v = {"test_residue_docs": 0, "own_residue_docs": 0, "findlay_docs": 0, "same_variant_docs": 0, "n_retrievals": 0}
    for pos, alt, ids in audit + audit_tr:
        if not ids: continue
        d = C.loc[ids]
        v["n_retrievals"] += len(ids)
        v["test_residue_docs"] += int(d.aa_pos.isin(te_pos).sum())
        v["own_residue_docs"] += int((d.aa_pos == pos).sum())
        v["findlay_docs"] += int(d.findlay_derived.sum())
        v["same_variant_docs"] += int(((d.aa_pos == pos) & (d.aa_alt == alt)).sum())
    return v
