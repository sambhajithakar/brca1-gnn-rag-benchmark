"""Scenario 2 (interpretation of already characterised variants): retrieval and citation-grounded synthesis.

This scenario is not a prediction benchmark. The index contains every document, including Findlay 2018 results and
ClinVar classifications for the query variant; the task is to retrieve the right evidence and summarise it without
error. No language model is used (no API key is available); synthesis is deterministic and template-based, so
every sentence is generated from one document's structured fields and cites that document.

Retrievers compared
  bm25        BM25Okapi over tokenised document text
  tfidf       TF-IDF cosine (word 1-2 grams)
  hybrid      reciprocal-rank fusion of bm25 and tfidf (k=60)
  hybrid_vf   hybrid followed by a variant/residue filter: variant-specific documents are kept only if their
              normalised variant equals the query; residue documents only if aa_pos equals the query residue

Evaluation (queries: a fixed random sample of BRCA1 missense variants with at least one external assay record,
plus a random sample without, seed 20261025)
  relevant set   documents whose variant equals the query, residue_structure documents for the query residue,
                 domain documents whose interval contains the residue
  precision@k, recall@k (k = 5, 10, 20), wrong-variant rate (share of retrieved variant-specific docs that
  are about another variant), citation correctness (share of cited docs that are relevant), unsupported-statement
  rate (share of synthesis sentences whose factual content cannot be re-derived from the cited document's fields),
  fidelity (Findlay class and assay calls in the synthesis equal the source values).
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
OUT = HERE / "out"
SEED = 20261025
AA3 = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
       "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V"}
AA1 = {v: k for k, v in AA3.items()}


def tok(s):
    s = str(s)
    for k, v in AA3.items():  # normalise 3-letter protein notation to 1-letter tokens, keep both
        s = re.sub(rf"p\.{k}(\d+)([A-Z][a-z]{{2}})", lambda m: f"p.{v}{m.group(1)}{AA3.get(m.group(2), m.group(2))} " + m.group(0), s)
    return re.findall(r"[a-z0-9]+", s.lower())


def load_corpus():
    C = pd.read_csv(OUT / "evidence_corpus.tsv", sep="\t", low_memory=False)
    C = C[C.gene.isin(["BRCA1", "BRCA1/BRCA2"])].reset_index(drop=True)
    C["variant"] = C.variant.fillna("")
    return C


class Retrievers:
    def __init__(self, C):
        self.C = C
        self.toks = [tok(t) for t in C.text]
        self.bm25 = BM25Okapi(self.toks)
        self.tfv = TfidfVectorizer(tokenizer=tok, lowercase=False, ngram_range=(1, 2), min_df=1, token_pattern=None)
        self.X = self.tfv.fit_transform(C.text)

    def rank(self, q, method, k=20, qv=None, qpos=None):
        if method == "bm25":
            s = self.bm25.get_scores(tok(q)); return np.argsort(-s)[:k]
        if method == "tfidf":
            s = (self.X @ self.tfv.transform([q]).T).toarray().ravel(); return np.argsort(-s)[:k]
        b = np.argsort(-self.bm25.get_scores(tok(q)))[:500]
        t = np.argsort(-(self.X @ self.tfv.transform([q]).T).toarray().ravel())[:500]
        rrf = {}
        for lst in (b, t):
            for r, i in enumerate(lst): rrf[i] = rrf.get(i, 0) + 1.0 / (60 + r + 1)
        order = sorted(rrf, key=lambda i: -rrf[i])
        if method == "hybrid":
            return np.array(order[:k])
        keep = []
        for i in order:
            d = self.C.iloc[i]
            if d.variant and d.variant != qv: continue
            if d.doc_type == "residue_structure" and int(d.aa_pos) != qpos: continue
            if d.doc_type == "domain" and not (d.range_start <= qpos <= d.range_end): continue
            keep.append(i)
            if len(keep) == k: break
        return np.array(keep)


def relevant(C, qv, qpos):
    m = (C.variant == qv) | ((C.doc_type == "residue_structure") & (C.aa_pos == qpos)) | \
        ((C.doc_type == "domain") & (C.range_start <= qpos) & (C.range_end >= qpos))
    return set(np.nonzero(m.to_numpy())[0])


def synthesise(C, idx, qv, model_pred=None):
    """One sentence per cited document, generated from structured fields only."""
    S = []
    for i in idx:
        d = C.iloc[i]
        if d.doc_type == "findlay_result":
            sc = re.search(r"function score (-?\d+\.\d+)", d.text).group(1)
            S.append(dict(doc=d.doc_id, claim=("findlay_class", d.strength), text=f"In saturation genome editing (Findlay 2018), {d.variant} scored {sc}, classed as {d.strength} [{d.doc_id}]."))
        elif d.doc_type == "assay_result":
            dup = " (this track re-codes the Findlay 2018 data and is not independent evidence)" if bool(d.findlay_derived) else ""
            S.append(dict(doc=d.doc_id, claim=("assay_call", d.strength), text=f"Assay {d.source_id.split(':')[1]} reported {d.variant} as {d.strength}{dup} [{d.doc_id}]."))
        elif d.doc_type == "clinvar":
            S.append(dict(doc=d.doc_id, claim=("clinvar", d.strength), text=f"ClinVar ({d.source_id}) lists {d.variant} as {d.strength} [{d.doc_id}]."))
        elif d.doc_type == "residue_structure":
            S.append(dict(doc=d.doc_id, claim=("structure", d.text.split(": ", 1)[1]), text=f"Structural context: {d.text.split(': ', 1)[1]} [{d.doc_id}]."))
        elif d.doc_type == "domain":
            S.append(dict(doc=d.doc_id, claim=("domain", d.text), text=f"{d.text} [{d.doc_id}]."))
        else:
            S.append(dict(doc=d.doc_id, claim=("other", d.doc_type), text=f"Related record: {d.source} [{d.doc_id}]."))
    if model_pred is not None:
        S.append(dict(doc="MODEL", claim=("model", model_pred), text=f"Model prediction (out-of-fold, not evidence): function score {model_pred:.2f}."))
    return S


def verify(C, S, qv, qpos):
    """Re-derive each sentence's claim from the cited document; count unsupported and unfaithful statements."""
    by_id = C.set_index("doc_id")
    unsupported = unfaithful = cite_ok = 0; n = 0
    for s in S:
        if s["doc"] == "MODEL": continue
        n += 1
        d = by_id.loc[s["doc"]]
        kind, val = s["claim"]
        ok_rel = (d.variant == qv) or (d.doc_type == "residue_structure" and int(d.aa_pos) == qpos) or \
                 (d.doc_type == "domain" and d.range_start <= qpos <= d.range_end)
        cite_ok += int(ok_rel)
        if kind in ("findlay_class", "assay_call", "clinvar"):
            if val != d.strength: unfaithful += 1
            if d.variant not in s["text"]: unsupported += 1
            if d.variant != qv: unfaithful += 1  # sentence asserts a result for a different variant than queried
        elif kind == "structure":
            if val not in d.text: unsupported += 1
    return dict(n_statements=n, unsupported=unsupported, unfaithful=unfaithful, citation_correct=cite_ok)


def main():
    C = load_corpus()
    R = Retrievers(C)
    T = pd.read_csv(ROOT / "data/processed/brca1_findlay2018_missense_protein.tsv", sep="\t")
    ext = set(C[(C.doc_type == "assay_result") & ~C.findlay_derived].variant)
    rng = np.random.default_rng(SEED)
    with_ext = T[T.hgvs_p.isin(ext)].sample(150, random_state=SEED)
    without = T[~T.hgvs_p.isin(ext)].sample(150, random_state=SEED)
    Q = pd.concat([with_ext.assign(stratum="external_evidence"), without.assign(stratum="findlay_only")])
    f = ROOT / "results/predictions/fold_P0/lgbm_seqevo__all.tsv"
    oof = pd.read_csv(f, sep="\t").set_index("hgvs_p").pred if f.exists() else pd.Series(dtype=float)
    rows = []; examples = []
    for _, q in Q.iterrows():
        qv, qpos = q.hgvs_p, int(q.aa_pos)
        query = f"BRCA1 {qv} p.{AA1[q.aa_ref]}{qpos}{AA1[q.aa_alt]} residue {qpos} functional evidence"
        rel = relevant(C, qv, qpos)
        for method in ("bm25", "tfidf", "hybrid", "hybrid_vf"):
            top = R.rank(query, method, 20, qv, qpos)
            r = dict(variant=qv, stratum=q.stratum, method=method, n_relevant=len(rel))
            for k in (5, 10, 20):
                hit = len(set(top[:k]) & rel)
                r[f"p@{k}"] = hit / k; r[f"r@{k}"] = hit / len(rel) if rel else np.nan
            vs = [C.variant.iat[i] for i in top[:10] if C.variant.iat[i]]
            r["wrong_variant_rate@10"] = np.mean([v != qv for v in vs]) if vs else 0.0
            S = synthesise(C, top[:10], qv, oof.get(qv))
            v = verify(C, S, qv, qpos)
            r["citation_correctness"] = v["citation_correct"] / v["n_statements"]
            r["unsupported_rate"] = v["unsupported"] / v["n_statements"]
            r["unfaithful_rate"] = v["unfaithful"] / v["n_statements"]
            rows.append(r)
            if len(examples) < 8 and method in ("hybrid", "hybrid_vf") and q.stratum == "external_evidence":
                examples.append(dict(variant=qv, method=method, synthesis=[s["text"] for s in S], verification=v))
    M = pd.DataFrame(rows)
    M.to_csv(OUT / "scenario2_per_query.tsv", sep="\t", index=False)
    summ = M.groupby(["stratum", "method"]).mean(numeric_only=True).round(3)
    summ.to_csv(OUT / "scenario2_summary.tsv", sep="\t")
    json.dump(examples, open(OUT / "scenario2_examples.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    print(summ.to_string())


if __name__ == "__main__":
    main()
