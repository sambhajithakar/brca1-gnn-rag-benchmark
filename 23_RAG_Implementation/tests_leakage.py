"""Automated leakage tests for the Scenario 1 retriever (run before any Model D training; exits non-zero on failure).

T1  The Scenario 1 index contains no Findlay-derived document (findlay_result docs, track T127).
T2  For every outer fold of every partition, no retrieved document sits at a test residue, at the target's own residue,
    or describes the target variant; no Findlay-derived document is retrieved.
T3  Positive control: a deliberately leaky configuration (all residues allowed, own residue not excluded) must be
    detected by the same check. A test that cannot fail is not a test.
T4  The retrieved-evidence features of a test variant are unchanged when every outcome-bearing document at test
    residues is deleted from the corpus (they never depended on it).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evidence_retriever import StructRetriever, leakage_check  # noqa: E402

ROOT = HERE.parents[0]


class LeakyRetriever(StructRetriever):
    def retrieve(self, pos, allowed):
        out = []
        for q, d in list(self.neighbours(pos).items()) + [(pos, 0.0)]:
            if q not in self.docs_by_pos: continue
            g = self.docs_by_pos[q].copy(); g["dist"] = d; out.append(g)
        return pd.concat(out) if out else self.C.iloc[:0].assign(dist=[])


def main():
    T = pd.read_csv(ROOT / "data/processed/model_table_brca1.tsv", sep="\t")
    R = StructRetriever(T)
    fails = []
    # T1
    n_f = int(R.C.findlay_derived.sum()) + int((R.C.source_id == "BRCA1:T127").sum()) + int((R.C.doc_type == "findlay_result").sum())
    print("T1 Findlay-derived docs in Scenario 1 index:", n_f); fails += ["T1"] if n_f else []
    # T2
    for part in ("fold_P0", "fold_P1", "fold_P2", "fold_LODO", "fold_LOEO", "fold_random"):
        for k in sorted(T[part].dropna().unique()):
            te = np.nonzero((T[part] == k).to_numpy())[0]; tr = np.nonzero((T[part] != k).to_numpy())[0]
            v = leakage_check(T, R, tr, te)
            bad = {x: y for x, y in v.items() if x != "n_retrievals" and y}
            if bad: fails.append(f"T2 {part} fold {k}: {bad}")
        print("T2", part, "checked")
    # T3
    L = LeakyRetriever(T)
    te = np.nonzero((T.fold_P0 == 0).to_numpy())[0]; tr = np.nonzero((T.fold_P0 != 0).to_numpy())[0]
    v = leakage_check(T, L, tr, te)
    detected = v["test_residue_docs"] > 0 and v["own_residue_docs"] > 0 and v["same_variant_docs"] > 0
    print("T3 leaky configuration detected:", detected, v); fails += [] if detected else ["T3"]
    # T4
    allowed = np.setdiff1d(T.aa_pos.iloc[tr].unique(), T.aa_pos.iloc[te].unique())
    F1 = R.features(T.iloc[te], allowed)
    R2 = StructRetriever(T)
    te_pos = set(T.aa_pos.iloc[te])
    R2.C = R2.C[~R2.C.aa_pos.isin(te_pos)]
    R2.docs_by_pos = {p: g for p, g in R2.C.groupby("aa_pos")}
    F2 = R2.features(T.iloc[te], allowed)
    same = np.allclose(np.nan_to_num(F1, nan=-9), np.nan_to_num(F2, nan=-9))
    print("T4 features invariant to deleting test-residue outcome docs:", same); fails += [] if same else ["T4"]
    if fails:
        print("FAILED:", fails); sys.exit(1)
    print("ALL LEAKAGE TESTS PASSED")


if __name__ == "__main__":
    main()
