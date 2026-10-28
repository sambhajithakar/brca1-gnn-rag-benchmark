"""Stage 4b: residue graphs and per-residue node features.

For each structure (1JM7 model 1, 1T29) stores:
  nodes.tsv            node table (structure, chain, resnum, aa, features)
  dist_heavy.npy       pairwise minimum heavy-atom distance (A)
  dist_ca.npy          pairwise C-alpha distance (A)
  mpnn_logp.npy        ProteinMPNN conditional log-probabilities (nodes x 21), mean of 10 noise draws
Graph variants are generated at training time from these matrices (see graph_variants()).
Also writes graph_qc.json. No functional label is read here.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

ROOT = Path(__file__).resolve().parents[1]
SP = ROOT / "17_Protein_Structure_Processing/out"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
ALPH = "ACDEFGHIKLMNPQRSTVWYX"


def build(pdb):
    feats = pd.read_csv(SP / "residue_features.tsv", sep="\t")
    feats = feats[feats.structure == pdb].reset_index(drop=True)
    s = PDBParser(QUIET=True).get_structure(pdb, SP / f"clean_{pdb}.pdb")[0]
    res = {}
    for ch in s:
        for r in ch:
            if r.id[0] == " " or r.get_resname() == "SEP":
                res[(ch.id, r.id[1])] = r
    keys = list(zip(feats.chain, feats.resnum))
    heavy = [np.array([a.coord for a in res[k]]) for k in keys]
    ca = np.array([res[k]["CA"].coord for k in keys])
    n = len(keys)
    dh = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.min(np.linalg.norm(heavy[i][:, None] - heavy[j][None], axis=-1))
            dh[i, j] = dh[j, i] = d
    dca = np.linalg.norm(ca[:, None] - ca[None], axis=-1).astype(np.float32)
    npz = np.load(SP / f"mpnn/{pdb}/conditional_probs_only/clean_{pdb}.npz")
    lp = npz["log_p"].mean(0)
    seq_mpnn = "".join(ALPH[i] for i in npz["S"])
    # ProteinMPNN orders designed chain A first, then visible chain B, residues in file order.
    order = [k for k in keys if k[0] == "A"] + [k for k in keys if k[0] != "A"]
    idx = {k: i for i, k in enumerate(order)}
    lp_nodes = np.stack([lp[idx[k]] for k in keys])
    seq_nodes = "".join(seq_mpnn[idx[k]] for k in keys)
    match = np.mean([a == b or b == "X" for a, b in zip(feats.aa, seq_nodes)])
    p = np.exp(lp_nodes[:, :20]); p = p / p.sum(1, keepdims=True)
    feats["mpnn_entropy"] = -(p * np.log(p + 1e-9)).sum(1)
    feats["mpnn_lp_native"] = [lp_nodes[i, ALPH.index(a)] if a in ALPH[:20] else np.nan for i, a in enumerate(feats.aa)]
    d = OUT / pdb; d.mkdir(exist_ok=True)
    feats.to_csv(d / "nodes.tsv", sep="\t", index=False)
    np.save(d / "dist_heavy.npy", dh); np.save(d / "dist_ca.npy", dca); np.save(d / "mpnn_logp.npy", lp_nodes)
    qc = {"nodes": n, "chainA_nodes": int((feats.chain == "A").sum()), "mpnn_sequence_match": float(match)}
    for name, adj in graph_variants(feats, dh, dca).items():
        deg = adj.sum(1)
        qc[name] = {"edges_undirected": int(adj.sum() // 2), "mean_degree": float(deg.mean()), "min_degree": int(deg.min()),
                    "isolated_nodes": int((deg == 0).sum()), "n_components": int(components(adj))}
    return qc


def graph_variants(feats, dh, dca):
    n = len(feats)
    chain = feats.chain.to_numpy(dtype=object); rn = feats.resnum.to_numpy()
    seq = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if chain[i] == chain[j] and abs(rn[i] - rn[j]) == 1: seq[i, j] = True
    nonseq = ~seq & ~np.eye(n, dtype=bool)
    partner = chain[:, None] != chain[None, :]
    v = {}
    for c in (6, 8, 10):
        v[f"heavy{c}_with_partner"] = seq | (nonseq & (dh < c))
    v["heavy8_no_partner"] = (seq | (nonseq & (dh < 8))) & ~partner
    v["ca8_with_partner"] = seq | (nonseq & (dca < 8))
    v["sequence_only"] = seq.copy()
    return {k: a.astype(np.int8) for k, a in v.items()}


def components(adj):
    n = len(adj); seen = np.zeros(n, bool); c = 0
    for s in range(n):
        if seen[s]: continue
        c += 1; stack = [s]; seen[s] = True
        while stack:
            u = stack.pop()
            for w in np.nonzero(adj[u])[0]:
                if not seen[w]: seen[w] = True; stack.append(w)
    return c


if __name__ == "__main__":
    qc = {p: build(p) for p in ("1jm7", "1t29")}
    json.dump(qc, open(OUT / "graph_qc.json", "w"), indent=1)
    print(json.dumps(qc, indent=1))
