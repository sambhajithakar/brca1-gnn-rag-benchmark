"""Stage 4a: per-residue structural features from experimental BRCA1 structures.

Inputs : data/raw/Findlay2018_dmsview_mirror/1jm7.pdb (NMR, 14 models, BRCA1 RING chain A + BARD1 chain B)
         data/raw/Findlay2018_dmsview_mirror/1t29.pdb (X-ray, BRCA1 BRCT chain A + BACH1 phosphopeptide chain B)
Outputs: 17_Protein_Structure_Processing/out/residue_features.tsv (one row per residue node, all chains)
         17_Protein_Structure_Processing/out/clean_*.pdb (heavy atoms, first model) for ProteinMPNN
         17_Protein_Structure_Processing/out/structure_qc.json

Features are computed from coordinates only; no functional label is read here.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pydssp
from Bio.PDB import PDBIO, PDBParser, Select, Superimposer
from Bio.PDB.SASA import ShrakeRupley

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/Findlay2018_dmsview_mirror"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
THREE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H",
         "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
         "TYR": "Y", "VAL": "V", "SEP": "S"}
# Theoretical maximum ASA (Tien et al. 2013, empirical)
MAXASA = {"A": 121, "R": 265, "N": 187, "D": 187, "C": 148, "Q": 214, "E": 214, "G": 97, "H": 216, "I": 195, "L": 191,
          "K": 230, "M": 203, "F": 228, "P": 154, "S": 143, "T": 163, "W": 264, "Y": 255, "V": 165}
# Domain intervals from the Lyra et al. 2026 repository (src/domains.py, Apache-2.0)
def brca1_domain(pos):
    if pos <= 96: return "RING"
    if pos <= 110: return "RING_flank"
    if 1650 <= pos <= 1736: return "BRCT1"
    if 1737 <= pos <= 1764: return "BRCT_linker"
    if 1765 <= pos <= 1863: return "BRCT2"
    return "other"


class HeavyFirst(Select):
    def accept_model(self, m): return m.id == 0
    def accept_atom(self, a): return a.element != "H" and a.element != "D"
    def accept_residue(self, r): return r.id[0] == " " or r.get_resname() in ("SEP", "ZN")


def residues(chain):
    return [r for r in chain if r.id[0] == " " or r.get_resname() == "SEP"]


def process(pdb_id, partner_label):
    s = PDBParser(QUIET=True).get_structure(pdb_id, RAW / f"{pdb_id}.pdb")
    io = PDBIO(); io.set_structure(s); io.save(str(OUT / f"clean_{pdb_id}.pdb"), HeavyFirst())
    s1 = PDBParser(QUIET=True).get_structure(pdb_id, OUT / f"clean_{pdb_id}.pdb")
    m = s1[0]
    ShrakeRupley().compute(m, level="R")
    rows = []
    allres = []
    for ch in m:
        for r in residues(ch):
            allres.append((ch.id, r))
    # DSSP (pydssp, 3-state) per chain, backbone N, CA, C, O
    ss = {}
    for ch in m:
        rs = residues(ch)
        try:
            coord = np.array([[r[a].coord for a in ("N", "CA", "C", "O")] for r in rs], dtype=float)
            lab = pydssp.assign(coord, out_type="c3")
            for r, l in zip(rs, lab): ss[(ch.id, r.id[1])] = l
        except KeyError:
            for r in rs: ss[(ch.id, r.id[1])] = "-"
    # Zinc ligands: residues with a side-chain S/N/O atom within 3.0 A of a Zn atom
    zn = [a for a in m.get_atoms() if a.element == "ZN" or a.get_parent().get_resname() == "ZN"]
    heavy = {(c, r.id[1]): np.array([a.coord for a in r]) for c, r in allres}
    # NMR flexibility: C-alpha RMSF after superposition on model 1 (1JM7 only)
    rmsf = {}
    if len(s) > 1:
        ref = {(c.id, r.id[1]): r["CA"] for c in s[0] for r in residues(c) if "CA" in r}
        keys = sorted(ref)
        coords = []
        for mod in s:
            cur = {(c.id, r.id[1]): r["CA"] for c in mod for r in residues(c) if "CA" in r}
            sup = Superimposer(); sup.set_atoms([ref[k] for k in keys], [cur[k] for k in keys])
            rot, tran = sup.rotran
            coords.append(np.array([np.dot(cur[k].coord, rot) + tran for k in keys]))
        coords = np.stack(coords)
        dev = np.sqrt(((coords - coords.mean(0)) ** 2).sum(-1).mean(0))
        rmsf = dict(zip(keys, dev))
    # B-factors (X-ray only)
    bfac = {}
    if len(s) == 1:
        b = {(c, r.id[1]): np.mean([a.bfactor for a in r]) for c, r in allres}
        v = np.array(list(b.values())); bfac = {k: (x - v.mean()) / v.std() for k, x in b.items()}
    for c, r in allres:
        aa = THREE.get(r.get_resname(), "X")
        key = (c, r.id[1])
        my = heavy[key]
        n_contacts = 0; partner_min = np.inf
        for c2, r2 in allres:
            if (c2, r2.id[1]) == key: continue
            dmin = np.min(np.linalg.norm(my[:, None, :] - heavy[(c2, r2.id[1])][None, :, :], axis=-1))
            if dmin < 8.0 and abs(r2.id[1] - r.id[1]) > 1 or (dmin < 8.0 and c2 != c): n_contacts += 1
            if c2 != c: partner_min = min(partner_min, dmin)
        zn_lig = 0
        for z in zn:
            if any(np.linalg.norm(a.coord - z.coord) < 3.0 for a in r if a.element in ("S", "N", "O") and a.get_id() not in ("N", "O")):
                zn_lig = 1
        rows.append(dict(structure=pdb_id, chain=c, resnum=r.id[1], aa=aa,
                         protein=("BRCA1" if c == "A" else partner_label),
                         rsa=min(1.0, r.sasa / MAXASA.get(aa, 200)), ss3=ss.get(key, "-"),
                         n_contacts_8A=n_contacts, partner_min_dist=partner_min,
                         interface=int(partner_min < 5.0), zn_ligand=zn_lig,
                         ca_rmsf_nmr=rmsf.get(key, np.nan), bfactor_z=bfac.get(key, np.nan),
                         domain=(brca1_domain(r.id[1]) if c == "A" else "partner")))
    return pd.DataFrame(rows), {"models": len(s), "chains": {ch.id: len(residues(ch)) for ch in s[0]}, "zn_atoms_model1": len(zn)}


def main():
    a, qa = process("1jm7", "BARD1")
    b, qb = process("1t29", "BACH1_peptide")
    df = pd.concat([a, b], ignore_index=True)
    df.to_csv(OUT / "residue_features.tsv", sep="\t", index=False)
    qc = {"1jm7": qa, "1t29": qb,
          "zn_ligands_1jm7": a[(a.zn_ligand == 1)][["chain", "resnum", "aa"]].values.tolist(),
          "interface_residues_brca1_1jm7": int(a[(a.chain == "A")].interface.sum()),
          "interface_residues_brca1_1t29": int(b[(b.chain == "A")].interface.sum()),
          "ss3_counts_brca1": df[df.chain == "A"].ss3.value_counts().to_dict()}
    json.dump(qc, open(OUT / "structure_qc.json", "w"), indent=1, default=str)
    print(json.dumps(qc, indent=1, default=str))


if __name__ == "__main__":
    main()
