# Post hoc depth-matched control (written 2026-10-26T17:37:43Z, before any run)
Authorised by the researcher via decision card, 2026-10-26 17:36Z. Reason: audit finding F2. C-nograph (n_layers=0) is shallower than Model C (2–3 message-passing layers).
POST HOC. This is not part of plan 16A. It does not replace H1.

Control "gnn_C_depthctl":
- Identical to gnn_C_struct_annot: same inputs, head, layer grid [2, 3] chosen by the same residue-grouped inner CV, same seeds 0–4, same early stopping.
- Both adjacency matrices (sequence and spatial) are set to zero. Each layer therefore applies only W_self, so there is no information exchange between residues.

Comparisons, using the same residue-cluster bootstrap code as the primary analysis (evaluate.py, 2,000 resamples):
1. Model C − depthctl (graph effect at matched depth). Primary partition P0; also P1, P2, LODO, LOEO and random.
2. depthctl − C-nograph (effect of the extra depth alone).

Implementation and outputs:
- Implemented by wrapper run_depthctl.py, which imports the unchanged run_experiments.py and redirects outputs to this folder.
- No existing file is modified.
