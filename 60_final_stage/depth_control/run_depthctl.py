"""Post hoc depth-matched control (see SPEC.md). Imports the unchanged pipeline; writes only under this folder."""
import sys
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "20_GNN_Model_Source"))
import run_experiments as R  # noqa: E402

R.RES = HERE / "results"
_orig = R.load_graphs


def zero_adj_graphs(graph_variant="heavy8_with_partner", with_annot=False):
    return [(x, torch.zeros_like(a), torch.zeros_like(s)) for x, a, s in _orig(graph_variant, with_annot)]


R.load_graphs = zero_adj_graphs
if __name__ == "__main__":
    for part in sys.argv[1:]:
        R.run(part, "all", "struct", ["gnn_C_struct_annot"], "heavy8_with_partner", "_depthctl")
