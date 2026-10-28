"""v3 (2026-10-26) optional wrapper: run a pipeline script with torch deterministic algorithms enabled.
Usage: python det_run_v3.py script.py [args...]. Not used for the reported numbers."""
import runpy
import sys

import torch

torch.use_deterministic_algorithms(True)
sys.argv = sys.argv[1:]
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))
runpy.run_path(sys.argv[0], run_name="__main__")
