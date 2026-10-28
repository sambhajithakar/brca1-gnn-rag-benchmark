"""v3 (2026-10-26) preflight for run_all_v3.sh. Exits non-zero, with a message, on the first missing or wrong item:
Python version, installed packages versus requirements.lock.v3.txt, importability of every package the pipeline uses,
environment variables, external repository commits, and presence of every input listed in data/SHA256SUMS.v3.txt
(checksums themselves are verified by run_all_v3.sh with sha256sum -c)."""
import importlib
import os
import subprocess
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1]
def fail(msg):
    print("PREFLIGHT FAILED:", msg, file=sys.stderr); sys.exit(1)

if sys.version_info[:2] != (3, 13):
    fail(f"Python 3.13 required, found {sys.version.split()[0]}")
lock = {l.split("==")[0].lower().replace("_", "-"): l.split("==")[1] for l in (R / "requirements.lock.v3.txt").read_text().split() if "==" in l}
from importlib.metadata import distributions
have = {d.metadata["Name"].lower().replace("_", "-"): d.version for d in distributions()}
bad = [f"{k}=={v} (installed {have.get(k, 'missing')})" for k, v in lock.items() if have.get(k) != v]
if bad:
    fail("packages differ from requirements.lock.v3.txt: " + "; ".join(bad[:10]))
for mod in ("numpy", "pandas", "scipy", "sklearn", "lightgbm", "torch", "Bio", "pydssp", "einops", "tqdm", "rank_bm25", "statsmodels", "openpyxl", "matplotlib"):
    try:
        importlib.import_module(mod)
    except Exception as e:  # noqa: BLE001
        fail(f"cannot import {mod}: {e}")
ext = os.environ.get("P3_EXT")
if not ext:
    fail("environment variable P3_EXT (directory with external clones) is not set")
pins = {"ProteinMPNN": "8907e6671bfbfc92303b5f79c4b5e6ce47cdef57",
        "FunctionalAssaYIntegration_BRCA1-BRCA2-Large-integration": "908fb3760358cd6a7e8f3438b2185bd29004f9aa"}
for d, c in pins.items():
    p = Path(ext) / d
    if not (p / ".git").is_dir():
        fail(f"missing external repository {p} (run scripts/fetch_inputs_v3.sh)")
    head = subprocess.run(["git", "-C", str(p), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != c:
        fail(f"{d} is at {head}, expected {c}")
if not (Path(ext) / "ProteinMPNN/vanilla_model_weights/v_48_020.pt").is_file():
    fail("ProteinMPNN weights v_48_020.pt missing")
missing = [l.split(None, 1)[1] for l in (R / "data/SHA256SUMS.v3.txt").read_text().splitlines() if not (R / l.split(None, 1)[1]).is_file()]
if missing:
    fail(f"{len(missing)} input files missing, e.g. {missing[:3]} (run scripts/fetch_inputs_v3.sh)")
for tool in ("sha256sum", "git"):
    if subprocess.run(["which", tool], capture_output=True).returncode:
        fail(f"required tool not found: {tool}")
print(f"PREFLIGHT OK: Python {sys.version.split()[0]}, {len(lock)} pinned packages, external repos at pinned commits, all inputs present")
