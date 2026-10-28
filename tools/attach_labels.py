"""Re-attach the Findlay function score to the deposited prediction and interval tables.

The deposit omits the Findlay `score` column (third-party data). After `scripts/fetch_inputs_v3.sh` and the
dataset step of `scripts/run_all_v3.sh` have rebuilt data/processed/brca1_findlay2018_missense_protein.tsv from
the original, checksum-verified source, run from the repository root:

    python tools/attach_labels.py

Each table regains `score` in its original column position, so the verification scripts run unchanged.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LAB = pd.read_csv(ROOT / "data/processed/brca1_findlay2018_missense_protein.tsv", sep="\t", dtype=str, keep_default_na=False).set_index("hgvs_p")["score"]
TABLES = (list((ROOT / "results/predictions").rglob("*.tsv"))
          + list((ROOT / "60_final_stage/depth_control/results/predictions").rglob("*.tsv"))
          + list((ROOT / "26_Uncertainty_Calibration/out").glob("*_intervals.tsv")))

n = 0
for f in TABLES:
    t = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False)
    if "score" in t.columns:
        continue
    s = t.hgvs_p.map(LAB)
    assert s.notna().all() and (s != "").all(), f"{f}: {int(s.isna().sum())} variants without a Findlay score"
    t.insert(t.columns.get_loc("hgvs_p") + 1, "score", s)
    t.to_csv(f, sep="\t", index=False, lineterminator="\n"); n += 1
print(f"re-attached score to {n} tables")
