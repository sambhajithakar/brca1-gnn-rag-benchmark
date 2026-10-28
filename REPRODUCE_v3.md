# Reproducing Project 3 (v3, 2026-10-26)

Requirements: Linux, Python 3.13, git, curl, sha256sum, ~3 GB free disk, network access to github.com,
raw.githubusercontent.com, storage.googleapis.com and pypi.org. CPU only (4 cores: about 1 hour).

```sh
python3.13 -m venv env && env/bin/pip install -r requirements.lock.v3.txt
export P3_EXT=$PWD/../p3_external            # any empty directory
sh scripts/fetch_inputs_v3.sh                 # clones pinned repositories, downloads inputs, verifies checksums
PY=$PWD/env/bin/python sh scripts/run_all_v3.sh > run.log 2>&1
```

The run stops at the first problem (missing input, checksum mismatch, wrong package version, wrong repository commit,
failed command, failed leakage test). Every input, source, licence and pin is listed in INPUTS_AND_ENVIRONMENT_v3.md.
Graph-network outputs can differ in the last decimals between runs because PyTorch CPU scatter operations are not
deterministic; set P3_DETERMINISTIC=1 to force deterministic algorithms (the reported numbers were produced without it).
