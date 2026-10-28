#!/bin/sh
# v3 (2026-10-26): create a clean checkout containing exactly the files in MANIFEST_v3.txt (no data, no outputs).
# Usage: sh scripts/make_clean_checkout_v3.sh /path/to/new/empty/dir
set -eu
R=$(cd "$(dirname "$0")/.." && pwd)
T=${1:?target directory}
[ ! -e "$T" ] || { echo "target exists: $T" >&2; exit 1; }
mkdir -p "$T"
while IFS= read -r f; do
  [ -f "$R/$f" ] || { echo "manifest file missing: $f" >&2; exit 1; }
  mkdir -p "$T/$(dirname "$f")"; cp -p "$R/$f" "$T/$f"
done < "$R/MANIFEST_v3.txt"
echo "clean checkout: $(wc -l < "$R/MANIFEST_v3.txt") files in $T"
