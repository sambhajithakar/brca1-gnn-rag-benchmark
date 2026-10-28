#!/bin/sh
# v3 (2026-10-26). Re-creates every raw input of Project 3 from its public source at a pinned version, then verifies
# data/SHA256SUMS.v3.txt. Also clones the two external code repositories needed at run time.
# Usage: P3_EXT=/path/for/clones sh scripts/fetch_inputs_v3.sh      (run from anywhere; writes into this checkout)
# Network needed: github.com, raw.githubusercontent.com, storage.googleapis.com.
set -eu
R=$(cd "$(dirname "$0")/.." && pwd)
: "${P3_EXT:?P3_EXT must be set to a directory for external clones}"
mkdir -p "$P3_EXT"
die() { echo "FETCH ERROR: $*" >&2; exit 1; }
clone() {  # url dir commit
  if [ ! -d "$2/.git" ]; then git clone --quiet "$1" "$2" || die "clone failed: $1"; fi
  git -C "$2" checkout --quiet "$3" || die "commit $3 not found in $1"
  [ "$(git -C "$2" rev-parse HEAD)" = "$3" ] || die "wrong commit in $2"
}
clone https://github.com/dms-view/BRCA1 "$P3_EXT/dmsview_BRCA1" 81d1dce3472d0efdaffc0b7233fe091b3875caaf
clone https://github.com/phoebedace/BRCA1_SGE_HAP1_HMEC "$P3_EXT/BRCA1_SGE_HAP1_HMEC" 37fc69b72d583f271052f70f340d2b3862420f5a
clone https://github.com/FunctionalAssaYIntegration/BRCA1-BRCA2-Large-integration "$P3_EXT/FunctionalAssaYIntegration_BRCA1-BRCA2-Large-integration" 908fb3760358cd6a7e8f3438b2185bd29004f9aa
clone https://github.com/dauparas/ProteinMPNN "$P3_EXT/ProteinMPNN" 8907e6671bfbfc92303b5f79c4b5e6ce47cdef57

D=$R/data/raw
mkdir -p "$D/Findlay2018_dmsview_mirror" "$D/dace2025_repo/amplicon_fasta" "$D/lyra2026_repo" "$D/alphamissense" "$D/proteingym_ref"
S=$P3_EXT/dmsview_BRCA1/data/Findlay2018
for f in 1jm7.pdb 1t29.pdb Findlay2018_brctdomain.csv Findlay2018_brctdomain.md Findlay2018_ringdomain.csv Findlay2018_ringdomain.md Findlay2018_suppTable1.csv README.md build_data.ipynb; do
  cp "$S/$f" "$D/Findlay2018_dmsview_mirror/$f"
done
git -C "$P3_EXT/dmsview_BRCA1" log -1 --format='%H %ci' > "$D/Findlay2018_dmsview_mirror/MIRROR_COMMIT.txt"
S=$P3_EXT/BRCA1_SGE_HAP1_HMEC
for f in 230420_BRCA1_hg19_hg38_conversion.csv 240913_clinvar.txt 241024_EVE_scores_BRCA1_HUMAN.csv 250703_SpliceAI_scores_500bp_all_vars.csv amplicon_coords_hg19.txt; do
  cp "$S/input_files/$f" "$D/dace2025_repo/$f"
done
cp "$S/data_for_plots/Findlay_et_al_2018_data.csv" "$S/data_for_plots/SUPPLEMENTARY_TABLES_1-8.xlsx" "$D/dace2025_repo/"
cp "$S/LICENSE" "$D/dace2025_repo/LICENSE"
cp "$S"/input_files/fasta/*.fa "$D/dace2025_repo/amplicon_fasta/"
git -C "$S" log -1 --format='%H %ci' > "$D/dace2025_repo/SOURCE_COMMIT.txt"
S=$P3_EXT/FunctionalAssaYIntegration_BRCA1-BRCA2-Large-integration
cp "$S/data/SUPP_TABLES_BRCA12_APR_2026.xlsx" "$S/CITATION.cff" "$S/checksums.sha256" "$D/lyra2026_repo/"
cp "$S/data/LICENSE" "$D/lyra2026_repo/LICENSE"
git -C "$S" log -1 --format='%H %ci' > "$D/lyra2026_repo/SOURCE_COMMIT.txt"
curl -fsSL -o "$D/proteingym_ref/DMS_substitutions.csv" \
  https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/144fe22b07dfaeec2b366f2346203a9838a55b4c/reference_files/DMS_substitutions.csv || die "ProteinGym download"
# AlphaMissense (Cheng et al. 2023; CC BY-NC-SA 4.0): stream the official file and keep BRCA1 (P38398) and BRCA2 (P51587) rows.
if ! sha256sum -c --quiet --status - <<SUM 2>/dev/null
3fdaa6ca033382351a06aa7331ef313d897a5e5d43ff0f71d0ebd5df08e25be2  $D/alphamissense/AlphaMissense_aa_substitutions_BRCA1_BRCA2.tsv
SUM
then
  curl -fsSL https://storage.googleapis.com/dm_alphamissense/AlphaMissense_aa_substitutions.tsv.gz | zcat \
    | awk 'NR<=4 || $1=="P38398" || $1=="P51587"' > "$D/alphamissense/AlphaMissense_aa_substitutions_BRCA1_BRCA2.tsv.part" || die "AlphaMissense stream"
  mv "$D/alphamissense/AlphaMissense_aa_substitutions_BRCA1_BRCA2.tsv.part" "$D/alphamissense/AlphaMissense_aa_substitutions_BRCA1_BRCA2.tsv"
fi
cd "$R" && sha256sum -c --quiet data/SHA256SUMS.v3.txt || die "checksum mismatch after fetch"
echo "FETCH OK: all $(wc -l < "$R/data/SHA256SUMS.v3.txt") inputs verified"
