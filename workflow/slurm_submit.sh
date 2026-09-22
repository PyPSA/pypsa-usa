#!/usr/bin/env bash
# Snakemake --cluster submit wrapper.
#
#   slurm_submit.sh <rule> <threads> <mem_mb> <walltime> <jobscript>
#
# Exists because several rules compute mem_mb as a float
# (e.g. `(input.size // 150000) * attempt * 1.5` in build_electricity.smk), and
# `sbatch --mem 4500.0` is rejected. This rounds up to an integer MB and floors
# it at a usable minimum. Prints the job id (--parsable) for snakemake to track.
set -euo pipefail

rule="$1"; threads="$2"; mem_raw="$3"; walltime="$4"; jobscript="$5"

# Round up to whole MB; floor at 2000 MB so a tiny input.size can't request ~0.
mem_mb=$(awk -v m="$mem_raw" 'BEGIN{ v=int(m); if (v<m) v++; if (v<2000) v=2000; print v }')

# Guard against an empty/garbage walltime reaching sbatch.
case "$walltime" in
  *[0-9]*:*) ;;
  *) walltime="02:00:00" ;;
esac

exec sbatch \
  --parsable \
  -p "${PARTITION:-serc}" \
  --mail-type FAIL \
  --mail-user "${EMAIL:-ctehran@stanford.edu}" \
  -J "$rule" \
  -o "logs/slurm/${rule}-%j.out" \
  -e "logs/slurm/${rule}-%j.err" \
  -c "$threads" \
  --mem "${mem_mb}" \
  --time "$walltime" \
  "$jobscript"
