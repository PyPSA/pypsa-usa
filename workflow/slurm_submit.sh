#!/usr/bin/env bash
# Snakemake --cluster submit wrapper. Site-agnostic: carries no account,
# partition or mail address of its own.
#
#   slurm_submit.sh <rule> <threads> <mem_mb> <walltime> <jobscript>
#
# Site settings come from Slurm's own input environment variables, which
# sbatch reads whenever the matching flag is absent from the command line:
#
#   SBATCH_ACCOUNT     e.g. def-<pi> on Alliance clusters (Fir, Nibi, ...);
#                      leave unset on Sherlock, which has no accounts
#   SBATCH_PARTITION   e.g. serc on Sherlock; leave unset where the scheduler
#                      routes jobs by size (Alliance clusters)
#   SBATCH_MAIL_USER   address for job mail; SBATCH_MAIL_TYPE defaults to FAIL
#                      here when a mail user is set
#   SBATCH_QOS, SBATCH_CONSTRAINT, ...   anything else `man sbatch` lists
#
# Export them once (e.g. in ~/.bashrc); the driver's environment is inherited
# here, so they apply to every rule job.
#
# Also rounds the float mem_mb some rules produce
# (e.g. `(input.size // 150000) * attempt * 1.5` in build_electricity.smk);
# `sbatch --mem 4500.0` is rejected. Prints the job id (--parsable) for
# snakemake and slurm_status.sh to track.
set -euo pipefail

rule="$1"; threads="$2"; mem_raw="$3"; walltime="$4"; jobscript="$5"

# Round up to whole MB; floor at 2000 MB so a tiny input.size can't request ~0.
mem_mb=$(awk -v m="$mem_raw" 'BEGIN{ v=int(m); if (v<m) v++; if (v<2000) v=2000; print v }')

# Guard against an empty/garbage walltime reaching sbatch.
case "$walltime" in
  *[0-9]*:*) ;;
  *) walltime="02:00:00" ;;
esac

if [ -n "${SBATCH_MAIL_USER:-}" ]; then
  export SBATCH_MAIL_TYPE="${SBATCH_MAIL_TYPE:-FAIL}"
fi

mkdir -p logs/slurm

exec sbatch \
  --parsable \
  -J "$rule" \
  -o "logs/slurm/${rule}-%j.out" \
  -e "logs/slurm/${rule}-%j.err" \
  -c "$threads" \
  --mem "${mem_mb}" \
  --time "$walltime" \
  "$jobscript"
