#!/usr/bin/env bash
# RETIRED 2026-09-14 — replaced by workflow/run_slurm.sbatch.
#
# This script ran the Snakemake scheduler ON THE LOGIN NODE (a multi-day
# process on a shared front end, against a `--configfile` path that does not
# exist in this repository) and resolved partition/account/email through
# `--cluster-config`, which is deprecated in Snakemake 7 and removed in 8.
# Its replacement puts the scheduler inside a Slurm job, takes the config as
# an argument, and keeps every site setting in one tracked profile.
#
# See docs/hpc.md for the three run modes and which to pick.
cat >&2 <<'MSG'
workflow/run_slurm.sh is retired. Use instead:

    mkdir -p logs/controller
    sbatch workflow/run_slurm.sbatch <configfile> [targets...]     # per-rule jobs
    sbatch -c 16 --mem=160G --time=24:00:00 \
        workflow/run_slurm.sbatch --local <configfile>             # one node
    bash  workflow/run_slurm_all.sh                                # config matrix

Profile: workflow/snakemake_profiles/sherlock/   Docs: docs/hpc.md
MSG
exit 1
