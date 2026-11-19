#!/usr/bin/env bash
# Run PyPSA-USA on a Slurm cluster, submitting each snakemake rule as its own
# Slurm job. Every argument is passed to snakemake unchanged:
#
#   bash run_slurm.sh --configfile config/config.my_scenario.yaml
#   bash run_slurm.sh solve_all_networks --configfile config/config.my_scenario.yaml -n
#
# Put any target BEFORE --configfile: --configfile takes several values and
# would swallow a target written after it as another config file.
#
# This process is the snakemake scheduler and lives for the whole run. Run it
# inside a job, never on a login node, e.g.
#
#   sbatch --time=2-00:00:00 -c 2 --mem=8G -o logs/driver-%j.out \
#       --wrap "bash run_slurm.sh --configfile config/config.my_scenario.yaml"
#
# Site settings (account, partition, mail) are Slurm's own SBATCH_* variables;
# see slurm_submit.sh. Tuning, all optional:
#   JOBS            rule-jobs in flight at once            (default 20)
#   RESTART_TIMES   retries per failed rule; rules that
#                   scale memory by `attempt` get more     (default 0)
#   LATENCY_WAIT    seconds to wait for outputs to appear
#                   on a shared filesystem                 (default 60)
#   SNAKEMAKE       snakemake executable                   (default: repo .venv, else PATH)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

JOBS="${JOBS:-20}"
RESTART_TIMES="${RESTART_TIMES:-0}"
LATENCY_WAIT="${LATENCY_WAIT:-60}"
if [ -z "${SNAKEMAKE:-}" ]; then
  if [ -x "$HERE/../.venv/bin/snakemake" ]; then
    # The venv binary directly rather than `uv run`, which would re-resolve the
    # environment on every invocation. See AGENTS.md for the uv setup.
    SNAKEMAKE="$HERE/../.venv/bin/snakemake"
  else
    SNAKEMAKE="$(command -v snakemake || true)"
  fi
fi
[ -n "$SNAKEMAKE" ] || { echo "snakemake not found; activate the environment or set SNAKEMAKE" >&2; exit 2; }

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "WARNING: not inside a Slurm job. The scheduler runs for the whole workflow;" >&2
  echo "         most sites forbid that on a login node. See the header of this script." >&2
fi

# The uv-managed CPython ships no CA bundle -- ssl.get_default_verify_paths()
# returns cafile=None -- so every retrieve_* rule that downloads over HTTPS dies
# with CERTIFICATE_VERIFY_FAILED. (curl works, because it reads the system
# store, which is why a curl reachability test does not catch this.) Point
# Python at certifi. sbatch propagates the environment to the rule jobs.
if [ -z "${SSL_CERT_FILE:-}" ]; then
  _certifi=$("$(dirname "$SNAKEMAKE")/python" -c 'import certifi; print(certifi.where())' 2>/dev/null || true)
  if [ -n "$_certifi" ]; then
    export SSL_CERT_FILE="$_certifi" REQUESTS_CA_BUNDLE="$_certifi"
  else
    echo "WARNING: SSL_CERT_FILE unset; HTTPS retrieve rules may fail" >&2
  fi
fi

mkdir -p logs/slurm

# Our flags first, ending on a plain switch (--printshellcmds) so that
# --default-resources, which takes several values, cannot swallow the user's
# first argument; user arguments last, so they override ours.
exec "$SNAKEMAKE" \
  --cluster "bash slurm_submit.sh {rule} {threads} {resources.mem_mb} {resources.walltime}" \
  --cluster-status "bash slurm_status.sh" \
  --cluster-cancel scancel \
  --max-status-checks-per-second 1 \
  --jobs "$JOBS" \
  --latency-wait "$LATENCY_WAIT" \
  --restart-times "$RESTART_TIMES" \
  --rerun-incomplete \
  --default-resources "mem_mb=8000" "walltime='02:00:00'" \
  --printshellcmds \
  "$@"
