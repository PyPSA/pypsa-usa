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
# Site settings (account, partition, mail) are SBATCH_* environment variables;
# see slurm_submit.sh. Tuning, all optional:
#   JOBS            rule-jobs in flight at once            (default 20)
#   RESTART_TIMES   retries per failed rule; rules that
#                   scale memory by `attempt` get more     (default 0)
#   LATENCY_WAIT    seconds to wait for outputs to appear
#                   on a shared filesystem                 (default 60)
#   SNAKEMAKE       snakemake command: an executable, or a
#                   launcher such as "uv run snakemake"    (default: repo .venv, else PATH)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

JOBS="${JOBS:-20}"
RESTART_TIMES="${RESTART_TIMES:-0}"
LATENCY_WAIT="${LATENCY_WAIT:-60}"
snakemake_cmd=()
if [ -n "${SNAKEMAKE:-}" ]; then
  if command -v "$SNAKEMAKE" > /dev/null; then
    snakemake_cmd=("$SNAKEMAKE")  # one executable; its path may hold spaces
  else
    read -r -a snakemake_cmd <<< "$SNAKEMAKE"
  fi
elif [ -x "$HERE/../.venv/bin/snakemake" ]; then
  # The venv binary directly rather than `uv run`, which would re-resolve the
  # environment on every invocation. See AGENTS.md for the uv setup.
  snakemake_cmd=("$HERE/../.venv/bin/snakemake")
elif command -v snakemake > /dev/null; then
  snakemake_cmd=("$(command -v snakemake)")
fi
[ "${#snakemake_cmd[@]}" -gt 0 ] || { echo "snakemake not found; activate the environment or set SNAKEMAKE" >&2; exit 2; }

# The Python that goes with it: next to the executable, or behind the same
# launcher ("uv run snakemake" -> "uv run python").
if [ "${#snakemake_cmd[@]}" -eq 1 ]; then
  python_cmd=("$(dirname "$(command -v "${snakemake_cmd[0]}" || echo "${snakemake_cmd[0]}")")/python")
else
  python_cmd=("${snakemake_cmd[@]:0:${#snakemake_cmd[@]}-1}" python)
fi

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
  _certifi=$("${python_cmd[@]}" -c 'import certifi; print(certifi.where())' 2>/dev/null || true)
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
exec "${snakemake_cmd[@]}" \
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
