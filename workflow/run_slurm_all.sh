#!/usr/bin/env bash
# =============================================================================
# PyPSA-USA — launch one Slurm controller per config
# =============================================================================
# CH2's run_slurm_all.sh, with two changes:
#
#   * each config gets its own SLURM CONTROLLER JOB (sbatch workflow/
#     run_slurm.sbatch), not a backgrounded snakemake process on the login
#     node. Nothing here runs a scheduler outside a job.
#   * job ordering uses --dependency=afterok instead of watch_and_replot.sh,
#     which polled a controller log every 300 s for the literal string
#     "49 of 49 steps (100%) done" and hard-coded the step count.
#
# The experiment matrix stays what CH2 made it: an array of configfiles you
# comment in and out, never a code change.
#
#   bash workflow/run_slurm_all.sh                 # submit the array below
#   bash workflow/run_slurm_all.sh cfg1.yaml cfg2.yaml   # or an explicit list
#
#   PYPSA_CHAIN=1   run the controllers back to back (afterok chain) instead of
#                   concurrently; use when they share a cold data/ cache, since
#                   concurrent retrieve_* rules race on the same download.
#   PYPSA_REPORT=1  submit workflow/run_report.sbatch afterok on all of them.
#   PYPSA_DRYRUN=1  print the sbatch commands and exit.
#   PYPSA_CTRL_ARGS extra sbatch flags for every controller, e.g.
#                   "--time=12:00:00 --mem=16G".
#   PYPSA_SMK_ARGS  extra snakemake args/targets appended to every controller.
# =============================================================================
set -euo pipefail

# --- The matrix --------------------------------------------------------------
# Every entry must have a DISTINCT run.name; that is the invariant --nolock
# rests on and it is checked below. Paths are relative to workflow/.
CONFIGS=(
  "repo_data/config/config.tutorial.yaml"
  # "repo_data/config/config.california.yaml"
  # "repo_data/config/config.default.yaml"
  # "config/config.usa.benchmark.yaml"
  # "config/config.usa.benchmark.REM.yaml"
)

[ "$#" -gt 0 ] && CONFIGS=("$@")
if [ "${#CONFIGS[@]}" -eq 0 ]; then
    echo "ERROR: no configs. Uncomment entries in the CONFIGS array above, or" >&2
    echo "       pass configfiles as arguments." >&2
    exit 1
fi

# --- Where am I? -------------------------------------------------------------
REPO="${PYPSA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[ -f "$REPO/workflow/Snakefile" ] || { echo "ERROR: $REPO is not a pypsa-usa checkout" >&2; exit 1; }
cd "$REPO"

# sbatch does not create -o/-e directories; run_slurm.sbatch writes its own
# tee'd copy here too.
mkdir -p logs/controller

# --- run.name must differ across the matrix ----------------------------------
# A config that does not set run.name inherits it from config.default.yaml, so
# two such configs collide even though neither mentions a name. Report that as
# the collision it is rather than discovering it as clobbered results.
run_name() {
    awk '
        /^run:[[:space:]]*$/ { inrun = 1; next }
        /^[^[:space:]#]/     { inrun = 0 }
        inrun && $1 == "name:" {
            v = $2
            # Two subs, not one gsub with an alternation: POSIX awk matches
            # leftmost-LONGEST, so a single quote-stripping alternation would
            # match the whole value and delete it. (No apostrophes in this
            # comment: it lives inside a single-quoted awk program.)
            sub(/^["'"'"']/, "", v)
            sub(/["'"'"'].*$/, "", v)
            print v
            exit
        }
    ' "$1"
}

declare -a NAMES=()
for cfg in "${CONFIGS[@]}"; do
    case "$cfg" in
        /*) path="$cfg" ;;
        *)  path="$REPO/workflow/$cfg"
            [ -f "$path" ] || path="$REPO/$cfg" ;;
    esac
    [ -f "$path" ] || { echo "ERROR: no such configfile: $cfg" >&2; exit 1; }
    n="$(run_name "$path")"
    NAMES+=("${n:-<inherited-from-config.default.yaml>}")
done

dupes="$(printf '%s\n' "${NAMES[@]}" | sort | uniq -d)"
if [ -n "$dupes" ]; then
    echo "ERROR: these run.name values appear more than once in the matrix:" >&2
    printf '  %s\n' $dupes >&2
    echo "       Concurrent controllers with the same run.name write to the same" >&2
    echo "       resources/ and results/ paths. Give each config its own run.name," >&2
    echo "       or serialise them with PYPSA_CHAIN=1." >&2
    [ "${PYPSA_CHAIN:-0}" = "1" ] || exit 1
    echo "       PYPSA_CHAIN=1: serialising instead of failing." >&2
fi

# Concurrent controllers share one working directory, so snakemake's lock has
# to come off -- safe exactly because the names above are distinct. A chained
# run never has two schedulers live at once and keeps the lock.
if [ "${PYPSA_CHAIN:-0}" = "1" ]; then
    export PYPSA_NOLOCK=0
else
    export PYPSA_NOLOCK=1
fi

# --- Submit ------------------------------------------------------------------
submit() {
    if [ "${PYPSA_DRYRUN:-0}" = "1" ]; then
        echo "DRYRUN: $*" >&2
        echo "000000"
        return
    fi
    sbatch --parsable "$@"
}

# Flags are plain strings, not arrays: Sherlock runs bash 4.2 (CentOS 7), where
# an EMPTY array expanded under `set -u` is an "unbound variable" abort, and
# `${arr[@]+...}` inside a double-quoted string is a parse error there too.
# None of these values contain spaces, so unquoted expansion is exactly right.
IDS=()
PREV=""
for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    tag="$(basename "$cfg" .yaml)"
    DEP=""
    if [ "${PYPSA_CHAIN:-0}" = "1" ] && [ -n "$PREV" ]; then
        DEP="--dependency=afterok:$PREV"
    fi
    # shellcheck disable=SC2086
    jid="$(submit --job-name="smk-$tag" ${DEP:-} ${PYPSA_CTRL_ARGS:-} \
              workflow/run_slurm.sbatch "$cfg" ${PYPSA_SMK_ARGS:-})"
    echo "submitted $jid  $cfg  (run.name=${NAMES[$i]}) ${DEP:-}"
    IDS+=("$jid")
    PREV="$jid"
done

# --- Chain the report --------------------------------------------------------
if [ "${PYPSA_REPORT:-0}" = "1" ]; then
    deps="$(IFS=:; echo "${IDS[*]}")"
    rid="$(submit --job-name=pypsa-report --dependency="afterok:$deps" \
              ${PYPSA_REPORT_ARGS:-} workflow/run_report.sbatch)"
    echo "submitted $rid  run_report.sbatch  (afterok:$deps)"
fi

echo
echo "controllers: ${IDS[*]}"
echo "watch with:  squeue --me      (poll no more than once every 2 minutes)"
echo "logs:        logs/controller/ctrl-<jobid>.log"
