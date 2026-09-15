#!/usr/bin/env bash
# Snakemake 7 --cluster-status hook.
#
#   slurm-status.sh <slurm-job-id>   ->  prints exactly one of
#                                        success | failed | running
#
# Snakemake calls this once per poll per live job with the id that `sbatch
# --parsable` printed. Without it, a job the scheduler kills (OUT_OF_MEMORY,
# TIMEOUT, NODE_FAIL, PREEMPTED, or a plain scancel) never writes the
# exit-code marker Snakemake otherwise waits on, and the controller blocks
# until its own wall time. That is what CH2 hit on 2026-08-26: a controller
# killed mid-solve, children left running, and a repair sbatch written by hand
# to re-solve the two networks that were deleted.
#
# It is also the precondition for `restart-times: 2` in config.yaml doing
# anything: Snakemake can only re-run a job with attempt=2 (and therefore
# ask Slurm for 2x the memory, in the 17 rules whose mem_mb is a function of
# `attempt`) once it knows the first attempt failed.
#
# Only ever prints one of the three words; anything else makes Snakemake abort
# the whole run with "unknown job status".
set -uo pipefail

jobid="${1:-}"
if [[ -z "$jobid" ]]; then
    echo "failed"
    exit 0
fi

# `sbatch --parsable` prints "<jobid>" normally and "<jobid>;<clustername>" on
# a federated cluster. Keep the numeric id only.
jobid="${jobid%%;*}"

# `command` bypasses shell functions and aliases: some site profiles export a
# caching wrapper called `sacct`, and a wrapper that swallows or reformats the
# output would make every job look failed.
#
# -X: the allocation's own row, not the .batch / .extern steps, which report
# their own states and would make a COMPLETED job look CANCELLED.
# --parsable2 so a trailing delimiter does not become part of the state.
#
# A Slurm state is an uppercase token, optionally truncated with a trailing
# "+". Anything else on stdout is an error message -- "slurm_load_jobs error:
# Invalid job id specified", "Socket timed out on send/recv operation" -- and
# must NOT be read as a state, or a transient slurmctld hiccup kills a healthy
# job. Discard it and let the caller poll again.
state_of() {
    local raw="${1%% *}"
    [[ "$raw" =~ ^[A-Z_]+\+?$ ]] && printf '%s' "$raw"
}

state="$(state_of "$(command sacct -j "$jobid" -X --format=State --noheader --parsable2 2>/dev/null \
                     | head -n1)")"

# sacct lags a few seconds behind submission on a busy slurmdbd; squeue knows
# about the job immediately.
if [[ -z "$state" ]]; then
    state="$(state_of "$(command squeue -j "$jobid" -h -o %T 2>/dev/null | head -n1)")"
fi

case "$state" in
    COMPLETED)
        echo "success"
        ;;
    ""|PENDING|RUNNING|SUSPENDED|COMPLETING|CONFIGURING|REQUEUED|REQUEUE_HOLD|REQUEUE_FED|RESIZING|SIGNALING|STAGE_OUT|RESV_DEL_HOLD|SPECIAL_EXIT)
        # An empty state means neither accounting nor the queue has caught up
        # yet. Report "running": Snakemake polls again, and Sherlock's sacct
        # retains completed jobs, so this cannot loop forever in practice.
        echo "running"
        ;;
    *)
        # FAILED CANCELLED TIMEOUT OUT_OF_MEMORY NODE_FAIL PREEMPTED
        # BOOT_FAIL DEADLINE OUT_OF_ME+ (sacct truncates) and anything new.
        echo "failed"
        ;;
esac
