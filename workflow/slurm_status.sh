#!/usr/bin/env bash
# Snakemake --cluster-status hook. Prints exactly one of: success | failed | running
#
#   slurm_status.sh <jobid>
#
# Without it, snakemake 7 waits for marker files the jobscript writes when it
# exits. A job Slurm kills (OUT_OF_MEMORY, TIMEOUT, NODE_FAIL, preemption)
# never writes them, so the driver hangs forever.
#
# sacct is asked first and the controller (squeue) only when sacct cannot
# settle it. `running` is the answer whenever the state is not known for
# certain: snakemake asks again on the next poll, whereas a wrong `success` or
# `failed` makes it check outputs or resubmit while the job is still alive.
set -uo pipefail

jobid="${1%%;*}"   # --parsable may print "<jobid>;<cluster>"

# Ask the controller for the job's state. Sets $live to that state, or to
#   GONE     the controller no longer holds the job (finished and purged)
#   UNKNOWN  the controller did not answer (timeout, restart, ...)
# `-t all` so a job that finished moments ago still reports its final state.
live_state() {
  local out rc
  out=$(squeue -h -j "$jobid" -t all -o %T 2>&1)
  rc=$?
  if [ "$rc" -eq 0 ]; then
    # Skip anything that is not a bare state name (e.g. a warning on stderr).
    live=$(printf '%s\n' "$out" | grep -m1 -E '^[A-Z_]+$' || true)
    live="${live:-GONE}"
  elif printf '%s' "$out" | grep -qi 'invalid job id'; then
    live=GONE
  else
    live=UNKNOWN
  fi
}

state=$(sacct -j "$jobid" -X --noheader --parsable2 --format=State 2>/dev/null | head -n1)
state="${state%% *}"  # "CANCELLED by 1234" -> CANCELLED

case "$state" in
  ""|PREEMPTED|NODE_FAIL)
    # Empty: sacct lags just after submission, can be briefly unreachable, and
    # some sites run without accounting. PREEMPTED / NODE_FAIL: final only
    # where Slurm does not requeue; where it does, the job is back in the
    # queue and will run again under the same id. The controller knows which.
    live_state
    case "$live" in
      UNKNOWN)
        echo running
        exit 0 ;;
      GONE)
        if [ -z "$state" ]; then
          # Unknown to sacct and purged from the controller: it finished.
          # Report success; snakemake then checks the rule's outputs and fails
          # the job if they are missing, so a silent failure still surfaces.
          echo success
          exit 0
        fi ;;  # else keep sacct's PREEMPTED / NODE_FAIL: it was not requeued
      *)
        state="$live" ;;
    esac ;;
esac

case "$state" in
  COMPLETED)
    echo success ;;
  PENDING|CONFIGURING|RUNNING|COMPLETING|SUSPENDED|REQUEUED|REQUEUE_HOLD|REQUEUE_FED|RESIZING|SIGNALING|STAGE_OUT|STOPPED)
    echo running ;;
  *)
    # FAILED, TIMEOUT, OUT_OF_MEMORY, CANCELLED, BOOT_FAIL, DEADLINE, and
    # NODE_FAIL / PREEMPTED that the controller confirmed as final.
    echo failed ;;
esac
