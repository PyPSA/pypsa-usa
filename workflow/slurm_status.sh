#!/usr/bin/env bash
# Snakemake --cluster-status hook. Prints exactly one of: success | failed | running
#
#   slurm_status.sh <jobid>
#
# Without it, snakemake 7 waits for marker files the jobscript writes when it
# exits. A job Slurm kills (OUT_OF_MEMORY, TIMEOUT, NODE_FAIL, preemption)
# never writes them, so the driver hangs forever.
set -uo pipefail

jobid="${1%%;*}"   # --parsable may print "<jobid>;<cluster>"

state=$(sacct -j "$jobid" -X --noheader --parsable2 --format=State 2>/dev/null | head -n1)
state="${state%% *}"  # "CANCELLED by 1234" -> CANCELLED

if [ -z "$state" ]; then
  # sacct lags just after submission, and some sites run without accounting.
  # Fall back to the live queue.
  state=$(squeue -h -j "$jobid" -o %T 2>/dev/null | head -n1)
  if [ -z "$state" ]; then
    # Unknown to both: finished and already out of the queue. Report success;
    # snakemake then checks the rule's outputs and fails the job if they are
    # missing, so a silent failure still surfaces.
    echo success
    exit 0
  fi
fi

case "$state" in
  COMPLETED)
    echo success ;;
  PENDING|CONFIGURING|RUNNING|COMPLETING|SUSPENDED|REQUEUED|REQUEUE_HOLD|REQUEUE_FED|RESIZING|SIGNALING|STAGE_OUT|STOPPED)
    echo running ;;
  *)
    # FAILED, TIMEOUT, OUT_OF_MEMORY, CANCELLED, NODE_FAIL, PREEMPTED,
    # BOOT_FAIL, DEADLINE, ...
    echo failed ;;
esac
