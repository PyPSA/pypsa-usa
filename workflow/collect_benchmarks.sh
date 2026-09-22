#!/usr/bin/env bash
# Per-rule resource report for tuning Slurm requests.
#
#   bash collect_benchmarks.sh [SINCE]        # SINCE defaults to now-6hours
#
# sacct splits the data we need across two rows per job: the allocation row
# carries JobName/ReqMem, the .batch row carries MaxRSS. This joins them on the
# base job id and aggregates per rule, so `sacct -X` (which shows MaxRSS blank)
# is never the thing you look at.
set -uo pipefail

SINCE="${1:-now-6hours}"

sacct -S "$SINCE" -P --format=JobID,JobName,State,Elapsed,MaxRSS,ReqMem 2>/dev/null |
awk -F'|' '
  NR == 1 { next }
  {
    split($1, a, ".")
    id = a[1]
    if ($1 !~ /\./) { name[id] = $2; state[id] = $3; elapsed[id] = $4; req[id] = $6 }
    else if ($1 ~ /\.batch$/) { gsub(/K$/, "", $5); rss[id] = $5 + 0 }
  }
  END {
    for (id in name) {
      n = name[id]
      if (n == "batch" || n == "extern" || n == "" ) continue
      if (n ~ /^(bash|uv|python|snakemake|ca2040_probe)$/) continue
      cnt[n]++
      sum_rss[n] += rss[id]
      if (rss[id] > max_rss[n]) max_rss[n] = rss[id]
      split(elapsed[id], t, ":")
      secs = t[1] * 3600 + t[2] * 60 + t[3]
      if (secs > max_secs[n]) max_secs[n] = secs
      reqm[n] = req[id]
      if (state[id] !~ /COMPLETED/) bad[n] = bad[n] " " state[id]
    }
    printf "%-34s %5s %11s %11s %10s %8s  %s\n", "RULE", "N", "MAXRSS_MB", "PEAK_ELAPSED", "REQ_MEM", "RATIO", "NONCOMPLETE"
    for (n in cnt) {
      mb = max_rss[n] / 1024
      r = reqm[n]; gsub(/[MGn]$/, "", r)
      if (reqm[n] ~ /G/) r = r * 1024
      ratio = (mb > 0 && r > 0) ? sprintf("%.1fx", r / mb) : "-"
      printf "%-34s %5d %11.0f %11s %10s %8s  %s\n", \
        n, cnt[n], mb, \
        sprintf("%d:%02d:%02d", max_secs[n]/3600, (max_secs[n]%3600)/60, max_secs[n]%60), \
        reqm[n], ratio, bad[n]
    }
  }
' | { read -r hdr; echo "$hdr"; sort -k3 -nr; }
