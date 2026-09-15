#!/usr/bin/env bash
# =============================================================================
# PyPSA-USA — one CSV of per-job wall time and peak memory
# =============================================================================
# CH2's collect_solve_statistics.sh, retargeted: it harvested three CSVs out of
# benchmarks/ and the Gurobi logs; this one emits a SINGLE tidy CSV that merges
# the two independent measurements of the same jobs --
#
#   source=benchmark : snakemake's own `benchmark:` TSVs under workflow/
#                      benchmarks/ (max_rss = peak RSS of the rule process
#                      tree, sampled by snakemake itself)
#   source=sacct     : Slurm accounting for the child jobs the sherlock profile
#                      submitted (MaxRSS, Elapsed, ReqMem, State -- the only
#                      place an OOM kill or a TIMEOUT is visible at all)
#
# This is the memory-profiling input the project asks for, and it is the reason
# the per-rule ("cluster") executor exists: a monolithic run reports ONE peak
# for the whole DAG, which cannot tell you where the memory went.
#
# Pure text parsing plus sacct, so it is safe on a login node.
#
#   bash workflow/collect_solve_statistics.sh [outfile.csv]
#
#   PYPSA_BENCHMARKS  workflow/benchmarks      where the TSVs are
#   PYPSA_SACCT_SINCE now-7days                sacct -S
#   PYPSA_JOB_TAG     (unset = every smk- job) restrict sacct to one controller
# =============================================================================
set -euo pipefail

REPO="${PYPSA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BENCH="${PYPSA_BENCHMARKS:-$REPO/workflow/benchmarks}"
OUT="${1:-$REPO/workflow/results/solve_statistics.csv}"
SINCE="${PYPSA_SACCT_SINCE:-now-7days}"
TAG="${PYPSA_JOB_TAG:-}"

mkdir -p "$(dirname "$OUT")"

HEADER="source,rule,run,stem,jobid,state,wall_s,wall_hms,max_rss_mb,cpu_time_s,req_mem,partition"
echo "$HEADER" > "$OUT"

# --- 1. The rule list, so a path component can be recognised as a rule name ---
RULES="$(grep -hoE '^rule[[:space:]]+[A-Za-z0-9_]+' \
            "$REPO/workflow/Snakefile" "$REPO/workflow/rules/"*.smk 2>/dev/null \
         | awk '{print $2}' | sort -u)"

# --- 2. benchmarks/**  -------------------------------------------------------
# Layouts in the wild: benchmarks/<rule>/<interconnect>/<stem> and
# benchmarks/<run>/<rule>/<interconnect>/<stem>. Recover the rule by finding
# the path component that IS a rule name; fall back to the longest rule name
# the stem starts with, which is how report_benchmarks.py does it.
n_bench=0
if [ -d "$BENCH" ]; then
    while IFS= read -r f; do
        rel="${f#"$BENCH"/}"
        rule=""
        IFS='/' read -ra parts <<< "$rel"
        for p in "${parts[@]}"; do
            if printf '%s\n' "$RULES" | grep -qx -- "$p"; then rule="$p"; fi
        done
        stem="$(basename "$f")"
        if [ -z "$rule" ]; then
            # Tier 2, then tier 3: the longest rule name the stem starts with,
            # then the longest one it contains anywhere (stems carry wildcard
            # debris on both sides, e.g. elec_s300_add_electricity). A stem
            # that matches neither stays "unknown" rather than being guessed
            # at -- workflow/report_benchmarks.py is the richer mapper.
            rule="$(printf '%s\n' "$RULES" \
                    | awk -v s="$stem" 'index(s, $0) == 1 { if (length($0) > length(b)) b = $0 } END { print b }')"
        fi
        if [ -z "$rule" ]; then
            rule="$(printf '%s\n' "$RULES" \
                    | awk -v s="$stem" 'index(s, $0) > 0 { if (length($0) > length(b)) b = $0 } END { print b }')"
        fi
        run="${parts[0]}"
        [ "$run" = "$stem" ] && run=""
        # First data line only; snakemake writes one row per job.
        row="$(awk 'NR==2 { print; exit }' "$f")"
        [ -n "$row" ] || continue
        printf '%s\n' "$row" | awk -F'\t' -v r="${rule:-unknown}" -v run="$run" -v stem="$stem" \
            'BEGIN { OFS="," } { print "benchmark", r, run, stem, "", "", $1, $2, $3, $10, "", "" }' >> "$OUT"
        n_bench=$((n_bench + 1))
    done < <(find "$BENCH" -type f 2>/dev/null | sort)
fi

# --- 3. sacct ----------------------------------------------------------------
# MaxRSS lives on the .batch step, JobName/State/ReqMem on the allocation row,
# so both are requested and joined on the base job id. Job names are
# smk-<PYPSA_JOB_TAG>-<rule>, minted by snakemake_profiles/sherlock/config.yaml.
n_sacct=0
if command -v sacct >/dev/null 2>&1; then
    sacct -u "${USER}" -S "$SINCE" -P -n \
          --format=JobID,JobName%120,State,ElapsedRaw,Elapsed,MaxRSS,ReqMem,TotalCPU,Partition 2>/dev/null \
    | awk -F'|' -v OFS=',' -v tag="$TAG" '
        function tomb(v,   u, n) {
            if (v == "" || v == "0") return ""
            u = substr(v, length(v), 1); n = v + 0
            if (u == "K") return sprintf("%.2f", n / 1024)
            if (u == "M") return sprintf("%.2f", n)
            if (u == "G") return sprintf("%.2f", n * 1024)
            if (u == "T") return sprintf("%.2f", n * 1024 * 1024)
            return sprintf("%.2f", n / 1048576)   # bytes
        }
        {
            id = $1
            base = id; sub(/\..*$/, "", base)
            if (id ~ /\./) { if ($6 != "") rss[base] = $6; next }
            name[base] = $2; state[base] = $3; raw[base] = $4
            hms[base] = $5; req[base] = $7; cpu[base] = $8; part[base] = $9
            order[++k] = base
        }
        END {
            for (i = 1; i <= k; i++) {
                b = order[i]
                nm = name[b]
                if (nm !~ /^smk-/) continue
                if (tag != "" && index(nm, "smk-" tag "-") != 1) continue
                rule = nm
                sub(/^smk-[^-]*-/, "", rule)
                print "sacct", rule, "", nm, b, state[b], raw[b], hms[b], tomb(rss[b]), "", req[b], part[b]
            }
        }' >> "$OUT"
    n_sacct="$(awk -F, 'NR>1 && $1=="sacct"' "$OUT" | wc -l)"
fi

# --- 4. Summary --------------------------------------------------------------
echo "wrote $OUT  ($n_bench benchmark rows, $n_sacct sacct rows)"
echo
awk -F, 'NR > 1 {
    k = $1 "/" $2
    n[k]++
    if ($7 + 0 > w[k]) w[k] = $7 + 0
    if ($9 + 0 > m[k]) m[k] = $9 + 0
}
END {
    printf "%-12s %-32s %5s %12s %14s\n", "source", "rule", "n", "peak_wall_s", "peak_rss_MB"
    for (k in n) {
        split(k, a, "/")
        printf "%-12s %-32s %5d %12.0f %14.1f\n", a[1], a[2], n[k], w[k], m[k]
    }
}' "$OUT" | { read -r h; echo "$h"; sort -k5 -gr; }
