#!/usr/bin/env python3
"""Turn snakemake benchmark TSVs into per-rule Slurm resource recommendations.

    ../.venv/bin/python report_benchmarks.py [benchmarks_dir]

Snakemake writes one TSV per job under benchmarks/, with columns
s / h:m:s / max_rss / max_vms / max_uss / max_pss / io_in / io_out /
mean_load / cpu_time.  Memory is in MB, s in seconds.

The rule name is recovered from the path, which mirrors each rule's `log:`
layout, so it is stripped of run name, interconnect, and wildcard-bearing
prefixes/suffixes.

Recommendations use headroom on the observed peak:
  mem   = max(2 GB, ceil(1.5 x peak max_rss / 500 MB) * 500 MB)
  time  = max(10 min, ceil(3 x peak runtime / 5 min) * 5 min)
The multipliers differ because memory overrun is a hard kill while a generous
walltime only costs queue priority; 3x also absorbs a slower/contended node.
"""

import re
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmarks")

# Wildcard debris to strip from the file stem to recover a rule name. The simpl
# wildcard is a number or a word like "county"/"all", so anchor on the known
# forms -- a bare `_s\w+$` would eat `_shapes` and `_substations`.
SIMPL = r"(?:\d+|county|all)"
STEM_SUBS = [
    (rf"_?elec_s{SIMPL}(?:_c\d+[a-z]?)?", ""),  # elec_s75_c4 / elec_scounty
    (rf"_s{SIMPL}$", ""),  # trailing _s75 / _scounty
    (r"_c?\d{2,}[a-z]?$", ""),  # trailing cluster counts / horizons
    (r"^elec_", ""),
]


# Benchmark paths are not uniform: most are benchmarks/<run>/<interconnect>/<stem>,
# but some rules (cluster_network) write benchmarks/<rule>/<interconnect>/<stem>.
# So resolve against the real rule list rather than inferring from position.
KNOWN_RULES = set()
for smk in Path("rules").glob("*.smk"):
    KNOWN_RULES.update(re.findall(r"^rule\s+(\w+)\s*:", smk.read_text(), re.M))


def rule_from(path: Path) -> str:
    # A rule name appearing as a directory component wins outright.
    for part in path.parts:
        if part in KNOWN_RULES:
            return part
    stem = path.stem
    # Longest known rule that prefixes the stem, so build_renewable_profiles_onwind
    # keeps its tech suffix but build_shapes is not truncated.
    hits = [r for r in KNOWN_RULES if stem.startswith(r)]
    if hits:
        best = max(hits, key=len)
        return stem if stem.startswith(f"{best}_") and best != stem else best
    for pat, rep in STEM_SUBS:
        stem = re.sub(pat, rep, stem)
    stem = stem.strip("_")
    return stem or path.parent.name


rows = []
for f in sorted(root.rglob("*")):
    if not f.is_file():
        continue
    try:
        df = pd.read_csv(f, sep="\t")
    except Exception:
        continue
    if df.empty or "s" not in df.columns:
        continue
    parts = f.relative_to(root).parts
    run = parts[0] if len(parts) > 1 else "-"
    r = df.iloc[0]
    rows.append(
        {
            "rule": rule_from(f),
            "run": run,
            "secs": float(r["s"]),
            "max_rss": float(r.get("max_rss", 0) or 0),
            "max_pss": float(r.get("max_pss", 0) or 0),
            "cpu_time": float(r.get("cpu_time", 0) or 0),
            "mean_load": float(r.get("mean_load", 0) or 0),
        },
    )

if not rows:
    sys.exit(f"no benchmark files under {root}")

d = pd.DataFrame(rows)


def fmt_hms(secs: float) -> str:
    s = int(round(secs))
    return f"{s // 3600:d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def rec_mem(peak_mb: float) -> int:
    import math

    return max(2000, int(math.ceil(1.5 * peak_mb / 500.0) * 500))


def rec_time(peak_s: float) -> str:
    import math

    mins = max(10, int(math.ceil(3 * peak_s / 60.0 / 5.0) * 5))
    return f"{mins // 60:02d}:{mins % 60:02d}:00"


agg = (
    d.groupby("rule")
    .agg(
        n=("secs", "size"),
        peak_s=("secs", "max"),
        peak_rss=("max_rss", "max"),
        mean_load=("mean_load", "max"),
    )
    .sort_values("peak_rss", ascending=False)
)

agg["rec_mem_mb"] = agg.peak_rss.map(rec_mem)
agg["rec_walltime"] = agg.peak_s.map(rec_time)
agg["peak_time"] = agg.peak_s.map(fmt_hms)

print(
    f"{'RULE':<44} {'N':>2} {'PEAK_RSS_MB':>11} {'PEAK_TIME':>10} {'LOAD%':>6}   {'REC_MEM_MB':>10} {'REC_WALLTIME':>12}"
)
print("-" * 108)
for name, r in agg.iterrows():
    print(
        f"{name:<44} {int(r.n):>2} {r.peak_rss:>11.0f} {r.peak_time:>10} "
        f"{r.mean_load:>6.0f}   {int(r.rec_mem_mb):>10} {r.rec_walltime:>12}",
    )

print()
print(f"total jobs benchmarked: {len(d)}   distinct rules: {len(agg)}")
print(f"sum of peak runtimes (serial lower bound): {fmt_hms(agg.peak_s.sum())}")
