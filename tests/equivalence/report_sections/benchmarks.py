"""Rule-grouped benchmark table (runtime + memory per snakemake rule).

Benchmark keys in each side's manifest are paths relative to that side's
``benchmarks/`` directory.  The two sides name their benchmark files
differently (the V1-epic stack threads ``_s{simpl}`` through filenames, the
anchor does not), so classification into rules uses an ordered regex list
PER SIDE.  Keys matching no pattern are historic pollution from earlier,
unrelated runs and are ignored entirely.

Exports used by summary.py and dag.py (signatures are load-bearing):

- ``harness_benchmark_df(ctx)`` -> DataFrame with columns
  ``['rule', 'instance', 'side', 'wall_s', 'max_rss_mb']`` (side values are
  the internal 'candidate'/'anchor').
- ``rule_walltimes(ctx)`` -> ``dict[rule] = {'candidate': float|None,
  'anchor': float|None}`` (wall seconds summed over instances per side).
- ``render(ctx)`` -> HTML fragment.
"""

from __future__ import annotations

import html
import re

import pandas as pd

from ..paths import INTERCONNECT as IC

# Pipeline (build) order for the grouped table.
RULE_ORDER = [
    "build_fuel_prices",
    "build_electrical_demand",
    "build_renewable_profiles",
    "add_demand",
    "add_electricity",
    "cluster_network",
    "solve_network",
]

# Ordered (pattern, rule) lists per side; first match wins, no match => ignore.
_CAND_PATTERNS = [
    (r"equivalence/{IC}/build_fuel_prices$", "build_fuel_prices"),
    (r"equivalence/{IC}/(power)_build_demand_s(20)?$", "build_electrical_demand"),
    (r"equivalence/{IC}/build_renewable_profiles_\w+_2030_s(20)?$", "build_renewable_profiles"),
    (r"equivalence/{IC}/elec_s(20)?_add_demand$", "add_demand"),
    (r"equivalence/{IC}/elec_s(20)?_add_electricity$", "add_electricity"),
    (r"cluster_network/{IC}/elec_s(20)?_c\d+m?$", "cluster_network"),
    (r"equivalence/solve_network/{IC}/.*$", "solve_network"),
]
_ANCH_PATTERNS = [
    (r"equivalence/{IC}/build_fuel_prices$", "build_fuel_prices"),
    (r"equivalence/{IC}/power_build_demand$", "build_electrical_demand"),
    (r"equivalence/{IC}/build_renewable_profiles_\w+_2030$", "build_renewable_profiles"),
    (r"equivalence/{IC}/add_demand$", "add_demand"),
    (r"equivalence/{IC}/add_electricity$", "add_electricity"),
    (r"cluster_network/{IC}/elec_s(20)?_c\d+m?$", "cluster_network"),
    (r"equivalence/solve_network/{IC}/.*$", "solve_network"),
]
_PATTERNS = {"candidate": _CAND_PATTERNS, "anchor": _ANCH_PATTERNS}


def _classify(key: str, patterns: list[tuple[str, str]]) -> str | None:
    for pat, rule in patterns:
        if re.match(pat.format(IC=IC), key):
            return rule
    return None


def _num(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def harness_benchmark_df(ctx) -> pd.DataFrame:
    """Long-format harness benchmarks: one row per (rule, instance, side)."""
    rows = []
    for side in ("candidate", "anchor"):
        bench = (ctx["manifests"].get(side) or {}).get("benchmarks") or {}
        for key, vals in bench.items():
            rule = _classify(key, _PATTERNS[side])
            if rule is None:
                continue  # historic pollution
            rows.append(
                {
                    "rule": rule,
                    "instance": key.rsplit("/", 1)[-1],
                    "side": side,
                    "wall_s": _num(vals.get("s")),
                    "max_rss_mb": _num(vals.get("max_rss")),
                },
            )
    df = pd.DataFrame(rows, columns=["rule", "instance", "side", "wall_s", "max_rss_mb"])
    if not df.empty:
        df["rule"] = pd.Categorical(df["rule"], categories=RULE_ORDER, ordered=True)
        df = df.sort_values(["rule", "instance", "side"]).reset_index(drop=True)
        df["rule"] = df["rule"].astype(str)
    return df


def rule_walltimes(ctx) -> dict:
    """Per-rule wall seconds summed over instances, per side (None if absent)."""
    df = harness_benchmark_df(ctx)
    out: dict = {}
    for rule in RULE_ORDER:
        sub = df[df["rule"] == rule] if not df.empty else df
        entry = {}
        for side in ("candidate", "anchor"):
            vals = sub[sub["side"] == side]["wall_s"].dropna() if not sub.empty else []
            entry[side] = float(vals.sum()) if len(vals) else None
        out[rule] = entry
    return out


def _fmt_wall(v: float | None) -> str:
    return "&mdash;" if v is None else f"{v:.1f}"


def _fmt_rss(v: float | None) -> str:
    return "&mdash;" if v is None else f"{v:.0f}"


def _side_stats(sub: pd.DataFrame, side: str) -> tuple[float | None, float | None]:
    """(summed wall_s, peak max_rss_mb) for one side of a df slice."""
    s = sub[sub["side"] == side]
    wall = s["wall_s"].dropna()
    rss = s["max_rss_mb"].dropna()
    return (
        float(wall.sum()) if len(wall) else None,
        float(rss.max()) if len(rss) else None,
    )


def render(ctx) -> str:
    labels = ctx["labels"]
    cand, anch = labels["candidate"], labels["anchor"]
    df = harness_benchmark_df(ctx)

    parts = ["<h2>Runtime and memory by snakemake rule</h2>"]
    parts.append(
        f"<p>How long each pipeline stage took in the harness runs, from snakemake's "
        f"<code>benchmark:</code> files, grouped by the rule that owns them. Bold rows are "
        f"rules (wall time summed over that rule's instances, memory as the peak instance); "
        f"the indented rows underneath are the individual instances &mdash; one benchmark file "
        f"per wildcard expansion, so the filename tells you which network/wildcards it ran on. "
        f"The useful comparison is per-rule: {cand} vs {anch} on the same bold row.</p>",
    )

    if df.empty:
        parts.append('<p><span class="badge na">no data</span> No harness benchmarks matched in either manifest.</p>')
        return "".join(parts)

    head = (
        "<tr><th>Rule / instance</th>"
        f"<th>{cand} wall (s)</th><th>{cand} max RSS (MB)</th>"
        f"<th>{anch} wall (s)</th><th>{anch} max RSS (MB)</th></tr>"
    )
    body_rows = []
    for rule in RULE_ORDER:
        sub = df[df["rule"] == rule]
        if sub.empty:
            continue
        cw, cr = _side_stats(sub, "candidate")
        aw, ar = _side_stats(sub, "anchor")
        body_rows.append(
            f'<tr style="font-weight:bold;background:#f2f2f2"><td>{html.escape(rule)}</td>'
            f"<td>{_fmt_wall(cw)}</td><td>{_fmt_rss(cr)}</td>"
            f"<td>{_fmt_wall(aw)}</td><td>{_fmt_rss(ar)}</td></tr>",
        )
        for inst in sorted(sub["instance"].unique()):
            isub = sub[sub["instance"] == inst]
            iw_c, ir_c = _side_stats(isub, "candidate")
            iw_a, ir_a = _side_stats(isub, "anchor")
            body_rows.append(
                f'<tr><td style="padding-left:2em;color:#444">{html.escape(inst)}</td>'
                f"<td>{_fmt_wall(iw_c)}</td><td>{_fmt_rss(ir_c)}</td>"
                f"<td>{_fmt_wall(iw_a)}</td><td>{_fmt_rss(ir_a)}</td></tr>",
            )
    tw_c, tr_c = _side_stats(df, "candidate")
    tw_a, tr_a = _side_stats(df, "anchor")
    body_rows.append(
        f'<tr style="font-weight:bold;border-top:2px solid #999"><td>TOTAL</td>'
        f"<td>{_fmt_wall(tw_c)}</td><td>{_fmt_rss(tr_c)}</td>"
        f"<td>{_fmt_wall(tw_a)}</td><td>{_fmt_rss(tr_a)}</td></tr>",
    )
    parts.append(f"<table>{head}{''.join(body_rows)}</table>")

    parts.append(
        "<p><strong>Reading notes</strong></p><ol style='font-size:12px;color:#333'>"
        "<li>Rules own the walltimes; instances are wildcard expansions of the same rule "
        "(different simpl/cluster values, carriers, or horizons).</li>"
        f"<li>The {anch}'s demand, renewable-profile, and add_* benchmark paths carry no "
        "<code>_s&#123;simpl&#125;</code> suffix, so the prong-1 and prong-2 runs OVERWRITE the same "
        f"benchmark file &mdash; {anch} shows only the most recent run there, while {cand} shows "
        "both instances. This is a bookkeeping artifact, not a speedup.</li>"
        f"<li>The {anch} has no <code>benchmark:</code> directive on its "
        "<code>simplify_network</code> rule, so that stage is uninstrumented and absent here.</li>"
        "<li><code>max_rss</code> reads 0 on macOS (psutil cannot sample it there); rerun the "
        "harness on Linux for real memory numbers.</li>"
        f"<li>The cross-side TOTAL rows are NOT apples-to-apples &mdash; {cand} and {anch} have "
        "different instance counts (see note 2, plus solve-benchmark coverage differs). Compare "
        "the per-rule bold rows instead.</li></ol>",
    )
    return "".join(parts)
