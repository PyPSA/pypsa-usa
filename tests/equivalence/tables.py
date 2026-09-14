"""The master-vs-develop comparison table and its on-disk form.

This module owns the **single source of truth for tolerances**
(:data:`TOLERANCES`). ``compare.py`` imports them from here rather than defining
its own, so the stage-by-stage findings and the comparison table cannot disagree
about what "within tolerance" means. Each family carries **both** a relative
tolerance and an absolute floor: without the floor, solver noise of 1e-7 MW on a
carrier that neither side built reads as a -100 % difference.

A row of the comparison table is one ``(metric, key)`` pair with a verdict:

``equivalent``
    ``|delta| <= atol`` **or** ``|delta_pct| <= rtol``. The two branches agree.
``explained``
    Over tolerance, and a hot-fix id resolves for it: an id that is present in
    ``hotfixes.yaml`` and is **not** marked ``ported``. The id comes either from
    a waiver that names this metric/key/family, or from an ``expect`` glob on
    the hot-fix itself.
``UNEXPLAINED``
    Over tolerance with nothing valid to point at. This is what the benchmark is
    for; the run fails while any of these remain.
``MISSING``
    The metric could not be computed at all — a criterion silently vanishing
    from the table is worse than a difference, so it is a failure too.
``one-sided`` / ``undefined``
    A ratio metric (capacity factor, quantiles) that is defined on one side only,
    or on neither. Reported, not failed: the corresponding additive metric
    (capacity, dispatch) carries the substantive difference.

Whether an id is an admissible explanation is decided by
:func:`hotfixes.explains`, not here: an id that has no row in ``hotfixes.yaml``
is rejected, and so is one marked ``ported: true`` — once a fix is a commit on
``master-benchmark`` it runs on BOTH sides, so a difference tracing to it means
the port is broken, not that the difference is explained. Rejections are
reported verbatim in the ``hotfix`` column and the row stays ``UNEXPLAINED``;
when the registry is empty, nothing can be explained.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import pandas as pd

from .hotfixes import explains
from .hotfixes import load_hotfixes as _load_registry


@dataclass(frozen=True)
class Tolerance:
    """Relative tolerance (fraction) and absolute floor (the metric's own unit)."""

    rtol: float
    atol: float


#: Per-family tolerances. Single source of truth — ``compare.py`` imports these;
#: do not restate a tolerance anywhere else. ``atol`` is the noise floor below
#: which a difference is not a difference: 1 MW of capacity, 1 MWh of energy,
#: 1 MW of demand, 1e-3 of a capacity factor. The objective has no floor, so it
#: is judged purely on the relative tolerance.
TOLERANCES: dict[str, Tolerance] = {
    "objective": Tolerance(rtol=1e-3, atol=0.0),
    "capacity": Tolerance(rtol=5e-3, atol=1.0),
    "dispatch": Tolerance(rtol=5e-3, atol=1.0),
    "p_nom_max": Tolerance(rtol=1e-3, atol=1.0),
    "p_max_pu": Tolerance(rtol=1e-3, atol=1e-3),
    "demand": Tolerance(rtol=1e-3, atol=1.0),
}

#: (family, substrings) in resolution order — first match wins. Ordering
#: matters: ``capacity_factor_by_carrier`` is a dispatch-derived quantity and
#: must not be caught by the plain ``capacity`` rule.
_FAMILY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("objective", ("objective",)),
    ("p_nom_max", ("p_nom_max", "potential")),
    ("dispatch", ("dispatch", "energy", "capacity_factor")),
    ("p_max_pu", ("p_max_pu", "mean_cf", "meancf", "profile")),
    ("demand", ("demand", "load")),
    ("capacity", ("capacity", "p_nom")),
)

COMPARISON_COLUMNS = [
    "metric",
    "key",
    "master",
    "develop",
    "delta",
    "delta_pct",
    "tolerance_pct",
    "tolerance_abs",
    "verdict",
    "hotfix",
]

#: Verdicts that make a run fail.
FAILING_VERDICTS = ("UNEXPLAINED", "MISSING")

#: Sort order: what must be looked at first comes first.
VERDICT_ORDER = ("MISSING", "UNEXPLAINED", "one-sided", "undefined", "explained", "equivalent")

#: Markdown is for reading, not for archiving — the CSV beside it is complete.
MD_ROW_CAP = 60

#: Every metric name ``plots.collect_metrics`` can produce, with one example
#: per per-tech family. A hot-fix ``expect`` pattern is matched against these,
#: so a pattern that can never match any of them is dead configuration and
#: :func:`unmatched_expect_patterns` turns it into a test failure. Keep in step
#: with ``plots.collect_metrics``.
KNOWN_METRICS: tuple[str, ...] = (
    "objective",
    "capacity_existing_by_carrier",
    "capacity_opt_by_carrier",
    "p_nom_existing_by_zone_carrier",
    "dispatch_by_carrier",
    "capacity_factor_by_carrier",
    "demand_by_zone",
    "p_max_pu_quantiles_solar",
    "p_max_pu_quantiles_onwind",
    "p_nom_max_by_zone_solar",
    "p_nom_max_by_zone_onwind",
    "mean_cf_by_zone_solar",
    "mean_cf_by_zone_onwind",
)

#: Waiver fields that can attach a waiver to a comparison-table row. A waiver
#: naming none of them is a ``compare.py`` cell waiver (stage/component/column/
#: kind) and has nothing to say about this table.
_WAIVER_MATCH_KEYS = ("metric", "key", "family")


def tolerance_family(metric: str) -> str:
    """Resolve a metric name to its tolerance family.

    Raises ``ValueError`` rather than guessing a default: a metric nobody
    assigned a tolerance to is a gap in the evaluation criterion, not a row to
    wave through.
    """
    low = metric.lower()
    for family, needles in _FAMILY_RULES:
        if any(nd in low for nd in needles):
            return family
    raise ValueError(
        f"no tolerance family for metric {metric!r}; known families: {sorted(TOLERANCES)}",
    )


def tolerance_for(metric: str) -> Tolerance:
    """The :class:`Tolerance` that applies to ``metric``."""
    return TOLERANCES[tolerance_family(metric)]


def tolerance_pct(metric: str) -> float:
    """Relative tolerance for ``metric``, in percent."""
    return tolerance_for(metric).rtol * 100.0


def load_hotfixes(path: Path | None = None) -> dict[str, dict]:
    """The hot-fix registry as ``{id: entry}``; a missing file yields ``{}``.

    Delegates to :func:`hotfixes.load_hotfixes`, which owns ``hotfixes.yaml``
    and its schema (``id``, ``commit``, ``pr``, ``title``, ``confidence``,
    ``ported``, ``usa_noop``, optional ``expect``). Kept here as the name the
    table layer imports, so there is one loader, not two.
    """
    return _load_registry(path)


def _hotfix_matches(entry: dict, metric: str, key: str) -> bool:
    """Does this hot-fix's ``expect`` list name ``metric`` or ``metric/key``."""
    patterns = entry.get("expect") or []
    if isinstance(patterns, str):
        patterns = [patterns]
    candidates = (metric, f"{metric}/{key}")
    return any(fnmatch(c, str(p)) for p in patterns for c in candidates)


def pattern_is_satisfiable(pattern: str) -> bool:
    """Can this ``expect`` glob ever match a metric this harness produces."""
    return any(
        fnmatch(m, pattern) or fnmatch(f"{m}/key", pattern) for m in KNOWN_METRICS
    )


def unmatched_expect_patterns(registry: dict[str, dict]) -> dict[str, list[str]]:
    """``{hotfix id: patterns that can never match}`` — dead configuration.

    An ``expect`` pattern written in a vocabulary the matcher does not speak is
    worse than no pattern: it reads as a claim about which metrics a hot-fix can
    move while doing nothing. The registry test fails on any entry here.
    """
    out: dict[str, list[str]] = {}
    for hid, row in registry.items():
        patterns = row.get("expect") or []
        if isinstance(patterns, str):
            patterns = [patterns]
        dead = [str(p) for p in patterns if not pattern_is_satisfiable(str(p))]
        if dead:
            out[hid] = dead
    return out


def _waiver_hotfix(waivers: list[dict], metric: str, family: str, key: str) -> str | None:
    """The ``hotfix:`` id of the first waiver that NAMES this row, if any.

    A waiver must name at least one of ``metric``/``key``/``family``, and every
    field it does name must match. ``waivers.yaml`` entries carry
    ``stage``/``component``/``column``/``kind`` instead — those are cell waivers
    for ``compare.py`` and are ignored here. Treating their absent fields as
    wildcards would let one ``hotfix:``-tagged cell waiver explain every
    over-tolerance row in every family.
    """
    row = {"metric": metric, "key": key, "family": family}
    for w in waivers:
        if not w.get("hotfix"):
            continue
        named = [k for k in _WAIVER_MATCH_KEYS if w.get(k) not in (None, "*")]
        if not named:
            continue
        if all(w[k] == row[k] for k in named):
            return str(w["hotfix"])
    return None


def _explanation(
    metric: str,
    family: str,
    key: str,
    hotfixes: dict[str, dict],
    waivers: list[dict],
) -> tuple[str, bool]:
    """(hotfix cell, is_valid_explanation) for an over-tolerance row.

    An id explains the row only if it **resolves in the registry** and is not
    marked ``ported`` (a ported fix is on both sides, so it cannot be why they
    differ). Unresolvable and ported ids are reported with the reason attached
    and the row stays UNEXPLAINED.
    """
    ids: list[str] = []
    waived = _waiver_hotfix(waivers, metric, family, key)
    if waived:
        ids.append(waived)
    ids += [
        hid
        for hid, e in sorted(hotfixes.items())
        if isinstance(e, dict) and _hotfix_matches(e, metric, key) and hid not in ids
    ]
    if not ids:
        return "", False
    verdicts = {h: explains(h, hotfixes) for h in ids}
    usable = [h for h, (ok, _) in verdicts.items() if ok]
    if usable:
        return ",".join(usable), True
    return "; ".join(reason for _, reason in verdicts.values()), False


def _as_frame(obj) -> pd.DataFrame:
    """Accept a metric DataFrame or the objective Series; return a DataFrame."""
    if isinstance(obj, pd.Series):
        return obj.to_frame().T.set_index(pd.Index(["total_system_cost"], name="key"))
    return obj


def _key_label(key) -> str:
    """Flatten a (possibly MultiIndex) key to one readable string."""
    if isinstance(key, tuple):
        return " | ".join(str(k) for k in key)
    return str(key)


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _row_verdict(
    metric: str,
    family: str,
    key: str,
    master: float,
    develop: float,
    delta: float,
    pct: float,
    tol: Tolerance,
    hotfixes: dict[str, dict],
    waivers: list[dict],
) -> tuple[str, str]:
    """(verdict, hotfix cell) for one comparison row."""
    m_na, d_na = np.isnan(master), np.isnan(develop)
    if m_na and d_na:
        return "undefined", "undefined on both sides"
    if m_na or d_na:
        return "one-sided", "develop only" if m_na else "master only"
    if abs(delta) <= tol.atol or (np.isfinite(pct) and abs(pct) <= tol.rtol * 100.0):
        return "equivalent", ""
    cell, ok = _explanation(metric, family, key, hotfixes, waivers)
    return ("explained" if ok else "UNEXPLAINED"), cell


def comparison_table(
    metrics: dict[str, pd.DataFrame],
    hotfixes: dict[str, dict],
    waivers: list[dict] | None = None,
    missing: list[dict] | None = None,
) -> pd.DataFrame:
    """One row per ``(metric, key)`` with a verdict.

    ``metrics`` maps a metric name to a frame from ``metrics.py`` (columns
    ``master``/``develop``/``delta``/``delta_pct``); the objective Series is
    accepted too and appears as the single key ``total_system_cost``.

    ``hotfixes`` is the ``hotfixes.yaml`` registry as ``{id: entry}``; pass
    ``{}`` before that file exists, in which case nothing can be explained.
    ``waivers`` is optional and only its ``hotfix:`` field is consulted, and only
    on waivers that name a metric/key/family. ``missing`` is the list of metrics
    that could not be computed (``plots.collect_metrics`` fills it); each becomes
    a ``MISSING`` row so that a criterion cannot vanish from the table silently.

    A ``delta_pct`` of NaN with a finite master of 0 means an
    appear-from-nothing difference; it is over tolerance unless the absolute
    floor covers it. NaN on a side means the metric is undefined there.

    Rows are sorted by :data:`VERDICT_ORDER`, then ``|delta_pct|`` descending.
    """
    waivers = waivers or []
    rows = []
    for metric, obj in metrics.items():
        df = _as_frame(obj)
        if df is None or df.empty:
            continue
        family = tolerance_family(metric)
        tol = TOLERANCES[family]
        for key, r in df.iterrows():
            label = _key_label(key)
            master, develop = _f(r["master"]), _f(r["develop"])
            delta, pct = _f(r["delta"]), _f(r["delta_pct"])
            verdict, cell = _row_verdict(
                metric, family, label, master, develop, delta, pct, tol, hotfixes, waivers,
            )
            rows.append(
                {
                    "metric": metric,
                    "key": label,
                    "master": master,
                    "develop": develop,
                    "delta": delta,
                    "delta_pct": pct,
                    "tolerance_pct": tol.rtol * 100.0,
                    "tolerance_abs": tol.atol,
                    "verdict": verdict,
                    "hotfix": cell,
                },
            )
    rows += _missing_rows(missing or [])
    out = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
    if out.empty:
        return out
    order = {v: i for i, v in enumerate(VERDICT_ORDER)}
    magnitude = out["delta_pct"].abs().fillna(np.inf)
    out = out.assign(_o=out["verdict"].map(order).fillna(0), _m=magnitude)
    out = out.sort_values(["_o", "_m"], ascending=[True, False], kind="mergesort")
    return out.drop(columns=["_o", "_m"]).reset_index(drop=True)


def _missing_rows(missing: list[dict]) -> list[dict]:
    """One ``MISSING`` row per metric that could not be computed."""
    rows = []
    for entry in missing:
        metric = str(entry.get("metric", "<unknown>"))
        try:
            tol = tolerance_for(metric)
            tol_pct, tol_abs = tol.rtol * 100.0, tol.atol
        except (ValueError, KeyError):
            tol_pct = tol_abs = float("nan")
        rows.append(
            {
                "metric": metric,
                "key": "<not computed>",
                "master": float("nan"),
                "develop": float("nan"),
                "delta": float("nan"),
                "delta_pct": float("nan"),
                "tolerance_pct": tol_pct,
                "tolerance_abs": tol_abs,
                "verdict": "MISSING",
                "hotfix": str(entry.get("reason", "")),
            },
        )
    return rows


def verdict_counts(table: pd.DataFrame) -> dict[str, int]:
    """``{verdict: n}`` over every verdict, zeros included."""
    counts = dict.fromkeys(VERDICT_ORDER, 0)
    if not table.empty:
        counts.update({str(k): int(v) for k, v in table["verdict"].value_counts().items()})
    return counts


def n_failing(table: pd.DataFrame) -> int:
    """How many rows make the run fail (UNEXPLAINED + MISSING)."""
    counts = verdict_counts(table)
    return sum(counts.get(v, 0) for v in FAILING_VERDICTS)


def _fmt(x) -> str:
    """Format one number for the markdown table."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    ax = abs(float(x))
    if ax != 0 and (ax >= 1e5 or ax < 1e-3):
        return f"{x:.4g}"
    return f"{x:,.4f}".rstrip("0").rstrip(".")


def to_markdown(table: pd.DataFrame, cap: int = MD_ROW_CAP) -> str:
    """Render the comparison table as markdown, capped at ``cap`` rows."""
    c = verdict_counts(table)
    head = [
        "# Comparison: master-benchmark (baseline) vs develop",
        "",
        f"`delta` is **develop minus master**. Verdicts: {c['equivalent']} equivalent, "
        f"{c['explained']} explained, {c['one-sided']} one-sided, {c['undefined']} undefined, "
        f"**{c['UNEXPLAINED']} UNEXPLAINED**, **{c['MISSING']} MISSING**.",
        "",
        "A row is equivalent when `|delta|` is within the absolute floor (`tol abs`) "
        "OR `|delta %|` is within the relative tolerance (`tol %`).",
        "",
    ]
    if table.empty:
        return "\n".join([*head, "_No comparable metrics were produced._", ""])
    shown = table.head(cap)
    head += [
        "| metric | key | master | develop | delta | delta % | tol % | tol abs | verdict | hot-fix |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for _, r in shown.iterrows():
        head.append(
            f"| {r['metric']} | {r['key']} | {_fmt(r['master'])} | {_fmt(r['develop'])} | "
            f"{_fmt(r['delta'])} | {_fmt(r['delta_pct'])} | {_fmt(r['tolerance_pct'])} | "
            f"{_fmt(r['tolerance_abs'])} | {r['verdict']} | {r['hotfix'] or '-'} |",
        )
    if len(table) > cap:
        head += ["", f"_{len(table) - cap} more rows in comparison.csv_"]
    return "\n".join([*head, ""])


def write_tables(tables: dict[str, pd.DataFrame], outdir: Path) -> list[Path]:
    """Write ``<outdir>/tables/<name>.csv`` per frame, plus ``comparison.md``.

    ``tables`` must contain a ``comparison`` frame for the markdown to be
    written; every other entry is dumped as CSV beside it so that every number
    quoted anywhere has a machine-readable home.
    """
    tdir = Path(outdir) / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, df in tables.items():
        if df is None:
            continue
        p = tdir / f"{name}.csv"
        frame = _as_frame(df)
        # Keyed metric frames carry their key in the index; the comparison
        # table carries it in a column and has a meaningless RangeIndex.
        frame.to_csv(p, index=not isinstance(frame.index, pd.RangeIndex))
        written.append(p)
    if "comparison" in tables and tables["comparison"] is not None:
        md = tdir / "comparison.md"
        md.write_text(to_markdown(tables["comparison"]))
        written.append(md)
    return written


def export_all(run_dir: Path, ctx: object | None = None, result: dict | None = None) -> pd.DataFrame:
    """Build the comparison table for one run and write ``<run_dir>/tables/``.

    The entry point ``run.py`` calls. Metrics come from
    :func:`plots.run_metrics`, which memoises them for the process, so the table
    and the figures are drawn from the same numbers and the networks are read
    once. ``result`` is the findings dict from ``compare.run_comparison``; it is
    accepted so this hook has the whole run in hand, and is not needed to build
    the table. Returns the comparison frame.
    """
    from . import plots
    from .compare import load_waivers

    prong = int(getattr(ctx, "prong", 2))
    _artifacts, frames, missing = plots.run_metrics(prong)
    comparison = comparison_table(frames, load_hotfixes(), load_waivers(), missing)
    written = write_tables({**frames, "comparison": comparison}, run_dir)
    print(f"[equivalence] tables: {len(written)} file(s) under {Path(run_dir) / 'tables'}")
    return comparison
