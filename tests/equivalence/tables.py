"""The master-vs-develop comparison table and its on-disk form.

This module owns the **single source of truth for tolerances**
(:data:`TOLERANCES`). ``compare.py`` imports them from here rather than defining
its own, so the stage-by-stage findings and the comparison table cannot disagree
about what "within tolerance" means.

A row of the comparison table is one ``(metric, key)`` pair with a verdict:

``equivalent``
    ``|delta_pct| <= tolerance_pct``. The two branches agree on this quantity.
``explained``
    Over tolerance, and a hot-fix id resolves for it — either from a waiver
    tagged ``hotfix:`` or from a glob in that hot-fix's ``expect`` list. The
    difference is a known, documented consequence of a develop-side change.
``UNEXPLAINED``
    Over tolerance with nothing to point at. This is what the benchmark is for;
    the run fails while any of these remain.

A hot-fix marked ``ported: true`` is **not** a valid explanation: once it is a
commit on ``master-benchmark`` its effect is present on *both* sides, so it
cannot be the reason they differ. Such ids are rejected here with a message in
the ``hotfix`` column and the row stays ``UNEXPLAINED``.
"""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

#: Relative tolerance per metric family, as a fraction (not a percentage).
#: Single source of truth — ``compare.py`` imports these; do not restate them.
TOLERANCES = {
    "objective": 1e-3,
    "capacity": 5e-3,
    "dispatch": 5e-3,
    "p_nom_max": 1e-3,
    "p_max_pu": 1e-3,
    "demand": 1e-3,
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
    "verdict",
    "hotfix",
]

#: Markdown is for reading, not for archiving — the CSV beside it is complete.
MD_ROW_CAP = 60

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


def tolerance_pct(metric: str) -> float:
    """Tolerance for ``metric``, in percent."""
    return TOLERANCES[tolerance_family(metric)] * 100.0


def load_hotfixes(path: Path) -> dict[str, dict]:
    """Load ``hotfixes.yaml`` into ``{id: entry}``; a missing file gives ``{}``.

    The file is a YAML **list**; T4 owns writing it and its loader lives here so
    that the table can be built before it exists. Expected schema of one entry:

    ``id``
        ``HF-n``, bare (not zero-padded), matching the ``#`` column of
        ``memory/plans/hotfix-ledger.md``.
    ``commit``
        7- or 40-character sha of the develop commit that introduced it.
    ``pr``
        integer pull-request number.
    ``title``
        one-line description, non-empty.
    ``confidence``
        one of ``high``, ``medium``, ``low``, ``none`` — how strongly this
        hot-fix is expected to move a benchmark number.
    ``ported``
        boolean. ``true`` once the fix has been ported onto
        ``master-benchmark``, at which point it is present on both sides and is
        **no longer an admissible explanation** for a difference.
    ``expect`` *(optional)*
        list of glob patterns naming the metrics this hot-fix is expected to
        move, e.g. ``[capacity/*, dispatch/*, objective/*]``. Patterns are
        matched against ``"<family>/<key>"``, ``"<metric>/<key>"`` and
        ``"<metric>"``.
    """
    if not Path(path).exists():
        return {}
    raw = yaml.safe_load(Path(path).read_text()) or []
    if isinstance(raw, dict):  # tolerate a mapping form
        return {str(k): dict(v or {}, id=str(k)) for k, v in raw.items()}
    return {str(e["id"]): dict(e) for e in raw if isinstance(e, dict) and e.get("id")}


def _hotfix_matches(entry: dict, metric: str, family: str, key: str) -> bool:
    """Does this hot-fix declare that it moves ``metric``/``key``."""
    patterns = entry.get("expect") or []
    if isinstance(patterns, str):
        patterns = [patterns]
    candidates = (f"{family}/{key}", f"{metric}/{key}", metric, family)
    return any(fnmatch(c, str(p)) for p in patterns for c in candidates)


def _waiver_hotfix(waivers: list[dict], metric: str, family: str, key: str) -> str | None:
    """The ``hotfix:`` id of the first waiver matching this row, if any."""
    row = {"metric": metric, "key": key, "family": family}
    for w in waivers:
        if not w.get("hotfix"):
            continue
        if all(w.get(k) in (None, "*", row[k]) for k in _WAIVER_MATCH_KEYS):
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

    A ``ported: true`` hot-fix is on both sides, so it cannot explain a
    difference; it is reported with that reason attached and the row stays
    UNEXPLAINED.
    """
    ids: list[str] = []
    waived = _waiver_hotfix(waivers, metric, family, key)
    if waived:
        ids.append(waived)
    ids += [hid for hid, e in sorted(hotfixes.items()) if _hotfix_matches(e, metric, family, key) and hid not in ids]
    if not ids:
        return "", False
    usable = [h for h in ids if not _entry(hotfixes, h).get("ported", False)]
    if usable:
        return ",".join(usable), True
    return ",".join(f"{h} (ported; not an explanation)" for h in ids), False


def _entry(hotfixes: dict[str, dict], hid: str) -> dict:
    e = hotfixes.get(hid)
    return e if isinstance(e, dict) else {}


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


def comparison_table(
    metrics: dict[str, pd.DataFrame],
    hotfixes: dict[str, dict],
    waivers: list[dict] | None = None,
) -> pd.DataFrame:
    """One row per ``(metric, key)`` with a verdict.

    ``metrics`` maps a metric name to a frame from ``metrics.py`` (columns
    ``master``/``develop``/``delta``/``delta_pct``); the objective Series is
    accepted too and appears as the single key ``total_system_cost``.

    ``hotfixes`` is the ``hotfixes.yaml`` registry as ``{id: entry}``; pass
    ``{}`` before that file exists. ``waivers`` is optional and only its
    ``hotfix:`` field is consulted here.

    A ``delta_pct`` of NaN means master was zero while develop was not — an
    appear-from-nothing difference, treated as infinitely over tolerance.

    Rows are sorted UNEXPLAINED first, then by ``|delta_pct|`` descending.
    """
    waivers = waivers or []
    rows = []
    for metric, obj in metrics.items():
        df = _as_frame(obj)
        if df is None or df.empty:
            continue
        family = tolerance_family(metric)
        tol_pct = TOLERANCES[family] * 100.0
        for key, r in df.iterrows():
            pct = float(r["delta_pct"])
            over = (not np.isfinite(pct)) or abs(pct) > tol_pct
            label = _key_label(key)
            if over:
                cell, ok = _explanation(metric, family, label, hotfixes, waivers)
                verdict = "explained" if ok else "UNEXPLAINED"
            else:
                cell, verdict = "", "equivalent"
            rows.append(
                {
                    "metric": metric,
                    "key": label,
                    "master": float(r["master"]),
                    "develop": float(r["develop"]),
                    "delta": float(r["delta"]),
                    "delta_pct": pct,
                    "tolerance_pct": tol_pct,
                    "verdict": verdict,
                    "hotfix": cell,
                },
            )
    out = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
    if out.empty:
        return out
    order = {"UNEXPLAINED": 0, "explained": 1, "equivalent": 2}
    magnitude = out["delta_pct"].abs().fillna(np.inf).replace(np.nan, np.inf)
    out = out.assign(_o=out["verdict"].map(order), _m=magnitude)
    out = out.sort_values(["_o", "_m"], ascending=[True, False], kind="mergesort")
    return out.drop(columns=["_o", "_m"]).reset_index(drop=True)


def verdict_counts(table: pd.DataFrame) -> dict[str, int]:
    """``{verdict: n}`` over all three verdicts, zeros included."""
    counts = {"equivalent": 0, "explained": 0, "UNEXPLAINED": 0}
    if not table.empty:
        counts.update({str(k): int(v) for k, v in table["verdict"].value_counts().items()})
    return counts


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
    counts = verdict_counts(table)
    head = [
        "# Comparison: master-benchmark (baseline) vs develop",
        "",
        f"`delta` is **develop minus master**. Verdicts: {counts['equivalent']} equivalent, "
        f"{counts['explained']} explained, **{counts['UNEXPLAINED']} UNEXPLAINED**.",
        "",
    ]
    if table.empty:
        return "\n".join([*head, "_No comparable metrics were produced._", ""])
    shown = table.head(cap)
    head += [
        "| metric | key | master | develop | delta | delta % | tol % | verdict | hot-fix |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for _, r in shown.iterrows():
        head.append(
            f"| {r['metric']} | {r['key']} | {_fmt(r['master'])} | {_fmt(r['develop'])} | "
            f"{_fmt(r['delta'])} | {_fmt(r['delta_pct'])} | {_fmt(r['tolerance_pct'])} | "
            f"{r['verdict']} | {r['hotfix'] or '-'} |",
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
