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
    Over tolerance, and a **waiver names this row** and carries a ``hotfix:``
    tag that resolves in ``hotfixes.yaml``, is not ``ported`` and is not a
    ``usa_noop``. Nothing else grants it — see "expect is advisory" below. A
    waiver may also BOUND what it explains (:data:`WAIVER_BOUND_KEYS`:
    ``expect_sign``, ``max_abs_pct``); a row outside those bounds is not the
    difference that was signed off and stays ``UNEXPLAINED``.
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
is rejected, and so is one marked ``ported: true`` (a fix on
``master-benchmark`` runs on BOTH sides, so a difference tracing to it means the
port is broken) or ``usa_noop: true`` (it did nothing on this run at all).
Rejections are reported verbatim in the ``hotfix`` column and the row stays
``UNEXPLAINED``; when the registry is empty, nothing can be explained.

**``expect`` is advisory, and never grants a verdict.**

The registry's ``expect`` globs were once consulted directly: a row was
``explained`` if any unported hot-fix claimed it could move that metric. In the
metric-name vocabulary that turned out to cover **every one of the 13
:data:`KNOWN_METRICS`** between them, so ``UNEXPLAINED`` became unreachable and
the failing verdict could never fire. A safety net that catches everything is
not a safety net.

So ``expect`` now feeds one advisory column, ``candidates``: the hot-fix ids
that *claim* they could move this row, offered to the human doing the
attribution. It is unfiltered on purpose — a ported or no-op id showing up there
is itself worth knowing, because it says the port or the no-op claim is the
thing to check. The verdict is granted only by a waiver someone wrote against
this row, which is a decision with a name on it rather than a glob that happened
to match.
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
    "candidates",
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
#: kind) and has nothing to say about this table. The converse matters just as
#: much and is enforced by ``compare.is_waived``: a waiver that DOES name one of
#: these is a table-row waiver, and must not be read as a cell waiver whose four
#: absent fields wildcard onto every finding in the run.
WAIVER_MATCH_KEYS = ("metric", "key", "family")
_WAIVER_MATCH_KEYS = WAIVER_MATCH_KEYS

#: Fields that SCOPE a waiver to a particular run rather than selecting a row.
#: ``compare.is_waived`` honours both; so must this table, or a waiver written
#: for the western prong-1 leg silently explains a USA prong-2 difference. An
#: absent field means "any run", which is why they are checked separately from
#: _WAIVER_MATCH_KEYS: a waiver must NAME a row, but it need not name a run.
_WAIVER_SCOPE_KEYS = ("interconnect", "prong")


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


#: ``expect`` patterns that match everything and therefore say nothing. They are
#: treated as invalid rather than satisfiable: the whole point of the registry
#: lint is to catch a pattern that makes a claim it cannot support, and "this
#: hot-fix can move any metric at all" is the emptiest claim there is.
_VACUOUS_PATTERNS = frozenset({"*", "**", "*/*", "*/**", "**/*"})


def pattern_is_satisfiable(pattern: str) -> bool:
    """Can this ``expect`` glob ever name a row this harness produces.

    A row is addressed as ``metric`` or ``metric/key``. The key half is
    open-ended — carrier names, zone ids, quantiles — so a pattern is
    satisfiable when its metric half can match a real metric, whatever it says
    about the key: ``objective/total_system_cost`` and
    ``dispatch_by_carrier/CCGT`` both name rows that exist, and neither is a
    glob at all. What is NOT satisfiable is the old family vocabulary
    (``capacity/*``, whose head matches no metric) or a bare wildcard.
    """
    p = str(pattern).strip()
    if not p or p in _VACUOUS_PATTERNS:
        return False
    if any(fnmatch(m, p) for m in KNOWN_METRICS):
        return True
    head, sep, tail = p.partition("/")
    if not sep or not tail or head in _VACUOUS_PATTERNS:
        return False
    return any(fnmatch(m, head) for m in KNOWN_METRICS)


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


def _waiver_in_scope(waiver: dict, run: dict) -> bool:
    """Does this waiver apply to THIS run's interconnect and prong.

    An absent or ``'*'`` field on the WAIVER means "any run"; a field that is
    set must equal the run's, and is refused when the run does not say.
    ``compare.is_waived`` has always scoped cell waivers this way and the table
    did not, so a waiver written ``{interconnect: western, prong: 1}`` explained
    a whole-USA prong-2 row — the deferred western leg reaching forward to sign
    off a difference it never saw.
    """
    for k in _WAIVER_SCOPE_KEYS:
        want = waiver.get(k)
        if want in (None, "*"):
            continue
        if run.get(k) is None or str(want) != str(run[k]):
            return False
    return True


#: Optional BOUNDS a table waiver can put on the row it explains.
#:
#: A waiver used to be matched on ``metric``/``key`` alone, which made it a
#: blank cheque: the HF-24 waiver on ``p_nom_max_by_zone_solar/p8`` was written
#: for "+2,158 MW of potential master silently dropped" (+1.1 %), and it would
#: equally have explained a -50 % row or a 10,000x row — the opposite of the
#: fix's known direction, or a magnitude nothing in the hot-fix could produce.
#: The two bounds say what the waiver was measured on, so a row that walks out
#: of that range comes back as UNEXPLAINED instead of inheriting the signature.
#:
#: ``expect_sign``  '+' or '-': the sign of ``develop - master``.
#: ``max_abs_pct``  upper bound on ``|delta %|``.
WAIVER_BOUND_KEYS = ("expect_sign", "max_abs_pct")


def _bounds_violation(waiver: dict, delta: float, pct: float) -> str | None:
    """Which bound this waiver puts on the row is broken, or ``None``.

    Returns ``'sign'`` or ``'magnitude'``; a waiver carrying neither bound can
    never violate one. An undefined ``delta_pct`` (master is 0, develop is not —
    the appear-from-nothing case) counts as a magnitude violation whenever
    ``max_abs_pct`` is set: the bound cannot be shown to hold, and a waiver is a
    claim that has to be checkable.
    """
    want = waiver.get("expect_sign")
    if want in ("+", "-"):
        sign = "+" if delta > 0 else "-" if delta < 0 else "0"
        if sign != want:
            return "sign"
    cap = waiver.get("max_abs_pct")
    if cap is not None:
        try:
            cap = float(cap)
        except (TypeError, ValueError):
            return "magnitude"
        if not np.isfinite(pct) or abs(pct) > cap:
            return "magnitude"
    return None


def _matching_waivers(
    waivers: list[dict],
    metric: str,
    family: str,
    key: str,
    run: dict | None = None,
) -> list[dict]:
    """The in-scope, ``hotfix:``-tagged waivers that NAME this row, in file order.

    A waiver must name at least one of ``metric``/``key``/``family``, and every
    field it does name must match. ``waivers.yaml`` entries carry
    ``stage``/``component``/``column``/``kind`` instead — those are cell waivers
    for ``compare.py`` and are ignored here. Treating their absent fields as
    wildcards would let one ``hotfix:``-tagged cell waiver explain every
    over-tolerance row in every family.

    ``run`` carries this run's ``interconnect`` and ``prong``. A waiver that
    names either must agree with it, and a waiver that names one the caller
    could not supply is refused rather than assumed to match: a scoped waiver
    whose scope cannot be checked has not been shown to apply here.
    """
    row = {"metric": metric, "key": key, "family": family}
    out = []
    for w in waivers:
        if not w.get("hotfix"):
            continue
        named = [k for k in _WAIVER_MATCH_KEYS if w.get(k) not in (None, "*")]
        if not named:
            continue
        if not _waiver_in_scope(w, run or {}):
            continue
        if all(w[k] == row[k] for k in named):
            out.append(w)
    return out


def _waiver_hotfix(
    waivers: list[dict],
    metric: str,
    family: str,
    key: str,
    run: dict | None = None,
) -> str | None:
    """The ``hotfix:`` id of the first in-scope waiver that NAMES this row.

    Bounds-unaware; kept for callers that only need to know whether a row is
    claimed at all. :func:`_explanation` uses :func:`_matching_waivers` so it
    can check the bounds too.
    """
    matches = _matching_waivers(waivers, metric, family, key, run)
    return str(matches[0]["hotfix"]) if matches else None


def _hotfix_sort_key(hid: str) -> tuple[int, str]:
    head, _, num = str(hid).partition("-")
    return (int(num), head) if num.isdigit() else (1 << 30, str(hid))


def candidate_hotfixes(metric: str, key: str, hotfixes: dict[str, dict]) -> list[str]:
    """Ids whose ``expect`` claims this row — ADVISORY, never a verdict.

    Unfiltered by design: a ported or no-op id appearing here is worth seeing,
    because it says the port or the no-op claim is what to check. Use it to
    start an attribution, then record the answer as a waiver with a ``hotfix:``
    tag, which is what actually moves the verdict.
    """
    return sorted(
        (hid for hid, e in hotfixes.items() if isinstance(e, dict) and _hotfix_matches(e, metric, key)),
        key=_hotfix_sort_key,
    )


def _explanation(
    metric: str,
    family: str,
    key: str,
    hotfixes: dict[str, dict],
    waivers: list[dict],
    run: dict | None = None,
    delta: float = float("nan"),
    pct: float = float("nan"),
) -> tuple[str, bool]:
    """(hotfix cell, is_valid_explanation) for an over-tolerance row.

    ONLY a waiver that names this row can explain it, and only when its
    ``hotfix:`` tag resolves in the registry, is not ``ported`` (a ported fix is
    on both sides, so it cannot be why they differ) and is not a ``usa_noop``.
    A rejected tag is reported verbatim and the row stays UNEXPLAINED.

    A waiver that carries :data:`WAIVER_BOUND_KEYS` must also cover THIS row's
    sign and magnitude. Out of bounds it is reported as
    ``waiver HF-n bounds violated (sign|magnitude)`` and the row stays
    UNEXPLAINED — the measurement the waiver signed off is not the one in front
    of us. Several waivers may name the same row; the first one that both
    resolves and holds explains it.

    An ``expect`` glob does NOT explain anything; it only populates the advisory
    ``candidates`` column. Between them the registry's globs cover every known
    metric, so honouring them here made UNEXPLAINED unreachable.
    """
    matches = _matching_waivers(waivers, metric, family, key, run)
    if not matches:
        return "", False
    rejected = ""
    for w in matches:
        hid = str(w["hotfix"])
        ok, reason = explains(hid, hotfixes)
        if not ok:
            rejected = rejected or f"{hid}: {reason}"
            continue
        broken = _bounds_violation(w, delta, pct)
        if broken is None:
            return hid, True
        rejected = rejected or f"waiver {hid} bounds violated ({broken})"
    return rejected, False


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
    run: dict | None = None,
) -> tuple[str, str]:
    """(verdict, hotfix cell) for one comparison row."""
    m_na, d_na = np.isnan(master), np.isnan(develop)
    if m_na and d_na:
        return "undefined", "undefined on both sides"
    if m_na or d_na:
        return "one-sided", "develop only" if m_na else "master only"
    if abs(delta) <= tol.atol or (np.isfinite(pct) and abs(pct) <= tol.rtol * 100.0):
        return "equivalent", ""
    cell, ok = _explanation(metric, family, key, hotfixes, waivers, run, delta=delta, pct=pct)
    return ("explained" if ok else "UNEXPLAINED"), cell


def comparison_table(
    metrics: dict[str, pd.DataFrame],
    hotfixes: dict[str, dict],
    waivers: list[dict] | None = None,
    missing: list[dict] | None = None,
    interconnect: str | None = None,
    prong: int | None = None,
) -> pd.DataFrame:
    """One row per ``(metric, key)`` with a verdict.

    ``metrics`` maps a metric name to a frame from ``metrics.py`` (columns
    ``master``/``develop``/``delta``/``delta_pct``); the objective Series is
    accepted too and appears as the single key ``total_system_cost``.

    ``hotfixes`` is the ``hotfixes.yaml`` registry as ``{id: entry}``; pass
    ``{}`` before that file exists, in which case nothing can be explained. It
    is used for two separate things: to VALIDATE the ``hotfix:`` tag on a waiver
    (the only thing that can grant ``explained``), and to fill the advisory
    ``candidates`` column from the hot-fixes whose ``expect`` claims the row.
    ``waivers`` is optional and only its ``hotfix:`` field is consulted, and only
    on waivers that name a metric/key/family. ``interconnect`` and ``prong``
    scope them to this run: a waiver naming either must agree, exactly as
    ``compare.is_waived`` scopes cell waivers. Leave them ``None`` and an
    unscoped waiver still applies, while a scoped one is refused — its scope
    cannot be checked, so it has not been shown to apply. ``missing`` is the list of metrics
    that could not be computed (``plots.collect_metrics`` fills it); each becomes
    a ``MISSING`` row so that a criterion cannot vanish from the table silently.

    A ``delta_pct`` of NaN with a finite master of 0 means an
    appear-from-nothing difference; it is over tolerance unless the absolute
    floor covers it. NaN on a side means the metric is undefined there.

    Rows are sorted by :data:`VERDICT_ORDER`, then ``|delta_pct|`` descending.
    """
    waivers = waivers or []
    run = {"interconnect": interconnect, "prong": prong}
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
                metric,
                family,
                label,
                master,
                develop,
                delta,
                pct,
                tol,
                hotfixes,
                waivers,
                run,
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
                    "candidates": ",".join(candidate_hotfixes(metric, label, hotfixes)),
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
                "candidates": "",
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


def _big(x) -> str:
    """One large quantity, thousands-separated and unrounded, for a detail line.

    ``_fmt`` switches to ``%.4g`` above 1e5, which turns 620,823,223 MWh into
    ``6.208e+08``; a reader comparing two annual totals needs the digits.
    """
    v = _f(x)
    return "n/a" if not np.isfinite(v) else f"{v:,.0f}"


def _mw(x) -> str:
    """One capacity in MW, thousands-separated, for a one-line detail."""
    return f"{_big(x)} MW"


def _cluster_list(d: object) -> str:
    """``{cluster: MW}`` as ``p87 0 (2,158 MW), ...``; ``none`` when empty."""
    if not isinstance(d, dict) or not d:
        return "none"
    return ", ".join(f"{k} ({_mw(v)})" for k, v in sorted(d.items()))


def _signed_pct(delta: float, pct: float) -> str:
    """``pct`` carrying the sign of ``delta``; findings report it unsigned."""
    if not np.isfinite(pct):
        return "n/a %"
    sign = "-" if delta < 0 else "+"
    return f"{sign}{abs(pct):.4g} %"


def _generic_detail(detail: object, keys: int = 4) -> str:
    """Any other finding's detail, condensed: the first few ``k=v`` pairs."""
    if isinstance(detail, dict):
        if not detail:
            return "-"
        shown = [f"{k}={detail[k]}" for k in list(detail)[:keys]]
        more = len(detail) - len(shown)
        return ", ".join(shown) + (f", +{more} more" if more > 0 else "")
    text = str(detail)
    return text if len(text) <= 160 else text[:157] + "..."


def finding_detail(f: dict) -> str:
    """One finding's ``detail`` as a single human-readable line.

    The components a reader has to be able to judge from ``comparison.md``
    alone get a purpose-written line; everything else is condensed generically.
    ``cluster_set`` is the one this function exists for: after the prong-2
    rollup master is restricted to the common clusters, so a develop-only
    cluster makes no comparison-table row move at all and used to appear
    nowhere but ``findings_<prong>.json``.
    """
    detail = f.get("detail")
    d = detail if isinstance(detail, dict) else {}
    component = f.get("component")
    if component == "cluster_set" and d:
        return (
            f"{d.get('n_common')} common clusters (master {d.get('n_master')}, "
            f"develop {d.get('n_develop')}); develop-only: {_cluster_list(d.get('only_develop'))}; "
            f"master-only: {_cluster_list(d.get('only_master'))}"
        )
    if component == "system_potential_mw" and d:
        dev, mas = _f(d.get("develop")), _f(d.get("master"))
        return f"develop {_mw(dev)} vs master {_mw(mas)} ({_signed_pct(dev - mas, _f(d.get('rel_pct')))})"
    if component == "system_available_mw" and d:
        dev, mas = _f(d.get("develop_total_mwh")), _f(d.get("master_total_mwh"))
        ew = _f(d.get("energy_weighted_mean_rel_pct"))
        return (
            f"develop {_big(dev)} vs master {_big(mas)} MWh/y "
            f"(total {_signed_pct(dev - mas, abs(_f(d.get('total_rel_pct'))))}), "
            f"energy-weighted mean hourly {_fmt(ew)} %, "
            f"{d.get('hours_mismatched')}/{d.get('hours_compared')} h mismatched"
        )
    return _generic_detail(detail)


def finding_verdict(f: dict) -> str:
    """``waived <HF-id>`` / ``LIVE`` / ``LIVE (bounds violated: <bound>)``."""
    if f.get("waived"):
        who = f.get("waiver")
        return f"waived {who}" if who else "waived"
    failed = f.get("waiver_bound_failed")
    return f"LIVE (bounds violated: {failed})" if failed else "LIVE"


#: Columns of ``tables/findings.csv`` and of the markdown Findings section.
FINDINGS_COLUMNS = ("stage", "component", "column", "kind", "detail", "verdict")


def findings_rows(result: object) -> list[dict]:
    """One row per cell finding, in the order ``compare`` emitted them."""
    findings = result.get("findings") if isinstance(result, dict) else None
    return [
        {
            "stage": f.get("stage"),
            "component": f.get("component"),
            "column": f.get("column"),
            "kind": f.get("kind"),
            "detail": finding_detail(f),
            "verdict": finding_verdict(f),
        }
        for f in (findings or [])
        if isinstance(f, dict)
    ]


def findings_frame(result: object) -> pd.DataFrame:
    """``findings.csv``: the stage-by-stage findings with their verdicts."""
    return pd.DataFrame(findings_rows(result), columns=list(FINDINGS_COLUMNS))


def _cluster_notes(result: object) -> list[dict]:
    """The per-stage prong-2 rollup facts, from ``run_meta`` notes or findings.

    ``compare.run_comparison`` puts them in ``profile_cluster_sets`` whether or
    not the two cluster sets differed; a result carrying only findings (a
    hand-assembled one, or an older ``findings_<prong>.json``) falls back to the
    ``cluster_set`` findings, which carry the same fields.
    """
    if not isinstance(result, dict):
        return []
    notes = [n for n in (result.get("profile_cluster_sets") or []) if isinstance(n, dict) and n.get("rolled_up")]
    if notes:
        return notes
    return [
        {"stage": f.get("stage"), **f["detail"]}
        for f in (result.get("findings") or [])
        if isinstance(f, dict) and f.get("component") == "cluster_set" and isinstance(f.get("detail"), dict)
    ]


def cluster_note_lines(result: object) -> list[str]:
    """One line per rolled-up profile stage naming the population it used.

    A capacity-factor quantile taken over 18 clusters and one taken over 19 are
    not the same number, and the cluster the 19th stands for carries real MW.
    Since the rollup restricts master to the common set, that MW moves no row in
    the comparison table — so it has to be said in words, here.
    """
    out: list[str] = []
    for n in _cluster_notes(result):
        if n.get("n_common") is None:
            continue
        parts = []
        if n.get("only_develop"):
            parts.append(f"develop-only {_cluster_list(n['only_develop'])}")
        if n.get("only_master"):
            parts.append(f"master-only {_cluster_list(n['only_master'])}")
        out.append(
            f"Profile metrics use the {n['n_common']} common clusters ({n.get('stage')}); "
            f"one-sided clusters: {'; '.join(parts) if parts else 'none'} (see Findings).",
        )
    return out


def _md_cell(text: object) -> str:
    """Escape a value so it survives a markdown table cell intact.

    ``|`` would end the cell, and ``<index>`` — the column label every row-set
    finding carries — renders as an unknown HTML tag, i.e. as nothing at all.
    A disclosure section that silently drops the word it is disclosing is worse
    than no section.
    """
    out = str("-" if text is None else text)
    for a, b in (("|", "\\|"), ("<", "&lt;"), (">", "&gt;"), ("\n", " ")):
        out = out.replace(a, b)
    return out


def findings_markdown(result: object, cap: int = MD_ROW_CAP) -> list[str]:
    """The **Findings** section: every cell finding and why it is or is not live.

    ``comparison.md`` is the human-facing artifact, and until this section
    existed a finding that moved no comparison-table row — a develop-only
    cluster, a potential-MW delta — was visible only in ``findings_<prong>.json``
    and ``run_meta.json``. The section is written even when empty, so "nothing
    here" is a statement the run made rather than a section that failed to
    render.
    """
    lines = ["## Findings", ""]
    rows = findings_rows(result)
    if not rows:
        return [*lines, "_No stage-by-stage findings were recorded for this run._", ""]
    lines += [
        "Stage-by-stage artifact differences, independent of the comparison table above. "
        "`waived <HF-id>` cites the waiver that covers it; `LIVE` is unexplained.",
        "",
        "| " + " | ".join(FINDINGS_COLUMNS) + " |",
        "|" + "---|" * len(FINDINGS_COLUMNS),
    ]
    for r in rows[:cap]:
        lines.append("| " + " | ".join(_md_cell(r[c]) for c in FINDINGS_COLUMNS) + " |")
    if len(rows) > cap:
        lines += ["", f"_{len(rows) - cap} more findings in findings.csv_"]
    return [*lines, ""]


def to_markdown(table: pd.DataFrame, cap: int = MD_ROW_CAP, result: object = None) -> str:
    """Render the comparison table as markdown, capped at ``cap`` rows.

    ``result`` is ``compare.run_comparison``'s dict; when given, the verdict
    summary is followed by the cluster note and the **Findings** section, so a
    reader of this one file sees the differences the table cannot show.
    """
    c = verdict_counts(table)
    head = [
        "# Comparison: master-benchmark (baseline) vs develop",
        "",
        f"`delta` is **develop minus master**. Verdicts: {c['equivalent']} equivalent, "
        f"{c['explained']} explained, {c['one-sided']} one-sided, {c['undefined']} undefined, "
        f"**{c['UNEXPLAINED']} UNEXPLAINED**, **{c['MISSING']} MISSING**.",
        "",
    ]
    notes = cluster_note_lines(result)
    if notes:
        head += [*(f"_{line}_" for line in notes), ""]
    if result is not None:
        head += findings_markdown(result, cap)
    head += [
        "## Comparison table",
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


def write_tables(tables: dict[str, pd.DataFrame], outdir: Path, result: object = None) -> list[Path]:
    """Write ``<outdir>/tables/<name>.csv`` per frame, plus ``comparison.md``.

    ``tables`` must contain a ``comparison`` frame for the markdown to be
    written; every other entry is dumped as CSV beside it so that every number
    quoted anywhere has a machine-readable home.

    ``result`` is ``compare.run_comparison``'s dict. Given one, the findings get
    a machine-readable home of their own (``findings.csv``) and a human-readable
    one in ``comparison.md``'s **Findings** section — the two views of the same
    rows, written from the same function so they cannot disagree.
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
    if result is not None:
        fc = tdir / "findings.csv"
        findings_frame(result).to_csv(fc, index=False)
        written.append(fc)
    if "comparison" in tables and tables["comparison"] is not None:
        md = tdir / "comparison.md"
        md.write_text(to_markdown(tables["comparison"], result=result))
        written.append(md)
    return written


def export_all(run_dir: Path, ctx: object | None = None, result: dict | None = None) -> pd.DataFrame:
    """Build the comparison table for one run and write ``<run_dir>/tables/``.

    The entry point ``run.py`` calls. Metrics come from
    :func:`plots.run_metrics`, which memoises them for the process, so the table
    and the figures are drawn from the same numbers and the networks are read
    once. ``result`` is the findings dict from ``compare.run_comparison``; it is
    not needed to build the table, but it is what the **Findings** section of
    ``comparison.md`` and ``findings.csv`` are written from. Returns the
    comparison frame.

    The run's interconnect and prong come off ``ctx`` and scope the waivers, so
    a waiver written for the deferred western prong-1 leg cannot sign off a
    whole-USA prong-2 difference.
    """
    from . import plots
    from .compare import load_waivers
    from .paths import INTERCONNECT

    prong = int(getattr(ctx, "prong", 2))
    interconnect = str(getattr(ctx, "interconnect", None) or INTERCONNECT)
    _artifacts, frames, missing = plots.run_metrics(prong)
    comparison = comparison_table(
        frames,
        load_hotfixes(),
        load_waivers(),
        missing,
        interconnect=interconnect,
        prong=prong,
    )
    written = write_tables({**frames, "comparison": comparison}, run_dir, result=result)
    print(f"[equivalence] tables: {len(written)} file(s) under {Path(run_dir) / 'tables'}")
    return comparison
