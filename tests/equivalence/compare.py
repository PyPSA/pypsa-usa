"""Stage-by-stage artifact comparison for the equivalence harness.

Tolerance policy (spec D2/D7):
- floats: ``np.allclose(rtol=1e-3, atol=1e-8, equal_nan=True)``
- indexes / integers / strings: exact, after sorting
- solved network: objective and per-carrier capacity within
  ``tables.TOLERANCES``; the objective is normalized to
  ``objective + objective_constant`` on both sides, because the branches split
  the total between those two terms differently (HF-13) while the sum is
  invariant
- row-set and column-set differences are first-class findings
- representation-only differences are normalized before comparing: float-
  formatted integer labels ('35827.0' vs '35827') and Load names carrying a
  carrier suffix ('35827 AC' vs '35827', re-keyed on the load's bus)
- per-frame findings are capped at ``MAX_FINDINGS_PER_FRAME`` with a single
  'suppressed' finding so fan-out cannot drown the signal
- waivers (tests/equivalence/waivers.yaml) suppress exactly the signed-off
  deltas; each waiver must reference a deltas-ledger entry. A cell waiver may
  BOUND what it explains, with the bounds its finding's COMPONENT knows how to
  read (``CELL_BOUND_SPECS``); a finding outside those bounds stays live and
  carries a ``waiver_note`` naming the bound that failed.

Findings are dicts: {stage, component, column, kind, detail, waived}, plus
``waiver`` naming the entry that waived it, or ``waiver_note`` and
``waiver_bound_failed`` when a matching waiver's bounds were broken.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from .metrics import (
    aggregate_profile_to_clusters,
    cluster_sets,
    objective_constant,
    total_objective,
)
from .paths import EQ, INTERCONNECT, SIMPL2, UNTIL, ArtifactPair, prong_pairs, run_dir
from .tables import TOLERANCES, WAIVER_MATCH_KEYS

# Every tolerance below comes from ``tables.TOLERANCES``; none is restated here,
# so the stage-by-stage findings and the comparison table cannot disagree about
# what "within tolerance" means (T3 of memory/plans/harness-master-vs-develop.md).
# The generic per-cell tolerance is the p_max_pu family's, which is the same
# 1e-3 the p_nom_max and demand families use — those are the frames this
# stage-by-stage pass actually walks.
RTOL = TOLERANCES["p_max_pu"].rtol
OBJECTIVE_RTOL = TOLERANCES["objective"].rtol
CAPACITY_RTOL = TOLERANCES["capacity"].rtol
CAPACITY_ATOL = TOLERANCES["capacity"].atol
# The two system aggregates below are both in MW, not per-unit: sum(p_nom_max)
# is installable potential and sum(profile x p_nom_max) is national available
# power. So they take the 1 MW capacity floor, NOT the p_max_pu family's 1e-3,
# which is a floor on a capacity FACTOR and would be a thousandth of a watt
# here. Without a floor at all — which is how sum(p_nom_max) was checked — a
# carrier neither side built reads as a 100 % difference on solver noise.
POTENTIAL_ATOL = TOLERANCES["p_nom_max"].atol
PROFILE_ATOL = TOLERANCES["capacity"].atol
# Float-equality epsilon for cell comparison. Not a physical floor — the
# per-family physical floors are ``tables.TOLERANCES[...].atol``.
ATOL = 1e-8
MAX_FINDINGS_PER_FRAME = 50
WAIVERS_PATH = Path(__file__).parent / "waivers.yaml"

# Master artifacts float-format integer bus labels ('35827.0'); develop
# writes them bare ('35827'). Pure representation — normalize on both sides.
_FLOAT_INT_LABEL = re.compile(r"\d+\.0")


def _norm_label(x) -> str:
    """Str-ify a label, collapsing float-formatted integers to integer form."""
    s = str(x)
    return s[:-2] if _FLOAT_INT_LABEL.fullmatch(s) else s


def load_network(path: Path):
    import pypsa

    # Keep network frames on numpy object dtype under pandas 3 (matches
    # _helpers), so develop and master networks compare on equal footing.
    if hasattr(pypsa, "options"):
        pypsa.options.api.legacy_string_dtype = True

    if path.suffix == ".pkl":
        import dill

        with open(path, "rb") as fh:
            return dill.load(fh)
    return pypsa.Network(str(path))


def load_waivers() -> list[dict]:
    if WAIVERS_PATH.exists():
        return yaml.safe_load(WAIVERS_PATH.read_text()) or []
    return []


def load_busmap(develop_root: Path, simpl: str = SIMPL2) -> pd.Series | None:
    """Develop's ``busmap_s{simpl}.csv`` as substation id -> cluster bus.

    ``None`` when the file is absent (prong 1 never needs it, and a run stopped
    before ``cluster_resources`` has not written one). Both the index and the
    values are strings; the substation ids are bare ('35827') while master's
    profile bus labels are float-formatted ('35827.0'), which
    ``metrics.normalize_bus_ids`` reconciles at join time.
    """
    p = Path(develop_root) / f"{EQ}/busmaps/{INTERCONNECT}/busmap_s{simpl}.csv"
    if not p.exists():
        return None
    bm = pd.read_csv(p, index_col=0, dtype=str).iloc[:, 0]
    bm.index = bm.index.astype(str)
    return bm.astype(str)


def _num(x) -> float:
    """``x`` as a float, or NaN when it is not a number."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _sign(delta: float) -> str:
    return "+" if delta > 0 else "-" if delta < 0 else "0"


def _bound_system_available_mw(waiver: dict, detail: dict) -> str | None:
    """Bounds on ``sum_bus(profile * p_nom_max)``: annual totals + hourly mean.

    ``expect_sign``       sign of ``develop_total_mwh - master_total_mwh``.
    ``max_total_pct``     ``abs(annual total delta) / master``, %.
    ``max_mean_rel_pct``  the finding's ``mean_rel_pct``.

    ``worst_rel_pct`` is deliberately NOT boundable: at dawn and dusk master's
    available power is a few MW, so a fraction of a MW is a 100 % relative error
    and the statistic says nothing about the energy that differs.
    """
    dev, mas = _num(detail.get("develop_total_mwh")), _num(detail.get("master_total_mwh"))
    if {"expect_sign", "max_total_pct"} & set(waiver) and not (np.isfinite(dev) and np.isfinite(mas)):
        return "unmeasurable"
    if "expect_sign" in waiver and _sign(dev - mas) != waiver["expect_sign"]:
        return "expect_sign"
    if "max_total_pct" in waiver:
        cap = _num(waiver["max_total_pct"])
        pct = abs(dev - mas) / abs(mas) * 100.0 if mas else float("inf")
        if not np.isfinite(cap) or not np.isfinite(pct) or pct > cap:
            return "max_total_pct"
    if "max_mean_rel_pct" in waiver:
        cap = _num(waiver["max_mean_rel_pct"])
        mean_rel = _num(detail.get("mean_rel_pct"))
        if not np.isfinite(mean_rel):
            return "unmeasurable"
        if not np.isfinite(cap) or mean_rel > cap:
            return "max_mean_rel_pct"
    return None


def _bound_system_potential_mw(waiver: dict, detail: dict) -> str | None:
    """Bounds on ``sum(p_nom_max)``: the sign and size of the potential delta.

    ``expect_sign``   sign of ``develop - master``.
    ``max_abs_pct``   upper bound on the finding's ``abs(rel_pct)``.

    The finding carries no annual totals, so the ``system_available_mw`` bounds
    are unreadable here — which is exactly why they are rejected for this
    component rather than silently evaluated against absent fields.
    """
    dev, mas = _num(detail.get("develop")), _num(detail.get("master"))
    if "expect_sign" in waiver:
        if not (np.isfinite(dev) and np.isfinite(mas)):
            return "unmeasurable"
        if _sign(dev - mas) != waiver["expect_sign"]:
            return "expect_sign"
    if "max_abs_pct" in waiver:
        cap = _num(waiver["max_abs_pct"])
        pct = abs(_num(detail.get("rel_pct")))
        if not np.isfinite(pct):
            return "unmeasurable"
        if not np.isfinite(cap) or pct > cap:
            return "max_abs_pct"
    return None


#: Values ``expect_side`` may take: which side a one-sided cluster may sit on.
CLUSTER_SIDES = ("develop", "master", "either")


def _bound_cluster_set(waiver: dict, detail: dict) -> str | None:
    """Bounds on the prong-2 cluster row-set: which side, and how much capacity.

    ``expect_side``        ``develop`` / ``master`` / ``either`` — the side that
                           may carry one-sided clusters. A cluster appearing on
                           the OTHER side is a different phenomenon from the one
                           the waiver signed off, so it stays live.
    ``max_one_sided_mw``   upper bound on the ``p_nom_max`` summed over BOTH
                           ``only_*`` lists.
    ``max_one_sided_pct``  that same sum as a percentage of ``common_total_mw``.

    A one-sided cluster whose MW is not known (``NaN``) makes both magnitude
    bounds unmeasurable: the waiver claims a size, and an unknown size is not a
    small one.
    """
    sides = {s: detail.get(f"only_{s}") for s in ("develop", "master")}
    if any(not isinstance(v, dict) for v in sides.values()):
        return "unmeasurable"
    if "expect_side" in waiver:
        want = waiver["expect_side"]
        if want not in CLUSTER_SIDES:
            return "expect_side"
        if want != "either" and sides["master" if want == "develop" else "develop"]:
            return "expect_side"
    if not {"max_one_sided_mw", "max_one_sided_pct"} & set(waiver):
        return None
    mws = [_num(v) for side in sides.values() for v in side.values()]
    if any(not np.isfinite(m) for m in mws):
        return "unmeasurable"
    total = float(sum(mws))
    if "max_one_sided_mw" in waiver:
        cap = _num(waiver["max_one_sided_mw"])
        if not np.isfinite(cap) or total > cap:
            return "max_one_sided_mw"
    if "max_one_sided_pct" in waiver:
        cap = _num(waiver["max_one_sided_pct"])
        common = _num(detail.get("common_total_mw"))
        if not np.isfinite(common) or common <= 0:
            return "unmeasurable"
        if not np.isfinite(cap) or total / common * 100.0 > cap:
            return "max_one_sided_pct"
    return None


#: Optional BOUNDS a CELL waiver may put on the finding it explains, per
#: finding COMPONENT: ``component -> (allowed bound keys, checker)``.
#:
#: A cell waiver used to be matched on stage/component/column/kind alone, which
#: made it a blank cheque exactly as an unbounded table waiver was: the HF-24
#: entry for the two ``system_available_mw`` findings was written for "+0.25 %
#: of annual energy, from 450 MW of onwind and 2,566 MW of solar master drops"
#: - and unbounded it would equally explain a -30 % row, which is what a real
#: capacity-factor construction bug looks like.
#:
#: The bounds were then evaluated against ``develop_total_mwh`` /
#: ``master_total_mwh`` / ``mean_rel_pct`` alone, which only
#: ``system_available_mw`` findings carry — so the HF-24 waivers on
#: ``cluster_set`` (whose detail is ``only_develop`` / ``only_master`` dicts of
#: MW) and on ``system_potential_mw`` (``develop`` / ``master`` / ``rel_pct``)
#: could not be bounded AT ALL and stayed blank cheques: a develop-only cluster
#: of 50 GW, or a -44 % potential delta, was waived by the entry written for
#: 2,158 MW and +0.19 %. Each component now declares how its own detail is read.
#:
#: A component absent from this registry has no readable bounds, so naming one
#: on a waiver for it is rejected by ``test_waiver_bounds_are_well_formed``
#: rather than silently evaluating to "unmeasurable" at run time.
CELL_BOUND_SPECS: dict[str, tuple[tuple[str, ...], object]] = {
    "system_available_mw": (
        ("expect_sign", "max_total_pct", "max_mean_rel_pct"),
        _bound_system_available_mw,
    ),
    "system_potential_mw": (("expect_sign", "max_abs_pct"), _bound_system_potential_mw),
    "cluster_set": (("expect_side", "max_one_sided_mw", "max_one_sided_pct"), _bound_cluster_set),
}

#: Every bound key any component accepts, in registry order. The union is what
#: ``waivers.yaml`` may mention at all; :func:`cell_bound_keys_for` says which
#: subset is readable for a given component.
CELL_BOUND_KEYS = tuple(dict.fromkeys(k for keys, _ in CELL_BOUND_SPECS.values() for k in keys))

#: Components whose cell waivers must be bounded. Each is a finding that would
#: CATCH a real construction difference, so waiving one unbounded hollows out
#: the comparison it belongs to.
BOUNDED_COMPONENTS = tuple(CELL_BOUND_SPECS)


def cell_bound_keys_for(component: str | None) -> tuple[str, ...]:
    """The bound keys a cell waiver on ``component`` may carry (``()`` if none)."""
    spec = CELL_BOUND_SPECS.get(str(component or ""))
    return spec[0] if spec else ()


def _cell_bounds_violation(waiver: dict, finding: dict) -> str | None:
    """Which bound this waiver puts on the finding is broken, or ``None``.

    Returns the name of the failed bound, ``'unmeasurable'`` when the finding
    does not carry what the bound needs, or ``unreadable bound ...`` when the
    waiver names a bound this finding's component has no way to evaluate. A
    waiver carrying no bound at all can never violate one, so every pre-existing
    entry in ``waivers.yaml`` keeps behaving exactly as before.

    A bound that cannot be evaluated is a violation, not a pass: a waiver is a
    claim about a measured quantity, and a claim that cannot be checked has not
    been shown to hold here.
    """
    named = [k for k in CELL_BOUND_KEYS if k in waiver]
    if not named:
        return None
    component = str(finding.get("component") or "")
    spec = CELL_BOUND_SPECS.get(component)
    if spec is None:
        return f"unreadable bound {named[0]!r} for component {component!r}"
    allowed, check = spec
    unknown = [k for k in named if k not in allowed]
    if unknown:
        return f"unreadable bound {unknown[0]!r} for component {component!r}"
    detail = finding.get("detail")
    return check(waiver, detail if isinstance(detail, dict) else {})


def matching_waiver(finding: dict, waivers: list[dict]) -> tuple[dict | None, str | None, str | None]:
    """``(waiver that waives this finding, note, failed bound)``.

    The single place the cell-waiver match is decided; :func:`waiver_status` and
    ``run_comparison`` both read it, so the comparison table's verdict and the
    findings JSON can never disagree about WHICH waiver covered a finding.
    """
    note = failed = None
    for w in waivers:
        if any(k in w for k in WAIVER_MATCH_KEYS):
            continue
        if not all(
            w.get(k) in (None, "*", finding.get(k))
            for k in ("stage", "component", "column", "kind", "prong", "interconnect")
        ):
            continue
        bad = _cell_bounds_violation(w, finding)
        if bad is None:
            return w, None, None
        if note is None:
            who = w.get("hotfix") or w.get("ledger") or "waiver"
            note, failed = f"waiver {who} bounds violated ({bad})", bad
    return None, note, failed


def waiver_id(waiver: dict) -> str:
    """How a waiver is named in a verdict: its hot-fix, else its ledger row."""
    return str(waiver.get("hotfix") or waiver.get("ledger") or "waiver")


def waiver_status(finding: dict, waivers: list[dict]) -> tuple[bool, str | None]:
    """``(waived, note)`` for one finding under the CELL waivers.

    A waiver that names a comparison-table row (``metric``/``key``/``family``,
    :data:`tables.WAIVER_MATCH_KEYS`) is skipped outright. Its four cell fields
    are absent, and absent means "any" here — so without this guard a single
    ``{metric: p_nom_max_by_zone_onwind, prong: 2, interconnect: western}``
    table waiver would silently waive EVERY western prong-2 finding.

    MENTIONING one of those keys is enough to be skipped, whatever the value.
    Reading ``metric: '*'`` as "not a table waiver" put the worst case back:
    ``{metric: '*', key: '*', prong: 2, interconnect: western}`` names no cell
    field at all, so every absent field wildcards and the entry waives every
    western prong-2 finding — including ``system_available_mw``, the one that
    would catch a real capacity-factor difference. A waiver that names a metric
    is about the comparison table, even when the metric it names is "any".

    A matching waiver that carries :data:`CELL_BOUND_KEYS` waives the finding
    only while the finding stays inside those bounds. Outside them the finding
    stays LIVE and ``note`` names the bound that failed, so the reason it was
    not waived is on the record instead of being inferred from a silence.

    Which bounds a waiver MAY carry depends on the finding's component
    (:data:`CELL_BOUND_SPECS`): the three annual-total bounds read only a
    ``system_available_mw`` detail, so ``cluster_set`` and
    ``system_potential_mw`` have their own.
    """
    w, note, _ = matching_waiver(finding, waivers)
    return w is not None, note


def is_waived(finding: dict, waivers: list[dict]) -> bool:
    """Does any in-bounds CELL waiver cover this finding. See :func:`waiver_status`."""
    return waiver_status(finding, waivers)[0]


def _numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _rel_pct(a: float, b: float) -> float | None:
    """Relative difference of a vs master b, in percent (None if undefined)."""
    if np.isnan(a) or np.isnan(b) or b == 0:
        return None
    return round(abs(a - b) / abs(b) * 100.0, 3)


def _max_rel_pct(av, bv) -> float | None:
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(av - bv) / np.abs(bv)
    rel = rel[np.isfinite(rel)]
    return round(float(rel.max()) * 100.0, 3) if rel.size else None


def _normalize_frame(
    df: pd.DataFrame,
    side: str,
    stage: str,
    component: str,
    local: list[dict],
) -> pd.DataFrame:
    """Copy with labels normalized; dedup (with a finding) if labels collide."""
    df = df.copy()
    df.index = df.index.map(_norm_label)
    df.columns = df.columns.map(_norm_label)
    for axis, labels in (("index", df.index), ("columns", df.columns)):
        if labels.has_duplicates:
            dups = sorted(set(labels[labels.duplicated()]))[:5]
            local.append(
                {
                    "stage": stage,
                    "component": component,
                    "column": f"<{axis}>",
                    "kind": "duplicate_labels",
                    "detail": f"{side} {axis} labels collide after normalization: {dups}",
                },
            )
    if df.index.has_duplicates:
        df = df.loc[~df.index.duplicated()]
    if df.columns.has_duplicates:
        df = df.loc[:, ~df.columns.duplicated()]
    return df


def compare_frames(
    stage: str,
    component: str,
    dev: pd.DataFrame,
    mas: pd.DataFrame,
    findings: list[dict],
) -> None:
    """Compare two indexed DataFrames; append findings in place.

    Labels are normalized on both sides (float-formatted integers collapse to
    integer form). A row-set mismatch is reported once and the comparison
    proceeds on the intersection. Findings for one frame are capped at
    ``MAX_FINDINGS_PER_FRAME`` with a single 'suppressed' finding.
    """
    local: list[dict] = []
    dev = _normalize_frame(dev, "develop", stage, component, local)
    mas = _normalize_frame(mas, "master", stage, component, local)
    ci, ai = set(dev.index), set(mas.index)
    if ci != ai:
        local.append(
            {
                "stage": stage,
                "component": component,
                "column": "<index>",
                "kind": "row_set",
                "detail": f"develop-only={sorted(ci - ai)[:8]} (n={len(ci - ai)}), "
                f"master-only={sorted(ai - ci)[:8]} (n={len(ai - ci)})",
            },
        )
    common = sorted(ci & ai)
    if not common:
        findings.extend(local)
        return
    dev = dev.loc[common]
    mas = mas.loc[common]

    cc, ac = set(dev.columns), set(mas.columns)
    for col in sorted(cc ^ ac, key=str):
        local.append(
            {
                "stage": stage,
                "component": component,
                "column": str(col),
                "kind": "column_set",
                "detail": "develop-only" if col in cc else "master-only",
            },
        )
    for col in sorted(cc & ac, key=str):
        a, b = dev[col], mas[col]
        if _numeric(a) and _numeric(b):
            av, bv = a.astype(float).to_numpy(), b.astype(float).to_numpy()
            close = np.isclose(av, bv, rtol=RTOL, atol=ATOL, equal_nan=True)
            if not close.all():
                bad = np.flatnonzero(~close)
                worst = bad[np.argsort(-np.abs(np.nan_to_num(av[bad] - bv[bad])))[:5]]
                local.append(
                    {
                        "stage": stage,
                        "component": component,
                        "column": str(col),
                        "kind": "value",
                        "detail": {
                            "n_diff": len(bad),
                            "n_total": len(av),
                            "max_abs": float(np.nanmax(np.abs(av[bad] - bv[bad]))),
                            "max_rel_pct": _max_rel_pct(av[bad], bv[bad]),
                            "examples": [
                                {
                                    "id": common[i],
                                    "develop": None if np.isnan(av[i]) else float(av[i]),
                                    "master": None if np.isnan(bv[i]) else float(bv[i]),
                                    "rel_pct": _rel_pct(av[i], bv[i]),
                                }
                                for i in worst
                            ],
                        },
                    },
                )
        else:
            av = a.fillna("<NA>").astype(str)
            bv = b.fillna("<NA>").astype(str)
            neq = av != bv
            if neq.any():
                ex = av.index[neq][:5]
                local.append(
                    {
                        "stage": stage,
                        "component": component,
                        "column": str(col),
                        "kind": "value",
                        "detail": {
                            "n_diff": int(neq.sum()),
                            "n_total": len(av),
                            "examples": [{"id": str(i), "develop": av[i], "master": bv[i]} for i in ex],
                        },
                    },
                )
    if len(local) > MAX_FINDINGS_PER_FRAME:
        n_more = len(local) - MAX_FINDINGS_PER_FRAME
        local = local[:MAX_FINDINGS_PER_FRAME]
        local.append(
            {
                "stage": stage,
                "component": component,
                "column": "<frame>",
                "kind": "suppressed",
                "detail": f"suppressed {n_more} more findings for this frame",
            },
        )
    findings.extend(local)


def _loads_by_bus(n) -> tuple[pd.DataFrame, dict[str, str]] | None:
    """Static Load frame re-keyed on bus, plus name->bus map for _t columns.

    Master Load names carry a carrier suffix ('35827 AC') while develop
    names are bare bus ids — re-keying on the ``bus`` attribute makes the two
    comparable. Returns None when loads are not one-per-bus (caller falls
    back to name comparison and emits a finding).
    """
    loads = n.loads
    buses = loads["bus"].map(_norm_label)
    if buses.duplicated().any():
        return None
    static = loads.copy()
    static["bus"] = buses.to_numpy()
    static.index = pd.Index(buses.to_numpy(), name=loads.index.name)
    mapping = {str(name): bus for name, bus in zip(loads.index, buses)}
    return static, mapping


def _generator_p_nom_by_bus_carrier(n) -> pd.DataFrame:
    """p_nom summed by (bus, carrier) — generator-name independent."""
    g = n.generators
    if g.empty:
        return pd.DataFrame(columns=["p_nom"])
    key = g["bus"].map(_norm_label) + " | " + g["carrier"].astype(str)
    return g.groupby(key)["p_nom"].sum().to_frame("p_nom")


def compare_networks(pair: ArtifactPair, nc, na, findings: list[dict]) -> None:
    if list(map(str, nc.snapshots)) != list(map(str, na.snapshots)):
        findings.append(
            {
                "stage": pair.stage,
                "component": "Network",
                "column": "snapshots",
                "kind": "row_set",
                "detail": f"develop n={len(nc.snapshots)}, master n={len(na.snapshots)}",
            },
        )
    if pair.solve_stage:
        _compare_solved(pair, nc, na, findings)
        return
    comps = sorted(
        {c.name for c in nc.components if not c.static.empty} | {c.name for c in na.components if not c.static.empty},
    )
    for name in comps:
        dfc, dfa = nc.components[name].static, na.components[name].static
        if dfc.empty and dfa.empty:
            continue
        map_c = map_a = None
        if name == "Load":
            kc, ka = _loads_by_bus(nc), _loads_by_bus(na)
            if kc is None or ka is None:
                bad = [s for s, k in (("develop", kc), ("master", ka)) if k is None]
                findings.append(
                    {
                        "stage": pair.stage,
                        "component": "Load",
                        "column": "<index>",
                        "kind": "load_rekey",
                        "detail": f"loads not one-per-bus on {', '.join(bad)}; falling back to name comparison",
                    },
                )
            else:
                dfc, map_c = kc
                dfa, map_a = ka
        compare_frames(pair.stage, name, dfc, dfa, findings)
        pnl_c, pnl_a = nc.components[name].dynamic, na.components[name].dynamic
        for attr in sorted(set(pnl_c) | set(pnl_a)):
            tc = pnl_c.get(attr, pd.DataFrame())
            ta = pnl_a.get(attr, pd.DataFrame())
            if tc.empty and ta.empty:
                continue
            if map_c is not None and map_a is not None:
                tc = tc.rename(columns={c: map_c.get(str(c), str(c)) for c in tc.columns})
                ta = ta.rename(columns={c: map_a.get(str(c), str(c)) for c in ta.columns})
            compare_frames(pair.stage, f"{name}_t.{attr}", tc, ta, findings)
        if name == "Generator":
            # Generator naming differs structurally between branches (plant-id
            # vs 'bus carrier' names); the aggregate makes content equality
            # visible regardless of names.
            compare_frames(
                pair.stage,
                "Generator[bus,carrier]",
                _generator_p_nom_by_bus_carrier(nc),
                _generator_p_nom_by_bus_carrier(na),
                findings,
            )


def _capacity_by_carrier(n) -> pd.DataFrame:
    out = {}
    for comp, cap in (("generators", "p_nom_opt"), ("storage_units", "p_nom_opt")):
        df = getattr(n, comp)
        if not df.empty and cap in df:
            out[comp] = df.groupby("carrier")[cap].sum()
    return pd.DataFrame(out).fillna(0.0)


def _objective_constant(n) -> float:
    """``objective_constant`` of a network, 0.0 when absent/NaN."""
    return objective_constant(n)


def _total_objective(n) -> float:
    """Total system cost, invariant to how the two terms are split.

    Both pypsa 0.30 and pypsa 1.3 expose ``objective`` and
    ``objective_constant`` separately (verified 2026-09-14). What differs
    between the branches is how the total is split between them: on the CA leg
    master reported ``-204,665,929.13`` with a constant of ``1,133,255,860.00``
    while develop reported ``928,590,425.01`` with a constant of ``0.00``.
    Comparing either term alone therefore manufactures a difference the size of
    the constant even when the two solves agree, so both sides are normalized to
    ``objective + objective_constant`` — the total system cost either way — and
    the tolerance applies to that. Each side's constant is read from its own
    file (missing -> 0.0).

    The implementation lives in ``metrics.total_objective`` so the findings and
    the comparison table normalise identically (HF-13).
    """
    return total_objective(n)


def _compare_solved(pair: ArtifactPair, nc, na, findings: list[dict]) -> None:
    oc, oa = _total_objective(nc), _total_objective(na)
    if not np.isclose(oc, oa, rtol=OBJECTIVE_RTOL):
        findings.append(
            {
                "stage": pair.stage,
                "component": "Network",
                "column": "objective",
                "kind": "value",
                "detail": {
                    "develop": oc,
                    "master": oa,
                    "develop_raw": float(nc.objective),
                    "master_raw": float(na.objective),
                    "develop_constant": _objective_constant(nc),
                    "master_constant": _objective_constant(na),
                    "rel": abs(oc - oa) / max(abs(oa), 1e-9),
                    "rel_pct": round(abs(oc - oa) / max(abs(oa), 1e-9) * 100.0, 4),
                },
            },
        )
    cc, ca = _capacity_by_carrier(nc), _capacity_by_carrier(na)
    both = cc.reindex(
        index=sorted(set(cc.index) | set(ca.index)),
        columns=sorted(set(cc.columns) | set(ca.columns)),
    ).fillna(0.0)
    mas = ca.reindex_like(both).fillna(0.0)
    close = np.isclose(both.to_numpy(), mas.to_numpy(), rtol=CAPACITY_RTOL, atol=CAPACITY_ATOL)
    if not close.all():
        rows, cols = np.where(~close)
        findings.append(
            {
                "stage": pair.stage,
                "component": "Network",
                "column": "p_nom_opt_by_carrier",
                "kind": "value",
                "detail": [
                    {
                        "carrier": str(both.index[r]),
                        "component": str(both.columns[c]),
                        "develop": float(both.iloc[r, c]),
                        "master": float(mas.iloc[r, c]),
                        "rel_pct": _rel_pct(float(both.iloc[r, c]), float(mas.iloc[r, c])),
                    }
                    for r, c in zip(rows, cols)
                ],
            },
        )


def compare_profiles(
    pair: ArtifactPair,
    pc: Path,
    pa: Path,
    findings: list[dict],
    prong: int = 1,
    busmap: pd.Series | None = None,
    notes: list[dict] | None = None,
) -> None:
    """Compare two renewable-profile files; append findings in place.

    ``notes`` collects non-finding facts about the comparison — whether the
    prong-2 rollup ran and what the two cluster sets were — for
    ``run_meta.json``. They are NOT findings: a rollup that ran is not a
    difference, and the cluster-set difference gets its own finding below.
    """
    with xr.open_dataset(pc) as dc, xr.open_dataset(pa) as da_raw:
        # At prong 2 master's file is NODAL (substation buses) and develop's is
        # at s{simpl} cluster resolution. sum(p_nom_max) and
        # sum_bus(profile*p_nom_max) are both aggregation-invariant, so they
        # would compare either way — but they must be computed from the same
        # object the table and the figures use, or a reader cannot tell whether
        # a residual is physics or resolution. So master is rolled up onto the
        # cluster bus space first (HF-24's silent-drop delta survives the
        # rollup: it is a real missing-capacity difference, not a resolution
        # artifact).
        da = da_raw
        rolled_up = prong == 2 and busmap is not None and "bus" in da_raw.dims
        if rolled_up:
            da = aggregate_profile_to_clusters(da_raw, busmap)
            # A cluster on one side only is a ROW-SET difference and is reported
            # as one. Before this finding existed the only trace of develop's
            # p87 0 (96 MW onwind / 2,158 MW solar, HF-24's bus 37808) was a
            # smear across every pooled quantile row, which reads as a
            # capacity-factor difference rather than as missing capacity.
            info = cluster_sets(da, dc)
            if notes is not None:
                notes.append({"kind": "profile_rollup", "stage": pair.stage, "rolled_up": True, **info})
            if not info["equal"]:
                findings.append(
                    {
                        "stage": pair.stage,
                        "component": "cluster_set",
                        "column": "<index>",
                        "kind": "row_set",
                        "detail": {
                            "n_master": info["n_master"],
                            "n_develop": info["n_develop"],
                            "n_common": info["n_common"],
                            "only_develop": info["only_develop"],
                            "only_master": info["only_master"],
                            "only_develop_mw": info["only_develop_mw"],
                            "only_master_mw": info["only_master_mw"],
                            # master's p_nom_max over the SHARED clusters: the
                            # baseline the one-sided MW is a fraction OF, so a
                            # waiver can bound it relatively (max_one_sided_pct)
                            # and not only in absolute MW.
                            "common_total_mw": info["common_total_mw"],
                            "note": (
                                "p_nom_max in MW per one-sided cluster; the pooled profile metrics "
                                "are computed over the common clusters only"
                            ),
                        },
                    },
                )
        elif notes is not None and prong == 2:
            notes.append({"kind": "profile_rollup", "stage": pair.stage, "rolled_up": False})
        if "p_nom_max" in dc and "p_nom_max" in da:
            tc = float(dc["p_nom_max"].sum())
            ta = float(da["p_nom_max"].sum())
            if not np.isclose(tc, ta, rtol=RTOL, atol=POTENTIAL_ATOL):
                findings.append(
                    {
                        "stage": pair.stage,
                        "component": "system_potential_mw",
                        "column": "sum(p_nom_max)",
                        "kind": "value",
                        "detail": {
                            "develop": tc,
                            "master": ta,
                            "rel": abs(tc - ta) / max(abs(ta), 1e-9),
                            "rel_pct": round(abs(tc - ta) / max(abs(ta), 1e-9) * 100.0, 4),
                        },
                    },
                )
            if "profile" in dc and "profile" in da:
                sc = (dc["profile"] * dc["p_nom_max"]).sum("bus").to_pandas()
                sa = (da["profile"] * da["p_nom_max"]).sum("bus").to_pandas()
                # Compare on positional hours: the two sides may label the
                # same weather differently (e.g. horizon-relabeled years).
                n = min(len(sc), len(sa))
                vc, va = sc.to_numpy()[:n], sa.to_numpy()[:n]
                bad = ~np.isclose(vc, va, rtol=RTOL, atol=PROFILE_ATOL)
                if len(sc) != len(sa) or bad.any():
                    denom = np.maximum(np.abs(va), 1e-9)
                    rel = np.abs(vc - va) / denom
                    master_energy = float(np.abs(va).sum())
                    findings.append(
                        {
                            "stage": pair.stage,
                            "component": "system_available_mw",
                            "column": "sum_bus(profile*p_nom_max)",
                            "kind": "value",
                            "detail": {
                                "hours_compared": int(n),
                                "hours_mismatched": int(bad.sum()),
                                "len_develop": int(len(sc)),
                                "len_master": int(len(sa)),
                                # Per-hour relative errors. ``worst_rel_pct`` is
                                # dominated by dawn/dusk hours where master is a
                                # few MW, so a fraction of a MW reads as 100 %;
                                # it is reported, never bounded by a waiver.
                                "worst_rel_pct": round(float(rel.max()) * 100.0, 4),
                                "mean_rel_pct": round(float(rel.mean()) * 100.0, 4),
                                # sum|delta| / sum(master): the same error
                                # weighted by the energy each hour carries, so
                                # the near-zero hours stop dominating. This is
                                # the per-hour number worth reading beside the
                                # annual totals below.
                                "energy_weighted_mean_rel_pct": (
                                    round(float(np.abs(vc - va).sum()) / master_energy * 100.0, 4)
                                    if master_energy
                                    else None
                                ),
                                "develop_total_mwh": float(vc.sum()),
                                "master_total_mwh": float(va.sum()),
                                "total_rel_pct": (
                                    round((float(vc.sum()) - float(va.sum())) / abs(float(va.sum())) * 100.0, 4)
                                    if va.sum()
                                    else None
                                ),
                            },
                        },
                    )
        # Per-bus variable comparison is only meaningful when the two sides
        # share a bus space as BUILT (prong 1). At prong 2 develop is keyed by
        # simpl-cluster IDs and master by nodal IDs — zero overlap — so
        # skip the per-var loop and let the system aggregates above carry the
        # comparison instead of emitting hundreds of vacuous row_set findings.
        # The test is deliberately made against the unaggregated master
        # (``da_raw``): the rollup above exists to make the SYSTEM aggregates
        # and the table metrics comparable, not to manufacture a per-bus
        # comparison the two DAGs never actually shared.
        cb = {str(b) for b in dc.indexes.get("bus", [])}
        ab = {str(b) for b in da_raw.indexes.get("bus", [])}
        if not (cb & ab):
            return
        for var in sorted(set(dc.data_vars) | set(da_raw.data_vars)):
            if var not in dc.data_vars or var not in da_raw.data_vars:
                findings.append(
                    {
                        "stage": pair.stage,
                        "component": var,
                        "column": "<var>",
                        "kind": "column_set",
                        "detail": "develop-only" if var in dc.data_vars else "master-only",
                    },
                )
                continue
            vc, va = dc[var], da_raw[var]
            fc = vc.transpose("time", "bus").to_pandas() if "time" in vc.dims else vc.to_pandas().to_frame(var)
            fa = va.transpose("time", "bus").to_pandas() if "time" in va.dims else va.to_pandas().to_frame(var)
            fc.columns = fc.columns.map(str)
            fa.columns = fa.columns.map(str)
            compare_frames(pair.stage, var, fc.T, fa.T, findings)


def compare_pair(
    pair: ArtifactPair,
    develop_root: Path,
    master_root: Path,
    prong: int = 1,
    busmap: pd.Series | None = None,
    notes: list[dict] | None = None,
) -> list[dict]:
    findings: list[dict] = []
    pc, pa = develop_root / pair.develop, master_root / pair.master
    for side, p in (("develop", pc), ("master", pa)):
        if not p.exists():
            findings.append(
                {
                    "stage": pair.stage,
                    "component": "<file>",
                    "column": "<file>",
                    "kind": "missing_artifact",
                    "detail": f"{side}: {p}",
                },
            )
    if any(f["kind"] == "missing_artifact" for f in findings):
        return findings
    if pair.kind in ("network", "network_pkl_vs_nc"):
        compare_networks(pair, load_network(pc), load_network(pa), findings)
    elif pair.kind == "profile":
        compare_profiles(pair, pc, pa, findings, prong=prong, busmap=busmap, notes=notes)
    elif pair.kind == "demand_total":
        # The two demand CSVs are keyed at different granularities (master is
        # nodal pre-aggregation, develop substation-keyed), so per-bus
        # comparison is meaningless here — it is covered by the assembled
        # network's Load_t.p_set. Compare the clustering-invariant system
        # total instead.
        tc = float(pd.read_csv(pc, index_col=0).apply(pd.to_numeric, errors="coerce").sum().sum())
        ta = float(pd.read_csv(pa, index_col=0).apply(pd.to_numeric, errors="coerce").sum().sum())
        if not np.isclose(tc, ta, rtol=RTOL):
            findings.append(
                {
                    "stage": pair.stage,
                    "component": "demand",
                    "column": "system_total",
                    "kind": "value",
                    "detail": {
                        "develop": tc,
                        "master": ta,
                        "rel": abs(tc - ta) / max(abs(ta), 1e-9),
                        "rel_pct": round(abs(tc - ta) / max(abs(ta), 1e-9) * 100.0, 4),
                    },
                },
            )
    return findings


def prong2_aggregates(nc, na) -> list[dict]:
    """Clustering-invariant checks for prong 2 pre-solve networks."""
    findings: list[dict] = []
    lc = float(nc.loads_t.p_set.sum().sum())
    la = float(na.loads_t.p_set.sum().sum())
    if not np.isclose(lc, la, rtol=RTOL):
        findings.append(
            {
                "stage": "prong2_aggregates",
                "component": "Load",
                "column": "total_energy",
                "kind": "value",
                "detail": {"develop": lc, "master": la},
            },
        )
    gc = nc.generators.groupby("carrier").p_nom.sum()
    ga = na.generators.groupby("carrier").p_nom.sum()
    compare_frames(
        "prong2_aggregates",
        "Generator.p_nom_by_carrier",
        gc.to_frame("p_nom"),
        ga.to_frame("p_nom"),
        findings,
    )
    return findings


def run_comparison(prong: int, develop_root: Path, master_root: Path) -> dict:
    waivers = load_waivers()
    all_findings: list[dict] = []
    notes: list[dict] = []
    pairs = prong_pairs(prong)
    # Prong 2 only: the map that rolls master's nodal profiles onto develop's
    # cluster bus space (see compare_profiles).
    busmap = load_busmap(develop_root) if prong == 2 else None
    if prong == 2:
        # Pre-cluster per-bus artifacts differ by design at prong 2 (different
        # simpl-stage kmeans), so normally only clustered/solve stages compare.
        # Under UNTIL=assembled those don't exist; the profile and demand
        # pairs carry the comparison instead via their clustering-invariant
        # system aggregates (compare_profiles skips per-bus vars when the two
        # bus spaces are disjoint).
        keep = ("clustered_network",)
        pairs = [
            p
            for p in pairs
            if p.solve_stage or p.stage in keep or (UNTIL == "assembled" and p.kind in ("profile", "demand_total"))
        ]
    for pair in pairs:
        if prong == 2 and pair.stage == "clustered_network":
            pc, pa = develop_root / pair.develop, master_root / pair.master
            if pc.exists() and pa.exists():
                all_findings += prong2_aggregates(load_network(pc), load_network(pa))
            else:
                all_findings.append(
                    {
                        "stage": pair.stage,
                        "component": "<file>",
                        "column": "<file>",
                        "kind": "missing_artifact",
                        "detail": f"{pc if not pc.exists() else pa}",
                    },
                )
            continue
        all_findings += compare_pair(
            pair,
            develop_root,
            master_root,
            prong=prong,
            busmap=busmap,
            notes=notes,
        )
    for f in all_findings:
        f["prong"] = prong
        f["interconnect"] = INTERCONNECT
        # ``waiver_note``/``waiver_bound_failed`` are set only when a waiver
        # MATCHED this finding and its bounds did not hold: the finding stays
        # live, and why is on the record. ``waiver`` names the entry that DID
        # waive it, so the Findings section of comparison.md can say "waived
        # HF-24" instead of an unattributed "waived".
        w, note, failed = matching_waiver(f, waivers)
        f["waived"] = w is not None
        if w is not None:
            f["waiver"] = waiver_id(w)
        if note:
            f["waiver_note"] = note
        if failed:
            f["waiver_bound_failed"] = failed
    live = [f for f in all_findings if not f["waived"]]
    cluster_note = [n for n in notes if n.get("kind") == "profile_rollup"]
    result = {
        "prong": prong,
        "n_findings": len(all_findings),
        "n_live": len(live),
        "pass": not live,
        "profile_cluster_sets": cluster_note,
        "findings": all_findings,
    }
    # One run directory per run (plan D5): the findings sit beside
    # run_meta.json and both manifests.
    out = run_dir() / f"findings_{prong}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=str))
    _record_profile_cluster_sets(prong, cluster_note)
    return result


def _record_profile_cluster_sets(prong: int, cluster_note: list[dict]) -> None:
    """Put the rollup facts in ``run_meta.json``, beside the shas they belong to.

    ``master_profile_stage`` is rewritten from what actually happened rather
    than from the prong: ``build_context`` runs before any artifact exists, so
    at that point the rollup is a plan, not a fact. A run whose busmap was
    missing rolled nothing up, and the provenance record has to say so — a CF
    quantile taken over 544 substations and one taken over 19 clusters are not
    the same number.
    """
    from . import context

    rolled = [n for n in cluster_note if n.get("rolled_up")]
    fields: dict[str, object] = {"profile_cluster_sets": cluster_note}
    if prong == 1 or cluster_note:
        fields["master_profile_stage"] = context.master_profile_stage(
            prong,
            rolled_up=bool(rolled) if cluster_note else None,
        )
    context.update_run_meta(**fields)
