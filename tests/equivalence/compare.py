"""Stage-by-stage artifact comparison for the equivalence harness.

Tolerance policy (spec D2/D7):
- floats: ``np.allclose(rtol=1e-3, atol=1e-8, equal_nan=True)``
- indexes / integers / strings: exact, after sorting
- solved network: objective within 0.1%, per-carrier capacity within 0.5%;
  the objective is normalized to ``objective + objective_constant`` on both
  sides so the pypsa 0.30 vs v1 reporting conventions compare like for like
- row-set and column-set differences are first-class findings
- representation-only differences are normalized before comparing: float-
  formatted integer labels ('35827.0' vs '35827') and Load names carrying a
  carrier suffix ('35827 AC' vs '35827', re-keyed on the load's bus)
- per-frame findings are capped at ``MAX_FINDINGS_PER_FRAME`` with a single
  'suppressed' finding so fan-out cannot drown the signal
- waivers (tests/equivalence/waivers.yaml) suppress exactly the signed-off
  deltas; each waiver must reference a deltas-ledger entry.

Findings are dicts: {stage, component, column, kind, detail, waived}.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from .paths import INTERCONNECT, ArtifactPair, prong_pairs

REPO = Path(__file__).resolve().parents[2]
RTOL = 1e-3
ATOL = 1e-8
OBJECTIVE_RTOL = 1e-3
CAPACITY_RTOL = 5e-3
MAX_FINDINGS_PER_FRAME = 50
WAIVERS_PATH = Path(__file__).parent / "waivers.yaml"

# Anchor artifacts float-format integer bus labels ('35827.0'); candidate
# writes them bare ('35827'). Pure representation — normalize on both sides.
_FLOAT_INT_LABEL = re.compile(r"\d+\.0")


def _norm_label(x) -> str:
    """Str-ify a label, collapsing float-formatted integers to integer form."""
    s = str(x)
    return s[:-2] if _FLOAT_INT_LABEL.fullmatch(s) else s


def load_network(path: Path):
    import pypsa

    # Keep network frames on numpy object dtype under pandas 3 (matches
    # _helpers), so candidate and anchor networks compare on equal footing.
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


def is_waived(finding: dict, waivers: list[dict]) -> bool:
    for w in waivers:
        if all(
            w.get(k) in (None, "*", finding.get(k))
            for k in ("stage", "component", "column", "kind", "prong", "interconnect")
        ):
            return True
    return False


def _numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _rel_pct(a: float, b: float) -> float | None:
    """Relative difference of a vs anchor b, in percent (None if undefined)."""
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
    cand: pd.DataFrame,
    anch: pd.DataFrame,
    findings: list[dict],
) -> None:
    """Compare two indexed DataFrames; append findings in place.

    Labels are normalized on both sides (float-formatted integers collapse to
    integer form). A row-set mismatch is reported once and the comparison
    proceeds on the intersection. Findings for one frame are capped at
    ``MAX_FINDINGS_PER_FRAME`` with a single 'suppressed' finding.
    """
    local: list[dict] = []
    cand = _normalize_frame(cand, "candidate", stage, component, local)
    anch = _normalize_frame(anch, "anchor", stage, component, local)
    ci, ai = set(cand.index), set(anch.index)
    if ci != ai:
        local.append(
            {
                "stage": stage,
                "component": component,
                "column": "<index>",
                "kind": "row_set",
                "detail": f"candidate-only={sorted(ci - ai)[:8]} (n={len(ci - ai)}), "
                f"anchor-only={sorted(ai - ci)[:8]} (n={len(ai - ci)})",
            },
        )
    common = sorted(ci & ai)
    if not common:
        findings.extend(local)
        return
    cand = cand.loc[common]
    anch = anch.loc[common]

    cc, ac = set(cand.columns), set(anch.columns)
    for col in sorted(cc ^ ac, key=str):
        local.append(
            {
                "stage": stage,
                "component": component,
                "column": str(col),
                "kind": "column_set",
                "detail": "candidate-only" if col in cc else "anchor-only",
            },
        )
    for col in sorted(cc & ac, key=str):
        a, b = cand[col], anch[col]
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
                                    "candidate": None if np.isnan(av[i]) else float(av[i]),
                                    "anchor": None if np.isnan(bv[i]) else float(bv[i]),
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
                            "examples": [{"id": str(i), "candidate": av[i], "anchor": bv[i]} for i in ex],
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

    Anchor Load names carry a carrier suffix ('35827 AC') while candidate
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
                "detail": f"candidate n={len(nc.snapshots)}, anchor n={len(na.snapshots)}",
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
                bad = [s for s, k in (("candidate", kc), ("anchor", ka)) if k is None]
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
    val = getattr(n, "objective_constant", 0.0)
    try:
        val = float(val)
    except (TypeError, ValueError):
        return 0.0
    return val if np.isfinite(val) else 0.0


def _total_objective(n) -> float:
    """Total system cost, independent of the reporting convention.

    pypsa 0.30 reported ``Network.objective`` as the solver's objective value
    only and carried the fixed-cost offset separately in
    ``Network.objective_constant``; pypsa v1 / linopy 0.9 fold that constant
    into ``objective`` and leave ``objective_constant`` at 0. Comparing raw
    ``objective`` across the two conventions therefore manufactures a
    difference exactly the size of the constant, even when the two solves are
    identical. Both sides are normalized to ``objective + objective_constant``
    — the total system cost either way — and the existing tolerance applies to
    that. Each side's constant is read from its own file (missing -> 0.0).
    """
    return float(n.objective) + _objective_constant(n)


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
                    "candidate": oc,
                    "anchor": oa,
                    "candidate_raw": float(nc.objective),
                    "anchor_raw": float(na.objective),
                    "candidate_constant": _objective_constant(nc),
                    "anchor_constant": _objective_constant(na),
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
    anch = ca.reindex_like(both).fillna(0.0)
    close = np.isclose(both.to_numpy(), anch.to_numpy(), rtol=CAPACITY_RTOL, atol=1.0)
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
                        "candidate": float(both.iloc[r, c]),
                        "anchor": float(anch.iloc[r, c]),
                        "rel_pct": _rel_pct(float(both.iloc[r, c]), float(anch.iloc[r, c])),
                    }
                    for r, c in zip(rows, cols)
                ],
            },
        )


def compare_profiles(pair: ArtifactPair, pc: Path, pa: Path, findings: list[dict]) -> None:
    with xr.open_dataset(pc) as dc, xr.open_dataset(pa) as da:
        for var in sorted(set(dc.data_vars) | set(da.data_vars)):
            if var not in dc.data_vars or var not in da.data_vars:
                findings.append(
                    {
                        "stage": pair.stage,
                        "component": var,
                        "column": "<var>",
                        "kind": "column_set",
                        "detail": "candidate-only" if var in dc.data_vars else "anchor-only",
                    },
                )
                continue
            vc, va = dc[var], da[var]
            fc = vc.transpose("time", "bus").to_pandas() if "time" in vc.dims else vc.to_pandas().to_frame(var)
            fa = va.transpose("time", "bus").to_pandas() if "time" in va.dims else va.to_pandas().to_frame(var)
            fc.columns = fc.columns.map(str)
            fa.columns = fa.columns.map(str)
            compare_frames(pair.stage, var, fc.T, fa.T, findings)


def compare_pair(pair: ArtifactPair, cand_root: Path, anch_root: Path) -> list[dict]:
    findings: list[dict] = []
    pc, pa = cand_root / pair.candidate, anch_root / pair.anchor
    for side, p in (("candidate", pc), ("anchor", pa)):
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
        compare_profiles(pair, pc, pa, findings)
    elif pair.kind == "demand_total":
        # The two demand CSVs are keyed at different granularities (anchor is
        # nodal pre-aggregation, candidate substation-keyed), so per-bus
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
                        "candidate": tc,
                        "anchor": ta,
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
                "detail": {"candidate": lc, "anchor": la},
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


def run_comparison(prong: int, cand_root: Path, anch_root: Path) -> dict:
    waivers = load_waivers()
    all_findings: list[dict] = []
    pairs = prong_pairs(prong)
    if prong == 2:
        pairs = [p for p in pairs if p.solve_stage or p.stage in ("clustered_network",)]
    for pair in pairs:
        if prong == 2 and pair.stage == "clustered_network":
            pc, pa = cand_root / pair.candidate, anch_root / pair.anchor
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
        all_findings += compare_pair(pair, cand_root, anch_root)
    for f in all_findings:
        f["prong"] = prong
        f["interconnect"] = INTERCONNECT
        f["waived"] = is_waived(f, waivers)
    live = [f for f in all_findings if not f["waived"]]
    result = {
        "prong": prong,
        "n_findings": len(all_findings),
        "n_live": len(live),
        "pass": not live,
        "findings": all_findings,
    }
    suffix = "" if INTERCONNECT == "western" else f"_{INTERCONNECT}"
    out = REPO / "workflow" / "results" / "equivalence" / f"findings_{prong}{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=str))
    return result
