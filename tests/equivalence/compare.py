"""Stage-by-stage artifact comparison for the equivalence harness.

Tolerance policy (spec D2/D7):
- floats: ``np.allclose(rtol=1e-3, atol=1e-8, equal_nan=True)``
- indexes / integers / strings: exact, after sorting
- solved network: objective within 0.1%, per-carrier capacity within 0.5%
- row-set and column-set differences are first-class findings
- waivers (tests/equivalence/waivers.yaml) suppress exactly the signed-off
  deltas; each waiver must reference a deltas-ledger entry.

Findings are dicts: {stage, component, column, kind, detail, waived}.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from .paths import ArtifactPair, prong_pairs

REPO = Path(__file__).resolve().parents[2]
RTOL = 1e-3
ATOL = 1e-8
OBJECTIVE_RTOL = 1e-3
CAPACITY_RTOL = 5e-3
WAIVERS_PATH = Path(__file__).parent / "waivers.yaml"

# Candidate-only bookkeeping columns that exist because of the DAG
# restructuring itself (not results): compared as schema info, never values.
_T_SKIP_EMPTY = True


def load_network(path: Path):
    import pypsa

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
        if all(w.get(k) in (None, "*", finding.get(k)) for k in ("stage", "component", "column", "kind")):
            return True
    return False


def _numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def compare_frames(
    stage: str,
    component: str,
    cand: pd.DataFrame,
    anch: pd.DataFrame,
    findings: list[dict],
) -> None:
    """Compare two indexed DataFrames; append findings in place."""
    ci, ai = set(map(str, cand.index)), set(map(str, anch.index))
    if ci != ai:
        findings.append(
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
        return
    cand = cand.copy()
    anch = anch.copy()
    cand.index = cand.index.map(str)
    anch.index = anch.index.map(str)
    cand = cand.loc[common]
    anch = anch.loc[common]

    cc, ac = set(cand.columns), set(anch.columns)
    for col in sorted(cc ^ ac):
        findings.append(
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
                findings.append(
                    {
                        "stage": stage,
                        "component": component,
                        "column": str(col),
                        "kind": "value",
                        "detail": {
                            "n_diff": len(bad),
                            "n_total": len(av),
                            "max_abs": float(np.nanmax(np.abs(av[bad] - bv[bad]))),
                            "examples": [
                                {
                                    "id": common[i],
                                    "candidate": None if np.isnan(av[i]) else float(av[i]),
                                    "anchor": None if np.isnan(bv[i]) else float(bv[i]),
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
                findings.append(
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
        {c.name for c in nc.iterate_components()} | {c.name for c in na.iterate_components()},
    )
    for name in comps:
        dfc, dfa = nc.df(name), na.df(name)
        if dfc.empty and dfa.empty:
            continue
        compare_frames(pair.stage, name, dfc, dfa, findings)
        pnl_c, pnl_a = nc.pnl(name), na.pnl(name)
        for attr in sorted(set(pnl_c) | set(pnl_a)):
            tc = pnl_c.get(attr, pd.DataFrame())
            ta = pnl_a.get(attr, pd.DataFrame())
            if tc.empty and ta.empty:
                continue
            compare_frames(pair.stage, f"{name}_t.{attr}", tc, ta, findings)


def _capacity_by_carrier(n) -> pd.DataFrame:
    out = {}
    for comp, cap in (("generators", "p_nom_opt"), ("storage_units", "p_nom_opt")):
        df = getattr(n, comp)
        if not df.empty and cap in df:
            out[comp] = df.groupby("carrier")[cap].sum()
    return pd.DataFrame(out).fillna(0.0)


def _compare_solved(pair: ArtifactPair, nc, na, findings: list[dict]) -> None:
    oc, oa = float(nc.objective), float(na.objective)
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
                    "rel": abs(oc - oa) / max(abs(oa), 1e-9),
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
    elif pair.kind == "demand_csv":
        fc = pd.read_csv(pc, index_col=0)
        fa = pd.read_csv(pa, index_col=0)
        fc.columns, fa.columns = fc.columns.map(str), fa.columns.map(str)
        compare_frames(pair.stage, "demand", fc.T, fa.T, findings)
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
        f["waived"] = is_waived(f, waivers)
    live = [f for f in all_findings if not f["waived"]]
    result = {
        "prong": prong,
        "n_findings": len(all_findings),
        "n_live": len(live),
        "pass": not live,
        "findings": all_findings,
    }
    out = REPO / "workflow" / "results" / "equivalence" / f"findings_{prong}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=str))
    return result
