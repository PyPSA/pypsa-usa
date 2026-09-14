"""Numeric comparison metrics for the master-vs-develop equivalence benchmark.

This module is the evaluation criterion of ``memory/plans/harness-master-vs-develop.md``
(PROJECT.md section 3.2) expressed as code: the distributional statistics of the
inputs that reach ``solve_network`` (``p_max_pu``, ``p_nom_max``, ``p_nom``, the
geographic assignment of capacity, demand) and the solved-stage deltas
(objective, capacity, dispatch).

Contract, relied on by ``tables.py`` and ``plots.py``:

- **Nothing here reads a file.** Callers pass loaded ``pypsa.Network`` objects,
  loaded ``xarray.Dataset`` objects and already-built bus->zone maps, so the unit
  tests run on 3-bus networks and 24-hour datasets with no ``data/``,
  ``resources/`` or network access.
- Every function returns a pandas object indexed by its grouping key with
  columns exactly ``["master", "develop", "delta", "delta_pct"]``.
- ``delta`` is **develop minus master**, in the metric's own unit.
- ``delta_pct`` is ``100 * delta / master``. It is ``0.0`` when both sides are
  zero and ``NaN`` when master is zero but develop is not — an
  appear-from-nothing difference, which ``tables.comparison_table`` treats as
  infinitely over tolerance rather than as "no difference".

Zone grouping always comes from the ``reeds_zone`` BUS ATTRIBUTE, never from
string surgery on cluster names: cluster ids are ``p{zone}{subcluster} {i}`` with
no separator ("p101 1" is zone p10, sub-cluster 1), so a prefix split silently
produces labels that match almost nothing. That bug is what produced the first,
mostly-grey report maps; ``test_capacity_by_zone_carrier_uses_bus_attribute``
keeps it from coming back.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["master", "develop", "delta", "delta_pct"]

#: Index of the Series returned by :func:`objective_row`. The ``*_raw`` and
#: ``*_constant`` entries are carried for the reader and are NEVER compared —
#: see the HF-13 note in that function's docstring.
OBJECTIVE_INDEX = [
    "master",
    "develop",
    "delta",
    "delta_pct",
    "master_raw",
    "develop_raw",
    "master_constant",
    "develop_constant",
]

#: Components whose static frames carry a capacity attribute worth summing.
CAPACITY_COMPONENTS = ("generators", "storage_units")


def empty_frame(name: str = "key") -> pd.DataFrame:
    """An empty metric frame with the contract's index name and columns."""
    return pd.DataFrame(columns=COLUMNS, index=pd.Index([], name=name, dtype=object))


def normalize_bus_ids(idx, mapper: pd.Series) -> pd.Series:
    """Map bus ids onto a bus->value Series, tolerating '39762.0' vs '39762'.

    Master artifacts float-format integer bus labels; develop writes them bare.
    The difference is pure representation, so try the literal label first and
    fall back to the integer form for whatever did not resolve.

    Returns a Series indexed by the stringified input labels.
    """
    ids = pd.Index([str(b) for b in idx])
    out = pd.Series(ids, index=ids).map(mapper)
    miss = out.isna()
    if miss.any():
        norm = []
        for b in out.index[miss]:
            try:
                norm.append(str(int(float(b))))
            except (TypeError, ValueError):
                norm.append(b)
        out.loc[miss] = pd.Series(norm, index=out.index[miss]).map(mapper).to_numpy()
    return out


def _rel_pct(delta: pd.Series, master: pd.Series) -> pd.Series:
    """100 * delta / master, with 0/0 -> 0.0 and x/0 (x != 0) -> NaN."""
    m = master.to_numpy(dtype=float)
    d = delta.to_numpy(dtype=float)
    out = np.full(len(d), np.nan)
    nz = m != 0
    with np.errstate(divide="ignore", invalid="ignore"):
        out[nz] = 100.0 * d[nz] / m[nz]
    out[(~nz) & (d == 0.0)] = 0.0
    return pd.Series(out, index=master.index)


def frame(
    master: pd.Series,
    develop: pd.Series,
    name: str = "key",
    fill: float | None = 0.0,
) -> pd.DataFrame:
    """Assemble the four-column metric frame from two keyed Series.

    The two sides are aligned on the union of their keys.

    ``fill`` decides what a one-sided key means, and the two cases are NOT the
    same:

    - **Additive metrics** (capacity MW, dispatch MWh, demand MW, installable
      potential MW) pass ``fill=0.0``. A carrier or zone absent on one side
      genuinely has zero capacity there, so 0.0 is the honest reading and the
      resulting -100 % is a real difference.
    - **Ratio metrics** (capacity factor, capacity-factor quantiles) pass
      ``fill=None`` and keep NaN. A carrier with no capacity has an *undefined*
      capacity factor, not a capacity factor of zero; filling it with 0.0 would
      manufacture a -100 % difference out of a quantity that does not exist, and
      an all-NaN row would read as "equivalent, 0 vs 0".
      ``tables.comparison_table`` gives those rows the verdicts ``undefined``
      and ``one-sided``.
    """
    master = pd.Series(master, dtype=float)
    develop = pd.Series(develop, dtype=float)
    if master.empty and develop.empty:
        return empty_frame(name)
    idx = master.index.union(develop.index, sort=False)
    try:
        idx = idx.sort_values()
    except TypeError:
        # Mixed-type keys (p_max_pu_quantiles mixes float quantiles with the
        # labels 'mean' and 'p_nom_max_weighted_mean'); keep declaration order.
        pass
    idx.name = name
    m = master.reindex(idx).astype(float)
    d = develop.reindex(idx).astype(float)
    if fill is not None:
        m, d = m.fillna(fill), d.fillna(fill)
    delta = d - m
    return pd.DataFrame(
        {"master": m, "develop": d, "delta": delta, "delta_pct": _rel_pct(delta, m)},
        index=idx,
    )[COLUMNS]


def _zone_of_bus(n) -> pd.Series:
    """Bus -> ``reeds_zone``, from the BUS ATTRIBUTE. Raises when it is absent."""
    if "reeds_zone" not in n.buses.columns:
        raise ValueError(
            "buses have no 'reeds_zone' attribute; zone grouping must come from the "
            "bus attribute, never from string surgery on cluster names",
        )
    return n.buses["reeds_zone"].astype(str)


def _capacity_series(n, attr: str) -> pd.Series:
    """``attr`` summed by carrier over Generator + StorageUnit, in MW."""
    parts = []
    for comp in CAPACITY_COMPONENTS:
        df = getattr(n, comp, None)
        if df is None or df.empty or attr not in df.columns:
            continue
        parts.append(df.groupby(df["carrier"].astype(str))[attr].sum())
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts).groupby(level=0).sum()


def _capacity_zone_series(n, attr: str) -> pd.Series:
    """``attr`` summed by (reeds_zone, carrier) over Generator + StorageUnit, MW."""
    zone = _zone_of_bus(n)
    parts = []
    for comp in CAPACITY_COMPONENTS:
        df = getattr(n, comp, None)
        if df is None or df.empty or attr not in df.columns:
            continue
        keys = [df["bus"].astype(str).map(zone).fillna("<unmapped>"), df["carrier"].astype(str)]
        parts.append(df.groupby(keys, observed=True)[attr].sum())
    if not parts:
        return pd.Series(dtype=float, index=pd.MultiIndex.from_arrays([[], []]))
    out = pd.concat(parts).groupby(level=[0, 1]).sum()
    out.index = out.index.set_names(["zone", "carrier"])
    return out


def capacity_by_carrier(n_master, n_develop, attr: str = "p_nom") -> pd.DataFrame:
    """Existing (``p_nom``) or solved (``p_nom_opt``) capacity, MW, by carrier.

    Summed over Generator and StorageUnit. A carrier missing on one side reads
    as 0 MW there.
    """
    return frame(_capacity_series(n_master, attr), _capacity_series(n_develop, attr), name="carrier")


def capacity_by_zone_carrier(n_master, n_develop, attr: str = "p_nom") -> pd.DataFrame:
    """Capacity, MW, by ``(reeds_zone, carrier)`` — the geographic-assignment criterion.

    The zone comes from the ``reeds_zone`` bus attribute. Buses whose zone does
    not resolve are collected under the literal key ``"<unmapped>"`` so they are
    visible rather than silently dropped.
    """
    m = _capacity_zone_series(n_master, attr)
    d = _capacity_zone_series(n_develop, attr)
    out = frame(m, d, name="zone_carrier")
    if not out.empty:
        out.index = pd.MultiIndex.from_tuples(list(out.index), names=["zone", "carrier"])
    return out


def _weights(n) -> pd.Series:
    """Snapshot weightings for energy integration (hours per snapshot)."""
    w = getattr(n, "snapshot_weightings", None)
    if w is None or len(w) == 0:
        return pd.Series(1.0, index=n.snapshots)
    if isinstance(w, pd.DataFrame):
        col = "generators" if "generators" in w.columns else w.columns[0]
        return w[col].astype(float)
    return pd.Series(w, dtype=float)


def _dispatch_series(n) -> pd.Series | None:
    """Annual energy, MWh, by carrier. ``None`` when the network has no solution.

    Generator dispatch is integrated with ``snapshot_weightings.generators``.
    StorageUnit dispatch is appended as its **positive part only** (discharge);
    charging is consumption and would otherwise cancel real generation.
    """
    gt = getattr(n, "generators_t", None)
    p = None if gt is None else gt.get("p")
    if p is None or p.empty:
        return None
    w = _weights(n).reindex(p.index).fillna(1.0)
    energy = p.mul(w, axis=0).sum(axis=0)
    carrier = n.generators["carrier"].astype(str).reindex(energy.index)
    out = energy.groupby(carrier).sum()

    st = getattr(n, "storage_units_t", None)
    sp = None if st is None else st.get("p")
    if sp is not None and not sp.empty:
        ws = _weights(n).reindex(sp.index).fillna(1.0)
        s_energy = sp.clip(lower=0.0).mul(ws, axis=0).sum(axis=0)
        s_carrier = n.storage_units["carrier"].astype(str).reindex(s_energy.index)
        out = pd.concat([out, s_energy.groupby(s_carrier).sum()]).groupby(level=0).sum()
    return out


def dispatch_by_carrier(n_master, n_develop) -> pd.DataFrame:
    """Annual energy, MWh, by carrier.

    Returns an **empty frame with the contract's columns** — never an exception
    — when either side carries no solution, since a one-sided dispatch table is
    not a comparison.
    """
    m, d = _dispatch_series(n_master), _dispatch_series(n_develop)
    if m is None or d is None:
        return empty_frame("carrier")
    return frame(m, d, name="carrier")


def _capacity_factor_series(n) -> pd.Series | None:
    """Fleet capacity factor by carrier: energy / (capacity * weighted hours)."""
    energy = _dispatch_series(n)
    if energy is None:
        return None
    attr = "p_nom_opt" if "p_nom_opt" in n.generators.columns else "p_nom"
    cap = _capacity_series(n, attr)
    hours = float(_weights(n).sum())
    denom = (cap * hours).reindex(energy.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        cf = energy / denom.replace(0.0, np.nan)
    return cf.astype(float)


def capacity_factor_by_carrier(n_master, n_develop) -> pd.DataFrame:
    """Realised fleet capacity factor (dimensionless, 0-1) by carrier.

    Energy divided by ``capacity * sum(snapshot_weightings)``. Empty when either
    side has no solution, for the same reason as :func:`dispatch_by_carrier`.
    """
    m, d = _capacity_factor_series(n_master), _capacity_factor_series(n_develop)
    if m is None or d is None:
        return empty_frame("carrier")
    return frame(m, d, name="carrier", fill=None)


def objective_constant(n) -> float:
    """``objective_constant`` of a network, 0.0 when absent, NaN or unparseable."""
    val = getattr(n, "objective_constant", 0.0)
    try:
        val = float(val)
    except (TypeError, ValueError):
        return 0.0
    return val if np.isfinite(val) else 0.0


def total_objective(n) -> float:
    """``objective + objective_constant`` — the total, invariant to the split (HF-13)."""
    return float(n.objective) + objective_constant(n)


def objective_row(n_master, n_develop) -> pd.Series:
    """Total system cost, normalised across the two branches.

    **HF-13** (hot-fix ledger; deltas-ledger DL-15 class C). Both pypsa 0.30 and
    pypsa 1.3 expose ``objective`` and ``objective_constant`` as two separate
    attributes — verified 2026-09-14. (The ledger's wording, that v1 "folds the
    offset into ``objective`` and leaves ``objective_constant`` at 0", is not a
    property of the library and should not be relied on.) What the two branches
    do differ on is **how the total is split between the two terms**: on the CA
    leg master reported ``-204,665,929.13`` with a constant of
    ``1,133,255,860.00`` while develop reported ``928,590,425.01`` with a
    constant of ``0.00``. Comparing either term on its own therefore manufactures
    a difference the size of the constant even when the two solves agree.

    The **sum** is the total system cost and is invariant to that split, so both
    sides are normalised to ``objective + objective_constant`` and the
    ``objective`` tolerance applies to that. Normalising is also what lets a
    0.30-written network be read under 1.3 without the split mattering. On the CA
    leg the normalised values agree to a relative difference of 5.3e-07.

    Returns a Series indexed by :data:`OBJECTIVE_INDEX`. The ``*_raw`` and
    ``*_constant`` entries exist for the reader; they are never compared.
    """
    m_raw, d_raw = float(n_master.objective), float(n_develop.objective)
    m_c, d_c = objective_constant(n_master), objective_constant(n_develop)
    m, d = m_raw + m_c, d_raw + d_c
    delta = d - m
    pct = float(_rel_pct(pd.Series([delta]), pd.Series([m])).iloc[0])
    return pd.Series(
        {
            "master": m,
            "develop": d,
            "delta": delta,
            "delta_pct": pct,
            "master_raw": m_raw,
            "develop_raw": d_raw,
            "master_constant": m_c,
            "develop_constant": d_c,
        },
    ).reindex(OBJECTIVE_INDEX)


def profile_values(ds) -> np.ndarray:
    """Flattened finite ``profile`` values of a renewable-profile Dataset."""
    if "profile" not in ds:
        return np.array([], dtype=float)
    v = np.asarray(ds["profile"].values, dtype=float).ravel()
    return v[np.isfinite(v)]


def _potential_weighted_mean_cf(ds) -> float:
    """Potential-weighted mean capacity factor of a profile Dataset."""
    if "profile" not in ds or "p_nom_max" not in ds:
        return float("nan")
    cf = np.asarray(ds["profile"].mean("time").values, dtype=float)
    pot = np.asarray(ds["p_nom_max"].values, dtype=float)
    total = np.nansum(pot)
    if total == 0 or not np.isfinite(total):
        return float("nan")
    return float(np.nansum(cf * pot) / total)


def p_max_pu_quantiles(
    ds_master,
    ds_develop,
    quantiles: tuple[float, ...] = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0),
) -> pd.DataFrame:
    """Capacity-factor distribution of two renewable-profile Datasets.

    Each Dataset carries a ``profile`` variable over ``(time, bus)``. Quantiles
    are taken over the **flattened** ``(time, bus)`` array, so the two sides stay
    comparable when their bus spaces differ (which they do at prong 2, where
    develop is keyed by simpl-cluster ids and master by nodal ids and the overlap
    is empty).

    Index: one entry per requested quantile (the float itself), plus ``"mean"``
    and ``"p_nom_max_weighted_mean"``.
    """
    vm, vd = profile_values(ds_master), profile_values(ds_develop)
    keys: list[object] = [*quantiles, "mean", "p_nom_max_weighted_mean"]
    m_vals = [float(np.quantile(vm, q)) if vm.size else float("nan") for q in quantiles]
    d_vals = [float(np.quantile(vd, q)) if vd.size else float("nan") for q in quantiles]
    m_vals += [float(vm.mean()) if vm.size else float("nan"), _potential_weighted_mean_cf(ds_master)]
    d_vals += [float(vd.mean()) if vd.size else float("nan"), _potential_weighted_mean_cf(ds_develop)]
    idx = pd.Index(keys, name="quantile", dtype=object)
    return frame(
        pd.Series(m_vals, index=idx), pd.Series(d_vals, index=idx), name="quantile", fill=None,
    )


def _p_nom_max_by_zone(ds, zone: pd.Series) -> pd.Series:
    """``p_nom_max`` summed by reeds_zone, MW."""
    if "p_nom_max" not in ds:
        return pd.Series(dtype=float)
    pot = ds["p_nom_max"].to_pandas()
    pot.index = pot.index.map(str)
    z = normalize_bus_ids(pot.index, zone)
    return pot.groupby(z.fillna("<unmapped>").to_numpy()).sum()


def p_nom_max_by_zone(
    ds_master,
    ds_develop,
    zone_master: pd.Series,
    zone_develop: pd.Series,
) -> pd.DataFrame:
    """Installable potential, MW, by ``reeds_zone``.

    The caller supplies the two bus->zone maps (``plots._zone_maps`` builds
    them), because deriving them needs files and this module reads none.
    """
    return frame(
        _p_nom_max_by_zone(ds_master, zone_master),
        _p_nom_max_by_zone(ds_develop, zone_develop),
        name="zone",
    )


def _mean_cf_by_zone(ds, zone: pd.Series) -> pd.Series:
    """Potential-weighted mean capacity factor by reeds_zone."""
    if "profile" not in ds or "p_nom_max" not in ds:
        return pd.Series(dtype=float)
    pot = ds["p_nom_max"].to_pandas()
    pot.index = pot.index.map(str)
    cf = ds["profile"].mean("time").to_pandas()
    cf.index = cf.index.map(str)
    z = normalize_bus_ids(pot.index, zone).fillna("<unmapped>").to_numpy()
    denom = pot.groupby(z).sum()
    return (cf * pot).groupby(z).sum() / denom.replace(0.0, np.nan)


def mean_cf_by_zone(ds_master, ds_develop, zone_master: pd.Series, zone_develop: pd.Series) -> pd.DataFrame:
    """Potential-weighted mean capacity factor (dimensionless) by ``reeds_zone``."""
    return frame(
        _mean_cf_by_zone(ds_master, zone_master),
        _mean_cf_by_zone(ds_develop, zone_develop),
        name="zone",
        fill=None,
    )


def available_power(ds) -> pd.Series:
    """National available power, ``sum_bus(profile * p_nom_max)``, MW per hour."""
    if "profile" not in ds or "p_nom_max" not in ds:
        return pd.Series(dtype=float)
    return (ds["profile"] * ds["p_nom_max"]).sum("bus").to_pandas().astype(float)


def _demand_stats(n) -> pd.Series:
    """(zone, stat) -> MW, with stat in {mean, peak}, from ``loads_t.p_set``."""
    lt = getattr(n, "loads_t", None)
    p_set = None if lt is None else lt.get("p_set")
    if p_set is None or p_set.empty:
        return pd.Series(dtype=float, index=pd.MultiIndex.from_arrays([[], []]))
    zone = _zone_of_bus(n)
    load_zone = n.loads["bus"].astype(str).map(zone).fillna("<unmapped>")
    cols = pd.Series(p_set.columns, index=p_set.columns).map(load_zone).fillna("<unmapped>")
    by_zone = p_set.T.groupby(cols.to_numpy()).sum().T
    stats = {}
    for z in by_zone.columns:
        stats[(str(z), "mean")] = float(by_zone[z].mean())
        stats[(str(z), "peak")] = float(by_zone[z].max())
    idx = pd.MultiIndex.from_tuples(sorted(stats), names=["zone", "stat"])
    return pd.Series([stats[k] for k in idx], index=idx, dtype=float)


def demand_by_zone(n_master, n_develop) -> pd.DataFrame:
    """Mean and peak demand, MW, per ``reeds_zone``, from ``loads_t.p_set``.

    The index is a ``(zone, stat)`` MultiIndex with ``stat`` in
    ``{"mean", "peak"}``, so the frame keeps the four-column contract instead of
    growing statistic columns.
    """
    out = frame(_demand_stats(n_master), _demand_stats(n_develop), name="zone_stat")
    if not out.empty:
        out.index = pd.MultiIndex.from_tuples(list(out.index), names=["zone", "stat"])
    return out
