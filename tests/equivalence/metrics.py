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

import logging
import re

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

COLUMNS = ["master", "develop", "delta", "delta_pct"]

#: Bus-dimensioned profile variables that are EXTENSIVE (MW, MW-potential, a
#: raw cell weight): a cluster's value is the sum of its members'.
PROFILE_SUM_VARS = ("p_nom_max", "potential", "weight")

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


def float_int_label(label) -> str | None:
    """``'39762.0' -> '39762'``; ``None`` when the label is not that form.

    The ONLY representation difference this harness reconciles is master's
    float-formatted integer bus labels against develop's bare ones. Everything
    else stays unmapped, deliberately:

    - ``'35827.5'`` is not an integer id at all. ``int(float(b))`` would truncate
      it onto bus 35827 and quietly merge two different things.
    - ``'035827'`` is a different string for the same integer, but it is not the
      float formatting this exists for; a zero-padded id is a foreign labelling
      convention and is dropped (and counted) rather than guessed at.

    A label that does not resolve is unmapped, which makes it a *counted drop*
    in :func:`aggregate_profile_to_clusters` rather than an invisible merge.
    """
    s = str(label).strip()
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v) or not float(v).is_integer():
        return None
    norm = str(int(v))
    return norm if s == norm or re.fullmatch(re.escape(norm) + r"\.0+", s) else None


def normalize_bus_ids(idx, mapper: pd.Series) -> pd.Series:
    """Map bus ids onto a bus->value Series, tolerating '39762.0' vs '39762'.

    Master artifacts float-format integer bus labels; develop writes them bare.
    The difference is pure representation, so try the literal label first and
    fall back to the integer form (:func:`float_int_label`) for whatever did not
    resolve. A label that is neither — ``'35827.5'``, ``'035827'`` — stays NaN,
    so the caller counts it as a drop instead of absorbing it into a neighbour.

    Returns a Series indexed by the stringified input labels.
    """
    ids = pd.Index([str(b) for b in idx])
    out = pd.Series(ids, index=ids).map(mapper)
    miss = out.isna()
    if miss.any():
        norm = [float_int_label(b) for b in out.index[miss]]
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


def aggregate_profile_to_clusters(
    ds: xr.Dataset,
    busmap: pd.Series,
    weight: str = "p_nom_max",
) -> xr.Dataset:
    """Aggregate a NODAL renewable-profile Dataset onto its ``{simpl}`` clusters.

    This is the prong-2 pre-step that makes the two sides' profile metrics
    comparable at all. Master builds ``profile_{tech}.nc`` at SUBSTATION
    resolution (western: 544 onwind / 808 solar buses) while develop builds
    ``profile_{tech}_s{simpl}.nc`` at cluster resolution (19 / 20 buses). Every
    distributional statistic — quantiles, means, per-zone weighted means — is
    resolution-dependent, so comparing the two files as they sit measures the
    clustering, not the refactor. Passing master through this function first
    puts both sides on the same bus space.

    ``busmap`` is the develop-side ``busmap_s{simpl}.csv``: substation id ->
    cluster bus. Master's profile bus labels are float-formatted integers
    ('35827.0') where the busmap's index is bare ('35827'); the join goes
    through :func:`normalize_bus_ids`, so either form resolves.

    Aggregation rules, one per variable class:

    - ``profile`` — ``weight``-weighted mean over the cluster's member buses.
      With the default ``weight="p_nom_max"`` this makes
      ``sum_bus(profile * p_nom_max)`` **exactly invariant** under aggregation,
      which is what lets the national available-power series stay a like-for-like
      comparison (``test_aggregate_preserves_available_power``).
    - :data:`PROFILE_SUM_VARS` (``p_nom_max``, ``potential``, ``weight``) —
      summed; they are extensive.
    - every other bus-dimensioned variable (``average_distance``) —
      ``p_nom_max``-weighted mean; they are intensive per unit of capacity.
    - variables without a ``bus`` dimension pass through untouched.

    Buses present in ``ds`` but **absent from** ``busmap`` are dropped, with a
    ``logging.warning`` naming the count — never silently, because a silent drop
    is exactly the master-side bug (HF-24) this prong is trying to measure. A
    cluster whose total weight is zero would divide 0/0, so its ``profile`` is
    **NaN** and counted. NaN, not 0.0: the pooled quantile metrics drop
    non-finite values, so a zero-weight master cluster contributes nothing —
    exactly as an all-NaN develop cluster does. Filled with 0.0 it would instead
    push 8,760 zeros into master's pool and none into develop's, moving every
    low quantile by the asymmetry alone. Both counts are also written to the
    returned Dataset's ``attrs`` as ``eq_dropped_buses`` and
    ``eq_zero_weight_clusters`` so a caller can assert on them.

    The output carries the SAME variable names and dims as the input, so every
    metric in this module accepts it unchanged.
    """
    if "bus" not in ds.dims:
        raise ValueError("dataset has no 'bus' dimension; nothing to aggregate")

    buses = pd.Index([str(b) for b in ds.indexes["bus"]], name="bus")
    mapped = normalize_bus_ids(buses, pd.Series(busmap).astype(str))
    keep = mapped.notna().to_numpy()
    n_dropped = int((~keep).sum())
    if n_dropped:
        logger.warning(
            "aggregate_profile_to_clusters: %d of %d buses are absent from the busmap "
            "and were dropped (first few: %s)",
            n_dropped,
            len(buses),
            ", ".join(str(b) for b in buses[~keep][:5]),
        )
    sub = ds.isel(bus=np.flatnonzero(keep))
    n_keep = int(keep.sum())
    labels = np.asarray(mapped.to_numpy()[keep], dtype=object)
    pos = pd.RangeIndex(n_keep)

    def _weights(name: str) -> pd.Series:
        """Positional weight vector for an intensive average (NaN -> 0)."""
        if name in sub:
            v = np.nan_to_num(np.asarray(sub[name].values, dtype=float), nan=0.0)
        else:
            v = np.ones(n_keep, dtype=float)
        return pd.Series(v, index=pos)

    w_profile = _weights(weight)
    w_intensive = _weights("p_nom_max")
    den_profile = w_profile.groupby(labels).sum()
    den_intensive = w_intensive.groupby(labels).sum()
    clusters = pd.Index(den_profile.index, name="bus")
    n_zero_weight = int((den_profile == 0).sum())
    if n_zero_weight:
        logger.warning(
            "aggregate_profile_to_clusters: %d of %d clusters have zero total '%s'; "
            "their profile is NaN (undefined), not 0.0",
            n_zero_weight,
            len(clusters),
            weight,
        )

    data_vars: dict[str, object] = {}
    for name, da in sub.data_vars.items():
        if "bus" not in da.dims:
            data_vars[name] = da
            continue
        w = w_profile if name == "profile" else w_intensive
        den = den_profile if name == "profile" else den_intensive
        if "time" in da.dims:
            f = da.transpose("time", "bus").to_pandas()
            f.columns = pos
            if name in PROFILE_SUM_VARS:
                agg = f.T.groupby(labels).sum().T
            else:
                # den.where(den != 0) is NaN for a zero-weight cluster, so the
                # quotient is NaN — an undefined intensive quantity, left
                # undefined. See the zero-weight note in the docstring.
                num = f.mul(w, axis=1).T.groupby(labels).sum().T
                agg = num.div(den.where(den != 0), axis=1)
            data_vars[name] = (("time", "bus"), agg.reindex(columns=clusters).to_numpy())
        else:
            s = pd.Series(np.asarray(da.values, dtype=float), index=pos)
            if name in PROFILE_SUM_VARS:
                agg1 = s.groupby(labels).sum()
            else:
                agg1 = (s * w).groupby(labels).sum() / den.where(den != 0)
            data_vars[name] = (("bus",), agg1.reindex(clusters).to_numpy())

    coords = {"bus": clusters.to_numpy()}
    if "time" in sub.coords:
        coords["time"] = sub["time"].to_numpy()
    out_ds = xr.Dataset(data_vars, coords=coords, attrs=dict(ds.attrs))
    out_ds.attrs["eq_dropped_buses"] = n_dropped
    out_ds.attrs["eq_zero_weight_clusters"] = n_zero_weight
    out_ds.attrs["eq_aggregated_from_buses"] = int(len(buses))
    return out_ds


def _bus_labels(ds) -> pd.Index:
    """The dataset's bus coordinate as plain strings."""
    if "bus" not in ds.dims:
        raise ValueError("dataset has no 'bus' dimension")
    return pd.Index([str(b) for b in ds.indexes["bus"]], name="bus")


def _p_nom_max_by_bus(ds) -> pd.Series:
    """``p_nom_max`` per bus, MW; an empty Series when the variable is absent."""
    if "p_nom_max" not in ds:
        return pd.Series(dtype=float)
    return pd.Series(np.asarray(ds["p_nom_max"].values, dtype=float), index=_bus_labels(ds))


def cluster_sets(ds_master, ds_develop) -> dict:
    """Which clusters each side carries, and what the one-sided ones are worth.

    A cluster present on ONE side only is not a small difference: pooled over
    ``(time, bus)``, its whole profile lands in one side's quantile pool and
    nothing lands in the other's. On the western smoke leg develop has one such
    cluster (``p87 0``: 96 MW of onwind, 2,158 MW of solar, HF-24's bus 37808),
    and it alone moved the onwind quantile deltas from -0.09/+0.51/0.00 % to
    -7.98/-2.73/+1.33 % (p25/p50/p95). Left unnamed, that reads as a
    capacity-factor difference the refactor caused.

    Returns ``{n_master, n_develop, n_common, only_master, only_develop,
    only_master_mw, only_develop_mw, equal}``, where the two ``only_*`` entries
    map cluster id -> its ``p_nom_max`` in MW. JSON-serialisable throughout, so
    it goes straight into a finding's ``detail`` and into ``run_meta.json``.
    """
    bm, bd = set(_bus_labels(ds_master)), set(_bus_labels(ds_develop))
    pm, pd_ = _p_nom_max_by_bus(ds_master), _p_nom_max_by_bus(ds_develop)
    only_master = {b: float(pm.get(b, float("nan"))) for b in sorted(bm - bd)}
    only_develop = {b: float(pd_.get(b, float("nan"))) for b in sorted(bd - bm)}
    return {
        "n_master": len(bm),
        "n_develop": len(bd),
        "n_common": len(bm & bd),
        "only_master": only_master,
        "only_develop": only_develop,
        "only_master_mw": float(np.nansum(list(only_master.values()))) if only_master else 0.0,
        "only_develop_mw": float(np.nansum(list(only_develop.values()))) if only_develop else 0.0,
        "equal": bm == bd,
    }


#: ``attrs`` keys :func:`common_cluster_subset` stamps on both returned Datasets.
CLUSTER_SET_ATTRS = (
    "eq_common_clusters",
    "eq_only_master_clusters",
    "eq_only_develop_clusters",
    "eq_only_master_mw",
    "eq_only_develop_mw",
)


def common_cluster_subset(ds_master, ds_develop) -> tuple[xr.Dataset, xr.Dataset, dict]:
    """Restrict both Datasets to the clusters they SHARE; report the rest.

    The pooled profile metrics (:func:`p_max_pu_quantiles`,
    :func:`mean_cf_by_zone`, :func:`p_nom_max_by_zone`) are meaningful only over
    a common population. A cluster one side does not have is a **row-set**
    difference, not a distributional one, and pooling it in leaks into every
    quantile row — so it is taken out here and reported by
    :func:`cluster_sets` instead, where a waiver can name it once rather than a
    dozen times.

    Returns ``(master_subset, develop_subset, info)``; ``info`` is what
    :func:`cluster_sets` returned, and the same facts are stamped on both
    subsets' ``attrs`` (:data:`CLUSTER_SET_ATTRS`) so a caller holding only a
    Dataset can still say what was removed.
    """
    info = cluster_sets(ds_master, ds_develop)
    lm, ld = _bus_labels(ds_master), _bus_labels(ds_develop)
    common = sorted(set(lm) & set(ld))
    pos_m = {b: i for i, b in enumerate(lm)}
    pos_d = {b: i for i, b in enumerate(ld)}
    m = ds_master.isel(bus=[pos_m[b] for b in common])
    d = ds_develop.isel(bus=[pos_d[b] for b in common])
    stamp = {
        "eq_common_clusters": info["n_common"],
        "eq_only_master_clusters": list(info["only_master"]),
        "eq_only_develop_clusters": list(info["only_develop"]),
        "eq_only_master_mw": info["only_master_mw"],
        "eq_only_develop_mw": info["only_develop_mw"],
    }
    for side in (m, d):
        side.attrs.update(stamp)
    if not info["equal"]:
        logger.warning(
            "common_cluster_subset: master has %d clusters and develop %d; "
            "master-only=%s (%.1f MW), develop-only=%s (%.1f MW). The pooled profile "
            "metrics are computed over the %d common clusters only.",
            info["n_master"],
            info["n_develop"],
            list(info["only_master"]) or "-",
            info["only_master_mw"],
            list(info["only_develop"]) or "-",
            info["only_develop_mw"],
            info["n_common"],
        )
    return m, d, info


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
    are taken over the flattened ``(time, bus)`` array.

    **Both inputs must be at the same bus resolution.** Flattening does not make
    two different resolutions comparable — it makes the difference invisible.
    A quantile of the pooled ``(time, bus)`` values is a property of the bus
    population as much as of the weather: 544 substation-level wind sites and
    the 19 clusters they aggregate into have genuinely different CF
    distributions (cluster averaging cuts the tails), so comparing the two
    measures the clustering and is guaranteed to differ no matter what the
    refactor did. At prong 2 the caller therefore passes master's nodal file
    through :func:`aggregate_profile_to_clusters` first; prong 1 is already
    bus-for-bus.

    Same resolution is not enough — it must also be the same bus SET. A cluster
    only one side has contributes its whole 8,760-hour column to one pool and
    nothing to the other, which is a row-set difference masquerading as a
    distributional one. :func:`common_cluster_subset` takes those out and
    reports them separately; the caller runs it before this function.

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
        pd.Series(m_vals, index=idx),
        pd.Series(d_vals, index=idx),
        name="quantile",
        fill=None,
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
