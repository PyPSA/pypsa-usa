"""Unit tests for ``capacity_weighted_bus_aggregation``.

The function computes a two-stage weighting: availability-weighted over the
GODEEEP cells *within each substation*, then ``p_nom_max``-weighted (NREL
installable capacity) across the substations of an ``s{simpl}`` cluster. The
alternative, ``weighted_bus_aggregation``, takes one availability-weighted mean
over every cell of the cluster; the two agree only when MW of NREL caps per unit
availability is uniform across the cluster's substations (western onwind:
p10/p90 spread 9.5x, 5.4 % mean hourly mismatch).

Neither is master's math, and these tests make no parity claim. Master's cluster
CF comes from ``simplify_network`` → pypsa 0.30.2 ``aggregateoneport``, which
weights ``p_max_pu`` by **existing** ``p_nom`` and degenerates to a plain
arithmetic mean where a cluster holds no existing capacity (12 of 20 western
onwind clusters). Develop keeps installable-capacity weighting deliberately —
hot-fix HF-25, deltas ledger DL-19.

These tests pin the collapse of the two stages into develop's single
scatter-add:

  * the one-step result equals the explicit two-stage mean,
  * it degenerates to the availability weighting when capacity density is flat,
  * a substation with zero availability keeps its MW (master drops it — HF-24),
  * a cell is assigned to the cluster of its SUBSTATION, not of its own
    polygon, so substation-level cross-polygon leakage survives the rollup.

Synthetic inputs only: a 2x3 "grid" of GODEEEP cells and 4 timesteps, no
``data/`` access.

Tolerance note: the public function returns ``profile`` as float32 for schema
parity with ``weighted_bus_aggregation`` (downstream code is unchanged), so the
spec's 1e-12 agreement is not observable through it; the reference is built in
float64 and compared at float32 resolution (rtol 1e-6).
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from nrel_exclusion.aggregate_godeeep_weighted import (
    capacity_weighted_bus_aggregation,
    weighted_bus_aggregation,
)

pytestmark = pytest.mark.fast

NT = 4
NY, NX = 2, 3  # 6 cells


def make_cf(seed: int = 0) -> xr.DataArray:
    """(time, south_north, west_east) capacity factors, distinct per cell."""
    rng = np.random.default_rng(seed)
    values = rng.uniform(0.05, 0.95, size=(NT, NY, NX))
    return xr.DataArray(
        values,
        dims=("time", "south_north", "west_east"),
        coords={"time": pd.date_range("2019-01-01", periods=NT, freq="h")},
        name="capacity_factor",
    )


def make_avail(values) -> xr.DataArray:
    return xr.DataArray(
        np.asarray(values, dtype=float).reshape(NY, NX),
        dims=("south_north", "west_east"),
    )


def mapping_from(assignment: dict[str, list[tuple[int, int]]]) -> pd.DataFrame:
    """[name, NS, EW] rows from {substation: [(NS, EW), ...]}."""
    rows = [{"name": sub, "NS": ns, "EW": ew} for sub, cells in assignment.items() for (ns, ew) in cells]
    return pd.DataFrame(rows, columns=["name", "NS", "EW"])


def two_stage_reference(
    cf: xr.DataArray,
    avail: xr.DataArray,
    mapping_sub: pd.DataFrame,
    busmap: pd.Series,
    caps_pnom: pd.Series,
) -> pd.DataFrame:
    """The two stages written out explicitly: per-substation mean, then MW mean.

    Substations with zero availability are DROPPED here, so the reference is the
    plain two-stage construction without the MW-retention rule the function
    applies. This is NOT a model of master's weighting (master weights the
    second stage by existing p_nom; see HF-25).
    """
    av = np.nan_to_num(avail.values.astype(float), nan=0.0)
    cfv = cf.values.astype(float)

    sub_cf, sub_w = {}, {}
    for sub, rows in mapping_sub.groupby("name"):
        w = np.array([av[int(r.NS), int(r.EW)] for r in rows.itertuples()])
        if w.sum() <= 0:
            continue  # no availability -> no stage-1 profile, so no weight
        stacked = np.stack([cfv[:, int(r.NS), int(r.EW)] for r in rows.itertuples()], axis=1)
        sub_cf[sub] = (stacked * w).sum(axis=1) / w.sum()
        sub_w[sub] = float(caps_pnom.get(sub, 0.0))

    out: dict[str, np.ndarray] = {}
    for sub, profile in sub_cf.items():
        cluster = busmap.get(str(int(float(sub))), busmap.get(sub))
        num, den = out.setdefault(cluster, [np.zeros(NT), 0.0])
        num += profile * sub_w[sub]
        out[cluster] = [num, den + sub_w[sub]]

    return pd.DataFrame(
        {c: (num / den if den > 0 else np.full(NT, np.nan)) for c, (num, den) in out.items()},
    )


def test_capacity_weighting_equals_two_stage():
    """6 cells, 3 substations, 1 cluster: one-step == explicit two-stage.

    The two-stage construction is availability-within-substation then
    p_nom_max-across-substations; no claim is made about master here.
    """
    cf = make_cf(seed=1)
    avail = make_avail([[0.9, 0.1, 0.4], [0.2, 0.7, 0.3]])
    mapping_sub = mapping_from(
        {
            "10.0": [(0, 0), (0, 1)],
            "20.0": [(0, 2), (1, 0)],
            "30.0": [(1, 1), (1, 2)],
        },
    )
    busmap = pd.Series({"10": "p1 0", "20": "p1 0", "30": "p1 0"})
    # Strongly non-uniform MW per unit availability: 10.0 carries 1000 MW on
    # avail 1.0, 30.0 carries 5 MW on avail 1.0.
    caps = pd.Series({"10.0": 1000.0, "20.0": 200.0, "30.0": 5.0})

    got = capacity_weighted_bus_aggregation(cf, avail, mapping_sub, busmap, caps)
    expected = two_stage_reference(cf, avail, mapping_sub, busmap, caps)

    assert list(got["bus"].values) == ["p1 0"]
    np.testing.assert_allclose(
        got["profile"].sel(bus="p1 0").values.astype(float),
        expected["p1 0"].to_numpy(),
        rtol=1e-6,
        atol=1e-9,
    )
    # The result must actually differ from the flat availability weighting,
    # otherwise the test would pass on the old code path too.
    flat = weighted_bus_aggregation(cf, avail, mapping_sub.assign(name="p1 0"))
    assert not np.allclose(
        got["profile"].sel(bus="p1 0").values.astype(float),
        flat["profile"].sel(bus="p1 0").values.astype(float),
        rtol=1e-3,
    )


def test_uniform_capacity_density_reduces_to_availability():
    """p_nom_max_sub proportional to A_sub => the two weightings agree."""
    cf = make_cf(seed=2)
    avail = make_avail([[0.9, 0.1, 0.4], [0.2, 0.7, 0.3]])
    mapping_sub = mapping_from(
        {
            "10.0": [(0, 0), (0, 1)],
            "20.0": [(0, 2), (1, 0)],
            "30.0": [(1, 1), (1, 2)],
        },
    )
    busmap = pd.Series({"10": "p1 0", "20": "p1 0", "30": "p1 0"})
    density = 250.0  # MW per unit availability, identical for every substation
    caps = pd.Series(
        {
            "10.0": density * (0.9 + 0.1),
            "20.0": density * (0.4 + 0.2),
            "30.0": density * (0.7 + 0.3),
        },
    )

    got = capacity_weighted_bus_aggregation(cf, avail, mapping_sub, busmap, caps)
    # Same cells, one cluster, plain availability weighting.
    flat = weighted_bus_aggregation(cf, avail, mapping_sub.assign(name="p1 0"))

    np.testing.assert_allclose(
        got["profile"].sel(bus="p1 0").values.astype(float),
        flat["profile"].sel(bus="p1 0").values.astype(float),
        rtol=1e-6,
        atol=1e-9,
    )


def test_zero_availability_substation_retains_capacity():
    """A substation master silently drops keeps its MW on develop (HF-24).

    Substation 30.0 sits on two fully excluded cells (A_sub = 0) but carries
    96 MW of NREL p_nom_max. Master drops it from the profile file and loses its
    NREL capacity; here its MW rides on the cluster's availability-weighted CF,
    so the cluster denominator is the full 96 + 104 MW and the profile stays
    finite.
    """
    cf = make_cf(seed=3)
    avail = make_avail([[0.5, 0.5, 0.5], [0.5, 0.0, 0.0]])
    mapping_sub = mapping_from(
        {
            "10.0": [(0, 0), (0, 1)],
            "20.0": [(0, 2), (1, 0)],
            "30.0": [(1, 1), (1, 2)],  # both cells excluded
        },
    )
    busmap = pd.Series({"10": "p1 0", "20": "p1 0", "30": "p1 0"})
    caps = pd.Series({"10.0": 60.0, "20.0": 44.0, "30.0": 96.0})

    got = capacity_weighted_bus_aggregation(cf, avail, mapping_sub, busmap, caps)
    profile = got["profile"].sel(bus="p1 0").values.astype(float)
    assert np.isfinite(profile).all()

    # The retained 96 MW rides on the cluster's availability-weighted CF, so
    # the result is the MW-weighted blend of the two-stage answer (on 104 MW)
    # and the flat availability answer (on 96 MW).
    two_stage = two_stage_reference(cf, avail, mapping_sub, busmap, caps)["p1 0"].to_numpy()
    flat = weighted_bus_aggregation(cf, avail, mapping_sub.assign(name="p1 0"))
    flat = flat["profile"].sel(bus="p1 0").values.astype(float)
    expected = (104.0 * two_stage + 96.0 * flat) / 200.0
    np.testing.assert_allclose(profile, expected, rtol=1e-6, atol=1e-9)

    # And the plain two-stage answer, which ignores the 96 MW, is NOT what we
    # produce (nor is it master's answer — master weights by existing p_nom).
    assert not np.allclose(profile, two_stage, rtol=1e-3)


def test_cell_assigned_by_substation_not_cell_polygon():
    """A cell lands in the cluster of its substation, not of its own polygon.

    Cell (1, 2) belongs to substation 30.0 (cluster p2 0) even though the
    neighbouring cells of the same grid row belong to cluster p1 0. Its CF must
    show up only in p2 0. This is what keeps substation-level assignment (and
    its cross-polygon leakage) intact through the rollup to the cluster.
    """
    cf = make_cf(seed=4)
    avail = make_avail([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    mapping_sub = mapping_from(
        {
            "10.0": [(0, 0), (0, 1), (0, 2)],
            "20.0": [(1, 0), (1, 1)],
            "30.0": [(1, 2)],
        },
    )
    busmap = pd.Series({"10": "p1 0", "20": "p1 0", "30": "p2 0"})
    caps = pd.Series({"10.0": 30.0, "20.0": 20.0, "30.0": 10.0})

    got = capacity_weighted_bus_aggregation(cf, avail, mapping_sub, busmap, caps)
    assert list(got["bus"].values) == ["p1 0", "p2 0"]

    # p2 0 is exactly the single cell (1, 2).
    np.testing.assert_allclose(
        got["profile"].sel(bus="p2 0").values.astype(float),
        cf.values[:, 1, 2],
        rtol=1e-6,
        atol=1e-9,
    )
    # p1 0 never sees it: with uniform availability, 10.0 is the mean of row 0
    # and 20.0 the mean of the first two cells of row 1, p_nom_max-weighted
    # 30:20.
    sub10 = cf.values[:, 0, :].mean(axis=1)
    sub20 = cf.values[:, 1, :2].mean(axis=1)
    np.testing.assert_allclose(
        got["profile"].sel(bus="p1 0").values.astype(float),
        (30.0 * sub10 + 20.0 * sub20) / 50.0,
        rtol=1e-6,
        atol=1e-9,
    )
