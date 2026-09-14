"""Fast unit tests for ``tests.equivalence.metrics``.

3-bus networks and 24-hour datasets only; no ``data/``, no ``resources/``, no
network access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.equivalence import metrics

from .conftest import make_network, make_objective, make_profile

pytestmark = pytest.mark.fast


def test_capacity_by_carrier_identical(tiny_pair):
    m, d = tiny_pair
    out = metrics.capacity_by_carrier(m, d, attr="p_nom")
    assert list(out.columns) == metrics.COLUMNS
    assert set(out.index) == {"solar", "onwind", "CCGT", "battery"}
    assert (out["delta"] == 0.0).all()
    assert (out["delta_pct"] == 0.0).all()
    assert out.loc["solar", "master"] == pytest.approx(100.0)
    assert out.loc["battery", "master"] == pytest.approx(50.0)


def test_capacity_by_carrier_known_delta():
    master = make_network()
    develop = make_network()
    develop.generators.loc["b0 solar", "p_nom"] *= 1.1
    out = metrics.capacity_by_carrier(master, develop, attr="p_nom")
    assert out.loc["solar", "delta_pct"] == pytest.approx(10.0)
    assert out.loc["solar", "delta"] == pytest.approx(10.0)
    others = out.drop(index="solar")
    assert (others["delta_pct"] == 0.0).all()


def test_capacity_by_carrier_p_nom_opt(tiny_pair):
    m, d = tiny_pair
    out = metrics.capacity_by_carrier(m, d, attr="p_nom_opt")
    assert out.loc["solar", "develop"] == pytest.approx(120.0)
    assert out.loc["battery", "develop"] == pytest.approx(60.0)


def test_capacity_by_zone_carrier_uses_bus_attribute():
    """The zone must come from ``buses.reeds_zone``, never from the bus name.

    The adversarial names below are the real cluster-id form: ``p101 1`` is zone
    ``p10``, sub-cluster 1, with no separator — any prefix split on the name
    produces zone labels that match almost nothing.
    """
    names = ("p101 1", "p102 1", "p203 1")
    master = make_network(bus_names=names, zones=("p10", "p10", "p20"))
    develop = make_network(bus_names=names, zones=("p10", "p10", "p20"))
    out = metrics.capacity_by_zone_carrier(master, develop, attr="p_nom")
    assert set(out.index.get_level_values("zone")) == {"p10", "p20"}
    assert out.loc[("p10", "solar"), "master"] == pytest.approx(100.0)
    assert out.loc[("p20", "CCGT"), "master"] == pytest.approx(300.0)
    assert ("p101", "solar") not in out.index


def test_capacity_by_zone_carrier_requires_the_attribute(tiny_network):
    tiny_network.buses.drop(columns=["reeds_zone"], inplace=True)
    with pytest.raises(ValueError, match="reeds_zone"):
        metrics.capacity_by_zone_carrier(tiny_network, tiny_network)


def test_dispatch_by_carrier_respects_snapshot_weightings():
    master = make_network(weighting=1.0)
    develop = make_network(weighting=3.0)
    out = metrics.dispatch_by_carrier(master, develop)
    assert list(out.columns) == metrics.COLUMNS
    for carrier in ("solar", "onwind", "CCGT"):
        assert out.loc[carrier, "develop"] == pytest.approx(3.0 * out.loc[carrier, "master"])
        assert out.loc[carrier, "delta_pct"] == pytest.approx(200.0)


def test_dispatch_storage_counts_discharge_only():
    n = make_network()
    out = metrics.dispatch_by_carrier(n, n)
    # 4 snapshots alternating +20/-20 -> two discharge hours of 20 MWh each.
    assert out.loc["battery", "master"] == pytest.approx(40.0)


def test_dispatch_missing_solution_returns_empty(tiny_network):
    unsolved = make_network(solved=False)
    out = metrics.dispatch_by_carrier(unsolved, tiny_network)
    assert out.empty
    assert list(out.columns) == metrics.COLUMNS


def test_capacity_factor_by_carrier(tiny_pair):
    m, d = tiny_pair
    out = metrics.capacity_factor_by_carrier(m, d)
    # solar: 10 MW for 4 h over p_nom_opt 120 MW * 4 h
    assert out.loc["solar", "master"] == pytest.approx(10.0 / 120.0)
    assert (out["delta"] == 0.0).all()


def test_capacity_factor_missing_solution_returns_empty(tiny_network):
    out = metrics.capacity_factor_by_carrier(make_network(solved=False), tiny_network)
    assert out.empty
    assert list(out.columns) == metrics.COLUMNS


def test_objective_row_normalises_constant():
    """HF-13 regression guard: never compare ``n.objective`` raw."""
    master = make_objective(100.0, 10.0)
    develop = make_objective(110.0, 0.0)
    row = metrics.objective_row(master, develop)
    assert list(row.index) == metrics.OBJECTIVE_INDEX
    assert row["master"] == pytest.approx(110.0)
    assert row["develop"] == pytest.approx(110.0)
    assert row["delta_pct"] == pytest.approx(0.0)
    # The raw values are carried for the reader and must stay different, or the
    # test would pass for a comparator that never normalised anything.
    assert row["master_raw"] == pytest.approx(100.0)
    assert row["develop_raw"] == pytest.approx(110.0)
    assert row["master_constant"] == pytest.approx(10.0)
    assert row["develop_constant"] == pytest.approx(0.0)


def test_objective_row_ledger_ca_numbers():
    """The ledger's measured CA-leg identity (HF-13): relative difference 5.3e-07."""
    master = make_objective(-204_665_929.13, 1_133_255_860.00)
    develop = make_objective(928_590_425.01, 0.0)
    row = metrics.objective_row(master, develop)
    assert abs(row["delta_pct"]) < 1e-4
    assert row["delta_pct"] / 100.0 == pytest.approx(5.3e-07, rel=0.05)
    assert row["master_raw"] != row["develop_raw"]


def test_objective_row_raw_comparison_would_fail():
    """Sanity check that the guard above has teeth."""
    master = make_objective(-204_665_929.13, 1_133_255_860.00)
    develop = make_objective(928_590_425.01, 0.0)
    raw_rel = abs(develop.objective - master.objective) / abs(master.objective)
    assert raw_rel > 1.0  # comparing raw manufactures a >100% difference


def test_objective_constant_missing_is_zero():
    class _Bare:
        objective = 5.0

    assert metrics.objective_constant(_Bare()) == 0.0
    assert metrics.total_objective(_Bare()) == pytest.approx(5.0)


def test_objective_constant_nan_is_zero():
    assert metrics.objective_constant(make_objective(1.0, float("nan"))) == 0.0


def test_p_max_pu_quantiles_disjoint_bus_spaces():
    """Prong 2 has no shared bus ids; the distribution must still compare."""
    ds_m = make_profile(n_bus=3, n_time=24, seed=2, bus_prefix="m")
    ds_d = make_profile(n_bus=5, n_time=24, seed=3, bus_prefix="d")
    assert not set(ds_m.bus.values) & set(ds_d.bus.values)
    out = metrics.p_max_pu_quantiles(ds_m, ds_d)
    assert list(out.columns) == metrics.COLUMNS
    assert "mean" in out.index
    assert "p_nom_max_weighted_mean" in out.index
    assert 0.5 in out.index
    expected = float(np.median(np.asarray(ds_m["profile"].values).ravel()))
    assert out.loc[0.5, "master"] == pytest.approx(expected)
    assert out.loc["mean", "develop"] == pytest.approx(float(np.asarray(ds_d["profile"].values).mean()))


def test_p_max_pu_quantiles_identical_datasets(tiny_profiles):
    m, d = tiny_profiles
    out = metrics.p_max_pu_quantiles(m, d)
    assert np.allclose(out["delta"].to_numpy(), 0.0)


def test_p_nom_max_by_zone():
    ds = make_profile(n_bus=3, n_time=24, seed=4)
    zone = pd.Series({"b0": "p1", "b1": "p1", "b2": "p2"})
    out = metrics.p_nom_max_by_zone(ds, ds, zone, zone)
    assert list(out.index) == ["p1", "p2"]
    assert out.loc["p1", "master"] == pytest.approx(1000.0 + 2000.0)
    assert (out["delta"] == 0.0).all()


def test_p_nom_max_by_zone_tolerates_float_formatted_ids():
    """Master writes '39762.0' where develop writes '39762' — pure representation."""
    ds = make_profile(n_bus=2, n_time=4, seed=5, bus_prefix="")
    ds = ds.assign_coords(bus=["39762.0", "39763.0"])
    zone = pd.Series({"39762": "p1", "39763": "p2"})
    out = metrics.p_nom_max_by_zone(ds, ds, zone, zone)
    assert set(out.index) == {"p1", "p2"}
    assert "<unmapped>" not in out.index


def test_mean_cf_by_zone_is_potential_weighted():
    ds = make_profile(n_bus=2, n_time=8, seed=6)
    zone = pd.Series({"b0": "p1", "b1": "p1"})
    out = metrics.mean_cf_by_zone(ds, ds, zone, zone)
    cf = ds["profile"].mean("time").to_pandas().to_numpy()
    pot = ds["p_nom_max"].to_pandas().to_numpy()
    assert out.loc["p1", "master"] == pytest.approx(float((cf * pot).sum() / pot.sum()))


def test_demand_by_zone_mean_and_peak(tiny_pair):
    m, d = tiny_pair
    out = metrics.demand_by_zone(m, d)
    assert out.index.names == ["zone", "stat"]
    # zone p1 = buses b0 + b1 -> 100 + 200 MW, doubled in the first snapshot
    assert out.loc[("p1", "peak"), "master"] == pytest.approx(600.0)
    assert out.loc[("p1", "mean"), "master"] == pytest.approx((600.0 + 300.0 * 3) / 4)
    assert (out["delta"] == 0.0).all()


def test_frame_appear_from_nothing_is_nan_pct():
    master = pd.Series({"solar": 0.0})
    develop = pd.Series({"solar": 5.0})
    out = metrics.frame(master, develop)
    assert out.loc["solar", "delta"] == pytest.approx(5.0)
    assert np.isnan(out.loc["solar", "delta_pct"])


def test_frame_zero_on_both_sides_is_zero_pct():
    out = metrics.frame(pd.Series({"x": 0.0}), pd.Series({"x": 0.0}))
    assert out.loc["x", "delta_pct"] == 0.0


def test_frame_one_sided_key_reads_as_zero():
    out = metrics.frame(pd.Series({"a": 1.0}), pd.Series({"b": 2.0}))
    assert out.loc["a", "develop"] == 0.0
    assert out.loc["b", "master"] == 0.0
    assert out.loc["a", "delta_pct"] == pytest.approx(-100.0)


def test_available_power_shape():
    ds = make_profile(n_bus=3, n_time=24, seed=7)
    s = metrics.available_power(ds)
    assert len(s) == 24
    assert s.iloc[0] == pytest.approx(
        float((ds["profile"].isel(time=0) * ds["p_nom_max"]).sum()),
    )


# ---------------------------------------------------------------------------
# Verifier defect 4: additive metrics fill with 0.0; ratio metrics keep NaN.
# ---------------------------------------------------------------------------


def test_frame_fill_none_keeps_nan_on_one_side():
    out = metrics.frame(pd.Series({"a": np.nan}), pd.Series({"a": 0.4}), fill=None)
    assert np.isnan(out.loc["a", "master"])
    assert out.loc["a", "develop"] == pytest.approx(0.4)
    assert np.isnan(out.loc["a", "delta"])
    assert np.isnan(out.loc["a", "delta_pct"])


def test_frame_fill_none_keeps_a_one_sided_key_nan():
    out = metrics.frame(pd.Series({"a": 1.0}), pd.Series({"b": 2.0}), fill=None)
    assert np.isnan(out.loc["a", "develop"])
    assert np.isnan(out.loc["b", "master"])


def test_capacity_factor_keeps_nan_for_a_carrier_with_no_capacity():
    """A carrier with no capacity has an undefined CF, not a CF of zero."""
    master = make_network()
    develop = make_network()
    develop.generators.loc["b2 CCGT", ["p_nom", "p_nom_opt"]] = 0.0
    out = metrics.capacity_factor_by_carrier(master, develop)
    assert np.isnan(out.loc["CCGT", "develop"])
    assert out.loc["CCGT", "develop"] != 0.0


def test_capacity_stays_zero_filled():
    """Additive metrics must keep the 0-fill: absent capacity really is 0 MW."""
    master = make_network()
    develop = make_network()
    develop.generators.drop(index="b2 CCGT", inplace=True)
    out = metrics.capacity_by_carrier(master, develop, attr="p_nom")
    assert out.loc["CCGT", "develop"] == 0.0
    assert out.loc["CCGT", "delta_pct"] == pytest.approx(-100.0)


def test_p_max_pu_quantiles_keep_nan_on_an_empty_side():
    ds = make_profile(n_bus=2, n_time=8, seed=40)
    empty = ds.isel(bus=slice(0, 0))
    out = metrics.p_max_pu_quantiles(ds, empty)
    assert np.isnan(out.loc["mean", "develop"])
    assert not np.isnan(out.loc["mean", "master"])


def test_mean_cf_by_zone_keeps_nan_for_a_zero_potential_zone():
    ds = make_profile(n_bus=2, n_time=8, seed=41)
    ds["p_nom_max"][:] = 0.0
    zone = pd.Series({"b0": "p1", "b1": "p1"})
    out = metrics.mean_cf_by_zone(ds, ds, zone, zone)
    assert np.isnan(out.loc["p1", "master"])
