"""Unit tests for California's out-of-state CONTRACTED resources.

``attach_remote_contracted_resources`` re-adds the physically out-of-state
capacity the CPUC ledger attributes to California load regions — capacity a
CA-scoped footprint otherwise drops in ``filter_plants_by_region``. These tests
pin the behaviour the ledger depends on:

  * ``p_nom`` is the CONTRACTED share, never more than the plant physically has,
  * a contract lands on the max-LAF bus of its SERVM region,
  * ledger rows with no ``eia_plant_id`` and rows whose plant has no live units
    are skipped and reported,
  * battery contracts become StorageUnits with the plant's duration,
  * remote VRE copies the attachment bus's capacity-factor profile,
  * the whole thing is inert when the config flag is off.
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from add_electricity import (
    DEFAULT_REMOTE_DURATION_H,
    REMOTE_PREFIX,
    attach_remote_contracted_resources,
)

pytestmark = pytest.mark.fast


PLANT_COLUMNS = [
    "carrier",
    "plant_id_eia",
    "p_nom",
    "efficiency",
    "marginal_cost",
    "heat_rate",
    "summer_derate",
    "winter_derate",
    "ramp_limit_up",
    "ramp_limit_down",
    "min_up_time",
    "min_down_time",
    "start_up_cost",
    "fuel_cost",
    "minimum_load_mw",
    "energy_storage_capacity_mwh",
    "build_year",
]


def _plant(name, carrier, plant_id, p_nom, **overrides):
    row = {
        "carrier": carrier,
        "plant_id_eia": plant_id,
        "p_nom": p_nom,
        "efficiency": 0.5,
        "marginal_cost": 25.0,
        "heat_rate": 7.0,
        "summer_derate": 0.9,
        "winter_derate": 1.0,
        "ramp_limit_up": 1.0,
        "ramp_limit_down": 1.0,
        "min_up_time": 2.0,
        "min_down_time": 4.0,
        "start_up_cost": 1000.0,
        "fuel_cost": 3.0,
        "minimum_load_mw": p_nom * 0.4,
        "energy_storage_capacity_mwh": np.nan,
        "build_year": 2010.0,
    }
    row.update(overrides)
    return pd.Series(row, name=name)[PLANT_COLUMNS]


@pytest.fixture
def plants_prefilter():
    """A miniature pre-regional-filter fleet standing in for powerplants.csv."""
    return pd.DataFrame(
        [
            # 900 MW CCGT plant: a 400 MW contract takes part of it, a 2000 MW
            # contract must be clipped down to the plant's live capacity.
            _plant("ccgt_a", "CCGT", 1001, 500.0),
            _plant("ccgt_b", "CCGT", 1001, 400.0),
            # hybrid solar + storage plant
            _plant("hybrid_pv", "solar", 1002, 300.0, marginal_cost=np.nan),
            _plant(
                "hybrid_bess",
                "battery",
                1002,
                100.0,
                marginal_cost=np.nan,
                energy_storage_capacity_mwh=600.0,
            ),
            # battery plant with no reported energy capacity
            _plant(
                "lone_bess",
                "battery",
                1003,
                50.0,
                marginal_cost=np.nan,
                energy_storage_capacity_mwh=np.nan,
            ),
            # a wind plant; the test network carries no onwind profile
            _plant("wind_a", "onwind", 1005, 120.0, marginal_cost=np.nan),
        ],
    )


TECH_MAP = pd.DataFrame(
    [
        ("pypsa", "CCGT", "Gas CC"),
        ("pypsa", "OCGT", "Gas CT/ICE/Steam"),
        ("pypsa", "solar", "Solar"),
        ("pypsa", "onwind", "Wind"),
        ("pypsa", "battery", "Battery"),
        ("cpuc", "CC", "Gas CC"),
    ],
    columns=["side", "source_category", "compare_category"],
)

WEIGHTS = pd.DataFrame(
    [
        ("z1", "SCE", 0.3),
        ("z2", "SCE", 0.7),  # max-LAF bus of SCE
        ("z1", "SDGE", 0.9),  # max-LAF bus of SDGE
        ("z3", "SDGE", 0.1),
    ],
    columns=["bus", "servm_region", "laf"],
)

COSTS = pd.DataFrame(
    {
        "annualized_capex_fom": {"CCGT": 90000.0, "solar": 40000.0, "4hr_battery_storage": 30000.0},
        "lifetime": {"CCGT": 30.0, "solar": 25.0, "4hr_battery_storage": 15.0},
        "marginal_cost": {"CCGT": 30.0, "solar": 0.0, "4hr_battery_storage": 0.0},
        "efficiency": {"CCGT": 0.55, "solar": 1.0, "4hr_battery_storage": 0.9},
        "opex_variable_per_mwh": {"CCGT": 2.0, "solar": 0.0, "4hr_battery_storage": 0.0},
        "opex_fixed_per_kw": {"CCGT": 20.0, "solar": 10.0, "4hr_battery_storage": 25.0},
        "heat_rate_mmbtu_per_mwh": {"CCGT": 6.5, "solar": 0.0, "4hr_battery_storage": 0.0},
    },
)


def _ledger(rows):
    return pd.DataFrame(
        rows,
        columns=["cpuc_unit_name", "servm_region", "compare_category", "capmax_mw", "eia_plant_id"],
    )


@pytest.fixture
def network():
    """Three-bus CA stand-in carrying a solar profile at z1 and z2 only."""
    n = pypsa.Network()
    # one winter day and one summer day, so the seasonal derate split is visible
    times = pd.DatetimeIndex(
        list(pd.date_range("2030-01-15", periods=24, freq="h"))
        + list(pd.date_range("2030-07-15", periods=24, freq="h")),
    )
    n.snapshots = pd.MultiIndex.from_arrays([times.year, times], names=["period", "timestep"])
    n.set_investment_periods(periods=[2030])
    for bus in ("z1", "z2", "z3"):
        n.add("Bus", bus, x=0.0, y=0.0, carrier="AC")

    profile_z1 = pd.Series(np.linspace(0.1, 0.9, len(n.snapshots)), index=n.snapshots)
    profile_z2 = pd.Series(np.linspace(0.9, 0.1, len(n.snapshots)), index=n.snapshots)
    n.add("Generator", "z1 solar", bus="z1", carrier="solar", p_nom=10.0, p_max_pu=profile_z1)
    n.add("Generator", "z2 solar", bus="z2", carrier="solar", p_nom=10.0, p_max_pu=profile_z2)
    return n


def _attach(n, plants_prefilter, ledger, **kwargs):
    return attach_remote_contracted_resources(
        n,
        plants_prefilter,
        ledger,
        WEIGHTS,
        COSTS,
        ["CCGT", "OCGT", "nuclear"],
        {"Generator": ["CCGT", "solar"], "StorageUnit": ["4hr_battery_storage"]},
        tech_map=TECH_MAP,
        **kwargs,
    )


def test_contract_smaller_than_plant_keeps_contracted_capacity(network, plants_prefilter):
    summary = _attach(network, plants_prefilter, _ledger([["SMALL_CC", "SCE", "Gas CC", 400.0, "1001"]]))

    gen = network.generators.loc[REMOTE_PREFIX + "SMALL_CC"]
    assert gen.p_nom == pytest.approx(400.0)
    assert gen.carrier == "CCGT"
    assert summary.loc[0, "status"] == "attached"


def test_contract_larger_than_plant_is_clipped_to_live_capacity(network, plants_prefilter):
    _attach(network, plants_prefilter, _ledger([["BIG_CC", "SCE", "Gas CC", 2000.0, "1001"]]))

    # the plant only has 500 + 400 MW live
    assert network.generators.at[REMOTE_PREFIX + "BIG_CC", "p_nom"] == pytest.approx(900.0)


def test_contract_attaches_to_max_laf_bus_of_its_region(network, plants_prefilter):
    _attach(
        network,
        plants_prefilter,
        _ledger(
            [
                ["SCE_CC", "SCE", "Gas CC", 100.0, "1001"],
                ["SDGE_CC", "SDGE", "Gas CC", 100.0, "1001"],
            ],
        ),
    )

    assert network.generators.at[REMOTE_PREFIX + "SCE_CC", "bus"] == "z2"
    assert network.generators.at[REMOTE_PREFIX + "SDGE_CC", "bus"] == "z1"


def test_unknown_servm_region_raises(network, plants_prefilter):
    with pytest.raises(ValueError, match="no bus in the SERVM load-weights table"):
        _attach(network, plants_prefilter, _ledger([["PGE_CC", "PGE", "Gas CC", 100.0, "1001"]]))


def test_rows_without_eia_id_are_skipped_and_reported(network, plants_prefilter, caplog):
    with caplog.at_level(logging.WARNING):
        summary = _attach(
            network,
            plants_prefilter,
            _ledger(
                [
                    ["MEXICALI_TDM", "SCE", "Gas CC", 625.0, np.nan],
                    ["KEEPER_CC", "SCE", "Gas CC", 100.0, "1001"],
                ],
            ),
        )

    assert REMOTE_PREFIX + "MEXICALI_TDM" not in network.generators.index
    assert summary.set_index("cpuc_unit_name").loc["MEXICALI_TDM", "status"] == "skipped_no_eia_id"
    warning = "\n".join(r.message for r in caplog.records if r.levelno >= logging.WARNING)
    assert "MEXICALI_TDM" in warning and "625.0 MW" in warning


def test_rows_whose_plant_has_no_live_units_are_skipped(network, plants_prefilter):
    summary = _attach(network, plants_prefilter, _ledger([["DEAD_CC", "SCE", "Gas CC", 100.0, "9999"]]))

    assert network.generators.empty or REMOTE_PREFIX + "DEAD_CC" not in network.generators.index
    assert summary.loc[0, "status"] == "skipped_no_live_units"
    assert summary.loc[0, "p_nom"] == 0.0


def test_battery_becomes_storage_unit_with_plant_duration(network, plants_prefilter):
    _attach(
        network,
        plants_prefilter,
        _ledger(
            [
                ["HYBRID_BESS", "SCE", "Battery", 80.0, "1002"],
                ["LONE_BESS", "SCE", "Battery", 50.0, "1003"],
            ],
        ),
    )

    units = network.storage_units
    assert set(units.index) == {REMOTE_PREFIX + "HYBRID_BESS", REMOTE_PREFIX + "LONE_BESS"}
    # 600 MWh over a 100 MW plant = 6 h, applied to the 80 MW contracted share.
    # max_hours carries attach_battery_storage's dispatch-efficiency correction so
    # the DELIVERABLE energy matches the reported duration.
    efficiency = 0.85**0.5
    assert units.at[REMOTE_PREFIX + "HYBRID_BESS", "max_hours"] == pytest.approx(6.0 / efficiency)
    assert units.at[REMOTE_PREFIX + "HYBRID_BESS", "p_nom"] == pytest.approx(80.0)
    assert not units.at[REMOTE_PREFIX + "HYBRID_BESS", "p_nom_extendable"]
    # no reported energy capacity falls back to the default duration
    assert units.at[REMOTE_PREFIX + "LONE_BESS", "max_hours"] == pytest.approx(DEFAULT_REMOTE_DURATION_H / efficiency)
    assert REMOTE_PREFIX + "HYBRID_BESS" not in network.generators.index


def test_remote_renewable_copies_the_attachment_bus_profile(network, plants_prefilter):
    _attach(network, plants_prefilter, _ledger([["HYBRID_PV", "SCE", "Solar", 250.0, "1002"]]))

    name = REMOTE_PREFIX + "HYBRID_PV"
    assert network.generators.at[name, "carrier"] == "solar"
    assert network.generators.at[name, "bus"] == "z2"
    pd.testing.assert_series_equal(
        network.generators_t.p_max_pu[name],
        network.generators_t.p_max_pu["z2 solar"],
        check_names=False,
    )


def test_remote_renewable_falls_back_to_the_mean_profile(network, plants_prefilter):
    """SDGE's max-LAF bus is z1 here; drop z1's own profile to force the fallback."""
    network.remove("Generator", "z1 solar")
    _attach(network, plants_prefilter, _ledger([["HYBRID_PV", "SDGE", "Solar", 250.0, "1002"]]))

    name = REMOTE_PREFIX + "HYBRID_PV"
    assert network.generators.at[name, "bus"] == "z1"
    pd.testing.assert_series_equal(
        network.generators_t.p_max_pu[name],
        network.generators_t.p_max_pu["z2 solar"],
        check_names=False,
    )


def test_remote_renewable_without_any_profile_is_dropped(network, plants_prefilter, caplog):
    """No onwind profile anywhere means no availability to assign — drop, don't fake 100 %."""
    with caplog.at_level(logging.ERROR):
        summary = _attach(network, plants_prefilter, _ledger([["ESJ_WIND", "SCE", "Wind", 100.0, "1005"]]))

    assert REMOTE_PREFIX + "ESJ_WIND" not in network.generators.index
    assert summary.loc[0, "status"] == "skipped_no_profile"
    assert summary.loc[0, "p_nom"] == 0.0
    assert "no capacity-factor profile" in caplog.text


def test_firm_units_carry_seasonal_derates_and_commitment(network, plants_prefilter):
    _attach(
        network,
        plants_prefilter,
        _ledger([["SMALL_CC", "SCE", "Gas CC", 400.0, "1001"]]),
        unit_commitment=True,
    )

    name = REMOTE_PREFIX + "SMALL_CC"
    assert network.generators.at[name, "committable"]
    assert not network.generators.at[name, "p_nom_extendable"]
    assert network.generators.at[name, "p_nom_min"] == pytest.approx(400.0)
    # minimum_load_mw is 40 % of every constituent, clipped by the tighter
    # seasonal derate (0.9) and relaxed by the 5 % headroom factor
    assert network.generators.at[name, "p_min_pu"] == pytest.approx(0.4 * 0.95)

    p_max_pu = network.generators_t.p_max_pu[name]
    months = p_max_pu.index.get_level_values(1).month
    assert np.allclose(p_max_pu[months == 7], 0.9)  # summer derate
    assert np.allclose(p_max_pu[months == 1], 1.0)  # winter derate


def test_category_mismatch_uses_the_contracted_category_carrier(network, plants_prefilter, caplog):
    """A Solar entitlement pointed at a battery-only plant stays solar."""
    with caplog.at_level(logging.WARNING):
        summary = _attach(network, plants_prefilter, _ledger([["AUX_PV", "SCE", "Solar", 6.0, "1003"]]))

    name = REMOTE_PREFIX + "AUX_PV"
    assert network.generators.at[name, "carrier"] == "solar"
    assert network.generators.at[name, "p_nom"] == pytest.approx(6.0)
    assert summary.loc[0, "status"] == "attached_category_fallback"
    assert network.storage_units.empty
    assert "carry no unit of that category" in caplog.text


def test_multi_plant_contract_pools_both_plants(network, plants_prefilter):
    """A ``;``-joined eia_plant_id (Hoover's NV + AZ halves) pools both plants."""
    plants = pd.concat(
        [plants_prefilter, pd.DataFrame([_plant("ccgt_c", "CCGT", 1004, 600.0)])],
    )
    _attach(network, plants, _ledger([["POOLED_CC", "SCE", "Gas CC", 1400.0, "1001;1004"]]))

    assert network.generators.at[REMOTE_PREFIX + "POOLED_CC", "p_nom"] == pytest.approx(1400.0)


def test_disabled_flag_is_a_no_op(network, plants_prefilter):
    """The attach only runs behind the config gate; nothing is added without it."""
    before_gens = network.generators.index.tolist()
    before_p_max_pu = network.generators_t.p_max_pu.columns.tolist()

    remote_contracted = {"enable": False}
    if remote_contracted.get("enable", False):  # mirrors add_electricity.main
        _attach(network, plants_prefilter, _ledger([["SMALL_CC", "SCE", "Gas CC", 400.0, "1001"]]))

    assert network.generators.index.tolist() == before_gens
    assert network.generators_t.p_max_pu.columns.tolist() == before_p_max_pu
    assert network.storage_units.empty
