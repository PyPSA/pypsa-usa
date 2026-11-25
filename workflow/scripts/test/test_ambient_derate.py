"""Unit tests for the temperature-dependent derate of thermal units (``conventional.ambient_derate``)."""

import os
import sys

import numpy as np
import pandas as pd
import pypsa
import pytest
import xarray as xr

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from _helpers import stamp_weather_calendar
from add_electricity import apply_seasonal_capacity_derates, seasonal_derate_exempt_carriers
from ambient_derate import (
    align_to_snapshots,
    ambient_derate_enabled,
    apply_ambient_derate,
    forced_outage_rate,
    load_bus_temperature,
    load_outage_curves,
)
from build_bus_temperature import area_weights
from constants import AMBIENT_DERATE_MAX_FORCED_OUTAGE, AMBIENT_DERATE_PRIME_MOVER
from scipy.sparse import csr_matrix

pytestmark = pytest.mark.fast

CURVES_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "repo_data",
    "plants",
    "outage_forced_temperature_murphy2019.csv",
)

ON = {"ambient_derate": {"enable": True, "source": "murphy2019", "replaces_seasonal_derate": True}}


@pytest.fixture(scope="module")
def curves():
    return load_outage_curves(CURVES_PATH)


# --- the forced outage curve ------------------------------------------------


def test_vendored_table_has_every_prime_mover_from_minus_15_to_35(curves):
    assert set(curves.columns) == {
        "combined_cycle",
        "combustion_turbine",
        "diesel",
        "hydro_and_psh",
        "nuclear",
        "steam",
    }
    assert curves.index.tolist() == list(range(-15, 40, 5))
    assert curves.at[-15, "combined_cycle"] == pytest.approx(0.149)
    assert curves.at[35, "nuclear"] == pytest.approx(0.124)


def test_curve_returns_the_table_at_tabulated_temperatures(curves):
    for pm in curves:
        rate = forced_outage_rate(curves[pm], curves.index.to_numpy())
        np.testing.assert_allclose(rate, curves[pm].to_numpy())


def test_curve_interpolates_linearly_between_points(curves):
    cc = curves["combined_cycle"]
    assert forced_outage_rate(cc, 12.5) == pytest.approx((0.025 + 0.028) / 2)
    assert forced_outage_rate(cc, 31.0) == pytest.approx(0.041 + (0.072 - 0.041) / 5)


def test_curve_extrapolates_with_the_slope_of_the_end_points(curves):
    cc = curves["combined_cycle"]
    assert forced_outage_rate(cc, -20.0) == pytest.approx(0.149 + (0.149 - 0.081))
    assert forced_outage_rate(cc, 40.0) == pytest.approx(0.072 + (0.072 - 0.041))
    steam = curves["steam"]
    assert forced_outage_rate(steam, 37.0) == pytest.approx(0.140 + (0.140 - 0.114) / 5 * 2)


def test_curve_is_capped_at_0_4_and_floored_at_0(curves):
    ct = curves["combustion_turbine"]
    assert AMBIENT_DERATE_MAX_FORCED_OUTAGE == 0.4
    # 0.199 + 0.1 per 5 degC below -15 passes 0.4 near -25 degC
    assert forced_outage_rate(ct, -40.0) == pytest.approx(0.4)
    assert forced_outage_rate(ct, -24.0) == pytest.approx(0.199 + 0.1 / 5 * 9)
    falling = pd.Series([0.05, 0.01], index=[0, 5])
    assert forced_outage_rate(falling, 20.0) == 0.0


# --- the carrier mapping ----------------------------------------------------


def test_carrier_mapping_follows_reeds_prime_mover_groups(curves):
    assert set(AMBIENT_DERATE_PRIME_MOVER.values()) <= set(curves.columns)
    expected = {
        "CCGT": "combined_cycle",
        "CCGT-95CCS": "combined_cycle",
        "OCGT": "combustion_turbine",
        "OCHT": "combustion_turbine",
        "oil": "diesel",
        "coal": "steam",
        "coal-99CCS": "steam",
        "biomass": "steam",
        "waste": "steam",
        "nuclear": "nuclear",
        "SMR": "nuclear",
    }
    for carrier, pm in expected.items():
        assert AMBIENT_DERATE_PRIME_MOVER[carrier] == pm
    for carrier in ("hydro", "geothermal", "EGS", "solar", "onwind", "ror"):
        assert carrier not in AMBIENT_DERATE_PRIME_MOVER
    assert "hydro_and_psh" not in AMBIENT_DERATE_PRIME_MOVER.values()


# --- settings ---------------------------------------------------------------


def test_disabled_by_default_and_when_off():
    assert not ambient_derate_enabled(None)
    assert not ambient_derate_enabled({"ambient_derate": {"enable": False, "source": "cpuc_servm"}})
    assert ambient_derate_enabled(ON)


def test_reserved_cpuc_servm_source_raises():
    with pytest.raises(NotImplementedError, match="cpuc_servm"):
        ambient_derate_enabled({"ambient_derate": {"enable": True, "source": "cpuc_servm"}})


def test_unknown_source_and_stacking_raise():
    with pytest.raises(ValueError, match="Unknown"):
        ambient_derate_enabled({"ambient_derate": {"enable": True, "source": "nerc_gads"}})
    with pytest.raises(ValueError, match="replaces_seasonal_derate"):
        ambient_derate_enabled(
            {"ambient_derate": {"enable": True, "source": "murphy2019", "replaces_seasonal_derate": False}},
        )


def test_ucap_and_ambient_derate_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        ambient_derate_enabled(ON, {"enable": True, "forced_outage_rates": {"CCGT": 6.1}})
    assert ambient_derate_enabled(ON, {"enable": False})
    # ucap alone is fine
    assert not ambient_derate_enabled(None, {"enable": True})


# --- the seasonal derate is replaced, not stacked ---------------------------


def _seasonal_network():
    n = pypsa.Network()
    times = pd.DatetimeIndex(
        list(pd.date_range("2030-01-15", periods=3, freq="h")) + list(pd.date_range("2030-07-15", periods=3, freq="h")),
    )
    n.snapshots = pd.MultiIndex.from_arrays([times.year, times], names=["period", "timestep"])
    n.set_investment_periods(periods=[2030])
    n.add("Bus", "z1", carrier="AC")
    plants = pd.DataFrame(
        {"carrier": ["CCGT", "geothermal"], "summer_derate": [0.8, 0.9], "winter_derate": [1.0, 0.95]},
        index=["1", "2"],
    )
    n.add("Generator", ["C1", "C2"], bus="z1", carrier=plants.carrier.to_numpy(), p_nom=100.0)
    return n, plants


def test_seasonal_derate_skips_thermal_carriers_when_enabled():
    exempt = seasonal_derate_exempt_carriers(ON, {"enable": False})
    assert "CCGT" in exempt and "geothermal" not in exempt

    n, plants = _seasonal_network()
    apply_seasonal_capacity_derates(n, plants, ["CCGT", "geothermal"], n.snapshots, exempt_carriers=exempt)
    assert "C1" not in n.generators_t.p_max_pu.columns  # thermal: left to the ambient derate
    july = n.generators_t.p_max_pu.index.get_level_values(1).month == 7
    assert np.allclose(n.generators_t.p_max_pu.loc[july, "C2"], 0.9)  # geothermal keeps its seasonal derate


def test_seasonal_derate_applies_to_every_carrier_when_disabled():
    assert seasonal_derate_exempt_carriers({"ambient_derate": {"enable": False}}) == ()
    n, plants = _seasonal_network()
    apply_seasonal_capacity_derates(n, plants, ["CCGT", "geothermal"], n.snapshots)
    july = n.generators_t.p_max_pu.index.get_level_values(1).month == 7
    assert np.allclose(n.generators_t.p_max_pu.loc[july, "C1"], 0.8)


# --- applying it to a network ------------------------------------------------


@pytest.fixture
def temperature(multi_period_base_network):
    """One weather day: z1 cold to mild, z2 hot, z3 mild; covers the extrapolated ends."""
    hours = multi_period_base_network.snapshots.get_level_values(1)[:24]
    return pd.DataFrame(
        {
            "z1": np.linspace(-30.0, 10.0, 24),
            "z2": np.linspace(25.0, 45.0, 24),
            "z3": np.full(24, 12.0),
        },
        index=pd.DatetimeIndex(hours),
    )


@pytest.fixture
def network(multi_period_base_network):
    n = multi_period_base_network
    sns = n.snapshots
    # existing committable CCGT with a stale seasonal-style profile that must be replaced
    n.add(
        "Generator",
        "z1 CCGT existing",
        bus="z1",
        carrier="CCGT",
        p_nom=300.0,
        committable=True,
        p_min_pu=0.5,
        p_max_pu=pd.Series(0.9, index=sns),
        build_year=2000,
        lifetime=60,
    )
    # new build at another bus, static p_max_pu below one (new nuclear takes the fleet mean)
    n.add(
        "Generator",
        "z2 nuclear 2040",
        bus="z2",
        carrier="nuclear",
        p_nom_extendable=True,
        p_max_pu=0.93,
        build_year=2040,
        lifetime=60,
    )
    n.add("Generator", "z2 OCHT retrofit_2030", bus="z2", carrier="OCHT", p_nom_extendable=True, build_year=2030)
    # non-committable must-run unit carrying a static p_min_pu
    n.add("Generator", "z3 coal", bus="z3", carrier="coal", p_nom=200.0, p_min_pu=0.4)
    hydro_profile = pd.Series(np.linspace(0.2, 0.6, len(sns)), index=sns)
    n.add("Generator", "z3 hydro", bus="z3", carrier="hydro", p_nom=50.0, p_max_pu=hydro_profile)
    n.add("Generator", "z3 geothermal", bus="z3", carrier="geothermal", p_nom=20.0, p_max_pu=0.95)
    return n


def _expected(curves, temperature, carrier, bus, periods=2):
    rate = forced_outage_rate(curves[AMBIENT_DERATE_PRIME_MOVER[carrier]], temperature[bus].to_numpy())
    return np.tile(1.0 - rate, periods)


def test_thermal_units_take_one_minus_for_of_their_bus_temperature(network, curves, temperature):
    derated = apply_ambient_derate(network, curves, temperature)

    # the fixture's generic "gas" units carry no PyPSA-USA thermal carrier
    assert set(derated) == {"z1 CCGT existing", "z2 nuclear 2040", "z2 OCHT retrofit_2030", "z3 coal"}
    p_max_pu = network.generators_t.p_max_pu
    for gen, carrier, bus in [
        ("z1 CCGT existing", "CCGT", "z1"),  # replaced, not multiplied by the old 0.9
        ("z2 nuclear 2040", "nuclear", "z2"),
        ("z2 OCHT retrofit_2030", "OCHT", "z2"),
        ("z3 coal", "coal", "z3"),
    ]:
        np.testing.assert_allclose(p_max_pu[gen].to_numpy(), _expected(curves, temperature, carrier, bus))
        assert network.generators.at[gen, "p_max_pu"] == 1.0
    # -30 degC is extrapolated: 0.149 + 15 * (0.149 - 0.081) / 5
    assert p_max_pu["z1 CCGT existing"].iloc[0] == pytest.approx(1 - (0.149 + 3 * 0.068))
    # both investment periods read the same weather day
    first, second = p_max_pu.loc[2030, "z2 nuclear 2040"], p_max_pu.loc[2040, "z2 nuclear 2040"]
    np.testing.assert_allclose(first.to_numpy(), second.to_numpy())


def test_hydro_geothermal_and_renewables_are_untouched(network, curves, temperature):
    before = network.generators_t.p_max_pu[["z3 hydro", "wind1", "solar1"]].copy()
    apply_ambient_derate(network, curves, temperature)

    pd.testing.assert_frame_equal(network.generators_t.p_max_pu[["z3 hydro", "wind1", "solar1"]], before)
    assert "z3 geothermal" not in network.generators_t.p_max_pu.columns
    assert network.generators.at["z3 geothermal", "p_max_pu"] == 0.95


def test_minimum_stable_level_scales_with_availability(network, curves, temperature):
    apply_ambient_derate(network, curves, temperature)

    p_max = network.get_switchable_as_dense("Generator", "p_max_pu")
    p_min = network.get_switchable_as_dense("Generator", "p_min_pu")
    assert (p_min <= p_max + 1e-12).all().all()
    np.testing.assert_allclose(
        p_min["z1 CCGT existing"].to_numpy(),
        0.5 * _expected(curves, temperature, "CCGT", "z1"),
    )
    np.testing.assert_allclose(p_min["z3 coal"].to_numpy(), 0.4 * _expected(curves, temperature, "coal", "z3"))
    # units without a minimum stable level get no p_min_pu series
    assert "z2 nuclear 2040" not in network.generators_t.p_min_pu.columns


def test_bus_without_temperature_takes_its_linked_buses_mean(network, curves, temperature):
    network.add("Bus", "AZ_imports", carrier="imports")
    network.add("Link", ["AZ_z1", "AZ_z2"], bus0="AZ_imports", bus1=["z1", "z2"], p_nom=100.0)
    network.add("Generator", "R PALO VERDE", bus="AZ_imports", carrier="nuclear", p_nom=100.0)

    apply_ambient_derate(network, curves, temperature)

    linked = temperature[["z1", "z2"]].mean(axis=1).to_frame("AZ_imports")
    np.testing.assert_allclose(
        network.generators_t.p_max_pu["R PALO VERDE"].to_numpy(),
        _expected(curves, linked, "nuclear", "AZ_imports"),
    )


def test_temperature_must_cover_the_snapshot_hours(network, curves, temperature):
    with pytest.raises(ValueError, match="does not cover"):
        apply_ambient_derate(network, curves, temperature.iloc[:12])
    shifted = temperature.copy()
    shifted.index = shifted.index + pd.Timedelta("1D")
    with pytest.raises(ValueError, match="does not cover"):
        align_to_snapshots(shifted, network.snapshots)


def test_area_weights_average_the_cells_of_each_region_by_their_area():
    # region 0 covers all of cell 0 and half of cell 1, region 1 a quarter of cell 2
    indicator = csr_matrix(np.array([[1.0, 0.5, 0.0], [0.0, 0.0, 0.25]]))
    weights = area_weights(indicator).toarray()
    np.testing.assert_allclose(weights, [[2 / 3, 1 / 3, 0.0], [0.0, 0.0, 1.0]])
    # the mean of cell temperatures 10, 40 and -5 degC
    np.testing.assert_allclose(weights @ np.array([10.0, 40.0, -5.0]), [20.0, -5.0])


def test_area_weights_reject_a_region_outside_the_cutout():
    with pytest.raises(ValueError, match="do not overlap"):
        area_weights(csr_matrix(np.array([[1.0, 0.0], [0.0, 0.0]])))


def test_load_bus_temperature_reads_the_build_bus_temperature_output(tmp_path):
    time = pd.date_range("2019-01-01", periods=4, freq="h")
    da = xr.DataArray(
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
        coords={"time": time, "name": ["p1 0", "p2 0"]},
        dims=("time", "name"),
        name="temperature",
    )
    path = tmp_path / "temp_air_bus_s10_c4.nc"
    stamp_weather_calendar(da).to_netcdf(path)

    temperature = load_bus_temperature(str(path))
    assert temperature.columns.tolist() == ["p1 0", "p2 0"]
    assert temperature.loc[time[2], "p2 0"] == 6.0
