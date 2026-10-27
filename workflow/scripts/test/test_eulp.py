"""Tests for the EULP building-type aggregation into useful-energy loads."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from build_demand import ReadEulp
from build_eulp_loads import EQUIPMENT, J_TO_KWH, comstock_factors, read_comstock_metadata
from eulp import ComEulp, ResEulp

pytestmark = pytest.mark.fast

INTERVALS_PER_HOUR = 4
END_USES = ["electricity", "heating", "space_heating", "water_heating", "cooling", "gas_other"]


def _write_profile(path: Path, values: dict[str, float]) -> Path:
    """One year of 15-min intervals stamped at their end, constant per column."""
    stamps = pd.date_range("2018-01-01 00:15", "2019-01-01 00:00", freq="15min")
    df = pd.DataFrame({"upgrade": 0, "in.state": "CO", "timestamp": stamps})
    for column, value in values.items():
        df[column] = value
    df.to_csv(path, index=False)
    return path


def _hourly_mwh(value_per_interval: float) -> float:
    return value_per_interval * INTERVALS_PER_HOUR * 1e-3


def _com_factors(**overrides: float) -> pd.Series:
    columns = [c for cols in ComEulp._useful_groups.values() for c in cols]
    factors = pd.Series(1.0, index=columns)
    for column, value in overrides.items():
        factors[f"out.{column}.energy_consumption.kwh"] = value
    return factors


def test_residential_loads_are_delivered_energy(tmp_path):
    path = _write_profile(
        tmp_path / "single-family_detached.csv",
        {
            "out.load.heating.energy_delivered.kbtu": 10.0,
            "out.load.hot_water.energy_delivered.kbtu": 4.0,
            "out.load.cooling.energy_delivered.kbtu": 6.0,
            # final energy of the same end uses must not be added on top
            "out.natural_gas.heating.energy_consumption.kwh": 100.0,
            "out.electricity.heating.energy_consumption.kwh": 50.0,
            "out.electricity.cooling.energy_consumption.kwh": 2.0,
            "out.natural_gas.hot_water.energy_consumption.kwh": 5.0,
            # gas end uses other than heating stay final energy; propane ones are not modelled
            "out.natural_gas.range_oven.energy_consumption.kwh": 9.0,
            "out.natural_gas.lighting.energy_consumption.kwh": 7.0,
            "out.propane.range_oven.energy_consumption.kwh": 11.0,
            # electricity end uses, including HVAC fans
            "out.electricity.plug_loads.energy_consumption.kwh": 3.0,
            "out.electricity.heating_fans_pumps.energy_consumption.kwh": 1.0,
        },
    )
    data = ResEulp(path).data

    assert list(data.columns) == END_USES
    assert len(data) == 8760
    k = ResEulp.KBTU_TO_KWH
    np.testing.assert_allclose(data.space_heating, _hourly_mwh(10.0 * k))
    np.testing.assert_allclose(data.water_heating, _hourly_mwh(4.0 * k))
    np.testing.assert_allclose(data.cooling, _hourly_mwh(6.0 * k))
    np.testing.assert_allclose(data.heating, data.space_heating + data.water_heating)
    np.testing.assert_allclose(data.electricity, _hourly_mwh(3.0 + 1.0))
    np.testing.assert_allclose(data.gas_other, _hourly_mwh(9.0 + 7.0))


def test_residential_delivered_loads_clip_simulation_noise(tmp_path):
    path = _write_profile(
        tmp_path / "mobile_home.csv",
        {
            "out.load.heating.energy_delivered.kbtu": -1.0,
            "out.load.hot_water.energy_delivered.kbtu": 1.0,
            "out.load.cooling.energy_delivered.kbtu": 1.0,
        },
    )
    assert (ResEulp(path).space_heating == 0).all()


def test_residential_file_without_delivered_loads_fails(tmp_path):
    path = _write_profile(
        tmp_path / "mobile_home.csv",
        {"out.natural_gas.heating.energy_consumption.kwh": 1.0},
    )
    with pytest.raises(AssertionError, match="delivered-load"):
        ResEulp(path)


def test_building_types_of_different_stocks_do_not_add(tmp_path):
    res = ResEulp(
        _write_profile(
            tmp_path / "mobile_home.csv",
            {
                "out.load.heating.energy_delivered.kbtu": 1.0,
                "out.load.hot_water.energy_delivered.kbtu": 1.0,
                "out.load.cooling.energy_delivered.kbtu": 1.0,
            },
        ),
    )
    com = ComEulp(
        _write_profile(tmp_path / "warehouse.csv", {"out.natural_gas.heating.energy_consumption.kwh": 1.0}),
        factors=_com_factors(),
    )
    with pytest.raises(TypeError):
        res + com


def test_commercial_final_energy_is_converted_once(tmp_path):
    path = _write_profile(
        tmp_path / "largeoffice.csv",
        {
            "out.natural_gas.heating.energy_consumption.kwh": 10.0,
            "out.electricity.heating.energy_consumption.kwh": 4.0,
            "out.district_heating.heating.energy_consumption.kwh": 1.0,
            "out.natural_gas.water_systems.energy_consumption.kwh": 2.0,
            "out.electricity.water_systems.energy_consumption.kwh": 3.0,
            "out.electricity.cooling.energy_consumption.kwh": 6.0,
            "out.district_cooling.cooling.energy_consumption.kwh": 2.0,
            "out.natural_gas.interior_equipment.energy_consumption.kwh": 8.0,
            "out.electricity.interior_lighting.energy_consumption.kwh": 5.0,
        },
    )
    factors = _com_factors(
        **{
            "natural_gas.heating": 0.8,
            "electricity.heating": 1.1,
            "natural_gas.water_systems": 0.8,
            "electricity.cooling": 3.0,
        },
    )
    data = ComEulp(path, factors=factors).data

    np.testing.assert_allclose(data.space_heating, _hourly_mwh(10.0 * 0.8 + 4.0 * 1.1 + 1.0))
    np.testing.assert_allclose(data.water_heating, _hourly_mwh(2.0 * 0.8 + 3.0))
    np.testing.assert_allclose(data.cooling, _hourly_mwh(6.0 * 3.0 + 2.0))
    np.testing.assert_allclose(data.heating, data.space_heating + data.water_heating)
    # electric water heating moved to water_heating; gas cooking is its own final-energy load
    np.testing.assert_allclose(data.electricity, _hourly_mwh(5.0))
    np.testing.assert_allclose(data.gas_other, _hourly_mwh(8.0))


def test_commercial_requires_factors_for_every_column(tmp_path):
    path = _write_profile(
        tmp_path / "warehouse.csv",
        {"out.natural_gas.heating.energy_consumption.kwh": 1.0},
    )
    with pytest.raises(AssertionError, match="factors missing"):
        ComEulp(path)
    with pytest.raises(AssertionError, match="factors missing"):
        ComEulp(path, factors=_com_factors().iloc[1:])


def test_building_types_sum(tmp_path):
    values = {
        "out.load.heating.energy_delivered.kbtu": 1.0,
        "out.load.hot_water.energy_delivered.kbtu": 1.0,
        "out.load.cooling.energy_delivered.kbtu": 1.0,
    }
    a = ResEulp(_write_profile(tmp_path / "a.csv", values))
    b = ResEulp(_write_profile(tmp_path / "b.csv", values))
    pd.testing.assert_frame_equal(sum([b], a).data, a.data * 2)


def test_state_file_reads_into_every_service_fuel(tmp_path):
    values = {
        "out.load.heating.energy_delivered.kbtu": 1.0,
        "out.load.hot_water.energy_delivered.kbtu": 1.0,
        "out.load.cooling.energy_delivered.kbtu": 1.0,
        "out.natural_gas.range_oven.energy_consumption.kwh": 1.0,
        "out.electricity.plug_loads.energy_consumption.kwh": 1.0,
    }
    state_file = tmp_path / "CA.csv"
    ResEulp(_write_profile(tmp_path / "mobile_home.csv", values)).to_csv(state_file)

    demand = ReadEulp(str(state_file), "res").read_demand()

    fuels = set(demand.index.get_level_values("fuel"))
    assert fuels == {"electricity", "heat", "space_heat", "water_heat", "cool", "gas_other"}
    gas = demand.xs("gas_other", level="fuel").sum().sum()
    assert gas == pytest.approx(_hourly_mwh(1.0) * 8760)


def _write_metadata(path: Path, buildings: list[dict]) -> Path:
    """ComStock annual results: final energy in kWh, equipment energy in J."""
    final = [c.removesuffix(".kwh") for cols in ComEulp._useful_groups.values() for c in cols]
    rows = []
    for b in buildings:
        row = {"in.hvac_heat_type": b.get("type", "Furnace"), "weight": b.get("weight", 1.0)}
        row.update({c: b.get(c.removeprefix("out.").removesuffix(".energy_consumption"), 0.0) for c in final})
        for param in EQUIPMENT:
            value = b.get(param, 0.0)
            row[f"out.params.{param}"] = value / J_TO_KWH if param.endswith("..j") else value
        rows.append(row)
    pd.DataFrame(rows).to_parquet(path)
    return path


def test_comstock_factors_balance_each_building(tmp_path):
    path = _write_metadata(
        tmp_path / "CO.parquet",
        [
            # furnace: fuel coils at their efficiency
            {"weight": 2.0, "natural_gas.heating": 100.0, "gas_coil_average_efficiency": 0.8},
            # boiler with electric reheat
            {
                "type": "Boiler",
                "natural_gas.heating": 50.0,
                "boiler_gas..j": 50.0,
                "boiler_load..j": 40.0,
                "electricity.heating": 10.0,
                "natural_gas.water_systems": 10.0,
                "electricity.water_systems": 5.0,
            },
            # air-source heat pump with supplemental resistance, DX cooling
            {
                "type": "ASHP",
                "electricity.heating": 20.0,
                "dx_heating_total_dx_electric..j": 10.0,
                "dx_heating_total_dx_load..j": 30.0,
                "dx_heating_total_supplemental_electric..j": 5.0,
                "dx_heating_total_supplemental_load_electric..j": 5.0,
                "electricity.cooling": 10.0,
                "dx_cooling_electric..j": 10.0,
                "dx_cooling_load..j": 35.0,
            },
            # water-source heat pumps on a boiler loop, chiller and district cooling
            {
                "type": "WSHP",
                "natural_gas.heating": 10.0,
                "boiler_gas..j": 10.0,
                "boiler_load..j": 8.0,
                "electricity.heating": 4.0,
                "wa_hp_heating_electric..j": 4.0,
                "wa_hp_heating_load..j": 12.0,
                "electricity.cooling": 20.0,
                "chiller_average_cop..cop": 5.0,
                "district_cooling.cooling": 7.0,
            },
        ],
    )
    factors = comstock_factors(read_comstock_metadata(path))
    col = "out.{}.energy_consumption.kwh".format

    assert set(factors.index) == {c for cols in ComEulp._useful_groups.values() for c in cols}
    # gas: 2 x 100 x 0.8 + 50 x 0.8 + loop boiler 10 x 0.8, over 2 x 100 + 50 + 10
    assert factors[col("natural_gas.heating")] == pytest.approx((160 + 40 + 8) / 260)
    # electric: reheat 10 + heat pump 30 + supplemental 5 + resistance 5 + WSHP compressor 4
    assert factors[col("electricity.heating")] == pytest.approx((10 + 30 + 5 + 5 + 4) / 34)
    assert factors[col("electricity.cooling")] == pytest.approx((35 + 20 * 5) / 30)
    assert factors[col("district_cooling.cooling")] == pytest.approx(1.0)
    assert factors[col("natural_gas.water_systems")] == pytest.approx(0.8)
    assert factors[col("electricity.water_systems")] == pytest.approx(1.0)
    # no final energy in the state: a nominal factor on an all-zero profile
    assert factors[col("other_fuel.heating")] == 1.0
    assert np.isfinite(factors).all()
