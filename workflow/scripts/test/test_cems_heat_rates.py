"""CEMS heat rates are screened like the PUDL ones before they replace them (build_powerplants)."""

import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
duckdb = pytest.importorskip("duckdb")
from build_powerplants import screen_cems_heat_rates  # noqa: E402

COLUMN = "Heat Input (mmBtu/MWh)"


def test_peaker_with_start_up_fuel_is_dropped():
    oil = [11.0, 12.0, 13.0, 13.5, 14.0, 15.0, 16.0, 220.9]  # last: a peaker that barely ran
    gas = [7.0, 7.2, 7.4, 7.5, 7.6, 2.0]  # last: an efficiency above 1
    cems = pd.DataFrame(
        {
            COLUMN: oil + gas,
            "technology_description": ["Petroleum Liquids"] * len(oil)
            + ["Natural Gas Fired Combined Cycle"] * len(gas),
        }
    )
    kept = screen_cems_heat_rates(cems)
    assert 220.9 not in kept[COLUMN].tolist()
    assert 2.0 not in kept[COLUMN].tolist()
    assert len(kept) == len(cems) - 2
    # each technology is judged against its own spread: 7.6 stays for gas, 16 for oil
    assert {7.6, 16.0} <= set(kept[COLUMN])


def test_dropped_unit_keeps_its_pudl_heat_rate(tmp_path):
    from build_powerplants import apply_cems_heat_rates

    pytest.importorskip("openpyxl")
    heat_rates = [11.0, 12.0, 13.0, 13.5, 14.0, 15.0, 16.0, 220.9]
    units = [f"u{i}" for i in range(len(heat_rates))]
    cems = pd.DataFrame({"Facility ID": 1, "Unit ID": units, COLUMN: heat_rates})
    crosswalk = pd.DataFrame(
        {"CAMD_PLANT_ID": 1, "CAMD_UNIT_ID": units, "EIA_PLANT_ID": 10, "EIA_GENERATOR_ID": units},
    )
    plants = pd.DataFrame(
        {
            "plant_id_eia": 10,
            "generator_id": units,
            "technology_description": "Petroleum Liquids",
            "unit_heat_rate_mmbtu_per_mwh": 12.5,
            "unit_heat_rate_mmbtu_per_mwh_source": "pudl_reciepts",
        },
    )
    cems.to_excel(tmp_path / "cems.xlsx", index=False)
    crosswalk.to_csv(tmp_path / "crosswalk.csv", index=False)

    out = apply_cems_heat_rates(plants, tmp_path / "crosswalk.csv", tmp_path / "cems.xlsx").set_index("generator_id")
    assert out.at["u7", "unit_heat_rate_mmbtu_per_mwh"] == 12.5  # the peaker keeps its PUDL heat rate
    assert out.at["u7", "unit_heat_rate_mmbtu_per_mwh_source"] == "unit_heat_rate_mmbtu_per_mwh_source"
    assert out.at["u0", "unit_heat_rate_mmbtu_per_mwh"] == 11.0  # the others take CEMS
    assert out.at["u0", "unit_heat_rate_mmbtu_per_mwh_source"] == "cems"
    assert len(out) == len(plants)
