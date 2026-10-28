"""
Test the AEO fuel costs of the cost data.

``build_cost_data`` reads the AEO2026 electric power prices of natural gas,
steam coal and distillate fuel oil (Table 3) for the case of ``costs: aeo:
scenario`` from the EIA API, in 2025$/MMBtu, and writes them in 2023$/MWh_th.
The EIA layer is patched throughout.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import eia
from build_cost_data import build_aeo_fuel_costs

# 2025$/MMBtu of each fuel in 2025, 2030 and 2050
AEO_PRICES = {"gas": [4.0, 5.0, 6.0], "coal": [2.0, 2.5, 3.0], "oil": [20.0, 21.0, 22.0]}
TO_MWH = 3.412 / 1.06  # 2025$/MMBtu -> 2023$/MWh_th


@pytest.fixture
def patch_prices(monkeypatch):
    def _patch(units: str = "2025 $/MMBtu") -> None:
        def get_data(self, pivot=False):
            return pd.DataFrame(
                {"series-description": self.fuel, "value": AEO_PRICES[self.fuel], "units": units, "state": "USA"},
                index=pd.Index([2025, 2030, 2050], name="period"),
            )

        monkeypatch.setattr(eia.ProjectedFuelCost, "get_data", get_data)

    return _patch


def fuel_costs(aeo: pd.DataFrame) -> pd.Series:
    return aeo[aeo.parameter == "fuel_cost_real_per_mwhth"].set_index("pypsa-name").value


def test_url_targets_aeo2026_electric_power_prices():
    url = eia.ProjectedFuelCost("coal", "power", "KEY").get_api_call()

    assert url.startswith(f"{eia.API_BASE}aeo/2026/data/?api_key=KEY&")
    # AEO2026 has no Reference case: reference is the Counterfactual Baseline
    assert "facets[scenario][]=cb2026" in url
    assert "facets[seriesId][]=prce_real_elep_NA_stc_NA_NA_y13dlrpmmbtu" in url


def test_other_aeo2026_cases():
    url = eia.ProjectedFuelCost("gas", "power", "KEY", "high_oil_and_gas_supply").get_api_call()

    assert "facets[scenario][]=highogs" in url
    assert "facets[seriesId][]=prce_real_elep_NA_ng_NA_NA_y13dlrpmmbtu" in url


def test_unknown_case_is_rejected():
    with pytest.raises(eia.InputPropertyError):
        eia.ProjectedFuelCost("gas", "power", "KEY", "high").get_api_call()


def test_fuel_costs_are_2023_usd_per_mwh_th(patch_prices):
    patch_prices()

    costs = fuel_costs(build_aeo_fuel_costs("KEY", 2030, "reference"))

    assert costs["natural_gas"] == pytest.approx(5.0 * TO_MWH)
    assert costs["coal"] == pytest.approx(2.5 * TO_MWH)
    assert costs["distillate_fuel_oil"] == pytest.approx(21.0 * TO_MWH)
    assert costs[["CCGT", "OCGT", "CCGT-95CCS", "CCGT-97CCS"]].tolist() == pytest.approx([5.0 * TO_MWH] * 4)
    assert costs[["coal-95CCS", "coal-99CCS"]].tolist() == pytest.approx([2.5 * TO_MWH] * 2)


def test_years_outside_the_aeo_take_the_nearest_year(patch_prices):
    patch_prices()

    assert fuel_costs(build_aeo_fuel_costs("KEY", 2024, "reference"))["natural_gas"] == pytest.approx(4.0 * TO_MWH)


def test_other_dollar_year_is_rejected(patch_prices):
    patch_prices(units="2024 $/MMBtu")

    with pytest.raises(ValueError, match="2025 \\$/MMBtu"):
        build_aeo_fuel_costs("KEY", 2030, "reference")
