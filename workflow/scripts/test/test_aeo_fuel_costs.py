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


# every gas, coal and oil plant pays the AEO fuel costs


def test_oil_plants_take_the_distillate_cost(patch_prices):
    patch_prices()

    costs = fuel_costs(build_aeo_fuel_costs("KEY", 2030, "reference"))

    assert costs["oil"] == pytest.approx(21.0 * TO_MWH)


def test_plants_pay_the_aeo_fuel_cost_of_each_period():
    import pypsa
    from _helpers import get_multiindex_snapshots
    from add_extra_components import apply_power_fuel_prices

    n = pypsa.Network()
    n.snapshots = get_multiindex_snapshots(
        sns_config={"start": "2030-01-01 00:00", "end": "2030-01-01 01:00", "inclusive": "both"},
        invest_periods=[2030, 2050],
    )
    n.set_investment_periods(periods=[2030, 2050])
    n.add("Bus", "p1")
    n.add(
        "Generator",
        ["p1 CCGT existing", "p1 coal-95CCS_2050", "p1 oil", "p1 solar"],
        bus="p1",
        carrier=["CCGT", "coal-95CCS", "oil", "solar"],
        efficiency=[0.5, 0.4, 0.25, 1],
        marginal_cost=[30, 40, 100, 0],
    )
    # an existing plant's own fuel cost is replaced
    n.generators["fuel_cost"] = [3.0, float("nan"), 15.0, 0.0]
    n.generators["vom_cost"] = [2.0, float("nan"), 3.0, 0.0]
    costs_dict = {
        period: pd.DataFrame(
            {"fuel_cost_real_per_mwhth": fuel, "opex_variable_per_mwh": [0.0, 4.0, 0.0]},
            index=["CCGT", "coal-95CCS", "oil"],
        )
        for period, fuel in {2030: [10.0, 5.0, 50.0], 2050: [20.0, 6.0, 60.0]}.items()
    }

    apply_power_fuel_prices(n, costs_dict)

    mc = n.generators_t.marginal_cost.groupby(level="period").first()
    assert mc["p1 CCGT existing"].tolist() == pytest.approx([2 + 10 / 0.5, 2 + 20 / 0.5])
    # without a VOM of its own, a plant takes the VOM of its carrier's cost data
    assert mc["p1 coal-95CCS_2050"].tolist() == pytest.approx([4 + 5 / 0.4, 4 + 6 / 0.4])
    assert mc["p1 oil"].tolist() == pytest.approx([3 + 50 / 0.25, 3 + 60 / 0.25])
    assert "p1 solar" not in mc
    assert n.generators.at["p1 CCGT existing", "marginal_cost"] == pytest.approx(2 + 10 / 0.5)
