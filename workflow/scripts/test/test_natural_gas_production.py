"""Unit tests for the gas production efficiency and marginal cost in ``build_natural_gas``.

Gas production is priced at the AEO2026 Lower 48 average supply price, read
from the EIA API in 2025$/MMBtu and converted to 2023$/MWh_th, one price per
investment period. Its efficiency is the dry gas, net of lease and plant fuel,
per unit of marketed production. The EIA layer is patched throughout; nothing
here touches the network.
"""

import logging
import os
import sys

import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import eia
from _helpers import get_multiindex_snapshots
from build_natural_gas import GasProcessing, get_production_costs, get_production_efficiency
from constants import NG_Dol_MMBTU_2_MWH

pytestmark = pytest.mark.fast

PAYLOAD = {
    "response": {
        "data": [
            {
                "period": "2030",
                "seriesName": "Natural Gas : Supply Prices : Lower 48 Average",
                "value": "3.76067",
                "unit": "2025 $/MMBtu",
            },
            {
                "period": "2025",
                "seriesName": "Natural Gas : Supply Prices : Lower 48 Average",
                "value": "2.710237",
                "unit": "2025 $/MMBtu",
            },
        ],
    },
}


def _prices(units: str = "2025 $/MMBtu") -> pd.DataFrame:
    """Formatted ``SupplyCost`` data for 2025, 2030 and 2050."""
    return pd.DataFrame(
        {
            "series-description": "Natural Gas : Supply Prices : Lower 48 Average",
            "value": [2.0, 3.0, 4.0],
            "units": units,
            "state": "USA",
        },
        index=pd.Index([2025, 2030, 2050], name="period"),
    )


@pytest.fixture
def patch_prices(monkeypatch):
    def _patch(df: pd.DataFrame) -> None:
        monkeypatch.setattr(eia.SupplyCost, "get_data", lambda self, pivot=False: df)

    return _patch


@pytest.fixture
def patch_balance(monkeypatch):
    """Formatted ``GasBalance`` data holding one volume per item."""

    def _patch(volumes: dict[str, float], units: str = "MMCF") -> None:
        def get_data(self, pivot=False):
            values = [volumes[self.item]] if self.item in volumes else []
            return pd.DataFrame(
                {"series-description": self.item, "value": values, "units": units, "state": "U.S."},
                index=pd.Index([self.year] * len(values), name="period"),
            )

        monkeypatch.setattr(eia.GasBalance, "get_data", get_data)

    return _patch


def test_supply_cost_url_targets_aeo2026_lower48_average():
    url = eia.SupplyCost("gas", "KEY").get_api_call()

    assert url.startswith(f"{eia.API_BASE}aeo/2026/data/?api_key=KEY&")
    assert "facets[scenario][]=cb2026" in url
    assert "facets[seriesId][]=prce_sup_NA_NA_ng_NA_l48_y13dlrpmmbtu" in url


def test_supply_cost_is_indexed_by_year(monkeypatch):
    monkeypatch.setattr(eia.DataExtractor, "_request_eia_data", staticmethod(lambda url: PAYLOAD))

    df = eia.SupplyCost("gas", "KEY").get_data()

    assert df.index.tolist() == [2025, 2030]
    assert df.value.tolist() == [2.710237, 3.76067]
    assert (df.units == "2025 $/MMBtu").all()


def test_production_costs_are_2023_usd_per_mwh(patch_prices):
    patch_prices(_prices())

    costs = get_production_costs("KEY", [2030, 2050])

    assert costs.index.tolist() == [2030, 2050]
    assert costs[2030] == pytest.approx(3.0 / 1.06 * NG_Dol_MMBTU_2_MWH)
    assert costs[2050] == pytest.approx(4.0 / 1.06 * NG_Dol_MMBTU_2_MWH)


def test_periods_outside_aeo_years_take_nearest_year(patch_prices, caplog):
    patch_prices(_prices())

    with caplog.at_level(logging.WARNING):
        costs = get_production_costs("KEY", [2020, 2030, 2060])

    assert costs[2020] == pytest.approx(2.0 / 1.06 * NG_Dol_MMBTU_2_MWH)
    assert costs[2060] == pytest.approx(4.0 / 1.06 * NG_Dol_MMBTU_2_MWH)
    assert "[2020, 2060]" in caplog.text


def test_other_dollar_year_is_rejected(patch_prices):
    patch_prices(_prices(units="2026 $/MMBtu"))

    with pytest.raises(ValueError, match="2025 \\$/MMBtu"):
        get_production_costs("KEY", [2030])


def test_production_efficiency_is_dry_gas_net_of_field_fuel(patch_balance):
    patch_balance({"marketed": 100.0, "dry": 93.0, "lease_and_plant_fuel": 5.0})

    assert get_production_efficiency("KEY", 2019) == pytest.approx(0.88)


def test_production_efficiency_needs_every_balance_item(patch_balance):
    patch_balance({"marketed": 100.0, "dry": 93.0})

    with pytest.raises(ValueError, match="lease_and_plant_fuel"):
        get_production_efficiency("KEY", 2019)


def test_production_efficiency_rejects_other_units(patch_balance):
    patch_balance({"marketed": 100.0, "dry": 93.0, "lease_and_plant_fuel": 5.0}, units="BCF")

    with pytest.raises(ValueError, match="MMCF"):
        get_production_efficiency("KEY", 2019)


LINKS = ["TX gas production", "CA gas production"]


def _build_production(**kwargs) -> pypsa.Network:
    n = pypsa.Network()
    n.snapshots = get_multiindex_snapshots(
        sns_config={"start": "2030-01-01 00:00", "end": "2030-01-01 01:00", "inclusive": "both"},
        invest_periods=[2030, 2040],
    )
    n.set_investment_periods(periods=[2030, 2040])
    n.add("Bus", ["TX gas", "CA gas"], carrier="gas", reeds_state=["TX", "CA"])

    # skip the EIA production capacity request
    production = GasProcessing.__new__(GasProcessing)
    production.interconnect = "usa"
    production._data = pd.DataFrame({"STATE": ["TX", "CA"], "p_nom": [100.0, 50.0]})

    production.build_infrastructure(
        n,
        capacity_multiplier=1,
        production_costs=pd.Series({2030: 10.0, 2040: 12.0}),
        **kwargs,
    )
    return n


KWARGS = {"efficiency": 0.88, "fom": 10725.4, "contraction": False}


def test_production_links_pay_the_wellhead_price_of_each_period():
    """The supply price is paid on the marketed gas the link draws, not scaled by its efficiency."""
    n = _build_production(**KWARGS)

    marginal_cost = n.links_t.marginal_cost[LINKS]
    assert marginal_cost.loc[2030].eq(10.0).all().all()
    assert marginal_cost.loc[2040].eq(12.0).all().all()


def test_production_links_deliver_the_dry_share_of_marketed_gas():
    n = _build_production(**KWARGS)

    assert (n.links.loc[LINKS, "efficiency"] == 0.88).all()
    # marketed production capacity is the input of the link
    assert n.links.loc[LINKS, "p_nom"].tolist() == [100.0, 50.0]
    assert (n.links.loc[LINKS, "capital_cost"] == 10725.4).all()
    assert not n.links.loc[LINKS, "p_nom_extendable"].any()


def test_production_contracts_down_to_zero():
    n = _build_production(**(KWARGS | {"contraction": True}))

    links = n.links.loc[LINKS]
    assert links.p_nom_extendable.all()
    assert (links.p_nom_min == 0).all()
    assert links.p_nom_max.tolist() == [100.0, 50.0]


@pytest.mark.parametrize("missing", list(KWARGS))
def test_production_links_need_every_parameter(missing):
    kwargs = KWARGS.copy()
    del kwargs[missing]

    with pytest.raises(KeyError, match=missing):
        _build_production(**kwargs)
