"""Unit tests for formatting EIA natural gas responses in ``eia.py``.

The raw frames mirror the API v2 response; nothing here touches the network.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import eia

pytestmark = pytest.mark.fast


def test_gas_production_keeps_only_the_requested_year():
    """The inclusive ``end={year+1}-01`` bound also returns January of the next year."""
    raw = pd.DataFrame(
        {
            "period": ["2018-01", "2018-12", "2019-01", "2018-12"],
            "area-name": ["CALIFORNIA", "CALIFORNIA", "CALIFORNIA", "U.S."],
            "series-description": [
                "California Natural Gas Marketed Production (MMcf)",
                "California Natural Gas Marketed Production (MMcf)",
                "California Natural Gas Marketed Production (MMcf)",
                "U.S. Natural Gas Marketed Production (MMcf)",
            ],
            "value": [10.0, 12.0, 99.0, 100.0],
            "units": ["MMCF"] * 4,
        },
    )

    df = eia._GasProduction("market", 2018, "key").format_data(raw)

    assert (df.index.year == 2018).all()
    assert df[df.state == "CA"].value.tolist() == [10.0, 12.0]


@pytest.mark.parametrize(
    ("item", "route", "process"),
    [
        ("marketed", "prod/sum", "VGM"),
        ("dry", "prod/sum", "FPD"),
        ("lease_and_plant_fuel", "cons/sum", "VGL"),
    ],
)
def test_gas_balance_url_targets_the_annual_us_total(item, route, process):
    url = eia.GasBalance(item, 2019, "KEY").get_api_call()

    assert url.startswith(f"{eia.API_BASE}natural-gas/{route}/data/?api_key=KEY&")
    assert "frequency=annual" in url
    assert f"facets[process][]={process}" in url
    assert "facets[duoarea][]=NUS" in url
    assert "start=2019&end=2019" in url


def test_gas_balance_is_indexed_by_year():
    raw = pd.DataFrame(
        {
            "period": ["2019"],
            "duoarea": ["NUS"],
            "area-name": ["U.S."],
            "process": ["VGL"],
            "series-description": ["U.S. Natural Gas Lease and Plant Fuel Consumption (MMcf)"],
            "value": ["1822940"],
            "units": ["MMCF"],
        },
    )

    df = eia._GasBalance("lease_and_plant_fuel", 2019, "key").format_data(raw)

    assert df.index.tolist() == [2019]
    assert df.value.tolist() == [1822940.0]
    assert df.state.tolist() == ["U.S."]


def test_gas_balance_rejects_unknown_items():
    with pytest.raises(eia.InputPropertyError):
        eia.GasBalance("gross", 2019, "KEY").get_api_call()
