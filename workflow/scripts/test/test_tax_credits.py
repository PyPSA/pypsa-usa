"""Tax credits per carrier and build year (``costs: itc_modifier`` / ``ptc_modifier``).

Each carrier takes one value for every build year or a mapping ``{build year:
value}``. ``add_extra_components.apply_itc`` scales the capital cost of each
component by the ITC of its build year, ``apply_ptc`` takes the PTC of its build
year off the marginal cost of each extendable vintage.
"""

import logging
import os
import sys

import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from _helpers import get_multiindex_snapshots
from add_extra_components import apply_itc, apply_ptc
from tax_credits import check_tax_credit_years, credit_for_year

pytestmark = pytest.mark.fast

MONETIZATION = 0.9  # apply_itc / apply_ptc keep 90% of the credit


def test_a_number_applies_to_every_build_year():
    assert credit_for_year(0.3, 2030, "itc_modifier", "solar") == 0.3
    assert credit_for_year(0.3, None, "itc_modifier", "solar") == 0.3


def test_a_mapping_applies_to_its_build_years_only():
    spec = {2030: 0.3, 2040: 0.0}
    assert credit_for_year(spec, 2030, "itc_modifier", "solar") == 0.3
    assert credit_for_year(spec, 2040, "itc_modifier", "solar") == 0.0
    assert credit_for_year(spec, 2035, "itc_modifier", "solar") is None
    assert credit_for_year({"2030": 27.5}, 2030, "ptc_modifier", "onwind") == 27.5
    assert credit_for_year(None, 2030, "ptc_modifier", "onwind") is None


def test_a_mapping_needs_the_build_year():
    with pytest.raises(ValueError, match="build year"):
        credit_for_year({2030: 0.3}, None, "itc_modifier", "solar")


@pytest.mark.parametrize(
    "key,spec",
    [("itc_modifier", 1.0), ("itc_modifier", -0.1), ("ptc_modifier", -5.0), ("itc_modifier", {2030: 1.2})],
)
def test_out_of_range_values_raise(key, spec):
    with pytest.raises(ValueError, match=key):
        credit_for_year(spec, 2030, key, "solar")


def test_years_that_are_no_investment_period_warn(caplog):
    with caplog.at_level(logging.WARNING):
        check_tax_credit_years({"itc_modifier": {"solar": {2030: 0.3, 2045: 0.0}, "onwind": 0.3}}, [2030, 2040])
    assert "2045" in caplog.text
    assert "onwind" not in caplog.text


PERIODS = [2030, 2040]


@pytest.fixture
def network():
    n = pypsa.Network()
    n.snapshots = get_multiindex_snapshots(
        sns_config={"start": "2030-01-01 00:00", "end": "2030-01-01 03:00", "inclusive": "both"},
        invest_periods=PERIODS,
    )
    n.set_investment_periods(periods=PERIODS)
    n.add("Bus", "z1")
    for year in PERIODS:
        n.add("Generator", f"solar {year}", bus="z1", carrier="solar", p_nom_extendable=True, build_year=year)
        n.add("Generator", f"wind {year}", bus="z1", carrier="onwind", p_nom_extendable=True, build_year=year)
        n.add("StorageUnit", f"battery {year}", bus="z1", carrier="4hr_battery_storage", build_year=year)
    n.add("Generator", "wind existing", bus="z1", carrier="onwind", p_nom=100, build_year=2010)
    n.generators["capital_cost"] = 100.0
    n.storage_units["capital_cost"] = 100.0
    return n


def test_a_number_credits_every_component_of_the_carrier(network):
    apply_itc(network, {"solar": 0.3, "4hr_battery_storage": 0.3})
    credited = 100.0 * (1 - MONETIZATION * 0.3)
    assert network.generators.loc[["solar 2030", "solar 2040"], "capital_cost"].to_list() == pytest.approx(
        [credited] * 2,
    )
    assert network.storage_units.capital_cost.to_list() == pytest.approx([credited] * 2)
    assert network.generators.loc[["wind 2030", "wind 2040"], "capital_cost"].to_list() == [100.0, 100.0]


def test_each_vintage_gets_the_itc_of_its_build_year(network):
    apply_itc(network, {"solar": {2030: 0.3, 2040: 0.1}, "4hr_battery_storage": {2040: 0.3}})
    assert network.generators.at["solar 2030", "capital_cost"] == pytest.approx(100.0 * (1 - MONETIZATION * 0.3))
    assert network.generators.at["solar 2040", "capital_cost"] == pytest.approx(100.0 * (1 - MONETIZATION * 0.1))
    # 2030 is not listed: no credit
    assert network.storage_units.at["battery 2030", "capital_cost"] == 100.0
    assert network.storage_units.at["battery 2040", "capital_cost"] == pytest.approx(100.0 * (1 - MONETIZATION * 0.3))


def test_no_itc_configured_is_a_noop(network):
    apply_itc(network, None)
    assert (network.generators.capital_cost == 100.0).all()


def _discounted_ptc(ptc, r, lifetime, credit_years=10):
    pv = MONETIZATION * ptc * (1 - (1 + r) ** -credit_years) / r
    return round(pv * r * (1 + r) ** lifetime / ((1 + r) ** lifetime - 1), 2)


def _costs(wacc):
    return pd.DataFrame({"wacc_real": [wacc], "lifetime": [30.0]}, index=["onwind"])


def test_each_vintage_gets_the_ptc_of_its_build_year(network):
    costs_dict = {2030: _costs(0.05), 2040: _costs(0.04)}
    apply_ptc(network, {"onwind": {2030: 27.5, 2040: 10.0}}, costs_dict)
    # each vintage is discounted at the WACC of its own build year
    assert network.generators.at["wind 2030", "marginal_cost"] == pytest.approx(-_discounted_ptc(27.5, 0.05, 30))
    assert network.generators.at["wind 2040", "marginal_cost"] == pytest.approx(-_discounted_ptc(10.0, 0.04, 30))
    assert network.generators.at["wind existing", "marginal_cost"] == 0.0


def test_a_build_year_not_listed_gets_no_ptc(network):
    apply_ptc(network, {"onwind": {2030: 27.5}}, {2030: _costs(0.05), 2040: _costs(0.05)})
    assert network.generators.at["wind 2040", "marginal_cost"] == 0.0


def test_a_number_gives_every_vintage_the_ptc(network):
    apply_ptc(network, {"onwind": 27.5}, {2030: _costs(0.05), 2040: _costs(0.05)})
    expected = -_discounted_ptc(27.5, 0.05, 30)
    assert network.generators.loc[["wind 2030", "wind 2040"], "marginal_cost"].to_list() == pytest.approx(
        [expected, expected],
    )
