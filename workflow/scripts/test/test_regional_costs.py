"""Regional component capex: the ReEDS county overnight-capex multipliers.

The real ReEDS table is read rather than mocked, so a data refresh that changes its
shape fails here.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from _helpers import calculate_annuity
from add_electricity import apply_regional_multipliers
from add_extra_components import regional_capital_costs
from cluster_network import clustering_for_n_clusters
from constants import REGIONAL_MULTIPLIER_COL
from regional_cost import (
    build_bus_multipliers,
    carrier_capital_charge_rate,
    carrier_multiplier,
    county_multiplier_table,
    load_reg_cap_cost_diff,
    overnight_delta_capital_cost,
    regional_capital_cost,
    renewable_interconnection_enabled,
)

REG_CAP_COST_DIFF = Path(__file__).parents[2] / "repo_data" / "costs" / "reg_cap_cost_diff_default.csv"


def _gas_costs(overnight: float = 1000.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "annualized_capex_fom": {"gas": 150_000.0},
            "capex_overnight_per_kw": {"gas": overnight},
            "cost_recovery_period_years": {"gas": 30.0},
            "wacc_real": {"gas": 0.05},
        },
    )


@pytest.mark.fast
def test_county_multiplier_table_expands_groups():
    table = county_multiplier_table(
        load_reg_cap_cost_diff(str(REG_CAP_COST_DIFF)),
        {"NUCLEAR": ["nuclear", "SMR"], "BATTERY|PVB": ["4hr_battery_storage"]},
    )
    raw = load_reg_cap_cost_diff(str(REG_CAP_COST_DIFF))
    assert table.loc["p01001", "nuclear"] == pytest.approx(1 + raw.loc["p01001", "NUCLEAR"])
    assert (table["nuclear"] == table["SMR"]).all()


@pytest.mark.fast
@pytest.mark.parametrize("column_to_tech", [None, {}])
def test_empty_column_to_tech_means_no_multipliers(base_network, column_to_tech):
    """The schema accepts a null mapping: the interconnection cost on its own."""
    raw = load_reg_cap_cost_diff(str(REG_CAP_COST_DIFF))
    table = county_multiplier_table(raw, column_to_tech)
    assert table.empty and table.index.equals(raw.index)

    cfg = {"enable": True, "renewable_interconnection": True, "column_to_tech": column_to_tech}
    assert build_bus_multipliers(base_network, cfg, str(REG_CAP_COST_DIFF), "county_shapes.shp") is None
    assert renewable_interconnection_enabled(cfg)
    # every carrier then keeps its uniform cost
    assert apply_regional_multipliers(base_network, _gas_costs(), None) == set()
    assert REGIONAL_MULTIPLIER_COL not in base_network.generators


@pytest.mark.fast
def test_enabled_multipliers_still_need_their_inputs(base_network):
    cfg = {"enable": True, "column_to_tech": {"NUCLEAR": ["nuclear"]}}
    with pytest.raises(ValueError, match="input is missing"):
        build_bus_multipliers(base_network, cfg, None, None)


@pytest.mark.fast
def test_regional_costs_use_the_capital_charge_rate():
    costs = pd.DataFrame(
        {"capital_charge_rate": [0.0342, 0.0], "cost_recovery_period_years": [30, 30], "wacc_real": [0.04, 0.04]},
        index=["solar", "CCGT"],
    )
    assert carrier_capital_charge_rate(costs, "solar") == 0.0342
    # load_costs zero-fills a missing rate: fall back to the capital recovery factor
    assert carrier_capital_charge_rate(costs, "CCGT") == pytest.approx(calculate_annuity(30, 0.04))


@pytest.mark.fast
def test_overnight_multiplier_leaves_fom_untouched():
    costs = pd.DataFrame(
        {
            "annualized_capex_fom": {"CCGT": 150_000.0},
            "capex_overnight_per_kw": {"CCGT": 1000.0},
            "cost_recovery_period_years": {"CCGT": 30.0},
            "wacc_real": {"CCGT": 0.05},
        },
    )
    bus_multipliers = pd.DataFrame({"CCGT": {"b1": 1.1, "b2": 0.9}})
    got = regional_capital_cost(costs, "CCGT", ["b1", "b2", "b1"], bus_multipliers)
    annuity = calculate_annuity(30.0, 0.05)
    assert got == pytest.approx(
        [150_000 + annuity * 0.1 * 1e6, 150_000 - annuity * 0.1 * 1e6, 150_000 + annuity * 0.1 * 1e6],
    )
    assert overnight_delta_capital_cost(150_000.0, 1000.0, annuity, 1.0) == pytest.approx(150_000.0)
    # unmapped carrier or disabled feature: uniform national value
    assert carrier_multiplier(bus_multipliers, "OCGT", ["b1"]) == 1.0
    assert regional_capital_cost(costs, "CCGT", ["b1"], None) == pytest.approx(150_000.0)


@pytest.mark.fast
def test_own_multiplier_replaces_the_bus_multiplier():
    costs = _gas_costs()
    bus_multipliers = pd.DataFrame({"gas": {"b1": 1.1, "b2": 0.9}})
    annuity = calculate_annuity(30.0, 0.05)

    got = regional_capital_cost(costs, "gas", ["b1", "b2", "b2"], bus_multipliers, own_multiplier=[1.3, np.nan, 0.0])

    # a component without a multiplier of its own (NaN, or the 0 an all-NaN group sums to) takes its bus's
    assert got == pytest.approx(
        [150_000 + annuity * 0.3 * 1e6, 150_000 - annuity * 0.1 * 1e6, 150_000 - annuity * 0.1 * 1e6],
    )
    # the feature being off wins over a stored multiplier
    assert regional_capital_cost(costs, "gas", ["b1"], None, own_multiplier=[1.3]) == pytest.approx(150_000.0)


@pytest.mark.fast
def test_vintages_take_the_multiplier_the_merged_generator_was_priced_with(base_network):
    """A multi-period run prices a merged generator like the single-period run does.

    ``add_electricity`` prices each generator with the county of its own bus and
    clustering averages those costs; the county of the clustered bus is a different
    multiplier.
    """
    n = base_network
    gas = ["gas1", "gas2"]
    n.generators.loc[gas, "bus"] = ["z2", "z3"]
    n.generators.loc[gas, "p_nom"] = [100.0, 300.0]
    n.generators.loc[gas, "p_nom_extendable"] = True
    costs = _gas_costs()
    annuity = calculate_annuity(30.0, 0.05)

    carriers = apply_regional_multipliers(n, costs, pd.DataFrame({"gas": {"z1": 1.0, "z2": 0.9, "z3": 1.2}}))

    assert carriers == {"gas"}
    assert n.generators.loc[gas, REGIONAL_MULTIPLIER_COL].tolist() == [0.9, 1.2]
    assert n.generators.at["gas2", "capital_cost"] == pytest.approx(150_000 + annuity * 0.2 * 1e6)

    clustered = clustering_for_n_clusters(
        n,
        2,
        custom_busmap=pd.Series({"z1": "z1", "z2": "X", "z3": "X"}),
        aggregate_carriers=["gas"],
    ).n
    merged = clustered.generators.loc[["X gas"]]
    multiplier = 0.25 * 0.9 + 0.75 * 1.2
    assert merged.at["X gas", REGIONAL_MULTIPLIER_COL] == pytest.approx(multiplier)
    assert merged.at["X gas", "capital_cost"] == pytest.approx(150_000 + annuity * (multiplier - 1) * 1e6)

    clustered_bus_multipliers = pd.DataFrame({"gas": {"z1": 1.0, "X": 1.5}})
    by_bus = regional_capital_costs(costs, merged.carrier, merged.bus, clustered_bus_multipliers)
    own = regional_capital_costs(
        costs,
        merged.carrier,
        merged.bus,
        clustered_bus_multipliers,
        merged[REGIONAL_MULTIPLIER_COL],
    )
    assert by_bus["X gas"] == pytest.approx(150_000 + annuity * 0.5 * 1e6)
    assert own["X gas"] == pytest.approx(merged.at["X gas", "capital_cost"])
    # a later vintage scales that year's overnight capex by the same multiplier
    later = regional_capital_costs(
        _gas_costs(overnight=800.0),
        merged.carrier,
        merged.bus,
        clustered_bus_multipliers,
        merged[REGIONAL_MULTIPLIER_COL],
    )
    assert later["X gas"] == pytest.approx(150_000 + annuity * (multiplier - 1) * 800e3)


@pytest.mark.fast
def test_attached_vintages_carry_the_multiplier(multi_period_base_network):
    from add_extra_components import attach_multihorizon_existing_generators

    n = multi_period_base_network
    n.generators["land_region"] = n.generators.bus
    n.generators.loc["gas2", REGIONAL_MULTIPLIER_COL] = 1.125
    gens = n.generators.loc[["gas2"]]
    costs = _gas_costs().assign(efficiency=0.5, opex_variable_per_mwh=0.0, heat_rate_mmbtu_per_mwh=0.0)
    annuity = calculate_annuity(30.0, 0.05)

    for year in n.investment_periods:
        attach_multihorizon_existing_generators(
            n,
            costs,
            gens,
            year,
            bus_multipliers=pd.DataFrame({"gas": {"z1": 1.0, "z2": 1.0, "z3": 1.5}}),
        )
        assert n.generators.at[f"gas2 {year}", "capital_cost"] == pytest.approx(150_000 + annuity * 0.125 * 1e6)
        assert n.generators.at[f"gas2 {year}", REGIONAL_MULTIPLIER_COL] == 1.125
