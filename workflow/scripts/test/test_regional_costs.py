"""Region- and voltage-resolved transmission costs and regional component capex.

The real ReEDS tables are read rather than mocked: the derivations (voltage
ratios, DC/AC ratio, cost level) are cheap assertions that catch a data refresh
changing shape.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from _helpers import calculate_annuity
from add_electricity import apply_regional_multipliers, update_transmission_costs
from add_extra_components import regional_capital_costs
from cluster_network import clustering_for_n_clusters
from constants import REGIONAL_MULTIPLIER_COL
from money import conversion_factor
from regional_cost import (
    BRANCH_CAPEX_COL,
    LINE_CAPEX_STRATEGIES,
    REEDS_TRANSMISSION_COST_YEAR,
    REFERENCE_VOLTAGE_KV,
    TransmissionCostModel,
    assign_branch_capex,
    build_bus_multipliers,
    carrier_capital_charge_rate,
    carrier_multiplier,
    county_multiplier_table,
    county_unit_cost_field,
    dc_ac_line_cost_ratio,
    great_circle_km,
    load_reg_cap_cost_diff,
    load_transmission_basecost,
    load_transmission_pair_costs,
    overnight_delta_capital_cost,
    pair_unit_costs,
    regional_capital_cost,
    renewable_interconnection_enabled,
    transmission_unit_costs,
    voltage_cost_anchors,
    voltage_cost_exponent,
    voltage_ratio,
)

REPO_DATA = Path(__file__).parents[2] / "repo_data" / "ReEDS_Constraints"
TRANSMISSION = REPO_DATA / "transmission"
BASECOST = TRANSMISSION / "rev_transmission_basecost.csv"
DISTANCE_COST = TRANSMISSION / "transmission_distance_cost_500kVac_county.csv"
REG_CAP_COST_DIFF = Path(__file__).parents[2] / "repo_data" / "costs" / "reg_cap_cost_diff_default.csv"
DOLLAR_YEAR = 2023


@pytest.fixture(scope="module")
def basecost():
    return load_transmission_basecost(str(BASECOST))


@pytest.fixture(scope="module")
def pair_table():
    return load_transmission_pair_costs(str(DISTANCE_COST))


@pytest.fixture(scope="module")
def model():
    return TransmissionCostModel.from_files(
        str(DISTANCE_COST),
        str(BASECOST),
        dollar_year=DOLLAR_YEAR,
        voltage_factor=True,
    )


def _costs():
    rows = {
        "HVAC overhead": {"capex_per_mw_km": 1541.0, "cost_recovery_period_years": 60.0, "wacc_real": 0.044},
        "HVDC overhead": {"capex_per_mw_km": 1026.53, "cost_recovery_period_years": 60.0, "wacc_real": 0.044},
        "HVDC submarine": {"capex_per_mw_km": 504.141, "cost_recovery_period_years": 60.0, "wacc_real": 0.044},
        "HVDC inverter pair": {"capex_per_mw_km": 0.0, "cost_recovery_period_years": 60.0, "wacc_real": 0.044},
    }
    costs = pd.DataFrame(rows).T
    annuity = calculate_annuity(60.0, 0.044)
    costs["annualized_capex_per_mw_km"] = annuity * costs["capex_per_mw_km"]
    costs["annualized_capex_per_mw"] = 0.0
    costs.loc["HVDC inverter pair", "annualized_capex_per_mw"] = annuity * 173.73e3
    return costs


# --------------------------------------------------------------------------
# Voltage and DC/AC relationships
# --------------------------------------------------------------------------


def test_voltage_anchors_are_region_medians_of_within_region_ratios(basecost):
    anchors = voltage_cost_anchors(basecost)
    assert list(anchors.index) == [69.0, 138.0, 230.0, 500.0]
    assert anchors.loc[500.0] == pytest.approx(1.0)
    regions = ["TEPPC", "SCE", "MISO", "Southeast"]
    for kv, label in [(69.0, "69ACsingle"), (138.0, "138ACsingle"), (230.0, "230ACsingle")]:
        within = basecost.loc[label, regions] / basecost.loc["500ACsingle", regions]
        assert anchors.loc[kv] == pytest.approx(float(within.median()))
    assert (anchors.diff().dropna() < 0).all()


def test_dc_ac_ratio_is_one_number_across_regions(basecost):
    assert dc_ac_line_cost_ratio(basecost) == pytest.approx(623.10 / 1506.22, rel=1e-5)
    tampered = basecost.copy()
    tampered.loc["500DCbipole", "MISO"] *= 1.5
    with pytest.raises(ValueError, match="differs across ReEDS cost regions"):
        dc_ac_line_cost_ratio(tampered)


def test_voltage_law_passes_through_500kv_and_decreases(basecost):
    exponent = voltage_cost_exponent(voltage_cost_anchors(basecost))
    assert voltage_ratio([REFERENCE_VOLTAGE_KV], exponent) == pytest.approx([1.0])
    assert exponent == pytest.approx(-0.8200, abs=1e-3)
    span = [69.0, 115.0, 138.0, 161.0, 230.0, 345.0, 500.0, 765.0]
    assert (np.diff(voltage_ratio(span, exponent)) < 0).all()
    assert voltage_ratio([np.nan, 0.0], exponent) == pytest.approx([1.0, 1.0])


# --------------------------------------------------------------------------
# County field and the four-level fallback
# --------------------------------------------------------------------------


def test_pair_table_is_symmetric_and_priced_per_great_circle_km(pair_table):
    raw = pd.read_csv(DISTANCE_COST, nrows=1)
    lookup = pair_unit_costs(pair_table)
    first = raw.iloc[0]
    expected = first.USD2004perMW / (first.length_miles * 1.609344)
    assert lookup.loc[(first.r, first.rr)] == pytest.approx(expected)
    assert lookup.loc[(first.rr, first.r)] == pytest.approx(expected)


def test_county_field_uses_only_the_nearest_pairs():
    table = pd.DataFrame(
        {
            "r": ["pA"] * 4,
            "rr": ["pB", "pC", "pD", "pE"],
            "length_km": [10.0, 20.0, 300.0, 400.0],
            "unit_cost": [1000.0, 1100.0, 5000.0, 6000.0],
        },
    )
    assert county_unit_cost_field(table, n_nearest=2).loc["pA"] == pytest.approx(1050.0)


def test_fallback_chain():
    pair_costs = pd.Series({("p06001", "p06003"): 2000.0, ("p06003", "p06001"): 2000.0})
    field = pd.Series({"p06001": 1000.0, "p06003": 1400.0, "p09110": 3000.0, "p09120": 3400.0})
    c0 = pd.Series({"pair": "p06001", "same": "p06001", "ct": "p09001", "none": np.nan}, dtype=object)
    c1 = pd.Series({"pair": "p06003", "same": "p06001", "ct": "p09003", "none": np.nan}, dtype=object)
    unit = transmission_unit_costs(c0, c1, pair_costs, field)
    assert unit["pair"] == pytest.approx(2000.0)  # exact county pair
    assert unit["same"] == pytest.approx(1000.0)  # endpoint county field
    assert unit["ct"] == pytest.approx(3200.0)  # state median (legacy CT FIPS)
    assert unit["none"] == pytest.approx(float(field.median()))  # national median


def test_reference_voltage_cost_level_against_the_national_value(pair_table):
    """Median county at 500 kV against the national 1541 USD/MW-km it replaces."""
    field = county_unit_cost_field(pair_table)
    median_500kv = float(field.median()) * conversion_factor(REEDS_TRANSMISSION_COST_YEAR, DOLLAR_YEAR)
    assert 1 / 1.5 < median_500kv / 1541 < 1.5


# --------------------------------------------------------------------------
# Branch pricing, clustering and annualization
# --------------------------------------------------------------------------


def _base_network():
    n = pypsa.Network()
    for name, x, v, county in [
        ("a", -100.0, 138.0, "p20001"),
        ("b", -99.0, 138.0, "p20003"),
        ("c", -98.0, 500.0, "p20005"),
        ("d", -97.0, 500.0, "p36061"),  # Manhattan: an expensive county
    ]:
        n.add("Bus", name, x=x, y=40.0, v_nom=v, carrier="AC")
        n.buses.loc[name, "county"] = county
    n.add("Line", "ab", bus0="a", bus1="b", s_nom=100.0, r=0.1, x=1.0, length=100.0)
    n.add("Line", "ab2", bus0="a", bus1="b", s_nom=300.0, r=0.1, x=1.0, length=100.0)
    n.add("Line", "bc", bus0="b", bus1="c", s_nom=400.0, r=0.1, x=1.0, length=100.0)
    n.add("Line", "cd", bus0="c", bus1="d", s_nom=500.0, r=0.1, x=1.0, length=100.0)
    n.lines["v_nom"] = [138.0, 69.0, 138.0, 500.0]
    n.add("Link", "dc_fwd", bus0="a", bus1="d", carrier="DC", p_nom=100.0, length=300.0)
    n.links["underwater_fraction"] = 0.0
    return n


def test_assign_branch_capex_resolves_voltage_and_region(model):
    n = _base_network()
    assign_branch_capex(n, model)
    unit = n.lines[BRANCH_CAPEX_COL]
    assert unit.notna().all() and (unit > 0).all()
    # lower voltage, same corridor -> dearer per MW-km
    assert unit["ab2"] > unit["ab"]
    # DC link carries the DC/AC line ratio at 500 kV
    ac_500 = model.ac_unit_cost(pd.Series(["p20001"]), pd.Series(["p36061"]), [500.0]).iloc[0]
    assert n.links.at["dc_fwd", BRANCH_CAPEX_COL] == pytest.approx(ac_500 * model.dc_ratio)


def test_voltage_factor_off_prices_every_line_at_500kv(model):
    n = _base_network()
    flat = TransmissionCostModel(**{**model.__dict__, "voltage_factor": False})
    assign_branch_capex(n, flat)
    assert n.lines.at["ab", BRANCH_CAPEX_COL] == pytest.approx(n.lines.at["ab2", BRANCH_CAPEX_COL])


def test_unit_cost_is_capacity_weighted_through_clustering(model):
    from pypsa.clustering.spatial import get_clustering_from_busmap

    n = _base_network()
    assign_branch_capex(n, model)
    before = n.lines[BRANCH_CAPEX_COL].copy()

    busmap = pd.Series({"a": "a", "b": "bc", "c": "bc", "d": "d"})
    clustered = get_clustering_from_busmap(
        n,
        busmap,
        line_length_factor=1.25,
        # in the pipeline v_nom is unified to 230 kV before any clustering
        line_strategies={**LINE_CAPEX_STRATEGIES, "v_nom": "max"},
        bus_strategies={"county": "first"},
    ).n

    merged = clustered.lines[clustered.lines.bus0.isin(["a", "bc"]) & clustered.lines.bus1.isin(["a", "bc"])]
    expected = (100 * before["ab"] + 300 * before["ab2"]) / 400
    assert float(merged[BRANCH_CAPEX_COL].iloc[0]) == pytest.approx(expected)
    # links are not aggregated; the column survives untouched
    assert clustered.links[BRANCH_CAPEX_COL].notna().all()


def test_update_transmission_costs_regional_and_national(model):
    costs = _costs()
    n = _base_network()
    assign_branch_capex(n, model)

    update_transmission_costs(n, costs, regional=True)
    km = great_circle_km(n, n.lines)
    annuity = calculate_annuity(60.0, 0.044)
    assert n.lines.capital_cost.to_numpy() == pytest.approx((annuity * n.lines[BRANCH_CAPEX_COL] * km).to_numpy())
    link_km = great_circle_km(n, n.links).iloc[0]
    assert n.links.at["dc_fwd", "capital_cost"] == pytest.approx(
        annuity * n.links.at["dc_fwd", BRANCH_CAPEX_COL] * link_km
        + costs.at["HVDC inverter pair", "annualized_capex_per_mw"],
    )

    update_transmission_costs(n, costs, regional=False)
    assert n.lines.capital_cost.to_numpy() == pytest.approx(
        (n.lines.length * costs.at["HVAC overhead", "annualized_capex_per_mw_km"]).to_numpy(),
    )


def test_update_transmission_costs_without_stored_column_is_national():
    costs = _costs()
    n = _base_network()
    update_transmission_costs(n, costs)
    assert n.lines.capital_cost.to_numpy() == pytest.approx(
        (n.lines.length * costs.at["HVAC overhead", "annualized_capex_per_mw_km"]).to_numpy(),
    )


# --------------------------------------------------------------------------
# Component overnight-capex multipliers
# --------------------------------------------------------------------------


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
