"""ReEDS wind/solar interconnection cost.

The interconnection tables shipped in repo_data are read as published.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from _helpers import calculate_annuity
from add_extra_components import apply_renewable_interconnection_costs
from cluster_network import clustering_for_n_clusters, generator_aggregation_strategies
from constants import (
    GENERATOR_CAPEX_STRATEGIES,
    RENEWABLE_INTERCONNECTION_COL,
    RENEWABLE_INTERCONNECTION_SHARE_COL,
)
from reeds_vre import bus_weighted_mean, load_interconnection_costs
from regional_cost import (
    renewable_interconnection_capital_cost,
    renewable_interconnection_enabled,
)

REPO_COSTS = Path(__file__).resolve().parents[2] / "repo_data" / "costs"

SOLAR = ["solar1", "solar2"]
#: merges the fixture's two eastern buses, which share every regional attribute
EAST_BUSMAP = pd.Series({"z1": "z1", "z2": "X", "z3": "X"})


def _costs() -> pd.DataFrame:
    """ATB-shaped cost rows: capex_per_kw = overnight + grid connection + finance."""
    rows = {
        "onwind": {"overnight": 940.0, "gcc": 100.0, "cff": 57.0, "fom": 25.0, "wacc": 0.032},
        "solar": {"overnight": 572.0, "gcc": 100.0, "cff": 22.0, "fom": 15.5, "wacc": 0.031},
    }
    out = {}
    for carrier, r in rows.items():
        annuity = calculate_annuity(30, r["wacc"])
        out[carrier] = {
            "capex_overnight_per_kw": r["overnight"],
            "capex_grid_connection_per_kw": r["gcc"],
            "capex_construction_finance_factor": r["cff"],
            "opex_fixed_per_kw": r["fom"],
            "cost_recovery_period_years": 30.0,
            "wacc_real": r["wacc"],
            "annualized_capex_fom": annuity * (r["overnight"] + r["gcc"] + r["cff"]) * 1e3 + r["fom"] * 1e3,
        }
    return pd.DataFrame(out).T


def _two_solar_sites(n, interconnection, p_nom=(100.0, 300.0)):
    """Put the fixture's two solar generators at the eastern buses, as ``add_electricity`` leaves them.

    Each carries its bus's ReEDS interconnection cost (NaN: no site), the share of
    its capacity that cost covers, and the capital cost priced from it.
    """
    interconnection = pd.Series(interconnection, index=SOLAR, dtype=float)
    n.generators.loc[SOLAR, "bus"] = ["z2", "z3"]
    n.generators.loc[SOLAR, "p_nom"] = list(p_nom)
    n.generators.loc[SOLAR, "land_region"] = "east"
    n.generators.loc[SOLAR, "capital_cost"] = renewable_interconnection_capital_cost(
        _costs(),
        "solar",
        interconnection,
    ).to_numpy()
    n.generators.loc[SOLAR, RENEWABLE_INTERCONNECTION_COL] = interconnection.to_numpy()
    n.generators.loc[SOLAR, RENEWABLE_INTERCONNECTION_SHARE_COL] = (interconnection > 0).to_numpy().astype(float)
    return n


def _merge_east(n, aggregation_strategies=None):
    """Cluster buses z2 and z3 into X the way ``cluster_network`` does, merging the solar generators."""
    return clustering_for_n_clusters(
        n,
        2,
        custom_busmap=EAST_BUSMAP,
        aggregate_carriers=["solar"],
        aggregation_strategies=aggregation_strategies or {},
    ).n


@pytest.mark.fast
def test_bus_weighted_mean_skips_unknown_sites():
    sites = pd.DataFrame(
        {
            "bus": ["a", "a", "a", "b"],
            "capacity": [1.0, 3.0, 5.0, 2.0],
            "interconnection_cost": [100.0, 200.0, np.nan, np.nan],
        },
    )
    out = bus_weighted_mean(sites, "interconnection_cost")
    assert out["a"] == pytest.approx((100 + 3 * 200) / 4)
    assert "b" not in out.index or np.isnan(out.get("b", np.nan))


@pytest.mark.fast
@pytest.mark.parametrize("tech", ["onwind", "solar", "offwind", "offwind_floating"])
def test_shipped_interconnection_tables_convert_dollar_year(tech):
    kind = "offshore" if tech.startswith("offwind") else "land"
    path = REPO_COSTS / f"interconnection_{kind}.h5"
    same = load_interconnection_costs(str(path), tech, dollar_year=2023)
    later = load_interconnection_costs(str(path), tech, dollar_year=2024)
    assert same.index.is_unique and same["interconnection_cost"].notna().all()
    assert (same["interconnection_cost"] > 0).all()
    ratio = (later["interconnection_cost"] / same["interconnection_cost"]).unique()
    assert len(np.unique(ratio.round(10))) == 1 and ratio[0] > 1  # one CPI factor, 2023 -> 2024


@pytest.mark.fast
def test_interconnection_replaces_atb_grid_connection_only():
    costs = _costs()
    annuity = calculate_annuity(30, costs.at["onwind", "wacc_real"])
    interconnection = pd.Series([250_000.0, np.nan, 0.0], index=["x", "y", "z"])

    out = renewable_interconnection_capital_cost(costs, "onwind", interconnection)

    base = costs.at["onwind", "annualized_capex_fom"]
    expected_x = annuity * ((940 + 57) * 1e3 + 250_000) + 25e3
    assert out["x"] == pytest.approx(expected_x)
    assert out["x"] == pytest.approx(base + annuity * (250_000 - 100e3))
    assert out["y"] == pytest.approx(base)  # no ReEDS site: keep the ATB cost
    assert out["z"] == pytest.approx(base)


@pytest.mark.fast
def test_share_defaults_to_the_whole_capacity():
    """A positive cost without a share (a network built before the share column) is priced in full."""
    costs = _costs()
    annuity = calculate_annuity(30, costs.at["onwind", "wacc_real"])
    interconnection = pd.Series([250_000.0, 250_000.0], index=["x", "y"])

    full = renewable_interconnection_capital_cost(costs, "onwind", interconnection)
    partly = renewable_interconnection_capital_cost(
        costs,
        "onwind",
        interconnection,
        share=pd.Series({"x": np.nan, "y": 0.5}),
    )

    assert partly["x"] == pytest.approx(full["x"])
    # half the capacity keeps its ATB grid connection (100 USD/kW)
    assert partly["y"] == pytest.approx(full["y"] + annuity * 0.5 * 100e3)


@pytest.mark.fast
def test_enable_switch_needs_both_keys():
    assert renewable_interconnection_enabled({"enable": True, "renewable_interconnection": True})
    assert not renewable_interconnection_enabled({"enable": False, "renewable_interconnection": True})
    assert not renewable_interconnection_enabled({"enable": True})
    assert not renewable_interconnection_enabled(None)


@pytest.mark.fast
def test_vintages_are_repriced_with_their_own_year_costs():
    costs_2030 = _costs()
    costs_2040 = _costs()
    costs_2040.loc["onwind", "capex_grid_connection_per_kw"] = 80.0
    costs_2040.loc["onwind", "annualized_capex_fom"] *= 0.9
    gens = pd.DataFrame(
        {
            "carrier": ["onwind", "onwind", "CCGT"],
            "bus": ["a", "b", "a"],
            RENEWABLE_INTERCONNECTION_COL: [300_000.0, np.nan, np.nan],
        },
        index=["a onwind", "b onwind", "a CCGT"],
    )
    uniform = pd.Series([1.0, 2.0, 3.0], index=gens.index)

    out_2030 = apply_renewable_interconnection_costs(uniform, costs_2030, gens)
    out_2040 = apply_renewable_interconnection_costs(uniform, costs_2040, gens)

    assert out_2030["a onwind"] == pytest.approx(
        renewable_interconnection_capital_cost(costs_2030, "onwind", pd.Series([300_000.0])).iloc[0],
    )
    assert out_2040["a onwind"] != pytest.approx(out_2030["a onwind"])
    # unpriced generators keep whatever capex they were given
    assert out_2040["b onwind"] == 2.0 and out_2040["a CCGT"] == 3.0


@pytest.mark.fast
def test_column_survives_clustering_of_merged_generators(base_network):
    """Without the strategy pypsa's consense check raises on differing values."""
    from pypsa.clustering.spatial import get_clustering_from_busmap

    n = _two_solar_sites(base_network, [100_000.0, 300_000.0], p_nom=(0.0, 0.0))

    with pytest.raises(ValueError, match=RENEWABLE_INTERCONNECTION_COL):
        get_clustering_from_busmap(
            n,
            EAST_BUSMAP,
            aggregate_generators_weighted=True,
            aggregate_generators_carriers=["solar"],
        )
    clustered = _merge_east(n)
    assert clustered.generators.at["X solar", RENEWABLE_INTERCONNECTION_COL] == pytest.approx(200_000.0)
    assert clustered.generators.at["X solar", RENEWABLE_INTERCONNECTION_SHARE_COL] == pytest.approx(1.0)


@pytest.mark.fast
def test_merged_generator_keeps_the_atb_grid_connection_for_its_unpriced_share(base_network):
    """A member without a ReEDS site keeps the ATB cost, in every investment year.

    The weighted sum counts that member as zero, so pricing the merged generator
    from the cost column alone would take the ATB grid connection off all of it.
    """
    interconnection = pd.Series([300_000.0, np.nan], index=SOLAR)
    weights = pd.Series([0.25, 0.75], index=SOLAR)  # p_nom 100 and 300
    clustered = _merge_east(_two_solar_sites(base_network, interconnection))
    merged = clustered.generators.loc[["X solar"]]

    assert merged.at["X solar", RENEWABLE_INTERCONNECTION_COL] == pytest.approx(0.25 * 300_000)
    assert merged.at["X solar", RENEWABLE_INTERCONNECTION_SHARE_COL] == pytest.approx(0.25)

    costs_2030 = _costs()
    costs_2040 = _costs()
    costs_2040.loc["solar", "capex_grid_connection_per_kw"] = 80.0
    costs_2040.loc["solar", "annualized_capex_fom"] *= 0.9
    for costs in (costs_2030, costs_2040):
        base = costs.at["solar", "annualized_capex_fom"]
        members = renewable_interconnection_capital_cost(costs, "solar", interconnection)
        assert members["solar2"] == pytest.approx(base)
        repriced = apply_renewable_interconnection_costs(pd.Series(base, index=merged.index), costs, merged)
        assert repriced["X solar"] == pytest.approx((members * weights).sum())
        if costs is costs_2030:
            # the capital cost clustering itself gave the merged generator
            assert repriced["X solar"] == pytest.approx(merged.at["X solar", "capital_cost"])


@pytest.mark.fast
def test_merged_generator_without_any_site_keeps_the_atb_cost(base_network):
    clustered = _merge_east(_two_solar_sites(base_network, [np.nan, np.nan]))
    merged = clustered.generators.loc[["X solar"]]
    costs = _costs()
    uniform = pd.Series(costs.at["solar", "annualized_capex_fom"], index=merged.index)

    assert apply_renewable_interconnection_costs(uniform, costs, merged)["X solar"] == pytest.approx(uniform.iloc[0])


@pytest.mark.fast
@pytest.mark.parametrize("aggregation_strategies", [{}, {"generators": None}, None])
def test_missing_generator_strategies_fall_back_to_the_defaults(base_network, aggregation_strategies):
    """``generators:`` left empty in the config arrives as null, not as a mapping."""
    assert generator_aggregation_strategies(aggregation_strategies) == GENERATOR_CAPEX_STRATEGIES
    assert generator_aggregation_strategies({"generators": {"fuel_cost": "mean"}}) == {
        **GENERATOR_CAPEX_STRATEGIES,
        "fuel_cost": "mean",
    }

    n = _two_solar_sites(base_network, [100_000.0, 300_000.0])
    clustered = _merge_east(n, aggregation_strategies={"generators": None})
    assert clustered.generators.at["X solar", RENEWABLE_INTERCONNECTION_COL] == pytest.approx(250_000.0)


@pytest.mark.fast
def test_split_generators_keep_the_interconnection_cost_of_their_vintages(multi_period_base_network):
    """split_retirement_gens re-adds the expandable units, which price every vintage."""
    from add_extra_components import attach_multihorizon_existing_generators, split_retirement_gens

    n = multi_period_base_network
    wind = ["wind1", "wind2"]
    n.generators.loc["wind1", "p_nom"] = 100.0
    n.generators["land_region"] = n.generators.bus
    n.generators.loc[wind, RENEWABLE_INTERCONNECTION_COL] = [300_000.0, 200_000.0]
    n.generators.loc[wind, RENEWABLE_INTERCONNECTION_SHARE_COL] = [0.5, 1.0]
    costs = _costs().assign(lifetime=30.0, efficiency=1.0, opex_variable_per_mwh=0.0, heat_rate_mmbtu_per_mwh=0.0)

    split_retirement_gens(n, costs, {"onwind"}, economic=False)

    assert n.generators.at["wind1", RENEWABLE_INTERCONNECTION_COL] == 300_000.0
    assert n.generators.at["wind1", RENEWABLE_INTERCONNECTION_SHARE_COL] == 0.5
    gens = n.generators[
        n.generators.p_nom_extendable
        & (n.generators.carrier == "onwind")
        & ~n.generators.index.str.contains("existing")
    ]
    expected = renewable_interconnection_capital_cost(costs, "onwind", pd.Series([300_000.0]), share=[0.5]).iloc[0]
    for year in n.investment_periods:
        attach_multihorizon_existing_generators(n, costs, gens, year)
        assert n.generators.at[f"wind1 {year}", "capital_cost"] == pytest.approx(expected)
        assert n.generators.at[f"wind1 {year}", RENEWABLE_INTERCONNECTION_SHARE_COL] == 0.5
