"""ReEDS wind/solar interconnection cost.

The interconnection tables shipped in repo_data are read as published.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from _helpers import calculate_annuity, get_multiindex_snapshots
from add_extra_components import apply_renewable_interconnection_costs
from constants import GENERATOR_CAPEX_STRATEGIES, RENEWABLE_INTERCONNECTION_COL
from reeds_vre import bus_weighted_mean, load_interconnection_costs
from regional_cost import (
    renewable_interconnection_capital_cost,
    renewable_interconnection_enabled,
)

REPO_COSTS = Path(__file__).resolve().parents[2] / "repo_data" / "costs"


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
def test_column_survives_clustering_of_merged_generators():
    """Without the strategy pypsa's consense check raises on differing values."""
    from pypsa.clustering.spatial import get_clustering_from_busmap

    n = pypsa.Network()
    n.add("Bus", ["a", "b"], x=[0, 1], y=[0, 0])
    n.add("Carrier", ["solar"])
    n.add("Generator", ["a solar", "b solar"], bus=["a", "b"], carrier="solar", capital_cost=[10.0, 20.0])
    n.generators[RENEWABLE_INTERCONNECTION_COL] = [100_000.0, 300_000.0]
    busmap = pd.Series({"a": "X", "b": "X"})

    with pytest.raises(ValueError, match=RENEWABLE_INTERCONNECTION_COL):
        get_clustering_from_busmap(
            n, busmap, aggregate_generators_weighted=True, aggregate_generators_carriers=["solar"]
        )
    clustered = get_clustering_from_busmap(
        n,
        busmap,
        aggregate_generators_weighted=True,
        aggregate_generators_carriers=["solar"],
        generator_strategies=dict(GENERATOR_CAPEX_STRATEGIES),
    ).n
    assert clustered.generators.at["X solar", RENEWABLE_INTERCONNECTION_COL] == pytest.approx(200_000.0)


@pytest.mark.fast
def test_split_generators_keep_the_interconnection_cost_of_their_vintages():
    """split_retirement_gens re-adds the expandable units, which price every vintage."""
    from add_extra_components import attach_multihorizon_existing_generators, split_retirement_gens

    n = pypsa.Network()
    n.snapshots = get_multiindex_snapshots(
        sns_config={"start": "2030-01-01 00:00", "end": "2030-01-01 01:00", "inclusive": "both"},
        invest_periods=[2030, 2040],
    )
    n.set_investment_periods(periods=[2030, 2040])
    n.add("Bus", ["a", "b"])
    n.add("Carrier", ["onwind"])
    n.add(
        "Generator",
        ["a onwind", "b onwind"],
        bus=["a", "b"],
        carrier="onwind",
        p_nom=[100.0, 0.0],
        p_nom_extendable=True,
        build_year=2030,
        capital_cost=1.0,
    )
    n.generators["land_region"] = ["a", "b"]
    n.generators[RENEWABLE_INTERCONNECTION_COL] = [300_000.0, 200_000.0]
    costs = _costs().assign(lifetime=30.0, efficiency=1.0, opex_variable_per_mwh=0.0, heat_rate_mmbtu_per_mwh=0.0)

    split_retirement_gens(n, costs, {"onwind"}, economic=False)

    assert n.generators.at["a onwind", RENEWABLE_INTERCONNECTION_COL] == 300_000.0
    gens = n.generators[n.generators.p_nom_extendable & ~n.generators.index.str.contains("existing")]
    for year in n.investment_periods:
        attach_multihorizon_existing_generators(n, costs, gens, year)
        priced = n.generators.at[f"a onwind {year}", "capital_cost"]
        assert priced == pytest.approx(
            renewable_interconnection_capital_cost(costs, "onwind", pd.Series([300_000.0])).iloc[0],
        )
