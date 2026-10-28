"""
Test the sector co2 limit constraints.

``add_sector_co2_constraints`` reads the co2 policy csv and caps, for each year,
the level of the matching ``-co2`` / ``-ch4`` emission stores at the last snapshot
of that investment period, per state or nationally. The stores start every period
empty, so that level is the emissions of the period alone.
Under PyPSA 1.x the component dimension of ``Store-e`` is ``name``; summing over
the old ``Store`` dimension raised ``KeyError: 'Store'`` in ``solve_network``.
"""

import os
import sys

import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from _helpers import get_multiindex_snapshots
from build_emission_tracking import build_co2_tracking
from opts.sector import add_sector_co2_constraints

STORES = ["UT res-co2", "UT res-ch4", "UT ind-co2", "NV res-co2"]
PERIODS = [2030, 2040]
LOAD = 10  # MW, met by a gas plant emitting 0.4 t/MWh_el
ANNUAL = LOAD * 0.4 * 8760  # t CO2

# Fixtures


@pytest.fixture
def emission_network():
    """Emission stores for two states and two sectors, as add_sectors names them."""
    n = pypsa.Network()
    n.snapshots = get_multiindex_snapshots(
        sns_config={"start": "2030-01-01 00:00", "end": "2030-01-01 02:00", "inclusive": "both"},
        invest_periods=[2030],
    )
    n.set_investment_periods(periods=[2030])

    n.add("Bus", "co2", carrier="co2")
    # a cost is needed only so create_model can build an objective
    n.add("Store", STORES, bus="co2", carrier="co2", e_nom=1e9, marginal_cost_storage=1.0)
    return n


@pytest.fixture
def two_period_network():
    """A gas plant emitting into the tracking stores of ``build_co2_tracking``, over two one year periods."""
    n = pypsa.Network()
    n.snapshots = get_multiindex_snapshots(
        sns_config={"start": "2030-01-01 00:00", "end": "2030-01-01 01:00", "inclusive": "both"},
        invest_periods=PERIODS,
    )
    n.set_investment_periods(periods=PERIODS)
    for col in n.snapshot_weightings.columns:
        n.snapshot_weightings[col] = 8760 / 2

    n.add("Bus", "UT", carrier="AC", STATE="UT")
    n.add("Load", "UT", bus="UT", p_set=LOAD)
    n.add("Generator", "UT clean", bus="UT", carrier="clean", p_nom=100, marginal_cost=100)
    n.add("Bus", "UT gas", carrier="gas", STATE="UT")
    n.add("Generator", "UT gas supply", bus="UT gas", carrier="gas", p_nom=1e4, marginal_cost=10)
    build_co2_tracking(n)
    n.add(
        "Link",
        "UT CCGT",
        bus0="UT gas",
        bus1="UT",
        bus2="UT pwr-co2",
        efficiency=0.5,
        efficiency2=0.2,  # t/MWh_th
        p_nom=100,
        carrier="CCGT",
    )
    return n


# Helpers


def build_model(n, tmp_path, policy):
    """Create the model and apply the co2 limits in ``policy`` (rows of the csv)."""
    fn = tmp_path / "sector_co2_limits.csv"
    pd.DataFrame(policy, columns=["sector", "state", "year", "co2_limit_mmt"]).to_csv(fn, index=False)
    n.optimize.create_model(multi_investment_periods=True)
    add_sector_co2_constraints(n, config={"sector": {"co2": {"policy": str(fn)}}})
    return n.model


def store_terms(model, constraint):
    """{store name: coefficient} of a single-row constraint on ``Store-e``."""
    labels = model.constraints[constraint].vars.values.ravel()
    coeffs = model.constraints[constraint].coeffs.values.ravel()
    terms = {}
    for label, coeff in zip(labels, coeffs):
        if label == -1:
            continue
        var, coords = model.variables.get_label_position(int(label))
        assert var == "Store-e"
        terms[coords["name"]] = float(coeff)
    return terms


def last_snapshot_label(model, n):
    return set(model.variables["Store-e"].labels.sel(snapshot=n.snapshots[-1]).values.ravel())


# Tests


def test_state_limit_sums_that_states_stores(emission_network, tmp_path):
    n = emission_network
    model = build_model(n, tmp_path, [["all", "UT", 2030, 5.0]])

    name = "GlobalConstraint-co2_limit-2030-UT"
    assert name in model.constraints
    assert store_terms(model, name) == {"UT res-co2": 1.0, "UT res-ch4": 1.0, "UT ind-co2": 1.0}
    assert set(model.constraints[name].vars.values.ravel()) <= last_snapshot_label(model, n)
    assert model.constraints[name].sign.item() == "<="
    assert float(model.constraints[name].rhs.item()) == 5.0e6


def test_state_sector_limit_only_takes_that_sector(emission_network, tmp_path):
    model = build_model(emission_network, tmp_path, [["res", "UT", 2030, 2.0]])

    name = "GlobalConstraint-co2_limit-2030-UT-res"
    assert store_terms(model, name) == {"UT res-co2": 1.0, "UT res-ch4": 1.0}
    assert float(model.constraints[name].rhs.item()) == 2.0e6


def test_national_limit_sums_every_emission_store(emission_network, tmp_path):
    model = build_model(emission_network, tmp_path, [["all", "all", 2030, 9.0]])

    name = "co2_limit-2030"
    assert store_terms(model, name) == dict.fromkeys(STORES, 1.0)
    assert float(model.constraints[name].rhs.item()) == 9.0e6


def test_each_year_caps_the_last_snapshot_of_its_period(two_period_network, tmp_path):
    n = two_period_network
    model = build_model(n, tmp_path, [["all", "UT", 2030, 1.0], ["all", "UT", 2040, 1.0]])

    for year in PERIODS:
        last = n.snapshots[n.snapshots.get_level_values("period") == year][-1]
        labels = set(model.variables["Store-e"].labels.sel(snapshot=last).values.ravel())
        assert set(model.constraints[f"GlobalConstraint-co2_limit-{year}-UT"].vars.values.ravel()) <= labels


def test_limits_cap_the_emissions_of_each_period(two_period_network, tmp_path):
    """A binding 2040 limit leaves 2030 alone, rather than capping the emissions of both periods."""
    n = two_period_network
    build_model(n, tmp_path, [["all", "UT", 2030, ANNUAL / 1e6], ["all", "UT", 2040, ANNUAL / 2 / 1e6]])
    n.optimize.solve_model(solver_name="glpk")

    emitted = n.links_t.p2["UT CCGT"].mul(-1).mul(n.snapshot_weightings.stores).groupby(level="period").sum()
    assert emitted.tolist() == pytest.approx([ANNUAL, ANNUAL / 2])
    # the tracking store starts every period empty
    assert n.stores_t.e["UT pwr-co2"].groupby(level="period").last().tolist() == pytest.approx(emitted.tolist())
