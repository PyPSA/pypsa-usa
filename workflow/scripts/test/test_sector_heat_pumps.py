"""
Test the cooling heat pump constraints.

A cooling heat pump is the same physical unit as its heating twin, so the two
must share one capacity: ``build_heat.add_service_heat_pumps_cooling`` names the
cooling link after the heating link with a ``-cool`` suffix, and
``add_cooling_heat_pump_constraints`` must (a) force the two nominal capacities
to be equal and (b) keep heating output + cooling output within the heating
capacity in every snapshot.
"""

import os
import sys

import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from _helpers import get_multiindex_snapshots
from opts.sector import add_cooling_heat_pump_constraints

HEAT_COP = 3.0
COOL_COP = 2.5

HEATING_HP = "p1 res-urban-ashp"
COOLING_HP = "p1 res-urban-ashp-cool"

# Fixtures


@pytest.fixture
def heat_pump_network():
    """One node with an extendable ASHP, its '-cool' twin and an air conditioner."""
    n = pypsa.Network()
    n.snapshots = get_multiindex_snapshots(
        sns_config={"start": "2030-01-01 00:00", "end": "2030-01-01 02:00", "inclusive": "both"},
        invest_periods=[2030],
    )
    n.set_investment_periods(periods=[2030])

    n.add("Bus", "p1", carrier="AC")
    n.add("Bus", "p1 res-urban-heat", carrier="res-urban-heat")
    n.add("Bus", "p1 res-urban-cool", carrier="res-urban-cool")

    # time varying COPs, as build_heat writes them
    heat_cop = pd.DataFrame(HEAT_COP, index=n.snapshots, columns=[HEATING_HP])
    cool_cop = pd.DataFrame(COOL_COP, index=n.snapshots, columns=[COOLING_HP])

    n.add(
        "Link",
        [HEATING_HP],
        bus0="p1",
        bus1="p1 res-urban-heat",
        carrier="res-urban-ashp",
        efficiency=heat_cop,
        capital_cost=100.0,
        p_nom_extendable=True,
        build_year=2030,
        lifetime=20,
    )
    n.add(
        "Link",
        [COOLING_HP],
        bus0="p1",
        bus1="p1 res-urban-cool",
        carrier="res-urban-ashp",
        efficiency=cool_cop,
        capital_cost=0.0,
        p_nom_extendable=True,
        build_year=2030,
        lifetime=20,
    )
    # an air conditioner also serves the cooling bus, but is not a heat pump
    n.add(
        "Link",
        ["p1 res-urban-air-con"],
        bus0="p1",
        bus1="p1 res-urban-cool",
        carrier="res-urban-air-con",
        efficiency=3.5,
        capital_cost=50.0,
        p_nom_extendable=True,
        build_year=2030,
        lifetime=20,
    )
    return n


# Helpers


def build_model(n):
    n.optimize.create_model(multi_investment_periods=True)
    add_cooling_heat_pump_constraints(n, config={})
    return n.model


def terms(constraint, **sel):
    """{variable label: coefficient} of the single constraint row picked by ``sel``."""
    row_vars = constraint.vars.isel(**sel).values.ravel()
    row_coeffs = constraint.coeffs.isel(**sel).values.ravel()
    return {int(v): float(c) for v, c in zip(row_vars, row_coeffs) if int(v) != -1}


def var_label(model, name, link, **sel):
    return int(model.variables[name].labels.sel(name=link).isel(**sel).item())


# Tests


def test_capacity_equality_constraint_pairs_the_twins(heat_pump_network):
    """p_nom of the cooling link is forced to equal p_nom of its heating link."""
    n = heat_pump_network
    model = build_model(n)

    assert "Link-ashp_cooling_capacity" in model.constraints

    con = model.constraints["Link-ashp_cooling_capacity"]
    coeffs = terms(con, name=0)

    assert coeffs == {
        var_label(model, "Link-p_nom", HEATING_HP): 1.0,
        var_label(model, "Link-p_nom", COOLING_HP): -1.0,
    }
    assert con.sign.item() == "="
    assert float(con.rhs.item()) == 0.0


def test_generation_constraint_includes_the_cooling_link(heat_pump_network):
    """
    Heating and cooling dispatch share the heating capacity.

    cop_heat * p_heat + cop_cool * p_cool - cop_heat * p_nom_heat <= 0
    """
    n = heat_pump_network
    model = build_model(n)

    assert "Link-ashp_cooling_generation" in model.constraints

    con = model.constraints["Link-ashp_cooling_generation"]

    for snapshot in range(len(n.snapshots)):
        coeffs = terms(con, snapshot=snapshot, name=0)
        assert coeffs == {
            var_label(model, "Link-p", HEATING_HP, snapshot=snapshot): HEAT_COP,
            var_label(model, "Link-p", COOLING_HP, snapshot=snapshot): COOL_COP,
            var_label(model, "Link-p_nom", HEATING_HP): -HEAT_COP,
        }, f"snapshot {snapshot}"
        assert con.sign.isel(snapshot=snapshot, name=0).item() == "<="
        assert float(con.rhs.isel(snapshot=snapshot, name=0).item()) == 0.0


def test_air_conditioner_is_not_constrained(heat_pump_network):
    """Only heat pumps are paired; the air conditioner keeps its own capacity."""
    n = heat_pump_network
    model = build_model(n)

    ac_p_nom = var_label(model, "Link-p_nom", "p1 res-urban-air-con")
    constrained = set()
    for name in ("Link-ashp_cooling_capacity", "Link-ashp_cooling_generation"):
        constrained |= {int(v) for v in model.constraints[name].vars.values.ravel()}

    assert ac_p_nom not in constrained


def test_no_gshp_means_no_gshp_constraints(heat_pump_network):
    """A network without ground source heat pumps gets no gshp constraints."""
    n = heat_pump_network
    model = build_model(n)

    assert "Link-gshp_cooling_capacity" not in model.constraints
    assert "Link-gshp_cooling_generation" not in model.constraints


def test_simultaneous_heating_and_cooling_needs_the_summed_capacity(heat_pump_network):
    """
    Solved behaviour: one unit cannot heat and cool at full output at once.

    ``build_heat`` gives the cooling twin the heating COP, and with equal COPs the
    constraint reduces to p_heat + p_cool <= p_nom. The heat load then needs
    30 / 3 = 10 MW of heat pump and the cool load another 10 MW in the same
    snapshot, so the shared unit must be sized at 20 MW.
    """
    n = heat_pump_network
    n.links_t.efficiency[COOLING_HP] = HEAT_COP
    n.add("Generator", "elec", bus="p1", p_nom=1e4, marginal_cost=0.01, build_year=2030, lifetime=20)
    n.add("Load", "heat load", bus="p1 res-urban-heat", p_set=[30.0, 0.0, 0.0])
    n.add("Load", "cool load", bus="p1 res-urban-cool", p_set=[30.0, 0.0, 0.0])
    # price the air conditioner out of the solution
    n.links.loc["p1 res-urban-air-con", "capital_cost"] = 1e6

    status, condition = n.optimize(
        multi_investment_periods=True,
        solver_name="highs",
        extra_functionality=lambda n, sns: add_cooling_heat_pump_constraints(n, config={}),
    )
    assert (status, condition) == ("ok", "optimal")

    assert n.links.at[HEATING_HP, "p_nom_opt"] == pytest.approx(20.0, abs=1e-4)
    assert n.links.at[COOLING_HP, "p_nom_opt"] == pytest.approx(20.0, abs=1e-4)
    assert n.links_t.p0.at[n.snapshots[0], HEATING_HP] == pytest.approx(10.0, abs=1e-4)
    assert n.links_t.p0.at[n.snapshots[0], COOLING_HP] == pytest.approx(10.0, abs=1e-4)
    assert n.links.at["p1 res-urban-air-con", "p_nom_opt"] == pytest.approx(0.0, abs=1e-4)


def test_missing_cooling_twin_is_rejected(heat_pump_network):
    """A heating heat pump without its '-cool' twin is a build error, not a silent pass."""
    n = heat_pump_network
    n.remove("Link", COOLING_HP)
    n.optimize.create_model(multi_investment_periods=True)

    with pytest.raises(AssertionError):
        add_cooling_heat_pump_constraints(n, config={})
