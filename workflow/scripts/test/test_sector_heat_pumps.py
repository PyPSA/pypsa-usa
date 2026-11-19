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
from types import SimpleNamespace

import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from _helpers import get_multiindex_snapshots
from opts.sector import add_cooling_heat_pump_constraints, add_gshp_capacity_constraint
from solve_network import freeze_prior_periods

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

    with pytest.raises(ValueError, match="Cooling links missing"):
        add_cooling_heat_pump_constraints(n, config={})


# Several heat pumps, and heat pumps with a fixed capacity


def add_heat_pump_pair(n, node, hp_type, cop, extendable=True, cooling_extendable=None, p_nom=0.0):
    """Add ``<node> res-total-<hp_type>`` and its '-cool' twin, both with COP ``cop``."""
    cooling_extendable = extendable if cooling_extendable is None else cooling_extendable
    heating = f"{node} res-total-{hp_type}"
    cooling = f"{heating}-cool"
    for bus in (node, f"{node} res-total-heat", f"{node} res-total-cool"):
        if bus not in n.buses.index:
            n.add("Bus", bus)
    common = dict(bus0=node, carrier=f"res-total-{hp_type}", build_year=2030, lifetime=20)
    n.add(
        "Link",
        [heating],
        bus1=f"{node} res-total-heat",
        efficiency=pd.DataFrame(cop, index=n.snapshots, columns=[heating]),
        capital_cost=100.0,
        p_nom_extendable=extendable,
        p_nom=p_nom,
        **common,
    )
    n.add(
        "Link",
        [cooling],
        bus1=f"{node} res-total-cool",
        efficiency=pd.DataFrame(cop, index=n.snapshots, columns=[cooling]),
        p_nom_extendable=cooling_extendable,
        p_nom=p_nom,
        **common,
    )
    return heating, cooling


@pytest.fixture
def empty_network():
    n = pypsa.Network()
    n.snapshots = get_multiindex_snapshots(
        sns_config={"start": "2030-01-01 00:00", "end": "2030-01-01 02:00", "inclusive": "both"},
        invest_periods=[2030],
    )
    n.set_investment_periods(periods=[2030])
    # one costed component, so that the model has an objective
    n.add("Bus", "hub")
    n.add("Generator", "elec", bus="hub", p_nom=1.0, marginal_cost=1.0, build_year=2030, lifetime=20)
    return n


def link_of(model):
    """{variable label: link name} for the Link dispatch and capacity variables."""
    out = {}
    for name in ("Link-p", "Link-p_nom"):
        if name not in model.variables:
            continue
        labels = model.variables[name].labels
        for link in labels.coords["name"].values:
            for label in labels.sel(name=link).values.ravel():
                out[int(label)] = str(link)
    return out


def test_many_heat_pumps_get_one_row_per_pair(empty_network):
    """
    Each pair gets its own rows and only ever meets its own twin.

    The cooling links are added in the opposite order to the heating links, so a
    positional pairing (or a heating x cooling product) would mix the nodes up.
    """
    n = empty_network
    nodes = ["p1", "p2", "p3", "p4"]
    for i, node in enumerate(nodes):
        n.add("Bus", node)
        n.add("Bus", f"{node} res-total-heat")
        n.add("Bus", f"{node} res-total-cool")
        n.add(
            "Link",
            f"{node} res-total-ashp",
            bus0=node,
            bus1=f"{node} res-total-heat",
            efficiency=2.0 + i,
            p_nom_extendable=True,
            build_year=2030,
            lifetime=20,
        )
    for i, node in reversed(list(enumerate(nodes))):
        n.add(
            "Link",
            f"{node} res-total-ashp-cool",
            bus0=node,
            bus1=f"{node} res-total-cool",
            efficiency=2.0 + i,
            p_nom_extendable=True,
            build_year=2030,
            lifetime=20,
        )
    model = build_model(n)
    names = link_of(model)

    capacity = model.constraints["Link-ashp_cooling_capacity"]
    generation = model.constraints["Link-ashp_cooling_generation"]
    assert capacity.ncons == len(nodes)
    assert generation.ncons == len(nodes) * len(n.snapshots)

    for con in (capacity, generation):
        for row in con.vars.values.reshape(-1, con.vars.shape[-1]):
            links = {names[int(v)].removesuffix("-cool") for v in row if int(v) != -1}
            assert len(links) == 1, links


def test_brownfield_heat_pump_is_not_paired(heat_pump_network):
    """An '-ashp-existing' unit has no cooling twin and must not be asked for one."""
    n = heat_pump_network
    n.add(
        "Link",
        "p1 res-urban-ashp-existing",
        bus0="p1",
        bus1="p1 res-urban-heat",
        carrier="res-urban-ashp",
        efficiency=2.0,
        p_nom=5.0,
        build_year=2030,
        lifetime=20,
    )
    model = build_model(n)

    assert model.constraints["Link-ashp_cooling_capacity"].ncons == 1


def test_cooling_link_without_heating_twin_is_rejected(heat_pump_network):
    n = heat_pump_network
    n.remove("Link", HEATING_HP)
    n.optimize.create_model(multi_investment_periods=True)

    with pytest.raises(ValueError, match="without a heating heat pump"):
        add_cooling_heat_pump_constraints(n, config={})


def test_fixed_capacity_heat_pumps_share_the_fixed_capacity(empty_network):
    """
    Non-extendable pair (``heat_pump: false``, or frozen by a myopic horizon).

    There is no capacity variable, so no capacity equality; the dispatch limit uses
    the fixed capacity: cop * p_heat + cop * p_cool <= cop * p_nom.
    """
    n = empty_network
    heating, cooling = add_heat_pump_pair(n, "p1", "ashp", cop=3.0, extendable=False, p_nom=7.0)
    model = build_model(n)

    assert "Link-ashp_cooling_capacity" not in model.constraints

    con = model.constraints["Link-ashp_cooling_generation"]
    for snapshot in range(len(n.snapshots)):
        assert terms(con, snapshot=snapshot, name=0) == {
            var_label(model, "Link-p", heating, snapshot=snapshot): 3.0,
            var_label(model, "Link-p", cooling, snapshot=snapshot): 3.0,
        }
        assert float(con.rhs.isel(snapshot=snapshot, name=0).item()) == pytest.approx(21.0)


def test_fixed_and_extendable_pairs_mix(empty_network):
    """A frozen pair and a new extendable pair are constrained side by side."""
    n = empty_network
    fixed_heat, _ = add_heat_pump_pair(n, "p1", "ashp", cop=3.0, extendable=False, p_nom=7.0)
    ext_heat, ext_cool = add_heat_pump_pair(n, "p2", "ashp", cop=4.0, extendable=True)
    model = build_model(n)

    capacity = model.constraints["Link-ashp_cooling_capacity"]
    assert capacity.ncons == 1
    assert terms(capacity, name=0) == {
        var_label(model, "Link-p_nom", ext_heat): 1.0,
        var_label(model, "Link-p_nom", ext_cool): -1.0,
    }

    generation = model.constraints["Link-ashp_cooling_generation"]
    rhs = generation.rhs.isel(snapshot=0)
    assert float(rhs.sel(name=fixed_heat).item()) == pytest.approx(21.0)
    assert float(rhs.sel(name=ext_heat).item()) == 0.0
    ext_row = dict(
        zip(
            generation.vars.isel(snapshot=0).sel(name=ext_heat).values.tolist(),
            generation.coeffs.isel(snapshot=0).sel(name=ext_heat).values.tolist(),
        ),
    )
    assert ext_row[var_label(model, "Link-p_nom", ext_heat)] == -4.0


def test_fixed_heating_with_extendable_cooling_pins_the_cooling_capacity(empty_network):
    """The extendable twin is forced onto the fixed capacity of the other link."""
    n = empty_network
    _, cooling = add_heat_pump_pair(n, "p1", "ashp", cop=3.0, extendable=False, cooling_extendable=True, p_nom=7.0)
    model = build_model(n)

    con = model.constraints["Link-ashp_cooling_capacity"]
    assert terms(con, name=0) == {var_label(model, "Link-p_nom", cooling): -1.0}
    assert con.sign.item() == "="
    assert float(con.rhs.item()) == pytest.approx(-7.0)


def test_unequal_fixed_capacities_are_rejected(empty_network):
    n = empty_network
    _, cooling = add_heat_pump_pair(n, "p1", "ashp", cop=3.0, extendable=False, p_nom=7.0)
    n.links.loc[cooling, "p_nom"] = 9.0
    n.optimize.create_model(multi_investment_periods=True)

    with pytest.raises(ValueError, match="capacities differ"):
        add_cooling_heat_pump_constraints(n, config={})


# GSHP to ASHP capacity ratio


def gshp_snakemake(tmp_path, nodes, urban_fraction=0.75):
    pop = pd.DataFrame({"name": nodes, "urban_fraction": urban_fraction, "rural_fraction": 1 - urban_fraction})
    path = tmp_path / "pop_layout.csv"
    pop.to_csv(path, index=False)
    return SimpleNamespace(input=SimpleNamespace(pop_layout=str(path)))


def test_gshp_ratio_pairs_by_node_and_skips_fixed_pairs(empty_network, tmp_path):
    """ASHP - (urban / rural) * GSHP >= 0 for the extendable pair only."""
    n = empty_network
    add_heat_pump_pair(n, "p1", "ashp", cop=3.0, extendable=False, p_nom=9.0)
    add_heat_pump_pair(n, "p1", "gshp", cop=4.0, extendable=False, p_nom=3.0)
    ashp, _ = add_heat_pump_pair(n, "p2", "ashp", cop=3.0)
    gshp, _ = add_heat_pump_pair(n, "p2", "gshp", cop=4.0)
    n.optimize.create_model(multi_investment_periods=True)

    add_gshp_capacity_constraint(n, {}, gshp_snakemake(tmp_path, ["p1", "p2"]))

    con = n.model.constraints["Link-gshp_capacity_ratio"]
    assert con.ncons == 1
    assert terms(con, name=0) == {
        var_label(n.model, "Link-p_nom", ashp): 1.0,
        var_label(n.model, "Link-p_nom", gshp): -3.0,
    }
    assert con.sign.item() == ">="


def test_gshp_ratio_is_skipped_when_every_heat_pump_is_fixed(empty_network, tmp_path):
    n = empty_network
    add_heat_pump_pair(n, "p1", "ashp", cop=3.0, extendable=False, p_nom=9.0)
    add_heat_pump_pair(n, "p1", "gshp", cop=4.0, extendable=False, p_nom=3.0)
    n.optimize.create_model(multi_investment_periods=True)

    add_gshp_capacity_constraint(n, {}, gshp_snakemake(tmp_path, ["p1"]))

    assert "Link-gshp_capacity_ratio" not in n.model.constraints


# Myopic foresight


def test_second_myopic_horizon_solves_with_frozen_heat_pumps(tmp_path):
    """
    Regression: the heat pumps built in the first horizon are frozen for the second.

    ``freeze_prior_periods`` makes them non-extendable, so they no longer have a
    ``Link-p_nom`` variable. The second horizon must still build its model, and the
    frozen unit must keep sharing its capacity between heating and cooling.
    """
    n = pypsa.Network()
    n.snapshots = get_multiindex_snapshots(
        sns_config={"start": "2030-01-01 00:00", "end": "2030-01-01 02:00", "inclusive": "both"},
        invest_periods=[2030, 2040],
    )
    n.set_investment_periods(periods=[2030, 2040])
    n.config = {"electricity": {}}

    heating, cooling = add_heat_pump_pair(n, "p1", "ashp", cop=3.0)
    add_heat_pump_pair(n, "p1", "gshp", cop=3.0)
    n.add("Generator", "elec", bus="p1", p_nom=1e4, marginal_cost=0.01, build_year=2030, lifetime=40)
    # expensive backstops, so the second horizon stays feasible if the frozen heat pump is short
    n.add("Generator", "heat backstop", bus="p1 res-total-heat", p_nom=1e4, marginal_cost=1e3, build_year=2030)
    n.add("Generator", "cool backstop", bus="p1 res-total-cool", p_nom=1e4, marginal_cost=1e3, build_year=2030)
    n.add("Load", "heat load", bus="p1 res-total-heat", p_set=30.0)
    n.add("Load", "cool load", bus="p1 res-total-cool", p_set=30.0)
    n.generators.loc[["heat backstop", "cool backstop"], "lifetime"] = 40
    snakemake = gshp_snakemake(tmp_path, ["p1"])

    def extra_functionality(n, sns):
        add_cooling_heat_pump_constraints(n, config={})
        add_gshp_capacity_constraint(n, {}, snakemake)

    for period in n.investment_periods:
        status, condition = n.optimize(
            snapshots=n.snapshots[n.snapshots.get_level_values(0) == period],
            multi_investment_periods=True,
            solver_name="highs",
            extra_functionality=extra_functionality,
        )
        assert (status, condition) == ("ok", "optimal")
        freeze_prior_periods(n, period)

    assert not n.links.loc[[heating, cooling], "p_nom_extendable"].any()
    sns_2040 = n.snapshots[n.snapshots.get_level_values(0) == 2040]
    shared = n.links_t.p0.loc[sns_2040, heating] + n.links_t.p0.loc[sns_2040, cooling]
    assert (shared <= n.links.at[heating, "p_nom"] + 1e-4).all()
