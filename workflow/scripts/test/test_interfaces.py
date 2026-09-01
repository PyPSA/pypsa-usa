"""
Test the aggregate transmission interface limits and the external-region build.

This module contains tests for the RESOLVE/NARIS style interface constraints
applied to the electricity import/export links in PyPSA-USA, and for the two
representations of out-of-footprint supply built by ``external_regions``.
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from _helpers import get_multiindex_snapshots
from external_regions import (
    GENERIC_IMPORT_CARRIER,
    add_external_regions,
    inbound_capacity_by_zone,
    map_remote_units_to_zones,
)
from opts.interfaces import (
    _boundary_links,
    _parse_regions,
    add_interface_transmission_limits,
)

TOL = 1e-4

SHIPPED_LIMITS = os.path.join(
    os.path.dirname(__file__),
    "../../repo_data/config/policy_constraints/transmission_interface_limits.csv",
)

# Fixtures


@pytest.fixture
def interface_network():
    """
    Build a small network with electricity import and export links.

    Mirrors the conventions of ``external_regions.add_elec_imports_exports``:
    external buses are named ``{zone}_imports`` / ``{zone}_exports`` with the
    matching carrier, import links run from the external bus into the model and
    export links run the other way.
    """
    n = pypsa.Network()

    n.snapshots = get_multiindex_snapshots(
        sns_config={"start": "2030-01-01 00:00", "end": "2030-01-01 03:00", "inclusive": "both"},
        invest_periods=[2030],
    )
    n.set_investment_periods(periods=[2030])

    for carrier in ("AC", "gas", "imports", "exports"):
        n.add("Carrier", carrier, co2_emissions=0)

    # Buses inside the model
    n.add(
        "Bus",
        ["CA_Z1", "TX_Z1"],
        carrier="AC",
        country="US",
        interconnect="western",
        nerc_reg=["WECC", "WECC"],
        reeds_state=["CA", "TX"],
        reeds_zone=["CA_Z1", "TX_Z1"],
    )

    # External trade buses. `_add_import_export_buses` stamps the outside zone
    # name onto `country`, which is what makes the carrier filter necessary.
    n.add(
        "Bus",
        ["p2_imports", "p5_imports"],
        carrier="imports",
        country=["p2", "p5"],
        interconnect="western",
    )
    n.add(
        "Bus",
        "p2_exports",
        carrier="exports",
        country="p2",
        interconnect="western",
    )

    # Trade links
    n.add(
        "Link",
        "CA_Z1_p2_imports",
        bus0="p2_imports",
        bus1="CA_Z1",
        carrier="imports",
        p_nom=500,
        marginal_cost=0,
    )
    n.add(
        "Link",
        "CA_Z1_p5_imports",
        bus0="p5_imports",
        bus1="CA_Z1",
        carrier="imports",
        p_nom=500,
        marginal_cost=0,
    )
    # Decoy: an import link landing outside region_1
    n.add(
        "Link",
        "TX_Z1_p2_imports",
        bus0="p2_imports",
        bus1="TX_Z1",
        carrier="imports",
        p_nom=500,
        marginal_cost=0,
    )
    n.add(
        "Link",
        "CA_Z1_p2_exports",
        bus0="CA_Z1",
        bus1="p2_exports",
        carrier="exports",
        p_nom=500,
        marginal_cost=0,
    )
    # Decoy: an internal AC link between two in-model buses
    n.add(
        "Link",
        "CA_Z1_TX_Z1",
        bus0="CA_Z1",
        bus1="TX_Z1",
        carrier="AC",
        p_nom=500,
    )

    # Generation and demand: imports are cheap, local gas is not
    n.add("Generator", "import_p2", bus="p2_imports", carrier="imports", p_nom=1000, marginal_cost=1)
    n.add("Generator", "import_p5", bus="p5_imports", carrier="imports", p_nom=1000, marginal_cost=1)
    n.add("Generator", "gas_ca", bus="CA_Z1", carrier="gas", p_nom=1000, marginal_cost=50)
    n.add("Generator", "gas_tx", bus="TX_Z1", carrier="gas", p_nom=1000, marginal_cost=50)
    # Expensive backup behind the export bus, so an export cap stays feasible
    n.add("Generator", "gas_p2", bus="p2_exports", carrier="gas", p_nom=1000, marginal_cost=100)

    n.add("Load", "load_ca", bus="CA_Z1", carrier="AC", p_set=pd.Series(300.0, index=n.snapshots))
    n.add("Load", "load_tx", bus="TX_Z1", carrier="AC", p_set=pd.Series(200.0, index=n.snapshots))
    n.add("Load", "load_p2", bus="p2_exports", carrier="exports", p_set=pd.Series(400.0, index=n.snapshots))

    return n


def write_limits(tmp_path, rows):
    """Write an interface limits CSV and return its path."""
    path = tmp_path / "transmission_interface_limits.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


# Tests


def test_parse_regions_handles_whitespace():
    assert _parse_regions("p9, p10,p11 ") == ["p9", "p10", "p11"]
    assert _parse_regions("p9") == ["p9"]
    assert _parse_regions("") == []


def test_boundary_links_selects_only_crossing_links(interface_network):
    n = interface_network

    imports = _boundary_links(n, ["CA_Z1"], ["p2", "p5"], "imports")
    assert sorted(imports) == ["CA_Z1_p2_imports", "CA_Z1_p5_imports"]

    exports = _boundary_links(n, ["CA_Z1"], ["p2", "p5"], "exports")
    assert sorted(exports) == ["CA_Z1_p2_exports"]

    # Only the named outside zones count
    assert sorted(_boundary_links(n, ["CA_Z1"], ["p5"], "imports")) == ["CA_Z1_p5_imports"]

    # Region_1 can be given as any of the labels get_region_buses matches on
    assert sorted(_boundary_links(n, ["CA"], ["p2", "p5"], "imports")) == [
        "CA_Z1_p2_imports",
        "CA_Z1_p5_imports",
    ]

    with pytest.raises(ValueError):
        _boundary_links(n, ["CA_Z1"], ["p2"], "both")


def test_no_matching_links_is_a_noop_not_an_error(interface_network, tmp_path):
    n = interface_network
    limits = write_limits(
        tmp_path,
        [
            # Neither region exists in this network
            {"interface": "NW_SW", "region_1": "p30", "region_2": "p33", "flow_12": 100, "flow_21": 100},
        ],
    )

    def extra_functionality(n, sns):
        add_interface_transmission_limits(n, limits)
        # the shipped RESOLVE table names ReEDS zones absent from this network
        add_interface_transmission_limits(n, SHIPPED_LIMITS)

    n.optimize(solver_name="glpk", multi_investment_periods=True, extra_functionality=extra_functionality)

    assert not [c for c in n.model.constraints if c.startswith("interface_limit-")]


def test_import_cap_binds(interface_network, tmp_path):
    n = interface_network
    cap = 100.0
    limits = write_limits(
        tmp_path,
        [
            {
                "interface": "CAISO_Imports",
                "region_1": "CA_Z1",
                "region_2": "p2, p5",
                "flow_12": 1e6,
                "flow_21": cap,
            },
        ],
    )

    def extra_functionality(n, sns):
        add_interface_transmission_limits(n, limits)

    n.optimize(solver_name="glpk", multi_investment_periods=True, extra_functionality=extra_functionality)

    assert "interface_limit-CAISO_Imports-imports" in n.model.constraints

    flow = n.links_t.p0[["CA_Z1_p2_imports", "CA_Z1_p5_imports"]].sum(axis=1)
    assert (flow <= cap + TOL).all()
    assert flow.max() >= cap - TOL, "import cap should bind in at least one snapshot"

    # The decoy import link into TX is outside the interface and stays free
    assert n.links_t.p0["TX_Z1_p2_imports"].max() > cap + TOL


def test_export_cap_uses_flow_12(interface_network, tmp_path):
    n = interface_network
    cap = 150.0
    limits = write_limits(
        tmp_path,
        [
            {
                "interface": "CAISO_Exports",
                "region_1": "CA_Z1",
                "region_2": "p2, p5",
                "flow_12": cap,
                "flow_21": 1e6,
            },
        ],
    )

    def extra_functionality(n, sns):
        add_interface_transmission_limits(n, limits)

    n.optimize(solver_name="glpk", multi_investment_periods=True, extra_functionality=extra_functionality)

    assert "interface_limit-CAISO_Exports-exports" in n.model.constraints

    flow = n.links_t.p0["CA_Z1_p2_exports"]
    assert (flow <= cap + TOL).all()
    assert flow.max() >= cap - TOL, "export cap should bind in at least one snapshot"


def test_disabled_flag_adds_no_constraints(interface_network, tmp_path):
    n = interface_network
    limits = write_limits(
        tmp_path,
        [
            {
                "interface": "CAISO_Imports",
                "region_1": "CA_Z1",
                "region_2": "p2, p5",
                "flow_12": 100,
                "flow_21": 100,
            },
        ],
    )
    config = {"model_topology": {"interface_transmission_limits": False}}

    def extra_functionality(n, sns):
        # mirrors the gate in solve_network.extra_functionality
        if config["model_topology"].get("interface_transmission_limits", False):
            add_interface_transmission_limits(n, limits)

    n.optimize(solver_name="glpk", multi_investment_periods=True, extra_functionality=extra_functionality)

    assert not [c for c in n.model.constraints if c.startswith("interface_limit-")]


# ---------------------------------------------------------------------------
# external_regions: the two import representations
# ---------------------------------------------------------------------------

MEMBERSHIP = pd.DataFrame(
    [
        {"ba": "p9", "st": "CA"},
        {"ba": "p12", "st": "NV"},
        {"ba": "p13", "st": "NV"},
        {"ba": "p28", "st": "AZ"},
    ],
)

# Mirrors the CA boundary: two NV zones (p13 much larger than p12), one AZ zone,
# and one outbound-only interface so the inbound/outbound split is exercised.
FLOWGATES = pd.DataFrame(
    [
        {"r": "p13", "rr": "p9", "value": 2000.0},
        {"r": "p12", "rr": "p9", "value": 500.0},
        {"r": "p28", "rr": "p9", "value": 1000.0},
        {"r": "p9", "rr": "p13", "value": 1500.0},
    ],
)


@pytest.fixture
def footprint_network():
    """A one-zone footprint (California's p9) with nothing external attached yet."""
    n = pypsa.Network()
    n.snapshots = get_multiindex_snapshots(
        sns_config={"start": "2030-01-01 00:00", "end": "2030-01-01 03:00", "inclusive": "both"},
        invest_periods=[2030],
    )
    n.set_investment_periods(periods=[2030])

    n.add("Carrier", "AC", co2_emissions=0)
    n.add("Carrier", "solar", co2_emissions=0)
    n.add(
        "Bus",
        "p9",
        carrier="AC",
        country="p9",
        interconnect="western",
        reeds_state="CA",
        reeds_zone="p9",
    )
    n.add(
        "Generator",
        "p9 solar",
        bus="p9",
        carrier="solar",
        p_nom=100,
        p_max_pu=pd.Series(0.5, index=n.snapshots),
    )
    return n


def wholesale_costs():
    """A `format_import_export_costs`-shaped price table for the internal zone."""
    return pd.DataFrame(
        {"zone": ["p9"], "value": [42.0], "units": ["usd/mwh"]},
        index=pd.to_datetime(["2030-01-01"]),
    )


def import_links(n):
    return n.links[n.links.carrier == "imports"]


def test_inbound_capacity_by_zone_ignores_outbound_rows(footprint_network):
    inbound = inbound_capacity_by_zone(footprint_network, FLOWGATES, "reeds_zone")
    assert inbound.to_dict() == {"p12": 500.0, "p13": 2000.0, "p28": 1000.0}


def test_generator_mode_builds_external_buses_generators_and_links(footprint_network):
    n = footprint_network
    add_external_regions(n, "imports", "generator", FLOWGATES, 30.0, co2_emissions=0.428, zone_col="reeds_zone")

    for zone in ("p12", "p13", "p28"):
        assert f"{zone}_imports" in n.buses.index
        assert n.buses.at[f"{zone}_imports", "carrier"] == "imports"

    gens = n.generators[n.generators.carrier == GENERIC_IMPORT_CARRIER]
    assert sorted(gens.bus) == ["p12_imports", "p13_imports", "p28_imports"]
    assert not gens.p_nom_extendable.any()
    # p_nom is the zone's total INBOUND interface capacity
    assert gens.set_index("bus")["p_nom"].to_dict() == {
        "p12_imports": 500.0,
        "p13_imports": 2000.0,
        "p28_imports": 1000.0,
    }
    assert (gens.marginal_cost == 30.0).all()

    links = import_links(n)
    assert sorted(links.index) == ["p9_p12_imports", "p9_p13_imports", "p9_p28_imports"]
    assert links.set_index("bus0")["p_nom"].to_dict() == {
        "p12_imports": 500.0,
        "p13_imports": 2000.0,
        "p28_imports": 1000.0,
    }
    assert (links.bus1 == "p9").all()
    assert not links.p_nom_extendable.any()

    # generator mode prices the GENERATOR, so the interface link is free
    assert (links.marginal_cost == 0).all()
    # and there is no bottomless import Store
    assert n.stores[n.stores.carrier == "imports"].empty


def test_store_mode_prices_the_links_and_keeps_the_store(footprint_network):
    n = footprint_network
    add_external_regions(n, "imports", "store", FLOWGATES, 30.0, co2_emissions=0.428, zone_col="reeds_zone")

    links = import_links(n)
    assert (links.marginal_cost == 30.0).all()
    assert sorted(n.stores[n.stores.carrier == "imports"].index) == [
        "p12_imports",
        "p13_imports",
        "p28_imports",
    ]
    assert n.generators[n.generators.carrier == GENERIC_IMPORT_CARRIER].empty


def test_representations_agree_on_interface_capacity(footprint_network):
    """The transfer capacity a zone can deliver is the same in both modes."""
    store = footprint_network.copy()
    add_external_regions(store, "imports", "store", FLOWGATES, 30.0, zone_col="reeds_zone")
    generator = footprint_network.copy()
    add_external_regions(generator, "imports", "generator", FLOWGATES, 30.0, zone_col="reeds_zone")

    pd.testing.assert_series_equal(
        import_links(store).set_index("bus0")["p_nom"].sort_index(),
        import_links(generator).set_index("bus0")["p_nom"].sort_index(),
    )


def test_generator_mode_moves_emissions_onto_the_generator_carrier(footprint_network):
    n = footprint_network
    add_external_regions(n, "imports", "generator", FLOWGATES, 30.0, co2_emissions=0.428, zone_col="reeds_zone")

    # links never carry emissions in PyPSA, so the factor must sit on the generator
    assert n.carriers.at[GENERIC_IMPORT_CARRIER, "co2_emissions"] == pytest.approx(0.428)
    assert n.carriers.at["imports", "co2_emissions"] == 0


def test_store_mode_keeps_emissions_on_the_imports_carrier(footprint_network):
    n = footprint_network
    add_external_regions(n, "imports", "store", FLOWGATES, 30.0, co2_emissions=0.428, zone_col="reeds_zone")
    assert n.carriers.at["imports", "co2_emissions"] == pytest.approx(0.428)


def test_generator_price_comes_from_the_wholesale_table(footprint_network):
    n = footprint_network
    add_external_regions(n, "imports", "generator", FLOWGATES, wholesale_costs(), zone_col="reeds_zone")

    name = f"p13_imports {GENERIC_IMPORT_CARRIER}"
    assert name in n.generators_t.marginal_cost.columns
    assert n.generators_t.marginal_cost[name].to_numpy() == pytest.approx(42.0)


@pytest.mark.parametrize("representation", ["store", "generator"])
def test_exports_are_identical_across_representations(footprint_network, representation):
    """Export construction is deliberately untouched by the representation switch."""
    n = footprint_network
    add_external_regions(n, "exports", representation, FLOWGATES, -30.0, zone_col="reeds_zone")

    links = n.links[n.links.carrier == "exports"]
    assert sorted(links.index) == ["p9_p13_exports"]
    assert links.at["p9_p13_exports", "bus0"] == "p9"
    assert links.at["p9_p13_exports", "bus1"] == "p13_exports"
    assert links.at["p9_p13_exports", "p_nom"] == 1500.0
    # the export revenue stays on the link in both modes
    assert links.at["p9_p13_exports", "marginal_cost"] == -30.0
    assert not n.stores[n.stores.carrier == "exports"].empty
    # nothing but export links may inject into the export bus
    assert n.generators[n.generators.bus == "p13_exports"].empty


# ---------------------------------------------------------------------------
# external_regions: contracted units behind the boundary
# ---------------------------------------------------------------------------


def test_map_remote_units_to_zones_prefers_the_largest_matching_zone():
    inbound = pd.Series({"p12": 500.0, "p13": 2000.0, "p28": 1000.0})
    zones = map_remote_units_to_zones(
        pd.Series({"R Hoover": "NV", "R Palo Verde": "AZ"}),
        ["p12", "p13", "p28"],
        inbound,
        "reeds_zone",
        MEMBERSHIP,
    )
    assert zones["R Hoover"] == "p13"  # NV, and p13 > p12
    assert zones["R Palo Verde"] == "p28"


def test_map_remote_units_to_zones_falls_back_and_warns(caplog):
    inbound = pd.Series({"p12": 500.0, "p13": 2000.0, "p28": 1000.0})
    with caplog.at_level(logging.WARNING):
        zones = map_remote_units_to_zones(
            pd.Series({"R Intermountain": "UT"}),
            ["p12", "p13", "p28"],
            inbound,
            "reeds_zone",
            MEMBERSHIP,
        )
    assert zones["R Intermountain"] == "p13"  # largest inbound capacity overall
    assert "R Intermountain" in caplog.text
    assert "UT" in caplog.text


def test_map_remote_units_to_zones_state_boundaries_need_no_membership():
    inbound = pd.Series({"NV": 2000.0, "AZ": 1000.0})
    zones = map_remote_units_to_zones(
        pd.Series({"R Apex": "NV"}),
        ["NV", "AZ"],
        inbound,
        "reeds_state",
        None,
    )
    assert zones["R Apex"] == "NV"


def remote_bundle(n):
    """A toy `build_remote_unit_bundle` output: one AZ firm unit, one UT VRE unit."""
    units = pd.DataFrame(
        [
            {
                "name": "R Palo Verde",
                "carrier": "nuclear",
                "bus": "p9",
                "p_nom": 600.0,
                "state": "AZ",
                "efficiency": 0.33,
                "marginal_cost": 8.0,
                "heat_rate": 10.0,
                "summer_derate": 1.0,
                "winter_derate": 1.0,
                "ramp_limit_up": 1.0,
                "ramp_limit_down": 1.0,
                "min_up_time": 0,
                "min_down_time": 0,
                "start_up_cost": 0.0,
                "fuel_cost": 1.0,
                "min_load_pu": 0.0,
                "build_year": 1988,
                "duration": 4.0,
            },
            {
                "name": "R Cape Solar",
                "carrier": "solar",
                "bus": "p9",
                "p_nom": 50.0,
                "state": "UT",
                "efficiency": 1.0,
                "marginal_cost": 0.0,
                "heat_rate": 0.0,
                "summer_derate": 1.0,
                "winter_derate": 1.0,
                "ramp_limit_up": 1.0,
                "ramp_limit_down": 1.0,
                "min_up_time": 0,
                "min_down_time": 0,
                "start_up_cost": 0.0,
                "fuel_cost": 0.0,
                "min_load_pu": 0.0,
                "build_year": 2024,
                "duration": 4.0,
            },
        ],
    ).set_index("name")
    return {
        "units": units,
        "vre_profiles": pd.DataFrame({"R Cape Solar": np.full(len(n.snapshots), 0.4)}, index=n.snapshots),
        "costs": pd.DataFrame(),
        "conventional_carriers": ["nuclear"],
        "unit_commitment": False,
    }


def test_remote_bundle_attaches_behind_the_matching_external_bus(footprint_network, caplog):
    n = footprint_network
    with caplog.at_level(logging.WARNING):
        add_external_regions(
            n,
            "imports",
            "generator",
            FLOWGATES,
            30.0,
            co2_emissions=0.428,
            zone_col="reeds_zone",
            remote_bundle=remote_bundle(n),
            membership=MEMBERSHIP,
        )

    # AZ unit lands behind the AZ boundary zone
    assert n.generators.at["R Palo Verde", "bus"] == "p28_imports"
    assert n.generators.at["R Palo Verde", "p_nom"] == 600.0
    assert not n.generators.at["R Palo Verde", "p_nom_extendable"]

    # UT has no direct CA interface -> fallback to the largest inbound zone
    assert n.generators.at["R Cape Solar", "bus"] == "p13_imports"
    assert "R Cape Solar" in caplog.text

    # the borrowed profile travels with the bundle
    assert n.generators_t.p_max_pu["R Cape Solar"].to_numpy() == pytest.approx(0.4)

    # and none of them sit inside the footprint any more
    assert "p9" not in set(n.generators.loc[["R Palo Verde", "R Cape Solar"], "bus"])


def test_remote_bundle_is_ignored_in_store_mode(footprint_network):
    n = footprint_network
    add_external_regions(
        n,
        "imports",
        "store",
        FLOWGATES,
        30.0,
        zone_col="reeds_zone",
        remote_bundle=remote_bundle(n),
        membership=MEMBERSHIP,
    )
    assert "R Palo Verde" not in n.generators.index


def test_unknown_representation_raises(footprint_network):
    with pytest.raises(ValueError, match="representation"):
        add_external_regions(footprint_network, "imports", "banana", FLOWGATES, 30.0, zone_col="reeds_zone")
