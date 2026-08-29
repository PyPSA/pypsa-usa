"""
Test the aggregate transmission interface limits.

This module contains tests for the RESOLVE/NARIS style interface constraints
applied to the electricity import/export links in PyPSA-USA.
"""

import os
import sys

import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from _helpers import get_multiindex_snapshots
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

    Mirrors the conventions of ``add_extra_components.add_elec_imports_exports``:
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
