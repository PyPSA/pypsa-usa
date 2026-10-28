"""
Test the CO2 storage, CO2 network and DAC components of ``add_extra_components``.

``add_co2_storage`` sends the CO2 captured by CCS plants to the capture bus of
their node, and in electricity-only studies vents what is not captured to an
atmosphere bus. ``add_dac`` adds one DAC per node with a CC plant, and
``add_co2_network`` links the capture buses along the transmission lines.
"""

import os
import sys

import pandas as pd
import pypsa
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from _helpers import get_multiindex_snapshots
from add_extra_components import add_co2_network, add_co2_storage, add_dac

GAS, COAL = 0.2, 0.3  # t CO2 / MWh_th
COSTS = pd.DataFrame({"co2_emissions": [GAS, COAL]}, index=["gas", "coal"])


def config(granularity: str = "node") -> dict:
    return {
        "plotting": {
            "tech_colors": {"co2": "#000000", "dac": "#000000"},
            "nice_names": {"co2": "CO2", "dac": "DAC"},
        },
        "dac": {
            "granularity": granularity,
            "capital_cost": 1e6,
            "lifetime": 20,
            "discount_rate": 0.07,
            "electricity_input": 2.5,
        },
        "co2": {"network": {"capital_cost": 1e6, "lifetime": 40, "discount_rate": 0.07, "marginal_cost": 4}},
        "model_topology": {"transmission_network": "reeds"},
    }


@pytest.fixture
def storage_csv(tmp_path):
    """CO2 storage of both nodes, listed in the opposite order of the CCS plants."""
    fn = tmp_path / "co2_storage.csv"
    storage = pd.DataFrame({"node": ["p2", "p1"], "potential [MtCO2]": [10, 20], "cost [USD/tCO2]": [5, 10]})
    storage.to_csv(fn, index=False)
    return str(fn)


def network() -> pypsa.Network:
    """Two nodes linked by an AC link, the first with a gas and a coal CCS plant."""
    n = pypsa.Network()
    n.snapshots = get_multiindex_snapshots(
        sns_config={"start": "2030-01-01 00:00", "end": "2030-01-01 01:00", "inclusive": "both"},
        invest_periods=[2030],
    )
    n.set_investment_periods(periods=[2030])
    n.add("Bus", ["p1", "p2"], carrier="AC")
    n.add("Link", "p1||p2", bus0="p1", bus1="p2", carrier="AC", length=100)
    n.add("Carrier", ["CCGT-95CCS", "coal-99CCS"], color="#000000", nice_name=["CCGT 95% CCS", "Coal 99% CCS"])
    return n


@pytest.fixture
def elec_network():
    """CCS plants as generators, as in electricity-only studies."""
    n = network()
    n.add(
        "Generator",
        ["p1 CCGT-95CCS_2030", "p1 coal-99CCS_2030", "p2 CCGT-95CCS_2030"],
        bus=["p1", "p1", "p2"],
        carrier=["CCGT-95CCS", "coal-99CCS", "CCGT-95CCS"],
        efficiency=[0.5, 0.4, 0.5],
        capital_cost=100,
    )
    n.generators_t.p_max_pu["p1 CCGT-95CCS_2030"] = 0.9
    return n


@pytest.fixture
def sector_network():
    """CCS plants as links drawing fuel from state buses, as in sector studies."""
    n = network()
    n.add("Bus", ["UT gas", "UT coal", "UT pwr-co2"])
    n.add(
        "Link",
        ["p1 CCGT-95CCS_2030", "p2 coal-99CCS_2030"],
        bus0=["UT gas", "UT coal"],
        bus1=["p1", "p2"],
        bus2="UT pwr-co2",
        carrier=["CCGT-95CCS", "coal-99CCS"],
        efficiency=[0.5, 0.4],
        capital_cost=100,
    )
    n.links_t.p_max_pu["p1 CCGT-95CCS_2030"] = 0.9
    return n


# electricity-only studies


def test_cc_generators_vent_and_capture_at_their_node(elec_network, storage_csv):
    n = elec_network
    add_co2_storage(n, config(), storage_csv, COSTS, sector=False)

    cc = n.links.loc[["p1 CCGT-95CC_2030", "p1 coal-99CC_2030", "p2 CCGT-95CC_2030"]]
    assert cc.bus1.tolist() == ["p1", "p1", "p2"]
    assert cc.bus2.tolist() == ["p1 atmosphere", "p1 atmosphere", "p2 atmosphere"]
    assert cc.bus3.tolist() == ["p1 co2 capture", "p1 co2 capture", "p2 co2 capture"]
    # one atmosphere store per node, however many plants vent to it
    assert sorted(n.stores.index[n.stores.index.str.endswith("atmosphere")]) == ["p1 atmosphere", "p2 atmosphere"]


def test_cc_generators_are_renamed_with_their_time_series(elec_network, storage_csv):
    n = elec_network
    add_co2_storage(n, config(), storage_csv, COSTS, sector=False)

    assert not n.generators.index.str.contains("CCS").any()
    assert n.generators.at["p1 CCGT-95CC_2030", "bus"] == "p1 CCGT-95CC_2030"
    assert n.generators.at["p1 CCGT-95CC_2030", "carrier"] == "CCGT-95CC"
    assert n.generators_t.p_max_pu.columns.tolist() == ["p1 CCGT-95CC_2030"]
    assert {"CCGT-95CC", "coal-99CC"} <= set(n.carriers.index)
    n.consistency_check(strict=["unknown_buses"])


def test_one_national_atmosphere(elec_network, storage_csv):
    n = elec_network
    add_co2_storage(n, config("nation"), storage_csv, COSTS, sector=False)

    assert (n.links.loc[n.links.bus2.str.endswith("atmosphere"), "bus2"] == "atmosphere").all()
    assert n.stores.index[n.stores.index.str.endswith("atmosphere")].tolist() == ["atmosphere"]


def test_one_dac_per_node_with_a_cc_generator(elec_network, storage_csv):
    n = elec_network
    add_co2_storage(n, config(), storage_csv, COSTS, sector=False)
    add_dac(n, config(), sector=False)

    dac = n.links[n.links.carrier == "dac"].sort_index()
    assert dac.index.tolist() == ["p1 dac", "p2 dac"]
    assert dac.bus0.tolist() == ["p1 atmosphere", "p2 atmosphere"]
    assert dac.bus1.tolist() == ["p1 co2 capture", "p2 co2 capture"]
    assert dac.bus2.tolist() == ["p1", "p2"]


def test_co2_network_follows_the_ac_links(elec_network, storage_csv):
    n = elec_network
    add_co2_storage(n, config(), storage_csv, COSTS, sector=False)
    add_co2_network(n, config())

    transport = n.links.loc["p1||p2 co2 transport"]
    assert (transport.bus0, transport.bus1) == ("p1 co2 capture", "p2 co2 capture")


# sector studies


def test_ccs_links_capture_at_their_node(sector_network, storage_csv):
    n = sector_network
    add_co2_storage(n, config(), storage_csv, COSTS, sector=True)

    cc = n.links.loc[["p1 CCGT-95CC_2030", "p2 coal-99CC_2030"]]
    assert cc.bus4.tolist() == ["p1 co2 capture", "p2 co2 capture"]
    assert cc.carrier.tolist() == ["CCGT-95CC", "coal-99CC"]
    assert n.links_t.p_max_pu.columns.tolist() == ["p1 CCGT-95CC_2030"]
    n.consistency_check(strict=["unknown_buses"])
