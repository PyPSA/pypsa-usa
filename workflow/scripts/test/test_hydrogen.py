"""Unit tests for the hydrogen sector: costs and ``build_hydrogen``."""

import os
import re
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
import pytest
from shapely.geometry import box

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from _helpers import get_gas_infra_contraction, get_multiindex_snapshots
from add_sectors import add_fuel_supply, get_period_fuel_price, get_plant_fuel_price
from build_cost_data import build_hydrogen_turbines, get_hydrogen_costs
from build_emission_tracking import build_ch4_tracking
from build_hydrogen import (
    add_retrofitted_pipelines,
    add_retrofitted_storage,
    build_hydrogen,
    get_pipeline_routes,
    get_storage_types,
    read_geopackage,
)
from money import deflator
from opts.bidirectional_link import DIRECTION_SUFFIX, add_bidirectional_link_constraints
from opts.retrofit import add_retrofit_constraints

pytestmark = pytest.mark.fast

HYDROGEN_COSTS = os.path.join(os.path.dirname(__file__), "..", "..", "repo_data", "costs", "hydrogen_costs.csv")

# CA and NV share a border; AZ touches NV only at a corner
STATES = gpd.GeoDataFrame(
    {"STATE": ["CA", "NV", "AZ"]},
    geometry=[box(-121, 35, -119, 37), box(-119, 35, -117, 37), box(-117, 37, -115, 39)],
    crs="EPSG:4269",
)
CENTERS = pd.DataFrame({"x": [-120.0, -118.0, -116.0], "y": [36.0, 36.0, 38.0]}, index=["CA", "NV", "AZ"])


@pytest.fixture
def storage_sites(tmp_path):
    """Salt under half of CA, hard rock under all of AZ and a sliver of NV."""
    path = tmp_path / "h2_storage_sites.gpkg"
    gpd.GeoDataFrame(geometry=[box(-121, 35, -120, 37)], crs="EPSG:4269").to_file(path, layer="salt")
    gpd.GeoDataFrame(
        geometry=[box(-117, 37, -115, 39), box(-119, 35, -118.99, 35.01)],
        crs="EPSG:4269",
    ).to_file(path, layer="hardrock")
    return str(path)


@pytest.fixture
def demand(tmp_path):
    path = tmp_path / "adp2024_h2_demand.csv"
    pd.DataFrame(
        {
            "scenario": ["central"] * 4 + ["baseline"] * 2,
            "state": ["CA", "CA", "NV", "NV", "CA", "NV"],
            "year": [2024, 2035, 2024, 2035, 2030, 2030],
            "demand_mwh": [8760.0, 8760.0 * 12, 0.0, 8760.0 * 2.2, 1.0, 1.0],
        },
    ).to_csv(path, index=False)
    return str(path)


def gas_pipelines(*pairs):
    return pd.DataFrame(pairs, columns=["STATE_FROM", "STATE_TO"])


def hydrogen_costs(year, dollar_year=2023):
    costs = get_hydrogen_costs(HYDROGEN_COSTS, year, dollar_year)
    return costs.pivot(index="pypsa-name", columns="parameter", values="value")


@pytest.fixture
def network(base_network):
    """z1 in CA, z2 and z3 in NV, with the state gas and h2-co2 buses of a sector model."""
    n = base_network
    n.buses.loc[["z1", "z2", "z3"], "STATE"] = ["CA", "NV", "NV"]
    for state in ("CA", "NV"):
        n.add("Bus", f"{state} gas", carrier="gas", STATE=state)
        n.add("Bus", f"{state} h2-co2", carrier="co2", STATE=state)
        n.add("Store", f"{state} h2-co2", bus=f"{state} h2-co2", carrier="co2", e_nom=np.inf)
    return n


OPTIONS = {
    "demand_scenario": "central",
    "production": ["h2_electrolysis", "h2_smr", "h2_smr_ccs"],
    "storage": True,
    "pipelines": True,
    "pipeline_length_factor": 1.25,
}


def build(n, demand, storage_sites, options=OPTIONS):
    build_hydrogen(
        n,
        costs={2030: hydrogen_costs(2030)},
        demand_path=demand,
        storage_sites_path=storage_sites,
        gas_pipelines=gas_pipelines(("CA", "NV")),
        states_geometry=STATES,
        center_points=CENTERS,
        options=options,
    )


###
# costs
###


def test_hydrogen_costs_cover_every_technology():
    df = pd.read_csv(HYDROGEN_COSTS)
    for tech, params in df.groupby("technology").parameter:
        assert {"investment", "fixed-om", "lifetime", "discount-rate"} <= set(params), tech


def test_hydrogen_costs_interpolate_and_annualize(tmp_path):
    path = tmp_path / "costs.csv"
    pd.DataFrame(
        {
            "technology": ["a"] * 6 + ["b"] * 4,
            "year": [2020, 2030, "", "", "", "", "", "", "", ""],
            "parameter": [
                *["investment", "investment", "fixed-om", "lifetime", "discount-rate", "stack-replacement"],
                *["investment", "fixed-om", "lifetime", "discount-rate"],
            ],
            "value": [1000.0, 500.0, 10.0, 20.0, 0.05, 0.0, 100.0, 1.0, 10.0, 0.0],
        },
    ).to_csv(path, index=False)

    costs = get_hydrogen_costs(path, 2024, 2023).pivot(index="pypsa-name", columns="parameter", values="value")

    assert costs.at["a", "investment"] == pytest.approx(800)
    annuity = 0.05 / (1 - 1.05**-20)
    assert costs.at["a", "annualized_capex_fom"] == pytest.approx(800 * annuity + 10)
    assert costs.at["b", "annualized_capex_fom"] == pytest.approx(100 / 10 + 1)
    # held constant outside the years given
    later = get_hydrogen_costs(path, 2050, 2023).pivot(index="pypsa-name", columns="parameter", values="value")
    assert later.at["a", "investment"] == pytest.approx(500)


def test_hydrogen_costs_are_in_the_model_dollar_year():
    base, later = hydrogen_costs(2030, 2023), hydrogen_costs(2030, 2025)
    factor = deflator(2023, 2025)
    for tech, parameter in (
        ("h2_electrolysis", "investment"),
        ("h2_electrolysis", "fixed-om"),
        ("h2_electrolysis", "annualized_capex_fom"),
        ("h2_smr", "variable-om"),
    ):
        assert later.at[tech, parameter] == pytest.approx(base.at[tech, parameter] * factor)
    # only the costs move
    assert later.at["h2_electrolysis", "efficiency"] == base.at["h2_electrolysis", "efficiency"]
    assert later.at["h2_electrolysis", "lifetime"] == base.at["h2_electrolysis", "lifetime"]


def test_electrolyzer_pays_its_stack_replacement():
    costs = hydrogen_costs(2030)
    r, life = 0.07, 20
    investment = costs.at["h2_electrolysis", "investment"] * (1 + 0.6 / (1 + r) ** 10)
    annuity = r / (1 - (1 + r) ** -life)
    expected = investment * annuity + costs.at["h2_electrolysis", "fixed-om"]
    assert costs.at["h2_electrolysis", "annualized_capex_fom"] == pytest.approx(expected)


def test_hydrogen_turbines_cost_ten_percent_more_than_gas():
    atb = pd.DataFrame(
        {
            "pypsa-name": ["OCGT", "CCGT"],
            "capex_overnight_per_kw": [1000.0, 1200.0],
            "capex_grid_connection_per_kw": [100.0, 100.0],
            # ATB construction financing cost in USD/kW: capex = OCC + GCC + CFC
            "capex_construction_finance_factor": [110.0, 130.0],
            "capex_per_kw": [1210.0, 1430.0],
            "heat_rate_mmbtu_per_mwh": [9.7, 6.2],
            "fuel_cost_real_per_mwhth": [13.0, 13.0],
            "co2_emissions": [0.18, 0.18],
        },
    )
    ocht, ccht = build_hydrogen_turbines(atb)

    assert ocht["pypsa-name"].item() == "OCHT"
    assert ccht["pypsa-name"].item() == "CCHT"
    assert ocht.capex_overnight_per_kw.item() == pytest.approx(1100)
    # the grid connection and construction financing costs of the gas plant are added as they are
    assert ocht.capex_per_kw.item() == pytest.approx(1100 + 100 + 110)
    assert ccht.capex_per_kw.item() == pytest.approx(1320 + 100 + 130)
    assert ccht.capex_construction_finance_factor.item() == pytest.approx(130)
    assert ocht.heat_rate_mmbtu_per_mwh.item() == 9.7
    assert ocht.fuel_cost_real_per_mwhth.item() == pytest.approx(20 * 3.412)
    assert ocht.co2_emissions.item() == 0


###
# network
###


def test_storage_type_is_the_cheapest_available(storage_sites):
    types = get_storage_types(STATES, storage_sites, ["CA", "NV", "AZ"])
    # NV's hard rock covers less than 1% of the state
    assert types.to_dict() == {"CA": "salt", "NV": "pipe", "AZ": "hardrock"}


def test_pipeline_routes_follow_transmission_and_gas_pipelines():
    n = pypsa.Network()
    n.add("Bus", ["ca", "nv", "az", "ut", "id", "mx"], carrier="AC", STATE=["CA", "NV", "AZ", "UT", "ID", "MX"])
    n.add("Bus", ["CA gas", "OR gas"], carrier="gas", STATE=["CA", "OR"])
    n.add("Line", "ca-nv", bus0="ca", bus1="nv")
    n.add("Link", "az-ut", bus0="az", bus1="ut", carrier="AC")  # AC transmission of a transport model
    n.add("Link", "ut-id", bus0="ut", bus1="id", carrier="DC")  # DC does not count
    n.add("Line", "ca-mx", bus0="ca", bus1="mx")  # MX is not a state in the model
    n.add("Link", "gas pipe", bus0="CA gas", bus1="OR gas")  # not electricity
    gas = gas_pipelines(("NV", "CA"), ("OR", "CA"), ("AZ", None))

    routes = get_pipeline_routes(n, ["AZ", "CA", "ID", "NV", "OR", "UT"], gas)

    # NV-CA is joined both ways, but is one route
    assert routes == [("AZ", "UT"), ("CA", "NV"), ("CA", "OR")]


def test_demand_is_interpolated_and_spread_over_the_year(network, demand, storage_sites):
    build(network, demand, storage_sites)

    loads = network.loads.loc[["CA H2", "NV H2"]]
    assert loads.bus.tolist() == ["CA H2", "NV H2"]
    assert (loads.carrier == "H2").all()

    # flat at the annual demand over 8760 hours; 2030 lies 6/11 of the way from 2024 to 2035
    p_set = network.loads_t.p_set[["CA H2", "NV H2"]]
    assert (p_set.nunique() == 1).all()
    assert p_set["CA H2"].iloc[0] == pytest.approx(1 + 11 * 6 / 11)
    assert p_set["NV H2"].iloc[0] == pytest.approx(2.2 * 6 / 11)


def test_production_links(network, demand, storage_sites):
    build(network, demand, storage_sites)
    costs = hydrogen_costs(2030)
    links = network.links

    electrolysis = links[links.carrier == "h2_electrolysis"]
    assert electrolysis.bus0.tolist() == ["z1", "z2", "z3"]
    assert electrolysis.bus1.tolist() == ["CA H2", "NV H2", "NV H2"]
    assert (electrolysis.build_year == 2030).all()
    assert electrolysis.index[0] == "z1 h2_electrolysis_2030"

    smr = links.loc["z2 h2_smr_ccs_2030"]
    efficiency = 1 / costs.at["h2_smr_ccs", "gas-input"]
    assert (smr.bus0, smr.bus1, smr.bus2, smr.bus4) == ("NV gas", "NV H2", "NV h2-co2", "z2")
    assert smr.efficiency == pytest.approx(efficiency)
    assert smr.efficiency2 == pytest.approx(costs.at["h2_smr_ccs", "co2-emitted"] * efficiency)
    assert smr.efficiency4 == pytest.approx(-costs.at["h2_smr_ccs", "electricity-input"] * efficiency)
    assert smr.capital_cost == pytest.approx(costs.at["h2_smr_ccs", "annualized_capex_fom"] * efficiency)
    # port 3 is left to the methane leakage; without CO2 storage the captured CO2 goes nowhere
    assert smr.get("bus3", "") == ""
    assert "bus5" not in links.columns


def test_captured_co2_goes_to_co2_storage(network, demand, storage_sites):
    network.add("Bus", "z2 co2 capture", carrier="co2")
    build(network, demand, storage_sites)
    costs = hydrogen_costs(2030)

    smr = network.links.filter(like="h2_smr_ccs", axis=0)
    assert smr.bus5.tolist() == ["", "z2 co2 capture", ""]
    efficiency = 1 / costs.at["h2_smr_ccs", "gas-input"]
    assert smr.efficiency5.iloc[1] == pytest.approx(costs.at["h2_smr_ccs", "co2-captured"] * efficiency)


def test_methane_leakage_of_reformers(network, demand, storage_sites):
    build(network, demand, storage_sites, OPTIONS | {"production": ["h2_smr"]})
    network.add("Bus", ["CA gas production", "NV gas production"], carrier="gas production")
    network.add("Link", "NV gas production", bus0="NV gas production", bus1="NV gas", carrier="gas production")

    build_ch4_tracking(network, gwp=30, upstream_leakage_rate=0.01, downstream_leakage_rate=0.02)

    smr = network.links.loc["z2 h2_smr_2030"]
    assert (smr.bus3, smr.efficiency3, smr.bus4) == ("NV gas-ch4", pytest.approx(0.6), "z2")
    assert network.links.at["NV gas production", "bus3"] == "NV gas-ch4"
    # the port is registered, so links added later leave it unconnected
    network.add("Link", "later", bus0="z1", bus1="CA H2")
    assert network.links.at["later", "bus3"] == ""


def test_geopackage_is_read_without_gdal(storage_sites):
    for layer in ("salt", "hardrock"):
        read = read_geopackage(storage_sites, layer)
        expected = gpd.read_file(storage_sites, layer=layer)
        assert read.crs == expected.crs
        assert read.geometry.geom_equals(expected.geometry).all()


def test_repo_storage_sites_read_like_gdal():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "repo_data", "hydrogen", "h2_storage_sites.gpkg")
    for layer in ("salt", "hardrock"):
        read = read_geopackage(path, layer).to_crs("EPSG:5070")
        expected = gpd.read_file(path, layer=layer).to_crs("EPSG:5070")
        assert len(read) == len(expected)
        assert read.area.sum() == pytest.approx(expected.area.sum())


def test_storage(network, demand, storage_sites):
    build(network, demand, storage_sites)
    costs = hydrogen_costs(2030)

    stores = network.stores.loc[["CA H2 storage_2030", "NV H2 storage_2030"]]
    assert stores.carrier.tolist() == ["h2_storage_salt", "h2_storage_pipe"]
    # a 30% cushion; the cost is per MWh of working capacity
    assert stores.e_min_pu.tolist() == pytest.approx([0.3, 0.3])
    assert stores.capital_cost.tolist() == pytest.approx(
        [
            costs.at["h2_storage_salt", "annualized_capex_fom"] * 0.7,
            costs.at["h2_storage_pipe", "annualized_capex_fom"] * 0.7,
        ],
    )
    assert stores.e_nom_extendable.all()

    charger = network.links.loc["CA H2 storage charger_2030"]
    assert (charger.bus0, charger.bus1) == ("CA H2", "CA H2 storage")
    assert charger.efficiency == pytest.approx(1 - costs.at["h2_storage_charger", "compression-loss"])
    discharger = network.links.loc["CA H2 storage discharger_2030"]
    assert (discharger.bus0, discharger.bus1) == ("CA H2 storage", "CA H2")
    assert charger.p_nom_extendable and discharger.p_nom_extendable


def test_pipelines(network, demand, storage_sites):
    build(network, demand, storage_sites)
    costs = hydrogen_costs(2030)

    pipes = network.links[network.links.carrier == "h2_pipeline"]
    assert sorted(pipes.index) == ["CA NV h2_pipeline_fwd_2030", "CA NV h2_pipeline_rev_2030"]
    assert all(DIRECTION_SUFFIX.match(x) for x in pipes.index)

    fwd = pipes.loc["CA NV h2_pipeline_fwd_2030"]
    assert (fwd.bus0, fwd.bus1) == ("CA H2", "NV H2")
    length = 1.25 * pypsa.geo.haversine_pts(np.array([[-120.0, 36.0]]), np.array([[-118.0, 36.0]]))[0]
    assert fwd.length == pytest.approx(length)
    # compression and leakage, both per distance as for natural gas
    loss_per_1000km = (costs.at["h2_pipeline", "compression-loss"] + costs.at["h2_pipeline", "leakage"]) / 100
    assert fwd.efficiency == pytest.approx(1 - loss_per_1000km * length / 1000)
    assert pipes.capital_cost.sum() == pytest.approx(costs.at["h2_pipeline", "annualized_capex_fom"] * length)


FORMATIONS = ("salt-dome", "depleted-field", "aquifer")

# gas storage per formation: a CA salt dome and aquifer and a NV depleted field
GAS_STORAGE = (
    ("CA salt-dome", "CA", {"salt-dome": 1.0}, {"salt-dome": 1.0}, 1000.0, 0.2, 10.0),
    ("CA aquifer", "CA", {"aquifer": 1.0}, {"aquifer": 1.0}, 500.0, 0.5, 5.0),
    ("NV depleted-field", "NV", {"depleted-field": 1.0}, {"depleted-field": 1.0}, 2000.0, 0.5, 4.0),
)


def add_gas_infrastructure(n, reservoirs=GAS_STORAGE, shares=True, contraction=True):
    """
    Gas storage, with the shares of its formations in its working gas and withdrawal
    as ``GasStorage`` gives them, and pipelines between CA and NV both ways. With
    ``contraction``, as for retrofitting, they are optimised up to their capacity.
    """
    for name, state, working_shares, withdrawal_shares, energy, cushion, withdrawal in reservoirs:
        storage = f"{name} gas storage"
        n.add("Bus", storage, carrier="gas storage")
        formation_shares = {
            **{f"working_share_{x}": working_shares.get(x, 0.0) for x in FORMATIONS},
            **{f"withdrawal_share_{x}": withdrawal_shares.get(x, 0.0) for x in FORMATIONS},
        }
        n.add(
            "Store",
            storage,
            bus=storage,
            carrier="gas storage",
            e_nom=energy,
            e_nom_max=energy,
            e_nom_extendable=contraction,
            e_min_pu=cushion,
            e_cyclic_per_period=True,
            **(formation_shares if shares else {}),
        )
        injection = 0.37 * withdrawal
        n.add(
            "Link",
            [f"{name} charge gas storage", f"{name} discharge gas storage"],
            bus0=[f"{state} gas", storage],
            bus1=[storage, f"{state} gas"],
            carrier="gas storage",
            p_nom=[injection, withdrawal],
            p_nom_max=[injection, withdrawal],
            p_nom_extendable=contraction,
        )
    n.add(
        "Link",
        ["CA NV pipeline", "NV CA pipeline"],
        bus0=["CA gas", "NV gas"],
        bus1=["NV gas", "CA gas"],
        carrier="gas pipeline",
        p_nom=[100.0, 50.0],
        p_nom_max=[100.0, 50.0],
        p_nom_extendable=contraction,
        length=300.0,
        capital_cost=1.0,
    )


def test_retrofitted_storage(network, demand, storage_sites):
    add_gas_infrastructure(network)
    build(network, demand, storage_sites, OPTIONS | {"retrofit": {"storage": True}})
    costs = hydrogen_costs(2030)

    stores = network.stores[network.stores.carrier == "h2_storage_retrofitted"]
    assert stores.index.tolist() == [
        "CA salt-dome H2 storage retrofitted_2030",
        "CA aquifer H2 storage retrofitted_2030",
        "NV depleted-field H2 storage retrofitted_2030",
    ]
    assert stores.retrofit_from.tolist() == [
        "CA salt-dome gas storage",
        "CA aquifer gas storage",
        "NV depleted-field gas storage",
    ]
    assert stores.retrofit_factor.tolist() == pytest.approx([0.2551] * 3)
    assert stores.e_nom_max.tolist() == pytest.approx([0.2551 * 1000, 0.2551 * 500, 0.2551 * 2000])
    # the same cushion share; the cost is per MWh of working capacity
    assert stores.e_min_pu.tolist() == pytest.approx([0.2, 0.5, 0.5])
    assert stores.capital_cost.tolist() == pytest.approx(
        [
            costs.at["h2_storage_retrofitted_salt", "annualized_capex_fom"] * 0.8,
            costs.at["h2_storage_retrofitted_aquifer", "annualized_capex_fom"] * 0.5,
            costs.at["h2_storage_retrofitted_depleted", "annualized_capex_fom"] * 0.5,
        ],
    )

    charger = network.links.loc["CA salt-dome H2 storage retrofitted charger_2030"]
    assert (charger.bus0, charger.bus1) == ("CA H2", "CA salt-dome H2 storage retrofitted")
    assert charger.carrier == "h2_storage_charger"
    assert charger.efficiency == pytest.approx(1 - costs.at["h2_storage_charger", "compression-loss"])
    # new compressors inject through the wells, at most at the hydrogen withdrawal rate of the
    # reservoir (salt caverns withdraw hydrogen faster than porous reservoirs)
    assert charger.p_nom_max == pytest.approx(0.8 * 10)
    assert charger.capital_cost == pytest.approx(
        costs.at["h2_storage_charger_retrofitted_salt", "annualized_capex_fom"]
    )
    # they take nothing from the gas injection
    assert pd.isna(charger.retrofit_from)
    discharger = network.links.loc["NV depleted-field H2 storage retrofitted discharger_2030"]
    assert (discharger.bus0, discharger.bus1) == ("NV depleted-field H2 storage retrofitted", "NV H2")
    assert (discharger.retrofit_factor, discharger.p_nom_max) == pytest.approx((0.4, 0.4 * 4))
    assert discharger.capital_cost == pytest.approx(
        costs.at["h2_storage_discharger_retrofitted_depleted", "annualized_capex_fom"],
    )
    assert discharger.retrofit_from == "NV depleted-field discharge gas storage"


def test_retrofitted_gas_storage_must_contract(network, demand, storage_sites):
    add_gas_infrastructure(network, contraction=False)
    with pytest.raises(ValueError, match="must contract"):
        build(network, demand, storage_sites, OPTIONS | {"retrofit": {"storage": True}})


def test_state_storage_takes_the_mean_of_its_formations(network, demand, storage_sites):
    # a quarter of NV's working gas but half its withdrawal in a salt dome
    add_gas_infrastructure(
        network,
        reservoirs=[
            ("NV", "NV", {"salt-dome": 0.25, "aquifer": 0.75}, {"salt-dome": 0.5, "aquifer": 0.5}, 2000.0, 0.5, 4.0),
        ],
    )
    build(network, demand, storage_sites, OPTIONS | {"retrofit": {"storage": True}})
    costs = hydrogen_costs(2030)

    def annualized(technology, formation):
        return costs.at[f"{technology}_{formation}", "annualized_capex_fom"]

    store = network.stores.loc["NV H2 storage retrofitted_2030"]
    assert store.retrofit_from == "NV gas storage"
    assert store.retrofit_factor == pytest.approx(0.2551)
    # the store weighs the formations by their hydrogen working gas
    salt, aquifer = 0.25 * 0.2551, 0.75 * 0.2551
    expected = salt * annualized("h2_storage_retrofitted", "salt") + aquifer * annualized(
        "h2_storage_retrofitted", "aquifer"
    )
    assert store.capital_cost == pytest.approx(expected / (salt + aquifer) * 0.5)

    # the rates by their hydrogen withdrawal
    salt, aquifer = 0.5 * 0.8, 0.5 * 0.4
    discharger = network.links.loc["NV H2 storage retrofitted discharger_2030"]
    assert discharger.retrofit_factor == pytest.approx(0.6)
    assert discharger.p_nom_max == pytest.approx(0.6 * 4)
    expected = salt * annualized("h2_storage_discharger_retrofitted", "salt") + aquifer * annualized(
        "h2_storage_discharger_retrofitted",
        "aquifer",
    )
    assert discharger.capital_cost == pytest.approx(expected / (salt + aquifer))


def test_storage_retrofitting_needs_the_formation_shares(network, demand, storage_sites):
    add_gas_infrastructure(network, shares=False)
    with pytest.raises(ValueError, match="formation shares"):
        build(network, demand, storage_sites, OPTIONS | {"retrofit": {"storage": True}})


def test_retrofitted_pipelines(network, demand, storage_sites):
    add_gas_infrastructure(network)
    build(network, demand, storage_sites, OPTIONS | {"retrofit": {"pipelines": True}})
    costs = hydrogen_costs(2030)

    pipes = network.links[network.links.carrier == "h2_pipeline_retrofitted"]
    assert sorted(pipes.index) == [
        "CA NV pipeline h2_pipeline_retrofitted_fwd_2030",
        "CA NV pipeline h2_pipeline_retrofitted_rev_2030",
        "NV CA pipeline h2_pipeline_retrofitted_fwd_2030",
        "NV CA pipeline h2_pipeline_retrofitted_rev_2030",
    ]
    assert all(DIRECTION_SUFFIX.match(x) for x in pipes.index)

    fwd = pipes.loc["NV CA pipeline h2_pipeline_retrofitted_fwd_2030"]
    rev = pipes.loc["NV CA pipeline h2_pipeline_retrofitted_rev_2030"]
    assert (fwd.bus0, fwd.bus1, rev.bus0, rev.bus1) == ("NV H2", "CA H2", "CA H2", "NV H2")
    assert fwd.p_nom_max == rev.p_nom_max == pytest.approx(0.6 * 50)
    # the forward link takes the gas capacity, the reverse one follows it
    assert fwd.retrofit_from == "NV CA pipeline"
    assert pd.isna(rev.retrofit_from)
    # along the gas pipeline, losing as much as a new hydrogen pipeline
    assert fwd.length == 300
    loss_per_1000km = (costs.at["h2_pipeline", "compression-loss"] + costs.at["h2_pipeline", "leakage"]) / 100
    assert fwd.efficiency == pytest.approx(1 - loss_per_1000km * 0.3)
    assert fwd.capital_cost + rev.capital_cost == pytest.approx(
        costs.at["h2_pipeline_retrofitted", "annualized_capex_fom"] * 300,
    )
    assert network.links.loc[["CA NV pipeline", "NV CA pipeline"], "p_nom_extendable"].all()


def test_nothing_to_retrofit_without_natural_gas(network, demand, storage_sites):
    build(network, demand, storage_sites, OPTIONS | {"retrofit": {"storage": True, "pipelines": True}})

    assert not network.stores.carrier.str.contains("retrofitted").any()
    assert not network.links.carrier.str.contains("retrofitted").any()


def test_retrofitting_shares_the_gas_capacity():
    n = pypsa.Network()
    n.set_snapshots(pd.MultiIndex.from_product([[2030], pd.date_range("2030-01-01", periods=4, freq="h")]))
    n.set_investment_periods([2030])
    n.add("Bus", ["CA gas", "NV gas"], carrier="gas", STATE=["CA", "NV"])
    n.add("Bus", ["CA H2", "NV H2"], carrier="H2", STATE=["CA", "NV"], x=[-120.0, -118.0], y=36.0)
    add_gas_infrastructure(n)
    costs = hydrogen_costs(2030)
    add_retrofitted_storage(n, costs, 2030, CENTERS)
    add_retrofitted_pipelines(n, costs, 2030)
    # hydrogen in NV in the first hour, for demand in CA in the last
    n.add("Generator", "NV H2", bus="NV H2", p_nom=100, p_max_pu=pd.Series([1.0, 0, 0, 0], index=n.snapshots))
    n.add("Load", "CA H2", bus="CA H2", p_set=pd.Series([0, 0, 0, 1.0], index=n.snapshots))

    def extra_functionality(n, sns):
        add_bidirectional_link_constraints(n)
        add_retrofit_constraints(n)

    status, _ = n.optimize(solver_name="highs", multi_investment_periods=True, extra_functionality=extra_functionality)
    assert status == "ok"

    stores, links = n.stores, n.links
    h2_store = stores.filter(like="H2 storage retrofitted", axis=0)
    assert h2_store.e_nom_opt.sum() > 0
    h2_pipe = links.loc["NV CA pipeline h2_pipeline_retrofitted_fwd_2030", "p_nom_opt"]
    assert links.loc[["CA NV pipeline h2_pipeline_retrofitted_fwd_2030"], "p_nom_opt"].sum() + h2_pipe > 0

    # gas left + hydrogen / factor is at most the existing capacity
    gas_store = stores.loc[h2_store.retrofit_from, "e_nom_opt"].to_numpy()
    total = gas_store + h2_store.e_nom_opt.to_numpy() / 0.2551
    assert (total <= np.array([1000, 500, 2000]) + 1e-6).all()
    gas_pipe = links.loc["NV CA pipeline", "p_nom_opt"]
    assert gas_pipe + h2_pipe / 0.6 <= 50 + 1e-6
    # gas capacity with a fixed cost and no use retires
    assert gas_pipe == pytest.approx(0, abs=1e-6)

    # a reservoir withdraws the share of the gas rate that it holds of the gas energy
    for reservoir, power, withdrawal, energy in (
        ("CA salt-dome", 0.8, 10, 1000),
        ("CA aquifer", 0.4, 5, 500),
        ("NV depleted-field", 0.4, 4, 2000),
    ):
        share = stores.at[f"{reservoir} H2 storage retrofitted_2030", "e_nom_opt"] / (0.2551 * energy)
        discharger = links.at[f"{reservoir} H2 storage retrofitted discharger_2030", "p_nom_opt"]
        assert discharger / (power * withdrawal) == pytest.approx(share, abs=1e-6)
        # hydrogen injects through the retrofitted wells, gas through the wells left to it
        charger = links.at[f"{reservoir} H2 storage retrofitted charger_2030", "p_nom_opt"]
        assert charger <= discharger + 1e-6
        gas_charge, gas_discharge = links.loc[
            [f"{reservoir} charge gas storage", f"{reservoir} discharge gas storage"],
            "p_nom_opt",
        ]
        assert gas_charge <= gas_discharge + 1e-6


def test_injection_through_the_wells_of_each_part():
    """Hydrogen and gas inject at most what their wells withdraw, at full use of the reservoir."""
    n = pypsa.Network()
    n.set_snapshots(pd.MultiIndex.from_product([[2030], pd.date_range("2030-01-01", periods=2, freq="h")]))
    n.set_investment_periods([2030])
    n.add("Bus", ["CA gas", "NV gas"], carrier="gas", STATE=["CA", "NV"])
    n.add("Bus", ["CA H2", "NV H2"], carrier="H2", STATE=["CA", "NV"], x=[-120.0, -118.0], y=36.0)
    add_gas_infrastructure(n, reservoirs=GAS_STORAGE[:1])
    add_retrofitted_storage(n, hydrogen_costs(2030), 2030, CENTERS)
    store = "CA salt-dome H2 storage retrofitted_2030"
    charger, discharger = (f"CA salt-dome H2 storage retrofitted {x}_2030" for x in ("charger", "discharger"))
    # retrofit half of the reservoir, and inject as much hydrogen and gas as possible
    n.stores.loc[store, ["e_nom_min", "e_nom_max"]] = 0.5 * 0.2551 * 1000
    n.links.loc[[charger, "CA salt-dome charge gas storage"], "capital_cost"] = -1
    # while gas withdrawal costs, so that it keeps only what the gas injection needs
    n.links.loc["CA salt-dome discharge gas storage", "capital_cost"] = 0.5

    def extra_functionality(n, sns):
        add_retrofit_constraints(n)

    status, _ = n.optimize(solver_name="highs", multi_investment_periods=True, extra_functionality=extra_functionality)
    assert status == "ok"

    links = n.links.p_nom_opt
    # half the wells withdraw hydrogen at 0.8 times the gas rate; new compressors inject as much
    assert links[discharger] == pytest.approx(0.5 * 0.8 * 10)
    assert links[charger] == pytest.approx(links[discharger])
    # the gas keeps its compressors, and withdrawal enough to inject through
    assert links["CA salt-dome charge gas storage"] == pytest.approx(3.7)
    assert links["CA salt-dome discharge gas storage"] == pytest.approx(3.7)


@pytest.mark.parametrize(
    ("setting", "retrofit", "hydrogen", "expected"),
    [
        ({}, {}, True, {"production": False, "pipeline": False, "storage": False}),
        ({"production": True, "storage": True}, {}, True, {"production": True, "pipeline": False, "storage": True}),
        # retrofitting contracts the gas storage and pipelines it takes from
        ({}, {"storage": True, "pipelines": True}, True, {"production": False, "pipeline": True, "storage": True}),
        # but only with hydrogen in the model
        ({}, {"storage": True, "pipelines": True}, False, {"production": False, "pipeline": False, "storage": False}),
    ],
)
def test_gas_infrastructure_contraction(setting, retrofit, hydrogen, expected):
    sector = {"natural_gas": {"infra_contraction": setting}, "hydrogen": {"retrofit": retrofit}}
    assert get_gas_infra_contraction(sector, hydrogen) == expected


def test_switches(network, demand, storage_sites):
    build(network, demand, storage_sites, OPTIONS | {"production": ["h2_smr"], "storage": False, "pipelines": False})

    assert set(network.links.carrier[network.links.carrier.str.startswith("h2")]) == {"h2_smr"}
    assert network.stores.carrier.str.startswith("h2_storage").sum() == 0


def test_coal_supply_is_priced_at_the_plants_fuel_cost(base_network):
    n = base_network
    n.buses.loc[["z1", "z2", "z3"], "STATE"] = ["CA", "NV", "NV"]
    n.add("Bus", ["CA coal", "NV coal", "AZ coal"], carrier="coal", STATE=["CA", "NV", "AZ"])
    n.add("Generator", ["g1", "g2", "g3"], bus=["z1", "z2", "z3"], carrier="coal", p_nom=[100, 100, 300])
    n.generators.loc[["g1", "g2", "g3"], "fuel_cost"] = [4.0, 2.0, 3.0]  # USD/MMBtu

    add_fuel_supply(n, "coal", get_plant_fuel_price(n, "coal", ["coal"]))

    supply = n.generators.loc[["CA coal supply", "NV coal supply", "AZ coal supply"]]
    national = (400 + 200 + 900) / 500
    assert supply.marginal_cost.tolist() == pytest.approx([4.0 * 3.412, 2.75 * 3.412, national * 3.412])
    assert (supply.carrier == "coal supply").all()


def test_coal_and_oil_supply_are_priced_at_the_aeo_fuel_costs_of_each_period():
    n = pypsa.Network()
    n.snapshots = get_multiindex_snapshots(
        sns_config={"start": "2030-01-01 00:00", "end": "2030-01-01 01:00", "inclusive": "both"},
        invest_periods=[2030, 2040],
    )
    n.set_investment_periods(periods=[2030, 2040])
    n.add("Bus", ["CA lpg", "NV lpg"], carrier="lpg", STATE=["CA", "NV"])
    costs = {
        2030: pd.DataFrame({"fuel_cost_real_per_mwhth": [10.0, 50.0]}, index=["coal", "oil"]),
        2040: pd.DataFrame({"fuel_cost_real_per_mwhth": [12.0, 60.0]}, index=["coal", "oil"]),
    }

    add_fuel_supply(n, "lpg", get_period_fuel_price(n, "lpg", costs, "oil"))

    marginal_cost = n.generators_t.marginal_cost[["CA lpg supply", "NV lpg supply"]]
    assert marginal_cost.loc[2030].to_numpy() == pytest.approx(50.0)
    assert marginal_cost.loc[2040].to_numpy() == pytest.approx(60.0)

    costs[2040].loc["oil", "fuel_cost_real_per_mwhth"] = np.nan
    with pytest.raises(ValueError, match="2040"):
        get_period_fuel_price(n, "lpg", costs, "oil")


def test_vintage_names_match_the_bidirectional_pairing():
    assert DIRECTION_SUFFIX.match("CA NV h2_pipeline_rev_2040")["base"] == "CA NV h2_pipeline"
    assert re.fullmatch(r".*_\d{4}", "z1 h2_smr_2040")
