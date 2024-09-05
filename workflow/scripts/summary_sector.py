"""
Calcualtes summary statistics for sector coupling studies.
"""

import logging
from typing import Optional

import pandas as pd
import pypsa
from eia import Emissions, Seds, TransportationDemand

logger = logging.getLogger(__name__)

###
# HELPERS
###


# def get_load_name_per_sector(sector: str) -> list[str]:
#     match sector:
#         case "res" | "com":
#             return ["elec", "urban-heat", "rural-heat", "cool"]
#         case "ind":
#             return ["elec", "heat"]
#         case "trn":
#             vehicles = ("lgt", "med", "hvy", "bus")
#             fuels = ("elec", "lpg")
#             return [f"{f}-{v}" for v in vehicles for f in fuels]
#         case _:
#             raise NotImplementedError


def _get_buses_in_state(n: pypsa.Network, state: str) -> list[str]:
    """
    Returns buses in a specified state.
    """
    return n.buses[n.buses.STATE == state].index.to_list()


def _get_loads_in_state(n: pypsa.Network, state: str) -> list[str]:
    """
    Returns buses in a specified state.
    """
    buses = _get_buses_in_state(n, state)
    return n.loads[n.loads.bus.isin(buses)].index.to_list()


def _get_links_in_state(n: pypsa.Network, state: str) -> list[str]:
    """
    Returns links in a specified state.
    """
    buses = _get_buses_in_state(n, state)
    return n.links[n.links.bus0.isin(buses) | n.links.bus1.isin(buses)].index.to_list()


def _get_gens_in_state(n: pypsa.Network, state: str) -> list[str]:
    """
    Returns buses in a specified state.
    """
    buses = _get_buses_in_state(n, state)
    return n.generators[n.generators.bus.isin(buses)].index.to_list()


def _get_stores_in_state(n: pypsa.Network, state: str) -> list[str]:
    """
    Returns buses in a specified state.
    """
    buses = _get_buses_in_state(n, state)
    return n.stores[n.stores.bus.isin(buses)].index.to_list()


def _filter_link_on_sector(n: pypsa.Network, sector: str) -> pd.DataFrame:
    """
    Filters network links to exclude dummy links.
    """
    match sector:
        case "res" | "res-urban" | "res-rural" | "com" | "com-urban" | "com-rural":
            return n.links[
                (n.links.carrier.str.startswith(sector))
                & ~(n.links.carrier.str.endswith("-store"))
                & ~(n.links.carrier.str.contains("-water"))  # hot water heaters
            ]
        case "ind":
            return n.links[
                (n.links.carrier.str.startswith(sector))
                & ~(n.links.carrier.str.endswith("-store"))
            ]
        case "trn":
            return n.links[
                (n.links.carrier.str.startswith(sector))
                & ~(n.links.carrier.str.endswith("-veh"))
                & ~(n.links.carrier.str.endswith("-boat"))
                & ~(n.links.carrier.str.endswith("-rail"))
                & ~(n.links.carrier.str.endswith("-air"))
            ]
        case _:
            raise NotImplementedError


def _filter_link_on_carrier(n: pypsa.Network, carriers: list[str]) -> pd.DataFrame:
    return n.links[n.links.carrier.isin(carriers)]


def _filter_gens_on_carrier(n: pypsa.Network, carriers: list[str]) -> pd.DataFrame:
    return n.generators[n.generators.carrier.isin(carriers)]


###
# GETTERS
###


def get_load_per_sector_per_fuel(n: pypsa.Network, sector: str, fuel: str, period: int):
    """
    Time series load per bus per fuel per sector.
    """
    loads = n.loads[
        (n.loads.carrier.str.startswith(sector)) & (n.loads.carrier.str.endswith(fuel))
    ]
    return n.loads_t.p[loads.index].loc[period]


def get_hp_cop(n: pypsa.Network, state: Optional[str] = None) -> pd.DataFrame:
    """
    Com and res hps have the same cop.
    """
    cops = n.links_t.efficiency

    if state:
        links = _get_links_in_state(n, state)
        cops = cops[[x for x in cops.columns if x in links]]

    ashp = cops[[x for x in cops.columns if x.endswith("ashp")]]
    gshp = cops[[x for x in cops.columns if x.endswith("gshp")]]

    return ashp.join(gshp)


def get_capacity_per_link_per_node(
    n: pypsa.Network,
    sector: str,
    include_elec: bool = False,
    group_existing: bool = True,
    state: Optional[str] = None,
) -> pd.Series:

    df = _filter_link_on_sector(n, sector)

    if not include_elec:
        df = df[~df.carrier.str.endswith("elec-infra")].copy()

    if state:
        links = _get_links_in_state(n, state)
        df = df[df.index.isin(links)]

    df = df[["carrier", "p_nom_opt"]]
    df["node"] = df.index.map(lambda x: x.split(f" {sector}-")[0])

    if group_existing:
        df["node"] = df.node.map(lambda x: x.split(" existing")[0])

    return df.reset_index(drop=True).groupby(["node", "carrier"]).sum().squeeze()


def get_total_capacity_per_node(
    n: pypsa.Network,
    sector: str,
    include_elec: bool = False,
    group_existing: bool = True,
    state: Optional[str] = None,
) -> pd.Series:

    df = _filter_link_on_sector(n, sector)

    if not include_elec:
        df = df[~df.carrier.str.endswith("elec-infra")].copy()

    if state:
        links = _get_links_in_state(n, state)
        df = df[df.index.isin(links)]

    df = df[["p_nom_opt"]]
    df["node"] = df.index.map(lambda x: x.split(f" {sector}-")[0])

    if group_existing:
        df["node"] = df.node.map(lambda x: x.split(" existing")[0])

    return df.reset_index(drop=True).groupby(["node"]).sum().squeeze()


def get_capacity_per_node(
    n: pypsa.Network,
    sector: str,
    include_elec: bool = False,
    group_existing: bool = True,
    state: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    total = get_total_capacity_per_node(n, sector, include_elec, group_existing, state)
    df = get_capacity_per_link_per_node(
        n,
        sector,
        include_elec,
        group_existing,
        state,
    ).to_frame()
    df["total"] = df.index.get_level_values("node").map(total)
    df["percentage"] = (df.p_nom_opt / df.total).round(4) * 100
    return df


def get_power_capacity_per_carrier(
    n: pypsa.Network,
    carriers: str | list[str],
    group_existing: bool = True,
    state: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:

    if isinstance(carriers, str):
        carriers = [carriers]

    links = _filter_link_on_carrier(n, carriers)
    gens = _filter_gens_on_carrier(n, carriers)

    if state:
        link_names = _get_links_in_state(n, state)
        links = links[links.index.isin(link_names)]
        gen_names = _get_gens_in_state(n, state)
        gens = gens[gens.index.isin(gen_names)]

    gens["node"] = gens["bus"]
    links["node"] = links["bus1"]

    cols = ["carrier", "p_nom_opt", "node"]

    df = pd.concat([links[cols], gens[cols]])

    return df.reset_index(drop=True).groupby(["node", "carrier"]).sum().squeeze()


def get_brownfield_capacity_per_state(
    n: pypsa.Network,
    sector: str,
    state: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:

    assert sector in (
        "res",
        "com",
        "ind",
        "trn",
        "res-rural",
        "res-urban",
        "com-rural",
        "com-urban",
    )

    df = get_capacity_per_node(n, sector, group_existing=False, state=state)

    data = []

    for carrier in df.index.get_level_values("carrier").unique():
        temp = df[df.index.get_level_values("carrier") == carrier].droplevel("carrier")
        temp["existing"] = temp.index.map(lambda x: True if "existing" in x else False)

        data.append(
            [
                carrier,
                temp[temp.existing == True].p_nom_opt.sum(),
                temp[temp.existing == False].p_nom_opt.sum(),
            ],
        )

    capacity = pd.DataFrame(data, columns=["carrier", "Existing", "New Build"])

    return capacity.set_index("carrier")


def get_sector_production_timeseries(
    n: pypsa.Network,
    sector: str,
    state: Optional[str] = None,
) -> pd.DataFrame:
    """
    Gets timeseries production to meet sectoral demand.

    Rememeber units! Transport will be in units of kVMT or similar.

    Note: can not use statistics module as multi-output links for co2 tracking
    > n.statistics.supply("Link", nice_names=False, aggregate_time=False).T
    """

    links = _filter_link_on_sector(n, sector).index.to_list()
    df = n.links_t.p1[links].mul(-1).mul(n.snapshot_weightings.generators, axis=0)

    if state:
        links = _get_links_in_state(n, state)
        return df[[x for x in df.columns if x in links]]
    else:
        return df


def get_sector_production_timeseries_by_carrier(
    n: pypsa.Network,
    sector: str,
    state: Optional[str] = None,
) -> pd.DataFrame:
    """
    Gets timeseries production by carrier.
    """

    df = get_sector_production_timeseries(n, sector, state=state).T
    df.index = df.index.map(n.links.carrier)
    return df.groupby(level=0).sum().T


def get_sector_max_production_timeseries(
    n: pypsa.Network,
    sector: str,
    state: Optional[str] = None,
) -> pd.DataFrame:
    """
    Max production timeseries at a carrier level.
    """
    eff = n.get_switchable_as_dense("Link", "efficiency")

    if state:
        links_in_state = _get_links_in_state(n, state)
        eff = eff[[x for x in eff.columns if x in links_in_state]]

    links_in_sector = _filter_link_on_sector(n, sector).index.to_list()
    links_in_sector_state = [x for x in links_in_sector if x in eff.columns]
    eff = eff[links_in_sector_state]

    cap = n.links.loc[eff.columns].p_nom_opt
    return eff.mul(cap).mul(n.snapshot_weightings.generators, axis=0)


def get_load_factor_timeseries(
    n: pypsa.Network,
    sector: str,
    include_elec: bool = False,
    state: Optional[str] = None,
) -> pd.DataFrame:

    max_prod = get_sector_max_production_timeseries(n, sector, state=state)
    act_prod = get_sector_production_timeseries(n, sector, state=state)

    max_prod = (
        max_prod.rename(columns={x: x.split(f"{sector}-")[1] for x in max_prod.columns})
        .T.groupby(level=0)
        .sum()
        .T
    )
    act_prod = (
        act_prod.rename(columns={x: x.split(f"{sector}-")[1] for x in act_prod.columns})
        .T.groupby(level=0)
        .sum()
        .T
    )

    lf = act_prod.div(max_prod).mul(100).round(3)

    if include_elec:
        return lf
    else:
        return lf[[x for x in lf.columns if not x.endswith("-infra")]]


def get_emission_timeseries_by_sector(
    n: pypsa.Network,
    sector: Optional[str] = None,
    state: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Cummulative emissions by sector in MT.
    """
    if sector:
        stores_in_sector = [x for x in n.stores.index if f"{sector}-co2" in x]
    else:
        stores_in_sector = [x for x in n.stores.index if f"-co2" in x]
    emissions = n.stores_t.e[stores_in_sector].mul(1e-6)

    if state:
        stores_in_state = _get_stores_in_state(n, state)
        return emissions[[x for x in emissions.columns if x in stores_in_state]]
    else:
        return emissions


def get_historical_emissions(
    sectors: str | list[str],
    year: int,
    api: str,
) -> pd.DataFrame:
    """
    Emissions by state/sector in units of million metric tons.
    """

    dfs = []

    if isinstance(sectors, str):
        sectors = [sectors]

    for sector in sectors:
        dfs.append(
            Emissions(sector, year, api)
            .get_data(pivot=True)
            .rename(index={f"{year}": f"{sector}"}),
        )

    df = pd.concat(dfs)
    df.index.name = "sector"
    return df


def get_historical_end_use_consumption(
    sectors: str | list[str],
    year: int,
    api: str,
) -> pd.DataFrame:
    """
    End-Use consumption by state/sector in units of MWh.
    """

    dfs = []

    if isinstance(sectors, str):
        sectors = [sectors]

    for sector in sectors:
        dfs.append(
            Seds("consumption", sector, year, api)
            .get_data(pivot=True)
            .rename(index={f"{year}": f"{sector}"}),
        )

    df = pd.concat(dfs)
    df.index.name = "sector"

    # convert billion BTU to MWH
    return df.mul(293.07)


def get_end_use_consumption(
    n: pypsa.Network,
    sector: str,
    state: Optional[str] = None,
) -> pd.DataFrame:
    """
    Gets timeseries energy consumption in MWh.

    - Will get "p_set" load for "res", "com", "ind"
    - WIll get "p0" link for "trn"
    """

    def get_service_consumption(
        n: pypsa.Network,
        sector: str,
        state: Optional[str] = None,
    ) -> pd.DataFrame:
        assert sector in ("res", "com", "ind")
        loads = n.loads[n.loads.carrier.str.startswith(sector)]
        if state:
            l = _get_loads_in_state(n, state)
            loads = loads[loads.index.isin(l)]
        df = n.loads_t.p[loads.index].mul(n.snapshot_weightings["objective"], axis=0).T
        df.index = df.index.map(n.loads.carrier).map(lambda x: x.split("-")[1:])
        df.index = df.index.map(lambda x: "-".join(x))
        return df.groupby(level=0).sum().T

    def get_transport_consumption(
        n: pypsa.Network,
        state: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Takes load from p0 link as loads are in kVMT or similar.
        """
        loads = n.links[
            (n.links.carrier.str.startswith("trn-"))
            & ~(n.links.carrier.str.endswith("-veh"))
            & ~(n.links.carrier == ("trn-air"))
            & ~(n.links.carrier == ("trn-rail"))
            & ~(n.links.carrier == ("trn-boat"))
        ]
        if state:
            l = _get_links_in_state(n, state)
            loads = loads[loads.index.isin(l)]
        df = n.links_t.p0[loads.index].mul(n.snapshot_weightings["objective"], axis=0).T
        df.index = df.index.map(lambda x: x.split("trn-")[1])
        return df.groupby(level=0).sum().T

    match sector:
        case "res" | "com" | "ind":
            return get_service_consumption(n, sector, state)
        case "trn":
            return get_transport_consumption(n, state)
        case _:
            raise NotImplementedError


def get_end_use_load_timeseries(
    n: pypsa.Network,
    sector: str,
    sns_weight: bool = True,
    state: Optional[str] = None,
) -> pd.DataFrame:
    """
    Gets timeseries load per node.

    - Residential, Commercial, Industrial are in untis of MWh
    """

    assert sector in ("res", "com", "ind")

    loads = n.loads[n.loads.carrier.str.startswith(sector)]

    if state:
        loads_in_state = _get_loads_in_state(n, state)
        loads = loads[loads.index.isin(loads_in_state)]

    df = n.loads_t.p[loads.index]
    if sns_weight:
        df = df.mul(n.snapshot_weightings["objective"], axis=0)
    return df.T


def get_end_use_load_timeseries_carrier(
    n: pypsa.Network,
    sector: str,
    sns_weight: bool = True,
    state: Optional[str] = None,
) -> pd.DataFrame:
    """
    Gets timeseries load per node per carrier.

    - Residential, Commercial, Industrial are in untis of MWh
    """

    df = get_end_use_load_timeseries(n, sector, sns_weight).T
    if state:
        buses = _get_loads_in_state(n, state)
        df = df[[x for x in df.columns if x in buses]]
    df = df.T
    df.index = df.index.map(n.loads.carrier).map(lambda x: x.split("-")[1:])
    df.index = df.index.map(lambda x: "-".join(x))
    return df.groupby(level=0).sum().T


def get_transport_consumption_by_mode(
    n: pypsa.Network,
    state: Optional[str] = None,
) -> pd.DataFrame:
    df = get_end_use_consumption(n, "trn", state)
    df = df.rename(columns={x: "-".join(x.split("-")[1:]) for x in df.columns})
    df = df.T.groupby(level=0).sum().T
    return df


def get_historical_transport_consumption_by_mode(api: str) -> pd.DataFrame:
    """
    Will return data in units of MWh.
    """

    vehicles = [
        "light_duty",
        "med_duty",
        "heavy_duty",
        "bus",
        "rail_passenger",
        "boat_shipping",
        "rail_shipping",
        "air",
        "boat_international",
        "boat_recreational",
        "military",
        "lubricants",
        "pipeline",
    ]

    dfs = []
    for vehicle in vehicles:
        dfs.append(TransportationDemand(vehicle, 2020, api, units="btu").get_data())
    df = pd.concat(dfs)
    df = df.set_index("series-description")["value"]
    return df.mul(1e9).mul(0.29307)  # quads -> mmbtu -> MWh
