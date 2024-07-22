"""
Calcualtes summary statistics for sector coupling studies.
"""

import logging
from typing import Optional

import pandas as pd
import pypsa
from eia import Emissions, Seds

logger = logging.getLogger(__name__)

###
# HELPERS
###


def get_load_name_per_sector(sector: str) -> list[str]:
    match sector:
        case "res" | "com":
            return ["elec", "urban-heat", "rural-heat", "cool"]
        case "ind":
            return ["elec", "heat"]
        case "trn":
            vehicles = ("lgt", "med", "hvy", "bus")
            fuels = ("elec", "lpg")
            return [f"{f}-{v}" for v in vehicles for f in fuels]
        case _:
            raise NotImplementedError


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


def get_hp_cop(n: pypsa.Network) -> pd.DataFrame:
    """
    Com and res hps have the same cop.
    """
    cops = n.links_t.efficiency
    ashp = cops[[x for x in cops.columns if x.endswith("res-urban-ashp")]]
    ashp = ashp.rename(
        columns={x: x.replace("res-urban-ashp", "ashp") for x in ashp.columns},
    )
    gshp = cops[[x for x in cops.columns if x.endswith("res-rural-gshp")]]
    gshp = gshp.rename(
        columns={x: x.replace("res-rural-gshp", "gshp") for x in gshp.columns},
    )
    return ashp.join(gshp)


def get_capacity_per_link_per_node(
    n: pypsa.Network,
    sector: str,
    include_elec: bool = False,
) -> pd.Series:
    if include_elec:
        df = n.links[
            (n.links.carrier.str.startswith(sector))
            & ~(n.links.carrier.str.endswith("-store"))
        ]
    else:
        df = n.links[
            (n.links.carrier.str.startswith(sector))
            & ~(n.links.carrier.str.endswith("elec-infra"))
            & ~(n.links.carrier.str.endswith("-store"))
        ]
    df = df[["carrier", "p_nom_opt"]]
    df["node"] = df.index.map(lambda x: x.split(f" {sector}-")[0])
    df["carrier"] = df.carrier.map(lambda x: x.split(f"{sector}-")[1])
    return df.reset_index(drop=True).groupby(["node", "carrier"]).sum().squeeze()


def get_total_capacity_per_node(
    n: pypsa.Network,
    sector: str,
    include_elec: bool = False,
) -> pd.Series:
    if include_elec:
        df = n.links[
            (n.links.carrier.str.startswith(sector))
            & ~(n.links.carrier.str.endswith("-store"))
        ]
    else:
        df = n.links[
            (n.links.carrier.str.startswith(sector))
            & ~(n.links.carrier.str.endswith("elec-infra"))
            & ~(n.links.carrier.str.endswith("-store"))
        ]
    df = df[["p_nom_opt"]]
    df["node"] = df.index.map(lambda x: x.split(f" {sector}-")[0])
    return df.reset_index(drop=True).groupby(["node"]).sum().squeeze()


def get_capacity_per_node(
    n: pypsa.Network,
    sector: str,
    include_elec: bool = False,
) -> pd.DataFrame:
    total = get_total_capacity_per_node(n, sector, include_elec)
    df = get_capacity_per_link_per_node(n, sector, include_elec).to_frame()
    df["total"] = df.index.get_level_values("node").map(total)
    df["percentage"] = (df.p_nom_opt / df.total).round(4) * 100
    return df


def get_sector_production_timeseries(
    n: pypsa.Network,
    sector: str,
    include_storage: bool = False,
) -> pd.DataFrame:
    """
    Gets timeseries production to meet sectoral demand.

    Gets p1 supply for service and industry techs. Gets p0 withdrawl for
    transport production (as p1 will be in units of kVMT).

    Note: can not use statistics module as multi-output links for co2 tracking
    > n.statistics.supply("Link", nice_names=False, aggregate_time=False).T
    """

    def get_service_production_timeseries(
        n: pypsa.Network,
        sector: str,
    ) -> pd.DataFrame:
        assert sector in ("res", "com", "ind")
        if include_storage:
            links = [x for x in n.links.index if f"{sector}-" in x]
        else:
            links = [
                x
                for x in n.links.index
                if f"{sector}-" in x and not x.endswith("charger")
            ]
        return n.links_t.p1[links].mul(-1).mul(n.snapshot_weightings.generators, axis=0)

    def get_transport_production_timeseries(n: pypsa.Network) -> pd.DataFrame:
        """
        Takes load from p0 link as loads are in kVMT.
        """
        if include_storage:
            links = n.links[
                (n.links.carrier.str.startswith("trn-"))
                & ~(n.links.index.str.endswith("infra"))
            ].index.to_list()
        else:
            links = n.links[
                (n.links.carrier.str.startswith("trn-"))
                & ~(n.links.index.str.endswith("infra"))
                & ~(n.links.index.str.endswith("charger"))
            ].index.to_list()

        return n.links_t.p0[links].mul(n.snapshot_weightings["objective"], axis=0)

    match sector:
        case "res" | "com" | "ind":
            return get_service_production_timeseries(n, sector)
        case "trn":
            return get_transport_production_timeseries(n)
        case _:
            raise NotImplementedError


def get_sector_production_timeseries_by_carrier(
    n: pypsa.Network,
    sector: str,
) -> pd.DataFrame:
    """
    Gets timeseries production by carrier.
    """

    df = get_sector_production_timeseries(n, sector).T
    df.index = df.index.map(n.links.carrier)
    return df.groupby(level=0).sum().T


def get_sector_max_production_timeseries(n: pypsa.Network, sector: str) -> pd.DataFrame:
    """
    Max production timeseries at a carrier level.
    """
    eff = n.get_switchable_as_dense("Link", "efficiency")
    links = [x for x in eff.columns if f"{sector}-" in x and not x.endswith("charger")]
    eff = eff[links]
    cap = n.links.loc[links].p_nom_opt
    return eff.mul(cap).mul(n.snapshot_weightings.generators, axis=0)


def get_load_factor_timeseries(
    n: pypsa.Network,
    sector: str,
    include_elec: bool = False,
) -> pd.DataFrame:
    max_prod = get_sector_max_production_timeseries(n, sector)
    act_prod = get_sector_production_timeseries(n, sector)

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
    **kwargs,
) -> pd.DataFrame:
    """
    Cummulative emissions by sector in MT.
    """
    if sector:
        stores = [x for x in n.stores.index if f"{sector}-co2" in x]
    else:
        stores = [x for x in n.stores.index if f"-co2" in x]
    return n.stores_t.e[stores].mul(1e-6)


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


def get_end_use_consumption(n: pypsa.Network, sector: str, year: int) -> pd.DataFrame:
    """
    Gets timeseries energy consumption in MWh.
    """

    def get_service_consumption(n: pypsa.Network, sector: str) -> pd.DataFrame:
        assert sector in ("res", "com", "ind")
        loads = n.loads[n.loads.carrier.str.startswith(sector)]
        df = n.loads_t.p[loads.index].mul(n.snapshot_weightings["objective"], axis=0).T
        df.index = df.index.map(n.loads.carrier).map(lambda x: x.split("-")[1:])
        df.index = df.index.map(lambda x: "-".join(x))
        return df.groupby(level=0).sum().T

    def get_transport_consumption(n: pypsa.Network) -> pd.DataFrame:
        """
        Takes load from p0 link as loads are in kVMT.
        """
        loads = n.links[
            (n.links.carrier.str.startswith("trn-"))
            & ~(n.links.index.str.endswith("infra"))
        ].index.to_list()
        df = n.links_t.p0[loads].mul(n.snapshot_weightings["objective"], axis=0).T
        df.index = df.index.map(n.loads.carrier).map(lambda x: x.split("-")[1:])
        df.index = df.index.map(lambda x: "-".join(x))
        return df.groupby(level=0).sum().T

    match sector:
        case "res" | "com" | "ind":
            df = get_service_consumption(n, sector)
        case "trn":
            df = get_transport_consumption(n)
        case _:
            raise NotImplementedError

    return df


def get_end_use_load_timeseries(
    n: pypsa.Network,
    sector: str,
    sns_weight: bool = True,
) -> pd.DataFrame:
    """
    Gets timeseries load per node.

    - Residential, Commercial, Industrial are in untis of MWh
    - Transport is in units of kVMT
    """

    loads = n.loads[n.loads.carrier.str.startswith(sector)]
    df = n.loads_t.p[loads.index]
    if sns_weight:
        df = df.mul(n.snapshot_weightings["objective"], axis=0)
    return df.T


def get_end_use_load_timeseries_carrier(
    n: pypsa.Network,
    sector: str,
    sns_weight: bool = True,
) -> pd.DataFrame:
    """
    Gets timeseries load per node per carrier.

    - Residential, Commercial, Industrial are in untis of MWh
    - Transport is in units of kVMT
    """

    df = get_end_use_load_timeseries(n, sector, sns_weight)
    df.index = df.index.map(n.loads.carrier).map(lambda x: x.split("-")[1:])
    df.index = df.index.map(lambda x: "-".join(x))
    return df.groupby(level=0).sum().T
