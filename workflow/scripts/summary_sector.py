"""
Calcualtes summary statistics for sector coupling studies.
"""

import logging
from typing import Optional

import pandas as pd
import pypsa
from eia import Emissions

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
    Gets timeseries production to meet sectoral demand. Gets p1 supply.

    Note: can not use statistics module as multi-output links for co2 tracking
    > n.statistics.supply("Link", nice_names=False, aggregate_time=False).T
    """
    if include_storage:
        links = [x for x in n.links.index if f"{sector}-" in x]
    else:
        links = [
            x for x in n.links.index if f"{sector}-" in x and not x.endswith("charger")
        ]
    prod = n.links_t.p1[links].mul(-1)
    return prod.mul(n.snapshot_weightings.generators, axis=0)


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
