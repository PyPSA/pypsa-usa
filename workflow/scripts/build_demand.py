# PyPSA USA Authors
"""
Builds the demand data for the PyPSA network. This module is responsible for cleaning and transforming electricity demand data from the NREL Electrification Futures Study, EIA, and GridEmissions to be used in the `add_electricity` module. 

**Relevant Settings**

.. code:: yaml

    network_configuration:

    snapshots:
        start:
        end:
        inclusive:

    scenario:
    interconnect:
    planning_horizons:


**Inputs**

    - base_network:  
    - ads_renewables:
    - ads_2032:
    - eia: (GridEmissions data file)
    - efs: (NREL EFS Load Forecasts)

**Outputs**

    - demand: Path to the demand CSV file.
"""


import logging
import os
import random
from itertools import product
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Union

import constants as const
import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from _helpers import configure_logging
from _helpers import export_network_for_gis_mapping
from _helpers import local_to_utc
from _helpers import test_network_datatype_consistency
from _helpers import update_p_nom_max
from scipy import sparse
from shapely.geometry import Point
from shapely.prepared import prep
from sklearn.neighbors import BallTree

idx = pd.IndexSlice

logger = logging.getLogger(__name__)


def prepare_ads_demand(
    n: pypsa.Network,
    demand_path: str,
) -> pd.DataFrame:
    demand = pd.read_csv(demand_path, index_col=0)
    data_year = 2032
    demand.columns = demand.columns.str.removeprefix("Load_")
    demand.columns = demand.columns.str.removesuffix(".dat")
    demand.columns = demand.columns.str.removesuffix(f"_{data_year}")
    demand.columns = demand.columns.str.removesuffix(f"_[18].dat: {data_year}")
    demand["CISO-PGAE"] = (
        demand.pop("CIPV") + demand.pop("CIPB") + demand.pop("SPPC")
    )  # TODO: #37 Create new Zone for SPPC
    demand["BPAT"] = demand.pop("BPAT") + demand.pop("TPWR") + demand.pop("SCL")
    demand["IPCO"] = demand.pop("IPFE") + demand.pop("IPMV") + demand.pop("IPTV")
    demand["PACW"] = demand.pop("PAID") + demand.pop("PAUT") + demand.pop("PAWY")
    demand["Arizona"] = demand.pop("SRP") + demand.pop("AZPS")
    demand.drop(columns=["Unnamed: 44", "TH_Malin", "TH_Mead", "TH_PV"], inplace=True)
    ba_list_map = {
        "CISC": "CISO-SCE",
        "CISD": "CISO-SDGE",
        "VEA": "CISO-VEA",
        "WAUW": "WAUW_SPP",
    }
    demand.rename(columns=ba_list_map, inplace=True)
    demand.set_index("Index", inplace=True)

    demand.index = n.snapshots
    n.buses["load_dissag"] = n.buses.balancing_area.replace({"": "missing_ba"})
    intersection = set(demand.columns).intersection(n.buses.ba_load_data.unique())
    demand = demand[list(intersection)]
    set_load_allocation_factor(n)
    return disaggregate_demand_to_buses(n, demand)


def prepare_eia_demand(
    n: pypsa.Network,
    demand_path: str,
) -> pd.DataFrame:
    logger.info("Building Load Data using EIA demand")
    demand = pd.read_csv(demand_path, index_col=0)
    demand.index = pd.to_datetime(demand.index)
    demand = demand.loc[n.snapshots.intersection(demand.index)]  # filter by snapshots
    demand = demand[~demand.index.duplicated(keep="last")]
    demand.index = n.snapshots

    # Combine EIA Demand Data to Match GIS Shapes
    demand["Arizona"] = demand.pop("SRP") + demand.pop("AZPS")
    n.buses["load_dissag"] = n.buses.balancing_area.replace(
        {"^CISO.*": "CISO", "^ERCO.*": "ERCO"},
        regex=True,
    )

    n.buses["load_dissag"] = n.buses.load_dissag.replace({"": "missing_ba"})
    intersection = set(demand.columns).intersection(n.buses.load_dissag.unique())
    demand = demand[list(intersection)]

    set_load_allocation_factor(n)
    return disaggregate_demand_to_buses(n, demand)

def prepare_efs_demand(
    n: pypsa.Network,
    planning_horizons: list[str],
) -> pd.DataFrame:
    logger.info("Building Load Data using EFS demand")

    demand = pd.read_csv(snakemake.input.efs)
    demand = demand.loc[demand.Year == planning_horizons[0]]
    demand.drop(
        columns=[
            "Electrification",
            "TechnologyAdvancement",
            "Sector",
            "Subsector",
            "Year",
        ],
        inplace=True,
    )
    # TODO: We are throwing out great data here on the sector and subsector loads. Revisit this.

    demand = demand.groupby(["State", "LocalHourID"]).sum().reset_index()

    demand["DateTime"] = pd.Timestamp(
        year=planning_horizons[0],
        month=1,
        day=1,
    ) + pd.to_timedelta(demand["LocalHourID"] - 1, unit="H")
    demand["UTC_Time"] = demand.groupby(["State"])["DateTime"].transform(local_to_utc)
    demand.drop(columns=["LocalHourID", "DateTime"], inplace=True)
    demand.set_index("UTC_Time", inplace=True)

    demand = demand.pivot(columns="State", values="LoadMW")
    n.buses["load_dissag"] = n.buses.state
    intersection = set(demand.columns).intersection(
        [const.STATE_2_CODE.get(item, item) for item in n.buses.load_dissag.unique()],
    )
    demand = demand[list(intersection)]
    demand.columns = [
        {v: k for k, v in const.STATE_2_CODE.items()}.get(item, item)
        for item in demand.columns
    ]

    #This block is use to align the demand data hours with the snapshot hours
    hoy = (demand.index.dayofyear - 1) * 24 + demand.index.hour
    demand.index = hoy
    demand_new = pd.DataFrame(columns=demand.columns)
    for column in demand.columns:
        col = demand[column].reset_index()
        demand_new[column] = col.groupby('UTC_Time').apply(lambda group: group.loc[group.drop(columns='UTC_Time').first_valid_index()]).drop(columns='UTC_Time')

    demand_new.index = n.snapshots
    n.buses.rename(columns={"LAF_states": "LAF"}, inplace=True)
    return disaggregate_demand_to_buses(n, demand_new)


def set_load_allocation_factor(n: pypsa.Network) -> pd.DataFrame:
    """
    Defines Load allocation factor for each bus according to load_dissag for
    balancing areas.
    """
    n.buses.Pd = n.buses.Pd.fillna(0)
    group_sums = n.buses.groupby("load_dissag")["Pd"].transform("sum")
    n.buses["LAF"] = n.buses["Pd"] / group_sums


def disaggregate_demand_to_buses(
    n: pypsa.Network,
    demand: pd.DataFrame,
) -> pd.DataFrame:
    """
    Zone power demand is disaggregated to buses proportional to Pd.
    """
    demand_aligned = demand.reindex(
        columns=n.buses["load_dissag"].unique(),
        fill_value=0,
    )
    bus_demand = pd.DataFrame()
    for load_dissag in n.buses["load_dissag"].unique():
        LAF = n.buses.loc[n.buses["load_dissag"] == load_dissag, "LAF"]
        zone_bus_demand = (
            demand_aligned[load_dissag].values.reshape(-1, 1) * LAF.values.T
        )
        bus_demand = pd.concat(
            [bus_demand, pd.DataFrame(zone_bus_demand, columns=LAF.index)],
            axis=1,
        )
    bus_demand.index = n.snapshots
    n.buses.drop(columns=["LAF"], inplace=True)
    return bus_demand.fillna(0)


def main(snakemake):
    params = snakemake.params
    configuration = snakemake.config["network_configuration"]
    interconnection = snakemake.wildcards["interconnect"]
    planning_horizons = snakemake.params["planning_horizons"]


    snapshot_config = snakemake.params["snapshots"]
    sns_start = pd.to_datetime(snapshot_config["start"]) #+ " 08:00:00")
    sns_end = pd.to_datetime(snapshot_config["end"]) #+ " 06:00:00")
    sns_inclusive = snapshot_config["inclusive"]

    n = pypsa.Network(snakemake.input.base_network)

    n.set_snapshots(
        pd.date_range(
            freq="h",
            start=sns_start,
            end=sns_end,
            inclusive=sns_inclusive,
        ),
    )
    Nyears = n.snapshot_weightings.objective.sum() / 8760.0

    if configuration == "ads":
        demand_per_bus = prepare_ads_demand(n, "data/WECC_ADS/processed/load_2032.csv")
    elif configuration == "pypsa-usa":
        demand_per_bus = prepare_efs_demand(n, snakemake.params.get("planning_horizons")) if snakemake.params.get("planning_horizons") else prepare_eia_demand(n, snakemake.input["eia"][0])
    else:
        raise ValueError("Invalid demand_type. Supported values are 'ads', and 'pypsa-usa'.")

    demand_per_bus.to_csv(snakemake.output.demand, index=True)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_demand", interconnect="western")
    configure_logging(snakemake)
    main(snakemake)
