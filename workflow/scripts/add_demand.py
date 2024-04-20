"""
Adds demand to the network.

Depending on study, the load will all be aggregated to a single load
type, or distributed to different sectors and end use fuels.
"""

import logging
from pathlib import Path

import pandas as pd
import pypsa
from _helpers import configure_logging, mock_snakemake

logger = logging.getLogger(__name__)


def attach_demand(n: pypsa.Network, df: pd.DataFrame, load_name: str, carrier: str):
    """
    Add demand to network from specified configuration setting.

    Returns network with demand added.
    """
    df.index = pd.to_datetime(df.index)
    buses = df.columns
    df = df.rename({x: f"{x}{load_name}" for x in buses})
    n.madd(
        "Load",
        df.columns,
        bus=buses,
        p_set=df,
        carrier=carrier,
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake("add_demand", interconnect="western")
    configure_logging(snakemake)

    demand_files = snakemake.input.demand
    n = pypsa.Network(snakemake.input.network)

    if isinstance(demand_files, str):
        demand_files = [demand_files]

    load_name_mapper = {
        "power": "",
        "residential": "res",
        "commercial": "com",
        "industry": "ind",
        "transport": "trn",
    }

    load_carrier_mapper = {
        "electricity": "AC",
        "heating": "heat",
        "cooling": "cool",
    }

    for demand_file in demand_files:
        parsed_name = Path(demand_file).name.split("_")
        sector = parsed_name[0]
        end_use = parsed_name[1]

        df = pd.read_csv(demand_file, index_col=0)
        sector_name = load_name_mapper[sector]
        carrier_name = load_carrier_mapper[end_use]
        attach_demand(n, df, sector_name, carrier_name)
        logger.info(f"{sector} {end_use} demand added to network")

    n.export_to_netcdf(snakemake.output.network)
