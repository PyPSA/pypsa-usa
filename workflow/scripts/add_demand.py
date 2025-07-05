"""
Adds demand to the network.

Depending on study, the load will all be aggregated to a single load
type, or distributed to different sectors and end use fuels.
"""

import logging
from pathlib import Path

import pandas as pd
import pypsa
from _helpers import configure_logging, get_multiindex_snapshots, mock_snakemake
from constants_sector import (
    TRANSPORT_FUELS,
    SecCarriers,
    SecNames,
)

logger = logging.getLogger(__name__)


def attach_demand(n: pypsa.Network, df: pd.DataFrame, carrier: str, suffix: str):
    """
    Add demand to network from specified configuration setting.

    Returns network with demand added.
    """
    df.index = pd.to_datetime(df.index)
    assert len(df.index) == len(
        n.snapshots,
    ), "Demand time series length does not match network snapshots"
    df.index = n.snapshots
    n.madd(
        "Load",
        df.columns,
        suffix=suffix,
        bus=df.columns,
        p_set=df,
        carrier=carrier,
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake("add_demand", interconnect="western")
    configure_logging(snakemake)

    demand_files = snakemake.input.demand
    n = pypsa.Network(snakemake.input.network)

    sectors = snakemake.params.sectors

    # add snapshots
    sns_config = snakemake.params.snapshots
    planning_horizons = snakemake.params.planning_horizons

    n.snapshots = get_multiindex_snapshots(sns_config, planning_horizons)
    n.set_investment_periods(periods=planning_horizons)

    if isinstance(demand_files, str):
        demand_files = [demand_files]

    if sectors == "E" or sectors == "":  # electricity only
        assert len(demand_files) == 1

        suffix = ""
        carrier = "AC"

        df = pd.read_csv(demand_files[0], index_col=0)
        attach_demand(n, df, carrier, suffix)
        logger.info("Electricity demand added to network")

    else:  # sector files
        for demand_file in demand_files:
            parsed_name = Path(demand_file).name.split("_")
            parsed_name[-1] = parsed_name[-1].split(".pkl")[0]

            if len(parsed_name) == 2:
                sector = parsed_name[0].upper()
                end_use = parsed_name[1].upper().replace("-", "_")

                sec_name = SecNames[sector].value
                sec_car = SecCarriers[end_use].value

                if sector == "transport":
                    carrier = f"{sec_name}-{sec_car}-{TRANSPORT_FUELS[end_use]}"
                else:
                    carrier = f"{sec_name}-{sec_car}"

                log_statement = f"{sector} {end_use} demand added to network"

            else:
                raise NotImplementedError

            suffix = f"-{carrier}"

            df = pd.read_pickle(demand_file)
            attach_demand(n, df, carrier, suffix)
            logger.info(log_statement)

    n.export_to_netcdf(snakemake.output.network)
