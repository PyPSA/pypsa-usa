# By PyPSA-USA Authors
"""
**Description**

Build_fuel_prices.py is a script that prepares data for dynamic fuel prices to be used in the `add_electricity` module. Data is input from `retrieve_caiso_data` and `retrieve_eia_data` to create a combined dataframe with all dynamic fuel prices available. The prices are modified to be on an hourly basis to match the network snapshots, and converted to $/MWh_thermal. The output is a CSV file containing the hourly fuel prices for each Balancing Authority and State.

**Relevant Settings**

.. code:: yaml

    fuel_year:
    snapshots:

**Inputs**

- ''data/caiso_ng_prices.csv'': A CSV file containing the daily average fuel prices for each Balancing Authority in the WEIM.
- ''data/eia_ng_prices.csv'': A CSV file containing the monthly average fuel prices for each State.

**Outputs**

- ''data/ng_fuel_prices.csv'': A CSV file containing the hourly fuel prices for each Balancing Authority and State.
"""

import logging
import sys
from pathlib import Path
from typing import List

import constants as const
import eia
import pandas as pd
from _helpers import configure_logging, mock_snakemake

logger = logging.getLogger(__name__)


###
# Helper functions
###


def make_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Makes the index hourly.
    """

    start = df.index.min()
    end = (
        pd.to_datetime(start)
        .to_period("Y")
        .to_timestamp("Y")
        .to_period("Y")
        .to_timestamp("Y")
        + pd.offsets.MonthEnd(0)
        + pd.Timedelta(hours=23)
    )
    hourly_df = pd.DataFrame(
        index=pd.date_range(start=start, end=end + pd.Timedelta(days=1), freq="h"),
    )
    return (
        hourly_df.merge(df, how="left", left_index=True, right_index=True)
        .ffill()
        .bfill()
    )


###
# Eia state level data
###


def get_state_ng_power_prices(sns: pd.date_range, eia_api: str) -> pd.DataFrame:
    df = (
        eia.FuelCosts("gas", "power", sns.year[0], eia_api).get_data(pivot=True)
        * 1000
        / const.NG_MWH_2_MMCF
    )  # $/MCF -> $/MWh
    return make_hourly(df)


def get_state_coal_power_prices(sns: pd.date_range, eia_api: str) -> pd.DataFrame:
    eia_coal = (
        eia.FuelCosts("coal", "power", sns.year[0], eia_api).get_data(pivot=True)
        * const.COAL_dol_ton_2_MWHthermal
    )
    return make_hourly(eia_coal)


# note, new functions to add must include the **kwargs argument

###
# Gas BA level data
###


def prepare_caiso_ng_power_prices(
    caiso_fn: str,
    sns: pd.DatetimeIndex = None,
) -> pd.DataFrame:
    caiso_ng = pd.read_csv(caiso_fn)
    caiso_ng["PRC"] = caiso_ng["PRC"] * const.NG_Dol_MMBTU_2_MWH
    caiso_ng = caiso_ng.rename(
        columns={"PRC": "dol_mwh_th", "Balancing Authority": "balancing_area"},
    )
    year = sns[0].year
    caiso_ng.day_of_year = pd.to_datetime(caiso_ng.day_of_year, format="%j").map(
        lambda dt: dt.replace(year=year),
    )
    return caiso_ng.rename(columns={"day_of_year": "period"})


def get_caiso_ng_power_prices(
    sns: pd.date_range,
    filepath: str,
    **kwargs,
) -> pd.DataFrame:

    # pypsa-usa name: caiso name
    ba_mapper = {
        "CISO-PGAE": "CISO",
        "CISO-SCE": "CISO",
        "CISO-SDGE": "CISO",
        "CISO-VEA": "CISO",
        "Arizona": "AZPS",
        # "NYISO": "NYISO",
        # "CAISO": "CAISO",
        "BANC": "BANCSMUD",
    }

    df = prepare_caiso_ng_power_prices(filepath, sns)
    df = df.pivot(
        index="period",
        columns="balancing_area",
        values="dol_mwh_th",
    )

    for pypsa_usa_name, caiso_name in ba_mapper.items():
        df[pypsa_usa_name] = df[caiso_name]

    return make_hourly(df)


###
# Coal BA level data
###


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_fuel_prices", interconnect="western")
    configure_logging(snakemake)

    snapshot_config = snakemake.config["snapshots"]
    sns_start = pd.to_datetime(snapshot_config["start"])
    sns_end = pd.to_datetime(snapshot_config["end"])
    sns_inclusive = snapshot_config["inclusive"]

    eia_api = snakemake.params.api_eia

    snapshots = pd.date_range(
        freq="h",
        start=sns_start,
        end=sns_end,
        inclusive=sns_inclusive,
    )

    function_mapper = {
        "caiso_ng_power_prices": get_caiso_ng_power_prices,
    }

    # state level prices are always attempted
    if not eia_api:
        state_ng_power_prices = pd.DataFrame(index=snapshots)
        state_coal_power_prices = pd.DataFrame(index=snapshots)
    else:
        state_ng_power_prices = get_state_ng_power_prices(snapshots, eia_api)
        state_coal_power_prices = get_state_coal_power_prices(snapshots, eia_api)

    # get any regional specific prices
    # only gas level ba implemented right now, but can be replicated for any
    # energy carrier and any geography (states, reeds, ect.)
    if not snakemake.input.gas_balancing_area:
        ba_ng_power_prices = pd.DataFrame(index=snapshots)
    else:
        ba_data = []
        for filepath in snakemake.input.gas_balancing_area:
            file_name = Path(filepath).stem

            assert (
                file_name in function_mapper
            ), f"Can not find {file_name} in dynamic fuel price mapper"

            ba_data.append(function_mapper[file_name](filepath=filepath, sns=snapshots))
        ba_ng_power_prices = pd.concat(ba_data, axis=1)

    # filter all data on snapshots and return
    state_ng_power_prices = state_ng_power_prices.loc[snapshots]
    state_coal_power_prices = state_coal_power_prices.loc[snapshots]
    ba_ng_power_prices = ba_ng_power_prices.loc[snapshots]

    state_ng_power_prices.to_csv(
        snakemake.output.state_ng_fuel_prices,
        index_label="snapshot",
    )
    state_coal_power_prices.to_csv(
        snakemake.output.state_coal_fuel_prices,
        index_label="snapshot",
    )
    ba_ng_power_prices.to_csv(
        snakemake.output.ba_ng_fuel_prices,
        index_label="snapshot",
    )
