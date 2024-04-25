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

import pandas as pd
import constants as const
from _helpers import mock_snakemake, configure_logging
import eia
from typing import List


def prepare_caiso(caiso_fn: str, sns: pd.DatetimeIndex = None) -> pd.DataFrame:
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
    # new_index = pd.date_range(start=df.index.min(), end=df.index.max()+pd.Timedelta(days=1) + pd.Timedelta(days=1), freq="h")
    hourly_df = pd.DataFrame(
        index=pd.date_range(start=start, end=end + pd.Timedelta(days=1), freq="h")
    )
    return (
        hourly_df.merge(df, how="left", left_index=True, right_index=True)
        .ffill()
        .bfill()
    )


def get_ng_prices(
    sns: pd.date_range, interconnects: list[str], eia_api: str = None
) -> pd.DataFrame:

    if eia_api:
        eia_ng = (
            eia.FuelCosts("gas", "power", sns.year[0], eia_api)
            .get_data()
            .drop(columns=["series-description", "units"])
            .reset_index()
        )
        eia_ng["value"] = (
            eia_ng["value"].astype(float) * 1000 / const.NG_MWH_2_MMCF
        )  # $/MCF -> $/MWh
        eia_ng = eia_ng.pivot(
            index="period",
            columns="state",
            values="value",
        )
        eia_ng = make_hourly(eia_ng)

        if sns.year[0] == 2023: 
            eia_ng.index = eia_ng.index + pd.offsets.DateOffset(years=1)
    else:
        eia_ng = pd.DataFrame()

    if "western" in interconnects:
        caiso_ng = prepare_caiso(snakemake.input.caiso_ng_prices, snapshots)
        caiso_ng = caiso_ng.pivot(
            index="period",
            columns="balancing_area",
            values="dol_mwh_th",
        )
        caiso_ng = make_hourly(caiso_ng)
    else:
        caiso_ng = pd.DataFrame()
    
    ng = pd.concat([caiso_ng, eia_ng], axis=1)
    return ng.loc[sns]


def get_coal_prices(sns: pd.date_range, eia_api: str = None) -> pd.DataFrame:

    if eia_api:
        eia_coal = (
            eia.FuelCosts("coal", "power", sns.year[0], eia_api)
            .get_data()
            .drop(columns=["series-description", "unit"])
            .reset_index()
        )
        eia_coal = eia_coal.pivot(
            index="period",
            columns="state",
            values="price",
        )
        eia_coal = make_hourly(eia_coal)

        if sns.year[0] == 2023: 
            eia_coal.index = eia_coal.index + pd.offsets.DateOffset(years=1)

    else:
        eia_coal = pd.DataFrame()

    return eia_coal.loc[sns]


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

    interconnects = snakemake.config["scenario"]["interconnect"]
    if isinstance(interconnects, str):
        interconnects = [interconnects]

    snapshots = pd.date_range(
        freq="h",
        start=sns_start,
        end=sns_end,
        inclusive=sns_inclusive,
    )

    ng_prices = get_ng_prices(snapshots, interconnects, eia_api)
    ng_prices.to_csv(snakemake.output.ng_fuel_prices, index=False)

    coal_prices = get_coal_prices(snapshots, eia_api)
    ng_prices.to_csv(snakemake.output.coal_fuel_prices, index=False)
