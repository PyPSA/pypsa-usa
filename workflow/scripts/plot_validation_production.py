import logging
from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd
import pypsa
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)
from _helpers import configure_logging
from constants import EIA_930_REGION_MAPPER

sns.set_theme("paper", style="whitegrid")

rename_op = {
    "CCGT": "Natural gas",
    "hydro": "Hydro",
    "oil": "Oil",
    "onwind": "Onshore wind",
    "solar": "Solar",
    "nuclear": "Nuclear",
    "coal": "Coal",
    "geothermal": "Other",
}
selected_cols = [
    "Balancing Authority",
    "UTC Time at End of Hour",
    "Net Generation (MW) from Natural Gas (Adjusted)",
    "Net Generation (MW) from Coal (Adjusted)",
    "Net Generation (MW) from Nuclear (Adjusted)",
    "Net Generation (MW) from All Petroleum Products (Adjusted)",
    "Net Generation (MW) from Hydropower and Pumped Storage (Adjusted)",
    "Net Generation (MW) from Solar (Adjusted)",
    "Net Generation (MW) from Wind (Adjusted)",
    "Net Generation (MW) from Other Fuel Sources (Adjusted)",
    "Region",
]
rename_his = {
    "Net Generation (MW) from Natural Gas (Adjusted)": "Natural gas",
    "Net Generation (MW) from Hydropower and Pumped Storage (Adjusted)": "Hydro",
    "Net Generation (MW) from All Petroleum Products (Adjusted)": "Oil",
    "Net Generation (MW) from Wind (Adjusted)": "Onshore wind",
    "Net Generation (MW) from Solar (Adjusted)": "Solar",
    "Net Generation (MW) from Nuclear (Adjusted)": "Nuclear",
    "Net Generation (MW) from Coal (Adjusted)": "Coal",
    "Net Generation (MW) from Other Fuel Sources (Adjusted)": "Other",
}
colors = [
    "purple",
    "dimgray",
    "brown",
    "royalblue",
    "chocolate",
    "green",
    "lightskyblue",
    "crimson",
]
kwargs = dict(color=colors, ylabel="Production [GW]", xlabel="")

def plot_regional_timeseries_comparison(
    n: pypsa.Network,
):
    """
    """
    regions = n.buses.country.unique()
    regions_clean = [ba.split("-")[0] for ba in regions]
    regions = list(OrderedDict.fromkeys(regions_clean)) 
     #filter out a few regions
    regions = [ba for ba in regions if ba in ['CISO']]
    buses = n.buses.copy()
    buses["region"] = [ba.split("-")[0] for ba in buses.country]
    for region in regions:
        region_bus = buses.query(f"region == '{region}'").index
        optimized_region = create_optimized_by_carrier(n, order, buses.loc[region_bus].country)
        historic_region = create_historical_df(snakemake.input.historic_first, snakemake.input.historic_second, region=region)[0]
        plot_timeseries_comparison(
            historic = historic_region,
            optimized = optimized_region,
            save_path= Path(snakemake.output[0]).parents[0] / f"{region}_seasonal_stacked_plot.png",
        )


def plot_timeseries_comparison(
    historic: pd.DataFrame,
    optimized: pd.DataFrame,
    save_path: str,
):
    """
    plots a stacked plot for seasonal production for snapshots: January 2 - December 30 (inclusive)
    """

    fig, axes = plt.subplots(3, 1, figsize=(9, 9))
    optimized.resample("1D").mean().plot.area(
        ax=axes[0], **kwargs, legend=False, title="Optimized"
    )
    historic.resample("1D").mean().plot.area(
        ax=axes[1], **kwargs, legend=False, title="Historic"
    )

    diff = (optimized - historic).fillna(0).resample("1D").mean()
    diff.clip(lower=0).plot.area(
        ax=axes[2],
        **kwargs,
        title=r"$\Delta$ (Optimized - Historic)",
    )
    lim = axes[2].get_ylim()[1]
    diff.clip(upper=0).plot.area(ax=axes[2], **kwargs, legend=False)
    axes[2].set_ylim(bottom=-lim, top=lim)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(
        h[::-1],
        l[::-1],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        ncol=1,
        frameon=False,
        labelspacing=1,
    )
    fig.tight_layout()
    fig.savefig(save_path)


def plot_bar_carrier_production(
    historic: pd.DataFrame,
    optimized: pd.DataFrame,
    save_path: str,
):
    # plot by carrier
    data = pd.concat([historic, optimized], keys=["Historic", "Optimized"], axis=1)
    data.columns.names = ["Kind", "Carrier"]
    fig, ax = plt.subplots(figsize=(6, 6))
    df = data.groupby(level=["Kind", "Carrier"], axis=1).sum().sum().unstack().T
    df = df / 1e3  # convert to TWh
    df.plot.barh(ax=ax, xlabel="Electricity Production [TWh]", ylabel="")
    ax.set_title("Electricity Production by Carriers")
    ax.grid(axis="y")
    fig.savefig(save_path)


def plot_bar_production_deviation(
    historic: pd.DataFrame,
    optimized: pd.DataFrame,
    save_path: str,
):
    # plot strongest deviations for each carrier
    fig, ax = plt.subplots(figsize=(6, 10))
    diff = (optimized - historic).sum() / 1e3  # convert to TW
    diff = diff.dropna().sort_values()
    diff.plot.barh(
        xlabel="Optimized Production - Historic Production [TWh]",
        ax=ax,
    )
    ax.set_title("Strongest Deviations")
    ax.grid(axis="y")
    fig.savefig(save_path)


def create_optimized_by_carrier(n, order, region= None):
    """
    Create a DataFrame from the model output/optimized.
    """
    if region is not None:
        gen_p = n.generators_t["p"].loc[:, n.generators.bus.map(n.buses.country).isin(region)]
    else:
        gen_p = n.generators_t["p"]

    optimized = (
        gen_p.groupby(axis="columns", by=n.generators["carrier"])
        .sum()
        .loc["2019-01-02 00:00:00":"2019-12-30 23:00:00"]
    )

    # Combine CCGT and OCGT
    if "OCGT" in optimized.columns:
        optimized["CCGT"] = optimized["CCGT"] + optimized["OCGT"]
        optimized_comb = optimized.drop(["OCGT"], axis=1)

    # Rename and rearrange the columns
    optimized = optimized_comb.rename(columns=rename_op)
    optimized = optimized.reindex(order, axis=1, level=1)
    # Convert to GW
    optimized = optimized / 1e3
    return optimized


def create_historical_df(
    csv_path_1, 
    csv_path_2, 
    region= None,):
    """
    Create a DataFrame from the csv files containing historical data.
    """
    historic_first = pd.read_csv(
        csv_path_1,
        index_col=[0, 1],
        header=0,
        parse_dates=True,
        date_format="%m/%d/%Y %I:%M:%S %p",
        usecols=selected_cols,
    )
    historic_first = historic_first[
        historic_first.Region.map(EIA_930_REGION_MAPPER)
        == snakemake.wildcards.interconnect
    ]

    historic_second = pd.read_csv(
        csv_path_2,
        index_col=[0, 1],
        header=0,
        parse_dates=True,
        date_format="%m/%d/%Y %I:%M:%S %p",
        usecols=selected_cols,
    )
    historic_second = historic_second[
        historic_second.Region.map(EIA_930_REGION_MAPPER)
        == snakemake.wildcards.interconnect
    ]

    # Clean the data read from csv
    historic_first_df = (
        historic_first.fillna(0)
        .replace({",": ""}, regex=True)
        .drop(columns="Region", axis=1)
        .astype(float)
    )
    historic_second_df = (
        historic_second.fillna(0)
        .replace({",": ""}, regex=True)
        .drop(columns="Region", axis=1)
        .astype(float)
    )

    historic = pd.concat([historic_first_df, historic_second_df], axis=0)

    if region is not None:
        if region == 'Arizona': region = ['AZPS']
        historic = historic.loc[region]

    historic = (
        historic
        .groupby(["UTC Time at End of Hour"])
        .sum()
    )

    historic = historic.rename(columns=rename_his)
    historic[historic < 0] = (
        0  # remove negative values for plotting (low impact on results)
    )
    order = (historic.diff().abs().sum() / historic.sum()).sort_values().index
    historic = historic.reindex(order, axis=1, level=1)
    historic = historic / 1e3
    return historic, order


def get_regions(n):
    regions = n.buses.country.unique()
    regions_clean = [ba.split("0")[0] for ba in regions]
    regions_clean = [ba.split("-")[0] for ba in regions_clean]
    regions = list(OrderedDict.fromkeys(regions_clean))
    return regions


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(  # use Validation config
            "plot_validation_figures",
            interconnect="western",
            clusters=40,
            ll="v1.0",
            opts="Co2L2.0",
            sector="E",
        )
    configure_logging(snakemake)
    n = pypsa.Network(snakemake.input.network)

    buses = get_regions(n)

    historic, order = create_historical_df(
        snakemake.input.historic_first, 
        snakemake.input.historic_second, 
        )
    optimized = create_optimized_by_carrier(n, order)

    plot_regional_timeseries_comparison(
        n,
    )
    plot_timeseries_comparison(
        historic,
        optimized,
        save_path=snakemake.output["seasonal_stacked_plot"],
    )

    plot_bar_carrier_production(
        historic,
        optimized,
        save_path=snakemake.output["carrier_production_bar"],
    )

    plot_bar_production_deviation(
        historic,
        optimized,
        save_path=snakemake.output["production_deviation_bar"],
    )

