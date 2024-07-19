"""
Plots Sector Coupling Statistics.
"""

import logging
from math import ceil
from typing import Callable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import pypsa
import seaborn as sns
from _helpers import configure_logging, mock_snakemake
from add_electricity import sanitize_carriers
from plot_statistics import create_title
from summary_sector import (
    get_capacity_per_node,
    get_emission_timeseries_by_sector,
    get_end_use_consumption,
    get_historical_emissions,
    get_historical_end_use_consumption,
    get_hp_cop,
    get_load_factor_timeseries,
    get_load_name_per_sector,
    get_load_per_sector_per_fuel,
    get_sector_production_timeseries,
)

logger = logging.getLogger(__name__)


SECTOR_MAPPER = {
    "res": "residential",
    "com": "commercial",
    "pwr": "power",
    "ind": "industrial",
    "trn": "transport",
}

###
# HELPERS
###


def percent_difference(col_1: pd.Series, col_2: pd.Series) -> pd.Series:
    """
    Calculates percent difference between two columns of numbers.
    """
    return abs(col_1 - col_2).div((col_1 + col_2).div(2)).mul(100)


###
# PLOTTERS
###


def plot_load_per_sector(
    n: pypsa.Network,
    sector: str,
    sharey: bool = True,
    log: bool = True,
    **kwargs,
) -> tuple:
    """
    Load per bus per sector per fuel.
    """

    fuels = get_load_name_per_sector(sector)
    investment_period = n.investment_periods[0]

    nrows = ceil(len(fuels) / 2)

    fig, axs = plt.subplots(
        ncols=2,
        nrows=nrows,
        figsize=(14, 5 * nrows),
        sharey=sharey,
    )

    ylabel = "VMT" if sector == "trn" else "MW"

    row = 0
    col = 0

    for i, fuel in enumerate(fuels):

        row = i // 2
        col = i % 2

        df = get_load_per_sector_per_fuel(n, sector, fuel, investment_period)
        avg = df.mean(axis=1)

        palette = sns.color_palette(["lightgray"], df.shape[1])

        if nrows > 1:

            sns.lineplot(
                df,
                color="lightgray",
                legend=False,
                palette=palette,
                ax=axs[row, col],
            )
            sns.lineplot(avg, ax=axs[row, col])

            axs[row, col].set_xlabel("")
            axs[row, col].set_ylabel(ylabel)
            axs[row, col].set_title(f"{fuel} load")

            if log:
                axs[row, col].set(yscale="log")

        else:

            sns.lineplot(
                df,
                color="lightgray",
                legend=False,
                palette=palette,
                ax=axs[i],
            )
            sns.lineplot(avg, ax=axs[i])

            axs[i].set_xlabel("")
            axs[i].set_ylabel(ylabel)
            axs[i].set_title(f"{fuel} load")

            if log:
                axs[i].set(yscale="log")

    return fig, axs


def plot_hp_cop(n: pypsa.Network, **kwargs) -> tuple:
    """
    Plots gshp and ashp cops.
    """

    investment_period = n.investment_periods[0]

    cops = get_hp_cop(n).loc[investment_period]

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), sharey=True)

    for i, hp in enumerate(["ashp", "gshp"]):

        df = cops[[x for x in cops if x.endswith(hp)]]
        avg = df.mean(axis=1)

        palette = sns.color_palette(["lightgray"], df.shape[1])

        sns.lineplot(df, color="lightgray", legend=False, palette=palette, ax=axs[i])
        sns.lineplot(avg, ax=axs[i])

        axs[i].set_xlabel("")
        axs[i].set_ylabel("COP")
        axs[i].set_title(f"{hp}")

    return fig, axs


def plot_sector_production_timeseries(
    n: pypsa.Network,
    sharey: bool = False,
    **kwargs,
) -> tuple:
    """
    Plots timeseries production.
    """

    investment_period = n.investment_periods[0]

    sectors = ("res", "com", "ind")

    nrows = len(sectors)

    fig, axs = plt.subplots(
        ncols=1,
        nrows=nrows,
        figsize=(14, 5 * nrows),
        sharey=sharey,
    )

    for i, sector in enumerate(sectors):

        df = get_sector_production_timeseries(n, sector).loc[investment_period].T
        df.index = df.index.map(n.links.carrier)
        df = df.groupby(level=0).sum().T

        df.plot.area(ax=axs[i])
        axs[i].set_xlabel("")
        axs[i].set_ylabel("MW")
        axs[i].set_title(f"{sector}")

    return fig, axs


def plot_sector_production(n: pypsa.Network, sharey: bool = True, **kwargws) -> tuple:
    """
    Plots model period production.
    """

    investment_period = n.investment_periods[0]

    sectors = ("res", "com", "ind", "trn")

    nrows = ceil(len(sectors) / 2)

    fig, axs = plt.subplots(
        ncols=2,
        nrows=nrows,
        figsize=(14, 5 * nrows),
        sharey=sharey,
    )

    row = 0
    col = 0

    for i, sector in enumerate(sectors):

        row = i // 2
        col = i % 2

        df = get_sector_production_timeseries(n, sector).loc[investment_period].T
        df.index = df.index.map(n.links.carrier)
        df = df.groupby(level=0).sum().T.sum()

        if nrows > 1:

            df.plot.bar(ax=axs[row, col])
            axs[row, col].set_xlabel("")
            axs[row, col].set_ylabel("MWh")
            axs[row, col].set_title(f"{sector}")
            axs[row, col].tick_params(axis="x", labelrotation=45)

        else:

            df.plot.bar(ax=axs[i])
            axs[i].set_xlabel("")
            axs[i].set_ylabel("MWh")
            axs[i].set_title(f"{sector}")

    return fig, axs


def plot_sector_emissions(n: pypsa.Network, **kwargws) -> tuple:
    """
    Plots model period emissions by sector.
    """

    investment_period = n.investment_periods[0]

    sectors = ("res", "com", "ind", "trn", "pwr")

    data = []

    for sector in sectors:

        data.append(
            get_emission_timeseries_by_sector(n, sector)
            .loc[investment_period,]
            .iloc[-1]
            .values[0],
        )

    df = pd.DataFrame([data], columns=sectors)

    fig, axs = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(14, 5),
    )

    df.T.plot.bar(ax=axs, legend=False)
    axs.set_ylabel("MT CO2e")
    axs.set_title("CO2e Emissions by Sector")

    return fig, axs


def plot_state_emissions(n: pypsa.Network, **kwargws) -> tuple:
    """
    Plots stacked bar plot of state level emissions.
    """

    investment_period = n.investment_periods[0]

    state = "Texas"

    fig, axs = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(14, 5),
    )

    df = (
        get_emission_timeseries_by_sector(n)
        .loc[investment_period,]
        .iloc[-1]
        .to_frame(name=state)
    )
    df.index = df.index.map(lambda x: x.split("-co2")[0][-3:])
    df.T.plot.bar(stacked=True, ax=axs)

    axs.set_ylabel("MT CO2e")
    axs.set_title("CO2e Emissions by State")

    return fig, axs


def plot_capacity_per_node(
    n: pypsa.Network,
    sharey: bool = True,
    percentage: bool = True,
    **kwargs,
) -> tuple:
    """
    Plots capacity percentage per node.
    """

    sectors = ("res", "com", "ind")

    nrows = ceil(len(sectors) / 2)

    fig, axs = plt.subplots(
        ncols=2,
        nrows=nrows,
        figsize=(14, 5 * nrows),
        sharey=sharey,
    )

    row = 0
    col = 0

    data_col = "percentage" if percentage else "p_nom_opt"
    y_label = "Percentage (%)" if percentage else "Capacity (MW)"

    for i, sector in enumerate(sectors):

        row = i // 2
        col = i % 2

        df = get_capacity_per_node(n, sector)
        df = df.reset_index()[["node", "carrier", data_col]]
        df = df.pivot(columns="carrier", index="node", values=data_col)

        if nrows > 1:

            df.plot(kind="bar", stacked=True, ax=axs[row, col])
            axs[row, col].set_xlabel("")
            axs[row, col].set_ylabel(y_label)
            axs[row, col].set_title(f"{sector} Capacity")

        else:

            df.plot(kind="bar", stacked=True, ax=axs[i])
            axs[i].set_xlabel("")
            axs[i].set_ylabel(y_label)
            axs[i].set_title(f"{sector} Capacity")

    return fig, axs


def plot_sector_load_factor_timeseries(
    n: pypsa.Network,
    sharey: bool = True,
    **kwargs,
) -> tuple:
    """
    Plots timeseries of load factor resampled to days.
    """

    investment_period = n.investment_periods[0]

    sectors = ("res", "com", "ind")

    nrows = ceil(len(sectors) / 2)

    fig, axs = plt.subplots(
        ncols=2,
        nrows=nrows,
        figsize=(14, 5 * nrows),
        sharey=sharey,
    )

    row = 0
    col = 0

    for i, sector in enumerate(sectors):

        row = i // 2
        col = i % 2

        df = (
            get_load_factor_timeseries(n, sector)
            .loc[investment_period]
            .resample("d")
            .mean()
            .dropna()
        )

        if nrows > 1:

            df.plot(ax=axs[row, col])
            axs[row, col].set_xlabel("")
            axs[row, col].set_ylabel("Load Factor (%)")
            axs[row, col].set_title(f"{sector}")

        else:

            df.plot(ax=axs[i])
            axs[i].set_xlabel("")
            axs[i].set_ylabel("Load Factor (%)")
            axs[i].set_title(f"{sector}")

    return fig, axs


def plot_sector_load_factor_boxplot(
    n: pypsa.Network,
    sharey: bool = True,
    **kwargs,
) -> tuple:
    """
    Plots boxplot of load factors.
    """

    investment_period = n.investment_periods[0]

    sectors = ("res", "com", "ind")

    nrows = ceil(len(sectors) / 2)

    fig, axs = plt.subplots(
        ncols=2,
        nrows=nrows,
        figsize=(14, 6 * nrows),
        sharey=sharey,
    )

    row = 0
    col = 0

    for i, sector in enumerate(sectors):

        row = i // 2
        col = i % 2

        df = get_load_factor_timeseries(n, sector).loc[investment_period]

        if nrows > 1:

            sns.boxplot(df, ax=axs[row, col])
            axs[row, col].set_xlabel("")
            axs[row, col].set_ylabel("Load Factor (%)")
            axs[row, col].set_title(f"{sector}")
            axs[row, col].tick_params(axis="x", labelrotation=45)

        else:

            sns.boxplot(df, ax=axs[i])
            axs[i].set_xlabel("")
            axs[i].set_ylabel("Load Factor (%)")
            axs[i].set_title(f"{sector}")
            axs[i].tick_params(axis="x", labelrotation=45)

    return fig, axs


def plot_sector_emissions_validation(
    n: pypsa.Network,
    eia_api: str,
    sharey: bool = False,
    **kwargs,
) -> tuple:
    """
    Plots state by state sector emission comparison.
    """

    investment_period = n.investment_periods[0]

    historical_emissions = get_historical_emissions(
        ["residential", "commercial", "power", "industrial", "transport"],
        investment_period,
        eia_api,
    )

    modelled_emissions = (
        get_emission_timeseries_by_sector(n)
        .loc[investment_period,]
        .iloc[-1]
        .to_frame(name="Texas")
    )
    modelled_emissions.index = modelled_emissions.index.map(
        lambda x: x.split("-co2")[0][-3:],
    ).map(SECTOR_MAPPER)

    states = modelled_emissions.columns

    nrows = ceil(len(states) / 2)

    fig, axs = plt.subplots(
        ncols=2,
        nrows=nrows,
        figsize=(14, 6 * nrows),
        sharey=sharey,
    )

    row = 0
    col = 0

    for i, state in enumerate(states):

        row = i // 2
        col = i % 2

        modelled = modelled_emissions[state].to_frame(name="Modelled")
        historical = historical_emissions[state].to_frame(name="Actual")

        assert modelled.shape == historical.shape

        df = modelled.join(historical)
        df["Difference"] = percent_difference(df.Modelled, df.Actual)

        if nrows > 1:

            df[["Modelled", "Actual"]].T.plot.bar(ax=axs[row, col])
            axs[row, col].set_xlabel("")
            axs[row, col].set_ylabel("Emissions (MT)")
            axs[row, col].set_title(f"{state}")
            axs[row, col].tick_params(axis="x", labelrotation=0)

        else:

            df[["Modelled", "Actual"]].T.plot.bar(ax=axs[i])
            axs[i].set_xlabel("")
            axs[i].set_ylabel("Emissions (MT)")
            axs[i].set_title(f"{state}")
            axs[i].tick_params(axis="x", labelrotation=0)

    return fig, axs


def plot_state_emissions_validation(
    n: pypsa.Network,
    eia_api: str,
    **kwargs,
) -> tuple:
    """
    Plots total state emission comparison.
    """

    investment_period = n.investment_periods[0]

    historical_emissions = get_historical_emissions(
        "total",
        investment_period,
        eia_api,
    ).T.rename(columns={"total": "Actual"})

    modelled_emissions = (
        (
            get_emission_timeseries_by_sector(n)
            .loc[investment_period,]
            .iloc[-1]
            .to_frame(name="Texas")
        )
        .sum()
        .to_frame(name="Modelled")
    )

    states = modelled_emissions.index

    df = historical_emissions.T[states]
    if isinstance(df, pd.Series):
        df = df.to_frame(name="Modelled")

    df = df.T.join(modelled_emissions)

    fig, axs = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(14, 6),
    )

    df.plot.bar(ax=axs)
    axs.set_xlabel("")
    axs.set_ylabel("Emissions (MT)")
    axs.set_title(f"{investment_period} State Emissions")
    axs.tick_params(axis="x", labelrotation=0)

    return fig, axs


def plot_sector_consumption_validation(
    n: pypsa.Network,
    eia_api: str,
    **kwargs,
) -> tuple:
    """
    Plots sector energy consumption comparison.
    """

    investment_period = n.investment_periods[0]

    historical = get_historical_end_use_consumption(
        ["residential", "commercial", "industrial", "transport"],
        investment_period,
        eia_api,
    )

    data = []

    for sector in ("res", "com", "ind", "trn"):
        modelled = get_end_use_consumption(n, sector, investment_period).sum().sum()
        data.append([sector, modelled, historical.at[SECTOR_MAPPER[sector], "TX"]])

    df = pd.DataFrame(data, columns=["sector", "Modelled", "Actual"]).set_index(
        "sector",
    )
    df.index = df.index.map(SECTOR_MAPPER)

    fig, axs = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(14, 6),
    )

    df.plot.bar(ax=axs)
    axs.set_xlabel("")
    axs.set_ylabel("End Use Consumption (MWh)")
    axs.set_title(f"{investment_period} State Generation")
    axs.tick_params(axis="x", labelrotation=0)

    return fig, axs


###
# HELPERS
###


def save_fig(
    fn: Callable,
    n: pypsa.Network,
    save: str,
    title: str,
    wildcards: dict[str, any] = {},
    **kwargs,
) -> None:
    """
    Saves the result figure.
    """

    fig, _ = fn(n, **kwargs)

    fig_title = create_title(title, **wildcards)

    fig.suptitle(fig_title)
    fig.tight_layout()

    fig.savefig(save)


###
# PUBLIC INTERFACE
###

FIGURE_FUNCTION = {
    # production
    "load_factor_boxplot": plot_sector_load_factor_boxplot,
    "hp_cop": plot_hp_cop,
    "production_time_series": plot_sector_production_timeseries,
    "production_total": plot_sector_production,
    # capacity
    "end_use_capacity_per_node_absolute": plot_capacity_per_node,
    "end_use_capacity_per_node_percentage": plot_capacity_per_node,
    # emissions
    "emissions_by_sector": plot_sector_emissions,
    "emissions_by_state": plot_state_emissions,
    # validation
    "emissions_by_sector_validation": plot_sector_emissions_validation,
    "emissions_by_state_validation": plot_state_emissions_validation,
    "generation_by_state_validation": plot_sector_consumption_validation,
}

FIGURE_NICE_NAME = {
    # production
    "load_factor_boxplot": "Load Factor",
    "hp_cop": "Heat Pump Coefficient of Performance",
    "production_time_series": "End Use Technology Production",
    "production_total": "End Use Technology Production",
    # capacity
    "end_use_capacity_per_node_absolute": "Capacity Per Node",
    "end_use_capacity_per_node_percentage": "Capacity Per Node",
    # emissions
    "emissions_by_sector": "",
    "emissions_by_state": "",
    # validation
    "emissions_by_sector_validation": "",
    "emissions_by_state_validation": "",
    "generation_by_state_validation": "",
}

FN_ARGS = {
    "end_use_capacity_per_node_absolute": {"percentage": False},
}

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_sector_validate",
            interconnect="texas",
            clusters=20,
            ll="v1.0",
            opts="500SEG",
            sector="E-G",
        )
    configure_logging(snakemake)

    # extract shared plotting files
    n = pypsa.Network(snakemake.input.network)

    sanitize_carriers(n, snakemake.config)

    wildcards = snakemake.wildcards

    params = snakemake.params
    eia_api = params.get("eia_api", None)

    for f, f_path in snakemake.output.items():

        try:
            fn = FIGURE_FUNCTION[f]
        except KeyError as ex:
            logger.error(f"Must provide a function for plot {f}!")
            print(ex)

        try:
            title = FIGURE_NICE_NAME[f]
        except KeyError:
            title = f

        try:
            fn_inputs = FN_ARGS[f]
        except KeyError:
            fn_inputs = {}

        if eia_api:
            fn_inputs["eia_api"] = eia_api

        save_fig(fn, n, f_path, title, wildcards, **fn_inputs)
