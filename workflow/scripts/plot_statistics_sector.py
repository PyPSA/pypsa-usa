"""
Plots Sector Coupling Statistics.
"""

import logging
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import pypsa
import seaborn as sns
from _helpers import configure_logging, mock_snakemake
from add_electricity import sanitize_carriers
from constants import STATE_2_CODE, Month
from plot_statistics import create_title
from summary_sector import (  # get_load_name_per_sector,
    get_brownfield_capacity_per_state,
    get_capacity_per_node,
    get_emission_timeseries_by_sector,
    get_end_use_consumption,
    get_end_use_load_timeseries,
    get_end_use_load_timeseries_carrier,
    get_historical_emissions,
    get_historical_end_use_consumption,
    get_historical_transport_consumption_by_mode,
    get_hp_cop,
    get_load_factor_timeseries,
    get_load_per_sector_per_fuel,
    get_power_capacity_per_carrier,
    get_sector_production_timeseries,
    get_sector_production_timeseries_by_carrier,
    get_transport_consumption_by_mode,
)

logger = logging.getLogger(__name__)


SECTOR_MAPPER = {
    "res": "residential",
    "res-rural": "residential-rural",
    "res-urban": "residential-urban",
    "com": "commercial",
    "com-rural": "commercial-rural",
    "com-urban": "commercial-urban",
    "pwr": "power",
    "ind": "industrial",
    "trn": "transport",
    "pwr": "Power",
}

FIG_WIDTH = 14
FIG_HEIGHT = 6

SECTORS = {
    "res": "Residential",
    "com": "Commercial",
    "trn": "Transport",
    "ind": "Industrial",
    "pwr": "Power",
}

# figure save format
EXT = "png"

# class Sectors(StrEnum):
#     RES = "Residential"
#     COM = "Commercial"
#     IND = "Industrial"
#     TRN = "Transport"
#     PWR = "Power"

###
# HELPERS
###


def percent_difference(col_1: pd.Series, col_2: pd.Series) -> pd.Series:
    """
    Calculates percent difference between two columns of numbers.
    """
    return abs(col_1 - col_2).div((col_1 + col_2).div(2)).mul(100)


def is_urban_rural_split(n: pypsa.Network) -> bool:
    """
    Checks for urban/rural split based on com/res load names.
    """

    com_res_load = n.loads[(n.loads.index.str.contains("res-")) | (n.loads.index.str.contains("com-"))].index.to_list()

    rural_urban_loads = ["res-urban-", "res-rural-", "com-urban-", "com-rural-"]

    if any(x in y for x in rural_urban_loads for y in com_res_load):
        return True
    else:
        return False


def get_plotting_colors(n: pypsa.Network, nice_name: bool) -> dict[str, str]:

    if nice_name:
        return n.carriers.set_index("nice_name")["color"].to_dict()
    else:
        return n.carriers["color"].to_dict()


def get_sectors(n: pypsa.Network) -> list[str]:
    if is_urban_rural_split(n):
        return ("res-rural", "res-urban", "com-rural", "com-urban", "ind", "trn")
    else:
        return ("res", "com", "ind", "trn")


###
# PLOTTERS
###


# def plot_load_per_sector(
#     n: pypsa.Network,
#     sector: str,
#     sharey: bool = True,
#     log: bool = True,
#     **kwargs,
# ) -> tuple:
#     """
#     Load per bus per sector per fuel.
#     """

#     fuels = get_load_name_per_sector(sector)
#     investment_period = n.investment_periods[0]

#     nrows = ceil(len(fuels) / 2)

#     fig, axs = plt.subplots(
#         ncols=2,
#         nrows=nrows,
#         figsize=(14, 5 * nrows),
#         sharey=sharey,
#     )

#     ylabel = "VMT" if sector == "trn" else "MW"

#     row = 0
#     col = 0

#     for i, fuel in enumerate(fuels):

#         row = i // 2
#         col = i % 2

#         df = get_load_per_sector_per_fuel(n, sector, fuel, investment_period)
#         avg = df.mean(axis=1)

#         palette = sns.color_palette(["lightgray"], df.shape[1])

#         if nrows > 1:

#             sns.lineplot(
#                 df,
#                 color="lightgray",
#                 legend=False,
#                 palette=palette,
#                 ax=axs[row, col],
#             )
#             sns.lineplot(avg, ax=axs[row, col])

#             axs[row, col].set_xlabel("")
#             axs[row, col].set_ylabel(ylabel)
#             axs[row, col].set_title(f"{fuel} load")

#             if log:
#                 axs[row, col].set(yscale="log")

#         else:

#             sns.lineplot(
#                 df,
#                 color="lightgray",
#                 legend=False,
#                 palette=palette,
#                 ax=axs[i],
#             )
#             sns.lineplot(avg, ax=axs[i])

#             axs[i].set_xlabel("")
#             axs[i].set_ylabel(ylabel)
#             axs[i].set_title(f"{fuel} load")

#             if log:
#                 axs[i].set(yscale="log")

#     return fig, axs


def plot_hp_cop(n: pypsa.Network, state: Optional[str] = None, **kwargs) -> tuple:
    """
    Plots gshp and ashp cops.
    """

    investment_period = n.investment_periods[0]

    cops = get_hp_cop(n, state).loc[investment_period]

    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        sharey=True,
    )

    for i, hp in enumerate(["ashp", "gshp"]):

        df = cops[[x for x in cops if x.endswith(hp)]]
        avg = df.mean(axis=1)

        # Ignoring `palette` because no `hue` variable has been assigned.
        if df.empty:
            continue

        palette = sns.color_palette(["lightgray"], df.shape[1])

        try:

            sns.lineplot(
                df,
                color="lightgray",
                legend=False,
                palette=palette,
                ax=axs[i],
            )
            sns.lineplot(avg, ax=axs[i])

            axs[i].set_xlabel("")
            axs[i].set_ylabel("COP")
            axs[i].set_title(f"{hp}")

        except TypeError:  # no numeric data to plot
            logger.warning(f"No COP data to plot for {state}")

    return fig, axs


def plot_sector_production_timeseries(
    n: pypsa.Network,
    sector: str,
    state: Optional[str] = None,
    nice_name: Optional[bool] = True,
    remove_sns_weights: bool = True,
    resample: Optional[str] = None,
    resample_fn: Optional[callable] = None,
    **kwargs,
) -> tuple:
    """
    Plots timeseries production as area chart.
    """

    y_label = kwargs.get("ylabel", "MWh")

    assert sector in ("res", "com", "ind", "pwr", "trn")

    investment_periods = n.investment_periods

    nrows = ceil(len(investment_periods) / 2)

    fig, axs = plt.subplots(
        ncols=1,
        nrows=nrows,
        figsize=(FIG_WIDTH, FIG_HEIGHT * nrows),
    )

    colors = get_plotting_colors(n, nice_name=nice_name)

    df_all = get_sector_production_timeseries_by_carrier(
        n,
        sector=sector,
        state=state,
        resample=resample,
        resample_fn=resample_fn,
        remove_sns_weights=remove_sns_weights,
    )

    for row, period in enumerate(investment_periods):

        df = df_all.loc[period]

        if df.empty:
            logger.warning(f"No data to plot for {state}")
            continue

        if nice_name:
            df = df.rename(columns=n.carriers.nice_name.to_dict())

        try:

            if nrows > 1:

                df.plot(kind="area", ax=axs[row], color=colors)
                axs[row].set_xlabel("")
                axs[row].set_ylabel(y_label)
                # axs[row].set_title(f"{SECTOR_MAPPER[sector]}")
                axs[row].tick_params(axis="x", labelrotation=45)

            else:

                df.plot(kind="area", ax=axs, color=colors)
                axs.set_xlabel("")
                axs.set_ylabel(y_label)
                # axs.set_title(f"{SECTOR_MAPPER[sector]}")

        except TypeError:  # no numeric data to plot
            logger.warning(f"No data to plot for {state} (plot_sector_production_timeseries)")

    return fig, axs


def plot_sector_production(
    n: pypsa.Network,
    sector: str,
    state: Optional[str] = None,
    nice_name: Optional[bool] = True,
    **kwargs,
) -> tuple:
    """
    Plots model period production as bar chart.
    """

    y_label = kwargs.get("ylabel", "MWh")

    assert sector in ("res", "com", "ind", "pwr", "trn")

    investment_periods = n.investment_periods

    nrows = ceil(len(investment_periods) / 2)

    fig, axs = plt.subplots(
        ncols=1,
        nrows=nrows,
        figsize=(FIG_WIDTH, FIG_HEIGHT * nrows),
    )

    df_all = get_sector_production_timeseries_by_carrier(n, sector=sector, state=state)

    for row, period in enumerate(investment_periods):

        df = df_all.loc[period].sum(axis=0)

        if df.empty:
            logger.warning(f"No data to plot for {state}")
            continue

        if nice_name:
            df.index = df.index.map(n.carriers.nice_name)

        try:

            if nrows > 1:

                df.plot.bar(ax=axs[row])
                axs[row].set_xlabel("")
                axs[row].set_ylabel(y_label)
                axs[row].set_title(f"{SECTOR_MAPPER[sector]}")
                axs[row].tick_params(axis="x", labelrotation=45)

            else:

                df.plot.bar(ax=axs)
                axs.set_xlabel("")
                axs.set_ylabel(y_label)
                axs.set_title(f"{SECTOR_MAPPER[sector]}")

        except TypeError:  # no numeric data to plot
            logger.warning(f"No data to plot for {state} (plot_sector_production)")

    return fig, axs


def plot_sector_emissions(
    n: pypsa.Network,
    state: Optional[str] = None,
    **kwargws,
) -> tuple:
    """
    Plots model period emissions by sector.
    """

    investment_period = n.investment_periods[0]

    sectors = ("res", "com", "ind", "trn", "pwr", "ch4")

    data = []

    for sector in sectors:

        df = get_emission_timeseries_by_sector(n, sector, state=state)

        if df.empty:
            logger.warning(f"No data for {state}")
            continue

        data.append(
            df.loc[investment_period,].iloc[-1].values[0],
        )

    if not data:
        # empty data to be caught by type error below
        df = pd.DataFrame(data, columns=sectors)
    else:
        df = pd.DataFrame([data], columns=sectors)

    fig, axs = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
    )

    try:
        df.T.plot.bar(ax=axs, legend=False)
    except TypeError:  # no numeric data to plot
        logger.warning(f"No data to plot for {state}")

    axs.set_ylabel("MT CO2e")
    axs.set_title("CO2e Emissions by Sector")

    return fig, axs


def plot_state_emissions(
    n: pypsa.Network,
    state: Optional[str] = None,
    **kwargws,
) -> tuple:
    """
    Plots stacked bar plot of state level emissions.
    """

    investment_period = n.investment_periods[0]

    fig, axs = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
    )

    df = get_emission_timeseries_by_sector(n, state=state).loc[investment_period,].iloc[-1].to_frame(name=state)
    df.index = df.index.map(lambda x: x.split("-co2")[0][-3:])

    try:
        df.T.plot.bar(stacked=True, ax=axs)
    except TypeError:  # no numeric data to plot
        logger.warning(f"No data to plot for {state}")

    axs.set_ylabel("MT CO2e")
    axs.set_title("CO2e Emissions by State")

    return fig, axs


def plot_capacity_by_carrier(
    n: pypsa.Network,
    sharey: bool = True,
    state: Optional[str] = None,
    nice_name: Optional[bool] = True,
    **kwargs,
) -> tuple:
    """
    Bar plot of capacity by carrier.
    """

    sectors = get_sectors(n)

    nrows = len(sectors)

    fig, axs = plt.subplots(
        ncols=1,
        nrows=nrows,
        figsize=(FIG_WIDTH, FIG_HEIGHT * nrows),
        sharey=sharey,
    )

    for i, sector in enumerate(sectors):

        df = get_capacity_per_node(n, sector, group_existing=True, state=state)
        df = df.reset_index()[["carrier", "p_nom_opt"]]

        if df.empty:
            logger.warning(f"No data to plot for {state} sector {sector}")
            continue

        if nice_name:
            df["carrier"] = df.carrier.map(n.carriers.nice_name)

        df = df.groupby("carrier").sum()

        try:

            if nrows > 1:

                df.plot(kind="bar", stacked=False, ax=axs[i])
                axs[i].set_xlabel("")
                axs[i].set_ylabel("Capacity (MW)")
                axs[i].set_title(f"{sector} Capacity")
                axs[i].tick_params(axis="x", labelrotation=45)

            else:

                df.plot(kind="bar", stacked=False, ax=axs)
                axs.set_xlabel("")
                axs.set_ylabel("Capacity (MW)")
                axs.set_title(f"{sector} Capacity")
                axs.tick_params(axis="x", labelrotation=45)

        except TypeError:  # no numeric data to plot
            logger.warning(f"No data to plot for {state}")

    return fig, axs


def plot_capacity_per_node(
    n: pypsa.Network,
    sharey: bool = True,
    percentage: bool = True,
    state: Optional[str] = None,
    nice_name: Optional[bool] = True,
    **kwargs,
) -> tuple:
    """
    Plots capacity percentage per node.
    """

    sectors = get_sectors(n)

    nrows = len(sectors)

    fig, axs = plt.subplots(
        ncols=1,
        nrows=nrows,
        figsize=(FIG_WIDTH, FIG_HEIGHT * nrows),
        sharey=sharey,
    )

    data_col = "percentage" if percentage else "p_nom_opt"
    y_label = "Percentage (%)" if percentage else "Capacity (MW)"

    colors = get_plotting_colors(n, nice_name)

    for i, sector in enumerate(sectors):

        df = get_capacity_per_node(n, sector, group_existing=True, state=state)
        df = df.reset_index()[["node", "carrier", data_col]]

        if nice_name:
            df["carrier"] = df.carrier.map(n.carriers.nice_name)

        df = df.pivot(columns="carrier", index="node", values=data_col)

        try:

            if nrows > 1:

                df.plot(kind="bar", stacked=True, ax=axs[i], color=colors)
                axs[i].set_xlabel("")
                axs[i].set_ylabel(y_label)
                axs[i].set_title(f"{sector} Capacity")

            else:

                df.plot(kind="bar", stacked=True, ax=axs, color=colors)
                axs.set_xlabel("")
                axs.set_ylabel(y_label)
                axs.set_title(f"{sector} Capacity")

        except TypeError:  # no numeric data to plot
            logger.warning(f"No data to plot for {state}")

    return fig, axs


def plot_capacity_brownfield(
    n: pypsa.Network,
    sharey: bool = True,
    state: Optional[str] = None,
    nice_name: Optional[bool] = True,
    **kwargs,
) -> tuple:
    """
    Plots old and new capacity at a state level by carrier.
    """

    sectors = get_sectors(n)

    nrows = len(sectors)

    fig, axs = plt.subplots(
        ncols=1,
        nrows=nrows,
        figsize=(FIG_WIDTH, FIG_HEIGHT * nrows),
        sharey=sharey,
    )

    y_label = "Capacity (MW)"

    for i, sector in enumerate(sectors):

        df = get_brownfield_capacity_per_state(n, sector, state=state)

        if nice_name:
            df.index = df.index.map(n.carriers.nice_name)

        try:

            if nrows > 1:

                df.plot(kind="bar", ax=axs[i])
                axs[i].set_xlabel("")
                axs[i].set_ylabel(y_label)
                axs[i].set_title(f"{sector} Capacity")

            else:

                df.plot(kind="bar", ax=axs)
                axs.set_xlabel("")
                axs.set_ylabel(y_label)
                axs.set_title(f"{sector} Capacity")

        except TypeError:  # no numeric data to plot
            logger.warning(f"No data to plot for {state}")

    return fig, axs


def plot_power_capacity(
    n: pypsa.Network,
    carriers: list[str],
    sharey: bool = True,
    state: Optional[str] = None,
    nice_name: Optional[bool] = True,
    **kwargs,
) -> tuple:
    """
    Plots capacity of generators in the power sector.
    """

    sector = "pwr"

    nrows = 1

    fig, axs = plt.subplots(
        ncols=1,
        nrows=nrows,
        figsize=(FIG_WIDTH, FIG_HEIGHT * nrows),
        sharey=sharey,
    )

    df = get_power_capacity_per_carrier(
        n,
        carriers,
        group_existing=True,
        state=state,
    )
    df = df.reset_index()[["carrier", "p_nom_opt"]]

    if df.empty:
        logger.warning(f"No data to plot for {state} sector pwr")
        return fig, axs

    if nice_name:
        df["carrier"] = df.carrier.map(n.carriers.nice_name)

    df = df.groupby("carrier").sum()

    try:

        df.plot(kind="bar", stacked=False, ax=axs)
        axs.set_xlabel("")
        axs.set_ylabel("Capacity (MW)")
        axs.set_title(f"{SECTOR_MAPPER[sector]} Capacity")
        axs.tick_params(axis="x", labelrotation=45)

    except TypeError:  # no numeric data to plot
        logger.warning(f"No data to plot for {state}")

    return fig, axs


def plot_sector_load_factor_timeseries(
    n: pypsa.Network,
    sharey: bool = True,
    state: Optional[str] = None,
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
        figsize=(FIG_WIDTH, FIG_HEIGHT * nrows),
        sharey=sharey,
    )

    row = 0
    col = 0

    for i, sector in enumerate(sectors):

        row = i // 2
        col = i % 2

        df = get_load_factor_timeseries(n, sector, state=state).loc[investment_period].resample("d").mean().dropna()

        try:

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

        except TypeError:  # no numeric data to plot
            logger.warning(f"No data to plot for {state}")

    return fig, axs


def plot_sector_load_factor_boxplot(
    n: pypsa.Network,
    sector: str,
    state: Optional[str] = None,
    nice_name: Optional[bool] = True,
    **kwargs,
) -> tuple:
    """
    Plots boxplot of load factors.
    """

    assert sector in ("res", "com", "ind")

    investment_periods = n.investment_periods

    nrows = ceil(len(investment_periods) / 2)

    fig, axs = plt.subplots(
        ncols=1,
        nrows=nrows,
        figsize=(FIG_WIDTH, FIG_HEIGHT * nrows),
    )

    df_all = get_load_factor_timeseries(n, sector, state=state)

    for row, period in enumerate(investment_periods):

        df = df_all.loc[period]

        if nice_name:
            cols = df.columns
            df = df.rename(columns={x: f"{sector}-{x}" for x in cols}).rename(
                columns=n.carriers.nice_name.to_dict(),
            )

        try:

            if nrows > 1:

                sns.boxplot(df, ax=axs[row])
                axs[row].set_xlabel("")
                axs[row].set_ylabel("Load Factor (%)")
                # axs[row].set_title(f"{SECTORS[sector]} Load Factor for {period}")
                axs[row].tick_params(axis="x", labelrotation=45)

            else:

                sns.boxplot(df, ax=axs)
                axs.set_xlabel("")
                axs.set_ylabel("Load Factor (%)")
                # axs.set_title(f"{SECTORS[sector]} Load Factor for {period}")
                axs.tick_params(axis="x", labelrotation=45)

        except TypeError:  # no numeric data to plot
            logger.warning(f"No data to plot for {state}")

    return fig, axs


def plot_sector_emissions_validation(
    n: pypsa.Network,
    eia_api: str,
    state: Optional[str] = None,
    sharey: bool = False,
    **kwargs,
) -> tuple:
    """
    Plots state by state sector emission comparison.
    """

    investment_period = n.investment_periods[0]

    historical = get_historical_emissions(
        ["residential", "commercial", "power", "industrial", "transport"],
        investment_period,
        eia_api,
    )
    historical = historical.rename(columns=STATE_2_CODE)

    modelled = get_emission_timeseries_by_sector(n, state=None).loc[investment_period,].iloc[-1].to_frame(name="value")
    modelled["sector"] = modelled.index.map(lambda x: x.split("-co2")[0][-3:])
    modelled["sector"] = modelled.sector.map(SECTOR_MAPPER)
    modelled["state"] = modelled.index.map(lambda x: x.split(" ")[0])
    modelled = modelled.pivot(index="sector", columns="state", values="value")

    if state:
        if isinstance(state, list):
            states_to_plot = state
        else:
            states_to_plot = [state]
    else:
        states_to_plot = modelled.columns

    if state:  # plot at state level

        nrows = ceil(len(states_to_plot) / 2)

        fig, axs = plt.subplots(
            ncols=2,
            nrows=nrows,
            figsize=(FIG_WIDTH, FIG_HEIGHT * nrows),
            sharey=sharey,
        )

        row = 0
        col = 0

        for i, state_to_plot in enumerate(states_to_plot):

            row = i // 2
            col = i % 2

            try:
                m = modelled[state_to_plot].to_frame(name="Modelled")
                h = historical[state_to_plot].to_frame(name="Actual")
            except KeyError:
                logger.warning(f"No data for {state_to_plot}")
                continue

            assert m.shape == h.shape

            df = m.join(h)
            df["Difference"] = percent_difference(df.Modelled, df.Actual)

            try:

                if nrows > 1:

                    df[["Modelled", "Actual"]].T.plot.bar(ax=axs[row, col])
                    axs[row, col].set_xlabel("")
                    axs[row, col].set_ylabel("Emissions (MT)")
                    axs[row, col].set_title(f"{state_to_plot}")
                    axs[row, col].tick_params(axis="x", labelrotation=0)

                else:

                    df[["Modelled", "Actual"]].T.plot.bar(ax=axs[i])
                    axs[i].set_xlabel("")
                    axs[i].set_ylabel("Emissions (MT)")
                    axs[i].set_title(f"{state_to_plot}")
                    axs[i].tick_params(axis="x", labelrotation=0)

            except TypeError:  # no numeric data to plot
                logger.warning(f"No data to plot for {state}")

        return fig, axs

    else:  # plot at system level

        modelled = modelled.T
        historical = historical.T

        sectors = modelled.columns

        nrows = len(sectors)  # one sector per row

        fig, axs = plt.subplots(
            ncols=1,
            nrows=nrows,
            figsize=(FIG_WIDTH, FIG_HEIGHT * nrows),
            sharey=False,
        )

        for i, sector in enumerate(sectors):

            df = pd.concat(
                [
                    historical[sector].to_frame(name="Actual"),
                    modelled[sector].to_frame(name="Modelled"),
                ],
                axis=1,
            ).dropna()

            try:

                if nrows > 1:

                    df[["Actual", "Modelled"]].plot.bar(ax=axs[i])
                    axs[i].set_xlabel("")
                    axs[i].set_ylabel("Emissions (MT)")
                    axs[i].set_title(f"{sector} Emissions")
                    axs[i].tick_params(axis="x", labelrotation=0)

                else:

                    df[["Actual", "Modelled"]].plot.bar(ax=axs)
                    axs.set_xlabel("")
                    axs.set_ylabel("Emissions (MT)")
                    axs.set_title(f"{sector} Emissions")
                    axs.tick_params(axis="x", labelrotation=0)

            except TypeError:  # no numeric data to plot
                logger.warning(f"No emission data to plot for {sector}")

        return fig, axs


def plot_state_emissions_validation(
    n: pypsa.Network,
    eia_api: str,
    state: Optional[str] = None,
    **kwargs,
) -> tuple:
    """
    Plots total state emission comparison.
    """

    investment_period = n.investment_periods[0]

    historical = get_historical_emissions(
        "total",
        investment_period,
        eia_api,
    ).T.rename(columns={"total": "Actual"}, index=STATE_2_CODE)

    modelled = (
        get_emission_timeseries_by_sector(n, state=None).loc[investment_period,].iloc[-1].to_frame(name="Modelled")
    )
    modelled["state"] = modelled.index.map(lambda x: x.split(" ")[0])
    modelled = modelled.groupby("state").sum()

    if state:
        if isinstance(state, list):
            states_to_plot = state
        else:
            states_to_plot = [state]
    else:
        states_to_plot = modelled.index

    df = historical.T[states_to_plot]
    if isinstance(df, pd.Series):
        df = df.to_frame(name="Modelled")

    df = df.T.join(modelled)

    fig, axs = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
    )

    try:
        df.plot.bar(ax=axs)
        axs.set_xlabel("")
        axs.set_ylabel("Emissions (MT)")
        axs.set_title(f"{investment_period} State Emissions")
        axs.tick_params(axis="x", labelrotation=0)
    except TypeError:  # no numeric data to plot
        logger.warning(f"No data to plot for {state}")

    return fig, axs


def plot_system_emissions_validation_by_state(
    n: pypsa.Network,
    eia_api: str,
    **kwargs,
) -> tuple:
    """
    Plots all states modelled and historcal.
    """

    investment_period = n.investment_periods[0]

    historical = get_historical_emissions(
        "total",
        investment_period,
        eia_api,
    ).T.rename(columns={"total": "Actual"}, index=STATE_2_CODE)

    modelled = (
        get_emission_timeseries_by_sector(n, state=None)
        .loc[investment_period,]
        .iloc[-1]  # casue cumulative
        .to_frame(name="Modelled")
    )
    modelled["state"] = modelled.index.map(lambda x: x.split(" ")[0])
    modelled = modelled.groupby("state").sum()

    df = historical.join(modelled).dropna()

    fig, axs = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(FIG_WIDTH, 8),
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
    state: Optional[str] = None,
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
        modelled = get_end_use_consumption(n, sector, state).loc[investment_period].sum().sum()
        if state:
            data.append([sector, modelled, historical.at[SECTOR_MAPPER[sector], state]])
        else:
            data.append([sector, modelled, historical.loc[SECTOR_MAPPER[sector]].sum()])

    df = pd.DataFrame(data, columns=["sector", "Modelled", "Actual"]).set_index(
        "sector",
    )
    df.index = df.index.map(SECTOR_MAPPER)

    fig, axs = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
    )

    try:
        df.plot.bar(ax=axs)
        axs.set_xlabel("")
        axs.set_ylabel("End Use Consumption (MWh)")
        axs.set_title(f"{investment_period} State Generation")
        axs.tick_params(axis="x", labelrotation=0)
    except TypeError:  # no numeric data to plot
        logger.warning(f"No data to plot for {state}")

    return fig, axs


def plot_sector_load_timeseries(
    n: pypsa.Network,
    sector: str,
    sharey: bool = False,
    state: Optional[str] = None,
    **kwargs,
) -> tuple:

    investment_period = n.investment_periods[0]

    df = get_end_use_load_timeseries(n, sector, sns_weight=False, state=state).loc[investment_period].T
    df.index = df.index.map(n.loads.carrier).map(lambda x: x.split("-")[1:])
    df.index = df.index.map(lambda x: "-".join(x))
    df = df.T

    loads = df.columns.unique()
    nrows = len(loads)
    ylabel = "MW"

    fig, axs = plt.subplots(
        ncols=1,
        nrows=nrows,
        figsize=(FIG_WIDTH, FIG_HEIGHT * nrows),
        sharey=sharey,
    )

    for i, load in enumerate(loads):

        l = df[load]

        # sns.lineplot(l, ax=axs[i], legend=False)

        avg = l.mean(axis=1)

        # palette = sns.color_palette(["lightgray"])

        try:

            sns.lineplot(
                l,
                color="lightgray",
                legend=False,
                # palette=palette,
                ax=axs[i],
                errorbar=("ci", 95),
            )
            sns.lineplot(avg, ax=axs[i])

            axs[i].set_xlabel("")
            axs[i].set_ylabel(ylabel)
            axs[i].set_title(f"{SECTOR_MAPPER[sector]} {load} Load")

        except TypeError:
            logger.warning(f"No data to plot for {state}")

    return fig, axs


def plot_sector_load_bar(
    n: pypsa.Network,
    state: Optional[str] = None,
    **kwargs,
) -> tuple:

    investment_period = n.investment_periods[0]

    sectors = ("res", "com", "ind")

    nrows = ceil(len(sectors) / 2)

    fig, axs = plt.subplots(
        ncols=2,
        nrows=nrows,
        figsize=(FIG_WIDTH, FIG_HEIGHT * nrows),
    )

    row = 0
    col = 0

    title = state if state else "System"

    for i, sector in enumerate(sectors):

        row = i // 2
        col = i % 2

        df = get_end_use_load_timeseries_carrier(n, sector, sns_weight=True, state=state).loc[investment_period].sum()

        if df.empty:
            logger.warning(f"No data to plot for {state}")
            continue

        try:

            if nrows > 1:

                df.T.plot.bar(ax=axs[row, col])
                axs[row, col].set_xlabel("")
                axs[row, col].set_ylabel(f"Load (MWh)")
                axs[row, col].set_title(f"{title} {SECTOR_MAPPER[sector]}")
                axs[row, col].tick_params(axis="x", labelrotation=0)

            else:

                df.T.plot.bar(ax=axs[i])
                axs[i].set_xlabel("")
                axs[i].set_ylabel(f"Load (MWh)")
                axs[i].set_title(f"{title} {SECTOR_MAPPER[sector]}")
                axs[i].tick_params(axis="x", labelrotation=0)

        except TypeError:  # no numeric data to plot
            logger.warning(f"No data to plot for {state}")

    return fig, axs


def plot_transportation_by_mode_validation(
    n: pypsa.Network,
    eia_api: str,
    state: Optional[str] = None,
    **kwargs,
) -> tuple:

    # to pull in from snakemake inputs
    transport_ratios = {
        "Alabama": 2.05,
        "Alaska": 0.75,
        "Arizona": 2.12,
        "Arkansas": 1.08,
        "California": 9.87,
        "Colorado": 1.59,
        "Connecticut": 0.79,
        "Delaware": 0.3,
        "District of Columbia": 0.05,
        "Florida": 6.31,
        "Georgia": 3.23,
        "Hawaii": 0.55,
        "Idaho": 0.64,
        "Illinois": 3.31,
        "Indiana": 2.19,
        "Iowa": 1.09,
        "Kansas": 0.97,
        "Kentucky": 1.86,
        "Louisiana": 2.47,
        "Maine": 0.39,
        "Maryland": 1.52,
        "Massachusetts": 1.4747,
        "Michigan": 2.62,
        "Minnesota": 1.6,
        "Mississippi": 1.29,
        "Missouri": 2.05,
        "Montana": 0.45,
        "Nebraska": 0.76,
        "Nevada": 0.98,
        "New Hampshire": 0.3636,
        "New Jersey": 2.35,
        "New Mexico": 0.9,
        "New York": 3.71,
        "North Carolina": 3.02,
        "North Dakota": 0.49,
        "Ohio": 3.24,
        "Oklahoma": 1.7,
        "Oregon": 1.12,
        "Pennsylvania": 3.08,
        "Rhode Island": 0.2,
        "South Carolina": 1.8,
        "South Dakota": 0.38,
        "Tennessee": 2.52,
        "Texas": 11.86,
        "Utah": 1,
        "Vermont": 0.16,
        "Virginia": 2.74,
        "Washington": 2.29,
        "West Virginia": 0.71,
        "Wisconsin": 1.62,
        "Wyoming": 0.41,
    }
    ratios = {STATE_2_CODE[x]: y / 100 for x, y in transport_ratios.items()}

    investment_period = n.investment_periods[0]

    fig, axs = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
    )

    modelled = (
        get_transport_consumption_by_mode(n, state)
        .loc[investment_period]
        .rename(
            columns={
                "air-psg": "Air",
                "boat-ship": "Shipping, Domestic",
                "bus": "Bus Transportation",
                "hvy": "Freight Trucks",
                "lgt": "Light-Duty Vehicles",
                "med": "Commercial Light Trucks",
                "rail-psg": "Rail, Passenger",
                "rail-ship": "Rail, Freight",
            },
        )
        .sum()
    )

    historical = get_historical_transport_consumption_by_mode(eia_api)

    if state:
        data = pd.DataFrame(index=historical.index)
        data["historical"] = historical.mul(ratios[state])
    else:
        data = historical.to_frame(name="historical")

    data = data.join(modelled.to_frame(name="modelled")).fillna(0)

    try:

        data.plot.bar(ax=axs)
        axs.set_xlabel("")
        axs.set_ylabel(f"Energy Consumption by Transport Mode (MWh)")
        axs.tick_params(axis="x", labelrotation=45)

    except TypeError:  # no numeric data to plot
        logger.warning(f"No data to plot for {state}")

    return fig, axs


def plot_consumption(
    n: pypsa.Network,
    sector: str,
    state: Optional[str] = None,
    nice_name: Optional[bool] = True,
    **kwargs,
) -> tuple:

    assert sector in ("res", "com", "ind", "trn")

    investment_periods = n.investment_periods

    nrows = len(investment_periods)

    fig, axs = plt.subplots(
        ncols=1,
        nrows=nrows,
        figsize=(FIG_WIDTH, FIG_HEIGHT * nrows),
    )

    y_label = "Energy (MWh)"

    df_all = get_end_use_consumption(n, sector, state)

    for row, period in enumerate(investment_periods):

        df = df_all.loc[period]

        if nice_name:
            df = df.rename(columns=n.carriers.nice_name.to_dict())

        df = df.sum(axis=0).to_frame()

        try:

            if nrows > 1:

                df.plot(kind="bar", ax=axs[row], legend=False)
                axs[row].set_xlabel("")
                axs[row].set_ylabel(y_label)
                axs[row].tick_params(axis="x", labelrotation=45)

            else:

                df.plot(kind="bar", ax=axs)
                axs.set_xlabel("")
                axs.set_ylabel(y_label)
                axs.tick_params(axis="x", labelrotation=45)

        except TypeError:  # no numeric data to plot
            logger.warning(f"No data to plot for {state}")

    return fig, axs


def plot_system_consumption_validation_by_state(
    n: pypsa.Network,
    eia_api: str,
    **kwargs,
) -> tuple:

    states = [x for x in n.buses.STATE.unique() if x]  # remove non-classified buses

    sectors = ("res", "com", "ind", "trn")

    nrows = len(sectors)

    fig, axs = plt.subplots(
        ncols=1,
        nrows=nrows,
        figsize=(FIG_WIDTH, FIG_HEIGHT * nrows),
    )

    y_label = "Energy (MWh)"

    for i, sector in enumerate(sectors):

        historical = get_historical_end_use_consumption(
            SECTOR_MAPPER[sector],
            2020,
            eia_api,
        )

        data = []

        for state in states:
            modelled = get_end_use_consumption(n, sector, state)

            data.append([state, modelled.sum().sum(), historical[state].values[0]])

        df = pd.DataFrame(data, columns=["State", "Modelled", "Historical"]).set_index(
            "State",
        )

        try:

            if nrows > 1:

                df.plot(kind="bar", ax=axs[i])
                axs[i].set_xlabel("")
                axs[i].set_ylabel(y_label)
                axs[i].set_title(f"{sector} Production")

            else:

                df.plot(kind="bar", ax=axs)
                axs.set_xlabel("")
                axs.set_ylabel(y_label)
                axs.set_title(f"{sector} Production")

        except TypeError:  # no numeric data to plot
            logger.warning(f"No data to plot for {sector}")

    return fig, axs


###
# HELPERS
###


def save_fig(
    fn: Callable,
    n: pypsa.Network,
    save: str,
    title: str,
    wildcards: dict[str, Any] = None,
    **kwargs,
) -> None:
    """
    Saves the result figure.
    """

    fig, _ = fn(n, **kwargs)

    if not wildcards:
        wildcards = {}

    fig_title = create_title(title, **wildcards)

    fig.suptitle(fig_title)

    fig.tight_layout()

    fig.savefig(save)

    plt.close(fig)


###
# PUBLIC INTERFACE
###

FIGURE_FUNCTION = {
    # load
    "load_timeseries_residential": plot_sector_load_timeseries,
    "load_timeseries_commercial": plot_sector_load_timeseries,
    "load_timeseries_industrial": plot_sector_load_timeseries,
    "load_timeseries_transport": plot_sector_load_timeseries,
    "load_barplot": plot_sector_load_bar,
    # capacity
    "end_use_capacity_per_carrier": plot_capacity_by_carrier,
    "end_use_capacity_per_node_absolute": plot_capacity_per_node,
    "end_use_capacity_per_node_percentage": plot_capacity_per_node,
    "end_use_capacity_state_brownfield": plot_capacity_brownfield,
    "power_capacity_per_carrier": plot_power_capacity,
    # validation
    "emissions_by_sector_validation": plot_sector_emissions_validation,
    "emissions_by_state_validation": plot_state_emissions_validation,
    "generation_by_state_validation": plot_sector_consumption_validation,
    "transportation_by_mode_validation": plot_transportation_by_mode_validation,
    "system_consumption_validation": plot_system_consumption_validation_by_state,
    "system_emission_validation_state": plot_system_emissions_validation_by_state,
}

FIGURE_NICE_NAME = {
    # load
    "load_timeseries_residential": "",
    "load_timeseries_commercial": "",
    "load_timeseries_industrial": "",
    "load_timeseries_transport": "",
    "load_barplot": "Load per Sector per Fuel",
    # capacity
    "end_use_capacity_per_carrier": "Capacity by Carrier",
    "end_use_capacity_per_node_absolute": "Capacity Per Node",
    "end_use_capacity_per_node_percentage": "Capacity Per Node",
    "end_use_capacity_state_brownfield": "Brownfield Capacity Per State",
    # validation
    "emissions_by_sector_validation": "",
    "emissions_by_state_validation": "",
    "generation_by_state_validation": "",
    "transportation_by_mode_validation": "",
    "system_consumption_validation": "",
    "system_emission_validation_state": "",
}

FN_ARGS = {
    # capacity
    "end_use_capacity_per_node_absolute": {"percentage": False},
    "power_capacity_per_carrier": {  # TODO: Pull these from the config file
        "carriers": [
            "nuclear",
            "oil",
            "OCGT",
            "CCGT",
            "coal",
            "geothermal",
            "biomass",
            "onwind",
            "offwind",
            "offwind_floating",
            "solar",
            "hydro",
            "CCGT-95CCS",
            "CCGT-97CCS",
            "coal-95CCS",
            "coal-99CCS",
        ],
    },
    # production
    # loads
    "load_timeseries_residential": {"sector": "res"},
    "load_timeseries_commercial": {"sector": "com"},
    "load_timeseries_industrial": {"sector": "ind"},
    "load_timeseries_transport": {"sector": "trn"},
}


@dataclass
class PlottingData:
    name: str  # snakemake name
    fn: callable
    sector: Optional[str] = None  # None = 'system'
    fn_kwargs: Optional[dict[str, Any]] = None
    nice_name: Optional[str] = None
    plot_by_month: Optional[bool] = False


EMISSIONS_PLOTS = [
    {
        "name": "emissions_by_sector",
        "fn": plot_sector_emissions,
        "nice_name": "Emissions by Sector",
    },
    {
        "name": "emissions_by_state",
        "fn": plot_state_emissions,
        "nice_name": "Emissions by State",
    },
]

PRODUCTION_PLOTS = [
    {
        "name": "load_factor_boxplot",
        "fn": plot_sector_load_factor_boxplot,
        "nice_name": "Residential Load Factor",
        "sector": "res",
        "fn_kwargs": {},
    },
    {
        "name": "load_factor_boxplot",
        "fn": plot_sector_load_factor_boxplot,
        "nice_name": "Commercial Load Factor",
        "sector": "com",
        "fn_kwargs": {},
    },
    {
        "name": "load_factor_boxplot",
        "fn": plot_sector_load_factor_boxplot,
        "nice_name": "Industrial Load Factor",
        "sector": "ind",
        "fn_kwargs": {},
    },
    {
        "name": "hp_cop",
        "fn": plot_hp_cop,
        "nice_name": "Heat Pump Coefficient of Performance",
    },
    {
        "name": "production_time_series",
        "fn": plot_sector_production_timeseries,
        "nice_name": "Residential End Use Production",
        "plot_by_month": True,
        "sector": "res",
        "fn_kwargs": {
            "resample": "D",
            "resample_fn": pd.Series.sum,
        },
    },
    {
        "name": "production_time_series",
        "fn": plot_sector_production_timeseries,
        "nice_name": "Comemrcial End Use Production",
        "plot_by_month": True,
        "sector": "com",
        "fn_kwargs": {
            "resample": "D",
            "resample_fn": pd.Series.sum,
        },
    },
    {
        "name": "production_time_series",
        "fn": plot_sector_production_timeseries,
        "nice_name": "Industrial End Use Production",
        "plot_by_month": True,
        "sector": "ind",
        "fn_kwargs": {
            "resample": "D",
            "resample_fn": pd.Series.sum,
        },
    },
    {
        "name": "production_total",
        "fn": plot_sector_production,
        "nice_name": "Residential End Use Technology Production",
        "sector": "res",
    },
    {
        "name": "production_total",
        "fn": plot_sector_production,
        "nice_name": "Commercial End Use Technology Production",
        "sector": "com",
    },
    {
        "name": "production_total",
        "fn": plot_sector_production,
        "nice_name": "Industrial End Use Technology Production",
        "sector": "ind",
    },
    {
        "name": "system_consumption",
        "fn": plot_consumption,
        "nice_name": "Residenital Consumption",
        "sector": "res",
    },
    {
        "name": "system_consumption",
        "fn": plot_consumption,
        "nice_name": "Commercial Consumption",
        "sector": "com",
    },
    {
        "name": "system_consumption",
        "fn": plot_consumption,
        "nice_name": "Industrial Consumption",
        "sector": "ind",
    },
    {
        "name": "system_consumption",
        "fn": plot_consumption,
        "nice_name": "Transportation Consumption",
        "sector": "trn",
    },
]


def _initialize_metadata(data: dict[str, Any]) -> list[PlottingData]:
    return [PlottingData(**x) for x in data]


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_sector_production",
            simpl="33",
            opts="2190SEG",
            clusters="4m",
            ll="v1.0",
            sector_opts="",
            sector="E-G",
            planning_horizons="2019",
            interconnect="western",
            state="CA",
        )
        rootpath = ".."
    else:
        rootpath = "."
    configure_logging(snakemake)

    # extract shared plotting files
    n = pypsa.Network(snakemake.input.network)

    sanitize_carriers(n, snakemake.config)

    wildcards = snakemake.wildcards

    results_dir = Path(rootpath, snakemake.params.root_dir)

    params = snakemake.params
    result = params["result"]
    eia_api = params.get("eia_api", None)

    states = n.buses[n.buses.reeds_state != ""].reeds_state.unique().tolist()

    if result == "emissions":
        plotting_data = _initialize_metadata(EMISSIONS_PLOTS)
    if result == "production":
        plotting_data = _initialize_metadata(PRODUCTION_PLOTS)
    else:
        raise NotImplementedError

    for plot_data in plotting_data:

        fn = plot_data.fn
        title = plot_data.nice_name if plot_data.nice_name else plot_data.name

        # plot at system level
        if plot_data.sector:
            f_path = Path(results_dir, "system", result, plot_data.sector, f"{plot_data.name}.{EXT}")
        else:
            f_path = Path(results_dir, "system", result, f"{plot_data.name}.{EXT}")

        if not f_path.parent.exists():
            f_path.parent.mkdir(parents=True)

        if plot_data.fn_kwargs:
            fn_kwargs = plot_data.fn_kwargs
        else:
            fn_kwargs = {}
        fn_kwargs["state"] = None  # system level
        fn_kwargs["eia_api"] = eia_api

        if plot_data.sector:
            fn_kwargs["sector"] = plot_data.sector
        else:
            fn_kwargs["sector"] = None

        save_fig(fn, n, str(f_path), title, wildcards, **fn_kwargs)

        # plot each state

        for state in states:

            fn_kwargs["state"] = state

            if plot_data.sector:
                f_path = Path(results_dir, state, result, plot_data.sector, f"{plot_data.name}.{EXT}")
            else:
                f_path = Path(results_dir, state, result, f"{plot_data.name}.{EXT}")

            if not f_path.parent.exists():
                f_path.parent.mkdir(parents=True)

            save_fig(fn, n, str(f_path), title, wildcards, **fn_kwargs)

        if not plot_data.plot_by_month:
            continue

    # for f, f_path in snakemake.output.items():

    #     try:
    #         fn = FIGURE_FUNCTION[f]
    #     except KeyError as ex:
    #         logger.error(f"Must provide a function for plot {f}!")
    #         print(ex)

    #     try:
    #         title = FIGURE_NICE_NAME[f]
    #     except KeyError:
    #         title = f

    #     try:
    #         fn_inputs = FN_ARGS[f]
    #     except KeyError:
    #         fn_inputs = {}

    #     if eia_api:
    #         fn_inputs["eia_api"] = eia_api

    #     if state == "system":
    #         fn_inputs["state"] = None
    #     else:
    #         fn_inputs["state"] = state

    #     save_fig(fn, n, f_path, title, wildcards, **fn_inputs)
