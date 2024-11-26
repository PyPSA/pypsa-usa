"""
Plots Sector Coupling Statistics.
"""

import logging
from dataclasses import dataclass
from enum import Enum
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
from constants_sector import (
    AirTransport,
    AirTransportUnits,
    BoatTransport,
    BoatTransportUnits,
    RailTransport,
    RailTransportUnits,
    RoadTransport,
    RoadTransportUnits,
    Transport,
)
from plot_statistics import create_title
from summary_sector import (  # get_load_name_per_sector,
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

# figure save format
EXT = "png"

###
# HELPERS
###


def _get_month_name(month: Month) -> str:
    return month.name.capitalize()


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
    month: Optional[int] = None,
    **kwargs,
) -> tuple:
    """
    Plots timeseries production as area chart.
    """

    y_label = kwargs.get("ylabel", "MWh")

    assert sector in ("res", "com", "ind", "pwr")

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

        if month:
            df = df[df.index.get_level_values("timestep").month == month_i]

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
            logger.warning(
                f"No data to plot for {state} (plot_sector_production_timeseries)",
            )

    return fig, axs


def plot_transportation_production_timeseries(
    n: pypsa.Network,
    sector: str,
    vehicle: str,  # veh, air, rail, ect.. .
    modes: Enum,  # AirTransport, RoadTransport, ect..
    units: Enum,
    state: Optional[str] = None,
    nice_name: Optional[bool] = True,
    remove_sns_weights: bool = True,
    resample: Optional[str] = None,
    resample_fn: Optional[callable] = None,
    month: Optional[int] = None,
    **kwargs,
) -> tuple:
    """
    Plots timeseries production as area chart.
    """

    assert sector == "trn"

    def _filter_vehicle_type(df: pd.DataFrame, vehicle: str) -> pd.DataFrame:
        cols = [x for x in df.columns if x.split("-")[-2] == vehicle]
        return df[cols].copy()

    assert sector == "trn"

    diff_units = {x.value for x in units}

    investment_periods = n.investment_periods

    # one unit type per plot
    nrows = len(investment_periods) * len(diff_units)

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
    df_veh = _filter_vehicle_type(df_all, vehicle)

    if df_veh.empty:
        logger.warning(f"No data to plot for {state} sector {sector}")
        return fig, axs

    for row, period in enumerate(investment_periods):

        df_veh_period = df_veh.loc[period]

        if month:
            df_veh_period = df_veh_period[df_veh_period.index.get_level_values("timestep").month == month_i].copy()

        for i, unit in enumerate(diff_units):

            all_modes = [x.name for x in modes]
            modes_per_unit = [modes[x].value for x in all_modes if units[x].value == unit]

            df = df_veh_period[[x for x in df_veh_period.columns if x.split("-")[-1] in modes_per_unit]]

            if nice_name:
                df = df.rename(columns=n.carriers.nice_name.to_dict())

            try:

                if nrows > 1:

                    df.plot(kind="area", ax=axs[row + i], color=colors)
                    axs[row + i].set_xlabel("")
                    axs[row + i].set_ylabel(f"{unit}")
                    axs[row + i].tick_params(axis="x", labelrotation=45)

                else:

                    df.plot(kind="area", ax=axs, color=colors)
                    axs.set_xlabel("")
                    axs.set_ylabel(f"{unit}")
                    axs.tick_params(axis="x", labelrotation=45)

            except TypeError:  # no numeric data to plot
                logger.warning(
                    f"No data to plot for {state} (plot_sector_production_timeseries)",
                )

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
    sector: str,
    state: Optional[str] = None,
    nice_name: Optional[bool] = True,
    **kwargs,
) -> tuple:
    """
    Bar plot of capacity by carrier.
    """

    investment_periods = n.investment_periods

    nrows = len(investment_periods)

    fig, axs = plt.subplots(
        ncols=1,
        nrows=nrows,
        figsize=(FIG_WIDTH, FIG_HEIGHT * nrows),
    )

    df_all = get_capacity_per_node(n, sector=sector, state=state)
    df_all = df_all.reset_index()[["carrier", "p_nom_opt"]]

    for row, _ in enumerate(investment_periods):

        df = df_all.copy()

        if df.empty:
            logger.warning(f"No data to plot for {state} sector {sector}")
            continue

        if nice_name:
            df["carrier"] = df.carrier.map(n.carriers.nice_name)

        df = df.groupby("carrier").sum()

        try:

            if nrows > 1:

                df.plot(kind="bar", stacked=False, ax=axs[row])
                axs[row].set_xlabel("")
                axs[row].set_ylabel("Capacity (MW)")
                axs[row].set_title(f"{sector} Capacity")
                axs[row].tick_params(axis="x", labelrotation=45)

            else:

                df.plot(kind="bar", stacked=False, ax=axs)
                axs.set_xlabel("")
                axs.set_ylabel("Capacity (MW)")
                axs.set_title(f"{sector} Capacity")
                axs.tick_params(axis="x", labelrotation=45)

        except TypeError:  # no numeric data to plot
            logger.warning(f"No data to plot for {state}")

    return fig, axs


def plot_transportation_capacity_by_carrier(
    n: pypsa.Network,
    sector: str,
    vehicle: str,  # veh, air, rail, ect.. .
    modes: Enum,  # AirTransport, RoadTransport, ect..
    units: Enum,
    state: Optional[str] = None,
    nice_name: Optional[bool] = True,
    **kwargs,
) -> tuple:
    """
    Bar plot of capacity by carrier.
    """

    def _filter_vehicle_type(df: pd.DataFrame, vehicle: str) -> pd.DataFrame:
        df["vehicle"] = df.index.get_level_values("carrier").map(
            lambda x: x.split("-")[-2],
        )
        df = df[df.vehicle == vehicle].copy()
        return df.drop(columns="vehicle")

    assert sector == "trn"

    df_all = get_capacity_per_node(n, sector=sector, state=state)
    df_veh = _filter_vehicle_type(df_all, vehicle)
    df_veh = df_veh.reset_index()[["carrier", "p_nom_opt"]]

    diff_units = {x.value for x in units}

    investment_periods = n.investment_periods

    # one unit type per plot
    nrows = len(investment_periods) * len(diff_units)

    fig, axs = plt.subplots(
        ncols=1,
        nrows=nrows,
        figsize=(FIG_WIDTH, FIG_HEIGHT * nrows),
    )

    for row, _ in enumerate(investment_periods):

        df = df_veh.copy()

        if df.empty:
            logger.warning(f"No data to plot for {state} sector {sector}")
            continue

        df["mode"] = df.carrier.map(lambda x: x.split("-")[-1])

        for i, unit in enumerate(diff_units):

            all_modes = [x.name for x in modes]
            modes_per_unit = [modes[x].value for x in all_modes if units[x].value == unit]

            df_mode = df[df["mode"].isin(modes_per_unit)].copy().drop(columns="mode")

            if nice_name:
                df_mode["carrier"] = df_mode.carrier.map(n.carriers.nice_name)

            df_mode = df_mode.groupby("carrier").sum()

            try:

                if nrows > 1:

                    df_mode.plot(kind="bar", stacked=False, ax=axs[row + i])
                    axs[row + i].set_xlabel("")
                    axs[row + i].set_ylabel(f"Capacity ({unit})")
                    axs[row + i].set_title(f"{sector} {vehicle} Capacity")
                    axs[row + i].tick_params(axis="x", labelrotation=45)

                else:

                    df_mode.plot(kind="bar", stacked=False, ax=axs)
                    axs.set_xlabel("")
                    axs.set_ylabel(f"Capacity ({unit})")
                    axs.set_title(f"{sector} {vehicle} Capacity")
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

        df = get_capacity_per_node(n, sector=sector, state=state)
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
    sector: str,
    state: Optional[str] = None,
    nice_name: Optional[bool] = True,
    **kwargs,
) -> tuple:
    """
    Plots old and new capacity at a state level by carrier.
    """

    investment_periods = n.investment_periods

    nrows = len(investment_periods)

    fig, axs = plt.subplots(
        ncols=1,
        nrows=nrows,
        figsize=(FIG_WIDTH, FIG_HEIGHT * nrows),
    )

    y_label = "Capacity (MW)"

    for row, _ in enumerate(investment_periods):

        df = get_capacity_per_node(n, sector, state)

        if nice_name:
            nn = n.carriers.nice_name.to_dict()
            df.index = df.index.map(lambda x: (x[0], nn[x[1]]))

        df = df.droplevel("node")
        df = df.reset_index()[["carrier", "existing", "new"]].groupby("carrier").sum()

        try:

            if nrows > 1:

                df.plot(kind="bar", ax=axs[row])
                axs[row].set_xlabel("")
                axs[row].set_ylabel(y_label)
                axs[row].set_title(f"{sector} Capacity")
                axs.tick_params(axis="x", labelrotation=45)

            else:

                df.plot(kind="bar", ax=axs)
                axs.set_xlabel("")
                axs.set_ylabel(y_label)
                axs.set_title(f"{sector} Capacity")
                axs.tick_params(axis="x", labelrotation=45)

        except TypeError:  # no numeric data to plot
            logger.warning(f"No data to plot for {state}")

    return fig, axs


def plot_transportation_capacity_brownfield(
    n: pypsa.Network,
    sector: str,
    vehicle: str,  # veh, air, rail, ect.. .
    modes: Enum,  # AirTransport, RoadTransport, ect..
    units: Enum,
    state: Optional[str] = None,
    nice_name: Optional[bool] = True,
    **kwargs,
) -> tuple:
    """
    Plots old and new capacity at a state level by carrier.
    """

    def _filter_vehicle_type(df: pd.DataFrame, vehicle: str) -> pd.DataFrame:
        df["vehicle"] = df.index.get_level_values("carrier").map(
            lambda x: x.split("-")[-2],
        )
        df = df[df.vehicle == vehicle].copy()
        return df.drop(columns="vehicle")

    assert sector == "trn"

    df_all = get_capacity_per_node(n, sector=sector, state=state)
    df_veh = _filter_vehicle_type(df_all, vehicle)
    df_veh = df_veh.reset_index()[["carrier", "existing", "new"]]

    diff_units = {x.value for x in units}

    investment_periods = n.investment_periods

    # one unit type per plot
    nrows = len(investment_periods) * len(diff_units)

    fig, axs = plt.subplots(
        ncols=1,
        nrows=nrows,
        figsize=(FIG_WIDTH, FIG_HEIGHT * nrows),
    )

    for row, _ in enumerate(investment_periods):

        df = df_veh.copy()

        if df.empty:
            logger.warning(f"No data to plot for {state} sector {sector}")
            continue

        df["mode"] = df.carrier.map(lambda x: x.split("-")[-1])

        for i, unit in enumerate(diff_units):

            all_modes = [x.name for x in modes]
            modes_per_unit = [modes[x].value for x in all_modes if units[x].value == unit]

            df_mode = df[df["mode"].isin(modes_per_unit)].copy().drop(columns="mode")

            if nice_name:
                df_mode["carrier"] = df_mode.carrier.map(n.carriers.nice_name)

            df_mode = df_mode.groupby("carrier").sum()

            try:

                if nrows > 1:

                    df_mode.plot(kind="bar", stacked=False, ax=axs[row + i])
                    axs[row + i].set_xlabel("")
                    axs[row + i].set_ylabel(f"Capacity ({unit})")
                    axs[row + i].set_title(f"{sector} {vehicle} Capacity")
                    axs[row + i].tick_params(axis="x", labelrotation=45)

                else:

                    df_mode.plot(kind="bar", stacked=False, ax=axs)
                    axs.set_xlabel("")
                    axs.set_ylabel(f"Capacity ({unit})")
                    axs.set_title(f"{sector} {vehicle} Capacity")
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

# "end_use_capacity_per_node_absolute": plot_capacity_per_node,
# "end_use_capacity_per_node_percentage": plot_capacity_per_node,

# "end_use_capacity_per_node_absolute": "Capacity Per Node",
# "end_use_capacity_per_node_percentage": "Capacity Per Node",

# FIGURE_FUNCTION = {
#     # load
#     "load_timeseries_residential": plot_sector_load_timeseries,
#     "load_timeseries_commercial": plot_sector_load_timeseries,
#     "load_timeseries_industrial": plot_sector_load_timeseries,
#     "load_timeseries_transport": plot_sector_load_timeseries,
#     "load_barplot": plot_sector_load_bar,
# }


# FN_ARGS = {
#     # capacity
#     "end_use_capacity_per_node_absolute": {"percentage": False},
#     # production
#     # loads
#     "load_timeseries_residential": {"sector": "res"},
#     "load_timeseries_commercial": {"sector": "com"},
#     "load_timeseries_industrial": {"sector": "ind"},
#     "load_timeseries_transport": {"sector": "trn"},
# }


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
        "name": "production_time_series",
        "fn": plot_sector_production_timeseries,
        "nice_name": "Power End Use Production",
        "plot_by_month": True,
        "sector": "pwr",
        "fn_kwargs": {
            "resample": "D",
            "resample_fn": pd.Series.sum,
        },
    },
    {
        "name": "production_time_series",
        "fn": plot_transportation_production_timeseries,
        "nice_name": "Road Vehicle End Use Production",
        "plot_by_month": True,
        "sector": "trn",
        "fn_kwargs": {
            "resample": "D",
            "resample_fn": pd.Series.sum,
            "vehicle": Transport.ROAD.value,
            "modes": RoadTransport,
            "units": RoadTransportUnits,
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

CAPACITY_PLOTS = [
    {
        "name": "end_use_capacity_per_carrier",
        "fn": plot_capacity_by_carrier,
        "nice_name": "Residenital Capacity",
        "sector": "res",
    },
    {
        "name": "end_use_capacity_per_carrier",
        "fn": plot_capacity_by_carrier,
        "nice_name": "Commercial Capacity",
        "sector": "com",
    },
    {
        "name": "end_use_capacity_per_carrier",
        "fn": plot_capacity_by_carrier,
        "nice_name": "Industrial Capacity",
        "sector": "ind",
    },
    {
        "name": "air_end_use_capacity_per_carrier",
        "fn": plot_transportation_capacity_by_carrier,
        "nice_name": "Transportation Capacity",
        "sector": "trn",
        "fn_kwargs": {
            "vehicle": Transport.AIR.value,
            "modes": AirTransport,
            "units": AirTransportUnits,
        },
    },
    {
        "name": "boat_end_use_capacity_per_carrier",
        "fn": plot_transportation_capacity_by_carrier,
        "nice_name": "Transportation Capacity",
        "sector": "trn",
        "fn_kwargs": {
            "vehicle": Transport.BOAT.value,
            "modes": BoatTransport,
            "units": BoatTransportUnits,
        },
    },
    {
        "name": "rail_end_use_capacity_per_carrier",
        "fn": plot_transportation_capacity_by_carrier,
        "nice_name": "Transportation Capacity",
        "sector": "trn",
        "fn_kwargs": {
            "vehicle": Transport.RAIL.value,
            "modes": RailTransport,
            "units": RailTransportUnits,
        },
    },
    {
        "name": "road_end_use_capacity_per_carrier",
        "fn": plot_transportation_capacity_by_carrier,
        "nice_name": "Transportation Capacity",
        "sector": "trn",
        "fn_kwargs": {
            "vehicle": Transport.ROAD.value,
            "modes": RoadTransport,
            "units": RoadTransportUnits,
        },
    },
    {
        "name": "end_use_capacity_per_carrier",
        "fn": plot_capacity_by_carrier,
        "nice_name": "Power Capacity",
        "sector": "pwr",
    },
    {
        "name": "end_use_capacity_state_brownfield",
        "fn": plot_capacity_brownfield,
        "nice_name": "Residenital Brownfield Capacity",
        "sector": "res",
    },
    {
        "name": "end_use_capacity_state_brownfield",
        "fn": plot_capacity_brownfield,
        "nice_name": "Commercial Brownfield Capacity",
        "sector": "com",
    },
    {
        "name": "end_use_capacity_state_brownfield",
        "fn": plot_capacity_brownfield,
        "nice_name": "Industrial Browfield Capacity",
        "sector": "ind",
    },
    {
        "name": "end_use_capacity_state_brownfield",
        "fn": plot_capacity_brownfield,
        "nice_name": "Transportation Brownfield Capacity",
        "sector": "trn",
    },
    {
        "name": "end_use_capacity_state_brownfield",
        "fn": plot_capacity_brownfield,
        "nice_name": "Power Brownfield Capacity",
        "sector": "pwr",
    },
    {
        "name": "road_brownfield_end_use_capacity_per_carrier",
        "fn": plot_transportation_capacity_brownfield,
        "nice_name": "Transportation Brownfield Capacity",
        "sector": "trn",
        "fn_kwargs": {
            "vehicle": Transport.ROAD.value,
            "modes": RoadTransport,
            "units": RoadTransportUnits,
        },
    },
]


def _initialize_metadata(data: dict[str, Any]) -> list[PlottingData]:
    return [PlottingData(**x) for x in data]


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_sector_capacity",
            simpl="11",
            opts="3h",
            clusters="4m",
            ll="v1.0",
            sector_opts="",
            sector="E-G",
            planning_horizons="2018",
            interconnect="western",
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
    elif result == "production":
        plotting_data = _initialize_metadata(PRODUCTION_PLOTS)
    elif result == "capacity":
        plotting_data = _initialize_metadata(CAPACITY_PLOTS)
    else:
        raise NotImplementedError

    # plot at system level

    for plot_data in plotting_data:

        fn = plot_data.fn
        title = plot_data.nice_name if plot_data.nice_name else plot_data.name

        if plot_data.sector:
            f_path = Path(
                results_dir,
                "system",
                result,
                plot_data.sector,
                f"{plot_data.name}.{EXT}",
            )
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

        if not plot_data.plot_by_month:
            continue

        by_month_kwargs = fn_kwargs.copy()

        by_month_kwargs["resample"] = None
        by_month_kwargs["resample_fn"] = None

        months = {month.value: _get_month_name(month) for month in Month}

        for month_i, month_name in months.items():

            if plot_data.sector:
                f_path = Path(
                    results_dir,
                    "system",
                    result,
                    plot_data.sector,
                    plot_data.name,
                    f"{month_name}.{EXT}",
                )
            else:
                f_path = Path(
                    results_dir,
                    "system",
                    result,
                    plot_data.name,
                    f"{month_name}.{EXT}",
                )

            if not f_path.parent.exists():
                f_path.parent.mkdir(parents=True)

            by_month_kwargs["month"] = month_i

            save_fig(fn, n, str(f_path), title, wildcards, **by_month_kwargs)

    # plot each state

    for plot_data in plotting_data:

        fn = plot_data.fn
        title = plot_data.nice_name if plot_data.nice_name else plot_data.name

        for state in states:

            if plot_data.fn_kwargs:
                fn_kwargs = plot_data.fn_kwargs
            else:
                fn_kwargs = {}
            fn_kwargs["eia_api"] = eia_api

            if plot_data.sector:
                fn_kwargs["sector"] = plot_data.sector
            else:
                fn_kwargs["sector"] = None

            fn_kwargs["state"] = state

            if plot_data.sector:
                f_path = Path(
                    results_dir,
                    state,
                    result,
                    plot_data.sector,
                    f"{plot_data.name}.{EXT}",
                )
            else:
                f_path = Path(results_dir, state, result, f"{plot_data.name}.{EXT}")

            if not f_path.parent.exists():
                f_path.parent.mkdir(parents=True)

            save_fig(fn, n, str(f_path), title, wildcards, **fn_kwargs)

        if not plot_data.plot_by_month:
            continue

        by_month_kwargs = fn_kwargs.copy()

        by_month_kwargs["resample"] = None
        by_month_kwargs["resample_fn"] = None

        months = {month.value: _get_month_name(month) for month in Month}

        for month_i, month_name in months.items():

            if plot_data.sector:
                f_path = Path(
                    results_dir,
                    state,
                    result,
                    plot_data.sector,
                    plot_data.name,
                    f"{month_name}.{EXT}",
                )
            else:
                f_path = Path(
                    results_dir,
                    state,
                    result,
                    plot_data.name,
                    f"{month_name}.{EXT}",
                )

            if not f_path.parent.exists():
                f_path.parent.mkdir(parents=True)

            by_month_kwargs["month"] = month_i

            save_fig(fn, n, str(f_path), title, wildcards, **by_month_kwargs)
