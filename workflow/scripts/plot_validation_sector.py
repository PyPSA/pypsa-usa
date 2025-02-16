"""
Plots sector validation plots.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
import pypsa
from _helpers import configure_logging, mock_snakemake
from add_electricity import sanitize_carriers
from constants import STATE_2_CODE
from plot_statistics import create_title
from summary_natural_gas import get_historical_ng_prices, get_ng_price
from summary_sector import (
    get_emission_timeseries_by_sector,
    get_end_use_consumption,
    get_historical_emissions,
    get_historical_end_use_consumption,
    get_historical_power_production,
    get_historical_transport_consumption_by_mode,
    get_power_production_timeseries,
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
    "pwr": "power",
    "ch4": "methane",
}

FIG_WIDTH = 14
FIG_HEIGHT = 6

# figure save format
EXT = "png"


def percent_difference(col_1: pd.Series, col_2: pd.Series) -> pd.Series:
    """
    Calculates percent difference between two columns of numbers.
    """
    return abs(col_1 - col_2).div((col_1 + col_2).div(2)).mul(100)


def plot_sector_emissions_validation(
    n: pypsa.Network,
    eia_api: str,
    state: Optional[str] = None,
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

    nrows = 1

    fig, axs = plt.subplots(
        ncols=1,
        nrows=nrows,
        figsize=(FIG_WIDTH, FIG_HEIGHT * nrows),
        sharey=False,
    )

    modelled = modelled.T
    historical = historical.T

    for sector in modelled.columns:
        if sector not in historical.columns:
            historical[sector] = 0
    assert set(historical.columns) == set(modelled.columns)

    modelled = modelled.T
    historical = historical.T

    if state:  # plot at state level
        historical = historical[state].to_frame("Actual")
        modelled = modelled[state].to_frame("Modelled")

    else:  # plot at system level
        historical = historical[modelled.columns].sum(axis=1).to_frame("Actual")
        modelled = modelled.sum(axis=1).to_frame("Modelled")

    df = historical.join(modelled)

    try:
        df.plot.bar(ax=axs, stacked=False)
        axs.set_xlabel("")
        axs.set_ylabel("Emissions (MT)")
        axs.set_title("Emissions by Sector")
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


def _get_annual_generation(n: pypsa.Network, year: int, state) -> pd.DataFrame:
    """
    Only for comparing agaist EIA data.
    """
    df = get_power_production_timeseries(n, False, state)
    df = df.T
    df.index = df.index.map(pd.concat([n.links.carrier, n.generators.carrier]))
    # collapse CCS techs into a single tech
    df = df.rename(index={x: x.split("-9")[0] for x in df.index})
    df = df.rename(index={"OCGT": "gas", "CCGT": "gas"})  # only one historical gas
    df = df.rename(
        index={"onwind": "wind", "offwind_floating": "wind", "offwind_fixed": "wind"},
    )
    df = df.groupby(level=0).sum().T
    return df.loc[year].sum().to_frame(name="modelled")


def plot_power_generation_validation(
    n: pypsa.Network,
    eia_api: str,
    state: Optional[str] = None,
    **kwargs,
) -> tuple:
    investment_period = n.investment_periods[0]

    modelled = _get_annual_generation(n, investment_period, state)

    historical = get_historical_power_production(
        investment_period,
        eia_api,
    )
    if not state:
        historical = historical.loc["U.S."].to_frame("actual")
    else:
        historical = historical.loc[state].to_frame("actual")

    df = historical.join(modelled, how="outer").fillna(0)

    fig, axs = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
    )

    try:
        df.plot.bar(ax=axs)
        axs.set_xlabel("")
        axs.set_ylabel("MWh")
        axs.set_title("Electric Power Production")
        axs.tick_params(axis="x", labelrotation=45)
    except TypeError:  # no numeric data to plot
        logger.warning(f"No data to plot for {state}")

    return fig, axs


def plot_ng_price_validation(
    n: pypsa.Network,
    eia_api: str,
    state: Optional[str] = None,
    **kwargs,
) -> tuple:
    investment_period = n.investment_periods[0]

    modelled = get_ng_price(n)
    modelled = {x: y.loc[investment_period] for x, y in modelled.items()}

    historical_power = get_historical_ng_prices(investment_period, "power", eia_api)
    historical_residential = get_historical_ng_prices(
        investment_period,
        "residential",
        eia_api,
    )
    historical_commercial = get_historical_ng_prices(
        investment_period,
        "commercial",
        eia_api,
    )
    historical_industrial = get_historical_ng_prices(
        investment_period,
        "industrial",
        eia_api,
    )

    if not state:
        historical_power = historical_power["U.S."].to_frame("Power")
        historical_residential = historical_residential["U.S."].to_frame("Residential")
        historical_commercial = historical_commercial["U.S."].to_frame("Commercial")
        historical_industrial = historical_industrial["U.S."].to_frame("Industrial")
        modelled = pd.concat(list(modelled.values()), axis=1).mean(axis=1).to_frame(name="Modelled")
    else:
        historical_power = historical_power[state].to_frame("Power")
        historical_residential = historical_residential[state].to_frame("Residential")
        historical_commercial = historical_commercial[state].to_frame("Commercial")
        historical_industrial = historical_industrial[state].to_frame("Industrial")
        modelled = modelled[state].mean(axis=1).to_frame(name="Modelled")

    df = (
        modelled.join(historical_power, how="left")
        .join(historical_residential, how="left")
        .join(historical_commercial, how="left")
        .join(historical_industrial, how="left")
    )

    fig, axs = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
    )

    try:
        df.plot.line(ax=axs)
        axs.set_xlabel("")
        axs.set_ylabel("$/MMBTU")
        axs.set_title("Natural Gas Prices")
        axs.tick_params(axis="x", labelrotation=45)
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
        axs.set_ylabel("Energy Consumption by Transport Mode (MWh)")
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


@dataclass
class PlottingData:
    name: str  # snakemake name
    fn: callable
    system_only: bool
    sector: Optional[str] = None  # None = 'system'
    fn_kwargs: Optional[dict[str, Any]] = None
    nice_name: Optional[str] = None


VALIDATION_PLOTS = [
    {
        "name": "natual_gas_price",
        "fn": plot_ng_price_validation,
        "nice_name": "Natural Gas by State",
        "system_only": False,
    },
    {
        "name": "power_generation",
        "fn": plot_power_generation_validation,
        "nice_name": "Power Generation by State",
        "system_only": False,
    },
    {
        "name": "emissions_by_sector_validation",
        "fn": plot_sector_emissions_validation,
        "nice_name": "Emissions by Sector",
        "system_only": False,
    },
    {
        "name": "emissions_by_state_validation",
        "fn": plot_state_emissions_validation,
        "nice_name": "Emissions by State",
        "system_only": True,
    },
    {
        "name": "generation_by_state_validation",
        "fn": plot_sector_consumption_validation,
        "nice_name": "Energy Consumption",
        "system_only": False,
    },
    # {
    #     "name": "transportation_by_mode_validation",
    #     "fn": plot_transportation_by_mode_validation,
    #     "nice_name": "Residenital Capacity",
    #     "sector": "res",
    # },
    # {
    #     "name": "system_consumption_validation",
    #     "fn": plot_system_consumption_validation_by_state,
    #     "nice_name": "Residenital Capacity",
    #     "sector": "res",
    # },
    # {
    #     "name": "system_emission_validation_state",
    #     "fn": plot_system_emissions_validation_by_state,
    #     "nice_name": "Residenital Capacity",
    #     "sector": "res",
    # },
]


def _initialize_metadata(data: dict[str, Any]) -> list[PlottingData]:
    return [PlottingData(**x) for x in data]


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


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_sector_validation",
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
    eia_api = params.get("eia_api", None)

    states = n.buses[n.buses.reeds_state != ""].reeds_state.unique().tolist()

    plotting_data = _initialize_metadata(VALIDATION_PLOTS)

    for plot_data in plotting_data:
        fn = plot_data.fn
        title = plot_data.nice_name if plot_data.nice_name else plot_data.name

        if plot_data.sector:
            f_path = Path(
                results_dir,
                "system",
                "validation",
                plot_data.sector,
                f"{plot_data.name}.{EXT}",
            )
        else:
            f_path = Path(
                results_dir,
                "system",
                "validation",
                f"{plot_data.name}.{EXT}",
            )

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

        if plot_data.system_only:
            continue

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
                    "validation",
                    plot_data.sector,
                    f"{plot_data.name}.{EXT}",
                )
            else:
                f_path = Path(
                    results_dir,
                    state,
                    "validation",
                    f"{plot_data.name}.{EXT}",
                )

            if not f_path.parent.exists():
                f_path.parent.mkdir(parents=True)

            save_fig(fn, n, str(f_path), title, wildcards, **fn_kwargs)
