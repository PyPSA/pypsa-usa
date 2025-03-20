"""
Plots static and interactive charts to analyze system results.

**Inputs**

A solved network

**Outputs**

System level charts for:
    - Hourly production
    - Generator costs
    - Generator capacity

    .. image:: _static/plots/production-area.png
        :scale: 33 %

    .. image:: _static/plots/costs-bar.png
        :scale: 33 %

    .. image:: _static/plots/capacity-bar.png
        :scale: 33 %

Emission charts for:
    - Accumulated emissions

    .. image:: _static/plots/emissions-area.png
        :scale: 33 %
"""

import logging
import math
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
import seaborn as sns
from _helpers import configure_logging
from add_electricity import sanitize_carriers
from summary import (
    get_demand_timeseries,
    get_energy_timeseries,
    get_fuel_costs,
    get_generator_marginal_costs,
    get_node_emissions_timeseries,
    get_tech_emissions_timeseries,
)

logger = logging.getLogger(__name__)

# Global Plotting Settings
TITLE_SIZE = 16


def get_color_palette(n: pypsa.Network) -> pd.Series:
    """Returns colors based on nice name."""
    colors = (n.carriers.reset_index().set_index("nice_name")).color

    # Initialize the additional dictionary
    additional = {
        "co2": "k",
    }

    # Loop through the carriers DataFrame
    for index, row in n.carriers.iterrows():
        if "battery" in index or "PHS" in index:
            color = row.color
            additional.update(
                {
                    f"{index}_charger": color,
                    f"{index}_discharger": color,
                },
            )

    return pd.concat([colors, pd.Series(additional)]).to_dict()


def create_title(title: str, **wildcards) -> str:
    """
    Standardizes wildcard writing in titles.

    Arguments:
        title: str
            Title of chart to plot
        **wildcards
            any wildcards to add to title
    """
    w = []
    for wildcard, value in wildcards.items():
        if wildcard == "interconnect":
            w.append(f"interconnect = {value}")
        elif wildcard == "clusters":
            w.append(f"#clusters = {value}")
        elif wildcard == "ll":
            w.append(f"ll = {value}")
        elif wildcard == "opts":
            w.append(f"opts = {value}")
        elif wildcard == "sector":
            w.append(f"sectors = {value}")
    wildcards_joined = " | ".join(w)
    return f"{title} \n ({wildcards_joined})"


## Data Gathering Functions ##
def get_currently_installed_capacity(n: pypsa.Network, region_name="nerc_reg") -> pd.DataFrame:
    """Returns a DataFrame with the currently installed capacity for each carrier and nerc region."""
    if region_name not in n.buses.columns:
        region_name = "all"
        n.buses[region_name] = "all"
    n.generators[region_name] = n.generators.bus.map(n.buses[region_name])
    existing_capacity = n.generators.groupby([region_name, "carrier"]).p_nom.sum().round(0)
    existing_capacity = existing_capacity.to_frame(name="Existing")
    n.storage_units[region_name] = n.storage_units.bus.map(n.buses[region_name])
    storage_units = n.storage_units.groupby([region_name, "carrier"]).p_nom.sum().round(0)
    storage_units = storage_units.to_frame(name="Existing")
    existing_capacity = pd.concat([existing_capacity, storage_units])

    # Groupby regions and carriers, then fix indexing
    existing_capacity = existing_capacity.groupby(existing_capacity.index).sum()
    existing_capacity = existing_capacity.reset_index()
    existing_capacity[["Region", "Carrier"]] = pd.DataFrame(
        existing_capacity["index"].tolist(),
        index=existing_capacity.index,
    )
    existing_capacity = existing_capacity.drop(columns="index")
    existing_capacity = existing_capacity.set_index(["Region", "Carrier"])

    nn_carriers = existing_capacity.index.get_level_values(1).map(n.carriers.nice_name)
    existing_capacity = existing_capacity.droplevel(1)
    existing_capacity = existing_capacity.set_index(nn_carriers, append=True)
    return existing_capacity


def get_statistics(n, statistic_name, component_types=["Generator", "StorageUnit"], region_name="nerc_reg"):
    """
    Prepare network statistics data for plotting by extracting, organizing by region and carrier.

    This function extracts specific statistics from the network, adds regional information,
    and groups the data by region and carrier for easier plotting and analysis.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object containing the data to analyze
    statistic_name : str
        The name of the statistic to extract (e.g., 'Optimal Capacity', 'Supply')
    component_types : list, default=["Generator", "StorageUnit"]
        List of component types to include in the statistics
    region_name : str, default="nerc_reg"
        The column name in n.buses that contains regional designation

    Returns
    -------
    pd.DataFrame
        DataFrame with statistics grouped by region and carrier, with index levels [Region, Carrier]
    """
    groupers = n.statistics.groupers
    df = n.statistics(groupby=groupers.get_name_bus_and_carrier).round(3)
    df = df.loc[component_types]

    # Add region information to the data
    if region_name not in n.buses.columns:
        region_name = "all"
        n.buses[region_name] = "all"
    gens = df.loc["Generator"].index.get_level_values(0)
    gens_reg = gens.map(n.generators.bus.map(n.buses[region_name])).to_series()
    su = df.loc["StorageUnit"].index.get_level_values(0)
    su_reg = su.map(n.storage_units.bus.map(n.buses[region_name])).to_series()
    region_components = pd.concat([gens_reg, su_reg])

    df = df.set_index(region_components, append=True)
    df = df.droplevel([0, 1, 2])
    df = df.reset_index()
    df = df.rename(columns={"level_0": "carrier", "level_1": "region"})
    df = df.set_index(["region", "carrier"])

    df_statistic = df[statistic_name]
    df_statistic = df_statistic.groupby(df_statistic.index).sum()
    df_statistic = df_statistic.reset_index()
    df_statistic[["Region", "Carrier"]] = pd.DataFrame(
        df_statistic["index"].tolist(),
        index=df_statistic.index,
    )
    df_statistic = df_statistic.drop(columns="index")
    df_statistic = df_statistic.set_index(["Region", "Carrier"])

    return df_statistic


#### Bar Plots ####
def plot_bar(data, n, save, title, ylabel, existing_cap=None):
    """
    Plot the data in a bar chart with subplots by region and carrier.

    Parameters.
    ----------
    - data: pd.DataFrame, data to plot
    - n: pypsa.Network
    - save: str, file path to save the plot
    - title: str, plot title
    - ylabel: str, y-axis label
    - existing_cap: pd.DataFrame, optional, existing capacity data
    """
    if existing_cap is not None:
        data.loc[existing_cap.index, "Existing"] = existing_cap
        data = data[["Existing"] + [col for col in data.columns if col != "Existing"]]
        retirements = data.diff(axis=1).clip(upper=0)
        retirements = retirements[(retirements < -0.001).any(axis=1)]
        retirements = retirements.fillna(0)
        data = pd.concat([data, retirements])

    data = data / 1e3  # Convert to GW
    palette = n.carriers.set_index("nice_name").color.to_dict()
    regions = data.index.get_level_values(0).unique()

    # Determine grid layout for subplots
    num_regions = len(regions)
    columns = min(5, num_regions)  # Limit to 5 columns
    rows = math.ceil(num_regions / columns)

    # Calculate legend parameters
    carriers = data.reset_index().Carrier.unique()
    num_carriers = len(carriers)

    # Determine optimal number of legend columns based on carrier name lengths
    max_name_length = max([len(str(carrier)) for carrier in carriers])
    # Fewer columns for longer names to prevent overlap
    if max_name_length > 15:
        legend_cols = 2
    elif max_name_length > 10:
        legend_cols = 3
    else:
        legend_cols = 4

    # Calculate needed legend height - more rows = more height needed
    legend_rows = math.ceil(num_carriers / legend_cols)
    legend_height_ratio = 0.07 * legend_rows  # Adjust this factor as needed

    # Create figure with extra space for legend
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 3, rows * 5 + legend_rows * 0.4),  # Add height for legend
        sharex=True,
        sharey=True,
    )

    # Ensure axes is a flattened array for consistent indexing
    if num_regions == 1:
        axes = [axes]  # Wrap single Axes object in a list
    else:
        axes = axes.flatten()

    for i, region in enumerate(regions):
        region_data = data.loc[region]
        region_data.T.plot(
            kind="bar",
            stacked=True,
            ax=axes[i],
            color=[palette.get(carrier) for carrier in region_data.index.get_level_values(0)],
            legend=False,
        )
        axes[i].axhline(0, color="black", linewidth=0.8)
        axes[i].set_title(region)
        axes[i].set_ylabel(ylabel)
        axes[i].set_xlabel("")

    # Remove unused subplots
    if num_regions > 1:
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

    # Create legend elements
    handles, labels = [], []
    for carrier in carriers:
        handle = plt.Rectangle((0, 0), 1, 1, color=palette[carrier])
        handles.append(handle)
        labels.append(f"{carrier}")

    # Calculate the right amount of space to reserve for the legend
    bottom_margin = min(0.25, legend_height_ratio)  # Cap at 25% of figure height

    # Apply tight layout to make room for the legend at the bottom
    plt.tight_layout(rect=[0, bottom_margin, 1, 0.95])

    # Add the legend with better formatting
    legend = fig.legend(
        handles,
        labels,
        title="Carrier",
        loc="lower center",
        ncol=legend_cols,
        fontsize=10,  # Slightly smaller font size
        bbox_to_anchor=(0.5, 0),  # Center at bottom
        frameon=True,  # Add frame around legend
        columnspacing=1.5,  # More space between columns
    )

    # Adjust the legend title
    legend.get_title().set_fontsize(11)

    # Add title
    fig.suptitle(title, y=0.98)

    # Save figure with tight bounding box
    fig.savefig(save, bbox_inches="tight", dpi=300)
    plt.close()


def plot_regional_capacity_additions_bar(n, region_name, save):
    """Plot capacity evolution by different region aggregations in a stacked bar plot."""
    data = get_statistics(n, statistic_name="Optimal Capacity", region_name=region_name)
    existing_cap = get_currently_installed_capacity(n, region_name=region_name)
    data.to_csv(f"{Path(save).parent.parent}/statistics/bar_regional_capacity_{region_name}.csv")
    plot_bar(data, n, save, "", "Capacity (GW)", existing_cap)


def plot_regional_production_bar(n, region_name, save):
    """Plot production evolution by NERC region in a stacked bar plot."""
    data = get_statistics(n, "Supply", region_name=region_name)
    data.to_csv(f"{Path(save).parent.parent}/statistics/bar_regional_production_{region_name}.csv")
    plot_bar(data, n, save, "", "Production (GWh)")


def plot_regional_emissions_bar(
    n: pypsa.Network,
    save: str,
) -> None:
    """PLOT OF CO2 EMISSIONS BY NERC REGION AND INVESTMENT PERIOD."""
    regional_emisssions_ts = get_node_emissions_timeseries(n).T.groupby(n.buses.nerc_reg).sum().T / 1e6
    regional_emissions = (
        regional_emisssions_ts.groupby(regional_emisssions_ts.index.get_level_values(0)).sum().round(3).T
    )

    # Determine grid layout for subplots
    regions = regional_emissions.index.get_level_values(0).unique()
    num_regions = len(regions)
    columns = min(5, num_regions)  # Limit to 5 columns
    rows = math.ceil(num_regions / columns)

    # Set up the figure and axes
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 2.5, rows * 5),
        sharex=True,
        sharey=True,
    )

    # Ensure axes is a flattened array for consistent indexing
    if num_regions == 1:
        axes = [axes]  # Wrap single Axes object in a list
    else:
        axes = axes.flatten()

    for i, region in enumerate(regions):
        region_data = regional_emissions.loc[region]
        region_data.T.plot(
            kind="bar",
            stacked=True,
            ax=axes[i],
            legend=False,
        )
        axes[i].axhline(0, color="black", linewidth=0.8)
        axes[i].set_title(region)
        axes[i].set_ylabel("MMtCo2")
        axes[i].set_xlabel("")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0.3, 1, 1])
    plt.subplots_adjust(wspace=0.4)

    plt.xlabel("")
    plt.ylabel("MMtCO2")

    plt.tight_layout()
    plt.savefig(save)
    plt.close()


def plot_emissions_bar(
    n: pypsa.Network,
    save: str,
) -> None:
    """PLOT OF CO2 EMISSIONS BY INVESTMENT PERIOD."""
    emisssions_ts = get_node_emissions_timeseries(n).T.sum().T / 1e6
    emissions = emisssions_ts.groupby(emisssions_ts.index.get_level_values(0)).sum().round(3).T

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(7, 4))
    emissions.T.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        legend=False,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("MMtCo2")
    ax.set_xlabel("")

    plt.tight_layout(rect=[0, 0.3, 1, 1])
    plt.subplots_adjust(wspace=0.4)

    plt.xlabel("CO2 Emissions [MMtCO2]")
    plt.ylabel("MMtCO2")

    plt.tight_layout()
    plt.savefig(save)
    plt.close()


def plot_seasonal_daily_production(
    n: pypsa.Network,
    carriers_2_plot: list[str],
    save: str,
) -> None:
    """
    Plot daily average production by energy carrier for each season and investment period.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network object containing the data
    carriers_2_plot : list[str]
        List of carriers to include in the plot
    save : str
        Path to save the plot
    wildcards : dict
        Additional parameters for the plot title
    """
    # Get data
    energy_mix = get_energy_timeseries(n).mul(1e-3)  # MW -> GW
    demand = get_demand_timeseries(n).mul(1e-3)  # MW -> GW

    # Process storage units and batteries
    for carrier in energy_mix.columns:
        if "battery" in carrier or carrier in snakemake.params.electricity["extendable_carriers"]["StorageUnit"]:
            energy_mix[carrier + "_discharger"] = energy_mix[carrier].clip(lower=0.0001)
            energy_mix[carrier + "_charger"] = energy_mix[carrier].clip(upper=-0.0001)
            energy_mix = energy_mix.drop(columns=carrier)
            carriers_2_plot.append(f"{carrier}" + "_charger")
            carriers_2_plot.append(f"{carrier}" + "_discharger")

    carriers_2_plot = list(set(carriers_2_plot))
    energy_mix = energy_mix[[x for x in carriers_2_plot if x in energy_mix]]
    energy_mix = energy_mix.rename(columns=n.carriers.nice_name)

    color_palette = get_color_palette(n)

    # Define seasons
    seasons = {
        "Winter": [12, 1, 2],
        "Spring": [3, 4, 5],
        "Summer": [6, 7, 8],
        "Fall": [9, 10, 11],
    }

    # Determine which seasons actually have data
    active_seasons = {}
    for investment_period in n.investment_periods:
        active_seasons[investment_period] = []
        for season_name, season_months in seasons.items():
            season_snapshots = n.snapshots[
                (n.snapshots.get_level_values(0) == investment_period)
                & (n.snapshots.get_level_values(1).month.isin(season_months))
            ]
            if len(season_snapshots) > 0:
                active_seasons[investment_period].append(season_name)

    # Count the maximum number of active seasons across all periods
    max_seasons_per_period = max(len(seasons_list) for seasons_list in active_seasons.values())
    if max_seasons_per_period == 0:
        print("No seasonal data available to plot.")
        return

    # Set up figure - dynamic columns based on available seasons
    num_periods = len(n.investment_periods)

    # Create figure with subplots - one row per investment period, columns only for active seasons
    fig, axs_dict = plt.subplots(
        nrows=num_periods,
        ncols=max_seasons_per_period,
        figsize=(4 * max_seasons_per_period, 4 * num_periods),
        sharey=True,
        sharex=True,
    )

    axs_dict = np.atleast_2d(axs_dict)

    # Track which carriers are actually plotted (for legend)
    plotted_carriers = set()

    # Calculate daily average for each season and investment period
    for i, investment_period in enumerate(n.investment_periods):
        period_active_seasons = active_seasons[investment_period]

        # Skip periods with no data
        if not period_active_seasons:
            continue

        for j, season_name in enumerate(period_active_seasons):
            season_months = seasons[season_name]

            # Get snapshots for this investment period and season
            season_snapshots = n.snapshots[
                (n.snapshots.get_level_values(0) == investment_period)
                & (n.snapshots.get_level_values(1).month.isin(season_months))
            ]

            # Get data for this period and season
            period_season_data = energy_mix.loc[season_snapshots].copy()
            period_season_demand = demand.loc[season_snapshots].copy()

            # Add hour of day information
            period_season_data["hour"] = period_season_data.index.get_level_values(1).hour
            period_season_demand["hour"] = period_season_demand.index.get_level_values(1).hour

            # Calculate daily average by hour
            daily_avg = period_season_data.groupby("hour").mean()
            daily_demand_avg = period_season_demand.groupby("hour").mean()

            # Keep track of carriers in this plot for legend
            plotted_carriers.update(set(daily_avg.columns))

            # Plot
            daily_avg.plot.area(
                ax=axs_dict[i, j],
                alpha=0.7,
                color=color_palette,
                stacked=True,
            )

            daily_demand_avg.plot.line(
                ax=axs_dict[i, j],
                ls="-",
                color="darkblue",
                linewidth=2,
            )

            # Set titles and labels
            axs_dict[i, j].set_title(f"{season_name} - {investment_period}")
            axs_dict[i, j].set_ylabel("Power [GW]")
            axs_dict[i, j].set_xlabel("Hour of Day [utc]")
            axs_dict[i, j].set_xticks(range(0, 24, 3))

            # Turn off individual legends for each plot
            axs_dict[i, j].get_legend().remove() if axs_dict[i, j].get_legend() else None

        # Hide unused axes
        for j in range(len(period_active_seasons), max_seasons_per_period):
            axs_dict[i, j].set_visible(False)

    # Create single legend below all plots
    handles, labels = [], []
    for carrier in plotted_carriers:
        color = color_palette.get(carrier, "gray")  # Default to gray if color not found
        handle = plt.Rectangle((0, 0), 1, 1, color=color)
        handles.append(handle)
        labels.append(carrier)

    # Add demand handle/label
    handles.append(plt.Line2D([0], [0], color="darkblue", linewidth=2))
    labels.append("Demand")

    # Calculate legend parameters
    num_items = len(handles)
    legend_cols = min(5, num_items)  # Limit to 5 columns, adjust as needed

    # Add overall title and adjust layout for legend space at bottom
    plt.tight_layout(rect=[0, 0.2, 1, 0.95])  # Reserve bottom 10% for legend
    fig.suptitle("Daily Average Production", fontsize=16)

    # Add legend at the bottom outside plots
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=legend_cols,
        fontsize=8,
        framealpha=0.8,  # Semi-transparent background
        title="Energy Carriers",
    )

    # Save figure
    save_path = Path(save)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_production_area(
    n: pypsa.Network,
    carriers_2_plot: list[str],
    save: str,
    **wildcards,
) -> None:
    """
    Plot timeseries production.

    Will plot an image for the entire time horizon, in addition to
    seperate monthly generation curves
    """
    # get data

    energy_mix = get_energy_timeseries(n).mul(1e-3)  # MW -> GW
    demand = get_demand_timeseries(n).mul(1e-3)  # MW -> GW

    for carrier in energy_mix.columns:
        if "battery" in carrier or carrier in snakemake.params.electricity["extendable_carriers"]["StorageUnit"]:
            energy_mix[carrier + "_discharger"] = energy_mix[carrier].clip(lower=0.0001)
            energy_mix[carrier + "_charger"] = energy_mix[carrier].clip(upper=-0.0001)
            energy_mix = energy_mix.drop(columns=carrier)
            carriers_2_plot.append(f"{carrier}" + "_charger")
            carriers_2_plot.append(f"{carrier}" + "_discharger")
    carriers_2_plot = list(set(carriers_2_plot))
    energy_mix = energy_mix[[x for x in carriers_2_plot if x in energy_mix]]
    energy_mix = energy_mix.rename(columns=n.carriers.nice_name)

    color_palette = get_color_palette(n)

    months = n.snapshots.get_level_values(1).month.unique()
    num_periods = len(n.investment_periods)
    base_plot_size = 4

    for month in ["all", *months.to_list()]:
        figsize = (14, (base_plot_size * num_periods))
        fig, axs = plt.subplots(figsize=figsize, ncols=1, nrows=num_periods)
        if not isinstance(axs, np.ndarray):  # only one horizon
            axs = np.array([axs])
        for i, investment_period in enumerate(n.investment_periods):
            if month == "all":
                sns = n.snapshots[n.snapshots.get_level_values(0) == investment_period]
            else:
                sns = n.snapshots[
                    (n.snapshots.get_level_values(0) == investment_period)
                    & (n.snapshots.get_level_values(1).month == month)
                ]
            energy_mix.loc[sns].droplevel("period").round(2).plot.area(
                ax=axs[i],
                alpha=0.7,
                color=color_palette,
            )
            demand.loc[sns].droplevel("period").round(2).plot.line(
                ax=axs[i],
                ls="-",
                color="darkblue",
            )

            suffix = "-" + datetime.strptime(str(month), "%m").strftime("%b") if month != "all" else ""

            axs[i].legend(bbox_to_anchor=(1, 1), loc="upper left")
            axs[i].set_ylabel("Power [GW]")
            axs[i].set_xlabel("")

        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.suptitle(create_title("Production [GW]", **wildcards))
        save = Path(save)
        fig.savefig(save.parent / (save.stem + suffix + save.suffix))
        plt.close()


def plot_hourly_emissions(n: pypsa.Network, save: str, **wildcards) -> None:
    """Plots snapshot emissions by technology."""
    # get data
    emissions = get_tech_emissions_timeseries(n).mul(1e-6)  # T -> MT
    zeros = emissions.columns[(np.abs(emissions) < 1e-7).all()]
    emissions = emissions.drop(columns=zeros)

    # plot
    color_palette = get_color_palette(n)

    fig, ax = plt.subplots(figsize=(14, 4))
    if not emissions.empty:
        emissions.plot.area(
            ax=ax,
            alpha=0.7,
            legend="reverse",
            color=color_palette,
        )

    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    ax.set_title(create_title("Technology Emissions", **wildcards))
    ax.set_ylabel("Emissions [MT]")
    fig.tight_layout()

    fig.savefig(save)
    plt.close()


def plot_accumulated_emissions_tech(n: pypsa.Network, save: str, **wildcards) -> None:
    """Creates area plot of accumulated emissions by technology."""
    # get data

    emissions = get_tech_emissions_timeseries(n).cumsum().mul(1e-6)  # T -> MT
    zeros = emissions.columns[(np.abs(emissions) < 1e-7).all()]
    emissions = emissions.drop(columns=zeros)

    # plot

    color_palette = get_color_palette(n)

    fig, ax = plt.subplots(figsize=(14, 4))
    if not emissions.empty:
        emissions.plot.area(
            ax=ax,
            alpha=0.7,
            legend="reverse",
            color=color_palette,
        )

    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    ax.set_title(create_title("Technology Accumulated Emissions", **wildcards))
    ax.set_ylabel("Emissions [MT]")
    fig.tight_layout()

    fig.savefig(save)
    plt.close()


def plot_accumulated_emissions(n: pypsa.Network, save: str, **wildcards) -> None:
    """Plots accumulated emissions."""
    # get data

    emissions = get_tech_emissions_timeseries(n).mul(1e-6).sum(axis=1)  # T -> MT
    emissions = emissions.cumsum().to_frame("co2")

    # plot

    color_palette = get_color_palette(n)

    fig, ax = plt.subplots(figsize=(14, 4))

    emissions.plot.area(
        ax=ax,
        alpha=0.7,
        legend="reverse",
        color=color_palette,
    )

    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    ax.set_title(create_title("Accumulated Emissions", **wildcards))
    ax.set_ylabel("Emissions [MT]")
    fig.tight_layout()
    fig.savefig(save)
    plt.close()


#### Panel / Mixed Plots ####
def plot_generator_data_panel(
    n: pypsa.Network,
    save: str,
    **wildcards,
):
    df_capex_expand = n.generators.loc[
        n.generators.p_nom_extendable & ~n.generators.index.str.contains("existing"),
        :,
    ]

    df_storage_units = n.storage_units.loc[n.storage_units.p_nom_extendable, :].copy()
    df_storage_units.loc[:, "efficiency"] = df_storage_units.efficiency_dispatch
    df_capex_expand = pd.concat([df_capex_expand, df_storage_units])

    df_efficiency = n.generators.loc[
        ~n.generators.carrier.isin(
            ["solar", "onwind", "offwind", "offwind_floating", "hydro", "load"],
        ),
        :,
    ]
    # Create a figure and subplots with 2 rows and 2 columns
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))

    # Plot on each subplot
    sns.lineplot(
        data=get_generator_marginal_costs(n),
        x="timestep",
        y="Value",
        hue="Carrier",
        ax=axes[0, 0],
    )
    sns.barplot(data=df_capex_expand, x="carrier", y="capital_cost", ax=axes[0, 1])
    sns.boxplot(data=df_efficiency, x="carrier", y="efficiency", ax=axes[1, 0])

    # Create line plot of declining capital costs
    sns.lineplot(
        data=df_capex_expand[df_capex_expand.build_year > 0],
        x="build_year",
        y="capital_cost",
        hue="carrier",
        ax=axes[2, 0],
    )

    cf_profiles = n.get_switchable_as_dense("Generator", "p_max_pu")
    fuel_costs = n.generators.marginal_cost * cf_profiles.sum()
    n.generators["lcoe"] = (n.generators.capital_cost + fuel_costs) / cf_profiles.sum()
    n.generators["cf"] = cf_profiles.mean()
    lcoe_plot_df = n.generators.loc[
        n.generators.p_nom_extendable & ~n.generators.index.str.contains("existing"),
        :,
    ]

    sns.boxplot(
        data=n.generators,
        x="cf",
        y="carrier",
        ax=axes[1, 1],
    )

    sns.boxplot(
        data=lcoe_plot_df,
        x="lcoe",
        y="carrier",
        ax=axes[2, 1],
    )

    # Set titles for each subplot
    axes[0, 0].set_title("Generator Marginal Costs")
    axes[0, 1].set_title("Extendable Capital Costs")
    axes[1, 0].set_title("Plant Efficiency")
    axes[1, 1].set_title("Capacity Factors by Carrier")
    axes[2, 0].set_title("Expansion Capital Costs by Carrier")
    axes[2, 1].set_title("LCOE by Carrier")

    # Set labels for each subplot
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("$ / MWh")
    # axes[0, 0].set_ylim(0, 200)
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("$ / MW-yr")
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("MWh_primary / MWh_elec")
    axes[1, 1].set_xlabel("p.u.")
    axes[1, 1].set_ylabel("")
    axes[2, 0].set_xlabel("Year")
    axes[2, 0].set_ylabel("$ / MW-yr")
    axes[2, 1].set_xlabel("$ / MWh")
    axes[2, 1].set_ylabel("")

    # Rotate x-axis labels for each subplot
    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=25)

    # Lay legend out horizontally
    axes[0, 0].legend(
        loc="upper left",
        bbox_to_anchor=(1, 1),
        ncol=1,
        fontsize="xx-small",
    )
    axes[2, 0].legend(fontsize="xx-small")

    fig.tight_layout()
    fig.savefig(save)
    plt.close()


def plot_lmp_heatmap(
    n: pypsa.Network,
    save: str,
    max_buses: int = 50,
    temporal_downsample: str | None = None,
    **wildcards,
) -> None:
    """
    Creates a heatmap of bus locational marginal prices (LMPs) over time.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network object containing the data
    save : str
        Path to save the plot
    max_buses : int, optional
        Maximum number of buses to include in plot
    temporal_downsample : str, optional
        Pandas frequency string for downsampling time axis (e.g., 'D' for daily)
    wildcards : dict
        Additional parameters for the plot title
    """
    # Extract LMP data
    lmp_data = n.buses_t.marginal_price.copy()

    # Check if we have multi-period data
    is_multi_period = hasattr(n, "investment_periods") and len(n.investment_periods) > 0

    # Process LMP data for all periods
    if is_multi_period:
        # Create a combined figure for all periods
        fig, axes = plt.subplots(
            len(n.investment_periods),
            1,
            figsize=(15, 5 * len(n.investment_periods)),
            sharex=True,
        )

        # If only one period, make axes iterable
        if len(n.investment_periods) == 1:
            axes = [axes]

        # Find global min/max for consistent color scale
        global_min = lmp_data.min().min()
        global_max = lmp_data.max().max()

        # For sorting buses consistently across periods
        all_periods_avg = lmp_data.groupby(level=0).mean().mean()
        sorted_buses = all_periods_avg.sort_values().index

        # Limit number of buses if needed
        if len(sorted_buses) > max_buses:
            # Take sampling from high, medium, and low prices
            num_each = max_buses // 3
            high_price_buses = sorted_buses[-num_each:]
            low_price_buses = sorted_buses[:num_each]

            # Get middle price buses
            mid_idx = len(sorted_buses) // 2
            remaining = max_buses - (len(high_price_buses) + len(low_price_buses))
            mid_price_buses = sorted_buses[mid_idx - remaining // 2 : mid_idx + remaining // 2]

            # Combine selections
            selected_buses = pd.Index(list(low_price_buses) + list(mid_price_buses) + list(high_price_buses))
        else:
            selected_buses = sorted_buses

        # Plot each period
        for i, period in enumerate(n.investment_periods):
            # Get data for this period
            period_data = lmp_data.loc[period]

            # Downsample time if specified
            if temporal_downsample:
                period_data = period_data.resample(temporal_downsample).mean()

            # Use consistent bus selection
            period_data = period_data[selected_buses]

            # Create the heatmap
            sns.heatmap(
                period_data.T,  # Transpose so buses are on y-axis and time on x-axis
                cmap="Reds",
                center=(global_min + global_max) / 2,  # Center colormap on the global mean
                vmin=global_min,
                vmax=global_max,
                cbar=(i == 0),  # Only show colorbar for first plot
                ax=axes[i],
                cbar_kws={"label": "LMP [$/MWh]"} if i == 0 else {},
            )

            # Rotate y tick labels 90 degrees clockwise and make font smaller
            axes[i].set_yticklabels(axes[i].get_yticklabels(), rotation=90, fontsize=5)

            # Add period label
            axes[i].set_title(f"Investment Period {period}")
            axes[i].set_ylabel("Bus")

            # Only show x-axis label for the last subplot
            if i == len(n.investment_periods) - 1:
                axes[i].set_xlabel("Time")
            else:
                axes[i].set_xlabel("")

            # Adjust x-axis ticks for readability
            if len(period_data.index) > 20:
                tick_indices = np.linspace(0, len(period_data.index) - 1, 10).astype(int)
                axes[i].set_xticks(tick_indices)
                axes[i].set_xticklabels([str(period_data.index[j]) for j in tick_indices], rotation=45)

        # Add overall title
        fig.suptitle(create_title("Bus LMP Heatmap", **wildcards), fontsize=16, y=0.98)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save the plot
        plt.savefig(save, bbox_inches="tight", dpi=300)
        plt.close()


# Fuel costs


def plot_fuel_costs(
    n: pypsa.Network,
    save: str,
) -> None:
    fuel_costs = get_fuel_costs(n)

    fuels = set(fuel_costs.index.get_level_values("carrier"))

    fig, axs = plt.subplots(len(fuels) + 1, 1, figsize=(20, 40))

    color_palette = n.carriers.color.to_dict()

    # plot error plot of all fuels
    df = fuel_costs.droplevel(["bus", "Generator"]).T.resample("d").mean().reset_index().melt(id_vars="timestep")
    sns.lineplot(
        data=df,
        x="timestep",
        y="value",
        hue="carrier",
        ax=axs[0],
        legend=True,
        palette=color_palette,
    )
    (axs[0].set_title("Daily Average Fuel Costs [$/MWh]"),)
    (axs[0].set_xlabel(""),)
    (axs[0].set_ylabel("$/MWh"),)

    # plot bus fuel prices for each fuel
    for i, fuel in enumerate(fuels):
        nice_name = n.carriers.at[fuel, "nice_name"]
        df = fuel_costs.loc[fuel, :, :].droplevel("Generator").T.resample("d").mean().T.groupby(level=0).mean().T
        sns.lineplot(
            data=df,
            legend=False,
            palette="muted",
            dashes=False,
            ax=axs[i + 1],
        )
        (axs[i + 1].set_title(f"Daily Average {nice_name} Fuel Costs per Bus [$/MWh]"),)
        (axs[i + 1].set_xlabel(""),)
        (axs[i + 1].set_ylabel("$/MWh"),)

    fig.savefig(save)
    plt.close()


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_statistics",
            interconnect="texas",
            clusters=7,
            ll="v1.00",
            opts="REM-400SEG",
            sector="E",
        )
    configure_logging(snakemake)

    # extract shared plotting files
    n = pypsa.Network(snakemake.input.network)
    onshore_regions = gpd.read_file(snakemake.input.regions_onshore)
    retirement_method = snakemake.params.retirement

    sanitize_carriers(n, snakemake.config)

    # carriers to plot
    carriers = (
        snakemake.params.electricity["conventional_carriers"]
        + snakemake.params.electricity["renewable_carriers"]
        + snakemake.params.electricity["extendable_carriers"]["Generator"]
        + snakemake.params.electricity["extendable_carriers"]["StorageUnit"]
        + snakemake.params.electricity["extendable_carriers"]["Store"]
        + snakemake.params.electricity["extendable_carriers"]["Link"]
        + ["battery_charger", "battery_discharger"]
    )
    carriers = list(set(carriers))  # remove any duplicates

    # Export Statistics Tables
    groupers = n.statistics.groupers
    n.statistics(groupby=groupers.get_name_bus_and_carrier).round(3).to_csv(
        snakemake.output.statistics_dissaggregated_name_bus_carrier,
    )
    n.statistics(groupby=groupers.get_bus_and_carrier).round(3).to_csv(
        snakemake.output.statistics_dissaggregated_bus_carrier,
    )
    n.statistics().round(2).to_csv(snakemake.output.statistics_summary)
    n.generators.to_csv(snakemake.output.generators)
    n.storage_units.to_csv(snakemake.output.storage_units)
    n.links.to_csv(snakemake.output.links)
    n.lines.to_csv(snakemake.output.lines)

    # Bar Plots
    regions_to_plot = ["reeds_state", "nerc_reg", "interconnect", "all"]
    for region in regions_to_plot:
        logger.info(f"Plotting regional statistics for {region}")
        plot_regional_capacity_additions_bar(
            n,
            region,
            snakemake.output[f"bar_regional_capacity_{region}.pdf"],
        )

        plot_regional_production_bar(
            n,
            region,
            snakemake.output[f"bar_regional_production_{region}.pdf"],
        )

    plot_regional_emissions_bar(
        n,
        snakemake.output["bar_regional_emissions.pdf"],
    )
    plot_emissions_bar(
        n,
        snakemake.output["bar_emissions.pdf"],
    )

    # Temporal Plots
    plot_seasonal_daily_production(
        n,
        carriers,
        snakemake.output["seasonal_daily_production.pdf"],
    )

    plot_production_area(
        n,
        carriers,
        snakemake.output["production_area.pdf"],
        **snakemake.wildcards,
    )
    plot_hourly_emissions(
        n,
        snakemake.output["emissions_area.pdf"],
        **snakemake.wildcards,
    )
    plot_accumulated_emissions_tech(
        n,
        snakemake.output["emissions_accumulated_tech.pdf"],
        **snakemake.wildcards,
    )
    plot_accumulated_emissions(
        n,
        snakemake.output["emissions_accumulated.pdf"],
        **snakemake.wildcards,
    )

    plot_fuel_costs(
        n,
        snakemake.output["fuel_costs.pdf"],
    )

    # LMP Heatmap
    plot_lmp_heatmap(
        n,
        snakemake.output["lmp_heatmap.pdf"],
        **snakemake.wildcards,
    )
    # potentially remove
    # plot_generator_data_panel(
    #     n,
    #     snakemake.output["generator_data_panel.pdf"],
    #     **snakemake.wildcards,
    # )

    # Thing to change to csv outputs
    # Box Plot
    # plot_region_lmps(
    #     n,
    #     snakemake.output["region_lmps.pdf"],
    #     **snakemake.wildcards,
    # )

#     shadow_prices = n.global_constraints.mu.round(3).reset_index()
