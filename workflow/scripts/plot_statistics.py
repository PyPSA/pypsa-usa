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

logger = logging.getLogger(__name__)
from _helpers import configure_logging
from add_electricity import sanitize_carriers
from plot_network_maps import get_color_palette
from summary import (
    get_demand_timeseries,
    get_energy_timeseries,
    get_fuel_costs,
    get_generator_marginal_costs,
    get_node_emissions_timeseries,
    get_tech_emissions_timeseries,
)

# Global Plotting Settings
TITLE_SIZE = 16


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


def stacked_bar_horizons(
    stats,
    variable,
    variable_units,
    carriers,
):
    carriers = carriers.set_index("nice_name")
    colors_ = carriers["color"]
    carriers_legend = carriers  # to track which carriers have non-zero values
    # Create subplots
    planning_horizons = stats[list(stats.keys())[0]].columns
    fig, axes = plt.subplots(
        nrows=len(planning_horizons),
        ncols=1,
        figsize=(8, 1.2 * len(planning_horizons)),
        sharex=True,
    )

    # Ensure axes is always iterable (even if there's only one planning horizon)
    if len(planning_horizons) == 1:
        axes = [axes]

    # Loop through each planning horizon
    for ax, horizon in zip(axes, planning_horizons):
        y_positions = np.arange(len(stats))  # One position for each scenario
        for j, (scenario, df) in enumerate(stats.items()):
            bottoms = np.zeros(
                len(df.columns),
            )  # Initialize the bottom positions for stacking
            # Stack the technologies for each scenario
            for i, technology in enumerate(df.index.unique()):
                values = df.loc[technology, horizon]
                values = values / (1e3) if "GW" in variable_units else values
                ax.barh(
                    y_positions[j],
                    values,
                    left=bottoms[j],
                    color=colors_[technology],
                    label=technology if j == 0 else "",
                )
                bottoms[j] += values
                carriers_legend.loc[technology, "value"] = values

        # Set the title for each subplot
        ax.text(
            1.01,
            0.5,
            f"{horizon}",
            transform=ax.transAxes,
            va="center",
            rotation="vertical",
        )
        ax.set_yticks(y_positions)  # Positioning scenarios on the y-axis
        ax.set_yticklabels(stats.keys())  # Labeling y-axis with scenario names
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    # Create legend handles and labels from the carriers DataFrame
    carriers_legend = carriers_legend[carriers_legend["value"] > 0.01]
    colors_ = carriers_legend["color"]
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=colors_[tech]) for tech in carriers_legend.index]
    # fig.legend(handles=legend_handles, labels=carriers.index.tolist(), loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=4, title='Technologies')
    ax.legend(
        handles=legend_handles,
        labels=carriers_legend.index.tolist(),
        loc="upper center",
        bbox_to_anchor=(0.5, -1.3),
        ncol=4,
        title="Technologies",
    )

    fig.subplots_adjust(hspace=0, bottom=0.5)
    fig.suptitle(f"{variable}", fontsize=12, fontweight="bold")
    plt.xlabel(f"{variable} {variable_units}")
    # fig.tight_layout()
    # plt.show(block=True)
    return fig


#### Bar Plots ####
def plot_capacity_additions_bar(
    n: pypsa.Network,
    carriers_2_plot: list[str],
    save: str,
    **wildcards,
) -> None:
    """
    Plots base capacity vs optimal capacity as a bar chart.
    """
    existing_capacity = n.generators.groupby("carrier").p_nom.sum().round(0)
    existing_capacity = existing_capacity.to_frame(name="Existing Capacity")
    storage_units = n.storage_units.groupby("carrier").p_nom.sum().round(0)
    storage_units = storage_units.to_frame(name="Existing Capacity")
    existing_capacity = pd.concat([existing_capacity, storage_units])
    existing_capacity.index = existing_capacity.index.map(n.carriers.nice_name)

    optimal_capacity = n.statistics.optimal_capacity()
    optimal_capacity = optimal_capacity[optimal_capacity.index.get_level_values(0).isin(["Generator", "StorageUnit"])]
    optimal_capacity.index = optimal_capacity.index.droplevel(0)
    optimal_capacity.reset_index(inplace=True)
    optimal_capacity.rename(columns={"index": "carrier"}, inplace=True)

    optimal_capacity.set_index("carrier", inplace=True)
    optimal_capacity.insert(0, "Existing", existing_capacity["Existing Capacity"])
    optimal_capacity = optimal_capacity.fillna(0)

    stats = {"": optimal_capacity}
    variable = "Optimal Capacity"
    variable_units = " GW"
    fig_ = stacked_bar_horizons(stats, variable, variable_units, n.carriers)
    fig_.savefig(save)
    plt.close()


def plot_production_bar(
    n: pypsa.Network,
    carriers_2_plot: list[str],
    save: str,
    **wildcards,
) -> None:
    """
    Plot diaptch per carrier.
    """
    energy_mix = n.statistics.supply().round(0)
    energy_mix = energy_mix[
        energy_mix.index.get_level_values("component").isin(
            ["Generator", "StorageUnit"],
        )
    ]
    energy_mix.index = energy_mix.index.droplevel(0)
    energy_mix = energy_mix.fillna(0)
    stats = {"": energy_mix}
    variable = "Energy Mix"
    variable_units = " GWh"

    fig_ = stacked_bar_horizons(stats, variable, variable_units, n.carriers)
    fig_.savefig(save)
    plt.close()


def plot_global_constraint_shadow_prices(
    n: pypsa.Network,
    save: str,
    **wildcards,
) -> None:
    """
    Plots shadow prices on global constraints.
    """
    shadow_prices = n.global_constraints.mu.round(3).reset_index()

    # plot data
    fig, ax = plt.subplots(figsize=(10, 10))

    sns.barplot(
        y=shadow_prices.GlobalConstraint,
        x=shadow_prices.mu,
        data=shadow_prices,
        color="purple",
        ax=ax,
    )

    ax.set_title(create_title("Shadow Prices on Constraints", **wildcards))
    ax.set_ylabel("")
    ax.set_xlabel("Shadow Price [$/MWh]")
    fig.tight_layout()
    fig.savefig(save)
    plt.close()


def get_currently_installed_capacity(n: pypsa.Network) -> pd.DataFrame:
    """
    Returns a DataFrame with the currently installed capacity for each carrier and nerc region.
    """
    n.generators["nerc_reg"] = n.generators.bus.map(n.buses.nerc_reg)
    existing_capacity = n.generators.groupby(["nerc_reg", "carrier"]).p_nom.sum().round(0)
    existing_capacity = existing_capacity.to_frame(name="Existing")
    n.storage_units["nerc_reg"] = n.storage_units.bus.map(n.buses.nerc_reg)
    storage_units = n.storage_units.groupby(["nerc_reg", "carrier"]).p_nom.sum().round(0)
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
    existing_capacity.set_index(["Region", "Carrier"], inplace=True)

    nn_carriers = existing_capacity.index.get_level_values(1).map(n.carriers.nice_name)
    existing_capacity = existing_capacity.droplevel(1)
    existing_capacity.set_index(nn_carriers, append=True, inplace=True)
    return existing_capacity


def get_statistics(n, column_name):
    """
    Prepare the statistics data for plotting by extracting and grouping by region and carrier.

    Parameters:
    - n: pypsa.Network
    - column_name: str, the name of the column to extract from statistics (e.g., 'Optimal Capacity', 'Supply')

    Returns:
    - pd.DataFrame: Prepared and grouped data
    """
    groupers = n.statistics.groupers
    df = n.statistics(groupby=groupers.get_name_bus_and_carrier).round(3)
    df = df.loc[["Generator", "StorageUnit"]]

    # Add nerc_region data
    gens = df.loc["Generator"].index.get_level_values(0)
    gens_reg = gens.map(n.generators.bus.map(n.buses.nerc_reg)).to_series()
    su = df.loc["StorageUnit"].index.get_level_values(0)
    su_reg = su.map(n.storage_units.bus.map(n.buses.nerc_reg)).to_series()
    nerc_reg = pd.concat([gens_reg, su_reg])

    df.set_index(nerc_reg, append=True, inplace=True)
    df = df.droplevel([0, 1, 2])
    df.reset_index(inplace=True)
    df.rename(columns={"level_0": "carrier", "level_1": "region"}, inplace=True)
    df.set_index(["region", "carrier"], inplace=True)

    df_selected = df[column_name]
    df_selected = df_selected.groupby(df_selected.index).sum()
    df_selected = df_selected.reset_index()
    df_selected[["Region", "Carrier"]] = pd.DataFrame(
        df_selected["index"].tolist(),
        index=df_selected.index,
    )
    df_selected = df_selected.drop(columns="index")
    df_selected.set_index(["Region", "Carrier"], inplace=True)

    return df_selected


def plot_bar(data, n, save, title, ylabel, is_capacity=False):
    """
    Plot the data in a bar chart with subplots by region and carrier.

    Parameters:
    - data: pd.DataFrame, data to plot
    - n: pypsa.Network
    - save: str, file path to save the plot
    - title: str, plot title
    - ylabel: str, y-axis label
    - is_capacity: bool, whether to add extra processing for capacities
    """
    if is_capacity:
        existing_cap = get_currently_installed_capacity(n)
        data.loc[existing_cap.index, "Existing"] = existing_cap
        data = data[["Existing"] + [col for col in data.columns if col != "Existing"]]
        retirements = data.diff(axis=1).clip(upper=0)
        retirements = retirements[(retirements < -0.001).any(axis=1)]
        retirements.fillna(0, inplace=True)
        data = pd.concat([data, retirements])

    data = data / 1e3  # Convert to GW

    palette = n.carriers.set_index("nice_name").color.to_dict()
    regions = data.index.get_level_values(0).unique()

    # Determine grid layout for subplots
    num_regions = len(regions)
    columns = min(5, num_regions)  # Limit to 5 columns
    rows = math.ceil(num_regions / columns)

    # Set up the figure and axes
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 2.5, rows * 5), sharex=True, sharey=True)
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

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    handles, labels = [], []
    for carrier in data.reset_index().Carrier.unique():
        handle = plt.Rectangle((0, 0), 1, 1, color=palette[carrier])
        handles.append(handle)
        labels.append(f"{carrier}")
    fig.legend(handles, labels, title="Carrier", loc="lower center", ncol=columns)

    plt.tight_layout(rect=[0, 0.3, 1, 1])
    plt.subplots_adjust(wspace=0.4)
    fig.suptitle(title)
    fig.savefig(save)
    plt.close()


def plot_regional_capacity_additions_bar(n, save):
    """
    Plot capacity evolution by NERC region in a stacked bar plot.
    """
    data = get_statistics(n, "Optimal Capacity")
    data.to_csv(f"{Path(save).parent.parent}/statistics/bar_regional_capacity.csv")
    plot_bar(data, n, save, "", "Capacity (GW)", is_capacity=True)


def plot_regional_production_bar(n, save):
    """
    Plot production evolution by NERC region in a stacked bar plot.
    """
    data = get_statistics(n, "Supply")
    data.to_csv(f"{Path(save).parent.parent}/statistics/bar_regional_production.csv")
    plot_bar(data, n, save, "", "Production (GWh)")


def plot_regional_emissions_bar(
    n: pypsa.Network,
    save: str,
) -> None:
    """
    PLOT OF CO2 EMISSIONS BY NERC REGION AND INVESTMENT PERIOD.
    """
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
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 2.5, rows * 5), sharex=True, sharey=True)
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
    """
    PLOT OF CO2 EMISSIONS BY INVESTMENT PERIOD.
    """
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


#### Temporal Plots ####


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

    for month in ["all"] + months.to_list():
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
            # axs[i].set_title(f"Production in {investment_period}")
            axs[i].set_ylabel("Power [GW]")
            axs[i].set_xlabel("")

        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.suptitle(create_title("Production [GW]", **wildcards))
        save = Path(save)
        fig.savefig(save.parent / (save.stem + suffix + save.suffix))
        plt.close()


def plot_hourly_emissions(n: pypsa.Network, save: str, **wildcards) -> None:
    """
    Plots snapshot emissions by technology.
    """
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
    """
    Creates area plot of accumulated emissions by technology.
    """
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
    """
    Plots accumulated emissions.
    """
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


def plot_capacity_factor_heatmap(n: pypsa.Network, save: str, **wildcards) -> None:
    """
    HEATMAP OF RENEWABLE CAPACITY FACTORS BY CARRIER.
    """
    df_long = n.generators_t.p.loc[n.investment_periods[0]].melt(var_name="bus", value_name="p", ignore_index=False)
    df_long["region"] = df_long["bus"].map(n.generators.bus.map(n.buses.country))
    df_long["carrier"] = df_long["bus"].map(n.generators.carrier)
    df_long["hour"] = df_long.index.hour
    df_long["month"] = df_long.index.month
    df_long.drop(columns="bus", inplace=True)
    df_long = df_long.drop(columns="region").groupby(["carrier", "month", "hour"]).mean().reset_index()

    # Get unique months for separate panels
    unique_months = df_long["month"].unique()

    # Prepare figure and axes
    fig, axs = plt.subplots(len(unique_months), 1, figsize=(12, len(unique_months) * 4), sharex=True)

    # Iterate over each month to create a panel
    for idx, month in enumerate(sorted(unique_months)):
        month_data = df_long[df_long["month"] == month]
        pivot_data = month_data.pivot(index="hour", columns="carrier", values="p")

        ax = axs[idx] if len(unique_months) > 1 else axs
        pivot_data.plot.area(ax=ax, title=f"Month: {month}", alpha=0.7)
        ax.set_ylabel("Mean Power (p)")
        ax.set_xlabel("Hour of the Day")
        ax.legend(title="Carrier", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.suptitle("Heatmap of Renewable Capacity Factors by by Carrier")
    plt.tight_layout()

    plt.savefig(save)
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


def plot_region_lmps(
    n: pypsa.Network,
    save: str,
    **wildcards,
) -> None:
    """
    Plots a box plot of LMPs for each region.
    """
    df_lmp = n.buses_t.marginal_price
    df_long = pd.melt(
        df_lmp.reset_index(),
        id_vars=["timestep"],
        var_name="bus",
        value_name="lmp",
    )
    df_long["season"] = df_long["timestep"].dt.quarter
    df_long["hour"] = df_long["timestep"].dt.hour
    df_long.drop(columns="timestep", inplace=True)
    df_long["region"] = df_long.bus.map(n.buses.country)

    plt.figure(figsize=(10, 10))

    sns.boxplot(
        df_long,
        x="lmp",
        y="region",
        width=0.5,
        fliersize=0.5,
        linewidth=1,
    )

    plt.title(create_title("LMPs by Region", **wildcards))
    plt.xlabel("LMP [$/MWh]")
    plt.ylabel("Region")
    plt.tight_layout()
    plt.savefig(save)
    plt.close()


#### Fuel costs


def plot_fuel_costs(
    n: pypsa.Network,
    save: str,
    **wildcards,
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
    axs[0].set_title("Daily Average Fuel Costs [$/MWh]"),
    axs[0].set_xlabel(""),
    axs[0].set_ylabel("$/MWh"),

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
        axs[i + 1].set_title(f"Daily Average {nice_name} Fuel Costs per Bus [$/MWh]"),
        axs[i + 1].set_xlabel(""),
        axs[i + 1].set_ylabel("$/MWh"),

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

    # mappers
    generating_link_carrier_map = {"fuel cell": "H2", "battery discharger": "battery"}

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
    n.statistics(groupby=groupers.get_name_bus_and_carrier).round(3).to_csv(snakemake.output.statistics_dissaggregated)
    n.statistics().round(2).to_csv(snakemake.output.statistics_summary)
    n.generators.to_csv(snakemake.output.generators)
    n.storage_units.to_csv(snakemake.output.storage_units)
    n.links.to_csv(snakemake.output.links)

    # Panel Plots
    plot_generator_data_panel(
        n,
        snakemake.output["generator_data_panel.pdf"],
        **snakemake.wildcards,
    )

    # Bar Plots
    plot_capacity_additions_bar(
        n,
        carriers,
        snakemake.output["capacity_additions_bar.pdf"],
        **snakemake.wildcards,
    )
    plot_production_bar(
        n,
        carriers,
        snakemake.output["production_bar.pdf"],
        **snakemake.wildcards,
    )
    plot_global_constraint_shadow_prices(
        n,
        snakemake.output["global_constraint_shadow_prices.pdf"],
        **snakemake.wildcards,
    )
    plot_regional_capacity_additions_bar(
        n,
        snakemake.output["bar_regional_capacity_additions.pdf"],
    )
    plot_regional_production_bar(
        n,
        snakemake.output["bar_regional_production.pdf"],
    )
    plot_regional_emissions_bar(
        n,
        snakemake.output["bar_regional_emissions.pdf"],
    )
    plot_emissions_bar(
        n,
        snakemake.output["bar_emissions.pdf"],
    )

    # Time Series Plots
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
        **snakemake.wildcards,
    )

    # Box Plot
    plot_region_lmps(
        n,
        snakemake.output["region_lmps.pdf"],
        **snakemake.wildcards,
    )
