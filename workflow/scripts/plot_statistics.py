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
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
import seaborn as sns

logger = logging.getLogger(__name__)
import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from _helpers import configure_logging
from add_electricity import add_nice_carrier_names, sanitize_carriers
from matplotlib.lines import Line2D
from summary import (
    get_capital_costs,
    get_demand_timeseries,
    get_energy_timeseries,
    get_fuel_costs,
    get_generator_marginal_costs,
    get_node_emissions_timeseries,
    get_tech_emissions_timeseries,
)

# Global Plotting Settings
TITLE_SIZE = 16


def get_color_palette(n: pypsa.Network) -> dict[str, str]:
    """
    Returns colors based on nice name.
    """

    colors = (n.carriers.reset_index().set_index("nice_name")).color

    additional = {
        "Battery Charge": n.carriers.loc["battery"].color,
        "Battery Discharge": n.carriers.loc["battery"].color,
        "battery_discharger": n.carriers.loc["battery"].color,
        "battery_charger": n.carriers.loc["battery"].color,
        "4hr_battery_storage_discharger": n.carriers.loc["4hr_battery_storage"].color,
        "4hr_battery_storage_charger": n.carriers.loc["4hr_battery_storage"].color,
        "co2": "k",
    }

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


#### Plot HTML ####
def plot_accumulated_emissions_tech_html(
    n: pypsa.Network,
    save: str,
    **wildcards,
) -> None:
    """
    Plots accumulated emissions by technology.
    """

    # get data

    emissions = get_tech_emissions_timeseries(n).cumsum().mul(1e-6)  # T -> MT
    zeros = emissions.columns[(np.abs(emissions) < 1e-7).all()]
    emissions = emissions.drop(columns=zeros)

    # plot

    color_palette = get_color_palette(n)

    fig = px.area(
        emissions,
        x=emissions.index,
        y=emissions.columns,
        color_discrete_map=color_palette,
    )

    title = create_title("Technology Accumulated Emissions", **wildcards)
    fig.update_layout(
        title=dict(text=title, font=dict(size=TITLE_SIZE)),
        xaxis_title="",
        yaxis_title="Emissions [MT]",
    )
    fig.write_html(save)


def plot_hourly_emissions_html(n: pypsa.Network, save: str, **wildcards) -> None:
    """
    Plots interactive snapshot emissions by technology.
    """

    # get data

    emissions = get_tech_emissions_timeseries(n).mul(1e-6)  # T -> MT
    zeros = emissions.columns[(np.abs(emissions) < 1e-7).all()]
    emissions = emissions.drop(columns=zeros)

    # plot

    color_palette = get_color_palette(n)

    fig = px.area(
        emissions,
        x=emissions.index,
        y=emissions.columns,
        color_discrete_map=color_palette,
    )

    title = create_title("Technology Emissions", **wildcards)
    fig.update_layout(
        title=dict(text=title, font=dict(size=TITLE_SIZE)),
        xaxis_title="",
        yaxis_title="Emissions [MT]",
    )
    fig.write_html(save)


def plot_production_html(
    n: pypsa.Network,
    carriers_2_plot: list[str],
    save: str,
    **wildcards,
) -> None:
    """
    Plots interactive timeseries production chart.
    """
    # get data

    energy_mix = get_energy_timeseries(n).mul(1e-3)  # MW -> GW

    # fix battery charge/discharge to only be positive
    if "battery" in energy_mix:
        col_rename = {
            "battery charger": "battery",
            "battery discharger": "battery",
        }
        energy_mix = energy_mix.rename(columns=col_rename)
        energy_mix = energy_mix.groupby(level=0, axis=1).sum()
        energy_mix["battery"] = energy_mix.battery.map(lambda x: max(0, x))

    carriers_2_plot.append("battery")
    energy_mix = energy_mix[[x for x in carriers_2_plot if x in energy_mix]]
    energy_mix = energy_mix.rename(columns=n.carriers.nice_name)
    energy_mix["Demand"] = get_demand_timeseries(n).mul(1e-3)  # MW -> GW

    color_palette = get_color_palette(n)

    fig = px.area(
        energy_mix,
        x=energy_mix.index,
        y=[c for c in energy_mix.columns if c != "Demand"],
        color_discrete_map=color_palette,
    )
    fig.add_trace(
        go.Scatter(
            x=energy_mix.index,
            y=energy_mix.Demand,
            mode="lines",
            name="Demand",
            line_color="darkblue",
        ),
    )
    title = create_title("Production [GW]", **wildcards)
    fig.update_layout(
        title=dict(text=title, font=dict(size=TITLE_SIZE)),
        xaxis_title="",
        yaxis_title="Power [GW]",
    )
    fig.write_html(save)


def plot_region_emissions_html(n: pypsa.Network, save: str, **wildcards) -> None:
    """
    Plots interactive region level emissions.
    """

    # get data
    emissions = get_node_emissions_timeseries(n).mul(1e-6)  # T -> MT
    emissions = emissions.T.groupby(n.buses.country).sum().T

    # plot data
    fig = px.area(
        emissions,
        x=emissions.index,
        y=emissions.columns,
    )

    title = create_title("Regional CO2 Emissions", **wildcards)
    fig.update_layout(
        title=dict(text=title, font=dict(size=TITLE_SIZE)),
        xaxis_title="",
        yaxis_title="Emissions [MT]",
    )
    fig.write_html(save)


def plot_node_emissions_html(n: pypsa.Network, save: str, **wildcards) -> None:
    """
    Plots interactive node level emissions.

    Performance issues of this with many nodes!!
    """

    # get data

    emissions = get_node_emissions_timeseries(n).mul(1e-6)  # T -> MT

    # plot

    fig = px.area(
        emissions,
        x=emissions.index,
        y=emissions.columns,
    )

    title = create_title("Node Emissions", **wildcards)
    fig.update_layout(
        title=dict(text=title, font=dict(size=TITLE_SIZE)),
        xaxis_title="",
        yaxis_title="Emissions [MT]",
    )
    fig.write_html(save)


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
    capacity = n.statistics()[["Optimal Capacity", "Installed Capacity"]]
    capacity = capacity[
        capacity.index.get_level_values(0).isin(["Generator", "StorageUnit"])
    ]
    capacity.index = capacity.index.droplevel(0)
    capacity.reset_index(inplace=True)
    capacity.rename(columns={"index": "carrier"}, inplace=True)
    capacity_melt = capacity.melt(
        id_vars="carrier",
        var_name="Capacity Type",
        value_name="Capacity",
    )

    color_palette = get_color_palette(n)
    color_mapper = [color_palette[carrier] for carrier in capacity.carrier]
    bar_height = 0.35

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.barh(
        capacity.carrier,
        capacity["Installed Capacity"],
        height=bar_height,
        align="center",
        color=color_mapper,
    )
    ax.barh(
        [i + bar_height for i in range(len(capacity))],
        capacity["Optimal Capacity"],
        height=bar_height,
        align="center",
        alpha=0.50,
        color=color_mapper,
    )
    ax.invert_yaxis()
    ax.set_yticks([i + bar_height / 2 for i in range(len(capacity))])

    legend_lines = [
        Line2D([0], [0], color="k", alpha=1, lw=7),
        Line2D([0], [0], color="k", alpha=0.5, lw=7),
    ]
    ax.legend(
        legend_lines,
        ["Installed Capacity", "Optimal Capacity"],
        loc="lower right",
        borderpad=0.75,
    )

    ax.set_title(create_title("System Capacity Additions", **wildcards))
    ax.set_ylabel("")
    ax.set_xlabel("Capacity [MW]")

    fig.tight_layout()
    fig.savefig(save)


def plot_production_bar(
    n: pypsa.Network,
    carriers_2_plot: list[str],
    save: str,
    **wildcards,
) -> None:
    """
    Plot diaptch per carrier.
    """

    # get data
    energy_mix = n.statistics.dispatch().mul(1e-3)  # MW -> GW
    energy_mix.name = "dispatch"
    energy_mix = energy_mix[
        energy_mix.index.get_level_values("component").isin(
            ["Generator", "StorageUnit"],
        )
    ]
    energy_mix = energy_mix.groupby("carrier").sum().reset_index()

    color_palette = get_color_palette(n)

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.barplot(
        data=energy_mix,
        y="carrier",
        x="dispatch",
        palette=color_palette,
    )

    ax.set_title(create_title("Dispatch [GWh]", **wildcards))
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(save)


def plot_costs_bar(
    n: pypsa.Network,
    carriers_2_plot: list[str],
    save: str,
    **wildcards,
) -> None:
    """
    Plot OPEX and CAPEX.
    """

    # get data

    opex = n.statistics.opex().mul(1e-6)  # $ -> M$
    capex = get_capital_costs(n).mul(1e-6)  # $ -> M$

    costs = pd.concat(
        [opex, capex],
        axis=1,
        keys=["OPEX", "CAPEX"],
    ).reset_index()
    costs = costs.groupby("carrier").sum().reset_index()  # groups batteries

    # plot data
    fig, ax = plt.subplots(figsize=(10, 10))
    color_palette = get_color_palette(n)

    sns.barplot(
        y="carrier",
        x="CAPEX",
        data=costs,
        alpha=0.6,
        ax=ax,
        palette=color_palette,
    )
    sns.barplot(
        y="carrier",
        x="OPEX",
        data=costs,
        ax=ax,
        left=costs["CAPEX"],
        palette=color_palette,
    )

    legend_lines = [
        Line2D([0], [0], color="k", alpha=1, lw=7),
        Line2D([0], [0], color="k", alpha=0.6, lw=7),
    ]
    ax.legend(legend_lines, ["OPEX", "CAPEX"], loc="lower right", borderpad=0.75)

    ax.set_title(create_title("Costs", **wildcards))
    ax.set_ylabel("")
    ax.set_xlabel("CAPEX & OPEX [M$]")
    fig.tight_layout()
    fig.savefig(save)


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


def plot_regional_capacity_additions_bar(
    n: pypsa.Network,
    save: str,
    **wildcards,
) -> None:
    """
    PLOT OF CAPACITY ADDITIONS BY STATE AND CARRIER (STACKED BAR PLOT)
    """
    exp_gens = n.generators.p_nom_opt - n.generators.p_nom
    exp_storage = n.storage_units.p_nom_opt - n.storage_units.p_nom

    expanded_capacity = pd.concat([exp_gens, exp_storage])
    expanded_capacity = expanded_capacity.to_frame(name="mw")
    mapper = pd.concat(
        [
            n.generators.bus.map(n.buses.country),
            n.storage_units.bus.map(n.buses.country),
        ],
    )
    expanded_capacity["region"] = expanded_capacity.index.map(mapper)
    carrier_mapper = pd.concat([n.generators.carrier, n.storage_units.carrier])
    expanded_capacity["carrier"] = expanded_capacity.index.map(carrier_mapper)

    palette = n.carriers.color.to_dict()

    expanded_capacity["positive"] = expanded_capacity["mw"] > 0
    df_sorted = expanded_capacity.sort_values(by=["region", "carrier"])

    # Correcting the bottoms for positive and negative values
    bottoms_pos = (
        df_sorted[df_sorted["positive"]].groupby("region")["mw"].cumsum()
        - df_sorted["mw"]
    )
    bottoms_neg = (
        df_sorted[~df_sorted["positive"]].groupby("region")["mw"].cumsum()
        - df_sorted["mw"]
    )

    # Re-initialize plot to address the legend and gap issues
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each carrier, adjusting handling for legend and correcting negative stacking
    for i, carrier in enumerate(df_sorted["carrier"].unique()):
        # Filter by carrier
        df_carrier = df_sorted[df_sorted["carrier"] == carrier]

        # Separate positive and negative
        df_pos = df_carrier[df_carrier["positive"]]
        df_neg = df_carrier[~df_carrier["positive"]]

        # Plot positives
        ax.barh(
            df_pos["region"],
            df_pos["mw"],
            left=bottoms_pos[df_pos.index],
            color=palette[carrier],
            edgecolor="w",
        )

        # Plot negatives
        ax.barh(
            df_neg["region"],
            df_neg["mw"],
            left=bottoms_neg[df_neg.index],
            color=palette[carrier],
            edgecolor="w",
        )

    # Adjust legend to include all carriers
    handles, labels = [], []
    for i, carrier in enumerate(df_sorted["carrier"].unique()):
        handle = plt.Rectangle((0, 0), 1, 1, color=palette[carrier], edgecolor="w")
        handles.append(handle)
        labels.append(f"{carrier}")

    ax.legend(handles, labels, title="Carrier")

    ax.set_title("Adjusted MW by Region and Carrier with Negative Values")
    ax.set_xlabel("MW")
    ax.set_ylabel("Region")

    fig.tight_layout()
    fig.savefig(save)


def plot_regional_emissions_bar(
    n: pypsa.Network,
    save: str,
    **wildcards,
) -> None:
    """
    PLOT OF CO2 EMISSIONS BY REGION.
    """
    regional_emisssions = (
        get_node_emissions_timeseries(n).T.groupby(n.buses.country).sum().T.sum() / 1e6
    )

    plt.figure(figsize=(10, 10))
    sns.barplot(
        x=regional_emisssions.values,
        y=regional_emisssions.index,
        palette="viridis",
    )

    plt.xlabel("CO2 Emissions [MMtCO2]")
    plt.ylabel("")
    plt.title(create_title("CO2 Emissions by Region", **wildcards))

    plt.tight_layout()
    plt.savefig(save)


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
        if "battery" in carrier:
            energy_mix[carrier + "_discharger"] = energy_mix[carrier].clip(lower=0.0001)
            energy_mix[carrier + "_charger"] = energy_mix[carrier].clip(upper=-0.0001)
            energy_mix = energy_mix.drop(columns=carrier)
            # carriers_2_plot.append("battery_charger")
            # carriers_2_plot.append("battery_discharger")
    energy_mix = energy_mix[[x for x in carriers_2_plot if x in energy_mix]]
    energy_mix = energy_mix.rename(columns=n.carriers.nice_name)

    energy_mix = energy_mix.rename(columns=n.carriers.nice_name)

    color_palette = get_color_palette(n)

    year = n.snapshots[0].year
    for timeslice in ["all"] + list(range(1, 12)):
        try:
            if not timeslice == "all":
                snapshots = n.snapshots.get_loc(f"{year}-{timeslice}")
            else:
                snapshots = slice(None, None)

            fig, ax = plt.subplots(figsize=(14, 4))

            energy_mix[snapshots].plot.area(
                ax=ax,
                alpha=0.7,
                color=color_palette,
            )
            demand[snapshots].plot.line(ax=ax, ls="-", color="darkblue")

            suffix = (
                "-" + datetime.strptime(str(timeslice), "%m").strftime("%b")
                if timeslice != "all"
                else ""
            )

            ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
            ax.set_title(create_title("Production [GW]", **wildcards))
            ax.set_ylabel("Power [GW]")
            fig.tight_layout()
            save = Path(save)
            fig.savefig(save.parent / (save.stem + suffix + save.suffix))
        except KeyError:
            # outside slicing range
            continue


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


def plot_curtailment_heatmap(n: pypsa.Network, save: str, **wildcards) -> None:
    curtailment = n.statistics.curtailment(aggregate_time=False)
    curtailment = curtailment[
        curtailment.index.get_level_values(0).isin(["StorageUnit", "Generator"])
    ].droplevel(0)
    curtailment = curtailment[curtailment.sum(1) > 0.001].T
    curtailment.index = (
        pd.to_datetime(curtailment.index)
        .tz_localize("utc")
        .tz_convert("America/Los_Angeles")
    )
    curtailment["month"] = curtailment.index.month
    curtailment["hour"] = curtailment.index.hour
    curtailment_group = curtailment.groupby(["month", "hour"]).mean()

    df_long = pd.melt(
        curtailment_group.reset_index(),
        id_vars=["month", "hour"],
        var_name="carrier",
        value_name="MW",
    )
    df_long

    carriers = df_long["carrier"].unique()
    num_carriers = len(carriers)

    rows = num_carriers // 3 + (num_carriers % 3 > 0)
    cols = min(num_carriers, 3)

    # Plotting with dynamic subplot creation based on the number of groups, with wrapping
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for i, carrier in enumerate(carriers):
        pivot_table = (
            df_long[df_long.carrier == carrier]
            .pivot(index="month", columns="hour", values="MW")
            .fillna(0)
        )
        sns.heatmap(pivot_table, ax=axes[i], cmap="viridis")
        axes[i].set_title(carrier)

    # Hide any unused axes if the number of groups is not a multiple of 3
    for j in range(i + 1, rows * cols):
        axes[j].set_visible(False)

    plt.suptitle(create_title("Heatmap of Curtailment by by Carrier", **wildcards))

    plt.tight_layout()
    plt.savefig(save)


def plot_capacity_factor_heatmap(n: pypsa.Network, save: str, **wildcards) -> None:
    """
    HEATMAP OF RENEWABLE CAPACITY FACTORS BY CARRIER.
    """
    df_long = n.generators_t.p_max_pu.melt(
        var_name="bus",
        value_name="p_max_pu",
        ignore_index=False,
    )
    df_long["region"] = df_long["bus"].map(n.generators.bus.map(n.buses.country))
    df_long["carrier"] = df_long["bus"].map(n.generators.carrier)
    df_long["hour"] = df_long.index.hour
    df_long["month"] = df_long.index.month
    df_long.drop(columns="bus", inplace=True)
    df_long = (
        df_long.drop(columns="region")
        .groupby(["carrier", "month", "hour"])
        .mean()
        .reset_index()
    )

    unique_groups = df_long["carrier"].unique()
    num_groups = len(unique_groups)

    rows = num_groups // 4 + (num_groups % 4 > 0)
    cols = min(num_groups, 4)

    # Plotting with dynamic subplot creation based on the number of groups, with wrapping
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for i, carrier in enumerate(unique_groups):
        pivot_table = (
            df_long[df_long.carrier == carrier]
            .pivot(index="month", columns="hour", values="p_max_pu")
            .fillna(0)
        )
        sns.heatmap(pivot_table, ax=axes[i], cmap="viridis")
        axes[i].set_title(carrier)

    # Hide any unused axes if the number of groups is not a multiple of 3
    for j in range(i + 1, rows * cols):
        axes[j].set_visible(False)

    plt.suptitle("Heatmap of Renewable Capacity Factors by by Carrier")

    plt.tight_layout()
    plt.savefig(save)


#### Panel / Mixed Plots ####


def plot_generator_data_panel(
    n: pypsa.Network,
    save: str,
    **wildcards,
):

    df_capex_expand = n.generators.loc[
        n.generators.index.str.contains("new")
        | n.generators.carrier.isin(
            [
                "nuclear",
                "solar",
                "onwind",
                "offwind",
                "offwind_floating",
                "geothermal",
                "oil",
                "hydro",
            ],
        ),
        :,
    ]
    df_capex_retire = n.generators.loc[
        ~n.generators.index.str.contains("new")
        & ~n.generators.carrier.isin(
            [
                "solar",
                "onwind",
                "offwind",
                "offwind_floating",
                "geothermal",
                "oil",
                "hydro",
                "nuclear",
            ],
        ),
        :,
    ]

    df_storage_units = n.storage_units
    df_storage_units["efficiency"] = df_storage_units.efficiency_dispatch
    df_capex_expand = pd.concat([df_capex_expand, df_storage_units])

    # Create a figure and subplots with 2 rows and 2 columns
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))

    # Plot on each subplot
    sns.lineplot(
        data=get_generator_marginal_costs(n),
        x="snapshot",
        y="Value",
        hue="Carrier",
        ax=axes[0, 0],
    )
    sns.barplot(data=df_capex_expand, x="carrier", y="capital_cost", ax=axes[0, 1])
    sns.boxplot(data=df_capex_expand, x="carrier", y="efficiency", ax=axes[1, 0])
    sns.barplot(data=df_capex_retire, x="carrier", y="capital_cost", ax=axes[1, 1])
    sns.histplot(
        data=n.generators,
        x="ramp_limit_up",
        hue="carrier",
        ax=axes[2, 0],
        bins=50,
        stat="density",
    )
    sns.barplot(
        data=n.generators.groupby("carrier").sum().reset_index(),
        y="p_nom",
        x="carrier",
        ax=axes[2, 1],
    )

    # Set titles for each subplot
    axes[0, 0].set_title("Generator Marginal Costs")
    axes[0, 1].set_title("Extendable Capital Costs")
    axes[1, 0].set_title("Energy Efficiency")
    axes[1, 1].set_title("Fixed O&M Costs of Retiring Units")
    axes[2, 0].set_title("Generator Ramp Up Limits")
    axes[2, 1].set_title("Existing Capacity by Carrier")

    # Set labels for each subplot
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("$ / MWh")
    axes[0, 0].set_ylim(0, 300)
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("$ / MW-yr")
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("MWh_primary / MWh_elec")
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("$ / MW-yr")
    axes[2, 0].set_xlabel("pu/hr")
    axes[2, 0].set_ylabel("count")
    axes[2, 1].set_xlabel("")
    axes[2, 1].set_ylabel("MW")

    # Rotate x-axis labels for each subplot
    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=35)

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
        id_vars=["snapshot"],
        var_name="bus",
        value_name="lmp",
    )
    df_long["season"] = df_long["snapshot"].dt.quarter
    df_long["hour"] = df_long["snapshot"].dt.hour
    df_long.drop(columns="snapshot", inplace=True)
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
    df = (
        fuel_costs.droplevel(["bus", "Generator"])
        .T.resample("d")
        .mean()
        .reset_index()
        .melt(id_vars="snapshot")
    )
    sns.lineplot(
        data=df,
        x="snapshot",
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
        df = (
            fuel_costs.loc[fuel, :, :]
            .droplevel("Generator")
            .T.resample("d")
            .mean()
            .T.groupby(level=0)
            .mean()
            .T
        )
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


# Pie Chart
def plot_california_emissions(
    n: pypsa.Network,
    save: str,
    **wildcards,
) -> None:
    """
    Plots a pie chart of emissions by carrier in California.
    """
    generator_emissions = n.generators_t.p * n.generators.carrier.map(
        n.carriers.co2_emissions,
    )
    ca_list = ["California", "CISO", "CISO_PGE", "CISO_SCE", "CISO_SDGE", "CISO_VEA"]
    ca_generator_emissions = generator_emissions.loc[
        :,
        n.generators.bus.map(n.buses.country).isin(ca_list),
    ]
    ca_generator_emissions = (
        ca_generator_emissions.groupby(n.generators.carrier, axis=1).sum().sum() / 1e6
    )
    ca_generator_emissions

    lines_bus0 = n.lines.bus0
    lines_bus1 = n.lines.bus1
    lines_bus0_region = lines_bus0[lines_bus0.map(n.buses.country).isin(ca_list)]

    region_lines = n.lines.loc[lines_bus0_region.index]
    inter_regional_lines = region_lines[
        ~region_lines.bus1.map(n.buses.country).isin(ca_list)
    ]

    ca_imports = (
        n.lines_t.p1.loc[:, inter_regional_lines.index].clip(lower=0) * 0.428 / 1e6
    )  # 0.428 is the average emissions factor for imports defined by CPUC
    ca_imports = pd.Series(ca_imports.sum().sum(), index=["imported_emissions"])
    ca_emissions = pd.concat([ca_generator_emissions, ca_imports])
    ca_emissions = ca_emissions.loc[ca_emissions > 0.0001]

    plt.figure(figsize=(8, 8))
    sns.barplot(x=ca_emissions.values, y=ca_emissions.index)

    plt.xlabel("CO2 Emissions [MMtCO2]")
    plt.title(create_title("California Total Emissions by Source", **wildcards))
    plt.tight_layout()
    plt.savefig(save)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_statistics",
            interconnect="western",
            clusters=80,
            ll="v1.0",
            opts="Ep-Co2L0.2",
            sector="E",
        )
    configure_logging(snakemake)

    # extract shared plotting files
    n = pypsa.Network(snakemake.input.network)
    onshore_regions = gpd.read_file(snakemake.input.regions_onshore)
    retirement_method = snakemake.params.retirement
    # n_hours = snakemake.config['solving']['options']['nhours']

    sanitize_carriers(n, snakemake.config)

    # mappers
    generating_link_carrier_map = {"fuel cell": "H2", "battery discharger": "battery"}

    # carriers to plot
    carriers = (
        snakemake.params.electricity["conventional_carriers"]
        + snakemake.params.electricity["renewable_carriers"]
        + snakemake.params.electricity["extendable_carriers"]["StorageUnit"]
        + snakemake.params.electricity["extendable_carriers"]["Store"]
        + snakemake.params.electricity["extendable_carriers"]["Link"]
        + ["battery_charger", "battery_discharger"]
    )
    carriers = list(set(carriers))  # remove any duplicates

    # plotting theme
    sns.set_theme("paper", style="darkgrid")

    # Bar Plots
    plot_capacity_additions_bar(
        n,
        carriers,
        snakemake.output["capacity_additions_bar.pdf"],
        **snakemake.wildcards,
    )
    plot_costs_bar(
        n,
        carriers,
        snakemake.output["costs_bar.pdf"],
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
        **snakemake.wildcards,
    )
    plot_regional_emissions_bar(
        n,
        snakemake.output["bar_regional_emissions.pdf"],
        **snakemake.wildcards,
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
    plot_curtailment_heatmap(
        n,
        snakemake.output["curtailment_heatmap.pdf"],
        **snakemake.wildcards,
    )
    plot_capacity_factor_heatmap(
        n,
        snakemake.output["capfac_heatmap.pdf"],
        **snakemake.wildcards,
    )
    plot_fuel_costs(
        n,
        snakemake.output["fuel_costs.pdf"],
        **snakemake.wildcards,
    )

    # HTML Plots
    plot_production_html(
        n,
        carriers,
        snakemake.output["production_area.html"],
        **snakemake.wildcards,
    )
    plot_hourly_emissions_html(
        n,
        snakemake.output["emissions_area.html"],
        **snakemake.wildcards,
    )
    plot_accumulated_emissions_tech_html(
        n,
        snakemake.output["emissions_accumulated_tech.html"],
        **snakemake.wildcards,
    )
    plot_region_emissions_html(
        n,
        snakemake.output["emissions_region.html"],
        **snakemake.wildcards,
    )

    # Panel Plots
    plot_generator_data_panel(
        n,
        snakemake.output["generator_data_panel.pdf"],
        **snakemake.wildcards,
    )

    # Box Plot
    plot_region_lmps(
        n,
        snakemake.output["region_lmps.pdf"],
        **snakemake.wildcards,
    )

    # if snakemake.wildcards["interconnect"] == "western":
    #     # California Emissions
    #     plot_california_emissions(
    #         n,
    #         Path(snakemake.output["region_lmps.pdf"]).parents[0]
    #         / "california_emissions.png",
    #         **snakemake.wildcards,
    #     )
