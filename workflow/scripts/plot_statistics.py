"""
Plots static and interactive charts to analyze system results.

**Inputs**

A solved network

**Outputs**

Capacity maps for:
    - Base capacity
    - New capacity
    - Optimal capacity (does not show existing unused capacity)
    - Optimal browfield capacity
    - Renewable potential capacity

    .. image:: _static/plots/capacity-map.png
        :scale: 33 %

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
    - Emissions map by node
    - Accumulated emissions

    .. image:: _static/plots/emissions-area.png
        :scale: 33 %

    .. image:: _static/plots/emissions-map.png
        :scale: 33 %
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict
from typing import List
from typing import Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
import seaborn as sns
from cartopy import crs as ccrs
from pypsa.plot import add_legend_circles
from pypsa.plot import add_legend_lines
from pypsa.plot import add_legend_patches

logger = logging.getLogger(__name__)
from _helpers import configure_logging
from summary import (
    get_demand_timeseries,
    get_energy_timeseries,
    get_node_emissions_timeseries,
    get_tech_emissions_timeseries,
    get_capacity_greenfield,
    get_capacity_brownfield,
    get_capacity_base,
    get_demand_base,
    get_capital_costs,
    get_generator_marginal_costs,
)
from add_electricity import (
    add_nice_carrier_names,
    sanitize_carriers,
)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs

import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs

import plotly.express as px
import plotly.graph_objects as go


# Global Plotting Settings
TITLE_SIZE = 16

def get_color_palette(n: pypsa.Network) -> pd.Series:
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
    emissions = emissions[
        [
            x
            for x in n.carriers[n.carriers.co2_emissions > 0].index
            if x in emissions.columns
        ]
    ]
    emissions = emissions.rename(columns=n.carriers.nice_name)

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
    emissions = emissions[
        [
            x
            for x in n.carriers[n.carriers.co2_emissions > 0].index
            if x in emissions.columns
        ]
    ]
    emissions = emissions.rename(columns=n.carriers.nice_name)

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

    # # fix battery charge/discharge to only be positive
    # if "battery" in energy_mix:
    #     col_rename = {
    #         "battery charger": "battery",
    #         "battery discharger": "battery",
    #     }
    #     energy_mix = energy_mix.rename(columns=col_rename)
    #     energy_mix = energy_mix.groupby(level=0, axis=1).sum()
    #     energy_mix["battery"] = energy_mix.battery.map(lambda x: max(0, x))

    # energy_mix = energy_mix[[x for x in carriers_2_plot if x in energy_mix]]
    # energy_mix = energy_mix.rename(columns=n.carriers.nice_name)
    # ########

    energy_mix["Demand"] = get_demand_timeseries(n).mul(1e-3)  # MW -> GW

    # plot

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
    opt_capacity: str = "greenfield",
    retirement_method: str = "economic",
    **wildcards,
) -> None:
    """
    Plots base capacity vs optimal capacity as a bar chart.
    """

    # get data

    nice_names = n.carriers.nice_name

    p_nom = (
        get_capacity_base(n)
        .to_frame("Base Capacity")
        .reset_index()
        .drop(columns=["bus"])
        .groupby("carrier")
        .sum()
    )

    if opt_capacity == "greenfield":
        p_nom_opt = get_capacity_greenfield(n, retirement_method)
    elif opt_capacity == "brownfield":
        p_nom_opt = get_capacity_brownfield(n, retirement_method)
    else:
        logger.error(
            f"Capacity method must be one of 'greenfield' or 'brownfield'. Recieved {opt_capacity}.",
        )
        raise NotImplementedError

    # depending on retirment method, column name may be p_nom_opt or p_max
    p_nom_opt = (
        p_nom_opt.reset_index()
        .drop(columns=["bus"])
        .groupby("carrier")
        .sum()
        .rename(columns={"p_nom_opt": "Optimal Capacity"})
    )

    capacity = p_nom.join(p_nom_opt)
    capacity = capacity[capacity.index.isin(carriers_2_plot)]
    capacity.index = capacity.index.map(nice_names)

    # plot data (option 1)
    # using seaborn with hues, but does not do tech color groups
    """
    capacity_melt = capacity.melt(id_vars="carrier", var_name="Capacity Type", value_name="Capacity")

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.barplot(data=capacity_melt, y="carrier", x="Capacity", hue="Capacity Type", ax=ax)

    ax.set_title(create_title("System Capacity Additions", **wildcards))
    ax.set_ylabel("")
    ax.set_xlabel("Capacity [MW]")
    fig.tight_layout()
    fig.savefig(save)
    """

    # plot data (option 2)
    # using matplotlib for tech group colours

    color_palette = get_color_palette(n)
    color_mapper = [color_palette[carrier] for carrier in capacity.index]
    bar_height = 0.35

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.barh(
        capacity.index,
        capacity["Base Capacity"],
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
        ["Base Capacity", "Optimal Capacity"],
        loc="lower right",
        borderpad=0.75,
    )

    ax.set_title(create_title("System Capacity Additions", **wildcards))
    ax.set_ylabel("")
    ax.set_xlabel("Capacity [MW]")
    fig.tight_layout()

    # save = Path(save)
    # fig.savefig(f"{save.parent}/{save.stem}_color{save.suffix}")
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
    # ax.set_xlabel("")
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


def plot_global_constraint_shadow_prices(n: pypsa.Network, save: str, **wildcards) -> None:
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
    emissions = emissions[
        [
            x
            for x in n.carriers[n.carriers.co2_emissions > 0].index
            if x in emissions.columns
        ]
    ]
    emissions = emissions.rename(columns=n.carriers.nice_name)

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
    emissions = emissions[
        [
            x
            for x in n.carriers[n.carriers.co2_emissions > 0].index
            if x in emissions.columns
        ]
    ]
    emissions = emissions.rename(columns=n.carriers.nice_name)

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

def plot_curtailment_heatmap(n: pypsa.Network, save: str, **wildcards) -> None:
    curtailment = n.statistics.curtailment(aggregate_time=False)
    curtailment = curtailment[curtailment.index.get_level_values(0).isin(['StorageUnit', 'Generator'])].droplevel(0)
    curtailment = curtailment[curtailment.sum(1)> 0.001].T
    curtailment.index = pd.to_datetime(curtailment.index).tz_localize('utc').tz_convert('America/Los_Angeles')
    curtailment['month'] = curtailment.index.month
    curtailment['hour'] = curtailment.index.hour
    curtailment_group = curtailment.groupby(['month', 'hour']).mean()

    df_long = pd.melt(curtailment_group.reset_index(), id_vars=['month','hour'], var_name='carrier', value_name='MW')
    df_long

    carriers = df_long['carrier'].unique()
    num_carriers = len(carriers)

    rows = num_carriers // 3 + (num_carriers % 3 > 0)
    cols = min(num_carriers, 3)

    # Plotting with dynamic subplot creation based on the number of groups, with wrapping
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for i, carrier in enumerate(carriers):
        pivot_table = df_long[df_long.carrier==carrier].pivot(index="month", columns="hour", values="MW").fillna(0)
        sns.heatmap(pivot_table, ax=axes[i], cmap="viridis")
        axes[i].set_title(carrier)

    # Hide any unused axes if the number of groups is not a multiple of 3
    for j in range(i + 1, rows * cols):
        axes[j].set_visible(False)

    plt.suptitle(create_title("Heatmap of Curtailment by by Carrier", **wildcards))

    plt.tight_layout()
    plt.savefig(save)

#### Panel / Mixed Plots ####

def plot_generator_data_panel(
    n: pypsa.Network,
    save: str,
    **wildcards,
):

    df_capex_expand = n.generators.loc[n.generators.index.str.contains("new") | n.generators.carrier.isin(['nuclear','solar','onwind','offwind','offwind_floating','geothermal','oil','hydro']),:]
    df_capex_retire = n.generators.loc[~n.generators.index.str.contains("new") & ~n.generators.carrier.isin(['solar','onwind','offwind','offwind_floating','geothermal','oil','hydro', 'nuclear']),:]

    df_storage_units = n.storage_units
    df_storage_units['efficiency'] = df_storage_units.efficiency_dispatch
    df_capex_expand = pd.concat([df_capex_expand, df_storage_units])

    # Create a figure and subplots with 2 rows and 2 columns
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))

    # Plot on each subplot
    sns.lineplot(data=get_generator_marginal_costs(n), x='snapshot', y='Value', hue='Carrier', errorbar='sd', ax=axes[0, 0]) 
    sns.barplot(data=df_capex_expand, x='carrier', y='capital_cost', ax=axes[0, 1])
    sns.boxplot(data=df_capex_expand, x='carrier', y='efficiency', ax=axes[1, 0])
    sns.barplot(data=df_capex_retire, x='carrier', y='capital_cost', ax=axes[1, 1])
    sns.histplot(data=n.generators, x='ramp_limit_up', hue='carrier', ax=axes[2, 0], bins=50, stat='density')
    sns.barplot(data=n.generators.groupby('carrier').sum().reset_index(), y='p_nom', x='carrier', ax=axes[2, 1])


    # Set titles for each subplot
    axes[0, 0].set_title('Generator Marginal Costs')
    axes[0, 1].set_title('Extendable Capital Costs')
    axes[1, 0].set_title('Energy Efficiency')
    axes[1, 1].set_title('Fixed O&M Costs of Retiring Units')
    axes[2, 0].set_title('Generator Ramp Up Limits')
    axes[2, 1].set_title('Existing Capacity by Carrier')

    # Set labels for each subplot
    axes[0, 0].set_xlabel('')
    axes[0, 0].set_ylabel('$ / MWh')
    axes[0, 1].set_xlabel('')
    axes[0, 1].set_ylabel('$ / MW-yr')
    axes[1, 0].set_xlabel('')
    axes[1, 0].set_ylabel('MWh_primary / MWh_elec')
    axes[1, 1].set_xlabel('')
    axes[1, 1].set_ylabel('$ / MW-yr')
    axes[2, 0].set_xlabel('pu/hr')
    axes[2, 0].set_ylabel('count')
    axes[2, 1].set_xlabel('')
    axes[2, 1].set_ylabel('MW')

    #Rotate x-axis labels for each subplot
    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=35)

    # Lay legend out horizontally
    axes[0, 0].legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1, fontsize='xx-small')
    axes[2, 0].legend( fontsize='xx-small')

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
    df_long = pd.melt(df_lmp.reset_index(), id_vars=['snapshot'], var_name='bus', value_name='lmp')
    df_long['season'] = df_long['snapshot'].dt.quarter
    df_long['hour'] = df_long['snapshot'].dt.hour
    df_long.drop(columns='snapshot', inplace=True)
    df_long["region"] = df_long.bus.map(n.buses.country)

    plt.figure(figsize=(10, 10))

    sns.boxplot(
        df_long, 
        x="lmp", 
        y="region",
        width=.5, 
        fliersize=0.5, 
        linewidth=1,
        
    )

    plt.title(create_title("LMPs by Region", **wildcards))
    plt.xlabel("LMP [$/MWh]")
    plt.ylabel("Region")
    plt.tight_layout()
    plt.savefig(save)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_figures",
            interconnect="western",
            clusters=30,
            ll="v1.15",
            opts="CO2L0.75-4H",
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
    )
    carriers = list(set(carriers))  # remove any duplicates

    # plotting theme
    sns.set_theme("paper", style="darkgrid")


    # Bar Plots
    plot_capacity_additions_bar(
        n,
        carriers,
        snakemake.output["capacity_additions_bar"],
        "greenfield",
        retirement_method,
        **snakemake.wildcards,
    )
    plot_costs_bar(
        n,
        carriers,
        snakemake.output["costs_bar"],
        **snakemake.wildcards,
    )
    plot_production_bar(
        n,
        carriers,
        snakemake.output["production_bar"],
        **snakemake.wildcards,
    )
    plot_global_constraint_shadow_prices(
        n,
        snakemake.output["global_constraint_shadow_prices"],
        **snakemake.wildcards,
    )


    # Time Series Plots
    plot_production_area(
        n,
        carriers,
        snakemake.output["production_area"],
        **snakemake.wildcards,
    )
    plot_hourly_emissions(
        n,
        snakemake.output["emissions_area"],
        **snakemake.wildcards,
    )
    plot_accumulated_emissions_tech(
        n,
        snakemake.output["emissions_accumulated_tech"],
        **snakemake.wildcards,
    )
    plot_curtailment_heatmap(
        n,
        snakemake.output["curtailment_heatmap"],
        **snakemake.wildcards,
    )


    # HTML Plots
    plot_production_html(
        n,
        carriers,
        snakemake.output["production_area_html"],
        **snakemake.wildcards,
    )
    plot_hourly_emissions_html(
        n,
        snakemake.output["emissions_area_html"],
        **snakemake.wildcards,
    )
    plot_accumulated_emissions_tech_html(
        n,
        snakemake.output["emissions_accumulated_tech_html"],
        **snakemake.wildcards,
    )
    plot_region_emissions_html(
        n,
        snakemake.output["emissions_region_html"],
        **snakemake.wildcards,
    )


    # Panel Plots
    plot_generator_data_panel(
        n,
        snakemake.output["generator_data_panel"],
        **snakemake.wildcards,
    )

    plot_region_lmps(
        n,
        snakemake.output["region_lmps"],
        **snakemake.wildcards,
    )