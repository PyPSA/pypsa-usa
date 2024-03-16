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
from typing import Tuple

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
    get_operational_costs,
    get_capital_costs,
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

    return pd.concat([colors, pd.Series(additional)])


def get_bus_scale(interconnect: str) -> float:
    """
    Scales lines based on interconnect size.
    """
    if interconnect != "usa":
        return 1e5
    else:
        return 4e4


def get_line_scale(interconnect: str) -> float:
    """
    Scales lines based on interconnect size.
    """
    if interconnect != "usa":
        return 2e3
    else:
        return 3e3


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


def remove_sector_buses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes buses for sector coupling.
    """

    num_levels = df.index.nlevels

    if num_levels > 1:
        condition = (df.index.get_level_values("bus").str.endswith(" gas")) | (
            df.index.get_level_values("bus").str.endswith(" gas storage")
        )
    else:
        condition = (
            (df.index.str.endswith(" gas"))
            | (df.index.str.endswith(" gas storage"))
            | (df.index.str.endswith(" gas import"))
            | (df.index.str.endswith(" gas export"))
        )
    return df.loc[~condition].copy()


def remove_sector_links(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes links for plotting capacity.
    """
    pass


def plot_emissions_map(
    n: pypsa.Network,
    regions: gpd.GeoDataFrame,
    save: str,
    **wildcards,
) -> None:

    # get data

    emissions = (
        get_node_emissions_timeseries(n)
        .groupby(level=0, axis=1)  # group columns
        .sum()
        .sum()  # collaps rows
        .mul(1e-6)  # T -> MT
    )
    emissions = remove_sector_buses(emissions.T).T
    emissions.index.name = "bus"

    # plot data

    fig, ax = plt.subplots(
        figsize=(10, 10),
        subplot_kw={"projection": ccrs.EqualEarth(n.buses.x.mean())},
    )

    bus_scale = 1

    with plt.rc_context({"patch.linewidth": 0.1}):
        n.plot(
            bus_sizes=emissions / bus_scale,
            bus_colors="k",
            bus_alpha=0.7,
            line_widths=0,
            link_widths=0,
            ax=ax,
            margin=0.2,
            color_geomap=None,
        )

    # onshore regions
    regions.plot(
        ax=ax,
        facecolor="whitesmoke",
        edgecolor="white",
        aspect="equal",
        transform=ccrs.PlateCarree(),
        linewidth=1.2,
    )
    ax.set_extent(regions.total_bounds[[0, 2, 1, 3]])

    # legend_kwargs = {"loc": "upper left", "frameon": False}
    # bus_sizes = [0.01, 0.1, 1]  # in Tonnes
    #
    # add_legend_circles(
    #     ax,
    #     [s / bus_scale for s in bus_sizes],
    #     [f"{s / 1000} Tonnes" for s in bus_sizes],
    #     legend_kw={"bbox_to_anchor": (1, 1), **legend_kwargs},
    # )

    title = create_title("Emissions (MTonne)", **wildcards)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=20)
    fig.tight_layout()

    fig.savefig(save)


def plot_region_emissions_html(n: pypsa.Network, save: str, **wildcards) -> None:
    """
    Plots interactive region level emissions.
    """

    # get data

    emissions = get_node_emissions_timeseries(n).mul(1e-6)  # T -> MT
    emissions = emissions.groupby(n.buses.country, axis=1).sum()

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
        color=color_palette.to_dict(),
    )

    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    ax.set_title(create_title("Accumulated Emissions", **wildcards))
    ax.set_ylabel("Emissions [MT]")
    fig.tight_layout()

    fig.savefig(save)


def plot_accumulated_emissions_tech(n: pypsa.Network, save: str, **wildcards) -> None:
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

    fig, ax = plt.subplots(figsize=(14, 4))

    emissions.plot.area(
        ax=ax,
        alpha=0.7,
        legend="reverse",
        color=color_palette.to_dict(),
    )

    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    ax.set_title(create_title("Technology Accumulated Emissions", **wildcards))
    ax.set_ylabel("Emissions [MT]")
    fig.tight_layout()

    fig.savefig(save)


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
        color_discrete_map=color_palette.to_dict(),
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
        color_discrete_map=color_palette.to_dict(),
    )

    title = create_title("Technology Emissions", **wildcards)
    fig.update_layout(
        title=dict(text=title, font=dict(size=TITLE_SIZE)),
        xaxis_title="",
        yaxis_title="Emissions [MT]",
    )
    fig.write_html(save)


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
        color=color_palette.to_dict(),
    )

    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    ax.set_title(create_title("Technology Emissions", **wildcards))
    ax.set_ylabel("Emissions [MT]")
    fig.tight_layout()

    fig.savefig(save)


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

    energy_mix = energy_mix[[x for x in carriers_2_plot if x in energy_mix]]
    energy_mix = energy_mix.rename(columns=n.carriers.nice_name)
    ########

    energy_mix["Demand"] = get_demand_timeseries(n).mul(1e-3)  # MW -> GW

    # plot

    color_palette = get_color_palette(n)

    fig = px.area(
        energy_mix,
        x=energy_mix.index,
        y=[c for c in energy_mix.columns if c != "Demand"],
        color_discrete_map=color_palette.to_dict(),
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
            carriers_2_plot.append("battery_charger")
            carriers_2_plot.append("battery_discharger")
    energy_mix = energy_mix[[x for x in carriers_2_plot if x in energy_mix]]
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
                color=color_palette.to_dict(),
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


def plot_production_bar(
    n: pypsa.Network,
    carriers_2_plot: list[str],
    save: str,
    **wildcards,
) -> None:
    """
    Plot production per carrier.
    """

    # get data

    energy_mix = (
        get_energy_timeseries(n)
        # .rename(columns={"battery charger": "battery", "battery discharger": "battery"})
        .groupby(level=0, axis=1)
        .sum()
        .sum()
        .mul(1e-3)  # MW -> GW
    )
    energy_mix = pd.DataFrame(energy_mix, columns=["Production"]).reset_index(
        names="carrier",
    )
    energy_mix = energy_mix[
        energy_mix.carrier.isin([x for x in carriers_2_plot if x != "battery"])
    ].copy()
    energy_mix["color"] = energy_mix.carrier.map(n.carriers.color)
    energy_mix["carrier"] = energy_mix.carrier.map(n.carriers.nice_name)

    # plot

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.barplot(data=energy_mix, y="carrier", x="Production", palette=energy_mix.color)

    ax.set_title(create_title("Production [GWh]", **wildcards))
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

    operational_costs = get_operational_costs(n).sum().mul(1e-9)  # $ -> M$
    capital_costs = get_capital_costs(n).mul(1e-9)  # $ -> M$

    costs = pd.concat(
        [operational_costs, capital_costs],
        axis=1,
        keys=["OPEX", "CAPEX"],
    ).reset_index()
    costs = costs[costs.carrier.isin(carriers_2_plot)]
    costs["carrier"] = costs.carrier.map(n.carriers.nice_name)
    costs = costs.groupby("carrier").sum().reset_index()  # groups batteries

    # plot data

    fig, ax = plt.subplots(figsize=(10, 10))
    color_palette = n.carriers.reset_index().set_index("nice_name").to_dict()["color"]
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


def plot_capacity_map(
    n: pypsa.Network,
    bus_values: pd.DataFrame,
    line_values: pd.DataFrame,
    link_values: pd.DataFrame,
    regions: gpd.GeoDataFrame,
    bus_scale=1,
    line_scale=1,
    title=None,
) -> tuple[plt.figure, plt.axes]:
    """
    Generic network plotting function for capacity pie charts at each node.
    """

    fig, ax = plt.subplots(
        figsize=(10, 10),
        subplot_kw={"projection": ccrs.EqualEarth(n.buses.x.mean())},
    )

    line_width = line_values / line_scale
    link_width = link_values / line_scale

    # temp hack for battery colors
    bus_colors = pd.concat(
        [
            n.carriers.color,
            # pd.Series(
            #     [n.carriers.color["battery"], n.carriers.color["battery"]],
            #     index=["battery charger", "battery discharger"],
            # ),
        ],
    )

    with plt.rc_context({"patch.linewidth": 0.1}):
        n.plot(
            bus_sizes=bus_values / bus_scale,
            bus_colors=bus_colors,
            bus_alpha=0.7,
            line_widths=line_width,
            link_widths=0 if link_width.empty else link_width,
            line_colors="teal",
            ax=ax,
            margin=0.2,
            color_geomap=None,
        )

    # onshore regions
    regions.plot(
        ax=ax,
        facecolor="whitesmoke",
        edgecolor="white",
        aspect="equal",
        transform=ccrs.PlateCarree(),
        linewidth=1.2,
    )
    ax.set_extent(regions.total_bounds[[0, 2, 1, 3]])

    legend_kwargs = {"loc": "upper left", "frameon": False}
    bus_sizes = [5000, 10e3, 50e3]  # in MW
    line_sizes = [2000, 5000]  # in MW

    add_legend_circles(
        ax,
        [s / bus_scale for s in bus_sizes],
        [f"{s / 1000} GW" for s in bus_sizes],
        legend_kw={"bbox_to_anchor": (1, 1), **legend_kwargs},
    )
    add_legend_lines(
        ax,
        [s / line_scale for s in line_sizes],
        [f"{s / 1000} GW" for s in line_sizes],
        legend_kw={"bbox_to_anchor": (1, 0.8), **legend_kwargs},
    )
    add_legend_patches(
        ax,
        n.carriers.color,
        n.carriers.nice_name,
        legend_kw={"bbox_to_anchor": (1, 0), **legend_kwargs, "loc": "lower left"},
    )
    if not title:
        ax.set_title(f"Capacity (MW)", fontsize=TITLE_SIZE, pad=20)
    else:
        ax.set_title(title, fontsize=TITLE_SIZE, pad=20)
    fig.tight_layout()

    return fig, ax


def plot_demand_map(
    n: pypsa.Network,
    regions: gpd.GeoDataFrame,
    carriers: list[str],
    save: str,
    **wildcards,
) -> None:
    """
    Plots map of network nodal demand.
    """

    # get data

    bus_values = get_demand_base(n).mul(1e-3)
    line_values = n.lines.s_nom
    link_values = n.links.p_nom.replace(0)

    # plot data
    title = create_title("Network Demand", **wildcards)
    interconnect = wildcards.get("interconnect", None)
    bus_scale = get_bus_scale(interconnect) if interconnect else 1
    line_scale = get_line_scale(interconnect) if interconnect else 1

    fig, ax = plt.subplots(
        figsize=(10, 10),
        subplot_kw={"projection": ccrs.EqualEarth(n.buses.x.mean())},
    )

    line_width = line_values / line_scale
    link_width = link_values / line_scale

    with plt.rc_context({"patch.linewidth": 0.1}):
        n.plot(
            bus_sizes=bus_values / bus_scale,
            # bus_colors=None,
            bus_alpha=0.7,
            line_widths=line_width,
            link_widths=0 if link_width.empty else link_width,
            line_colors="teal",
            ax=ax,
            margin=0.2,
            color_geomap=None,
        )

    # onshore regions
    regions.plot(
        ax=ax,
        facecolor="whitesmoke",
        edgecolor="white",
        aspect="equal",
        transform=ccrs.PlateCarree(),
        linewidth=1.2,
    )
    ax.set_extent(regions.total_bounds[[0, 2, 1, 3]])

    legend_kwargs = {"loc": "upper left", "frameon": False}
    bus_sizes = [5000, 10e3, 50e3]  # in MW
    line_sizes = [2000, 5000]  # in MW

    add_legend_circles(
        ax,
        [s / bus_scale for s in bus_sizes],
        [f"{s / 1000} GW" for s in bus_sizes],
        legend_kw={"bbox_to_anchor": (1, 1), **legend_kwargs},
    )
    add_legend_lines(
        ax,
        [s / line_scale for s in line_sizes],
        [f"{s / 1000} GW" for s in line_sizes],
        legend_kw={"bbox_to_anchor": (1, 0.8), **legend_kwargs},
    )
    add_legend_patches(
        ax,
        n.carriers.color,
        n.carriers.nice_name,
        legend_kw={"bbox_to_anchor": (1, 0), **legend_kwargs, "loc": "lower left"},
    )
    if not title:
        ax.set_title(f"Total Annual Demand (MW)", fontsize=TITLE_SIZE, pad=20)
    else:
        ax.set_title(title, fontsize=TITLE_SIZE, pad=20)
    fig.tight_layout()
    fig.savefig(save)


def plot_base_capacity_map(
    n: pypsa.Network,
    regions: gpd.GeoDataFrame,
    carriers: list[str],
    save: str,
    **wildcards,
) -> None:
    """
    Plots map of base network capacities.
    """

    # get data

    bus_values = get_capacity_base(n)
    bus_values = bus_values[bus_values.index.get_level_values(1).isin(carriers)]
    bus_values = remove_sector_buses(bus_values).groupby(by=["bus", "carrier"]).sum()

    line_values = n.lines.s_nom
    link_values = n.links.p_nom.replace(0)

    # plot data

    title = create_title("Base Network Capacities", **wildcards)
    interconnect = wildcards.get("interconnect", None)
    bus_scale = get_bus_scale(interconnect) if interconnect else 1
    line_scale = get_line_scale(interconnect) if interconnect else 1

    fig, _ = plot_capacity_map(
        n=n,
        bus_values=bus_values,
        line_values=line_values,
        # link_values=link_values,
        link_values=pd.DataFrame(),
        regions=regions,
        line_scale=line_scale,
        bus_scale=bus_scale,
        title=title,
    )
    fig.savefig(save)


def plot_opt_capacity_map(
    n: pypsa.Network,
    regions: gpd.GeoDataFrame,
    carriers: list[str],
    save: str,
    opt_capacity: str = "greenfield",
    retirement_method: str = "economic",
    **wildcards,
) -> None:
    """
    Plots map of optimal network capacities.
    """

    # get data

    if opt_capacity == "greenfield":
        bus_values = get_capacity_greenfield(n, retirement_method)
    elif opt_capacity == "brownfield":
        bus_values = get_capacity_brownfield(n, retirement_method)
    else:
        logger.error(
            f"Capacity method must be one of 'greenfield' or 'brownfield'. Recieved {opt_capacity}.",
        )
        raise NotImplementedError

    # a little awkward to fix color plotting referece issue
    bus_values = bus_values[bus_values.index.get_level_values("carrier").isin(carriers)]
    bus_values = (
        remove_sector_buses(bus_values)
        .reset_index()
        .groupby(by=["bus", "carrier"])
        .sum()
        .squeeze()
    )

    line_values = n.lines.s_nom_opt
    # link_values = n.links.p_nom_opt

    # plot data

    title = create_title(f"Optimal {opt_capacity} Network Capacities", **wildcards)
    interconnect = wildcards.get("interconnect", None)
    bus_scale = get_bus_scale(interconnect) if interconnect else 1
    line_scale = get_line_scale(interconnect) if interconnect else 1

    fig, _ = plot_capacity_map(
        n=n,
        bus_values=bus_values.copy(),
        line_values=line_values,
        link_values=n.links.p_nom.replace(0),
        regions=regions,
        line_scale=line_scale,
        bus_scale=bus_scale,
        title=title,
    )
    fig.savefig(save)


def plot_new_capacity_map(
    n: pypsa.Network,
    regions: gpd.GeoDataFrame,
    carriers: list[str],
    save: str,
    opt_capacity: str = "greenfield",
    retirement_method: str = "economic",
    **wildcards,
) -> None:
    """
    Plots map of new capacity.
    """

    # get data

    bus_pnom = get_capacity_base(n)
    if opt_capacity == "greenfield":
        bus_pnom_opt = get_capacity_greenfield(n, retirement_method)
    elif opt_capacity == "brownfield":
        bus_pnom_opt = get_capacity_brownfield(n, retirement_method)
    else:
        logger.error(
            f"Capacity method must be one of 'greenfield' or 'brownfield'. Recieved {opt_capacity}.",
        )
        raise NotImplementedError

    # awkward processing to fix color plotting issue

    bus_values = bus_pnom_opt - bus_pnom
    bus_values = bus_values[
        (bus_values > 0) & (bus_values.index.get_level_values(1).isin(carriers))
    ]
    bus_values = (
        remove_sector_buses(bus_values)
        .reset_index()
        .groupby(by=["bus", "carrier"])
        .sum()
        .squeeze()
    )

    line_snom = n.lines.s_nom
    line_snom_opt = n.lines.s_nom_opt
    line_values = line_snom_opt - line_snom

    # link_pnom = n.links.p_nom
    # link_pnom_opt = n.links.p_nom_opt
    # link_values = link_pnom_opt - link_pnom

    # plot data

    title = create_title("New Network Capacities", **wildcards)
    interconnect = wildcards.get("interconnect", None)
    bus_scale = get_bus_scale(interconnect) if interconnect else 1
    line_scale = get_line_scale(interconnect) if interconnect else 1

    fig, _ = plot_capacity_map(
        n=n,
        bus_values=bus_values,
        line_values=line_values,
        # link_values=n.links.p_nom.replace(0),
        link_values=pd.DataFrame(),
        regions=regions,
        line_scale=line_scale,
        bus_scale=bus_scale,
        title=title,
    )
    fig.savefig(save)


def plot_renewable_potential(
    n: pypsa.Network,
    regions: gpd.GeoDataFrame,
    save: str,
    **wildcards,
) -> None:
    """
    Plots wind and solar resource potential by node.
    """

    # get data

    renew = n.generators[
        (n.generators.p_nom_max != np.inf)
        & (
            n.generators.carrier.isin(
                ["onwind", "offwind", "offwind_floating", "solar"],
            )
        )
    ]
    bus_values = renew.groupby(["bus", "carrier"]).p_nom_max.sum()

    # do not show lines or links
    line_values = pd.Series(0, index=n.lines.s_nom.index)
    link_values = pd.Series(0, index=n.links.p_nom.index)

    # plot data

    title = create_title("Renewable Capacity Potential", **wildcards)
    interconnect = wildcards.get("interconnect", None)
    bus_scale = get_bus_scale(interconnect) if interconnect else 1

    bus_scale *= 12  # since potential capacity is so big

    fig, ax = plot_capacity_map(
        n=n,
        bus_values=bus_values,
        line_values=line_values,
        link_values=link_values,
        regions=regions,
        bus_scale=bus_scale,
        title=title,
    )

    # only show renewables in legend
    fig.artists[-2].remove()  # remove line width legend
    fig.artists[-1].remove()  # remove existing colour legend
    renew_carriers = n.carriers[
        n.carriers.index.isin(["onwind", "offwind", "offwind_floating", "solar"])
    ]
    add_legend_patches(
        ax,
        renew_carriers.color,
        renew_carriers.nice_name,
        legend_kw={"bbox_to_anchor": (1, 0), "frameon": False, "loc": "lower left"},
    )

    fig.savefig(save)


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

    # color_palette = get_color_palette(n)
    color_palette = n.carriers.reset_index().set_index("nice_name").to_dict()["color"]
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

    # create plots
    plot_base_capacity_map(
        n,
        onshore_regions,
        carriers,
        snakemake.output["capacity_map_base.pdf"],
        **snakemake.wildcards,
    )
    plot_opt_capacity_map(
        n,
        onshore_regions,
        carriers,
        snakemake.output["capacity_map_optimized.pdf"],
        "greenfield",
        retirement_method,
        **snakemake.wildcards,
    )
    plot_opt_capacity_map(
        n,
        onshore_regions,
        carriers,
        snakemake.output["capacity_map_optimized_brownfield.pdf"],
        "brownfield",
        retirement_method,
        **snakemake.wildcards,
    )
    plot_new_capacity_map(
        n,
        onshore_regions,
        carriers,
        snakemake.output["capacity_map_new.pdf"],
        "greenfield",
        retirement_method,
        **snakemake.wildcards,
    )
    plot_demand_map(
        n,
        onshore_regions,
        carriers,
        snakemake.output["demand_map.pdf"],
        **snakemake.wildcards,
    )
    plot_capacity_additions_bar(
        n,
        carriers,
        snakemake.output["capacity_additions_bar.pdf"],
        "greenfield",
        retirement_method,
        **snakemake.wildcards,
    )
    plot_costs_bar(n, carriers, snakemake.output["costs_bar.pdf"], **snakemake.wildcards)
    plot_production_bar(
        n,
        carriers,
        snakemake.output["production_bar.pdf"],
        **snakemake.wildcards,
    )
    plot_production_area(
        n,
        carriers,
        snakemake.output["production_area.pdf"],
        **snakemake.wildcards,
    )
    plot_production_html(
        n,
        carriers,
        snakemake.output["production_area.html"],
        **snakemake.wildcards,
    )
    plot_hourly_emissions(n, snakemake.output["emissions_area.pdf"], **snakemake.wildcards)
    plot_hourly_emissions_html(
        n,
        snakemake.output["emissions_area.html"],
        **snakemake.wildcards,
    )
    plot_accumulated_emissions(
        n,
        snakemake.output["emissions_accumulated.pdf"],
        **snakemake.wildcards,
    )
    plot_accumulated_emissions_tech(
        n,
        snakemake.output["emissions_accumulated_tech.pdf"],
        **snakemake.wildcards,
    )
    plot_accumulated_emissions_tech_html(
        n,
        snakemake.output["emissions_accumulated_tech.html"],
        **snakemake.wildcards,
    )
    # plot_node_emissions_html(n, snakemake.output["emissions_node_html"], **snakemake.wildcards)
    plot_region_emissions_html(
        n,
        snakemake.output["emissions_region.html"],
        **snakemake.wildcards,
    )
    plot_emissions_map(
        n,
        onshore_regions,
        snakemake.output["emissions_map.pdf"],
        **snakemake.wildcards,
    )
    plot_renewable_potential(
        n,
        onshore_regions,
        snakemake.output["renewable_potential_map.pdf"],
        **snakemake.wildcards,
    )
