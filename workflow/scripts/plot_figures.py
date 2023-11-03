"""Plots reults from optimzation"""

import sys
import os
import pypsa
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import logging
from datetime import datetime
from cartopy import crs as ccrs
from pypsa.plot import add_legend_circles, add_legend_lines, add_legend_patches

import logging
logger = logging.getLogger(__name__)
from _helpers import configure_logging

import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs

import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs


# Global Plotting Settings
TITLE_SIZE = 16


# def create_area_plot(production, demand, colors, snapshots, output_path, timeslice):
#     fig, ax = plt.subplots(figsize=(14, 4))
#     production[snapshots].plot.area(ax=ax, color=colors, alpha=0.7, legend="reverse")
#     ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
#     ax.set_ylabel("Power [GW]")
#     ax.set_xlabel("")
#     fig.tight_layout()
#     suffix = (
#         "-" + datetime.strptime(str(timeslice), "%m").strftime("%b")
#         if timeslice != "all"
#         else ""
#     )
#     fig.savefig(output_path.parent / (output_path.stem + suffix + output_path.suffix))


# def create_total_production_bar(total_production, colors, output_path):
#     fig, ax = plt.subplots()
#     total_production.div(1e3).plot.bar(color=colors, ax=ax)
#     ax.set_ylabel("Total production [TWh]")
#     ax.set_xlabel("")
#     fig.savefig(output_path)


# def create_cost_bar(operational_costs, capital_costs, colors, output_path):
#     fig, ax = plt.subplots()

#     costs = pd.concat([operational_costs, capital_costs], axis=1, keys=["OPEX", "CAPEX"])
#     costs = costs.reset_index("carrier")

#     s1 = sns.barplot(y="carrier", x="CAPEX", data=costs, alpha=0.6, ax=ax, palette=colors)
#     s2 = sns.barplot(y="carrier", x="OPEX", data=costs, ax=ax, left=costs["CAPEX"], palette=colors)

#     ax.set_ylabel("")
#     ax.set_xlabel("CAPEX & OPEX [$]")
#     fig.savefig(output_path)

# def main(snakemake):

#     carriers_gen = n.generators.carrier
#     carriers_storageUnits = n.storage_units.carrier
#     production = n.generators_t.p.groupby(carriers_gen, axis=1).sum().rename(columns=n.carriers.nice_name)/ 1e3
#     # storage = network.storage_units_t.p.groupby(carriers_storageUnits, axis=1).sum()/1e3
#     # storage_charge = storage[storage > 0].fillna(0).rename(columns={'battery':'Battery Discharging'})
#     # storage_discharge = storage[storage < 0].fillna(0).rename(columns={'battery':'Battery Charging'})
#     # energymix = pd.concat([production, storage_charge, storage_discharge], axis=1)
#     demand = n.loads_t.p.sum(1).rename("Demand") / 1e3

#     if n_hours < 8760:
#         enddate = pd.to_datetime(snakemake.config["snapshots"]["end"])
#         year = enddate.year
#         for timeslice in list(range(1, enddate.month)) + ["all"]:
#             snapshots = (
#                 n.snapshots.get_loc(f"{year}-{timeslice}")
#                 if timeslice != "all"
#                 else slice(None, None)
#             )
#             create_area_plot(production, demand, colors, snapshots, snakemake.output.operation_area, timeslice)
#     else:
#         for timeslice in list(range(1, 12)) + ["all"]:
#             snapshots = (
#                 n.snapshots.get_loc(f"{year}-{timeslice}")
#                 if timeslice != "all"
#                 else slice(None, None)
#             )
#             create_area_plot(production, demand, colors, snapshots, snakemake.output.operation_area, timeslice)

#     create_total_production_bar(total_production, colors, snakemake.output.operation_bar)

#     operational_costs = (
#         (production * n.generators.marginal_cost)
#         .groupby(carriers, axis=1)
#         .sum()
#         .rename(columns=n.carriers.nice_name)
#     ).sum()

#     capital_costs = (
#         n.generators.eval("p_nom_opt * capital_cost")
#         .groupby(carriers)
#         .sum()
#         .rename(n.carriers.nice_name)
#     )

#     create_cost_bar(operational_costs, capital_costs, colors, snakemake.output.cost_bar)

def get_bus_scale(interconnect: str) -> float:
    """Scales lines based on interconnect size"""
    if interconnect != "usa":
        return 1e5
    else:
        return 4e4
    
def get_line_scale(interconnect: str) -> float:
    """Scales lines based on interconnect size"""
    if interconnect != "usa":
        return 2e3
    else:
        return 4e3

def create_title(title: str, **wildcards) -> str:
    """Standardizes wildcard writing in titles
    
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
    wildcards_joined = " | ".join(w)
    return f"{title} \n ({wildcards_joined})"

def plot_costs_bar(n: pypsa.Network, save:str, **wildcards) -> None:
    """Plot OPEX and CAPEX"""
    
    # get data 
    
    carriers = n.generators.carrier
    carrier_nice_names = n.carriers.nice_name
    production = n.generators_t.p
    marginal_cost = n.generators.marginal_cost
    
    operational_costs = (
        (production * marginal_cost)
        .groupby(carriers, axis=1)
        .sum()
        .rename(columns=carrier_nice_names)
    ).sum()

    capital_costs = (
        n.generators.eval("p_nom_opt * capital_cost")
        .groupby(carriers)
        .sum()
        .rename(n.carriers.nice_name)
    )
    costs = pd.concat([operational_costs, capital_costs], axis=1, keys=["OPEX", "CAPEX"]).reset_index()
    
    # plot 
    
    fig, ax = plt.subplots(figsize=(10, 10))
    color_palette = n.carriers.set_index("nice_name").to_dict()["color"]
    sns.barplot(y="carrier", x="CAPEX", data=costs, alpha=0.6, ax=ax, palette=color_palette)
    sns.barplot(y="carrier", x="OPEX", data=costs, ax=ax, left=costs["CAPEX"], palette=color_palette)
    
    ax.set_title(create_title("Costs", **wildcards))
    ax.set_ylabel("")
    ax.set_xlabel("CAPEX & OPEX [$]")
    fig.tight_layout()
    fig.savefig(save)

def plot_capacity_map(n: pypsa.Network, bus_values: pd.DataFrame, regions: gpd.GeoDataFrame, bus_scale=1, line_scale=1, title = None) -> (plt.figure, plt.axes):
    """
    Generic network plotting function for capacity pie charts at each node
    """

    fig, ax = plt.subplots(
        figsize=(10, 10), subplot_kw={"projection": ccrs.EqualEarth(n.buses.x.mean())}
    )

    with plt.rc_context({"patch.linewidth": 0.1}):
        n.plot(
            bus_sizes=bus_values / bus_scale,
            bus_alpha=0.7,
            line_widths=n.lines.s_nom / line_scale,
            link_widths=n.links.p_nom / line_scale,
            line_colors="teal",
            ax=ax,
            margin=0.2,
            color_geomap=None
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

def plot_base_capacity(n: pypsa.Network, regions: gpd.GeoDataFrame, save: str, **wildcards) -> None:
    """
    Plots base network capacities
    """
    gen_pnom = n.generators.groupby(["bus", "carrier"]).p_nom.sum()
    storage_pnom = n.storage_units.groupby(["bus", "carrier"]).p_nom.sum()
    bus_pnom = pd.concat([gen_pnom, storage_pnom])
    
    title = create_title("Base Network Capacities", **wildcards)
    interconnect = wildcards.get("interconnect", None)
    bus_scale = get_bus_scale(interconnect=wildcards["interconnect"]) if interconnect else 1
    line_scale = get_line_scale(interconnect=wildcards["interconnect"]) if interconnect else 1
    
    fig, _ = plot_capacity_map(
        n=n, 
        bus_values=bus_pnom,
        regions=regions,
        line_scale=line_scale,
        bus_scale=bus_scale,
        title=title
    )
    fig.savefig(save)

def plot_opt_capacity(n: pypsa.Network, regions: gpd.GeoDataFrame, save: str, **wildcards) -> None:
    """
    Plots optimal network capacities
    """
    gen_pnom_opt = n.generators.groupby(["bus", "carrier"]).p_nom_opt.sum()
    storage_pnom_opt = (n.storage_units.groupby(["bus", "carrier"]).p_nom_opt.sum())
    bus_pnom_opt = pd.concat([gen_pnom_opt, storage_pnom_opt])
    
    title = create_title("Optimal Network Capacities", **wildcards)
    interconnect = wildcards.get("interconnect", None)
    bus_scale = get_bus_scale(interconnect=wildcards["interconnect"]) if interconnect else 1
    line_scale = get_line_scale(interconnect=wildcards["interconnect"]) if interconnect else 1
    
    fig, _ = plot_capacity_map(
        n=n, 
        bus_values=bus_pnom_opt,
        regions=regions,
        line_scale=line_scale,
        bus_scale=bus_scale,
        title=title
    )
    fig.savefig(save)

def plot_new_capacity(n: pypsa.Network, regions: gpd.GeoDataFrame, save: str, **wildcards) -> None:
    """Plots new capacity"""
    gen_pnom = n.generators.groupby(["bus", "carrier"]).p_nom.sum()
    gen_pnom_opt = n.generators.groupby(["bus", "carrier"]).p_nom_opt.sum()
    gen_pnom_new = gen_pnom_opt - gen_pnom
    
    storage_pnom = n.storage_units.groupby(["bus", "carrier"]).p_nom.sum()
    storage_pnom_opt = (n.storage_units.groupby(["bus", "carrier"]).p_nom_opt.sum())
    storage_pnom_new = storage_pnom_opt - storage_pnom
    
    bus_pnom_new = pd.concat([gen_pnom_new, storage_pnom_new])
    
    title = create_title("New Network Capacities", **wildcards)
    interconnect = wildcards.get("interconnect", None)
    bus_scale = get_bus_scale(interconnect=wildcards["interconnect"]) if interconnect else 1
    line_scale = get_line_scale(interconnect=wildcards["interconnect"]) if interconnect else 1
    
    fig, _ = plot_capacity_map(
        n=n, 
        bus_values=bus_pnom_new,
        regions=regions,
        line_scale=line_scale,
        bus_scale=bus_scale,
        title=title
    )
    fig.savefig(save)

if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake(
            'plot_figures', 
            interconnect='western',
            clusters=60,
            ll='vopt',
            opts='Co2L0.75',
        )
    configure_logging(snakemake)
    
    # extract shared plotting files 
    n = pypsa.Network(snakemake.input.network)
    onshore_regions = gpd.read_file(snakemake.input.regions_onshore)
    n_clusters = snakemake.wildcards.clusters
    ll = snakemake.wildcards.ll
    opts = snakemake.wildcards.opts
    n_hours = snakemake.config['solving']['options']['nhours']
    
    # mappers 
    generating_link_carrier_map = {"fuel cell": "H2", "battery discharger": "battery"}
    
    # plotting theme
    sns.set_theme("paper", style="darkgrid")
    
    # create plots
    plot_base_capacity(n, onshore_regions, snakemake.output["capacity_map_base"], **snakemake.wildcards)
    plot_opt_capacity(n, onshore_regions, snakemake.output["capacity_map_optimized"], **snakemake.wildcards)
    plot_new_capacity(n, onshore_regions, snakemake.output["capacity_map_new"], **snakemake.wildcards)
    plot_costs_bar(n, snakemake.output["costs_bar"], **snakemake.wildcards)
    
