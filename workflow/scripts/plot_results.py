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


if __name__ == "__main__":
    os.chdir(os.getcwd())
    sns.set_theme("paper", style="darkgrid")

    logger = logging.getLogger(__name__)
    rootpath = "."

    n = pypsa.Network(snakemake.input.network)
    regions_onshore = gpd.read_file(snakemake.input.regions_onshore)
    n_clusters = snakemake.wildcards.clusters
    # import pdb; pdb.set_trace()
    # TODO
    n.carriers.loc["wind", "color"] = "lightblue"
    n.carriers.loc["offwind", "color"] = "lightblue"
    n.carriers.loc["offwind", "nice_name"] = "Offshore Wind"
    n.carriers.loc["ng", "color"] = "indianred"
    # n.carriers.loc["dfo", "color"] = "k"
    # n.carriers.loc["other", "color"] = "grey"

    # TODO
    generating_link_carrier_map = {"fuel cell": "H2", "battery discharger": "battery"}


    TITLE_SIZE = 16

    ######
    ### Capacity Plots
    ######

    if snakemake.wildcards.interconnect != "usa":
        bus_scale = 1e5
    else:
        bus_scale = 4e4

    if snakemake.wildcards.interconnect != "usa":
        line_scale = 2e3
    else:
        line_scale = 4e3

    if snakemake.config['solving']['options']['load_shedding']: 
        n.carriers.rename(index={'Load':'load'}, inplace=True)
        n.carriers.loc['load','nice_name'] = 'Load'
        n.carriers.loc['load','color'] = '#70af1d'

    ###### 
    ## Capacity Map
    # Map of all the CURRENT capacities in the Network. Including buses, lines, and links
    ######

    fig, ax = plt.subplots(
        figsize=(8, 8), subplot_kw={"projection": ccrs.EqualEarth(n.buses.x.mean())}
    )
    gens_df = n.generators.drop(n.generators.index[n.generators.carrier == 'load'])
    generation_capacity = gens_df.groupby(["bus", "carrier"]).p_nom.sum()
    storage_capacity = (
        n.links.query("carrier in @generating_link_carrier_map")
        .groupby(["bus1", "carrier"])
        .p_nom.sum()
    )
    storage_capacity = storage_capacity.rename(index=generating_link_carrier_map, level=1)
    buses = pd.concat([generation_capacity, storage_capacity])

    with plt.rc_context({"patch.linewidth": 0.1}):
        n.plot(
            bus_sizes=buses / bus_scale,
            bus_alpha=0.7,
            line_widths=n.lines.s_nom / line_scale,
            link_widths=n.links.p_nom / line_scale,
            line_colors="teal",
            ax=ax,
            margin=0.2,
            color_geomap=None,
        )
    regions_onshore.plot(
        ax=ax,
        facecolor="whitesmoke",
        edgecolor="white",
        aspect="equal",
        transform=ccrs.PlateCarree(),
        linewidth=1.2,
    )
    ax.set_extent(regions_onshore.total_bounds[[0, 2, 1, 3]])

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
    ax.set_title(f'Base Network Capacities  (#clusters = {n_clusters})', fontsize=TITLE_SIZE)
    fig.tight_layout()
    fig.savefig(snakemake.output.capacity_map_base)

    # import pdb; pdb.set_trace()

    ###### 
    ## Capacity Map
    # Map of all the added OPTIMIZED capacities in the Network. Including buses, lines, and links
    ######

    fig, ax = plt.subplots(
        figsize=(8, 8), subplot_kw={"projection": ccrs.EqualEarth(n.buses.x.mean())}
    )
    generation_capacity = gens_df.groupby(["bus", "carrier"]).p_nom_opt.sum()
    storage_capacity = (
        n.links.query("carrier in @generating_link_carrier_map")
        .groupby(["bus1", "carrier"])
        .p_nom_opt.sum()
    )
    storage_capacity = storage_capacity.rename(index=generating_link_carrier_map, level=1)
    buses = pd.concat([generation_capacity, storage_capacity])
    with plt.rc_context({"patch.linewidth": 0.1}):
        n.plot(
            bus_sizes=buses / bus_scale,
            bus_alpha=0.7,
            line_widths=n.lines.s_nom_opt / line_scale,
            link_widths=n.links.p_nom_opt / line_scale,
            line_colors="teal",
            ax=ax,
            margin=0.2,
            color_geomap=None,
        )
    regions_onshore.plot(
        ax=ax,
        facecolor="whitesmoke",
        edgecolor="white",
        aspect="equal",
        transform=ccrs.PlateCarree(),
        linewidth=1.2,
    )
    ax.set_extent(regions_onshore.total_bounds[[0, 2, 1, 3]])

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
    ax.set_title(f'Optimized Network Capacities  (#clusters = {n_clusters})', fontsize=TITLE_SIZE)
    fig.tight_layout()
    fig.savefig(snakemake.output.capacity_map_optimized)

    ####
    ## Capacity Bar Plot
    # Bar plot of all the added optimized capacities in the Network
    ####

    fig, ax = plt.subplots()
    capacities = n.generators.groupby("carrier").p_nom_opt.sum()
    capacities.rename(n.carriers.nice_name, inplace=True)
    colors = n.carriers.set_index("nice_name").color[capacities.index]
    capacities.div(1e3).plot.bar(color=colors, ax=ax)
    ax.set_ylabel("Total capacity [GW]")
    ax.set_xlabel("")
    ax.set_title(f'Optimized Network Capacities (#clusters = {n_clusters})', fontsize=TITLE_SIZE)
    fig.savefig(snakemake.output.capacity_bar)


    ######
    ### Operation Plots
    ######

    if snakemake.wildcards.interconnect != "usa":
        bus_scale = 5e7
    else:
        bus_scale = 5e7

    if snakemake.wildcards.interconnect != "usa":
        line_scale = 1e6
    else:
        line_scale = 1e7

    # import pdb; pdb.set_trace()
    ######
    ## Operation Map
    # Map of all the optimized operation in the Network. Including buses, lines, and links
    ######

    fig, ax = plt.subplots(
        figsize=(8, 8), subplot_kw={"projection": ccrs.EqualEarth(n.buses.x.mean())}
    )
    with plt.rc_context({"patch.linewidth": 0.1}):
        n.plot(
            bus_sizes=n.generators_t.p.sum()
            .groupby([n.generators.bus, n.generators.carrier])
            .sum()
            / bus_scale,
            bus_alpha=0.7,
            line_widths=n.lines_t.p0.sum() / line_scale,
            link_widths=n.links_t.p0.sum() / line_scale,
            line_colors="teal",
            ax=ax,
            margin=0.2,
            color_geomap=None,
        )
    regions_onshore.plot(
        ax=ax,
        facecolor="whitesmoke",
        edgecolor="white",
        aspect="equal",
        transform=ccrs.PlateCarree(),
        linewidth=1.2,
    )
    ax.set_extent(regions_onshore.total_bounds[[0, 2, 1, 3]])


    legend_kwargs = {"loc": "upper left", "frameon": False}
    bus_sizes = [1000000, 2000000, 5000000]  # in MW
    line_sizes = [2000000, 5000000]  # in MW
    add_legend_circles(
        ax,
        [s / bus_scale for s in bus_sizes],
        [f"{s / 1e6} TWh" for s in bus_sizes],
        legend_kw={"bbox_to_anchor": (1, 1), **legend_kwargs},
    )
    add_legend_lines(
        ax,
        [s / line_scale for s in line_sizes],
        [f"{s / 1e6} TWh" for s in line_sizes],
        legend_kw={"bbox_to_anchor": (1, 0.8), **legend_kwargs},
    )
    add_legend_patches(
        ax,
        n.carriers.color,
        n.carriers.nice_name,
        legend_kw={"bbox_to_anchor": (1, 0.0), **legend_kwargs, "loc": "lower left"},
    )
    fig.tight_layout()
    ax.set_title("Optimized Network Operations [MWh]", fontsize= TITLE_SIZE)
    fig.savefig(snakemake.output.operation_map)
        

    ####
    ## Operations Time-Series Plots
    ####
    carriers = n.generators.carrier
    production = (
        n.generators_t.p.groupby(carriers, axis=1)
        .sum()
        .rename(columns=n.carriers.nice_name)
        / 1e3
    )
    production = production.loc[:, production.sum() > 0.1]
    demand = n.loads_t.p.sum(1).rename("Demand") / 1e3
    colors = n.carriers.set_index("nice_name").color[production.columns]

    if snakemake.config['solving']['options']['nhours'] < 8760:
        nhours= snakemake.config['solving']['options']['nhours']
        enddate = pd.to_datetime('2019-01-01') + pd.Timedelta(nhours%24,'h')
        enddate

        for timeslice in list(range(1, enddate.month)) + ["all"]:
            snapshots = (
                n.snapshots.get_loc(f"2019-{timeslice}")
                if timeslice != "all"
                else slice(None, None)
            )
            fig, ax = plt.subplots(figsize=(14, 4))
            production[snapshots].plot.area(ax=ax, color=colors, alpha=0.7, legend="reverse")
            # demand.plot.line(ax=ax, ls='-', color='darkblue')
            ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
            ax.set_ylabel("Power [GW]")
            ax.set_xlabel("")
            fig.tight_layout()
            suffix = (
                "-" + datetime.strptime(str(timeslice), "%m").strftime("%b")
                if timeslice != "all"
                else ""
            )
            path = Path(snakemake.output.operation_area)
            fig.savefig(path.parent / (path.stem + suffix + path.suffix))

    else:
        for timeslice in list(range(1, 12)) + ["all"]:
            snapshots = (
                n.snapshots)
            fig, ax = plt.subplots(figsize=(14, 4))
            production[snapshots].plot.area(ax=ax, color=colors, alpha=0.7, legend="reverse")
            # demand.plot.line(ax=ax, ls='-', color='darkblue')
            ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
            ax.set_ylabel("Power [GW]")
            ax.set_xlabel("")
            fig.tight_layout()
            suffix = (
                "-" + datetime.strptime(str(timeslice), "%m").strftime("%b")
                if timeslice != "all"
                else ""
            )
            path = Path(snakemake.output.operation_area)
            fig.savefig(path.parent / (path.stem + suffix + path.suffix))



    fig, ax = plt.subplots()
    total_production = n.snapshot_weightings.generators @ production
    total_production.div(1e3).plot.bar(color=colors, ax=ax)
    ax.set_ylabel("Total production [TWh]")
    ax.set_xlabel("")
    fig.savefig(snakemake.output.operation_bar)


    fig, ax = plt.subplots()

    production = n.generators_t.p
    operational_costs = (
        (production * n.generators.marginal_cost)
        .groupby(carriers, axis=1)
        .sum()
        .rename(columns=n.carriers.nice_name)
    ).sum()

    capital_costs = (
        n.generators.eval("p_nom_opt * capital_cost")
        .groupby(carriers)
        .sum()
        .rename(n.carriers.nice_name)
    )

    costs = pd.concat([operational_costs, capital_costs], axis=1, keys=["OPEX", "CAPEX"])
    costs = costs.reset_index("carrier")


    s1 = sns.barplot(y="carrier", x="CAPEX", data=costs, alpha=0.6, ax=ax, palette=colors)
    s2 = sns.barplot(
        y="carrier", x="OPEX", data=costs, ax=ax, left=costs["CAPEX"], palette=colors
    )

    ax.set_ylabel("")
    ax.set_xlabel("CAPEX & OPEX [$]")
    fig.savefig(snakemake.output.cost_bar)