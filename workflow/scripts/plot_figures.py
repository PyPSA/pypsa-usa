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

def plot_capacity_map(n, bus_values, generating_link_carrier_map, regions_onshore, n_clusters):
    '''
    Plot the map of network where each node is a pie chart defined by the input of bus_values
    '''
    if snakemake.wildcards.interconnect != "usa":
        bus_scale = 1e5
    else:
        bus_scale = 4e4
    if snakemake.wildcards.interconnect != "usa":
        line_scale = 2e3
    else:
        line_scale = 4e3


    fig, ax = plt.subplots(
        figsize=(8, 8), subplot_kw={"projection": ccrs.EqualEarth(n.buses.x.mean())}
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
    
    return fig, ax


def create_area_plot(production, demand, colors, snapshots, output_path, timeslice):
    fig, ax = plt.subplots(figsize=(14, 4))
    production[snapshots].plot.area(ax=ax, color=colors, alpha=0.7, legend="reverse")
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    ax.set_ylabel("Power [GW]")
    ax.set_xlabel("")
    fig.tight_layout()
    suffix = (
        "-" + datetime.strptime(str(timeslice), "%m").strftime("%b")
        if timeslice != "all"
        else ""
    )
    fig.savefig(output_path.parent / (output_path.stem + suffix + output_path.suffix))


def create_total_production_bar(total_production, colors, output_path):
    fig, ax = plt.subplots()
    total_production.div(1e3).plot.bar(color=colors, ax=ax)
    ax.set_ylabel("Total production [TWh]")
    ax.set_xlabel("")
    fig.savefig(output_path)


def create_cost_bar(operational_costs, capital_costs, colors, output_path):
    fig, ax = plt.subplots()

    costs = pd.concat([operational_costs, capital_costs], axis=1, keys=["OPEX", "CAPEX"])
    costs = costs.reset_index("carrier")

    s1 = sns.barplot(y="carrier", x="CAPEX", data=costs, alpha=0.6, ax=ax, palette=colors)
    s2 = sns.barplot(y="carrier", x="OPEX", data=costs, ax=ax, left=costs["CAPEX"], palette=colors)

    ax.set_ylabel("")
    ax.set_xlabel("CAPEX & OPEX [$]")
    fig.savefig(output_path)

def main(snakemake):
    logger = logging.getLogger(__name__)

    os.chdir(os.getcwd())
    rootpath = "."

    #Import Files
    n = pypsa.Network(snakemake.input.network)
    regions_onshore = gpd.read_file(snakemake.input.regions_onshore)
    
    #Parameters & Wildcards
    n_clusters = snakemake.wildcards.clusters
    ll = snakemake.wildcards.ll
    opts = snakemake.wildcards.opts
    n_hours = snakemake.config['solving']['options']['nhours']

    #Plotting Settings
    sns.set_theme("paper", style="darkgrid")

    generating_link_carrier_map = {"fuel cell": "H2", "battery discharger": "battery"}

    #Base Capacity Map 
    gen_pnom = n.generators.groupby(["bus", "carrier"]).p_nom.sum()
    storage_capacity = (n.storage_units.groupby(["bus", "carrier"]).p_nom.sum())
    bus_base_capacities = pd.concat([gen_pnom, storage_capacity])
    fig, ax = plot_capacity_map(n, bus_base_capacities, generating_link_carrier_map, regions_onshore, n_clusters)
    fig.savefig(snakemake.output[0])

    #Optimized Capacity Map
    gen_pnom_opt = n.generators.groupby(["bus", "carrier"]).p_nom_opt.sum()
    storage_capacity_opt = (n.storage_units.groupby(["bus", "carrier"]).p_nom_opt.sum())
    bus_opt_capacities = pd.concat([gen_pnom_opt, storage_capacity_opt])
    fig, ax = plot_capacity_map(n, bus_opt_capacities, generating_link_carrier_map, regions_onshore, n_clusters)
    fig.savefig(snakemake.output[1])

    #Built Capacity Map
    gen_built = gen_pnom_opt - gen_pnom
    storage_built = storage_capacity_opt - storage_capacity
    buses_built_capacities = pd.concat([gen_built, storage_built])
    fig, ax = plot_capacity_map(n, buses_built_capacities, generating_link_carrier_map, regions_onshore, n_clusters)
    fig.savefig(snakemake.output[2])


    carriers_gen = n.generators.carrier
    carriers_storageUnits = n.storage_units.carrier
    production = n.generators_t.p.groupby(carriers_gen, axis=1).sum().rename(columns=n.carriers.nice_name)/ 1e3
    # storage = network.storage_units_t.p.groupby(carriers_storageUnits, axis=1).sum()/1e3
    # storage_charge = storage[storage > 0].fillna(0).rename(columns={'battery':'Battery Discharging'})
    # storage_discharge = storage[storage < 0].fillna(0).rename(columns={'battery':'Battery Charging'})
    # energymix = pd.concat([production, storage_charge, storage_discharge], axis=1)
    demand = n.loads_t.p.sum(1).rename("Demand") / 1e3

    if n_hours < 8760:
        enddate = pd.to_datetime(snakemake.config["snapshots"]["end"])
        year = enddate.year
        for timeslice in list(range(1, enddate.month)) + ["all"]:
            snapshots = (
                n.snapshots.get_loc(f"{year}-{timeslice}")
                if timeslice != "all"
                else slice(None, None)
            )
            create_area_plot(production, demand, colors, snapshots, snakemake.output.operation_area, timeslice)
    else:
        for timeslice in list(range(1, 12)) + ["all"]:
            snapshots = (
                n.snapshots.get_loc(f"{year}-{timeslice}")
                if timeslice != "all"
                else slice(None, None)
            )
            create_area_plot(production, demand, colors, snapshots, snakemake.output.operation_area, timeslice)

    create_total_production_bar(total_production, colors, snakemake.output.operation_bar)

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

    create_cost_bar(operational_costs, capital_costs, colors, snakemake.output.cost_bar)



if __name__ == "__main__":
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake('plot_figures', 
                                    interconnect='western',
                                    clusters=60,
                                    ll='vopt',
                                    opts='[Co2L0.75]',
                                    )
    main(snakemake)
