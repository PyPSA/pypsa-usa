import logging
from collections import OrderedDict
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
import seaborn as sns

logger = logging.getLogger(__name__)
from _helpers import configure_logging, get_snapshots
from constants import EIA_930_REGION_MAPPER, EIA_BA_2_REGION, STATE_2_CODE
from eia import Emissions
from plot_network_maps import (
    create_title,
    get_bus_scale,
    get_line_scale,
    plot_capacity_map,
)
from plot_statistics import (
    plot_california_emissions,
    plot_capacity_factor_heatmap,
    plot_curtailment_heatmap,
    plot_generator_data_panel,
    plot_region_lmps,
    plot_regional_emissions_bar,
    plot_fuel_costs,
)
from summary import get_node_emissions_timeseries

sns.set_theme("paper", style="whitegrid")

DPI=300
EIA_carrier_names = {
    "CCGT": "Natural gas",
    "OCGT": "Natural gas",
    "hydro": "Hydro",
    "oil": "Oil",
    "onwind": "Onshore wind",
    "solar": "Solar",
    "nuclear": "Nuclear",
    "coal": "Coal",
    "load": "Load shedding",
}
GE_carrier_names = {
    "NG": "Net Generation",
    "GAS": "Natural gas",
    "D": "Demand",
    "TI": "Total Interchange",
    "OIL": "Oil",
    "WAT": "Hydro",
    "WND": "Onshore wind",
    "SUN": "Solar",
    "NUC": "Nuclear",
    "COL": "Coal",
    "UNK": "Unknown",
    "OTH": "Other",
}


def add_missing_carriers(df1, df2):
    # Create new columns for historic for missing carriers in optimized
    for carrier in df1.columns:
        if carrier not in df2.columns:
            df2[carrier] = 0
    for carrier in df2.columns:
        if carrier not in df1.columns:
            df1[carrier] = 0
    return df1, df2


def plot_timeseries_comparison(
    historic: pd.DataFrame,
    optimized: pd.DataFrame,
    save_path: str,
    colors=None,
    title="Electricity Production by Carrier",
    **wildcards,
):
    """
    plots a stacked plot for seasonal production for snapshots: January 2 - December 30 (inclusive)
    """
    historic, optimized = add_missing_carriers(historic, optimized)

    kwargs = dict(color=colors, ylabel="Production [GW]", xlabel="", linewidth=0)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9))

    optimized_resampled = optimized.resample("1D").mean()
    optimized_resampled.plot.area(
        ax=axes[0],
        **kwargs,
        legend=False,
        title="Optimized",
    )
    order = optimized_resampled.columns

    historic_resampled = historic.resample("1D").mean()[order]
    historic_resampled.plot.area(
        ax=axes[1],
        **kwargs,
        legend=False,
        title="Historic",
    )

    diff = (optimized - historic).fillna(0).resample("1D").mean()
    diff.clip(lower=0).plot.area(
        ax=axes[2],
        title=r"$\Delta$ (Optimized - Historic)",
        legend=False,
        **kwargs,
    )
    diff.clip(upper=0).plot.area(ax=axes[2], **kwargs, legend=False)

    lower_lim = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0], axes[2].get_ylim()[0])
    upper_lim = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1], axes[2].get_ylim()[1])
    axes[0].set_ylim(bottom=lower_lim, top=upper_lim)
    axes[1].set_ylim(bottom=lower_lim, top=upper_lim)

    # Set the limits for the difference subplot
    diff_lim_upper = diff.clip(lower=0).sum(axis=1).max()
    diff_lim_lower = diff.clip(upper=0).sum(axis=1).min()
    axes[2].set_ylim(
        bottom=min(lower_lim, diff_lim_lower),
        top=max(upper_lim, diff_lim_upper),
    )

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(
        h[::-1],
        l[::-1],
        loc="lower right",
        bbox_to_anchor=(1, 0),
        ncol=1,
        frameon=True,
        labelspacing=0.1,
    )
    plt.suptitle(create_title(title, **wildcards))
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI)
    plt.close()


def plot_bar_carrier_production(
    historic: pd.DataFrame,
    optimized: pd.DataFrame,
    save_path: str,
    **wildcards,
):
    # plot by carrier
    data = pd.concat([historic, optimized], keys=["Historic", "Optimized"], axis=1)
    data.columns.names = ["Kind", "Carrier"]
    fig, ax = plt.subplots(figsize=(6, 6))
    df = data.T.groupby(level=["Kind", "Carrier"]).sum().T.sum().unstack().T
    df = df / 1e3  # convert to TWh
    df = df.drop("Unknown", axis=0)
    df.plot.barh(ax=ax, xlabel="Electricity Production [TWh]", ylabel="")
    ax.set_title(create_title("Electricity Production by Carriers", **wildcards))
    ax.grid(axis="y")
    fig.savefig(save_path, dpi =DPI)


def create_optimized_by_carrier(n, order, region_buses=None):
    """
    Create a DataFrame from the model output/optimized.
    """
    if region_buses is not None:
        gen_p = n.generators_t["p"].loc[
            :,
            n.generators.bus.isin(region_buses),
        ]
        # bus0 flow (pos if branch is withdrawing from region 0)
        # Pos = exports from region 0
        # Neg = imports to region 0
        interface_lines_b0 = n.lines[
            (n.lines.bus0.isin(region_buses) & ~n.lines.bus1.isin(region_buses))
        ]
        interface_links_b0 = n.links[
            (n.links.bus0.isin(region_buses) & ~n.links.bus1.isin(region_buses))
        ]

        # bus1 branch flow (pos if branch is withdrawing from region 1)
        # Pos = imports to region 0
        # Neg = exports from region 0
        interface_lines_b1 = n.lines[
            (n.lines.bus1.isin(region_buses) & ~n.lines.bus0.isin(region_buses))
        ]
        interface_links_b1 = n.links[
            (n.links.bus1.isin(region_buses) & ~n.links.bus0.isin(region_buses))
        ]

        # imports positive, exports negative
        flows = n.lines_t.p1.loc[:, interface_lines_b0.index].sum(axis=1)
        flows += n.lines_t.p0.loc[:, interface_lines_b1.index].sum(axis=1)
        flows += n.links_t.p1.loc[:, interface_links_b0.index].sum(axis=1)
        flows += n.links_t.p0.loc[:, interface_links_b1.index].sum(axis=1)
        imports = flows.apply(lambda x: x if x > 0 else 0)
        exports = flows.apply(lambda x: x if x < 0 else 0)

    else:
        gen_p = n.generators_t["p"]
        imports = None

    optimized = gen_p.T.groupby(by=n.generators["carrier"]).sum().T

    # Combine other carriers into "carrier"
    other_carriers = optimized.columns.difference(EIA_carrier_names.keys())
    # other_carriers = other_carriers.drop("Natural gas")
    optimized["Other"] = optimized[other_carriers].sum(axis=1)
    optimized = optimized.drop(columns=other_carriers)

    # Combine CCGT and OCGT
    if "OCGT" in optimized.columns and "CCGT" in optimized.columns:
        optimized["Natural gas"] = optimized.pop("CCGT") + optimized.pop("OCGT")
    elif "OCGT" in optimized.columns:
        optimized["Natural gas"] = optimized.pop("OCGT")
    elif "CCGT" in optimized.columns:
        optimized["Natural gas"] = optimized.pop("CCGT")

    # adding imports/export to df after cleaning up carriers
    if imports is not None:
        optimized["imports"] = imports
        optimized["exports"] = exports

    optimized = optimized.rename(columns=EIA_carrier_names)
    optimized.loc[
        :,
        optimized.columns.str.contains("Load"),
    ] /= 1e3  # correct load shedding units
    optimized.index = optimized.index.get_level_values(1)
    return optimized / 1e3


def create_historic_region_data(
    n: pypsa.Network,
    historic_all_ba: pd.DataFrame,
    ge_interchange: pd.DataFrame,
    region: str,
    emissions: bool = False,
):
    region_mapper = {
        "reeds_zone": {
            "BPAT": ["PGE", "BPAT", "GRID", "CHPD", "DOPD", "PSEI", "AVRN", "TPWR"],
            "AZPS": ["AZPS", "SRP", "HGMA", "DEAA"],
            "WALC": ["WALC", "GRIF"],
            "NWMT": ["NWMT", "GWA", "WWA"],
            "CISO": ["CISO", "BANC", "IID", "LDWP", "TIDC"],
            "SEC": ["SEC", "GVL", "TEC", "JEA", "FMPP", "FPC"],
            "FPL": ["FPL", "HST"],
        },
        "balancing_area": {
            "Arizona": ["AZPS", "SRP"],
            "WALC": ["WALC", "GRIF"],
            "NWMT": ["NWMT", "GWA", "WWA"],
        },
    }

    aggregation_zone = snakemake.config["clustering"]["cluster_network"][
        "aggregation_zones"
    ]
    regions = (
        region_mapper[aggregation_zone][region]
        if region in region_mapper[aggregation_zone]
        else [region]
    )

    historic_region = historic_all_ba.loc[regions].groupby(level=1).sum()

    if not emissions:
        index_split = ge_interchange.index.get_level_values(0).str.split("-")
        index_split = pd.DataFrame(
            [x for x in ge_interchange.index.get_level_values(0).str.split("-")],
        )
        from_region = index_split[0].isin(regions)
        to_region = index_split[1].isin(regions)
        selected_transfers = ge_interchange[
            ge_interchange.index[from_region & ~to_region]
        ]
        selected_transfers = selected_transfers.groupby(level=1).sum()

        historic_region["imports"] = selected_transfers.clip(upper=0) * -1
        historic_region["exports"] = selected_transfers.clip(lower=0) * -1

    return historic_region


def plot_regional_comparisons(
    n: pypsa.Network,
    historic_all_ba: pd.DataFrame,
    ge_interchange: pd.DataFrame,
    colors=None,
    order=None,
    **wildcards,
):
    """
    Plot regional comparison of results.
    """
    Path.mkdir(
        Path(snakemake.output[0]).parents[0] / "regional_timeseries",
        exist_ok=True,
    )
    buses = n.buses.copy()

    if (
        snakemake.config["clustering"]["cluster_network"]["aggregation_zones"]
        == "reeds_zone"
    ):
        regions = n.buses.reeds_ba.unique()
        regions = list(OrderedDict.fromkeys(regions))
        buses["region"] = buses.reeds_ba
    else:  # For Balancing Authority Aggregation
        regions = n.buses.country.unique()
        regions_clean = [ba.split("-")[0] for ba in regions]
        regions = list(OrderedDict.fromkeys(regions_clean))
        buses["region"] = [ba.split("-")[0] for ba in buses.country]

    historic_all_ba["imports"] = historic_all_ba["Total Interchange"].clip(upper=0) * -1
    historic_all_ba["exports"] = historic_all_ba["Total Interchange"].clip(lower=0)
    historic_all_ba = historic_all_ba.drop(columns=["Total Interchange"])

    diff = pd.DataFrame()
    # regions = [ba for ba in regions if ba in ["CISO"]] # Run to only produce ciso
    for region in regions:
        if region == "ERCO" and snakemake.wildcards.interconnect == "eastern":
            continue
        if region == "SWPP" and snakemake.wildcards.interconnect == "texas":
            continue
        region_buses = buses.query(f"region == '{region}'").index

        historic_region = create_historic_region_data(
            n,
            historic_all_ba,
            ge_interchange,
            region,
        )

        order = historic_region.columns
        optimized_region = create_optimized_by_carrier(
            n,
            order,
            region_buses,
        )

        # Create new columns for historic for missing carriers in optimized
        historic_region, optimized_region = add_missing_carriers(
            historic_region,
            optimized_region,
        )

        # Plot Timeseries Comparison
        plot_timeseries_comparison(
            historic=historic_region,
            optimized=optimized_region,
            save_path=Path(snakemake.output[0]).parents[0]
            / "regional_timeseries"
            / f"{region}_seasonal_stacked_plot.png",
            colors=colors,
            title=f"{region} Electricity Production by Carrier",
            **snakemake.wildcards,
        )
        # Calculate Production Deviation by percentage
        total_region = historic_region.sum().sum()
        diff[region] = (
            (optimized_region.sum() - historic_region.sum()) / total_region * 1e2
        )

    # Plot Bar Production Differences of Regions
    fig, ax = plt.subplots(figsize=(10, 6))
    diff.T.plot(kind="barh", stacked=True, ax=ax, color=colors)
    ax.set_xlabel("Production Deviation [% of Total]")
    ax.set_ylabel("Region")
    ax.set_title(
        create_title("Generation Deviation by Region and Carrier", **wildcards),
    )
    plt.legend(title="Carrier", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    fig.savefig(
        Path(snakemake.output[0]).parents[0] / "production_deviation_by_region.png", dpi =DPI
    )


def get_regions(n):
    regions = n.buses.country.unique()
    regions_clean = [ba.split("0")[0] for ba in regions]
    regions_clean = [ba.split("-")[0] for ba in regions_clean]
    regions = list(OrderedDict.fromkeys(regions_clean))
    return regions


def plot_load_shedding_map(
    n: pypsa.Network,
    save: str,
    regions: gpd.GeoDataFrame,
    **wildcards,
):

    load_curtailment = n.generators_t.p.filter(regex="^(.*load).*$")
    load_curtailment_sum = load_curtailment.sum() / 1e3  # convert to MW

    # split the generator name into a multi index where the first level is the bus and the second level is the carrier name
    multi_index = load_curtailment_sum.index.str.rsplit(" ", n=1, expand=True)
    multi_index.rename({0: "bus", 1: "carrier"}, inplace=True)
    load_curtailment_sum.index = multi_index
    bus_values = load_curtailment_sum

    bus_values = bus_values[bus_values.index.get_level_values(1).isin(n.carriers.index)]
    line_values = n.lines.s_nom

    # plot data
    title = create_title("Load Shedding", **wildcards)
    interconnect = wildcards.get("interconnect", None)
    bus_scale = get_bus_scale(interconnect) if interconnect else 1
    line_scale = get_line_scale(interconnect) if interconnect else 1

    fig, _ = plot_capacity_map(
        n=n,
        bus_values=bus_values,
        line_values=line_values,
        link_values=n.links.p_nom.replace(to_replace={pd.NA: 0}),
        regions=regions,
        line_scale=line_scale,
        bus_scale=bus_scale,
        title=title,
    )
    fig.savefig(save, dpi =DPI)


def plot_line_loading_map(
    n: pypsa.Network,
    save: str,
    regions: gpd.GeoDataFrame,
    **wildcards,
):
    gen = (
        n.generators.assign(g=n.generators_t.p.mean())
        .groupby(["bus", "carrier"])
        .g.sum()
    )

    line_values = 50

    # plot data
    title = create_title("Line Loading", **wildcards)
    interconnect = wildcards.get("interconnect", None)
    bus_scale = get_bus_scale(interconnect) if interconnect else 1
    line_scale = get_line_scale(interconnect) if interconnect else 1

    line_loading = n.lines_t.p0.abs().mean() / n.lines.s_nom / n.lines.s_max_pu * 100
    link_loading = n.links_t.p0.abs().mean() / n.links.p_nom / n.links.p_max_pu * 100
    norm = plt.Normalize(vmin=0, vmax=100)

    fig, _ = plot_capacity_map(
        n=n,
        bus_values=gen / 5e3,
        line_values=line_values,
        link_values=n.links.p_nom.replace(to_replace={pd.NA: 0}),
        regions=regions,
        flow="mean",
        line_scale=line_scale,
        bus_scale=bus_scale,
        line_colors=line_loading,
        link_colors=link_loading,
        line_cmap="plasma",
        line_norm=norm,
        title=title,
    )

    # plt.colorbar(
    #     plt.cm.ScalarMappable(cmap="plasma", norm=norm),
    #     label="Relative line loading [%]",
    #     shrink=0.6,
    #     ax=_,
    # )

    fig.savefig(save, dpi =DPI)


def plot_generator_cost_stack(
    n: pypsa.Network,
    save: str,
    **wildcards,
):
    marginal_costs = n.get_switchable_as_dense("Generator", "marginal_cost")
    marginal_costs = marginal_costs.mean().rename("marginal_cost")
    marginal_costs.loc[marginal_costs < 0.1] = 0.5
    marginal_costs = pd.DataFrame(marginal_costs)
    marginal_costs["p_nom"] = marginal_costs.index.map(n.generators.p_nom)
    marginal_costs["carrier"] = marginal_costs.index.map(n.generators.carrier)
    df = marginal_costs[marginal_costs.index.map(n.generators.carrier) != "load"]

    # Sort by marginal cost
    df_sorted = df.sort_values(by="marginal_cost")

    # Generate plot
    fig, ax = plt.subplots()

    # Variables for plotting
    cumulative_capacity = np.cumsum(df_sorted["p_nom"]) - df_sorted["p_nom"]
    marginal_costs = df_sorted["marginal_cost"]
    capacities = df_sorted["p_nom"]

    colors = n.carriers.color.to_dict()
    # Create stack plot
    for i in range(len(df_sorted)):
        ax.barh(
            y=0,
            width=capacities.iloc[i],
            left=cumulative_capacity.iloc[i],
            height=marginal_costs.iloc[i],
            align="edge",
            linewidth=0,
            color=colors[df_sorted["carrier"].iloc[i]],
        )

    fig.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=colors[carrier], label=carrier)
            for carrier in colors
        ],
        loc="upper left",
        bbox_to_anchor=(0.12, 0.875),
        title="Carrier",
    )

    ax.set_xlabel("Capacity [MW]")
    ax.set_ylabel("Marginal Cost [USD/MWh]")
    ax.set_title(create_title("Average Generator Merit Order Curve", **wildcards))
    fig.savefig(save, dpi =DPI)


def plot_state_emissions_historical_bar(
    n: pypsa.Network,
    ge_emissions: pd.DataFrame,
    save: str,
    snapshots: pd.date_range,
    eia_api: str,
    **wildcards,
) -> None:
    """
    Compares regional annual emissions to the year.
    """

    year = snapshots[0].year

    sectors = wildcards["sector"].split("-")
    historical_emissions = []

    if snakemake.params.eia_api:
        if "T" in sectors:
            historical_emissions.append(
                Emissions("transport", year, eia_api).get_data(),
            )
        if "I" in sectors:
            historical_emissions.append(
                Emissions("industrial", year, eia_api).get_data(),
            )
        if "H" in sectors:
            historical_emissions.append(
                Emissions("commercial", year, eia_api).get_data(),
            )
            historical_emissions.append(
                Emissions("residential", year, eia_api).get_data(),
            )
        historical_emissions.append(Emissions("power", year, eia_api).get_data())

        historical_emissions = pd.concat(historical_emissions)
        historical = (
            historical_emissions.reset_index()[["value", "state"]]
            .set_index("state")
            .rename(columns={"value": "Historical"})
        )

    optimized = pd.DataFrame(
        get_node_emissions_timeseries(n).T.groupby(n.buses.country).sum().T.sum() / 1e6,
        columns=["Optimized"],
    )

    region_mapper = (
        n.buses[["country", "reeds_state"]]
        .drop_duplicates()
        .set_index("country")["reeds_state"]
        .to_dict()
    )
    optimized["region"] = optimized.index.map(region_mapper)
    optimized = optimized.groupby("region").sum()
    CODE_2_STATE = {v: k for k, v in STATE_2_CODE.items()}
    optimized.index = optimized.index.map(CODE_2_STATE)
    optimized.index.name = "state"

    optimized.sort_index(inplace=True)

    historical = historical.loc[optimized.index]
    final = optimized.join(historical).reset_index()

    final = pd.melt(final, id_vars=["state"], value_vars=["Optimized", "Historical"])
    final["value"] = final.value.astype("float")

    # final = final[~final["state"].str.contains("Texas")]

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.barplot(
        data=final,
        y="state",
        x="value",
        hue="variable",
        orient="horizontal",
        ax=ax,
    )
    ax.set_title(create_title("CO2 Emissions by Region", **wildcards))
    ax.set_xlabel("CO2 Emissions [MMtCO2]")
    ax.set_ylabel("")
    fig.savefig(save, dpi =DPI)


def plot_ba_emissions_historical_bar(
    n: pypsa.Network,
    ge_emissions: pd.DataFrame,
    save: str,
    snapshots: pd.date_range,
    eia_api: str,
    **wildcards,
) -> None:
    """
    Compares regional annual emissions to the year.
    """

    year = snapshots[0].year

    sectors = wildcards["sector"].split("-")
    historical_emissions = []

    historical = pd.Series()
    for region in ge_emissions.index.get_level_values(0).unique():
        region_em = create_historic_region_data(
            n,
            ge_emissions,
            None,
            region,
            emissions=True,
        ).sum(axis=0)["Net Generation"]
        region_em = region_em.sum() / 1e9
        historical.loc[region] = region_em
    historical.name = "Historical"
    historical = historical[historical.round(3) > 0]

    optimized = pd.DataFrame(
        get_node_emissions_timeseries(n).T.groupby(n.buses.country).sum().T.sum() / 1e6,
        columns=["Optimized"],
    )

    if (
        snakemake.config["clustering"]["cluster_network"]["aggregation_zones"]
        == "balancing_area"
    ):
        optimized.loc["CISO"] = optimized.loc[
            ["CISO-PGAE", "CISO-SCE", "CISO-SDGE", "CISO-VEA"]
        ].sum()
        optimized.drop(
            index=["CISO-PGAE", "CISO-SCE", "CISO-SDGE", "CISO-VEA"],
            inplace=True,
        )
    elif (
        snakemake.config["clustering"]["cluster_network"]["aggregation_zones"]
        == "reeds_zone"
    ):
        region_mapper = (
            n.buses[["country", "reeds_ba"]]
            .drop_duplicates()
            .set_index("country")["reeds_ba"]
            .to_dict()
        )
        optimized["region"] = optimized.index.map(region_mapper)
        optimized = optimized.groupby("region").sum()
        optimized.index.name = "country"

    optimized.sort_index(inplace=True)

    historical = historical.loc[optimized.index]
    final = optimized.join(historical).reset_index()

    final = pd.melt(final, id_vars=["country"], value_vars=["Optimized", "Historical"])
    final["value"] = final.value.astype("float")

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.barplot(
        data=final,
        y="country",
        x="value",
        hue="variable",
        orient="horizontal",
        ax=ax,
    )
    ax.set_title(create_title("CO2 Emissions by Region", **wildcards))
    ax.set_xlabel("CO2 Emissions [MMtCO2]")
    ax.set_ylabel("")
    fig.savefig(save, dpi =DPI)

def get_state_generation_mix(n: pypsa.Network, var='p'):
    gens = n.generators.copy()
    gens['state'] = gens.bus.map(n.buses.reeds_state)
    gens['state_carrier'] = gens['state'] + "_" + gens['carrier']
    #Group by state and carrier
    generation = n.generators_t[var].copy()
    generation = generation.T.groupby(gens['state_carrier']).sum().T
    generation.index = generation.index.droplevel(1)
    generation = generation.groupby('period').sum().T
    generation = generation / 1e3 # convert to GWh
    generation = generation.reset_index()
    generation.columns = ['state_carrier', 'generation']
    generation['state'] = generation['state_carrier'].str.split('_').str[0]
    generation['carrier'] = generation['state_carrier'].str.split('_').str[1:].str.join('_')
    generation_pivot = generation.pivot(index='state', columns='carrier', values='generation')
    if 'load' in generation_pivot.columns:
        generation_pivot.load = generation_pivot.load.mul(1e-3)
    return generation_pivot

def get_state_loads(n: pypsa.Network):
    loads = n.loads_t.p
    n.loads['state'] = n.loads.bus.map(n.buses.reeds_state)
    loads = loads.T.groupby(n.loads.state).sum().T
    loads = loads / 1e3 # convert to GW

def plot_state_generation_mix(
    n: pypsa.Network,
    save: str,
    **wildcards,
):
    """Creates a stacked bar chart for each state's generation mix"""
    generation_pivot = get_state_generation_mix(n)

    # Create Stacked Bar Plot for each State's Generation Mix
    colors = n.carriers.color.to_dict()
    fig, ax = plt.subplots(figsize=(10, 8))
    generation_pivot.plot(kind='bar', stacked=True, ax=ax, color=colors)
    ax.set_title(create_title("State Generation Mix", **wildcards))
    ax.set_xlabel("State")
    ax.set_ylabel("Generation Mix [GWh]")
    fig.savefig(save, dpi =DPI)

def plot_state_generation_capacities(
    n: pypsa.Network,
    save: str,
    **wildcards,
):
    """Creates a stacked bar chart for each state's generation mix"""
    n.generators['state'] = n.generators.bus.map(n.buses.reeds_state)
    n.generators['state_carrier'] = n.generators['state'] + "_" + n.generators['carrier']

    #Group by state and carrier
    generation = n.generators.groupby('state_carrier').p_nom.sum()
    generation = generation / 1e3 # convert to GW
    generation = generation.reset_index()
    generation.columns = ['state_carrier', 'capacity']
    generation['state'] = generation['state_carrier'].str.split('_').str[0]
    generation['carrier'] = generation['state_carrier'].str.split('_').str[1:].str.join('_')
    generation_pivot = generation.pivot(index='state', columns='carrier', values='capacity')
    generation_pivot.drop(columns=['load'], inplace=True)

    # Create Stacked Bar Plot for each State's Generation Mix
    colors = n.carriers.color.to_dict()
    fig, ax = plt.subplots(figsize=(10, 8))
    generation_pivot.plot(kind='bar', stacked=True, ax=ax, color=colors)
    ax.set_title(create_title("State Generation Capacities ", **wildcards))
    ax.set_xlabel("State")
    ax.set_ylabel("Generation Capacity [GW]")
    fig.savefig(save, dpi =DPI)

def plot_lmp_distribution_comparison(
    n: pypsa.Network,
    lmps_true: pd.DataFrame,
    save: str,
    **wildcards,
):
    lmps = n.buses_t.marginal_price.copy()
    ISOs = ['CISO', 'MISO', 'ERCO', 'ISNE', 'NYIS', 'PJM', 'SWPP']
    iso_buses = n.buses[n.buses.reeds_ba.isin(ISOs)]
    lmps_iso = lmps.loc[:, iso_buses.index]
    lmps_iso.index = lmps_iso.index.get_level_values(1)

    df_long = pd.melt(
        lmps_iso.reset_index(),
        id_vars=["timestep"],
        var_name="bus",
        value_name="lmp",
    )
    df_long["season"] = df_long["timestep"].dt.quarter
    df_long["hour"] = df_long["timestep"].dt.hour
    # df_long.drop(columns="timestep", inplace=True)
    df_long["region"] = df_long.bus.map(n.buses.reeds_ba)
    df_long["source"] = 'simulated'

    df_true = df_long.copy()
    df_true = df_true.region.isin(df_long.region.unique())
    df_true['source'] = 'historical'
    df_plot = pd.concat([df_long, df_true])

    sns.boxplot(
        df_plot,
        x="lmp",
        y="region",
        hue='source',
        width=0.5,
        fliersize=0.5,
        linewidth=1,
    )
    sns.despine(offset=10, trim=True)

    plt.title(create_title("LMPs by Region", **wildcards))
    plt.xlabel("LMP [$/MWh]")
    plt.ylabel("Region")
    plt.tight_layout()
    plt.savefig(save, dpi =DPI)

    return None

def main(snakemake):
    configure_logging(snakemake)
    n = pypsa.Network(snakemake.input.network)
    snapshots = n.snapshots.get_level_values(1)

    onshore_regions = gpd.read_file(snakemake.input.regions_onshore)
    offshore_regions = gpd.read_file(snakemake.input.regions_offshore)

    buses = get_regions(n)

    # Load Grid Emissions Electricity Data
    ge_all = pd.read_csv(snakemake.input.ge_all).drop(columns=["Unnamed: 0"])
    ge_all.period = pd.to_datetime(ge_all.period)
    ge_all = ge_all.loc[ge_all.period.isin(snapshots), :]
    ge_all = ge_all.rename(columns=lambda x: x[2:] if x.startswith("E_") else x)

    ge_all.set_index("period", inplace=True)
    ge_all.columns = pd.MultiIndex.from_tuples(
        ge_all.columns.str.split("_", expand=True).tolist(),
    )
    ge_all = ge_all.stack(level=0).swaplevel().sort_index(level=0)
    ge_all.columns = ge_all.columns.map(GE_carrier_names).fillna("Interchange")

    ge_all["interconnect"] = (
        ge_all.index.get_level_values(0).map(EIA_BA_2_REGION).map(EIA_930_REGION_MAPPER)
    )
    ge_interchange = ge_all.loc[ge_all.interconnect.isna(), "Interchange"] / 1e3
    ge_all = ge_all.loc[~ge_all.interconnect.isna()]

    ge_all = (
        ge_all.loc[ge_all.interconnect == snakemake.wildcards.interconnect].drop(
            columns="interconnect",
        )
        / 1e3
    )
    ge_all.loc["SRP", "Nuclear"] = 0  # Fix for double reported Palo Verde
    ge_interconnect = (
        ge_all.groupby("period")
        .sum()
        .drop(columns=["Demand", "Net Generation", "Total Interchange", "Interchange"])
    )
    order = ge_all.columns

    # Load GridEmissions CO2 Data
    ge_co2 = pd.read_csv(snakemake.input.ge_co2).drop(columns=["Unnamed: 0"])
    ge_co2.period = pd.to_datetime(ge_co2.period)
    ge_co2 = ge_co2.loc[ge_co2.period.isin(snapshots), :]
    ge_co2 = ge_co2.rename(columns=lambda x: x[4:] if x.startswith("CO2_") else x)
    ge_co2.set_index("period", inplace=True)
    ge_co2.columns = pd.MultiIndex.from_tuples(
        ge_co2.columns.str.split("_", expand=True).tolist(),
    )
    ge_co2 = ge_co2.stack(level=0).swaplevel().sort_index(level=0)
    ge_co2.columns = ge_co2.columns.map(GE_carrier_names).fillna("Interchange")

    # Create Optimized DataFrame
    optimized = create_optimized_by_carrier(n, order)

    # Create Colors for Plotting
    colors = n.carriers.rename(EIA_carrier_names).color.to_dict()
    colors["Other"] = "#ba91b1"
    colors["Unknown"] = "#bd71aa"
    colors["imports"] = "#7d1caf"
    colors["exports"] = "#d624d9"

    snapshots = get_snapshots(snakemake.params.snapshots)

    plot_lmp_distribution_comparison(
        n,
        None,
        snakemake.output["val_lmp_comparison.pdf"],
        **snakemake.wildcards,
    )

    plot_generator_data_panel(
        n,
        snakemake.output["val_generator_data_panel.pdf"],
        **snakemake.wildcards,
    )

    plot_generator_cost_stack(
        n,
        snakemake.output["val_generator_stack.pdf"],
        **snakemake.wildcards,
    )


    plot_fuel_costs(
        n,
        snakemake.output["val_fuel_costs.pdf"],
        **snakemake.wildcards,
    )

    plot_line_loading_map(
        n,
        snakemake.output["val_map_line_loading.pdf"],
        onshore_regions,
        **snakemake.wildcards,
    )

    plot_state_generation_mix(
        n,
        snakemake.output["val_mix_state_generation.pdf"],
        **snakemake.wildcards,
    )

    plot_state_generation_capacities(
        n,
        snakemake.output["val_cap_state_generation.pdf"],
        **snakemake.wildcards,
    )

    plot_state_emissions_historical_bar(
        n,
        ge_co2,
        snakemake.output["val_bar_state_emissions.pdf"],
        snapshots,
        snakemake.params.eia_api,
        **snakemake.wildcards,
    )


    # Regional Comparisons
    plot_regional_comparisons(
        n,
        ge_all.drop(columns=["Demand", "Net Generation", "Interchange"]),
        ge_interchange,
        colors=colors,
        **snakemake.wildcards,
    )

    plot_ba_emissions_historical_bar(
        n,
        ge_co2,
        snakemake.output["val_bar_regional_emissions.pdf"],
        snapshots,
        snakemake.params.eia_api,
        **snakemake.wildcards,
    )

    # Bar Production
    plot_bar_carrier_production(
        ge_interconnect,
        optimized,
        save_path=snakemake.output["carrier_production_bar.pdf"],
        **snakemake.wildcards,
    )

    # Time Series
    plot_timeseries_comparison(
        ge_interconnect,
        optimized,
        save_path=snakemake.output["seasonal_stacked_plot.pdf"],
        colors=colors,
        title="Electricity Production by Carrier",
        **snakemake.wildcards,
    )

    # Box Plot
    plot_region_lmps(
        n,
        snakemake.output["val_box_region_lmps.pdf"],
        **snakemake.wildcards,
    )

    plot_capacity_factor_heatmap(
        n,
        snakemake.output["val_heatmap_capacity_factor.pdf"],
        **snakemake.wildcards,
    )



    plot_load_shedding_map(
        n,
        snakemake.output["val_map_load_shedding.pdf"],
        onshore_regions,
        **snakemake.wildcards,
    )



    n.statistics().to_csv(snakemake.output["val_statistics"])
    # plot_curtailment_heatmap(
    #     n,
    #     snakemake.output["val_heatmap_curtailment.pdf"],
    #     **snakemake.wildcards,
    # )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_validation_figures",
            interconnect="texas",
            clusters=50,
            ll="v1.0",
            opts="Ep",
            sector="E",
        )
    main(snakemake)
