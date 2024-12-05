import logging
import warnings
from functools import reduce

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as po
import pypsa
import seaborn as sns
from _helpers import (
    calculate_annuity,
    configure_logging,
    is_transport_model,
    update_p_nom_max,
)
from add_electricity import update_transmission_costs
from constants import *
from pypsa.clustering.spatial import (
    busmap_by_greedy_modularity,
    busmap_by_hac,
    busmap_by_kmeans,
    get_clustering_from_busmap,
)

warnings.filterwarnings(action="ignore", category=UserWarning)

idx = pd.IndexSlice

logger = logging.getLogger(__name__)


def normed(x):
    return (x / x.sum()).fillna(0.0)


def weighting_for_country(n, x):
    conv_carriers = {"nuclear", "OCGT", "CCGT", "PHS", "hydro", "coal", "biomass"}
    generators = n.generators
    generators["carrier_base"] = generators.carrier.str.split().str[0]
    gen = generators.loc[generators.carrier_base.isin(conv_carriers)].groupby(
        "bus",
    ).p_nom.sum().reindex(
        n.buses.index,
        fill_value=0.0,
    ) + n.storage_units.loc[n.storage_units.carrier.isin(conv_carriers)].groupby(
        "bus",
    ).p_nom.sum().reindex(
        n.buses.index,
        fill_value=0.0,
    )
    load = n.loads_t.p_set.mean().groupby(n.loads.bus).sum()

    b_i = x.index
    g = normed(gen.reindex(b_i, fill_value=0))
    l = normed(load.reindex(b_i, fill_value=0))

    w = g + l
    return (w * (100.0 / w.max())).clip(lower=1.0).astype(int)


def get_feature_for_hac(n, buses_i=None, feature=None):
    if buses_i is None:
        buses_i = n.buses.index

    if feature is None:
        feature = "solar+onwind-time"

    carriers = feature.split("-")[0].split("+")
    if "offwind" in carriers:
        carriers.remove("offwind")
        carriers = np.append(
            carriers,
            n.generators.carrier.filter(like="offwind").unique(),
        )

    if feature.split("-")[1] == "cap":
        feature_data = pd.DataFrame(index=buses_i, columns=carriers)
        for carrier in carriers:
            gen_i = n.generators.query("carrier == @carrier").index
            attach = n.generators_t.p_max_pu[gen_i].mean().rename(index=n.generators.loc[gen_i].bus)
            feature_data[carrier] = attach

    if feature.split("-")[1] == "time":
        feature_data = pd.DataFrame(columns=buses_i)
        for carrier in carriers:
            gen_i = n.generators.query("carrier == @carrier").index
            attach = n.generators_t.p_max_pu[gen_i].rename(
                columns=n.generators.loc[gen_i].bus,
            )
            feature_data = pd.concat([feature_data, attach], axis=0)[buses_i]

        feature_data = feature_data.T
        # timestamp raises error in sklearn >= v1.2:
        feature_data.columns = feature_data.columns.astype(str)

    feature_data = feature_data.fillna(0)

    return feature_data


def distribute_clusters(n, n_clusters, focus_weights=None, solver_name="cbc"):
    """
    Determine the number of clusters per country.
    """
    L = (
        n.loads_t.p_set.mean()
        .groupby(n.loads.bus)
        .sum()
        .groupby([n.buses.country, n.buses.sub_network])
        .sum()
        .pipe(normed)
    )

    N = n.buses.groupby(["country", "sub_network"]).size()

    assert (
        n_clusters >= len(N) and n_clusters <= N.sum()
    ), f"Number of clusters must be {len(N)} <= n_clusters <= {N.sum()} for this selection of countries."

    if focus_weights is not None:
        total_focus = sum(list(focus_weights.values()))

        assert total_focus <= 1.0, "The sum of focus weights must be less than or equal to 1."

        for country, weight in focus_weights.items():
            L[country] = weight / len(L[country])

        remainder = [c not in focus_weights.keys() for c in L.index.get_level_values("country")]
        L[remainder] = L.loc[remainder].pipe(normed) * (1 - total_focus)

        logger.warning("Using custom focus weights for determining number of clusters.")

    assert np.isclose(
        L.sum(),
        1.0,
        rtol=1e-3,
    ), f"Country weights L must sum up to 1.0 when distributing clusters. Is {L.sum()}."

    m = po.ConcreteModel()

    def n_bounds(model, *n_id):
        return (1, N[n_id])

    m.n = po.Var(list(L.index), bounds=n_bounds, domain=po.Integers)
    m.tot = po.Constraint(expr=(po.summation(m.n) == n_clusters))
    m.objective = po.Objective(
        expr=sum((m.n[i] - L.loc[i] * n_clusters) ** 2 for i in L.index),
        sense=po.minimize,
    )

    if solver_name == "highs":
        solver_name = "ipopt"

    opt = po.SolverFactory(solver_name)
    if not opt.has_capability("quadratic_objective"):
        logger.warning(
            f"The configured solver `{solver_name}` does not support quadratic objectives. Falling back to `ipopt`.",
        )
        opt = po.SolverFactory("ipopt")

    results = opt.solve(m)
    assert results["Solver"][0]["Status"] == "ok", f"Solver returned non-optimally: {results}"

    return pd.Series(m.n.get_values(), index=L.index).round().astype(int)


def busmap_for_n_clusters(
    n,
    n_clusters,
    solver_name,
    focus_weights=None,
    algorithm="kmeans",
    feature=None,
    **algorithm_kwds,
):
    if algorithm == "kmeans":
        algorithm_kwds.setdefault("n_init", 1000)
        algorithm_kwds.setdefault("max_iter", 30000)
        algorithm_kwds.setdefault("tol", 1e-6)
        algorithm_kwds.setdefault("random_state", 0)

    def fix_country_assignment_for_hac(n):
        from scipy.sparse import csgraph

        # overwrite country of nodes that are disconnected from their country-topology
        for country in n.buses.country.unique():
            m = n[n.buses.country == country].copy()

            _, labels = csgraph.connected_components(
                m.adjacency_matrix(),
                directed=False,
            )

            component = pd.Series(labels, index=m.buses.index)
            component_sizes = component.value_counts()

            if len(component_sizes) > 1:
                disconnected_bus = component[component == component_sizes.index[-1]].index[0]

                neighbor_bus = n.lines.query(
                    "bus0 == @disconnected_bus or bus1 == @disconnected_bus",
                ).iloc[
                    0
                ][["bus0", "bus1"]]
                new_country = list(
                    set(n.buses.loc[neighbor_bus].country) - {country},
                )[0]

                logger.info(
                    f"overwriting country `{country}` of bus `{disconnected_bus}` "
                    f"to new country `{new_country}`, because it is disconnected "
                    "from its initial inter-country transmission grid.",
                )
                n.buses.at[disconnected_bus, "country"] = new_country
        return n

    if algorithm == "hac":
        feature = get_feature_for_hac(n, buses_i=n.buses.index, feature=feature)
        n = fix_country_assignment_for_hac(n)

    if (algorithm != "hac") and (feature is not None):
        logger.warning(
            f"Keyword argument feature is only valid for algorithm `hac`. "
            f"Given feature `{feature}` will be ignored.",
        )

    n.determine_network_topology()

    n_clusters = distribute_clusters(
        n,
        n_clusters,
        focus_weights=focus_weights,
        solver_name=solver_name,
    )

    def busmap_for_country(x):
        prefix = x.name[0] + x.name[1] + " "
        logger.debug(f"Determining busmap for country {prefix[:-1]}")
        if len(x) == 1:
            return pd.Series(prefix + "0", index=x.index)
        weight = weighting_for_country(n, x)

        if algorithm == "kmeans":
            return prefix + busmap_by_kmeans(
                n,
                weight,
                n_clusters[x.name],
                buses_i=x.index,
                **algorithm_kwds,
            )
        elif algorithm == "hac":
            return prefix + busmap_by_hac(
                n,
                n_clusters[x.name],
                buses_i=x.index,
                feature=feature.loc[x.index],
            )
        elif algorithm == "modularity":
            return prefix + busmap_by_greedy_modularity(
                n,
                n_clusters[x.name],
                buses_i=x.index,
            )
        else:
            raise ValueError(
                f"`algorithm` must be one of 'kmeans' or 'hac'. Is {algorithm}.",
            )

    return (
        n.buses.groupby(["country", "sub_network"], group_keys=False)
        .apply(busmap_for_country, include_groups=False)
        .squeeze()
        .rename("busmap")
    )


def clustering_for_n_clusters(
    n,
    n_clusters,
    custom_busmap=False,
    aggregate_carriers=None,
    line_length_factor=1.25,
    aggregation_strategies=dict(),
    solver_name="cbc",
    algorithm="hac",
    feature=None,
    focus_weights=None,
):
    if not isinstance(custom_busmap, pd.Series):
        busmap = busmap_for_n_clusters(
            n,
            n_clusters,
            solver_name,
            focus_weights,
            algorithm,
            feature,
        )
    else:
        busmap = custom_busmap

    line_strategies = aggregation_strategies.get("lines", dict())
    generator_strategies = aggregation_strategies.get("generators", dict())
    one_port_strategies = aggregation_strategies.get("one_ports", dict())

    clustering = get_clustering_from_busmap(
        n,
        busmap,
        aggregate_generators_weighted=True,
        aggregate_generators_carriers=aggregate_carriers,
        aggregate_one_ports=["Load", "StorageUnit"],
        line_length_factor=line_length_factor,
        line_strategies=line_strategies,
        generator_strategies=generator_strategies,
        one_port_strategies=one_port_strategies,
        scale_link_capital_costs=False,
    )

    return clustering


def add_itls(buses, itls, itl_cost, expansion=True):
    """
    Adds ITL limits to the network.

    Adds bi-directional links for all ITLS which are non-expandable.
    Adds a second link that is expandable with equal expansion in each
    direction.
    """
    if itl_cost is not None:
        itl_cost["interface"] = itl_cost.r + "||" + itl_cost.rr
        itl_cost = itl_cost[itl_cost.interface.isin(itls.interface)]
        itl_cost["USD2023perMW"] = itl_cost["USD2004perMW"] * (314.54 / 188.9)
        itl_cost["USD2023perMWyr"] = calculate_annuity(60, 0.025) * itl_cost["USD2023perMW"]
        itls = itls.merge(
            itl_cost[["interface", "length_miles", "USD2023perMWyr"]],
            on="interface",
            how="left",
        )
    else:
        itls["length_miles"] = 0
        itls["USD2023perMWyr"] = 0

    itls["p_min_pu_Rev"] = (-1 * (itls.mw_r0 / itls.mw_f0)).fillna(0)
    itls["efficiency"] = 1 - ((itls.length_miles / 100) * 0.01)

    # lines to add in reverse if forward direction is zero
    itls_rev = itls[itls.mw_f0 == 0].copy()
    itls_fwd = itls[itls.mw_f0 != 0]

    clustering.network.madd(
        "Link",
        names=itls_fwd.interface,  # itl name
        bus0=buses.loc[itls_fwd.r].index,
        bus1=buses.loc[itls_fwd.rr].index,
        p_nom=itls_fwd.mw_f0.values,
        p_nom_min=itls_fwd.mw_f0.values,
        p_max_pu=1.0,
        p_min_pu=itls_fwd.p_min_pu_Rev.values,
        length=0 if itl_cost is None else itls_fwd.length_miles.values,
        capital_cost=0 if itl_cost is None else itls_fwd.USD2023perMWyr.values,
        p_nom_extendable=False,
        efficiency=1 if itl_cost is None else itls_fwd.efficiency.values,
        carrier="AC",
    )

    clustering.network.madd(
        "Link",
        names=itls_rev.interface,  # itl name
        suffix="rev",
        bus0=buses.loc[itls_rev.r].index,
        bus1=buses.loc[itls_rev.rr].index,
        p_nom=itls_rev.mw_r0.values,
        p_nom_min=itls_rev.mw_r0.values,
        p_max_pu=0,
        p_min_pu=-1,
        length=0 if itl_cost is None else itls_rev.length_miles.values,
        capital_cost=0 if itl_cost is None else itls_rev.USD2023perMWyr.values,
        p_nom_extendable=False,
        efficiency=1 if itl_cost is None else itls_rev.efficiency.values,
        carrier="AC",
    )

    if not expansion:
        return

    # for tracking expansion of Zonal Links
    clustering.network.madd(
        "Link",
        names=itls.interface,  # itl name
        suffix="exp",
        bus0=buses.loc[itls.r].index,
        bus1=buses.loc[itls.rr].index,
        p_nom=0,
        p_nom_min=0,
        p_max_pu=1,
        p_min_pu=-1,
        length=0 if itl_cost is None else itls.length_miles.values,
        capital_cost=0 if itl_cost is None else itls.USD2023perMWyr.values,
        p_nom_extendable=False,
        efficiency=1 if itl_cost is None else itls.efficiency.values,
        carrier="AC_exp",
    )


def convert_to_transport(
    clustering,
    itl_fn,
    itl_cost_fn,
    itl_agg_fn,
    itl_agg_costs_fn,
    topological_boundaries,
    topology_aggregation,
):
    """
    Replaces all Lines according to Links with the transfer capacity specified
    by the ITLs.
    """
    clustering.network.mremove("Line", clustering.network.lines.index)
    buses = clustering.network.buses.copy()

    itls = pd.read_csv(itl_fn)
    itl_cost = pd.read_csv(itl_cost_fn)
    itls.columns = itls.columns.str.lower()
    itls_filt = itls[
        itls.r.isin(clustering.network.buses[f"{topological_boundaries}"])
        & itls.rr.isin(clustering.network.buses[f"{topological_boundaries}"])
    ]
    add_itls(buses, itls_filt, itl_cost)

    if itl_agg_fn:
        topology_aggregation_key = list(topology_aggregation.keys())[0]
        itl_agg = pd.read_csv(itl_agg_fn)
        itl_agg.columns = itl_agg.columns.str.lower()
        itl_agg = itl_agg.rename(columns={"transgrp": "r", "transgrpp": "rr"})
        itl_agg = itl_agg[
            itl_agg.r.isin(clustering.network.buses["country"]) | itl_agg.rr.isin(clustering.network.buses["country"])
        ]
        aggregated_buses = agg_busmap.rename(index=lambda x: x.strip(" 0"))
        non_agg_buses = buses[~buses.index.isin(agg_busmap.values)]
        non_agg_buses = non_agg_buses[
            non_agg_buses[topology_aggregation_key].isin(itl_agg.r)
            | non_agg_buses[topology_aggregation_key].isin(itl_agg.rr)
        ]
        virtual_buses = non_agg_buses.groupby(topology_aggregation_key)[non_agg_buses.columns].min()
        existing_x = non_agg_buses.groupby(topology_aggregation_key).x.mean()
        existing_y = non_agg_buses.groupby(topology_aggregation_key).y.mean()
        agg_x = buses.loc[agg_busmap.unique()].x.mean()
        agg_y = buses.loc[agg_busmap.unique()].y.mean()
        virtual_buses["x"] = 0.3 * agg_x + 0.7 * existing_x
        virtual_buses["y"] = 0.3 * agg_y + 0.7 * existing_y

        if not virtual_buses.empty:
            clustering.network.madd(
                "Bus",
                virtual_buses[topology_aggregation_key],
                country=virtual_buses[topology_aggregation_key],
                reeds_zone="na" if topology_aggregation_key == "trans_grp" else virtual_buses["reeds_zone"],
                reeds_ba="na" if topology_aggregation_key == "trans_grp" else virtual_buses["reeds_ba"],
                interconnect=virtual_buses["interconnect"],
                nerc_reg=virtual_buses["nerc_reg"],
                trans_reg=virtual_buses["trans_reg"],
                trans_grp=virtual_buses["trans_grp"],
                reeds_state=virtual_buses["reeds_state"],
                x=virtual_buses.x,
                y=virtual_buses.y,
            )
        itl_agg = itl_agg[
            itl_agg.r.isin(clustering.network.buses["country"])
            & itl_agg.rr.isin(clustering.network.buses["country"])
            & (itl_agg.r.isin(agg_busmap.values) | itl_agg.rr.isin(agg_busmap.values))
        ]

        buses = clustering.network.buses.copy()
        # itls from county to respective virtual bus
        itls_to_virtual = itls[
            (itls.r.isin(aggregated_buses.index) | itls.rr.isin(aggregated_buses.index))
            & ~(itls.r.isin(aggregated_buses.index) & itls.rr.isin(aggregated_buses.index))
        ]
        itls_to_virtual = itls_to_virtual[itls_to_virtual.r.isin(buses.index) | itls_to_virtual.rr.isin(buses.index)]
        # Update the itls virtual such that the existing bus : virtual bus
        itls_to_virtual.loc[itls_to_virtual.r.isin(non_agg_buses.index), "rr"] = non_agg_buses.loc[
            itls_to_virtual.r[itls_to_virtual.r.isin(non_agg_buses.index)],
            topology_aggregation_key,
        ].values
        itls_to_virtual.loc[itls_to_virtual.rr.isin(non_agg_buses.index), "r"] = non_agg_buses.loc[
            itls_to_virtual.rr[itls_to_virtual.rr.isin(non_agg_buses.index)],
            topology_aggregation_key,
        ].values

        itl_agg = pd.concat([itl_agg, itls_to_virtual])
        itl_agg_costs = None if itl_agg_costs_fn is None else pd.concat([itl_cost, pd.read_csv(itl_agg_costs_fn)])
        add_itls(buses, itl_agg, itl_agg_costs, expansion=False)
        itls = pd.concat([itls_filt, itl_agg])
    else:
        itls = itls_filt

    clustering.network.add("Carrier", "AC_exp", co2_emissions=0)
    logger.info("Replaced Lines with Links for zonal model configuration.")

    # Remove any disconnected buses
    unique_buses = buses.loc[itls.r].index.union(buses.loc[itls.rr].index).unique()
    disconnected_buses = clustering.network.buses.index[~clustering.network.buses.index.isin(unique_buses)]

    if len(disconnected_buses) > 0:
        logger.warning(
            f"Network configuration contains {len(disconnected_buses)} disconnected buses. ",
        )

    # Dissolve TX for particular zones according to default reeds configurations
    clustering.network.links.loc[
        clustering.network.links.bus0.isin(["p119"]) & clustering.network.links.bus1.isin(["p122"]),
        "p_nom",
    ] = 1e9
    clustering.network.links.loc[
        clustering.network.links.bus1.isin(["p119"]) & clustering.network.links.bus0.isin(["p122"]),
        "p_nom",
    ] = 1e9
    # Dissolve p124 and p99
    if "p124" in clustering.network.buses.index and "p99" in clustering.network.buses.index:
        clustering.network.add(
            "Link",
            "p124||p99",
            bus0="p124",
            bus1="p99",
            p_nom=1e9,
            p_nom_extendable=False,
            length=0,
            capital_cost=0,
            efficiency=1,
            carrier="AC",
        )

    return clustering


def cluster_regions(busmaps, input=None, output=None):
    busmap = reduce(lambda x, y: x.map(y), busmaps[1:], busmaps[0])

    for which in ("regions_onshore", "regions_offshore"):
        regions = gpd.read_file(getattr(input, which))
        regions = regions.reindex(columns=["name", "geometry"]).set_index("name")
        regions_c = regions.dissolve(busmap)
        regions_c.index.name = "name"
        regions_c = regions_c.reset_index()
        regions_c.to_file(getattr(output, which))


def plot_busmap_for_n_clusters(n, n_clusters, fn=None):
    busmap = busmap_for_n_clusters(n, n_clusters)
    cs = busmap.unique()
    cr = sns.color_palette("hls", len(cs))
    n.plot(bus_colors=busmap.map(dict(zip(cs, cr))))
    if fn is not None:
        plt.savefig(fn, bbox_inches="tight")
    del cs, cr


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "cluster_network",
            simpl="",
            clusters="7",
            interconnect="texas",
        )
    configure_logging(snakemake)

    params = snakemake.params
    solver_name = snakemake.config["solving"]["solver"]["name"]

    n = pypsa.Network(snakemake.input.network)

    n.set_investment_periods(
        periods=snakemake.params.planning_horizons,
    )

    topological_boundaries = params.topological_boundaries
    transport_model = is_transport_model(params.transmission_network)
    topology_aggregation = params.topology_aggregation

    exclude_carriers = params.cluster_network["exclude_carriers"]
    aggregate_carriers = set(n.generators.carrier) - set(exclude_carriers)
    conventional_carriers = set(params.conventional_carriers)
    non_aggregated_carriers = {}
    if snakemake.wildcards.clusters.endswith("m"):
        n_clusters = int(snakemake.wildcards.clusters[:-1])
        aggregate_carriers = set(params.conventional_carriers) & aggregate_carriers
        non_aggregated_carriers = set(n.generators.carrier) - aggregate_carriers
    elif snakemake.wildcards.clusters.endswith("c"):
        n_clusters = int(snakemake.wildcards.clusters[:-1])
        aggregate_carriers = aggregate_carriers - conventional_carriers
        non_aggregated_carriers = set(n.generators.carrier) - aggregate_carriers
    elif snakemake.wildcards.clusters == "all":
        n_clusters = len(n.buses)
    else:
        n_clusters = int(snakemake.wildcards.clusters)

    n.generators.loc[n.generators.carrier.isin(non_aggregated_carriers), "land_region"] = n.generators.loc[
        n.generators.carrier.isin(non_aggregated_carriers),
        "bus",
    ]

    if params.cluster_network.get("consider_efficiency_classes", False):
        carriers = []
        for c in aggregate_carriers:
            gens = n.generators.query("carrier == @c")
            low = gens.efficiency.quantile(0.10)
            high = gens.efficiency.quantile(0.90)
            if low >= high or low.round(2) == high.round(2):
                carriers += [c]
            else:
                labels = ["low", "medium", "high"]
                suffix = pd.cut(
                    gens.efficiency,
                    bins=[0, low, high, 1],
                    labels=labels,
                ).astype(str)
                carriers += [f"{c} {label} efficiency" for label in labels]
                n.generators.update(
                    {"carrier": gens.carrier + " " + suffix + " efficiency"},
                )
        aggregate_carriers = carriers

    if (n_clusters == len(n.buses)) and not transport_model:
        # Fast-path if no clustering is necessary
        busmap = n.buses.index.to_series()
        linemap = n.lines.index.to_series()
        clustering = pypsa.clustering.spatial.Clustering(
            n,
            busmap,
            linemap,
        )
    else:
        costs = pd.read_csv(snakemake.input.tech_costs)
        costs = costs.pivot(index="pypsa-name", columns="parameter", values="value")
        hvac_overhead_cost = costs.at["HVAC overhead", "annualized_capex_per_mw_km"]

        custom_busmap = params.custom_busmap
        if custom_busmap:
            custom_busmap = pd.read_csv(
                snakemake.input.custom_busmap,
                index_col=0,
                squeeze=True,
            )
            custom_busmap.index = custom_busmap.index.astype(str)
            logger.info(f"Imported custom busmap from {snakemake.input.custom_busmap}")

        if transport_model:
            # Prepare data for transport model
            itl_agg_fn = None
            itl_agg_costs_fn = None
            logger.info(
                f"Aggregating to transport model with {topological_boundaries} zones.",
            )
            match topological_boundaries:
                case "reeds_zone":
                    custom_busmap = n.buses.reeds_zone.copy()
                    itl_fn = snakemake.input.itl_reeds_zone
                    itl_cost_fn = snakemake.input.itl_costs_reeds_zone
                case "county":
                    custom_busmap = n.buses.county.copy()
                    itl_fn = snakemake.input.itl_county
                    itl_cost_fn = snakemake.input.itl_costs_county
                case _:
                    raise ValueError(f"Unknown aggregation zone {topological_boundaries}")

            if topology_aggregation:
                assert isinstance(topology_aggregation, dict), "topology_aggregation must be a dictionary."
                assert len(topology_aggregation) == 1, "topology_aggregation must contain exactly one key."

                # Extract the single key and value
                key, value = next(iter(topology_aggregation.items()))
                agg_busmap = n.buses[key][n.buses[key].isin(value)]
                logger.info(f"Aggregating {agg_busmap.unique()} {key} zones.")
                custom_busmap.update(agg_busmap.copy())
                n.buses.loc[agg_busmap.index, "country"] = agg_busmap
                if key == "trans_grp":
                    n.buses.loc[agg_busmap.index, "reeds_zone"] = "na"
                    n.buses.loc[agg_busmap.index, "reeds_ba"] = "na"
                    n.buses.loc[agg_busmap.index, "reeds_state"] = "na"
                    n.buses.loc[agg_busmap.index, "county"] = "na"
                itl_agg_fn = snakemake.input[f"itl_{key}"]
                itl_agg_costs_fn = snakemake.input.get(f"itl_costs_{key}", None)

            logger.info("Using Transport Model.")
            nodes_req = custom_busmap.unique()

            assert n_clusters == len(
                nodes_req,
            ), f"Number of clusters must be {len(nodes_req)} for current configuration."

            n.buses.interconnect = n.buses.nerc_reg.map(REEDS_NERC_INTERCONNECT_MAPPER)
            n.lines.drop(columns=["interconnect"], inplace=True)

        clustering = clustering_for_n_clusters(
            n,
            n_clusters,
            custom_busmap,
            aggregate_carriers,
            params.length_factor,
            params.aggregation_strategies,
            solver_name,
            params.cluster_network["algorithm"],
            params.cluster_network["feature"],
            params.focus_weights,
        )

        if transport_model:
            # Use Reeds Data
            clustering = convert_to_transport(
                clustering,
                itl_fn,
                itl_cost_fn,
                itl_agg_fn,
                itl_agg_costs_fn,
                topological_boundaries,
                topology_aggregation,
            )
        else:
            # Use standard transmission cost estimates
            update_transmission_costs(clustering.network, costs)

    update_p_nom_max(clustering.network)
    clustering.network.generators.land_region = clustering.network.generators.land_region.fillna(
        clustering.network.generators.bus,
    )

    if params.cluster_network.get("consider_efficiency_classes"):
        labels = [f" {label} efficiency" for label in ["low", "medium", "high"]]
        nc = clustering.network
        nc.generators["carrier"] = nc.generators.carrier.replace(labels, "", regex=True)

    clustering.network.meta = dict(
        snakemake.config,
        **dict(wildcards=dict(snakemake.wildcards)),
    )

    clustering.network.set_investment_periods(
        periods=snakemake.params.planning_horizons,
    )

    clustering.network.export_to_netcdf(snakemake.output.network)

    for attr in (
        "busmap",
        "linemap",
    ):  # also available: linemap_positive, linemap_negative
        getattr(clustering, attr).to_csv(snakemake.output[attr])

    cluster_regions((clustering.busmap,), snakemake.input, snakemake.output)
    n.consistency_check()
    logger.info(f"Saved clustered network to {snakemake.output.network}")
