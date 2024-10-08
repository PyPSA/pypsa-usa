"""
Creates networks clustered to ``{cluster}`` number of zones with aggregated
buses, generators and transmission corridors.

**Relevant Settings**

.. code:: yaml

    clustering:
      cluster_network:
      aggregation_strategies:

    focus_weights:

    solving:
        solver:
            name:

    lines:
        length_factor:

.. seealso::
    Documentation of the configuration file ``config/config.yaml`` at
    :ref:`renewable_cf`, :ref:`solving_cf`, :ref:`lines_cf`

**Inputs**

- ``resources/regions_onshore_elec_s{simpl}.geojson``: confer :ref:`simplify`
- ``resources/regions_offshore_elec_s{simpl}.geojson``: confer :ref:`simplify`
- ``resources/busmap_elec_s{simpl}.csv``: confer :ref:`simplify`
- ``networks/elec_s{simpl}.nc``: confer :ref:`simplify`
- ``data/custom_busmap_elec_s{simpl}_{clusters}.csv``: optional input

**Outputs**

- ``resources/regions_onshore_elec_s{simpl}_{clusters}.geojson``:
- ``resources/regions_offshore_elec_s{simpl}_{clusters}.geojson``:
- ``resources/busmap_elec_s{simpl}_{clusters}.csv``: Mapping of buses from ``networks/elec_s{simpl}.nc`` to ``networks/elec_s{simpl}_{clusters}.nc``;
- ``resources/linemap_elec_s{simpl}_{clusters}.csv``: Mapping of lines from ``networks/elec_s{simpl}.nc`` to ``networks/elec_s{simpl}_{clusters}.nc``;
- ``networks/elec_s{simpl}_{clusters}.nc``:

**Description**

.. note::

    **Why is clustering used both in** ``simplify_network`` **and** ``cluster_network`` **?**

        Consider for example a network ``networks/elec_s100_50.nc`` in which
        ``simplify_network`` clusters the network to 100 buses and in a second
        step ``cluster_network``` reduces it down to 50 buses.

        In preliminary tests, it turns out, that the principal effect of
        changing spatial resolution is actually only partially due to the
        transmission network. It is more important to differentiate between
        wind generators with higher capacity factors from those with lower
        capacity factors, i.e. to have a higher spatial resolution in the
        renewable generation than in the number of buses.

        The two-step clustering allows to study this effect by looking at
        networks like ``networks/elec_s100_50m.nc``. Note the additional
        ``m`` in the ``{cluster}`` wildcard. So in the example network
        there are still up to 100 different wind generators.

        In combination these two features allow you to study the spatial
        resolution of the transmission network separately from the
        spatial resolution of renewable generators.

    **Is it possible to run the model without the** ``simplify_network`` **rule?**

        No, the network clustering methods in the PyPSA module
        `pypsa.clustering.spatial <https://github.com/PyPSA/PyPSA/blob/master/pypsa/clustering/spatial.py>`_
        do not work reliably with multiple voltage levels and transformers.

.. tip::
    The rule :mod:`cluster_networks` runs
    for all ``scenario`` s in the configuration file
    the rule :mod:`cluster_network`.
"""

# SPDX-FileCopyrightText: : 2017-2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
# coding: utf-8
# ADAPTED FROM PyPSA-Eur for PyPSA-USA

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
from _helpers import calculate_annuity, configure_logging, update_p_nom_max
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
    ).p_nom.sum().reindex(n.buses.index, fill_value=0.0) + n.storage_units.loc[
        n.storage_units.carrier.isin(conv_carriers)
    ].groupby(
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
            attach = (
                n.generators_t.p_max_pu[gen_i]
                .mean()
                .rename(index=n.generators.loc[gen_i].bus)
            )
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

        assert (
            total_focus <= 1.0
        ), "The sum of focus weights must be less than or equal to 1."

        for country, weight in focus_weights.items():
            L[country] = weight / len(L[country])

        remainder = [
            c not in focus_weights.keys() for c in L.index.get_level_values("country")
        ]
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
    assert (
        results["Solver"][0]["Status"] == "ok"
    ), f"Solver returned non-optimally: {results}"

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
                disconnected_bus = component[
                    component == component_sizes.index[-1]
                ].index[0]

                neighbor_bus = n.lines.query(
                    "bus0 == @disconnected_bus or bus1 == @disconnected_bus",
                ).iloc[0][["bus0", "bus1"]]
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
        .apply(busmap_for_country)
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
    extended_link_costs=0,
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
    bus_strategies = aggregation_strategies.get("buses", dict())
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

    if not n.links.empty:
        nc = clustering.network
        nc.links["underwater_fraction"] = (
            n.links.eval("underwater_fraction * length").div(nc.links.length).dropna()
        )
        nc.links["capital_cost"] = nc.links["capital_cost"].add(
            (nc.links.length - n.links.length)
            .clip(lower=0)
            .mul(extended_link_costs)
            .dropna(),
            fill_value=0,
        )

    return clustering


def replace_lines_with_links(clustering, itl_fn):
    """
    Replaces all Lines according to Links with the transfer capacity specified
    by the ITLs.
    """
    lines = clustering.network.lines.copy()
    buses = clustering.network.buses.copy()

    itls = pd.read_csv(itl_fn)

    itls = itls[
        itls.r.isin(clustering.network.buses.reeds_zone)
        & itls.rr.isin(clustering.network.buses.reeds_zone)
    ]

    itl_cost = pd.read_csv(snakemake.input.itl_costs)
    itl_cost["interface"] = itl_cost.r + "||" + itl_cost.rr
    itl_cost = itl_cost[itl_cost.interface.isin(itls.interface)]
    itl_cost["USD2023perMW"] = itl_cost["USD2004perMW"] * (314.54 / 188.9)
    itl_cost["USD2023perMWyr"] = calculate_annuity(60, 0.025) * itl_cost["USD2023perMW"]
    itls = itls.merge(
        itl_cost[["interface", "length_miles", "USD2023perMWyr"]],
        on="interface",
        how="left",
    )

    itls["p_min_pu_Rev"] = (-1 * (itls.MW_r0 / itls.MW_f0)).fillna(0)

    # lines to add in reverse if forward direction is zero
    itls_rev = itls[itls.MW_f0 == 0].copy()
    itls_fwd = itls[itls.MW_f0 != 0]

    clustering.network.mremove("Line", clustering.network.lines.index)
    clustering.network.madd(
        "Link",
        names=itls_fwd.interface,  # itl name
        bus0=buses.loc[itls_fwd.r].index,
        bus1=buses.loc[itls_fwd.rr].index,
        p_nom=itls_fwd.MW_f0.values,
        p_nom_min=itls_fwd.MW_f0.values,
        p_max_pu=1.0,
        p_min_pu=itls_fwd.p_min_pu_Rev.values,
        length=itls_fwd.length_miles.values,
        capital_cost=itls_fwd.USD2023perMWyr.values,
        p_nom_extendable=False,
        carrier="AC_trans",
    )

    clustering.network.madd(
        "Link",
        names=itls_rev.interface,  # itl name
        suffix="rev",
        bus0=buses.loc[itls_rev.r].index,
        bus1=buses.loc[itls_rev.rr].index,
        p_nom=itls_rev.MW_r0.values,
        p_nom_min=itls_rev.MW_r0.values,
        p_max_pu=0,
        p_min_pu=-1,
        length=itls_rev.length_miles.values,
        capital_cost=itls_rev.USD2023perMWyr.values,
        p_nom_extendable=False,
        carrier="AC_trans",
    )

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
        length=itls.length_miles.values,
        capital_cost=itls.USD2023perMWyr.values,
        p_nom_extendable=False,
        carrier="DC",
    )

    logger.info(f"Replaced Lines with Links for zonal model configuration.")

    # Remove any disconnected buses
    unique_buses = buses.loc[itls.r].index.union(buses.loc[itls.rr].index).unique()
    disconnected_buses = clustering.network.buses.index[
        ~clustering.network.buses.index.isin(unique_buses)
    ]
    if len(disconnected_buses) > 0:
        logger.warning(
            f"Removed {len(disconnected_buses)} sub-network buses from the network.",
        )
        clustering.network.mremove("Bus", disconnected_buses)
        clustering.network.mremove(
            "Generator",
            clustering.network.generators.query("bus in @disconnected_buses").index,
        )
        clustering.network.mremove(
            "StorageUnit",
            clustering.network.storage_units.query("bus in @disconnected_buses").index,
        )
        clustering.network.mremove(
            "Store",
            clustering.network.stores.query("bus in @disconnected_buses").index,
        )
        clustering.network.mremove(
            "Load",
            clustering.network.loads.query("bus in @disconnected_buses").index,
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

    exclude_carriers = params.cluster_network["exclude_carriers"]
    aggregate_carriers = set(n.generators.carrier) - set(exclude_carriers)
    conventional_carriers = set(params.conventional_carriers)
    if snakemake.wildcards.clusters.endswith("m"):
        n_clusters = int(snakemake.wildcards.clusters[:-1])
        aggregate_carriers = set(params.conventional_carriers) & aggregate_carriers
    elif snakemake.wildcards.clusters.endswith("c"):
        n_clusters = int(snakemake.wildcards.clusters[:-1])
        aggregate_carriers = aggregate_carriers - conventional_carriers
    elif snakemake.wildcards.clusters == "all":
        n_clusters = len(n.buses)
    else:
        n_clusters = int(snakemake.wildcards.clusters)

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

    if n_clusters == len(n.buses):
        # Fast-path if no clustering is necessary
        busmap = n.buses.index.to_series()
        linemap = n.lines.index.to_series()
        clustering = pypsa.clustering.spatial.Clustering(
            n,
            busmap,
            linemap,
            linemap,
            pd.Series(dtype="O"),
        )
    else:
        Nyears = (
            n.snapshot_weightings.loc[n.investment_periods[0]].objective.sum() / 8760.0
        )

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

        if params.replace_lines_with_links:
            custom_busmap = n.buses.reeds_zone
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
            hvac_overhead_cost,
            params.focus_weights,
        )
        if params.replace_lines_with_links:
            clustering = replace_lines_with_links(
                clustering,
                snakemake.input.itls,
            )
            N = clustering.network.buses.reeds_zone.unique()
            assert n_clusters == len(
                N,
            ), f"Number of clusters must be {len(N)} to model as transport model."
        else:
            update_transmission_costs(clustering.network, costs)

    update_p_nom_max(clustering.network)

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
