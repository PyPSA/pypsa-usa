# BY PyPSA-USA Authors
"""Optional k-means clustering of the substation-level network to ``{simpl}`` clusters.

Second stage of the topology-aggregation pipeline. Consumes the substation-level
topology produced by :mod:`aggregate_to_substations` (buses, lines, links only —
no loads, generators, or storage attached) and either:

- When ``{simpl}`` is empty: pass-through the substation-level network and regions.
- When ``{simpl}=N``: k-means cluster to N buses using ``n.buses.Pd`` as the static
  per-bus weight (i.e. ``weighting_strategy=population``), then dissolve the
  substation Voronoi cells accordingly.

Because the input network has no time-varying data attached, this script always
uses the ``population`` weighting branch regardless of the configured
``simplify_network.weighting_strategy``. Demand-capacity weighting requires loads
to be attached, which only happens at the downstream ``cluster_network`` step.

HAC clustering is no longer supported; ``cluster_network.busmap_for_n_clusters``
will raise if selected.
"""

import logging

import geopandas as gpd
import pandas as pd
import pypsa
from _helpers import configure_logging, update_p_nom_max
from cluster_network import cluster_regions, clustering_for_n_clusters

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "cluster_simpl",
            interconnect="texas",
            simpl="50",
        )
    configure_logging(snakemake)
    params = snakemake.params
    solver_name = snakemake.config["solving"]["solver"]["name"]

    n = pypsa.Network(snakemake.input.network)

    if snakemake.wildcards.simpl:
        configured_strategy = params.simplify_network.get(
            "weighting_strategy",
            "population",
        )
        if configured_strategy != "population":
            logger.info(
                "cluster_simpl runs before loads/generators are attached; using "
                "weighting_strategy='population' (n.buses.Pd) regardless of "
                "configured '%s'.",
                configured_strategy,
            )

        clustering = clustering_for_n_clusters(
            n,
            int(snakemake.wildcards.simpl),
            focus_weights=params.focus_weights,
            solver_name=solver_name,
            algorithm=params.simplify_network["algorithm"],
            aggregation_strategies=params.aggregation_strategies,
            weighting_strategy="population",
        )
        busmap = clustering.busmap
        n = clustering.network

        cluster_regions((busmap,), snakemake.input, snakemake.output)
    else:
        for which in ("regions_onshore", "regions_offshore"):
            regions = gpd.read_file(getattr(snakemake.input, which))
            regions.to_file(getattr(snakemake.output, which))
        busmap = pd.Series(n.buses.index, index=n.buses.index, name="cluster_bus")

    busmap.index = busmap.index.astype(str)
    busmap = busmap.astype(str)
    busmap.index.name = "sub_id"
    busmap.name = "cluster_bus"
    busmap.to_csv(snakemake.output.busmap)

    update_p_nom_max(n)

    n.export_to_netcdf(snakemake.output.network)
