# BY PyPSA-USA Authors
"""Optional k-means clustering of the substation-level network to ``{simpl}`` clusters.

Second stage of the topology-aggregation pipeline. Consumes the substation-level
network produced by :mod:`aggregate_to_substations` and either:

- When ``{simpl}`` is empty: pass-through the substation-level network and regions.
- When ``{simpl}=N``: k-means cluster to N buses using static demand / generation
  weighting, then dissolve the substation Voronoi cells accordingly.

This script preserves the behavior of the former second half of
``simplify_network.py`` for the k-means / modularity algorithms. HAC is no
longer supported; ``cluster_network.busmap_for_n_clusters`` will raise if
selected.
"""

import logging

import geopandas as gpd
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
        n.set_investment_periods(periods=snakemake.params.planning_horizons)

        # Drop timeseries before clustering — preserves the historical
        # simplify_network behavior; k-means and modularity use static weights
        # only.
        n.loads_t.p = n.loads_t.p.iloc[:, 0:0]
        n.loads_t.q = n.loads_t.q.iloc[:, 0:0]
        for attr in [
            "p",
            "q",
            "state_of_charge",
            "mu_state_of_charge_set",
            "mu_energy_balance",
            "mu_lower",
            "mu_upper",
            "spill",
            "p_dispatch",
            "p_store",
        ]:
            n.storage_units_t[attr] = n.storage_units_t[attr].iloc[:, 0:0]

        # Patch for pypsa io clustering bug with build_years for new gens
        n.generators.build_year += 0.001

        clustering = clustering_for_n_clusters(
            n,
            int(snakemake.wildcards.simpl),
            focus_weights=params.focus_weights,
            solver_name=solver_name,
            algorithm=params.simplify_network["algorithm"],
            aggregation_strategies=params.aggregation_strategies,
            weighting_strategy=params.simplify_network.get("weighting_strategy", None),
        )
        n = clustering.network

        cluster_regions((clustering.busmap,), snakemake.input, snakemake.output)
    else:
        for which in ("regions_onshore", "regions_offshore"):
            regions = gpd.read_file(getattr(snakemake.input, which))
            regions.to_file(getattr(snakemake.output, which))

    update_p_nom_max(n)

    n.export_to_netcdf(snakemake.output.network)
