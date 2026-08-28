# BY PyPSA-USA Authors
"""Optional k-means clustering of the substation-level network to ``{simpl}`` clusters.

Second stage of the topology-aggregation pipeline. Consumes the substation-level
topology produced by :mod:`aggregate_to_substations` (buses, lines, links only —
no loads, generators, or storage attached) and either:

- When ``{simpl}`` is empty: pass-through the substation-level network and regions.
- When ``{simpl}=N``: k-means cluster to N buses using ``n.buses.load_weight`` as the static
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
from _helpers import configure_logging, log_network_schema, plot_geojson, update_p_nom_max
from cluster_network import cluster_regions, clustering_for_n_clusters

logger = logging.getLogger(__name__)


def resolve_simpl_mode(value: str) -> str:
    """Map a `{simpl}` wildcard value to its dispatch branch.

    Returns one of:
      - "identity": pass-through (empty string)
      - "county":   fast-path using county FIPS as busmap
      - "kmeans":   numeric value -> N-cluster k-means

    Raises ValueError for anything else, listing the recognized values.
    """
    if value == "":
        return "identity"
    if value == "county":
        return "county"
    if value.isdigit():
        return "kmeans"
    raise ValueError(
        f"Unknown simpl wildcard value {value!r}. Recognized values are: "
        f'"" (identity pass-through), "county" (county FIPS fast-path), '
        f"or a positive integer (k-means).",
    )


def build_county_busmap(n: "pypsa.Network") -> "pd.Series":
    """Construct a sub_id -> '<reeds_zone>_<county_fips>' busmap.

    Used by the simpl='county' fast-path. The county field is the 5-digit FIPS
    GEOID assigned in build_base_network from county_shapes.GEOID, which is
    nationally unique; the reeds_zone prefix is added for human readability
    when inspecting clustered networks.
    """
    if "county" not in n.buses.columns or n.buses.county.isna().any():
        raise ValueError(
            "simpl='county' requires every substation bus to carry a non-null "
            "'county' attribute. This attribute is dropped by "
            "aggregate_to_substations when topological_boundaries='state'. "
            "Set model_topology.topological_boundaries to 'county' (or "
            "'reeds_zone') in your config, or use a numeric {simpl} wildcard.",
        )
    return (n.buses.reeds_zone.astype(str) + "_" + n.buses.county.astype(str)).rename("busmap")


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
    schema_entry = log_network_schema(n, stage="entry")

    if snakemake.wildcards.simpl:
        configured_strategy = params.simplify_network.get(
            "weighting_strategy",
            "population",
        )
        if configured_strategy != "population":
            logger.info(
                "cluster_simpl runs before loads/generators are attached; using "
                "weighting_strategy='population' (n.buses.load_weight) regardless of "
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
        n = clustering.n

        cluster_regions((busmap,), snakemake.input, snakemake.output)
    else:
        for which in ("regions_onshore", "regions_offshore"):
            regions = gpd.read_file(getattr(snakemake.input, which))
            # Substation-level region names carry a float ".0" suffix (e.g.
            # "35827.0") while the pass-through network keeps bare bus IDs
            # ("35827"). Normalize so regions_s{simpl} names match
            # n.buses.index — the invariant downstream consumers (e.g. the
            # godeeep path of build_renewable_profiles) rely on. The kmeans
            # branch gets the same normalization inside cluster_regions().
            try:
                regions["name"] = pd.to_numeric(regions["name"]).astype(int).astype(str)
            except (ValueError, TypeError):
                pass  # non-numeric names are already canonical
            out_path = getattr(snakemake.output, which)
            regions.to_file(out_path)
            plot_geojson(out_path)
        busmap = pd.Series(n.buses.index, index=n.buses.index, name="cluster_bus")

    busmap.index = busmap.index.astype(str)
    busmap = busmap.astype(str)
    busmap.index.name = "sub_id"
    busmap.name = "cluster_bus"
    busmap.to_csv(snakemake.output.busmap)

    update_p_nom_max(n)

    log_network_schema(n, stage="exit", baseline=schema_entry)
    n.export_to_netcdf(snakemake.output.network)
