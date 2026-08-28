# cluster_simpl county fast-path

**Status:** Design approved 2026-05-21. Implementation plan to follow.

## Motivation

`cluster_simpl` is the first stage of the simplify-early clustering pipeline. With a numeric `{simpl}` wildcard it runs k-means over the substation-level network. On full-interconnect inputs this dominates wall time of the topology-aggregation stack, even though the user often just wants the network rolled up to the county level — for which k-means is unnecessary, because every substation bus already carries a county FIPS code from `build_base_network` / `aggregate_to_substations`.

We add a non-numeric sentinel value `simpl="county"` that bypasses k-means entirely and uses the existing county FIPS as a direct busmap. This is a pure-speed change: no new science, no change in carrier-selectivity semantics.

## Scope

In scope:
- New recognized value `simpl="county"` for the `{simpl}` wildcard.
- Fast-path branch in `workflow/scripts/cluster_simpl.py`.
- A clear error when the input network doesn't carry a usable `county` column.
- Unit test covering the fast path.

Out of scope (deliberately, YAGNI):
- Fast paths for `simpl="state"` / `simpl="reeds_zone"` / `simpl="ba"`. Easy to add later; not requested.
- New config knobs. The fast path is selected purely via the wildcard.
- Any change to `cluster_network`, `add_electricity`, or downstream rules. They consume `{simpl}` opaquely and don't need to know whether the value was numeric or `"county"`.
- Any change to the existing `simpl=""` (identity) or `simpl=<N>` (k-means) branches.

## Wildcard

The current `Snakefile` `wildcard_constraints` entry is `simpl="[a-zA-Z0-9]*|all"`. This regex already matches `"county"`, so no regex change is required.

Branching in `cluster_simpl.py`:

| `wildcards.simpl` | Behavior |
|---|---|
| `""` | Identity pass-through (unchanged). |
| `"county"` | **New fast path** — county busmap, no k-means. |
| `<digits>` (e.g. `"50"`) | K-means via `clustering_for_n_clusters` (unchanged). |
| anything else | Raise `ValueError` with the list of recognized sentinels. |

The catch-all error is new — today the `else` branch silently treats unrecognized values as the identity pass-through, which is a footgun.

## Fast-path body

```python
if snakemake.wildcards.simpl == "county":
    if "county" not in n.buses.columns or n.buses.county.isna().any():
        raise ValueError(
            "simpl='county' requires every substation bus to carry a non-null "
            "'county' attribute. This is dropped by aggregate_to_substations "
            "when topological_boundaries='state'. Set topological_boundaries "
            "to 'county' (or 'reeds_zone') in model_topology, or use a numeric "
            "{simpl} wildcard."
        )

    busmap = n.buses.reeds_zone.astype(str) + "_" + n.buses.county.astype(str)

    clustering = get_clustering_from_busmap(
        n,
        busmap,
        aggregate_generators_weighted=True,
        aggregate_one_ports=["Load", "StorageUnit"],
        line_length_factor=1.25,
        bus_strategies={"Pd": "sum", "LAF_state": "sum"},
        line_strategies=params.aggregation_strategies.get("lines", {}),
        generator_strategies=params.aggregation_strategies.get("generators", {}),
        one_port_strategies=params.aggregation_strategies.get("one_ports", {}),
        scale_link_capital_costs=False,
    )
    n = clustering.network
    cluster_regions((clustering.busmap,), snakemake.input, snakemake.output)
    busmap = clustering.busmap
```

Then the existing tail of the script runs unchanged:

```python
busmap.index = busmap.index.astype(str)
busmap = busmap.astype(str)
busmap.index.name = "sub_id"
busmap.name = "cluster_bus"
busmap.to_csv(snakemake.output.busmap)
update_p_nom_max(n)
log_network_schema(n, stage="exit", baseline=schema_entry)
n.export_to_netcdf(snakemake.output.network)
```

`get_clustering_from_busmap` is already imported via `from cluster_network import cluster_regions, clustering_for_n_clusters`. We'll add a direct import of `get_clustering_from_busmap` from `pypsa.clustering.spatial`.

## Cluster bus IDs

Format: `<reeds_zone>_<county_FIPS>`, e.g. `p9_06001`.

- `county` is the 5-digit FIPS GEOID (`build_base_network.py:537-538` assigns it from `county_shapes.GEOID`), which is already nationally unique.
- `reeds_zone` is enforced one-per-county by the mode-lookup at `build_base_network.py:442-444`, so the prefix is well-defined.
- The prefix is for human readability when inspecting clustered networks; downstream code treats the bus index as opaque.

## Composition with `cluster_network`

The existing `{clusters}` wildcard remains the carrier-selectivity knob — no semantic change.

| `simpl=county, clusters=` | Behavior |
|---|---|
| `all` | Identity fast-path in `cluster_network` (line 897 already short-circuits when `n_clusters == len(n.buses)`). Bus topology stays one-per-county. Thermal plants attached at `add_electricity` keep one row per plant; no per-carrier aggregation. **This is the typical full-county run.** |
| `<num_counties>c` | Renewables aggregated per-bus-per-carrier (one combined wind generator per county, etc.). Conventionals retain one row per plant. **The typical "renewables-only roll-up" run.** |
| `<num_counties>` (no suffix) | Aggregate all carriers per-county-per-carrier (one combined CCGT per county, etc.). |
| `<M>` where `M < num_counties` | K-means in `cluster_network` rolls counties further down to M groups. |

Important: `clusters=all` does NOT merge thermal plants into one-per-county aggregates. Their `bus` was set to the county cluster bus back in `add_electricity`, so they're spatially at county resolution, but each plant keeps its identity. This is desirable for unit commitment and ramp-constrained modeling.

## File paths produced

Following the existing category-first resources layout, with `{simpl}` = `"county"`:

- `resources/<RDIR>/networks/<interconnect>/elec_scounty.nc`
- `resources/<RDIR>/busmaps/<interconnect>/busmap_scounty.csv`
- `resources/<RDIR>/geospatial/<interconnect>/regions_onshore_scounty.geojson`
- `resources/<RDIR>/geospatial/<interconnect>/regions_offshore_scounty.geojson`
- End-to-end: `…/elec_scounty_c{clusters}_ec_l{ll}_{opts}_{sector}.nc`

No rule changes needed; the existing path templates use `{simpl}` opaquely.

## Edge cases

1. **Wrong `topological_boundaries`**: `aggregate_to_substations.py:236-237` and the `state` branch of `cols2drop` drop the `county` column. The fast-path raises a clear error in this case (see the snippet above).
2. **Offshore content**: at this pipeline stage the network is the substation-level grid (onshore power-system substations only — offshore wind enters later via `build_renewable_profiles` / `add_electricity`). Offshore is represented only by `regions_offshore.geojson`, not by buses. The county FIPS field is therefore expected on every bus; the guard above is the right enforcement mechanism.
3. **EGS rule**: `aggregate_egs` consumes `busmap_s{simpl}.csv` and remaps NREL substation-keyed supply curves. The fast-path busmap is keyed on `sub_id → cluster_bus`, which is exactly the schema `aggregate_egs` expects. No change needed.
4. **Schema-logging**: keep the existing `log_network_schema(n, stage="entry")` / `stage="exit"` calls so the network-schema catalog initiative (see `2026-05-21-network-schema-tracking-design.md`) captures the new branch.

## Testing

Add to `workflow/scripts/test/test_cluster_simpl.py` (new file):

1. **Fast-path happy path**: build a small fixture network with 4 substation buses carrying `county` FIPS `"06001"`, `"06001"`, `"06037"`, `"06037"` and `reeds_zone` `"p9"`, `"p9"`, `"p10"`, `"p10"`. Run the fast path. Assert:
   - Output bus count = 2.
   - Bus IDs are `{"p9_06001", "p10_06037"}`.
   - Busmap CSV maps the 4 input sub_ids to the 2 county cluster IDs.
2. **Error path**: same fixture but drop the `county` column. Assert `ValueError` mentioning `topological_boundaries`.
3. **Unknown sentinel**: invoke with `wildcards.simpl="foo"`, assert it raises (the new catch-all branch).

These tests parallel the structure of existing tests in `workflow/scripts/test/` and use the same mock-snakemake conventions. They are not end-to-end; they're focused unit tests on the new branch.

## Migration / docs

- Add a short note to `workflow/config/config.default.yaml` near `clustering.simplify_network` explaining the new `simpl="county"` wildcard value.
- Update any existing docs that enumerate `{simpl}` values (search `docs/source/` for `simpl` references during implementation).
- No backwards compatibility concern: this only adds a new accepted wildcard value; all existing values keep their current behavior.

## Open implementation questions (not blocking)

- Whether to also recognize an alias `simpl="counties"` (or any spelling variant). I'd say no — pick one spelling and stick with it. This spec uses `"county"`.
- Whether the catch-all error for unrecognized non-numeric sentinels (e.g. `simpl="foo"`) belongs to this change or a separate hygiene PR. I'd include it here; it's three lines and removes a real footgun adjacent to the new branch.
