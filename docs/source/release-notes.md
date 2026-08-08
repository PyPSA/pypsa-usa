(release-notes)=
# Release Notes

## Upcoming release (v1 line, in development)

The v1 development line restructures the workflow around **early spatial
aggregation** ("simplify-early") and modernizes the repository layout. If you are
migrating a workflow or custom configuration from an earlier checkout, these are the
changes you will notice:

### Workflow restructuring

- **`simplify_network` was split and the DAG reordered.** The old single rule became
  `aggregate_to_substations` (topology-only reduction to substations) followed by
  `cluster_simpl` (kmeans/modularity clustering to `{simpl}` zones). Renewable
  profiles, demand construction, and `add_electricity` now run *after* clustering, at
  `{simpl}` resolution instead of nodal resolution. This cuts peak memory of the
  heavy rules several-fold and speeds up the data pipeline substantially. See
  {doc}`model-workflow`.
- **HAC clustering was removed.** Supported clustering algorithms are `kmeans` and
  `modularity`. The config section `clustering: simplify_network:` keeps its
  historical name but feeds the `cluster_simpl` rule.
- **EGS supply curves are remapped through the cluster busmap.** A new
  `aggregate_egs` rule converts substation-keyed NREL EGS supply curves to cluster
  buses (capacity-weighted means for intensive quantities, sums for extensive ones).

### Repository layout

- **`resources/` is organized category-first**: `networks/`, `busmaps/`, `profiles/`,
  `geospatial/`, `costs/`, `prices/`, `demand/`, `powerplants/`, ... — instead of
  per-interconnect flat folders. File names are unchanged.
- Unused config keys and never-read rule parameters were removed, and
  `workflow/config/` is kept in sync with the canonical templates in
  `workflow/repo_data/config/`.

### Correctness fixes validated by an equivalence harness

The refactor was validated by a pipeline-equivalence harness comparing the
restructured DAG against the pre-refactor baseline on a California test system. The
harness caught and the line fixes four real bugs, all of which also affected the old
pipeline or would have silently shifted results:

- Demand disaggregation conservation error (state totals now conserved exactly).
- Hydro plants dropped during attachment under some clusterings.
- Transmission `length_factor` applied twice in capital costs.
- Empty renewable profiles attached as zero-output generators.

The full engineering change-log, including per-change expected effects on model
results, is maintained in the repository at
[`docs/CHANGELOG-v1-epic.md`](https://github.com/PyPSA/pypsa-usa/blob/master/docs/CHANGELOG-v1-epic.md).

### Documentation

- New Model Description section ({doc}`model-workflow`, {doc}`model-components`,
  {doc}`model-constraints`, {doc}`model-network-schema`).
- Configuration-reference pages are regenerated from the live config templates and
  guarded by tests; both workflow DAG diagrams were regenerated from the current
  rule graph.

## Earlier releases

For changes prior to the v1 line, see the
[GitHub releases page](https://github.com/PyPSA/pypsa-usa/releases).
