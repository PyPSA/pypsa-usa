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
  `cluster_resources` (kmeans/modularity clustering to `{simpl}` zones). Renewable
  profiles, demand construction, and `add_electricity` now run *after* clustering, at
  `{simpl}` resolution instead of nodal resolution. This cuts peak memory of the
  heavy rules several-fold and speeds up the data pipeline substantially. See
  {doc}`model-workflow`.
- **HAC clustering was removed.** Supported clustering algorithms are `kmeans` and
  `modularity`. The config section `clustering: simplify_network:` keeps its
  historical name but feeds the `cluster_resources` rule.
- **EGS supply curves are remapped through the cluster busmap.** A new
  `aggregate_egs` rule converts substation-keyed NREL EGS supply curves to cluster
  buses (capacity-weighted means for intensive quantities, sums for extensive ones).

### Repository layout

- **`resources/` is organized category-first**: `networks/`, `busmaps/`, `profiles/`,
  `geospatial/`, `costs/`, `prices/`, `demand/`, `powerplants/`, ... — instead of
  per-interconnect flat folders. File names are unchanged.
- Unused config keys and never-read rule parameters were removed. The canonical config
  templates live in `workflow/repo_data/config/` and the workflow loads its whole layered
  base from there; `workflow/config/` is untracked and holds only the per-user files
  (`config.api.yaml`, `config.slurm.yaml`, your own scenario configs) seeded by
  `init_pypsa_usa.sh`.
- **`config.cluster.yaml` was renamed `config.slurm.yaml`.** The old name collided with the
  `{clusters}` wildcard and the `clustering:` section while only ever describing the job
  scheduler. Its dead per-rule `{rule}: {walltime: ...}` blocks are gone; the single
  top-level `walltime:` block is the one the rules actually read. Rename your copy under
  `workflow/config/` to keep your account and partition settings.
- **Scenario configs are overlays, not forks.** `config.default.yaml` is itself a loaded
  layer, so a `--configfile` only needs the keys it changes, and every top-level key is now
  owned by exactly one layered file. A full copy still works — it simply overrides every key
  it repeats. See {doc}`config-configuration`.
- **The merged configuration is validated** against `workflow/schemas/config.schema.yaml` at
  parse time. A misspelled key inside a closed section (`electricity:`, `model_topology:`,
  `clustering:`, `solving:`, ...) now fails immediately instead of silently falling back to a
  default several rules later.
- **The EIA API key can be supplied as `$EIA_API_KEY`**, which takes precedence over
  `config/config.api.yaml` and keeps the key out of your files entirely.
- **`renewable_land_access` now defaults to `reference`.** The shipped templates previously
  paired `renewable.dataset: godeeep` with a null land-access setting, a combination
  `build_renewable_profiles` rejects — no shipped config could build renewable profiles from a
  clean checkout. The default is NREL's central land-access assumption; expect
  `retrieve_nrel_exclusion_artifact` jobs in every godeeep DAG. Null remains valid only with
  `dataset: atlite`.

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
[`docs/CHANGELOG-v1-epic.md`](https://github.com/PyPSA/pypsa-usa/blob/develop/docs/CHANGELOG-v1-epic.md).

### California / CPUC SERVM

- **New demand source `electricity: demand: profile: servm`** — CPUC SERVM 2026 IRP hourly
  load for the six California load regions (PGE, SCE, SDGE, IID, LADWP, NCNC), retrieved
  per forecast year from files.cpuc.ca.gov. Nine forecast years are published (2026, 2028,
  2030, 2032, 2035, 2037, 2040, 2042, 2045) and `planning_horizons` is restricted to them.
  `electricity: demand: scenario: servm_weather_years` picks one weather year out of the
  stacked 2000-2024 record. Only `Net Load` is dispatched; the full component split is
  written to a new component-resolved zonal artifact
  (`power_zonal_components_s{simpl}.parquet`), which is now produced for every demand
  profile. See {ref}`servm-demand`.
- **SERVM load-allocation weights** — a new `build_servm_load_weights` rule composes the
  base→substation→cluster busmaps into a fractional `(SERVM region, bus)` table, so a cluster
  that straddles two regions (Los Angeles County holds both LDWP and CISO-SCE buses) receives
  the sum of its share of each.
- **Interface transmission limits are live.** `model_topology: interface_transmission_limits`
  and `electricity: transmission_interface_limits` were previously dead keys. They now apply
  the RESOLVE interface table as a per-snapshot cap on the *aggregate* flow across each
  interface. The constraint scopes to the import/export links, so it is inert when trade is
  disabled; the resulting understatement for `region_2` entries inside the footprint (notably
  `p8` in California-only runs) is documented in {doc}`data-transmission`.
- **New maintained config `config.california.yaml`** — a runnable California-only model on
  SERVM demand with the CAISO interface caps and imports/exports enabled, at REeDS-zone
  resolution (`clusters: 4`) with a commented county-resolution alternative
  (`clusters: 58`, `simpl: county`).
- **Phase-2 hook `conventional: ambient_derate`** — reserved for CPUC SERVM unit-specific
  ambient-temperature derates. It is not implemented; enabling it raises `NotImplementedError`
  in `add_electricity`. When it lands it replaces the EIA-860 seasonal derate rather than
  stacking on it.

### Documentation

- New Model Description section ({doc}`model-workflow`, {doc}`model-components`,
  {doc}`model-constraints`, {doc}`model-network-schema`).
- Configuration-reference pages are regenerated from the live config templates and
  guarded by tests; both workflow DAG diagrams were regenerated from the current
  rule graph.

### Breaking changes since 2026-08-31

- **The GenX-style operational reserve constraint was removed** (`6eacc8cc`).
  `add_operational_reserve_margin` and its `Generator-p` reserve bound are gone from
  `opts/reserves.py`, along with the call in `solve_network`, the
  `electricity: operational_reserve` block in `config.default.yaml`, its schema entry and
  its config-table rows. The formulation was never verified against its implementation. No
  `{opts}` token was involved — it was triggered by
  `electricity: operational_reserve: activate: true` — so **a config that still carries an
  `electricity: operational_reserve:` block now fails schema validation at parse time**
  (`electricity:` is a closed section); delete the block. The energy reserve margin
  (`ERM` token, `add_ERM_constraints`) is unchanged and is the active resource-adequacy
  mechanism.

### Correctness fixes since 2026-08-31

- **Portfolio standards (`RPS`) worked at all only in the unit tests** (`7da43cb1`).
  `add_RPS_constraints` read `snakemake.params.planning_horizons`, which `rule
  solve_network` never declared, so the `RPS` token raised `AttributeError` in the
  workflow; the horizons now come from `n.investment_periods`, which is also correct under
  `foresight: myopic`.
- **ReEDS CES trajectories were silently dropped** (`bc2131d4`). `ces_fraction.csv` is
  wide (`st, 2010, ..., 2050`); after melting, its `planning_horizon` values were strings
  and were filtered out against integer horizons, so 0 of 359 CES rows survived and no
  ReEDS CES target was ever enforced. RPS rows, already long with integer years, were
  unaffected. Expect CES targets to start binding.
- **One empty RPS zone no longer cancels the remaining RPS constraints** (`6b1732ee`). A
  zone with no eligible generators returned from the whole function, dropping every
  remaining (zone, horizon, carrier) constraint; it now logs a warning and continues.
- **Land-use limits subtract existing and frozen capacity from the potential**
  (`46be26d7`, `43096a8a`). `p_nom_max` from `build_renewable_profiles` is a *gross*
  land potential, and nothing was subtracted from it. Under `foresight: myopic`,
  `freeze_prior_periods` makes each horizon's builds non-extendable, so those MW left the
  constraint's left-hand side while the right-hand side kept the full potential and the
  same land could be built on again every horizon. The right-hand side is now the group
  potential net of the non-extendable capacity of the same carrier and land region that is
  active in the period (peak across the modelled periods), clipped at zero, and exhausted
  groups are named in a warning. **This changes results for myopic multi-horizon runs**
  (less renewable headroom in later horizons); perfect-foresight runs whose
  `land_region` generators are all extendable are unchanged. See
  [Land-use limits](model-constraints.md#land-use-limits).
- **Vintaged bidirectional links are now coupled** (`e4305b15`). `add_itls` names
  future-horizon interface links `<interface>_fwd_<year>` / `<interface>_rev_<year>`, but
  the pairing regex was anchored on `_fwd`/`_rev` at the end of the name, so those links
  were made extendable without the equal-expansion constraint and a multi-horizon solve
  could build one direction only, at half the capital cost of a line. Pairing now happens
  within a vintage.
- **Heat-pump cooling links are back in the dispatch constraint** (`b6607b0d`,
  sector studies only). The selection looked for a `-cooling` suffix while
  `build_heat` names the links `<heating link>-cool`, so it matched nothing and the
  dispatch constraint collapsed into PyPSA's own capacity bound — heating and cooling could
  each run at full capacity in the same snapshot although they are one physical unit.
- **`electricity: transmission_interface_limits` is now actually read** (`4754bc5b`).
  `solve_network` and `solve_network_validation` hard-coded the interface-limits CSV path,
  so pointing the key at another file had no effect. The shipped default path is unchanged,
  so existing configs resolve the same file and results are unaffected.

### Trade with external regions

- **Generator-based imports and boundary-metered CPUC contracted resources**
  ([#810](https://github.com/PyPSA/pypsa-usa/pull/810)). A new
  `electricity: imports: representation` key selects how the world outside the footprint is
  represented. `store` (the default) is the previous behaviour: a bottomless `Store` on each
  external bus feeding priced links into the footprint. `generator` instead puts a priced
  import generator (carrier `unspecified_imports`, rated at the zone's inbound interface
  capacity) on the external bus and leaves the links into the footprint unpriced, and it
  attaches California's CPUC-contracted out-of-state units at that external bus — *behind*
  the model boundary — so their deliveries are metered against the import volume and
  interface caps instead of looking like in-state generation. Import CO2 moves onto the
  `unspecified_imports` carrier. The external-region code moved out of
  `add_extra_components.py` into a new `workflow/scripts/external_regions.py`.
  `config.california.yaml` ships `representation: generator` and prices imports at a flat
  `costs: 60` \$/MWh, because the EIA `wholesale` series is retail-priced
  ([#807](https://github.com/PyPSA/pypsa-usa/issues/807)). See {doc}`california-model`.

### Performance, environment and tooling

- **EIA API responses are cached on disk** (`8443b1be`). `DataExtractor._request_eia_data`
  memoises every response under `$EIA_CACHE_DIR` (default `data/eia/cache`, relative to
  `workflow/`, already gitignored), keyed on the API-key-stripped, parameter-sorted URL, so
  the key never lands in a cache file and the directory is shareable. Every EIA consumer
  (fuel costs, demand, trade, emissions, ...) benefits; cache failures are logged and never
  fatal, and non-200 responses are not cached.
- **`build_fuel_prices` only calls the EIA API when dynamic fuel pricing is on**
  (`b8eca6d3`). A default power-only run has
  `conventional: dynamic_fuel_price: enable: false` and never reads the state gas/coal price
  tables, but the script fetched them anyway whenever a key was present. The EIA key is
  therefore no longer needed for a default run; {doc}`about-install` lists the features that
  do need it.
- **The unused `AC_exp` carrier was dropped** (`b68d2502`) along with `add_itls`' ignored
  `expansion` argument and the carrier's colour entries in two validation maps. No link,
  constraint or figure changes.
- **Benchmark directives on every rule, plus Slurm resource tooling**
  ([#809](https://github.com/PyPSA/pypsa-usa/pull/809)). Eleven rules — including
  `cluster_resources` and `add_extra_components` — had no `benchmark:` directive, so
  snakemake recorded no wall time or peak RSS for them. Benchmark paths mirror each rule's
  log path and carry every wildcard. `run_slurm.sh` now takes a list of overlay configs and
  runs each as its own `snakemake --cluster` invocation; `slurm_submit.sh` wraps `sbatch`
  because several rules compute a float `mem_mb`, which `sbatch --mem` rejects;
  `collect_benchmarks.sh` and `report_benchmarks.py` turn the results into per-rule memory
  and walltime recommendations.
- **The conda environment was re-pinned so it solves again**
  ([#802](https://github.com/PyPSA/pypsa-usa/pull/802)). After the PyPSA v1 migration
  bumped pandas to 3.0.5 and geopandas to 1.1.4, `workflow/envs/environment.yaml` no longer
  solved on conda-forge and micromamba died at environment-creation time. pytables, lxml,
  python, rasterio, dask, tsam and others are re-pinned against pyproject; `numpy` stays at
  1.26.0. The uv/`pyproject.toml` path was never affected.

### Documentation since 2026-08-31

- Every custom constraint in {doc}`model-constraints` was rewritten in a uniform
  *let / s.t.* form and corrected against the implementation; the Operational reserves
  section was removed with the constraint. The default technology-capacity targets and the
  REC trading zones are now mapped, and the `{simpl}`/`{clusters}` figures were regenerated.

## Earlier releases

For changes prior to the v1 line, see the
[GitHub releases page](https://github.com/PyPSA/pypsa-usa/releases).
