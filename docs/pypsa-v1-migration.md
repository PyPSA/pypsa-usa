# PyPSA v1 migration (pypsa 1.2.4 / linopy 0.9.1)

This branch migrates the whole workflow from `pypsa==0.30.2` / `linopy==0.3.14`
to `pypsa==1.2.4` / `linopy==0.9.1`. It is a redo of upstream
[PyPSA/pypsa-usa#762](https://github.com/PyPSA/pypsa-usa/pull/762) against the
v1-epic tree (which had diverged substantially: simplify-early DAG, `opts/`
constraint modules, equivalence harness), verified against the real pypsa 1.2.4
API rather than ported line-by-line.

**Why 1.2.4 and not 1.3.0:** pypsa 1.3.0 requires `pandas>=3.0`; the repo pins
`pandas==2.2.2` and a pandas-3 bump (copy-on-write, string dtype) is a separate
migration. 1.2.4 is the newest v1 that runs on pandas 2.x. All code written
here uses only API that behaves identically on 1.3.0, so the later bump should
be pins-only.

## Migration map

| Pre-1.0 API | v1 replacement | Where |
|---|---|---|
| `n.madd` / `n.mremove` | `n.add` / `n.remove` (removed, not deprecated) | everywhere (~180 call sites) |
| `n.add(..., names=)` | `n.add(..., name=)` | add_sectors, build_natural_gas, add_extra_components, cluster_network |
| `x = n.madd(...)` (returned Index) | v1 `add` returns `None`; compute names explicitly | add_extra_components H2 buses |
| `clustering.network` | `clustering.n` (dataclass field rename) | cluster_network, cluster_simpl, aggregate_to_substations |
| `n.df(c)` / `n.pnl(c)` | `n.components[c].static` / `.dynamic` | ~20 sites incl. equivalence harness |
| `n.iterate_components(X)` | iterate `n.components` (yields typed `Components`) | _helpers, summary, solve_network, prepare_network, plot_network_maps, add_sectors |
| `pypsa.descriptors.get_switchable_as_dense(n, ...)` | `n.get_switchable_as_dense(...)` method | opts/policy, opts/reserves, plot_sankey_energy, add_extra_components, tests |
| `pypsa.descriptors.{get_activity_mask, get_bounds_pu, expand_series}` | `n.components[c].get_activity_mask/.get_bounds_pu` (xarray), `pypsa.common.expand_series` | opts/reserves |
| `pypsa.descriptors.Dict` | `pypsa.definitions.structures.Dict` | _helpers (mock_snakemake) |
| `pypsa.components.component_attrs` | `n.components[c].defaults` | add_electricity UC defaults, summary |
| `n.copy(with_time=False)` | `n.copy(snapshots=[])` | prepare_network |
| `pypsa.statistics.get_bus_and_carrier` etc. | `groupby=["bus", "carrier"]` string lists | summary, plot_statistics |
| statistics `comps=` / `aggregate_time=` | `components=` / `groupby_time=` | summary, summary_sector |
| `pypsa.pf.logger` | `logging.getLogger("pypsa.network.power_flow")` | solve_network |
| `from pypsa.components import Network` | `from pypsa import Network` | build_natural_gas |
| `n.get_extendable_i / get_non_extendable_i / get_committable_i` | `n.components[c].extendables / .fixed / .committables` | opts/reserves, opts/_helpers |

## Semantics changes that needed pinning (not just renames)

1. **Model variable dims are `"name"`** — linopy variables no longer carry
   `"Generator"` / `"Generator-ext"` dims. Every `.sel(Generator=...)`,
   `.rename({"Generator-ext": ...})`, `rename_axis("Generator-ext")` and
   grouper axis in `opts/` was rewritten to dim `"name"`. The two RESERVES
   operational-constraint builders now mirror pypsa v1's internal
   implementations.
2. **linopy 0.9 unstacks MultiIndex DataFrames** — multiplying a linopy
   expression by a `(period, timestep)`-indexed DataFrame produces dense
   `period × timestep` dims. Coefficient frames are wrapped in `DataArray(...)`
   first so the flat `snapshot` dim is preserved (opts/reserves).
3. **Cyclic storage defaults flipped** — v1 changed
   `e_cyclic_per_period` / `cyclic_state_of_charge_per_period` defaults
   True→False. All 15 cyclic adds now pin `*_per_period=True` explicitly to
   keep per-investment-period cyclicity (results-equivalence).
4. **Bus index is named `"name"`** (was `"Bus"`). The `bus2sub.csv` artifact
   keeps its legacy `Bus` header via `index_label="Bus"`; in-memory
   nearest-bus matching is now index-positional (also fixing two latent
   `.Bus`-attribute bugs in `build_base_network.match_missing_buses`).
5. **Empty component frames carry an int64 index** — `.str` accessors on
   possibly-empty component indexes need an `empty` guard
   (solve_network.freeze_prior_periods).
6. **UC ramp-limit fixes upstream** — first-snapshot ramp limits are now
   enforced and `ramp_limit_start_up/shut_down` defaults changed 1→NaN;
   unit-commitment runs may show small accepted deltas.
7. **pypsa bug (present through 1.3.0):** `Network.copy()` drops the hidden
   `name="snapshot"` attribute of MultiIndex snapshots, breaking the `c.da`
   xarray accessors on the copy (`dim_0` instead of `snapshot`). netCDF
   round-trips are unaffected, so production paths are safe; the unit-test
   conftest wraps `Network.copy` with a `set_snapshots(n.snapshots)` heal.
   Worth reporting upstream.

## Verification

- Unit tier: **45 passed / 1 skipped** (`workflow/scripts/test/`). The
  migration un-skipped and fixed 9 tests previously marked "pre-existing
  failure on v1-epic" (RPS, TCT, regional CO2 clustered, ERM multi-period) —
  their failures were old-stack artifacts. The one remaining skip
  (`test_e2e_solve_network_myopic`) was bisected: its fixture model is
  infeasible on pristine v1-epic under pypsa 0.32 too (pre-existing).
- Static tier: **72 passed** (`tests/static/`, includes full-DAG dry-runs).
- Not yet run: a full `data_model`/solve pipeline run and the Tier-C
  equivalence harness against develop anchors (needs data downloads + solver;
  the env itself changed, so deltas must be re-baselined per the harness
  conventions).

---

# PyPSA v1 data-storage features worth adopting

Surveyed from pypsa 1.0–1.3 release notes and the installed 1.2.4 API.
Ordered by leverage for this repo.

## Adopted in this migration

- **Typed components layer (`n.components` / `n.c`)** — `.static`, `.dynamic`,
  `.defaults`, `.extendables/.fixed/.committables`, `.get_activity_mask`,
  `.get_bounds_pu`, and xarray views `.da` / `.ds`. All migrated code now goes
  through it. The xarray views are the big win for constraint code: masks and
  bounds arrive dim-aligned with model variables (`opts/reserves.py` already
  uses `c.da.active` instead of hand-converting DataFrames).
- **Statistics groupers as string lists** — `groupby=["name", "bus", "carrier"]`
  replaces imported grouper functions.

## High-leverage candidates (next steps)

1. **Custom groupers on our custom columns**
   (`pypsa.statistics.groupers.add_grouper`). We can register the repo's own
   dimension columns once — `reeds_state`, `reeds_zone`, `reeds_ba`,
   `rec_trading_zone`, `interconnect`, `STATE` — and then write
   `n.statistics.supply(groupby=["reeds_state", "carrier"])` everywhere.
   Kills most of the hand-rolled `groupby(n.buses.reeds_state)` joins in
   `summary.py`, `plot_statistics.py`, `summary_sector.py`.

2. **`n.shapes` (GeoDataFrame component, serialized into the netCDF).**
   Today `regions_onshore/offshore_s{simpl}.geojson`, ReEDS shapes, and county
   tessellations travel as parallel artifacts keyed back to buses by path
   plumbing in the Snakefile. Storing bus regions in `n.shapes`
   (`component="Bus"`, `idx=<bus>`, `type="onshore_region"`) makes every
   downstream network self-describing — plotting and GIS exports stop
   re-joining geojson by wildcard. Adoption can be incremental: write shapes at
   `build_bus_regions`/`aggregate_to_substations`, keep the geojson artifacts
   until consumers are ported. (Cost: netCDF size; regions at `{simpl}`
   granularity are small.)

3. **`n.meta` as the provenance channel.** Already used in `cluster_network`
   (config + wildcards). Extend to every network-writing rule: rule name,
   config hash, wildcards, and the `log_network_schema` snapshot. The
   equivalence harness (`tests/equivalence/compare.py`) and the schema-tracking
   initiative then read provenance from the artifact itself instead of paths —
   and `n.meta` survives netCDF round-trips.

4. **`NetworkCollection` for multi-network analysis.** The myopic loop already
   writes `result_period_{p}.nc` per horizon, and validation compares
   scenarios/years. `pypsa.NetworkCollection([...])` gives cross-network
   `.statistics` with a scenario/period index in one call — a natural fit for
   `plot_statistics.py`'s side-by-side comparisons and for backcasting
   validation sweeps. (Experimental API; pin usage to simple statistics.)

5. **`Network.equals` in the equivalence harness.** `compare.py` hand-rolls
   frame diffs; `n.equals(other, log_mode="verbose")` can provide the fast
   exact-match path, with the hand-rolled tolerance/rekey logic kept only for
   the loud-diff reporting path.

6. **Temporal clustering module (v1.1: `n.cluster.temporal`).**
   `resample()/segment()` (TSAM-backed) can replace the hand-rolled
   `average_every_nhours` + `apply_time_segmentation` in `prepare_network.py`,
   including correct multi-period snapshot-weighting bookkeeping — the exact
   code the migration had to touch for `copy(snapshots=[])`.

7. **Stochastic scenarios (v1.0: `n.set_scenarios`).** Stores
   scenario-dimensioned data (e.g. weather years) inside one network via a
   `scenario` MultiIndex level, with two-stage stochastic optimization on top.
   Directly relevant to the `renewable_scenarios` (GODEEEP) machinery, which
   currently multiplies whole networks.

8. **`pypsa.options`.** Global/env-var config (`PYPSA_*`). Two concrete uses:
   `pypsa.options.api.legacy_string_dtype` for the eventual pandas-3 bump, and
   consistency-check verbosity control in batch runs.

## Noted, lower priority

- **Excel / cloud-path IO** (v0.34+): `import_from_excel`, `cloudpathlib`
  support for reading/writing artifacts on S3/GCS — useful for HPC result
  sync, no code depends on it today.
- **Process component** (v1.2): multi-port conversion component with per-bus
  rates; a cleaner model for the CCS/DAC `efficiency2/3` link patterns in
  `add_extra_components.py`/`build_natural_gas.py`, but a results-affecting
  remodel, not a storage swap.
- **Piecewise-linear costs, maintenance scheduling, phase-shifters** (v1.3):
  new modeling capabilities gated on the pandas-3/pypsa-1.3 bump.
