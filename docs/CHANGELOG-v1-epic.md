# Change-log: `v1-epic` relative to `develop`

Running record of every change on `v1-epic` relative to upstream `develop`.
Each entry says **what** changed, **why**, and its **expected effect on model
results** (None / Accepted delta / Under validation). Result differences that
survive the equivalence harness get a signed entry in the deltas ledger; this
file records the *code* changes themselves.

Conventions:
- Newest changes at the top within each section.
- Every entry cites its PR on `ktehranchi/pypsa-usa` (or commit, for
  un-PR'd work).
- "Results effect: None" is a claim that the equivalence harness (Tier C) will
  eventually enforce; until the harness is green, treat it as intent.

---

## Merged into `v1-epic`

### Simplify-early refactor (the core DAG restructuring)

- **[#9](https://github.com/ktehranchi/pypsa-usa/pull/9) + [#8](https://github.com/ktehranchi/pypsa-usa/pull/8) — split `simplify_network` and move clustering ahead of the heavy rules.**
  `simplify_network` (one rule doing substation aggregation + optional kmeans)
  became `aggregate_to_substations` (topology only → `elec_b.nc`) followed by
  `cluster_simpl` (kmeans/modularity → `elec_s{simpl}.nc`). Renewable
  profiles, demand construction, and `add_electricity` were repointed to run
  *after* clustering, at `{simpl}` granularity (~50–200 buses) instead of
  substation/nodal granularity (~750–3000 buses).
  *Why:* 5–10× peak-RSS reduction and large wall-time cuts in the per-bus
  rules (measured on Western: onwind profiles 142s→28s, `add_electricity`
  387s→116s, `power_build_demand` 352s→171s).
  *Results effect:* **Accepted delta (pending ledger entry).** Cluster
  assignments change because kmeans now runs on `Pd` static weights before
  demand exists; demand disaggregation is coarser per-cluster. Aggregate
  invariants (state demand totals, zonal capacity totals) are expected to
  hold; per-bus quantities differ by design.
  Also removed: HAC clustering (both stages), dead `to_substations` and
  `feature` config knobs.

- **[#11](https://github.com/ktehranchi/pypsa-usa/pull/11) — EGS supply curves aggregated through the `cluster_simpl` busmap.**
  New `aggregate_egs` rule remaps NREL substation-keyed EGS specs/profiles
  (capacity-weighted means for intensive, sums for extensive quantities) to
  cluster buses; `attach_egs` dropped its `bus2sub` join. `cluster_simpl` now
  exports `busmap_s{simpl}.csv`.
  *Why:* after the refactor, substation-keyed EGS data no longer matched
  network bus IDs — EGS generators silently failed to attach.
  *Results effect:* None intended for non-EGS runs; EGS runs restored to
  working (was silently broken post-#9).

### Cleanups

- **[#12](https://github.com/ktehranchi/pypsa-usa/pull/12) — `resources/` reorganized into category-first subfolders** (`networks/`,
  `busmaps/`, `profiles/`, `geospatial/`, `costs/`, `prices/`, `demand/`,
  `powerplants/`, ...) with matching Snakefile category constants. Filenames
  unchanged. *Results effect:* None (paths only).

- **[#10](https://github.com/ktehranchi/pypsa-usa/pull/10) — removed unused config keys and never-read snakemake rule params**
  (e.g. `electricity.prm`, `atlite.default_cutout`, leftover atlite-hydro
  keys, plotting thresholds; 58 lines deleted). Every removal verified to
  have zero code references. *Results effect:* None.

- **[#7](https://github.com/ktehranchi/pypsa-usa/pull/7) — `build_powerplants` pre-aggregates SCD tables.**
  *Why:* memory blowup fix. *Results effect:* None intended.

### Test infrastructure

- **[#13](https://github.com/ktehranchi/pypsa-usa/pull/13)/[#18](https://github.com/ktehranchi/pypsa-usa/pull/18)/[#19](https://github.com/ktehranchi/pypsa-usa/pull/19) — pytest scaffolding** (markers `fast`/`integration`,
  `tests/` skeleton, `test` extras). Merged, reverted, restored — net: in.
- **[#20](https://github.com/ktehranchi/pypsa-usa/pull/20) — Tier A static checks + Tier B fixture** (carries reviewed
  [#14](https://github.com/ktehranchi/pypsa-usa/pull/14) + [#15](https://github.com/ktehranchi/pypsa-usa/pull/15) content): DAG dry-run test, category-constant path
  validator, config-key AST check, `config.test.yaml` (CA slice), session-
  scoped snakemake build fixture. The path validator surfaced and fixed real
  stale-path bugs in `build_sector.smk` / `solve_electricity.smk` /
  `validate.smk`, and added missing `co2:` / `EGS.drilling_cost` defaults to
  `config.common.yaml` (these defaults are load-bearing for dry-runs).
  *Results effect:* None (the config defaults only unblock configs that
  previously crashed).
  Status: open, merge order #20 → [#16](https://github.com/ktehranchi/pypsa-usa/pull/16) (Tier B assertions) → [#17](https://github.com/ktehranchi/pypsa-usa/pull/17) (CI jobs).

## On local `v1-epic`, not yet pushed

- **Network schema tracking** — `log_network_schema` helper wired into the
  topology chain, electricity assembly, and solve scripts; seeded
  `docs/network-schema.md` catalog. *Results effect:* None (logging only).
- **`cluster_simpl` county fast-path helpers** — `resolve_simpl_mode` and
  `build_county_busmap` defined and unit-tested but **not yet wired into the
  script's dispatch** (`simpl="county"` would still crash). Wiring is planned
  Phase 3 work. *Results effect:* None yet.
- **repo_data config documentation/standardization commits.**
  *Results effect:* None.

## Uncommitted (working tree) — bugfixes found during refactor validation

All four are Phase 2 reconciliation candidates; each needs an equivalence-run
verdict (fix restores parity with anchor vs. accepted delta):

- **`add_electricity.py`** — (1) capital-cost regional multipliers now map
  buses to states via `reeds_state` → full state name (post-aggregation buses
  no longer carry `state`); (2) `match_plant_to_bus` skips the REEDS-zone
  first pass when plants lack a `country` column instead of crashing;
  (3) `attach_renewable_capacities_to_atlite` groups by bus identity instead
  of `sub_id` (post-aggregation, bus == substation).
- **`build_demand.py`** — state-zone lookup via `reeds_state` mapped to full
  state names (same root cause as above).
- **`build_powerplants.py`** — missing EIA-860 summer/winter derates filled
  with 1.0 (multi-unit combined-cycle sub-units otherwise propagate NaN into
  `p_max_pu`). *Note:* this one plausibly changes results vs anchor for
  affected plants — needs an explicit equivalence verdict.
- **`cluster_network.py`** — `cluster_regions` preserves the `country` column
  through dissolves (consumed by `match_plant_to_bus`).

## Planned (spec in progress)

- Merge latest `upstream/develop` into `v1-epic` and pin the new anchor
  (brings in upstream PRs #745–#766: pumped hydro, EER demand, StorageUnit
  max_hours fix, transmission cost data, ...).
- Tier C equivalence harness (CA now; USA deferred), two-prong pinned-busmap
  protocol, visual comparison reports, deltas ledger + waivers.
- `custom_busmap` repair (config key + pandas ≥2.0 `squeeze` fix) — required
  by the pinning protocol; currently vestigial and broken on both branches.
- County fast-path wiring in `cluster_simpl`.
- Profiling-driven memory/speed hit-list (candidates: dill-pickle handoff,
  demand pickle formats, profile chunking).
