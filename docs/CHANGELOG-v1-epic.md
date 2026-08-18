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

### Upstream sync (Phase 1)

- **Merged `upstream/develop` `e7f8bd70` (2026-07-22) into `v1-epic`** — the
  **anchor** commit for all equivalence runs. Brings in upstream PRs
  #745–#766 (NREL land-access exclusions, pumped hydro, EER demand profile,
  StorageUnit max_hours fix, 500 kV ITL cost data, EGS seismic mask defaults,
  myopic fixes, HOURS_PER_YEAR constant). Conflicts resolved: union of
  `bus_strategies` in `cluster_network` (kept `LAF_state`, took upstream's
  `rec_trading_zone`/`original_reeds_zone`); kept schema logging alongside
  upstream's `HOURS_PER_YEAR` in `prepare_network`; took upstream EGS
  defaults (`drilling_cost: advanced`, `seismic_exclusion: true`); folded
  upstream's EER `eer_file` option into the documented demand block; kept our
  CLAUDE.md. Post-merge fixes (caught by Tier A): upstream's new
  `postprocess.smk` block repointed from the old `Geospatial/` layout to the
  `GEOSPATIAL` category constant; `common.smk` now materializes
  `constants.py` into the snakemake source cache (upstream #764's import
  chain broke under a fresh cache). *Results effect:* carries upstream's
  intended changes; equivalence is measured against this same code, so no
  v1-epic-attributable delta.

## On local `v1-epic`, not yet pushed

- **Rename rule `cluster_simpl` -> `cluster_resources`** (user request
  2026-08-07; rule name, log path, walltime config key, living docs; the
  script file stays `workflow/scripts/cluster_simpl.py`). *Results effect:*
  None (identical code, outputs, and paths).
- **Report: % differences on every numerical delta** — findings now carry
  `max_rel_pct`/`rel_pct` (comparator), and all report narratives, map
  captions, and delta quotes state relative differences alongside absolutes.
  *Results effect:* None (reporting only).

- **Equivalence report revamped into a single explanatory artifact**
  (`tests/equivalence/report_sections/`, grilling session 2026-08-07):
  one `equivalence_report.html` with executive summary (verdict, provenance
  fingerprint, delta index, stage timeline), unified diff-DAG annotated with
  per-rule walltimes, stage-ordered narrative with born/inherited/masked
  delta tracking, 3-panel maps (V1-epic | anchor | difference) at the
  assembled and solved stages, and a rule-grouped benchmark table.
  "candidate" renders as "V1-epic" in all user-facing text (internal keys
  unchanged). *Results effect:* None (reporting only).

- **Fix 6.28% silent demand loss through transformer removal**
  (`workflow/scripts/aggregate_to_substations.py`). `remove_transformers`
  runs before demand exists in the reordered DAG and dropped
  transformer-secondary buses (420 on the CA slice) together with their
  `Pd`/`LAF_state` statics; `build_demand` allocates state demand by
  `LAF_state` without renormalizing, so the dropped share (CA LAF sum
  0.9367 vs 0.9995 base) vanished — total demand 9,541,668 vs anchor
  10,181,147 MWh (−6.281%, matching the dropped LAF to 5 significant
  digits), and 95 substations got zero load. `busmap_b.csv` also covered
  only 3,828/4,248 base buses, breaking base-keyed remaps. Fix: transfer
  `Pd`/`LAF_state` sums onto surviving buses through the trafo mapping and
  compose the trafo map into the exported busmap (mirrors the anchor's
  `trafo_map` composition, upstream simplify_network.py:246). Found by the
  Tier C system-total demand check. *Results effect:* restores demand
  conservation to anchor parity.

- **Fix double-applied `length_factor` in line/link capital costs**
  (`workflow/scripts/add_electricity.py:1091`). The simplify-early reorder
  put `add_electricity` after `aggregate_to_substations`, whose
  `assign_line_lengths(n, 1.25)` already folds `lines.length_factor` into
  `length`; the unchanged `update_transmission_costs(n, costs,
  params.length_factor)` call then multiplied by 1.25 again —
  `capital_cost = hav x 1.5625 x $/MW-km` instead of `x 1.25` (observed
  ratio 1.2486 on 2810/2811 lines vs anchor; TRANSMISSION_LIFETIME ruled
  out — both sides annuitize at CRP 60y / WACC 0.044). Found by the Tier C
  assembled-stage comparison. Masked in reeds transport-model runs (0
  lines survive clustering; ITL link costs recomputed identically — the
  clustered artifacts agree to full precision), but line-preserving runs
  (`transmission_network` != 'reeds', or `lv/lc` normalization in
  `prepare_network`) overpriced transmission expansion by 25%. Fix: pass
  `length_factor=1.0` at the post-aggregation call site, preserving the
  once-applied factor inside `length`. *Results effect:* restores parity
  with anchor for line CAPEX; no effect on this harness config's clustered
  or solved outputs (verified identical ITL costs).

- **`cluster_simpl` identity branch — normalize region bus names**
  (`workflow/scripts/cluster_simpl.py`, plus guards in
  `build_renewable_profiles.py` and `add_electricity.py`). With `simpl=''`
  (pass-through), `cluster_simpl` copied `regions_{onshore,offshore}.geojson`
  verbatim, keeping the substation-level float-formatted `name` values
  (`"35827.0"`) while the network and `busmap_s.csv` carry bare bus IDs
  (`"35827"`). The godeeep path of `build_renewable_profiles` intersects
  profile buses (from region names, `"35827.0"`) with NREL caps buses
  (remapped through the busmap, `"35827"`) — empty intersection, so it
  silently wrote profiles with a 0-length `bus` dim, and `add_electricity`'s
  `attach_wind_and_solar` later crashed on `pd.concat([])` (`ValueError: No
  objects to concatenate`) after skipping every empty horizon. The kmeans
  (`simpl=N`) branch was never affected: `cluster_regions()` already
  normalizes float names. Fix: apply the same float→int→str normalization in
  the identity branch; add a `RuntimeError` in `build_renewable_profiles` when
  the three bus-ID spaces are non-empty but disjoint (fail at the producer,
  not two rules later); and in `attach_wind_and_solar`, skip a carrier whose
  horizon profiles have no buses (parity with the single-profile branch's
  empty-bus `continue`) instead of crashing. *Results effect:* `simpl=''`
  runs gain the onwind/solar generators that were silently dropped
  (profiles were empty); `simpl=N` runs unchanged.
- **`config.equivalence.yaml` — add `costs.aeo.scenario: reference`** (in the
  tracked canonical `workflow/repo_data/config/config.equivalence.yaml`, plus
  the gitignored `workflow/config/` working copy; the harness re-copies the
  canonical file into the anchor worktree on every provision).
  `build_cost_data` crashed with an `IndexError` in `build_aeo_fuel_costs`
  (empty lookup for `natural_gas`): the equivalence config had no `costs.aeo`
  key, so the script fell back to its hardcoded default `"Reference"`, which
  matches **zero** rows of PUDL's `model_case_eiaaeo` (values are lowercase
  snake_case, e.g. `reference`). Anchor (`upstream/develop`) is equally
  affected — `build_cost_data.py` is byte-identical on both branches — so the
  fix lives in the shared harness config, not code; value mirrors
  `config.default.yaml`. (The script's `"Reference"` fallback remains a latent
  upstream bug: it can never match the data.) *Results effect:* None (the rule
  previously produced no output at all; both branches now consume the
  identical AEO `reference` fuel-cost slice).
- **`config.equivalence.yaml` — add solve-stage policy-file keys under
  `electricity`**: `regional_Co2_limits`, `technology_capacity_targets`,
  `portfolio_standards` (paths mirror `config.default.yaml`; CSVs already
  ship in `config/policy_constraints/`). Both branches' `solve_network`
  crashed identically with `KeyError: 'regional_Co2_limits'` in
  `add_regional_co2limit` (`workflow/scripts/opts/policy.py:548`), invoked
  by the `REM` opt in the harness opts string `REM-3h`; the sibling keys are
  the same-style direct subscripts behind the `TCT` and `RPS` opts. The
  remaining `config.default.yaml` policy siblings
  (`transmission_interface_limits`, `SAFE_reservemargin`,
  `SAFE_regional_reservemargins`) plus `agg_p_nom_limits` are referenced
  nowhere in `scripts/` or `rules/` on either branch (dead config) and were
  deliberately omitted. With this, both solves complete
  (candidate objective -2.5629e+09, anchor -2.5506e+09, both optimal).
  *Results effect:* None attributable to v1-epic — shared harness config,
  identical constraint added on both sides; objective deltas fall to the
  harness comparison stage.
- **Network schema tracking** — `log_network_schema` helper wired into the
  topology chain, electricity assembly, and solve scripts; seeded
  `docs/network-schema.md` catalog. *Results effect:* None (logging only).
- **`cluster_simpl` county fast-path helpers** — `resolve_simpl_mode` and
  `build_county_busmap` defined and unit-tested but **not yet wired into the
  script's dispatch** (`simpl="county"` would still crash). Wiring is planned
  Phase 3 work. *Results effect:* None yet.
- **repo_data config documentation/standardization commits.**
  *Results effect:* None.
- **`attach_breakthrough_renewable_plants` — remap breakthrough plant bus_ids
  through the substation busmap chain**
  (`workflow/scripts/add_electricity.py:~981`, plus new `busmap_s` input on
  `rule add_electricity` in `workflow/rules/build_electricity.smk`). Bug: the
  function filtered `plants.query("bus_id in @n.buses.index")`, but after the
  simplify-early reorder the network at this stage is substation-level
  (`elec_s{simpl}_dem.nc`) while `data/breakthrough_network/base_grid/plant.csv`
  and `hydro.csv` reference RAW base-grid bus_ids. On the CA equivalence
  harness every western hydro plant was silently dropped (12,848 MW missing),
  and 9 *Eastern* plants whose raw ids collide with western substation ids
  (5266, 5267, 5290–5292, 5332, 5333, 5340, 5341 — 128.6 MW) attached at wrong
  buses. Fix, patterned after `aggregate_egs`: remap `plants.bus_id` raw →
  `sub_id` (via `bus2sub.csv`, float-format `"35827.0"` normalized to
  `"35827"`) → cluster bus (via `busmap_s{simpl}.csv`) before the membership
  filter; per-plant generators keep their numeric plant-id names and attach at
  the mapped bus, and raw ids from other interconnects drop out naturally
  (eliminating the collision attachments). All ids compared as plain
  integer-strings. *Results effect:* restores the 12.8 GW of CA hydro dropped
  by the refactor — verified on the harness: hydro total p_nom 12,976.8 MW
  (matches anchor to 0.00%), per-ReEDS-zone p9 = 10,247.2 MW / p10 =
  2,622.6 MW (anchor ≈ 10,247 / 2,622), zero Eastern-collision plant ids in
  the network.

## Merged 2026-08-07 — bugfixes found during refactor validation (PR #21)

Merged into v1-epic mid-harness-bringup: the first candidate equivalence
build independently re-hit the `build_demand` bug below (`n.buses.state`
missing post-aggregation), confirming the candidate cannot build without
these. Each still gets its equivalence verdict from the harness comparison
(fix restores parity with anchor vs. accepted delta):

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

## Population-based demand allocation (2026-08-18)

- New `electricity.demand.bus_allocation` toggle (default `population`):
  per-bus demand-allocation weights now come from 2020 Decennial Census
  county populations (`data/population/DECENNIALDHC2020.P1-Data.csv`, county
  population split evenly across substations, then buses) instead of the
  2016-vintage Breakthrough Energy `Pd` column. `breakthrough` restores the
  legacy behavior and is pinned in `config.equivalence.yaml` (the anchor has
  no population method).
- Canonical weight lives in `n.buses.load_weight` (new helper
  `workflow/scripts/build_bus_population.py`); `Pd` is retained untouched for
  reference. Consumers switched: `LAF_state`, `WritePopulation`,
  `WriteIndustrial` load-bus filter, clustering `population` weighting,
  substation aggregation.
- A/B harness: `uv run python -m tests.equivalence.ab` builds the pipeline
  twice (`ab_pd` vs `ab_pop` run names), checks conservation invariants and
  reports per-zone/per-bus allocation shifts and solve-level deltas.

## Planned (spec in progress)

- Merge latest `upstream/develop` into `v1-epic` and pin the new anchor
  (brings in upstream PRs #745–#766: pumped hydro, EER demand, StorageUnit
  max_hours fix, transmission cost data, ...).
- Tier C equivalence harness (CA now; USA deferred), config-only two-prong
  protocol (no fixtures, no anchor patches), visual comparison reports,
  deltas ledger + waivers.
- `custom_busmap` cleanup (config key + pandas ≥2.0 `squeeze` fix) — vestigial
  and broken on both branches; not on the harness critical path.
- County fast-path wiring in `cluster_simpl`.
- Profiling-driven memory/speed hit-list (candidates: dill-pickle handoff,
  demand pickle formats, profile chunking).
