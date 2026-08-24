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

- **Seam-plant fallback bounded to the model footprint in scoped runs
  (DL-13)** (`workflow/scripts/add_electricity.py`, commits d98cb93f and
  103f2194; harness adoption in
  `tests/equivalence/build.py::apply_seam_adoption`).
  `filter_plants_by_region` re-adds "must add" plants — those outside
  every ReEDS shape of the run's interconnect whose ReEDS membership
  contradicts their EIA `interconnection` — without testing them against
  the regions layers, and `match_plant_to_bus` then attaches whatever
  survives to the nearest bus at unbounded distance. Since DL-11 shrank
  those layers to the model footprint, a CA-only run picked up 23 plants /
  1,887.4 MW from as far away as Indiana (Hardy Hills Solar, 2,508 km).
  Now, when `model_topology.include` scopes the run, a must-add plant is
  kept only if it lies within `SEAM_PLANT_MAX_KM` = 100 km of the
  onshore+offshore regions (EPSG:5070); in-footprint plants are at
  distance 0 and always kept, and every drop is logged at WARNING with
  name, carrier, state, MW and distance. *Results effect:* **Accepted
  delta for scoped runs — DL-13, countersigned 2026-08-23.** Gated on
  `include`, so unfiltered interconnect runs (incl. the usa harness,
  `include: {}`) are byte-identical by construction. Of the 1,887.4 MW,
  1,725.0 MW actually reached the assembled network (onwind 1,416.5,
  solar 281.5, oil 27.0; the 162.4 MW of Fort Peck hydro never did, since
  hydro comes from the breakthrough files), and the change is symmetric:
  after mirroring onto the anchor, prong 1 is PASS with 0 live findings,
  objective rel 2.46e-06 and per-carrier existing capacity equal to
  4.5e-13 MW, while prong 2 leaves DL-9's absolute gaps (3,680.1 /
  3,586.6 MW) exactly unchanged. Third ADOPTED-FIX anchor patch after
  DL-11 and DL-12, and the first done by targeted string surgery rather
  than whole-file adoption, because `add_electricity.py` legitimately
  differs between the branches. Note: the first implementation unioned
  the region layers before measuring distance and crashed prong 2 with a
  GEOS "side location conflict" — reprojecting coarse cluster polygons to
  EPSG:5070 leaves 9 of 29 invalid — so the distance is now taken per
  region, which is mathematically identical (0.0 m difference where the
  union succeeds) and robust to invalid geometry.

- **`build_powerplants` EIA-860 pre-aggregation adopted onto the anchor
  (DL-12)** (candidate code already on `v1-epic`; harness adoption in
  `tests/equivalence/build.py::apply_powerplants_adoption`). v1-epic
  pre-aggregates the EIA-860 tables into `ges_latest` / `plants_latest` /
  `yg_latest` CTEs before the LEFT JOINs; upstream joins them raw, so ~24
  years of `report_date` vintages fan out per generator and reweight the
  means behind `heat_rate`, `fuel_cost` and `efficiency` (8,231 / 10,381 /
  8,380 differing cells in `powerplants.csv`; CA gas capacity-weighted
  fuel cost +0.0558 $/MMBtu). *Results effect:* **Accepted delta — DL-12,
  countersigned 2026-08-23.** Second ADOPTED-FIX anchor patch after DL-11:
  the anchor's script is replaced at provision time with the live
  candidate copy, sentinel-gated on the `ges_latest` CTE and guarded by an
  interface check against the pristine e7f8bd70 file. Restores prong-1
  solve exactness: objective rel 2.34e-2 -> ~2e-6, CCGT/OCGT `p_nom_opt`
  split eliminated, 0 live findings (prong 2 also PASS). Also regenerated
  the candidate's own stale `powerplants.csv` (predated its own derate
  fix; `--rerun-triggers mtime` never noticed). Known residual (own
  follow-up): `build_powerplants` is not bit-reproducible — DuckDB
  `first()` / tied `array_agg` picks perturb imputations at <=1e-5
  relative, below the harness's 1e-3 tolerance but non-zero.

- **Footprint-scoped empty-county sweep in `build_bus_regions` (DL-11)**
  (`workflow/scripts/build_bus_regions.py`,
  `workflow/rules/build_electricity.smk`, commit 88bede47; harness adoption
  in `tests/equivalence/build.py::apply_adopted_fix_patches`).
  When `model_topology.include` scopes a run (e.g. `reeds_state: [CA]`), the
  empty-county sweep now only considers counties inside the ReEDS zones
  retained in the filtered network, instead of the whole interconnect. Before
  the fix a CA-only run's onshore regions covered 2,930,688 km2 (86% outside
  CA), which passed the entire WECC fleet through
  `filter_plants_by_region`'s sjoin — 215.5 GW of existing capacity attached
  to a CA-demand-only model (22.6 GW coal, 7.7 GW nuclear). After: 409,842
  km2 (0.2% border slivers) and an 84.5 GW fleet matching California's
  actual one carrier-by-carrier. Gated on `include` being set: unfiltered
  interconnect runs (incl. the usa harness, `include: {}`) are untouched.
  *Results effect:* **Accepted delta for scoped runs — DL-11, countersigned
  2026-08-23.** By user decision the same patch is applied to the anchor
  worktree (first ADOPTED-FIX anchor patch, distinct from the numbers-neutral
  build-infra category) so the CA harness keeps comparing like-for-like;
  patch application drops a one-shot `.eq-force-rerun` marker because
  `--rerun-triggers mtime` neither reruns on code changes nor revisits
  missing intermediates when the final target looks current. Known residual
  (own follow-up): the `plants_must_add` seam-plant fallback still leaks
  ~1.9 GW of out-of-footprint plants into scoped runs.

- **Out-of-footprint NREL caps: loud accounting + opt-in nearest-bus
  reassignment** (`workflow/scripts/build_renewable_profiles.py`,
  `workflow/scripts/nrel_exclusion/build_nrel_bus_capacities.py`, new
  `nrel_caps_reassign` config block in `config.common.yaml`; unit tests in
  `workflow/scripts/test/test_remap_caps.py`). The NREL caps files are
  rolled up against the NATIONAL substation tessellation (17,890 entries);
  in footprint-scoped runs `remap_caps_to_cluster` silently dropna()'d
  every out-of-footprint entry — CA prong-1: 17,340/17,890 entries,
  9.43 TW of 9.70 TW national onwind p_nom_max (97.3%), including two
  border regions holding 13.4% of the West's developable wind (see ledger
  CF-coverage amendment). Now: (1) an UNCONDITIONAL WARNING reports the
  dropped entry count, dropped MW, and % of the national total per
  technology; (2) a config-gated, DEFAULT-OFF
  `nrel_caps_reassign: {enable: false, max_km: 100}` reassigns each
  unmapped entry to the cluster of the geographically nearest in-footprint
  entry, but only within `max_km` — preventing distant-interconnect
  capacity from teleporting across seams. The published Zenodo caps
  artifacts carry no per-entry coordinates, so enabling the flag today
  raises a clear config error; `build_nrel_bus_capacities.py` now writes
  per-bus `x`/`y` (capacity-weighted site centroid, bus-centroid fallback)
  so the NEXT HPC regeneration (`build_nrel_artifacts.sh` — raw per-site
  NREL CSVs live only on HPC) carries them. The long-term fix (option a)
  remains HPC-side: re-roll the caps against each run's own region
  geometry instead of the national tessellation.
  *Results effect:* **None by default** — flag off adds logging only
  (verified byte-identical remap output vs pre-change code on the CA
  prong-1 artifacts). Enabling the flag is a scenario choice that changes
  p_nom_max/potential on border buses and needs its own ledger entry.

- **Ledger amendments from the DL-2/DL-3/DL-9 + CF-coverage deep-dives**
  (2026-08-18, four low-effort investigations; full amendments in the deltas
  ledger): DL-2's +17.86% DC-link delta is an ANCHOR-side aggregation bug
  (pypsa link aggregation rescales the length-independent inverter-pair term
  by 1/1.25), not base-stage pricing — with a flagged follow-up that V1-epic
  now under-charges the DC-link km-term by 25% (~6% of link cost, 2 links);
  DL-3's palette diff was an uncommitted LOCAL plotting config on the
  candidate side (co2_emissions verified 0.000% different at every stage) —
  local config resynced from the template, class disappears on next rebuild;
  DL-9 confirmed MW-exact with the candidate's dropped onwind shown to be
  out-of-footprint plants snapped ~880 km into CA by the harness scoping
  (production full-Western s385 exposure measured: 1.54% onwind / 0.12%
  solar; prototype nearest-profiled-group fix recovers 100.0% with zero
  residual). CF-map coverage traced: 72.4% of regions have no onwind because
  NREL reference land access contains no eligible site (99.6% via absent
  caps entries; anchor set identical, 0.0%) — report captions now say
  "modeled resource exclusion, not missing data".
  *Results effect:* None (documentation, captions, and a local-config
  resync only).

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
