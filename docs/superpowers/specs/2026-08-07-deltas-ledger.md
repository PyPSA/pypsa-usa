# Deltas Ledger — v1-epic vs anchor (upstream/develop e7f8bd70)

One row per accepted result-difference between the candidate and the anchor
under the Tier C equivalence harness (CA prong 1, `config.equivalence.yaml`).
Every row must have a matching machine-readable waiver in
`tests/equivalence/waivers.yaml`, and vice versa. **Sign-off column:** all rows DL-1..DL-10 (including the 2026-08-18
amendments and the DL-2 follow-up decision) were countersigned by
ktehranchi on 2026-08-18.

| ID | Stage(s) | Delta | Root cause | Why accepted | Sign-off |
|----|----------|-------|------------|--------------|----------|
| DL-1 | assembled_substation_network | `Line.capital_cost` differs ≤0.2% on 2810/2811 lines (max $19/MW-yr) | Stage-ordering artifact: anchor prices lines pre-aggregation on base-network sum-of-segment haversine lengths (1.00112× endpoint); candidate prices post-aggregation on endpoint lengths. Same formula, same $/MW-km, same factor (after the double-`length_factor` fix, commit bd69126e). | Clustered-stage transmission (ITL links) is recomputed identically on both branches — verified equal to full precision. The assembled-stage residue never reaches results in any config: line-preserving configs re-derive costs downstream of lengths that themselves agree. | countersigned (ktehranchi, 2026-08-18; adjudicated Claude 2026-08-07) |
| DL-2 | assembled_substation_network | `Link.capital_cost` differs on the 2 DC links (15_fwd/15_rev: 10894.9 vs 9244.1) | Same stage-ordering artifact as DL-1: anchor cost computed at base stage where the DC link length was the shorter pre-aggregation value. | Same masking argument as DL-1; clustered ITL links identical. | countersigned (ktehranchi, 2026-08-18; adjudicated Claude 2026-08-07) |
| DL-3 | all network stages | `Carrier.color` differs (onwind, 4hr_battery_storage) | v1-epic updated plotting palette entries. | Pure plotting metadata; no solver input. | countersigned (ktehranchi, 2026-08-18; adjudicated Claude 2026-08-07) |
| DL-4 | all network stages | `Bus.control`/`Bus.generator` slack assignment differs (slack at p9 vs p10); `Generator.control` PQ/Slack/'' bookkeeping differs | pypsa assigns slack per sub-network from the first generator encountered; generator ordering differs with the DAG reorder. | Power-flow bookkeeping only; LOPF/expansion solves ignore `control`. | countersigned (ktehranchi, 2026-08-18; adjudicated Claude 2026-08-07) |
| DL-5 | assembled + clustered | Pebbly Beach Generating Station Hybrid (EIA 6704, Catalina Island; 11.3 MW oil + 1.0 MW battery) assigned to bus 36973 (candidate) vs 37317 (anchor) | `match_plant_to_bus` nearest-bus matching now runs against 1,975 substation coordinates instead of 4,248 nodal coordinates; the offshore island plant's nearest neighbor flips. | Inherent to substation granularity; 12.3 MW on an island interconnection; zone-level totals unaffected (both buses in p11). Waiver lands after the hydro-fix rebuild isolates this as the sole [bus,carrier] residual. | countersigned (ktehranchi, 2026-08-18; adjudicated Claude 2026-08-07) |
| DL-6 | assembled + clustered | Generator/StorageUnit element NAMES differ (candidate per-plant ids at assembled, anchor pre-aggregated `{bus} {carrier}`); anchor-only `Generator_t.p_max_pu` columns for aggregate names | Stage-ordering: anchor aggregates one-ports/generators at simplify (pre-comparison stage); candidate keeps per-plant granularity until cluster_network. Attach code byte-identical. | Content is guarded by the UNWAIVED `Generator[bus,carrier]` / StorageUnit aggregate comparisons (battery 21,254.0 MW, PHS 3,978.4 MW match exactly). Names are not results. | countersigned (ktehranchi, 2026-08-18; adjudicated Claude 2026-08-07) |
| DL-7 | clustered stages | Small conventional-generator parameter diffs at p8–p11: `fuel_cost` ≤ 2.55 $/MWh (n=17), `efficiency` ≤ 0.0028 (n≤9), `marginal_cost` ≤ 2.55 $/MWh | Non-composable aggregation strategies (`mean` for fuel_cost/vom): anchor aggregates plants→sub→zone (mean of means), candidate plants→zone (single mean). Same plants, same strategy config, different composition order. | Inherent to the single-step aggregation of the refactor; bounded and small; capacity totals exact. Flagged for the user: switching those strategies to capacity-weighted means would make aggregation composable and eliminate the delta class. | countersigned (ktehranchi, 2026-08-18; adjudicated Claude 2026-08-07) |
| DL-8 | all network stages | `Bus.Pd` static differs (99 buses at assembled; zonal sums differ ≤7%) | The demand-conservation fix makes the candidate transfer `Pd` through transformer removal; the anchor conserves actual Load components but leaves its `Pd` static stale after aggregation. | `Pd` post-demand-attachment is bookkeeping: `Load`/`Load_t.p_set` comparisons are finding-free (actual demand identical); `Pd` only feeds upstream kmeans weighting (prong-2, by-design territory). Candidate's accounting is the more correct one. | countersigned (ktehranchi, 2026-08-18; adjudicated Claude 2026-08-07) |
| DL-9 | prong 2 only (prong2_aggregates + solved) | Existing onwind p_nom 33,583.0 vs 29,164.4 MW (+15.2%), solar 49,014.8 vs 45,428.2 (+7.9%); objective rel 0.22%; p_nom_opt per carrier follows | `attach_renewable_capacities_to_atlite` silently drops existing plant capacity whose aggregation group has no renewable-profile generator (shared code, both branches; the code's own log cites "git issue #16"). The count of profile-less groups is a function of cluster geometry: at 20-cluster granularity 19/20 groups have onwind profiles; at the anchor's substation granularity only ~half of plant-bearing subs do. MW accounting closes exactly on both sides (Δattached = Δdropped: onwind 4,418.6, solar 3,586.6 MW); double-counting explicitly ruled out. | Inherent discretization consequence of by-design different cluster geometries; each side internally consistent and conserved through clustering to 0.1 MW. The candidate is strictly closer to plant-data ground truth (drops 6.1% vs 18.5% onwind, 0% vs 7.3% solar). Prong-1 exactness (which shares the attach code) stays fully guarded. NOTE for upstream: the silent-drop itself is a shared pre-existing limitation both branches suffer — flagged as follow-up work. | countersigned (ktehranchi, 2026-08-18; adjudicated Claude 2026-08-07) |

Resolution 2026-08-07 (post-fix rebuild): demand total and hydro FIXED in
code (commits cccf696f et al.) — no waivers needed; solved objective now
agrees at rel 6.6e-05 (gate 1e-3). All remaining findings are DL-5..DL-8
classes; waivers finalized in tests/equivalence/waivers.yaml.


## Amendments (2026-08-18, Fable deep-dive investigations)

- **DL-2 — mechanism corrected (waiver outcome unchanged).** The anchor's
  9,244.1 is not a "base-stage price": the anchor prices the DC links at base
  stage as 11,555.2 (53.819 km x 1.25 x 48.856 + 8,268.4 inverter pair), then
  its `simplify_network` aggregation (`get_clustering_from_busmap(...,
  line_length_factor=1.0)`) rescales `capital_cost` by new/old length = exactly
  1/1.25 — wrongly deflating the **length-independent inverter-pair term** by
  20.0% (8,268.4 -> 6,614.7). Delta decomposition: +25.00% on the inverter term,
  −0.11% on the km term. So the +17.86% is an **anchor-side aggregation bug**,
  masked in reeds transport configs (DC links dropped at clustering; ITL costs
  rebuilt identically). In line-preserving (tamu) runs it reaches the objective
  whenever links are extendable (`lv>1.0`/`lvopt`) or under `lc*` budgets.
  Follow-up flagged for the user: V1-epic's `length_factor=1.0` call correctly
  de-duplicates LINE costs, but DC-link lengths are NOT pre-multiplied by the
  1.25 routing factor, so V1-epic under-charges the DC-link km-term by 25%
  (~6% of total link cost, 2 links) — restoring the factor for links only
  would put V1-epic at 11,551.5 vs the anchor's true base price 11,555.2
  (0.03%). Modeling decision pending sign-off.
- **DL-3 — mechanism corrected (waiver outcome unchanged; root cause now
  FIXED at source).** The color differences did not come from a committed
  v1-epic palette change: no commit anywhere contains the candidate hexes.
  The candidate ran with an uncommitted, local 67-key slim
  `workflow/config/config.plotting.yaml` (vs the 274-key template the anchor
  used); only onwind and 4hr_battery_storage exist as carriers, which is why
  only they surfaced. `co2_emissions` — the ONLY Carrier column feeding the
  solve (GlobalConstraint via prepare_network) — is numerically identical
  (0.000% on every carrier at every stage). Remediation: local plotting
  config resynced from the repo_data template (2026-08-18); the color class
  disappears on the next candidate rebuild. Incidental: the anchor carries a
  stale EMPTY `DC` Carrier row at clustered stages (0 member components,
  co2 0.0) — DL-6 stage-ordering class, zero solver content.
- **DL-9 — confirmed MW-exact, scope sharpened.** New finding: the
  candidate's dropped onwind (2,183.0 MW, 6.10%) is entirely
  OUT-OF-FOOTPRINT interconnect plants snapped a median 878 km into CA buses
  by the CA-scoped harness — a harness scoping artifact, not in-state
  capacity; the anchor's drops (6,601.6 MW onwind 18.46%, 3,586.6 MW solar
  7.30%) are genuine local orphans (median 5.7 km from a profiled group).
  Production exposure measured on full-Western s385: onwind 1.54%, solar
  0.12%. Prototype fix (reassign to nearest profiled group in-zone) recovers
  100.0% of dropped MW with zero residual in every tested geometry and would
  null this waiver class. Scope confirmed prong-2 only; issue-#16 fix
  priority MEDIUM (matters most for high-resolution runs). Also flagged: the
  anchor-style sub_id grouping can double-attach when one sub has 2+ profiled
  buses, and a shared −111.8 MW solar offset (both sides, cancels in diffs)
  deserves a one-line audit.
- **Onwind CF coverage (context, not a delta):** 1,428 of 1,972 regions
  (72.4%, 20.4% of Western land area) have no onwind profile because the NREL
  reference land-access supply curve contains no eligible site there (urban
  Bay/LA/SD coast, Sierra counties); 99.6% via absent caps entries, joins
  lossless; anchor bus set identical (0.0% difference). Expected behavior.
  The `limited` variant is stricter still (−53.3% buses); no `open` artifacts
  on disk. Report captions updated to say "modeled resource exclusion, not
  missing data."

- **DL-2 follow-up DECIDED (user, 2026-08-18):** keep `length_factor=1.0` at
  the post-aggregation costing call — the pipeline must not edit length data
  at that stage. Recorded note: the length factor only modifies transmission
  costs on the TAMU (line-preserving) network; under the reeds transport
  model, lines and DC links are dropped at clustering and ITL link costs are
  rebuilt from the ReEDS distance-cost tables, so the factor never reaches
  the solve there. The 2-link DC km-term difference (~6% of link cost)
  is accepted under this decision.

- **CF-coverage note SUPERSEDED by source-data verification (user-requested,
  2026-08-18):** split verdict. (a) In-state CA missing regions: NREL's own
  reference-access raster corroborates the exclusion (median 0.0% developable
  in missing regions vs 8.1% kept; Inyo 0.0%, Tulare 0.0%, Fresno 1.2%,
  LA 0.3%, SD 1.7%); only 450 MW (0.17% of 265.8 GW) sits in missing CA
  regions; the sjoin is NOT buggy (49,264/49,323 national sites land inside
  polygons, 0.12% dropped). (b) REAL GAP out of state: caps are rolled up
  against the NATIONAL substation tessellation (17,890 entries) but the
  CA-focus run's busmap covers only 1,975 CA substations —
  `remap_caps_to_cluster` (build_renewable_profiles.py:50) silently
  dropna()s the 17,340 unmapped entries. Two giant border regions holding
  13.4% of the West's NREL-developable wind area (~100+ GW at the calibrated
  ~2.8 MW/km2) get 0 MW, and several kept giants are severely undercounted
  (e.g. 782k km2 region -> 186 MW). Shared identically by both sides (no
  equivalence delta) but diverges structurally from the legacy atlite path.
  Follow-up flagged: roll caps up against the run's own region geometry, or
  warn loudly on partial dropna.

- **DL-10 (usa interconnect, data stages; 2026-08-18): state-assignment
  demand split.** At national scope, per-bus `Load_t.p_set` differs on ~2,300
  buses (individual buses up to 394%) while the SYSTEM total conserves to
  0.0485% and state totals shift by at most 1.6% (KS; 11 further states
  0.3-0.9%). Root cause: the candidate allocates state demand to buses via
  ReEDS membership (`reeds_state`, the PR #21 fix), the anchor via the raw
  breakthrough `state` column — border-adjacent buses are assigned to
  different states, reshuffling demand between neighboring states. Same
  family: 93/38,974 generator `capital_cost` values (max 8.4%, regional capex
  multipliers keyed on the same state assignment) and one 296 MW CCGT
  placement flip (bus 35790, DL-5 nearest-bus class). Invisible in CA-only
  runs (every bus is California on both sides). **Scientific decision for
  the user:** which state assignment is authoritative — ReEDS membership
  (consistent with all other zonal machinery) or the raw breakthrough
  column? Waived provisionally (usa-scoped) with reeds_state as the
  candidate behavior. | countersigned (ktehranchi, 2026-08-18; adjudicated Claude 2026-08-18) |

- **Out-of-footprint caps follow-up IMPLEMENTED, default-off (2026-08-18):**
  the CF-coverage amendment's flagged follow-up ("warn loudly on partial
  dropna") is now code: `remap_caps_to_cluster` unconditionally WARNs with
  the dropped entry count, dropped MW, and % of the national total per
  technology (CA prong-1 onwind: 17,340/17,890 entries, 9.43 of 9.70 TW,
  97.3%). A new `nrel_caps_reassign: {enable: false, max_km: 100}` config
  key opts into reassigning unmapped entries to the nearest in-footprint
  bus within max_km; the published caps .nc files carry no per-entry
  coordinates, so enabling it today raises a config error until the HPC
  rollup (`build_nrel_bus_capacities.py`, which now writes per-bus x/y) is
  regenerated. **No new delta vs anchor:** the flag defaults OFF and the
  flag-off remap output was verified byte-identical
  (xr.testing.assert_identical) to the pre-change code on the CA prong-1
  artifacts — this entry is documentation of a behavior-neutral change.
  Enabling the flag is a scenario choice that will need its own ledger
  entry (delta class: border-bus p_nom_max/potential/weight increases)
  the first time it is used in an equivalence-checked run. | needs no
  countersignature while default-off |

- **DL-9/CF-coverage root cause found: interconnect-wide empty-county sweep
  (prototype 2026-08-22, branch `proto/footprint-scoped-regions`, commit
  ccfe4b77 — NOT on v1-epic).** `build_bus_regions`'s empty-county sweep
  (upstream PR #723) tests counties against the FULL interconnect ReEDS
  footprint, ignoring `model_topology.include`, and glues every busless
  county onto the nearest retained bus. A CA-only run's onshore regions
  therefore covered 2,930,688 km2 — 86.0% outside California, ~7x the
  state. Consequences measured on the CA harness (candidate side, both
  prongs rebuilt from build_bus_regions with the prototype): (a)
  `filter_plants_by_region`'s sjoin passed the whole WECC fleet — 215.5 GW
  existing capacity attached to a CA-demand-only model (22.6 GW coal,
  7.7 GW nuclear, 29.2 GW onwind, 45.4 GW solar); after scoping the sweep
  to the network's ReEDS zones the fleet is 84.5 GW and matches CA's
  actual one carrier-by-carrier (coal 62.5 MW = Argus Cogen, nuclear
  2,323 MW = Diablo Canyon, CCGT+OCGT 39.1 GW, geothermal 2.77 GW; hydro
  unchanged at 12,976.8 MW — it attaches by bus_id, not geometry). (b)
  Regions after: 409,842 km2, 0.2% out-of-state slivers; count unchanged
  (1,972). (c) Border-bus godeeep CFs shift: onwind 9/544 buses (max
  |dCF| 0.147, cap-wtd mean CF −2.41%), solar 11/808 (max 0.047,
  +0.24%). (d) Caps-derived p_nom_max/weight sums UNCHANGED (0.000%) —
  the out-of-footprint caps drop is an independent bug. (e) Demand
  identical (10,181,147 MWh, 1,674 load buses). (f) Solved objective
  moves −91.3% (prong 1) / −89.4% (prong 2); p_nom_opt flips (CCGT
  0→12.6 GW, coal 22.7→0.06 GW, onwind 29.2→4.2 GW). This is the parent
  of DL-9's "plants snapped a median 878 km" observation. Shared
  identically by the anchor, so the CA harness stayed green while both
  sides simulated most of WECC's fleet against CA demand. Fix is gated on
  `include` being set — unfiltered interconnect runs byte-identical.
  ADOPTING IT BREAKS CA-HARNESS EQUIVALENCE BY DESIGN (the anchor keeps
  the contamination): needs its own ledger row, a decision on how the CA
  harness re-baselines, and user countersignature before landing on
  v1-epic. Before/after artifacts + patch preserved in the session
  scratchpad (`before_footprint_fix/`, `after_footprint_fix/`,
  `footprint_scoped_regions.patch`). | superseded by DL-11 below (adopted
  2026-08-23) |

- **DL-11 (scoped runs, all stages; ADOPTED on both sides): footprint-scoped
  empty-county sweep.** `build_bus_regions` restricts the empty-county
  nearest-bus assignment to the ReEDS zones present in the
  `model_topology.include`-filtered network (v1-epic commit 88bede47, from
  prototype ccfe4b77; quantification in the 2026-08-22 amendment above:
  regions 2.93M→0.41M km2, existing fleet 215.5→84.5 GW, demand and
  caps-derived p_nom_max unchanged, objective −91.3%/−89.4%). USER DECISION
  2026-08-23: fold into v1-epic AND mirror onto the anchor, so the CA
  harness compares two footprint-correct pipelines instead of freezing the
  contamination. Implemented as the first ADOPTED-FIX anchor patch
  (`tests/equivalence/build.py::apply_adopted_fix_patches`, marker-idempotent,
  documented exception to the D10 "no anchor patches" rule — unlike the
  build-infra category this one changes numbers BY DESIGN, identically on
  both sides). Because the harness builds with `--rerun-triggers mtime`
  (code changes never invalidate outputs, and missing intermediates are NOT
  revisited when the final target looks current — both observed 2026-08-23),
  a newly applied patch also drops a one-shot `.eq-force-rerun` marker that
  `build_side` turns into `-R build_bus_regions` and clears on success.
  Unfiltered runs (usa harness `include: {}`) untouched, so
  whole-US results and all non-scoped configs carry no delta. KNOWN
  RESIDUAL (2026-08-23 adversarial review, empirically reproduced):
  `filter_plants_by_region`'s `plants_must_add` seam-plant fallback bypasses
  the region sjoin and, via the unbounded second-pass nearest-bus match,
  still attaches ~27 out-of-footprint seam plants / 1,890.6 MW to CA buses
  (Buffalo Ridge II SD 210 MW, Hardy Hills IN 195 MW, Fort Peck MT 162 MW,
  ~1.1 GW NM wind/solar). Pre-existing and shared by both sides (no
  equivalence delta); spun off as its own follow-up fix requiring its own
  ledger entry. NOTE: DL-9's
  recorded magnitudes (existing onwind 33,583.0 vs 29,164.4 MW etc.) were
  measured under the pre-DL-11 contamination; its mechanism (issue-#16
  profile-group drops vs cluster geometry) is unchanged but the numbers are
  superseded by the post-DL-11 rerun recorded below. | countersigned
  (ktehranchi, 2026-08-23) |

- **DL-12 (all stages, prong 1; ADOPTED on both sides): EIA-860 history
  pre-aggregated before the `build_powerplants` LEFT JOINs.** Upstream's
  `load_eia_operable_data` / `load_heat_rates_data` join
  `out_eia__yearly_generators` and `core_eia860__scd_*` raw, so each
  generator fans out against ~24 `report_date` vintages before the aggregate
  collapses them; v1-epic pre-aggregates in `ges_latest` / `plants_latest` /
  `yg_latest` CTEs so each generator contributes once. On the shared
  `powerplants.csv` (31,405 rows x 55 columns, identical index both sides)
  this moved 8,231 `fuel_cost`, 10,381 `heat_rate`, 8,380 `efficiency` and
  13,679 `marginal_cost` cells; CA gas capacity-weighted fuel cost was
  offset +0.0558 $/MMBtu (5.0808 vs 5.0250). The duplicate-weighted means
  are demonstrably wrong — upstream credits Watson Cogen (1987 vintage) a
  physically impossible 62% efficiency — and this accounted for essentially
  all of the live prong-1 divergence. USER DECISION 2026-08-23: adopt the
  candidate query on the anchor rather than sign the divergence as a delta.
  Implemented as the SECOND ADOPTED-FIX anchor patch
  (`tests/equivalence/build.py::apply_powerplants_adoption`, precedent
  DL-11): a dynamic whole-file adoption that overwrites the anchor's
  `workflow/scripts/build_powerplants.py` with the live candidate copy at
  provision time — the two rule definitions are identical apart from the
  output path and the script is layout-agnostic — gated on the sentinel CTE
  `ges_latest`, guarded by a check that the PRISTINE anchor file (read via
  `git show e7f8bd70:`) lacks the sentinel and provides every
  `snakemake.input/params/output` key the candidate reads, and idempotent
  by content comparison so future drift re-applies and re-arms the forced
  rerun. The one-shot `.eq-force-rerun` writer is now merge-safe
  (`mark_force_rerun`) since two patches can queue rules in one provision.
  PUDL release is identical on both sides (v2025.2.0) and all five tracked
  rule inputs are byte-identical, so the query was the sole divergence.
  RESULT: prong 1 PASS, 0 live findings; solved objective rel
  2.34e-2 -> 2.1e-6 (full rebuild; 8.4e-8 incremental) and every carrier's
  `p_nom_opt` agrees to 0.01 MW — CCGT 12,563.33 and OCGT 8,434.51 on both
  sides, eliminating the 9.3%/11.4% split. Prong 2 PASS, 0 live / 3 total
  (all DL-9-class). SIDE FINDINGS: (i) the adoption also carries v1-epic's
  `set_derates` NaN->1.0 fill, a no-op once both sides rebuild from the
  same code; (ii) the candidate's own `powerplants.csv` was STALE with
  respect to its own script (built before the derate fix;
  `--rerun-triggers mtime` never noticed) — both sides regenerated; latent
  staleness of long-lived shared artifacts flagged for a harness guard;
  (iii) `build_powerplants` is not bit-reproducible: DuckDB `first()` and
  tied `array_agg(... ORDER BY report_date DESC)` picks perturb the
  weighted-mean imputations at <=1e-5 relative — the post-adoption
  cross-side residual (9,803 cells) is SMALLER than the same-side
  run-to-run residual (12,951 cells), i.e. parity is exact up to the
  code's own nondeterminism, >100x below the comparator's 1e-3 tolerance.
  Deterministic tie-breaks flagged as their own follow-up. | countersigned
  (ktehranchi, 2026-08-23) |

- **DL-7 re-scope (2026-08-23) — mechanism PARTLY disproven, bounds were
  stale and exceeded, scope now reduced to one column.** (1) SCOPE: the
  recorded root cause ("non-composable aggregation strategies, mean of
  means") never explained the `efficiency` and `marginal_cost` members of
  this class — those came from the SOURCE DATA (the `build_powerplants`
  join fan-out now signed as DL-12) and after the DL-12 adoption they
  produce ZERO findings at every stage in both prongs; their two waiver
  rows are deleted as dead. The mechanism IS correct for the surviving
  member: `generators.fuel_cost: mean` (plain, unweighted) is genuinely
  non-associative, so plants->sub->zone (anchor) differs from plants->zone
  (candidate). (2) BOUNDS: measured pre-DL-12, `fuel_cost` reached 8.4476
  $/MWh (3.3x the recorded 2.55), `efficiency` 0.004721 (1.7x) and
  `marginal_cost` 3.5141 $/MWh (1.4x); because these were stage-`'*'`
  value waivers they absorbed `marginal_cost` — the column that drives the
  objective — which is how a 2.34% objective error reached the solved
  stage with only two live findings. Any future waiver over a solver-input
  column should carry an explicit magnitude ceiling. RESTATED DL-7:
  clustered stages, prong 1 only: `Generator.fuel_cost` differs on 6 of 31
  clustered generators (6/45 at later stages), max 8.1277 $/MWh (40.24%)
  on `p10 oil` as of the post-DL-13 artifacts (8.4498/40.4% when first
  measured post-DL-12), with p9 oil 6.6% and p9/p10/p11 biomass <=10.8%.
  The DL-9 absolute gaps (3,680.1 / 3,586.6 MW) are invariant across
  DL-11, DL-12 AND DL-13 — the silent-drop mechanism is independent of
  all three adopted fixes. Accepted
  because the column is carried metadata: `marginal_cost`, `efficiency`
  and the solved objective are finding-free at 1e-3 and per-carrier
  `p_nom_opt` agrees to 0.01 MW. Prong 2 shows zero residuals. The
  original recommendation stands, narrower: switching `fuel_cost` (and
  `heat_rate`) to `capacity_weighted_average` would eliminate the class. |
  re-scoped 2026-08-23; restatement awaiting countersignature |

- **DL-9 recalibration post-DL-11/DL-12 (2026-08-23):** prong-2 residuals
  are now existing onwind 7,873.5 (cand) vs 4,193.4 MW (anchor)
  (+87.8% relative, gap 3,680.1 MW) and solar 25,033.5 vs 21,446.9
  (+16.7%, gap 3,586.6 MW); objective rel 21.2%. The SOLAR ABSOLUTE GAP IS
  UNCHANGED from the pre-DL-11 calibration (3,586.6 MW), proving DL-9's
  mechanism is intact and purely additive to DL-11's footprint scoping;
  the onwind gap shrank 4,418.6 -> 3,680.1 MW (738.5 MW of the old gap was
  out-of-footprint plants DL-11 removed). Percentages and the objective
  rel grew only because the CA-only base is ~5x smaller. NEW ISOLATION
  EVIDENCE: the anchor is insensitive to `{simpl}` — its prong-2 existing
  onwind/solar equal its prong-1 values exactly, while the candidate's
  move with cluster geometry; this cleanly isolates the issue-#16 silent
  drop to the geometry-dependent attach. All three prong-2 findings are
  DL-9-class and waived; magnitudes recorded here supersede the row's. |
  recalibrated 2026-08-23 |

- **DL-13 (scoped runs, all stages; ADOPTED on both sides): the
  `plants_must_add` seam-plant fallback bounded to the model footprint.**
  `filter_plants_by_region` unconditionally re-adds every plant outside all
  of the run's interconnect ReEDS shapes whose ReEDS membership disagrees
  with its EIA `interconnection` column — a guard so imprecise ReEDS shapes
  never delete a legitimate border plant. Since DL-11 the regions layers
  tile only the model footprint in scoped runs, so that add-back bypasses
  the now CA-sized region sjoin, and `match_plant_to_bus`'s second pass —
  which applies NO distance bound — attaches the survivors to the nearest
  in-footprint bus. This is DL-11's recorded KNOWN RESIDUAL, now measured
  exactly: 23 plants / 1,887.4 MW, every one >=890 km away (1,112.1 MW NM
  wind/solar, Buffalo Ridge II SD 210.0 MW, Hardy Hills Solar IN 195.0 MW
  at 2,508 km, Fort Peck MT hydro 162.4 MW over 4 units; DL-11's "~27 /
  1,890.6 MW" estimate superseded). FIX (v1-epic d98cb93f + 103f2194): in
  a footprint-scoped run keep only must-add plants within
  `SEAM_PLANT_MAX_KM` = 100 km of `regions_onshore` + `regions_offshore`
  in EPSG:5070; in-footprint plants are at distance 0 and always kept, so
  genuine near-seam plants still attach. `match_plant_to_bus` is
  deliberately left alone — its unbounded second pass is correct once the
  leak population is filtered upstream. Every drop logged at WARNING with
  name, carrier, state, MW, distance, plus a count/MW summary. GATE:
  applied only when `model_topology.include` is truthy, read from
  `snakemake.config` in `main()` and threaded as
  `filter_plants_by_region(footprint_scoped=...)`. With the gate off not
  one statement changes, so unfiltered interconnect/usa runs are
  byte-identical BY CONSTRUCTION — verified by evaluating the gate
  expression parsed from both sides' source against both configs
  (`{'reeds_state': ['CA']}`->True, `{}`->False). The gate is
  load-bearing, not cosmetic: against a full-western footprint the same
  population is mostly legitimate and an unconditional 100 km bound would
  delete 8 plants / 694.9 MW. USER DECISION 2026-08-23: fold into v1-epic
  AND mirror onto the anchor. THIRD ADOPTED-FIX anchor patch
  (`tests/equivalence/build.py::apply_seam_adoption`), and the first by
  targeted string surgery rather than DL-12's whole-file adoption, because
  v1-epic's `add_electricity.py` legitimately differs from the anchor's
  (simplify-early bus2sub/sub_id removal, the DL-1/DL-2 `length_factor=1.0`
  decision, schema logging). The whole `filter_plants_by_region` body is
  byte-identical between e7f8bd70 and v1-epic, so the anchor takes the
  same `footprint_scoped` plumbing; the constant and helper are sliced
  from the LIVE candidate file so both sides run the same text and drift
  re-applies. Rails: candidate sentinel, all four needles verified exactly
  once against the PRISTINE anchor file via `git show e7f8bd70:`, refusal
  if the pristine anchor already carries the sentinel, post-assembly
  checks for three sentinel occurrences and end-to-end wiring, idempotence
  BY CONTENT, and `mark_force_rerun(["add_electricity"])` on
  newly-applied. Verified AST-identical across sides in helper body,
  constant, signature, gated block and `main()` wiring. MEASURED EFFECT —
  SYMMETRIC, and larger than the pre-run estimate, which counted generator
  NAMES rather than capacity: 19 of the 23 plants / 1,725.0 MW do reach
  the assembled network (onwind 1,416.5, solar 281.5, oil 27.0), but only
  the oil plant is its own conventional generator — wind and solar
  capacity is folded onto per-bus atlite profile generators by
  `attach_renewable_capacities_to_atlite`, so the generator COUNT falls by
  1 while p_nom falls by 1,725.0 MW. The other 4 (Fort Peck hydro, 162.4
  MW) never reached the network: hydro is attached from the breakthrough
  base-grid files. Cross-check: pre-fix onwind (2,776.9 + 1,416.5 =
  4,193.4) and solar (21,165.4 + 281.5 = 21,446.9) reproduce DL-9's
  recorded anchor values exactly. Assembled stage, BOTH sides: existing
  p_nom 84,456.3 -> 82,731.3 MW, p10 oil 67.4 -> 40.4, onwind 4,193.4 ->
  2,776.9, solar 21,446.9 -> 21,165.4, generators 2,594 -> 2,593
  (candidate) and 1,771 -> 1,770 (anchor); max cross-side per-carrier
  residual 4.5e-13 MW. HARNESS RESULT: prong 1 PASS, 0 live / 72 total
  (finding classes identical to the DL-12 baseline), solved objective
  candidate -204,665,425.94 vs anchor -204,665,929.13, rel 2.46e-06,
  per-carrier `p_nom_opt` agreeing to 6e-4 MW; both sides moved together
  from ~-222,743,578.7 (+8.1%) as 1.7 GW of free existing renewables left
  and gas build rose (CCGT 12,563.3 -> 12,578.7, OCGT 8,434.5 -> 8,843.9
  MW). Prong 2 PASS, 0 live / 3 (all DL-9-class): onwind 6,457.0 vs
  2,776.9 and solar 24,752.0 vs 21,165.4, i.e. DL-9's absolute gaps
  3,680.1 and 3,586.6 MW are EXACTLY unchanged while both levels shift
  together — percentages rose (87.8%->132.5%, 16.7%->16.9%) only because
  the bases shrank. ROBUSTNESS DEFECT FOUND BY THE HARNESS (fixed in
  103f2194 before sign-off): the first implementation unioned the region
  layers before measuring distance and prong 2 died with `GEOSException:
  TopologyException: side location conflict` — the regions are 100% valid
  as stored in EPSG:4326, but reprojecting to EPSG:5070 leaves 9 of 29
  polygons self-intersecting or degenerate at simpl=20 (none at simpl=''),
  and GEOS `union_all` refuses invalid input. Since dist(p, U R) = min
  over R of dist(p, R), the union was replaced by a per-region minimum,
  robust to self-intersection; the two agree to 0.0 m on the simpl=''
  layer where the union works, and prong-1 numbers were bit-identical
  before and after. A regression test (self-intersecting bowtie +
  overlapping box) reproduces the GEOS failure against the union
  implementation. USA leg COMPLETE (2026-08-24):
  data-stage harness PASS, 0 live / 113 total findings — the identical
  count and class structure as the pre-DL-11/12/13 usa baseline, i.e. the
  three adopted fixes left the national comparison untouched. Gate
  verified on every axis: gate expression AST-identical on both sides and
  False for `include: {}`; candidate usa add_electricity log 0 seam-drop
  lines (vs 24 at CA); anchor usa log 0 lines. equivalence_report_usa.html
  regenerated, superseding the stale 2026-08-22 usa artifacts. |
  countersigned (ktehranchi, 2026-08-23) |

- **DL-15 (western, prong 1, all network stages; env-move re-baseline):
  pypsa 0.30 -> 1.3 SERIALIZATION AND OBJECTIVE-REPORTING CONVENTIONS.**
  (DL-14 is reserved by the `proto/nearshore-crs-fix` branch; this entry
  takes the next free id so the two merge without collision.) The
  candidate branch `claude/pypsa-v1-migration-data-storage-26840f` moves
  the stack to `pypsa==1.3.0` / `linopy==0.9.1` / `pandas==3.0.5` /
  `xarray==2026.7.0` while the anchor stays on `pypsa==0.30.2`, so the
  Tier-C comparison was re-baselined from scratch. CLEAN-BASELINE RESULT
  BEFORE ADJUDICATION: prong 1 FAIL, 16 live / 88 total; prong 2 PASS, 0
  live / 3 (unchanged, all DL-9-class). Every one of the 16 live findings
  falls into exactly three classes, each verified to carry zero physics
  content.
  (A) **`sub_network` topology metadata — 11 findings.** The candidate
  (pypsa v1) computes and serializes connected-component labels into
  `Bus.sub_network` ('0','1',...) and `Line.sub_network`, and exports
  `SubNetwork` component rows (7 at `assembled_substation_network`, 1 at
  each of `clustered_network` / `extra_components` / `prepared_network` /
  `sectored_network`); the 0.30 anchor wrote empty strings and no
  `SubNetwork` rows at all. Bus/Line diffs are total-population (1975/1975
  buses, 2811/2811 lines at assembled; 4/4 buses thereafter) precisely
  because one side is uniformly empty. `sub_network` is bookkeeping
  recomputed by `determine_network_topology` from Bus/Line/Link topology
  that is itself finding-free; the LOPF/expansion problem never reads it.
  (B) **`StorageUnit.cyclic_state_of_charge_per_period` — 4 findings**
  (clustered_network, extra_components, prepared_network,
  sectored_network; n=5 then 9 storage units). The candidate stores
  explicit `True` because the migration deliberately pins the pypsa-0.30
  default that v1 flipped True->False, preserving per-investment-period
  cyclicity. ANCHOR-FILE PROOF: the attribute IS NOT STORED IN THE ANCHOR
  NETCDF AT ALL — no `storage_units_cyclic_state_of_charge_per_period`
  data_var exists at any of the four stages (verified by opening each
  anchor file with xarray; only `storage_units_cyclic_state_of_charge` is
  present), because 0.30 omitted attributes left at their default. The
  harness loads BOTH sides under pypsa 1.3, whose default is now `False`,
  so the reader backfills `False` onto the anchor even though the anchor's
  actual solve-time behaviour was `True`. Both sides therefore behaved
  identically; the comparison manufactures the difference on read.
  (C) **`solved_network` `Network.objective` — 1 finding.** Candidate
  928,590,425.0093 with `objective_constant` 0.00; anchor
  -204,665,929.1272 with `objective_constant` 1,133,255,860.00. IDENTITY
  VERIFIED: candidate `objective` vs anchor `objective + objective_constant`
  gives 928,590,425.0093 vs 928,589,930.8728, relative difference
  **5.321e-07** — three orders of magnitude inside the 1e-3 objective gate
  and of the same order as the August baseline's own 6.7e-05 match. pypsa
  v1 / linopy 0.9 fold the fixed-cost offset into `objective` and leave
  `objective_constant` at 0; 0.30 reported the solver objective only and
  carried the offset separately.
  ADJUDICATION — two changes, deliberately different in kind.
  (1) CLASS C IS FIXED AS A COMPARATOR NORMALIZATION, NOT WAIVED:
  `compare.py::_compare_solved` now compares
  `objective + objective_constant` on both sides (each side's constant read
  from its own file, missing/NaN -> 0.0) via the new `_total_objective` /
  `_objective_constant` helpers, keeping the existing `OBJECTIVE_RTOL`
  (1e-3). This is a harness-correctness repair: the old raw-`objective`
  comparison was only valid while both sides used the same reporting
  convention, and would have silently mis-stated any future cross-version
  run. The prong-1 objective gate therefore stays LIVE — it is not
  suppressed, it is measured correctly. The finding detail now also carries
  `candidate_raw` / `anchor_raw` / `*_constant` for traceability. NOTE: the
  DL-9 prong-2 `solved_network` `Network.objective` waiver is unaffected
  (it suppresses a genuine physics difference, not this convention).
  (2) CLASSES A AND B ARE WAIVED. Class B was considered for normalization
  too — the correct generic rule is "a column absent from one side's file
  should be compared against that side's WRITING-era default, not the
  reading-era default" — but the harness has no clean hook for it:
  `compare_frames` sees fully-materialized DataFrames with defaults already
  backfilled, one compared pair is a dill pickle with no netCDF to
  interrogate, and a writing-era default table would have to be maintained
  per pypsa version per attribute. That is speculative machinery for a
  one-attribute problem, so the pragmatic targeted waiver was taken with
  the file-level proof recorded in the waiver's own `justification` field.
  SEVEN NEW WAIVERS in `tests/equivalence/waivers.yaml`, all
  `ledger: DL-15`: `Bus.sub_network` value, `Line.sub_network` value and
  `SubNetwork.<index>` row_set at `stage: '*'`; and
  `StorageUnit.cyclic_state_of_charge_per_period` value at each of the four
  named stages. All are scoped `interconnect: western` — the leg actually
  measured — and left unscoped by prong, which is symmetric but inert for
  prong 2 (whose only compared stages are `prong2_aggregates` and
  `solved_network`). KNOWN FOLLOW-UP: the usa leg's last baseline predates
  this environment move, so the same three classes must be re-verified —
  and these waivers extended to `interconnect: usa` — when usa is rebuilt
  on the 1.3 stack.
  RESULTS EFFECT: **None — all three are comparison artifacts of the
  pypsa 0.30 -> 1.3 serialization and reporting conventions.** Class A is
  metadata the optimizer never reads; class B is a default the anchor never
  stored and never behaved by; class C is the same total system cost under
  two accounting conventions, agreeing to 5.3e-07. No physical quantity
  moved: with the adjudication in place prong 1 is PASS, 0 live / **87**
  total findings, and prong 2 stays PASS, 0 live / 3 total. The total falls
  by one rather than staying at 88 precisely because class C was normalized
  rather than waived — the objective finding is no longer raised at all,
  whereas the 15 waived class-A/B findings are still counted and reported. |
  PENDING COUNTERSIGNATURE |

## Amendment (2026-08-28) — DL-16, harness recalibration after PyPSA#777

| DL-16 | all network stages (both prongs) | `Bus.load_weight` is a
  candidate-only column, and the `usa` leg compared two different demand
  pipelines | PyPSA#777 ("Population-based demand allocation") added a
  `load_weight` column that `build_base_network.py:54` writes on every Bus
  unconditionally, and made `population` the default `bus_allocation`. The
  anchor (`e7f8bd70`) has no census-population method at all. TWO SEPARATE
  DEFECTS followed. (1) `config.equivalence.yaml:77` pins
  `bus_allocation: breakthrough` so the western leg stays apples-to-apples,
  but `config.equivalence-usa.yaml` carried no such pin — and since
  `build.py` copies the candidate's config into the anchor worktree, BOTH
  sides ran unpinned. The candidate therefore weighted demand by 2020 census
  county population while the anchor weighted by Breakthrough `Pd`,
  diverging `Load_t.p_set`, `Bus.LAF_state`, the kmeans geometry
  (`cluster_network.py:68,82` weights on `load_weight`) and the objective by
  the full reallocation — every one of which would read as a migration
  regression. DL-13's recorded `usa` leg stopped reproducing. (2) The
  `load_weight` column survives every compared stage (`cols2drop` in
  `aggregate_to_substations.py:165-188` omits it; `cluster_network.py:269`
  sums it; `clean_bus_data` in `add_electricity.py:1058-1067` drops only
  `load_dissag`/`LAF`/`LAF_state`), so `compare.py:174-184` raised an
  unwaived `column_set` finding at five stages and `compare.py:546-549` set
  prong 1 to `pass: false` for a non-migration reason. | FIX, not a waiver
  of substance: the missing `bus_allocation: breakthrough` pin was added to
  `config.equivalence-usa.yaml`, restoring the like-for-like comparison. The
  column-presence finding is then waived (`Bus`/`load_weight`/`column_set`,
  stage `*`), and that waiver is only sound BECAUSE of the pin — with
  `breakthrough` pinned the column carries the anchor's own legacy `Pd`
  weighting, so it is numerically inert. Demand content remains guarded by
  the UNWAIVED `Load` / `Load_t.p_set` comparisons, and `load_weight`
  *value* findings are deliberately NOT waived (verified: the waiver matches
  `kind: column_set` only). | PENDING COUNTERSIGNATURE |

Note: both defects were present on `develop` from the moment PyPSA#777
merged; they were surfaced by the adversarial review of PyPSA#778 but are
not caused by it. Neither has been exercised against a live harness run —
the Tier-C re-baseline that PyPSA#778 requires is still outstanding, and
this amendment does not discharge it.
