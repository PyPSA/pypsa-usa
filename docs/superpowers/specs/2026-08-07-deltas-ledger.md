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
