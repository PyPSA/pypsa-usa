# Pipeline Evolution with Results-Equivalence Guarantee — Master Work Plan

**Date:** 2026-08-07
**Status:** Implemented 2026-08-07 — CA harness GREEN on both prongs (prong 1: 0 live/305, objective rel 6.6e-05; prong 2: 0 live/3, DL-9). See Implementation outcomes addendum.
**Author:** ktehranchi (with Claude)
**Vocabulary:** all capitalized terms defined in [`CONTEXT.md`](../../../CONTEXT.md)

## Goal

Evolve the PyPSA-USA pipeline so the pre-`solve_network` stages are fast and
memory-lean, under one hard constraint: **model data and results must be
equivalent to upstream `develop`**, with every intentional difference
explicitly signed off. The constraint is enforced by a Tier C equivalence
harness, not by promises.

This spec subsumes and sequences the four existing initiatives
(testing-strategy, network-schema-tracking, cluster-simpl county fast-path,
config-cleanup — see `docs/superpowers/specs/2026-05-21-*.md`) and the
already-merged simplify-early refactor (PRs #7–#12).

## Scope

**In:** electricity path (`build_base_network` → `prepare_network` →
`solve_network`); the CA harness; the change-log and deltas ledger;
profiling-driven performance work.

**Out (deferred, revisit after the CA harness is green):**
- The USA/HPC harness. Design is sketched in "USA harness (deferred)" below;
  no implementation now.
- Sector-coupling path (`add_sectors`, gas/heat). Inherits the wins; gets
  harness coverage later.
- Memory/wall-time *targets*. We instrument (file sizes, max RSS, wall time
  per rule) and report changes; we do not gate on thresholds.
- Per-PR CI gating on Tier C. Harness runs are manual/local until green once;
  the long-term home is a CI gate on `develop` → `master` promotion merges.

## Decisions (from the grilling session)

| # | Decision |
|---|----------|
| D1 | **Anchor** = upstream `develop` at `e7f8bd70` (2026-07-22), pinned after Phase 1 sync. Moving the anchor is an explicit, recorded event. |
| D2 | Pre-solve float comparison: `np.allclose(rtol=1e-3)`; indexes, dtypes, integers, strings compare exactly. |
| D3 | Comparison is stage-by-stage at every major artifact, to localize regressions to the producing rule. |
| D4 | Every equivalence run emits a self-contained HTML **visual report** (see "Visual reports"). |
| D5 | CA harness config: CA-only Western slice, two weeks of January (2019-01-06 → 2019-01-19), 2019 weather year. |
| D6 | Harness cadence: manual/local now → CI gate for `develop`→`master` merges once green. |
| D7 | Solve stage: objective within 0.1%, per-carrier capacities within 0.5%. |
| D8 | Accepted deltas: markdown **deltas ledger** (human-signed) + machine-readable **waivers** consumed by the comparison. 1:1 correspondence enforced. |
| D9 | Upstream sync: land test-stack PRs #20→#16→#17 first, then **merge** (not rebase) `upstream/develop` into `v1-epic`; create fork `develop` tracking upstream. |
| D10 | **Two-prong protocol** (below): config-only determinism — no pinned fixtures, no anchor-side patches (revised per user 2026-08-07). *Amended 2026-08-23:* two narrow, documented anchor-patch categories now exist — BUILD-INFRA (cannot change numbers; the #764 constants source-cache fix) and ADOPTED-FIX (changes numbers by design, applied identically to both sides after user countersignature; first instance DL-11, footprint-scoped empty-county sweep). Both live in `tests/equivalence/build.py` and are marker-idempotent. |
| D11 | Visual report ships the standard plot set (extensible). |
| D12 | Performance: profile-first. No pre-committed hit-list beyond county fast-path wiring and the `custom_busmap` repair. |
| D13 | Baseline builds are recomputed on demand (cheap at CA scale); no baseline artifacts or busmap fixtures in git. |

## Equivalence methodology

### Why prong 1 skips the `{simpl}` stage

The simplify-early refactor intentionally moved `{simpl}` kmeans ahead of
demand/RE construction and changed its weighting. Cluster assignments
therefore differ from the anchor **by design**, and per-bus artifacts are not
directly comparable in a production (`simpl=N`) configuration.

### The two-prong protocol (D10, revised 2026-08-07)

No fixtures, no patches to either side: equivalence runs use **basic
configuration options only**. This works because `cluster_network`'s busmap
generation is shared, unchanged code on both branches and its kmeans is
seeded (`random_state=0` default, both branches) — deterministic given
identical inputs. Stage-by-stage comparison (D3) guarantees inputs are
verified identical before the clustering stage is judged.

**Prong 1 — exact comparison.** Both sides run the identical config with
`simpl=""` (pass-through — supported on both branches), so both enter
`cluster_network` at substation granularity with identical bus IDs, and the
seeded, shared clustering code produces identical busmaps. Every downstream
artifact — demand, profiles, capacities, line parameters, the prepared
network — must match within D2 tolerances. Any delta is a real behavior
change. `simpl=""` is used because the `{simpl}`-stage kmeans is the one
place the branches intentionally differ (weighting moved to static `Pd`);
skipping it isolates the data-assembly path the constraint protects.

**Prong 2 — aggregate invariants.** Both sides run production-style
(`simpl=N`, live simpl-stage kmeans). Compare only clustering-invariant
quantities: annual demand per state (within D2), total capacity per carrier
per REEDS zone, system totals. This validates the clustering relocation
itself.

*Note:* the vestigial `custom_busmap` wiring (broken under pandas ≥ 2.0 on
both branches) is no longer on the critical path; it becomes an ordinary
cleanup candidate.

### Comparison points (D3)

| Stage artifact | Prong 1 assertion |
|---|---|
| demand CSVs / `elec_*_dem` | per-bus `p_set` timeseries |
| `profile_{tech}*.nc` | per-bus profiles, p_nom_max |
| `add_electricity` output | generator set, `p_nom`, marginal/capital costs per bus+carrier |
| `elec_*_c{clusters}.nc` | bus/line/link sets, impedances, capacities |
| `prepare_network` output | full component frames |
| solved network | objective (0.1%), per-carrier capacity (0.5%) |

The comparison library is schema-aware: it reads the network-schema catalog
(`docs/network-schema.md`) for per-column semantics, and reads waivers to skip
exactly the accepted deltas.

### Visual reports (D4, D11)

One self-contained HTML file per equivalence run:
- demand timeseries overlays (baseline vs candidate; per state and system
  total);
- RE profile duration curves per technology;
- capacity-by-carrier bar charts with delta annotations;
- per-bus scatter plots (baseline vs candidate, 1:1 line) for `p_set` means,
  `p_nom`, and marginal cost;
- a summary table: per-stage pass/fail, worst offender columns, active
  waivers.

### Deltas ledger and waivers (D8)

- Ledger: `docs/superpowers/specs/2026-08-07-deltas-ledger.md` — one row per
  accepted delta: description, root cause, justification, sign-off (user +
  date).
- Waivers: a `waivers.yaml` consumed by the comparison library — stage,
  component, column, scope, ledger reference. The harness fails if a waiver
  has no ledger row or vice versa.
- The change-log (`docs/CHANGELOG-v1-epic.md`) records *code* changes; the
  ledger records accepted *result* differences. A change-log entry claiming
  "Results effect: None" is a claim the harness enforces.

## CA harness

### Config (draft — for user confirmation, D5)

`workflow/repo_data/config/config.equivalence.yaml`, derived from
`config.test.yaml` (PR #20) with these deltas:

```yaml
run:
  name: "equivalence"
scenario:
  interconnect: [western]
  simpl: ['']          # prong 1: pass-through; prong 2 variant uses [20]
  clusters: [10]       # seeded kmeans in cluster_network (random_state=0, both branches)
  opts: [REM-3h]
  ll: [v1.0]
  planning_horizons: [2030]
model_topology:
  include:
    reeds_state: ['CA']
snapshots:
  start: "2019-01-06 00:00"   # two full Sun–Sat January weeks
  end: "2019-01-19 23:00"
  inclusive: both
renewable_weather_years: [2019]
```

All other keys inherit `config.test.yaml`'s choices (efs reference demand,
no unit commitment, economic retirement). The actual file lands in Phase 2
(it depends on #20 merging) and is presented for your modification before the
first equivalence run.

### Driver

`tests/equivalence/` contains:
- `run_baseline.py` — checks out the anchor SHA into a git worktree and
  builds artifacts into a dedicated run dir (no patches to the anchor).
  Reuses the artifacts if the (anchor SHA, config hash, data-bundle) manifest
  matches (D13).
- `compare.py` — the schema-aware artifact-diff library (also importable by
  future CI).
- `report.py` — visual report generation.
- pytest wrapper marked `equivalence` (excluded from `fast`/`integration`
  runs).

### Instrumentation (no targets)

Every harness run logs, per rule: wall time, max RSS (Linux only — macOS
reports 0), and output file sizes, appended to a small CSV so the
develop-vs-v1-epic change is documented over time.

## Phases

- **Phase 0 — land what's queued.** Merge #20 → #16 → #17 (re-targeting each
  to `v1-epic` as its predecessor lands). Commit the four working-tree
  bugfixes on a branch (they get their equivalence verdict in Phase 3).
  Push local `v1-epic`.
- **Phase 1 — sync with upstream (D9).** Create fork `develop` from
  `upstream/develop`. Merge `upstream/develop` (`e7f8bd70`) into `v1-epic` —
  one merge commit; the main conflict surface is upstream's intact
  `simplify_network` vs our split rules, plus upstream PRs #745–#766
  (pumped hydro, EER demand, StorageUnit max_hours, transmission costs).
  Pin the anchor. Tier A/B must pass post-merge.
- **Phase 2 — build the CA harness.** Comparison library + visual report;
  equivalence config (user confirms); baseline-build driver. Exit: both
  prongs run end-to-end and produce a report (red is expected).
- **Phase 3 — reconcile.** Work the delta list: each is fixed (restores
  parity) or waived (ledger row + waiver, user signs). The four bugfixes are
  adjudicated here. Exit: **CA harness green** — the constraint is now
  enforced, not promised.
- **Phase 4 — performance, harness-gated (D12).** Profiling pass (py-spy /
  memray) on the CA and Western builds; wire the county fast-path dispatch
  (helpers exist, unwired); then the profile-driven hit-list. Every perf PR:
  harness green + instrumentation delta recorded + change-log entry.
- **Phase 5 — promote (deferred trigger).** When the harness is green and
  stable: wire it as the CI gate for `develop`→`master` promotion (D6), and
  revisit the deferred items (USA harness, sector coverage, upstreaming to
  PyPSA/pypsa-usa — coordinate with upstream #762, the PyPSA v1 bump).

## USA harness (deferred — design sketch only)

Same config-only two-prong protocol and comparison library at
`interconnect: usa` scale;
baseline built once per anchor on HPC scratch with a manifest (anchor SHA,
config hash, data-bundle versions, checksums); rerun only at milestones or
when the anchor moves. No implementation until the CA harness is green.

## Success criteria

1. CA harness green: prong 1 within D2/D7, prong 2 invariants hold, zero
   unexplained deltas (every delta fixed or signed in the ledger).
2. Every merged PR from Phase 3 onward carries a change-log entry, and the
   harness stays green.
3. An instrumentation record (wall time, max RSS, file sizes per rule)
   documents the develop-vs-v1-epic change on the CA and Western builds.
4. `v1-epic` carries the merged upstream `develop` history (no divergence).

## Risks

| Risk | Mitigation |
|---|---|
| Phase 1 merge is large and conflict-heavy (simplify split vs upstream changes) | Do it right after the small test-stack PRs land; Tier A/B as the post-merge gate; timebox and fall back to re-applying v1-epic changes onto develop as fresh commits if the merge is unmanageable. |
| `simpl=""` path is rarely exercised and may itself be buggy on either branch | It's prong 1's foundation — smoke-test it first on both sides before building the comparison on top. |
| Seeded kmeans still differs across branches due to library-version drift in sklearn | environment.yaml is byte-identical (verified); the manifest records the resolved env so any drift is detectable. |
| Zenodo data-bundle drift between baseline and candidate builds | Manifest records bundle versions; both sides build from the same `data/` cache. |
| kmeans nondeterminism contaminates prong 2 | Prong 2 only asserts clustering-invariant aggregates; seeds pinned where exposed. |
| Restructuring scope creep ("improve the code first") | Guardrail: only restructure code the harness or a profiled hotspot touches. Docs are written as-you-touch (schema catalog, change-log). |

## Environment facts (verified 2026-08-07)

- `workflow/envs/environment.yaml` is byte-identical between `v1-epic` and
  `upstream/develop` (pypsa 0.30.2, atlite 0.3.0, linopy 0.3.14,
  pandas 2.2.2, snakemake 7.32.4, python 3.11.9): one env runs both sides.
- Both branches support `simpl=""` pass-through; `cluster_network` kmeans is
  seeded on both (`random_state=0`); both have vestigial, broken
  `custom_busmap` wiring (pandas ≥ 2.0 `squeeze` removal) — not used by the
  harness.
- Anchor-side pipeline shape: single `simplify_network` rule; profiles built
  at substation level, demand at nodal level, both pre-clustering.


## Implementation outcomes (2026-08-07 addendum)

- Config-only determinism (D10) held, with one refinement: the reeds
  transport-model path requires `clusters` = the footprint's ReEDS zone
  count (CA -> `4m`), and its zonal busmap is pure attribute membership —
  no kmeans anywhere in prong 1.
- The harness caught four real defects during bringup, all fixed and
  change-logged: AEO scenario-case mismatch (latent upstream, config fix);
  `simpl=''` region-name normalization (empty profiles); double-applied
  `length_factor` (25% line-CAPEX inflation in line-preserving configs);
  hydro busmap remap (12.8 GW dropped); plus the demand-conservation fix in
  `remove_transformers` (-6.28% silent demand loss). PR #21's four earlier
  bugfixes were adjudicated by the same runs.
- Prong 1: PASS. 305 findings, 0 live; 8 ledger entries DL-1..DL-8
  (provisional, awaiting user countersignature); solved objectives agree at
  rel 6.6e-05 against the 1e-3 gate.
- CA-slice per-rule wall times (candidate vs anchor, seconds):
  power_build_demand 139 vs 217; solar profiles 39 vs 61; onwind profiles
  34 vs 42; add_electricity 13 vs 15; add_demand 3.9 vs 5.5; cluster 7.2
  vs 10.2; solve 33.7 vs 32.3 (identical LP, as equivalence requires).
  max RSS not measurable on macOS (0 in benchmarks) — Linux/HPC runs
  needed for memory numbers, per the deferred USA-harness phase.
