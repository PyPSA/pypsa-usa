# CA Equivalence Harness — Implementation Plan (Phase 2–4 of the master spec)

> Executes `docs/superpowers/specs/2026-08-07-pipeline-equivalence-and-perf-design.md`.
> Orchestrated autonomously per user mandate 2026-08-07: build the harness, run both
> sides locally, debug via agents, document everything, benchmark, then PR
> `v1-epic` → `develop`. Steps use checkbox syntax for tracking.

**Goal:** A runnable Tier C harness that builds the CA-only pipeline on both
`v1-epic` (candidate, in-repo) and the anchor `upstream/develop@e7f8bd70`
(baseline, in a git worktree), compares artifacts stage-by-stage under the
config-only two-prong protocol, emits a self-contained HTML visual report and
a benchmark table, and populates the deltas ledger.

**Architecture:** New package `tests/equivalence/` with four modules and a CLI.
No changes to workflow rules or scripts except bug fixes discovered by runs
(each documented in `docs/CHANGELOG-v1-epic.md`). The anchor side may receive
**build-infrastructure patches only** (documented; never results-affecting).

**Tech stack:** Python 3.11, pypsa 0.30.2, xarray, pandas 2.2.2, dill,
geopandas, matplotlib 3.8 (PNGs base64-embedded in a single HTML file — no
external assets), pytest marker `equivalence`, snakemake 7.32.4 invoked with
`--scheduler greedy` (pattern: `tests/integration/conftest.py:129-152`).

---

## Verified facts the design rests on (from the 5-reader research workflow)

1. **Prefixes:** `run.name: equivalence` → both branches write
   `resources/equivalence/` + `results/equivalence/` (Snakefile:95-103 both).
   RESULTS-tree paths are string-identical across branches; the divergence is
   inside resources/ (v1 category dirs + `_s{simpl}` keys vs anchor
   `western/` flat + `Geospatial/`).
2. **Prong-1 pairing table** (candidate ↔ anchor), EQ=resources/equivalence:
   - demand: `EQ/demand/western/power_electricity_s.csv` ↔ `EQ/western/demand/power_electricity.csv`
   - profiles: `EQ/profiles/western/2030/profile_{onwind,solar}_s.nc` ↔ `EQ/western/2030/profile_{onwind,solar}.nc`
   - assembled substation network: `EQ/networks/western/elec_s_l_pp.pkl` (dill) ↔ `EQ/western/elec_s.nc` (pypsa)
     — the anchor's `elec_s.nc` is its *simplify_network* output (full network,
     substation level); its own `elec_base_network_l_pp.pkl` is **nodal** and has
     no candidate counterpart.
   - clustered: `EQ/networks/western/elec_s_c10.nc` ↔ `EQ/western/elec_s_c10.nc`
   - `_ec`, prepared `_lv1.0_REM-3h`, sectored `_E`: same filenames both sides
   - solved: `results/equivalence/western/networks/elec_s_c10_ec_lv1.0_REM-3h_E.nc` (identical path)
   - candidate-only (no anchor counterpart, skip): `elec_b.nc`, `busmap_b.csv`, `busmap_s.csv`, `elec_s.nc` (topology-only), `elec_s_dem.nc`
   - anchor-only (skip): `elec_base_network_dem.nc`, `elec_base_network_l_pp.pkl`
3. **simpl='' is safe end-to-end on both branches** — no unguarded
   `int(simpl)`; filenames unambiguous (wildcard excludes `_`); candidate
   identity branch writes identity busmap; anchor else-branch skips
   investment-periods/truncation but anchor `cluster_network.py:988` sets
   periods unconditionally. CA slice ≈ 1975 substations, 4 ReEDS zones →
   `distribute_clusters` bounds hold for clusters=10.
4. **Shared config feasibility:** one `config.equivalence.yaml` drives both
   branches iff it carries (for the anchor): `co2.storage` (parse-time read,
   anchor build_electricity.smk:894), `clustering.cluster_network.feature`
   (anchor cluster_network.py:1165), `clustering.simplify_network.feature`
   (prong 2 only, anchor simplify_network.py:294), plus branch-common:
   non-null `renewable_land_access` (godeeep hard-requires),
   `electricity.demand.*`, `clustering.*.algorithm`. v1-only keys ship in
   v1's tracked config.common.yaml; the anchor ignores extras.
5. **Anchor worktree needs:** symlink `workflow/data` (+ `workflow/cutouts`)
   from main checkout (both gitignored, untracked); copy the four gitignored
   layered configs from anchor `workflow/repo_data/config/` into
   `workflow/config/` (they are `configfile:`-loaded unconditionally);
   `touch` `data/costs/caiso_ng_power_prices.csv` (tracked-input mtime would
   retrigger `retrieve_caiso_data`); **infra patch**: anchor `common.smk`
   has the `constants` source-cache import bug (upstream #764) — apply the
   same `workflow.source_path("../scripts/constants.py")` fix we shipped on
   v1-epic (commit e43fa927). Infra-only; documented in change-log.
6. **Loaders:** networks `pypsa.Network(path)`; candidate assembled network
   via `dill.load`; profiles `xr.open_dataset` (vars: profile(time,bus),
   weight, p_nom_max, potential, average_distance(bus));
   demand CSV `pd.read_csv(index_col=0)` (columns = bus IDs, values rounded
   to 4 dp at write — build_demand.py:2649). Full-network comparison walks
   `n.iterate_components()` static `.df` + `.pnl` timeseries (pattern:
   prepare_network.py:172-176).
7. **Instrumentation:** snakemake `benchmark:` TSVs land in
   `workflow/benchmarks/equivalence/` (except `cluster_network`'s hard-coded
   shared path `benchmarks/cluster_network/...` — read from there); columns
   `s h:m:s max_rss ...`; max_rss reads 0 on macOS (note in report). File
   sizes via `os.stat` walk of both resources trees (no existing helper).
8. **Report tooling:** matplotlib 3.8 + stdlib base64 (no jinja2 dep to add;
   f-string HTML). Reusable shapes: `plot_timeseries_comparison`
   (plot_validation_production.py:81), paired-bar
   (`plot_bar_carrier_production`, :151), duration-curve kernel
   (plot_statistics.py:1049-1050).

---

## File structure

- **Create** `workflow/repo_data/config/config.equivalence.yaml` — shared
  harness config (both prongs; prong selected by `--config scenario={...}`
  override from the driver).
- **Create** `tests/equivalence/__init__.py`
- **Create** `tests/equivalence/paths.py` — `ArtifactPair` dataclass + the
  prong-aware candidate↔anchor path map (fact 2), loader-kind tags.
- **Create** `tests/equivalence/build.py` — `build_side(side, prong, until)`:
  candidate = snakemake in main checkout; anchor = worktree provisioning
  (fact 5) + snakemake there. Captures per-rule benchmarks + file sizes into
  `results/equivalence/manifest_{side}_{prong}.json` (anchor SHA, config
  hash, wall times, sizes).
- **Create** `tests/equivalence/compare.py` — loads each pair, applies D2
  (floats `allclose(rtol=1e-3, atol=1e-8, equal_nan=True)`; indexes/dtypes/
  strings exact after sorting; candidate-vs-anchor index alignment), D7 at
  solve stage (objective 0.1%, per-carrier capacity 0.5%), waiver filtering
  from `tests/equivalence/waivers.yaml`; writes
  `results/equivalence/findings_{prong}.json`.
- **Create** `tests/equivalence/report.py` — one self-contained HTML per
  prong: summary pass/fail table, demand overlays, profile duration curves,
  capacity-by-carrier bars, per-bus scatters, worst-offender tables, waiver
  list, benchmark table (candidate vs anchor wall time + sizes).
- **Create** `tests/equivalence/run.py` — CLI:
  `python -m tests.equivalence.run --prong {1,2} [--until STAGE] [--skip-solve]`.
- **Create** `tests/equivalence/test_equivalence.py` — pytest wrapper,
  marker `equivalence` (register in pyproject; exclude from `fast`).
- **Create** `tests/equivalence/waivers.yaml` — starts empty.
- **Modify** `pyproject.toml` — register `equivalence` marker.
- **Create on first delta** `docs/superpowers/specs/2026-08-07-deltas-ledger.md`.
- **Modify continuously** `docs/CHANGELOG-v1-epic.md`.

Comparison behavior details (D2/D3 refinements):
- Sort both sides' component indexes before compare; report set differences
  (missing/extra rows) as first-class deltas, then value deltas on the
  intersection.
- Column union: anchor-only columns and candidate-only columns are reported
  as schema deltas (waivable), not errors.
- Known-immaterial normalizations applied silently: dtype object-vs-string,
  netCDF float32 round-trip via the tolerance itself, empty-vs-absent `_t`
  DataFrames.
- Prong 2 compares only aggregate invariants: annual demand per state,
  p_nom (existing) per carrier per ReEDS zone pre-solve; objective +
  per-carrier optimal capacity post-solve.

## Task list

- [ ] **T1 config**: write `config.equivalence.yaml` (CA slice, Jan 6–19
  2019, clusters 10, prong-1 simpl [''], anchor-required keys from fact 4).
  Dry-run both prongs' DAGs on v1-epic resolve.
- [ ] **T2 paths+build**: implement paths.py + build.py; candidate prong-1
  build completes locally `--until add_sectors`-equivalent target.
- [ ] **T3 anchor build**: worktree provisioning + anchor prong-1 build
  completes (expect bugs → debug agents; document every fix).
- [ ] **T4 compare+report**: implement compare.py/report.py/run.py/tests;
  prong-1 comparison runs end-to-end, report renders.
- [ ] **T5 solve + prong 2**: solve both sides (Gurobi, seeded), run prong-2
  builds + aggregate comparison.
- [ ] **T6 reconcile (Phase 3)**: adjudicate findings — fix candidate bugs /
  provisional waivers with ledger entries (sign-off pending user).
- [ ] **T7 benchmark + docs**: benchmark table into change-log + report;
  update spec/ledger; Tier A+B green.
- [ ] **T8 PR**: push, open PR `v1-epic` → `develop` with report + ledger +
  change-log evidence.
