# `master-benchmark` — the equivalence baseline

## What this branch is

`master-benchmark` is the **baseline** the equivalence harness compares
`develop` against. It is cut from `master` (branch point `fbe5883f`) and carries,
as ordinary reviewable commits, everything `master` needs in order to *run* the
benchmark config and to be compared against `develop` *like for like*.

It replaces the old build-time patching machinery in
`tests/equivalence/build.py` (`apply_infra_patches`,
`apply_adopted_fix_patches`, `apply_powerplants_adoption`,
`apply_seam_adoption`, `apply_leap_day_adoption`), which rewrote a pinned
upstream worktree's source at provision time and needed a `.eq-force-rerun`
marker to defeat snakemake's `--rerun-triggers mtime`. The baseline that
actually ran was not a commit anyone could check out. Now it is.

Three refs name the benchmark:

- **`master`** — upstream baseline, untouched.
- **`master-benchmark`** — this branch: `master` + the commits below. **This is
  what the baseline side of every benchmark actually builds.**
- **`develop`** — the refactored branch under test.

## Rules for adding a commit here

1. **One commit per ported item.** Never a squash, never a rollup.
2. **Every commit carries trailers:**

   ```
   Category: runnability | comparability | resources
   Hot-fix: HF-<n>            # required for `comparability`
   Ported-from: <sha on develop or the harness branch>
   ```

3. **Three categories, and nothing else is admissible:**

   | category | meaning | may change numbers |
   |---|---|---|
   | `runnability` | `master` cannot execute the benchmark config without it | only insofar as it makes the run exist |
   | `comparability` | a develop fix adopted onto the baseline so the two sides compare like-for-like | **yes, by design** |
   | `resources` | `mem_mb` / `walltime` / `threads` / `benchmark:` directives so the case runs on Sherlock | **no — enforced by test** |

   A `resources` commit that touches anything other than resource/benchmark
   declarations is a bug; `tests/equivalence/test_master_benchmark_branch.py`
   enforces it.
4. **Every ported commit gets a `ported to master-benchmark: <sha>` mark on its
   row in `memory/plans/hotfix-ledger.md`** (project brain) and `ported: true`
   in `tests/equivalence/hotfixes.yaml`. A hot-fix marked ported is **excluded**
   from the candidate explanations for a residual master-vs-develop difference:
   its effect is on *both* sides. If a difference traces to a ported row, the
   port is broken — that is a finding, not an explanation.
5. **Rebase, never merge.** When `master` moves, `master-benchmark` is rebased
   onto it, the shas change, and `run_meta.json` records the new ones. This
   table must be regenerated when that happens.
6. The branch is **local** until Kamran asks for it to be pushed — `origin` is
   `PyPSA/pypsa-usa`.

## Commits

Branch point: `master` = `fbe5883f` ("Remove stray directory named with a single
space (#789)"). Newest last. One row per **ported item**; the commit that adds
or updates this file is itself excluded (it cannot carry its own sha), and it is
the only commit in `master..master-benchmark` without a `Hot-fix:` trailer.

| # | sha | category | hot-fix | what |
|---|---|---|---|---|
| 1 | `f65c90c` | runnability | HF-23 | `common.smk` materialises `constants.py` into the snakemake source cache, so its import resolves under a fresh cache (upstream #764) |
| 2 | `572534f` | runnability | HF-22 | `_drop_leap_day` on the renewable CF-selection window. **Inert on master** — `_helpers.get_snapshots` already defaults to `drop_leap_day=True`, so the 2012 window is 8,760 h before this ever runs. Ported for text parity with the harness branch only |
| 3 | `c10d60e` | comparability | HF-1 | `data_year` 2023 → 2025 in `build_powerplants.py` — the EIA-923 window is a hard-coded constant, so pinning `pudl_path` does not pin it |
| 4 | `93751fe` | comparability | HF-8 | adopt develop's `build_powerplants.py` wholesale: EIA-860 SCD tables pre-aggregated before the LEFT JOINs (DL-12) |
| 5 | `bce8437` | comparability | HF-9 | empty-county sweep scoped to the `model_topology.include` footprint (DL-11); no-op on the whole-USA config |
| 6 | `d4dd4ff` | comparability | HF-10 | must-add seam-plant fallback bounded to 100 km of the model footprint (DL-13); no-op on the whole-USA config |
| 7 | `b3afe50` | comparability | HF-3 | `load_powerplants` honours `electricity.honor_planned_retirements`; master had no code reading the key, so a config pin alone could not neutralise it |
| 8 | `969ae8b0` | comparability | HF-29 | `dissagregate_demand` renormalises the load allocation factor inside the demand key it consumes (state demand conserved; was +2.28 % on this branch), folds a DC demand column into Maryland unless a bus carries the key. The `state`-vs-`reeds_state` key difference stays on develop as HF-28 |

### Read this before attributing a difference

Commit 4 is a **wholesale file adoption**, exactly as the retired build-time
patch was. It therefore carries onto the baseline more than HF-8:

| also carried by commit 4 | ledger row |
|---|---|
| `data_year = 2025` (already set by commit 3) | HF-1 (`data_year` half) |
| `nerc_region` imputed from state for recent-vintage EIA-860 plants | HF-2 |
| `planned_generator_retirement_date` = latest filing as-is (a newer NULL = withdrawn announcement) | HF-4 |
| missing EIA-860 summer/winter derates filled `1.0` instead of NaN | HF-11 |
| `sanitize_uc_parameters` / `_clamp_series` — **NOT inert with unit commitment off** | HF-17 (part) |

All of those are on **both** sides of the benchmark and are marked ported in the
ledger. None of them may be used to explain a master-vs-develop difference.

`sanitize_uc_parameters` deserves its own warning. It is often described as
inert while `conventional.unit_commitment: false`, and that is wrong: it clamps
and fills `ramp_limit_up` / `ramp_limit_down` (among `min_up_time`,
`min_down_time`, `start_up_cost`) in `powerplants.csv`, and
`attach_conventional_generators` passes those to `n.add` unconditionally,
committable or not. They also feed
`clustering.aggregation_strategies.generators` (see HF-16). So **baseline
dispatch differs from bare `master`'s**. That is neutral for master-vs-develop —
both sides run it — but it is *not* neutral for master-vs-`master-benchmark`,
and any statement that this branch "only makes master runnable" is false.

HF-4 is load-bearing for commit 7: the retirement dates commit 7 honours are the
ones commit 4's query produces.

## What is NOT here, deliberately

- `pudl_path` (`s3://pudl.catalyst.coop/v2026.8.0`) and
  `electricity.honor_planned_retirements: true` are **config pins**, applied to
  both sides by the shared harness config
  (`workflow/repo_data/config/config.equivalence-usa.yaml`) via `--configfile`,
  which snakemake applies after every `configfile:` directive. They need no
  commit here.

  **⚠ Neither pin exists yet.** They are T4 of
  `memory/plans/harness-master-vs-develop.md` and are not in the harness config
  as of this branch. Until they land, the baseline loads master's own tracked
  `workflow/config/config.common.yaml` value `pudl_path:
  s3://pudl.catalyst.coop/v2025.2.0` while commit 3 asks for `data_year = 2025`
  — a release whose EIA-923 coverage ends before 2025, so the heat-rate /
  fuel-cost window is **empty**. `honor_planned_retirements` is unaffected in
  practice (commit 7 defaults it to `True`), but it is likewise unpinned. **Do
  not read a benchmark run off this branch until both pins are in the harness
  config.**
- The PyPSA 0.30 → 1.3 / pandas 2 → 3 stack move (HF-12/HF-13/HF-14) cannot be
  back-ported: it *is* the refactor. The baseline builds in its own `uv`
  environment (pypsa 0.30.2 / pandas 2.2.2 / python 3.11) from this worktree's
  own `pyproject.toml` / `uv.lock`.
- HF-5, HF-6, HF-7, HF-15, HF-16, HF-19, HF-20 are live develop-only
  differences and remain candidate explanations.
