# The equivalence harness: `master-benchmark` vs `develop`

How the benchmark compares the refactored branch against the baseline, how the
two sides avoid clobbering each other, how to run it on Sherlock, and how to
read what comes out.

Everything here lives in `tests/equivalence/`. The prose home for *why* a
difference exists is the hot-fix ledger in the project brain
(`memory/plans/hotfix-ledger.md`); `tests/equivalence/hotfixes.yaml` is its
machine-readable extract.

---

## 1. The three branches

| name | what it is |
|---|---|
| `master` | the released baseline. Never built directly. |
| `master-benchmark` | `master` **plus** the commits it needs in order to run the benchmark config at all and to be compared like-for-like. This is what actually builds the baseline side. |
| `develop` | the refactored branch under test. Builds in the main checkout. |

The baseline is a **branch, not a patched checkout**. Earlier versions of this
harness rewrote the baseline's source at provisioning time — string surgery with
sentinels, plus a marker file to defeat snakemake's mtime rerun triggers — which
meant the baseline that actually ran was not a commit anyone could check out.
Now it is a sha. `BENCHMARK-BRANCH.md` at the root of `master-benchmark` is that
branch's own manifest, and `tests/equivalence/test_master_benchmark_branch.py`
checks that the manifest, the commit trailers and the hot-fix registry all
agree.

### How a hot-fix lands on the baseline

A **hot-fix** is a change on `develop` that is not part of the core
restructuring — a data fix, a bug fix, a changed default — and that could by
itself move a benchmark number. Each one has a row `HF-<n>` in the ledger and in
`tests/equivalence/hotfixes.yaml`.

Some of them have to be on the baseline too, either because `master` cannot run
the config without them or because we want both sides compared on the fixed
behaviour. Adding one is three steps and no shortcuts:

1. **Commit it on `master-benchmark`**, one commit per ported item, never a
   squash, with these trailers:

   ```
   port(HF-8): pre-aggregate EIA-860 SCD tables before the LEFT JOINs

   <why master needs this to run or to compare — two or three lines>

   Category: comparability
   Hot-fix: HF-8
   Ported-from: 01f742f
   ```

   The category is one of `runnability` (master cannot execute the config
   without it), `comparability` (a develop fix adopted so the two sides compare
   like-for-like — this one *may* change numbers, by design) or `resources`
   (`mem_mb` / `walltime` / `threads` / `benchmark:` only, and a test enforces
   that such a commit touches nothing else).

2. **Regenerate `BENCHMARK-BRANCH.md`** so its table lists exactly the commits
   in `master..master-benchmark`, in order.

3. **Mark it ported in both ledgers** — `ported: true` plus `ported_sha:` in
   `tests/equivalence/hotfixes.yaml`, and a "ported to master-benchmark:
   `<sha>`" note on the row in `memory/plans/hotfix-ledger.md`.

Step 3 is the one that matters most. A ported hot-fix runs on **both** sides, so
it can no longer explain a difference between them. The harness enforces this:
citing a ported id as the explanation for an over-tolerance row is rejected with
a message saying so. If a difference really does trace to a ported row, the port
is broken — that is a finding, not an explanation.

When `master` moves, `master-benchmark` is **rebased** onto it, never merged.
The shas change and `run_meta.json` records the new ones.

---

## 2. How the two sides avoid clobbering each other

### Code: two git worktrees of one repository

`develop` builds in the main checkout. The baseline builds in
`.worktrees/master-benchmark`, checked out at the resolved `master-benchmark`
sha. Two worktrees means two separate `workflow/resources/`, `workflow/results/`,
`workflow/benchmarks/` and `.snakemake/` trees, so the two DAGs cannot see or
lock each other's artifacts.

Provisioning (`build.provision_baseline_worktree`) **refuses** a
`.worktrees/master-benchmark` directory that git does not list as a registered
worktree. A directory like that survives a move of the checkout and carries a
`.git` file pointing at *another* repository's gitdir; adopting it would run git
against the wrong repo.

Both sides also refuse to build when their tracked files differ from their
commit, because the sha the manifest records would then not describe the code
that ran. `EQ_ALLOW_DIRTY=1` builds anyway and records `dirty: true` with the
offending paths.

### Environment: one venv per worktree

Each worktree has its own `pyproject.toml` and `uv.lock`, so `uv run` inside it
resolves its own `.venv`:

| side | pypsa | linopy | pandas | python |
|---|---|---|---|---|
| `master-benchmark` | 0.30.2 | 0.3.14 | 2.2.2 | 3.11 |
| `develop` | 1.3.0 | 0.9.1 | 3.0.5 | ≥3.11 |

The resolved versions per side go into `run_meta.json`. `compare.py`,
`metrics.py` and `plots.py` all run in the **develop** environment and read both
sides' files, so some differences are artifacts of reading a pypsa-0.30 file
with pypsa 1.3 — those are the DL-15 class and are waived in `waivers.yaml`,
each with the proof that the two sides behaved identically and the comparison
manufactured the difference on read.

### Data: shared, symlinked, never copied

`workflow/data` and `workflow/cutouts` in the develop checkout are symlinks into
the `$GROUP_SCRATCH` cache; the baseline worktree's copies are symlinks to the
develop checkout's. Thirteen gigabytes, read-only in practice, one copy.

### Sequential, one node

Both sides build **in one SLURM job, one after the other**, on `-p serc`. Never
two concurrent jobs against a shared `data/` cache: the `retrieve_*` rules would
race on the same download targets. Per-rule job submission is also deliberately
not used for equivalence runs — it would put the two arms in different
environments and add scheduler nondeterminism to the thing being measured.

**Never on the login node.**

---

## 3. "The same config" is a checked statement

The two branches do not load the same set of config files. `master`'s Snakefile
loads `config/config.{cluster,common,plotting,api,sector}.yaml` with
`config.default.yaml` commented out; `develop` layers the tracked
`repo_data/config/*` set **including `config.default.yaml` as a base**, then
user overlays, then validates the result against a schema. So on `develop` an
omitted key resolves to the shipped default, and on `master` to a script's
inline fallback — or it raises. That is hot-fix HF-20, and the merge replay that
justified it ("no key changes value") was only ever run develop-vs-develop.

Two consequences:

- **`config.equivalence*.yaml` is deliberately self-contained.** It is not a
  sparse overlay like `config.tutorial.yaml`. Any key dropped from it would be
  supplied to develop by the default layer and missing on master.
- **A config-equivalence gate runs before either build.**
  `context.assert_config_equivalent()` dumps each side's fully merged config
  through that side's own Snakefile — replayed, never re-implemented here — and
  aborts on any differing key that is not on `context.CONFIG_DIFF_ALLOWLIST`
  with a signed reason. Running this gate *is* the replay HF-20 still owes.

Two values are pinned identically in both configs so they stop being
differences at all:

```yaml
pudl_path: s3://pudl.catalyst.coop/v2026.8.0   # newest release on develop
electricity:
  honor_planned_retirements: true
```

Snakemake applies `--configfile` **after** every `configfile:` directive, so
these win on both branches. Note that the `honor_planned_retirements` pin alone
was inert on the baseline — `master` carried no code reading the key — which is
why HF-3 was also ported as a commit.

The gate splits differences on whether the two sides can take different
branches, not on whether a key is present:

| kind | meaning | severity |
|---|---|---|
| `value` | both sides carry the key and disagree | **fatal** |
| `default_only` | present on one side only, with a **truthy** value | **fatal** |
| `only_master` / `only_develop` | present on one side only, with a **falsy** value (`null`, `{}`, `[]`, `""`, `0`, `false`) | recorded |

The falsy/truthy split is the whole point, and it is not a nicety.
`config.get(key, {})` followed by `if cfg:` is the shape all over this workflow:
a falsy default-layer value takes the same branch as an absent key, and a truthy
one does not. `electricity.demand_response: {marginal_cost: 999999, shift: 0}`
is the case that proved it — develop-only, and truthy, so develop calls
`add_demand_response`, which adds a `demand_response` **Carrier** before its own
`shift == 0` early return, while the baseline's `.get(..., {})` adds nothing.
Three stages of unwaived Carrier `row_set` findings, from a key that looks inert.
It is now pinned on both sides.

Measured on the USA config on 2026-09-14: 3 `value`, 46 `default_only`, 4 falsy
presence-only. Every one of the 49 fatal ones is either pinned in the shared
config or carries a signed line in `CONFIG_DIFF_ALLOWLIST` saying what was
checked — dead keys nothing reads, blocks whose own `enable` is false, opts that
this run does not select, solver option sets it does not use, per-rule
`walltime` declarations the scheduler reads and no script does.

Where an entry is safe because of its **value** and not just its key, it also
has a guard in `_ALLOWLIST_GUARDS`, and an allowlisted key whose value moves out
from under its reason stops being allowlisted. `costs.atb` is allowed only while
develop's defaults still equal master's inline fallback; the `weighting_strategy`
keys only while neither side says `population`; the default-off blocks only while
they are off; the `walltime` blocks only while they contain nothing but
walltimes.

`EQ_STRICT_CONFIG_GATE=1` makes the falsy presence-only differences fatal too.
That is how HF-20's owed master-vs-develop replay gets discharged in full.

One normalisation is built in: `null`, `{}` and `[]` compare equal. The shared
config writes `model_topology.include: {}`, which arrives as `None` on develop
(snakemake's merge writes nothing for an empty-mapping update) and `{}` on
master. Both are falsy, so the HF-9 / HF-10 scoping gates evaluate `False` on
both sides. A *populated* include on one side still differs, which is the case
that matters.

When the gate does fail it lists **every** offending key with both values at
once. Each one is a decision: pin it identically in the shared config, or add it
to `CONFIG_DIFF_ALLOWLIST` with a reason someone signed. An allowlist grown by
copy-pasting the failure message defeats the gate.

`EQ_SKIP_CONFIG_GATE=1` skips the check entirely. It records that it was skipped
in `run_meta.json`, so a run made without the gate can never be mistaken for one
made with it.

---

## 4. Running it on Sherlock

One driver, `tests/equivalence/run_equivalence.sbatch`, parameterised entirely
by environment variables. Submit **from the repository root** — it resolves the
checkout from `$SLURM_SUBMIT_DIR` and contains no repository path of its own.

```bash
cd /path/to/pypsa-usa
sbatch tests/equivalence/run_equivalence.sbatch
```

| knob | default | what it does |
|---|---|---|
| `EQ_INTERCONNECT` | `usa` | footprint; selects `config.equivalence-<ic>.yaml` |
| `EQ_PRONG` | `2` | 1 = `simpl=''` pass-through, 2 = `simpl=$EQ_SIMPL` |
| `EQ_SIMPL` | `300` | prong-2 granularity |
| `EQ_CLUSTERS` | `134` | develop-dialect `{clusters}`; translated for the baseline |
| `EQ_OPTS` | `3h` | the emissions-unconstrained twin. `REM-3h` (national 1 Mt CO2 cap) is **opt-in** — see below |
| `EQ_UNTIL` | *(unset)* | `assembled` stops before prepare/solve |
| `EQ_BASELINE_REF` | `master-benchmark` | the baseline branch |
| `EQ_RUN_ID` | minted | names the run directory |
| `EQ_JOBS` | `8` | snakemake `-j` per side |
| `EQ_TIMEOUT` | derived | per-side wall-clock cap, seconds. Unset, it is computed from the job's own remaining time less a 900 s tail, split across the two sides, so a flat value can never outlive `--time` and get side one killed mid-rule |
| `EQ_CACHE` | `$GROUP_SCRATCH/kamran/pypsa-usa-eq-cache` | the shared `data/` + `cutouts/` cache |
| `EQ_STAGE_SRC` | *(unset)* | optional rsync top-up of the cache from another checkout |
| `EQ_ALLOW_DIRTY` | *(unset)* | `1` builds a dirty checkout and records it |
| `EQ_SKIP_CONFIG_GATE` | *(unset)* | `1` skips the config gate and records that |
| `EQ_STRICT_CONFIG_GATE` | *(unset)* | `1` makes presence-only config differences fatal too |

The driver loads `gcc/12.4.0` (GCC 14 hard-errors on the datrie source build),
`system git/2.45.1` (system git 1.8 predates `git worktree`) and `devel uv`;
redirects `UV_CACHE_DIR`, `PIP_CACHE_DIR`, `XDG_CACHE_HOME` and `MPLCONFIGDIR`
off `$HOME`; exports `GRB_LICENSE_FILE` explicitly; refuses to replace a
`workflow/data` that is not already a symlink; verifies the cached
`nrel_exclusion` and GODEEEP inputs are national rather than a leftover regional
footprint; clears stale snakemake locks on both sides; and installs a trap that
kills its children and `scancel`s any tagged Slurm job on **any** exit path,
including the pre-timeout `TERM`.

### Why `EQ_OPTS` defaults to the unconstrained twin

`3h` is the emissions-unconstrained run. The national 1 Mt CO2 cap (`REM-3h`)
looked **infeasible at USA scale** on 2026-09-01 — a barrier infeasibility
verdict plus a stalled disambiguation simplex — and a failed solve aborts the
whole harness before the comparison is written. Running the unconstrained twin
first secures the master build, the comparison and the benchmark data before the
REM question is settled. `paths.OPTS` carries the same default; the two must
agree, or bash and python would mint different run ids for the same run.

### Starting from an empty cache

The three footprint-dependent directories — `nrel_exclusion/`, `godeeep/`,
`zenodo/` — are **excluded from `EQ_STAGE_SRC` even when it is set**, so a cold
`EQ_CACHE` is not filled by staging. It does not need to be: the workflow's own
`retrieve_*` rules fetch them from the Zenodo records declared in
`config.equivalence-usa.yaml`, and the driver detects the cold cache, says so,
and continues. Expect the first run to be substantially longer.

What the driver *will* refuse is an input that is **present and too small**.
That is the distinction that matters: absent means "not fetched yet"; a 68 KB
`caps_onwind_reference.nc` where a 400 KB+ national one belongs means a regional
footprint is sitting in the national cache, which builds ERCOT profiles on both
sides and looks like a clean comparison. Delete it and let the retrieve rules
refetch, or restage a verified copy.

Everything else — the ~13 GB of `data/` and `cutouts/` that is not
footprint-dependent — either comes from `EQ_STAGE_SRC` (which covers both
directories) or is retrieved by the workflow.

### Recovering from a killed job

The driver traps `TERM` (including Slurm's pre-timeout signal, 180 s before the
wall) and kills the run's whole **process group**, so snakemake's rule processes
die with it rather than running on until the node is reclaimed. That is CH2's
missing piece: one of its sbatch scripts exists only to repair the two networks a
controller killed mid-solve deleted.

A job that dies without the trap firing (an OOM kill, a node failure) leaves
snakemake metadata marking the in-flight outputs incomplete. `build.py` passes
`--rerun-incomplete`, so the next run simply rebuilds them; the run directory
means a resumed build picks up where the dead one stopped. If snakemake still
refuses a specific file, clear just that file's metadata by hand rather than
wiping the tree:

```bash
cd workflow                      # or .worktrees/master-benchmark/workflow
uv run snakemake --cleanup-metadata resources/equivalence/networks/usa/elec_s300.nc
```

Stale directory locks from a killed predecessor are swept automatically at job
start (`snakemake --unlock` on both sides).

### `EQ_STAGE_SRC` is off for a reason

The footprint-dependent directories (`nrel_exclusion/`, `godeeep/`, `zenodo/`)
in the cache are provenance-verified against published Zenodo records. A blanket
sync from another campaign's `data/` once imported Texas-only `nrel_exclusion`
artifacts into a USA run, and both sides happily built ERCOT profiles. Those
directories stay excluded even when `EQ_STAGE_SRC` is set.

### The fast loop

For a development loop that does not pay for the solve:

```bash
EQ_UNTIL=assembled EQ_OPTS=3h sbatch tests/equivalence/run_equivalence.sbatch
```

This builds both sides through `build_renewable_profiles` / demand /
`add_electricity` (develop) and `simplify_network` (master) and compares the
demand and profile artifacts — enough to ground the capacity-factor and demand
comparison without a full-year solve. Locally, `--skip-solve` on `run.py` does
the same for the final target.

For a single interactive check, `sh_dev -c 4 -m 16GB -t 02:00:00 -p serc` and
then `uv run --extra dev pytest -m fast tests/equivalence -q`. Never run
snakemake or a solve on the login node.

---

## 5. What a run produces

One directory per run, `workflow/results/equivalence/<run_id>/`:

```
run_meta.json          provenance: three shas, both environments, the config gate's verdict
manifest_master.json   baseline build: sha, config hash, per-rule wall time and max_rss
manifest_develop.json  develop build, same shape
findings_<prong>.json  every stage-by-stage difference, each marked waived or live
config_merged/         the two fully merged configs the gate compared
tables/comparison.csv  one row per (metric, key) with a verdict and its candidates
tables/comparison.md   the same, capped, for reading
figures/*.png          every figure...
figures/*.csv          ...each with the exact values it plots
logs/driver-<job>.log  what the driver did
```

### `run_meta.json`

`run_meta.json` is written **before** the gate runs and rewritten after it, so
a run the gate refuses still leaves provenance behind: `config_gate` reads
`pending` / `passed` / `failed` / `skipped`, and `config_gate_error` carries the
refusal. A run with no record is a run nobody can account for.

Three shas, not one: `master_sha` (the branch point),
`baseline_sha` (what actually built the baseline, plus how many commits it sits
above `master`) and `develop_sha` (plus `develop_dirty`). Also both sides'
resolved package versions, the config sha256, the `{clusters}` value **each
side actually ran**, the allowed config differences with their reasons, the
known code differences, the SLURM job id and the timestamp.

`clusters_develop` and `clusters_baseline` differ on purpose. Harness commit
`8371e9d` changed `{clusters}` semantics on develop — a plain integer now
aggregates conventional carriers only, and a trailing `s` restores all-carrier
aggregation. `master` does not carry that commit, so the harness translates the
value; without the translation the two sides would build different generator
sets and the comparison would be meaningless.

### At prong 2, master's profiles are rolled up first

The two branches do not build renewable profiles at the same resolution. Master
writes `profile_{tech}.nc` keyed by **substation** (western: 544 onwind, 808
solar buses); develop writes `profile_{tech}_s{simpl}.nc` keyed by **cluster**
(19 and 20). Every profile statistic is a property of that bus population as
much as of the weather — pooling `(time, bus)` values from 544 sites and from
the 19 clusters they roll up into gives different quantiles no matter what the
refactor did. Compared as they sit, the `p_max_pu_quantiles_*` rows measure the
clustering, and they cannot come out equivalent even in principle.

So at prong 2 the harness passes master's file through
`metrics.aggregate_profile_to_clusters` with develop's own
`busmap_s{simpl}.csv` *before* any profile metric is taken: `profile` becomes
the `p_nom_max`-weighted mean per cluster, `p_nom_max` / `potential` / `weight`
are summed, `average_distance` is a capacity-weighted mean. `p_nom_max`
weighting is what makes `sum_bus(profile * p_nom_max)` — the national
available-power series — exactly invariant under the rollup, so the system
aggregate means the same thing before and after. Master's buses are then cluster
ids, so its side of the zone-keyed metrics joins through develop's
cluster→`reeds_zone` map rather than the substation→zone chain. A bus missing
from the busmap is dropped with a logged count, never silently — and "missing"
is strict: only the float-formatted integer form (`'35827.0'` for `35827`) is
reconciled, so `'35827.5'` and `'035827'` are counted as drops rather than
truncated onto a real bus. A cluster whose total weight is zero gets a **NaN**
profile, not 0.0, so it contributes nothing to the pooled statistics — exactly
as develop's own all-NaN clusters do. Prong 1 compares two nodal files
bus-for-bus and is untouched.

#### A cluster on one side only is a row-set difference

Same resolution is not the same population. On the western leg the rollup leaves
master with 18 onwind / 19 solar clusters while develop has 19 / 20: `p87 0`
(96 MW of onwind, 2,158 MW of solar — HF-24's bus 37808) exists only on develop,
because master silently dropped the substation it is made of. Pooled in, that
one cluster put a whole 8,760-hour column into develop's quantile pool and
nothing into master's, which moved the onwind quantile deltas from
−0.09 / +0.51 / 0.00 % to −7.98 / −2.73 / +1.33 % (p25/p50/p95): a missing
cluster reading as a capacity-factor difference.

So after the rollup the harness compares the two cluster sets and emits one
finding — `stage: profile_{tech}`, `component: cluster_set`, `column: <index>`,
`kind: row_set` — whose detail names every one-sided cluster with its
`p_nom_max` in MW. The pooled metrics (`p_max_pu_quantiles_*`,
`mean_cf_by_zone_*`, `p_nom_max_by_zone_*`) are then computed over the
**common** clusters only. The one-sided capacity is not lost: it is in that
finding, and in the clustering-invariant `system_potential_mw` /
`system_available_mw` aggregates, which are still taken over everything each
side built.

`run_meta.json` records which object the numbers came from, in
`master_profile_stage`, and what the rollup found, in `profile_cluster_sets`.
The stage is derived from what actually happened, not from the prong:
`build_context` runs before either side builds and can only say the rollup is
planned; the comparison rewrites the field with `nodal->s{simpl}
(p_nom_max-weighted)` when the rollup ran, and `nodal (rollup did NOT run: ...)`
when there was no busmap to run it with.

### A waiver is bounded in sign and magnitude

A TABLE waiver in `tests/equivalence/waivers.yaml` may carry two optional
fields:

| field | meaning |
|---|---|
| `expect_sign` | `'+'` or `'-'`: the sign of `develop − master` the waiver was written for |
| `max_abs_pct` | upper bound on the row's `abs(delta %)` |

Without them a waiver is a blank cheque. The HF-24 entry on
`p_nom_max_by_zone_solar/p8` was written for "+2,158 MW of potential that master
silently dropped" (+1.1 %); matched on metric and key alone it would equally
have explained a −50 % row — the opposite of the fix's known direction — or a
10,000× one. A row outside the bounds is reported in the hot-fix column as
`waiver HF-n bounds violated (sign|magnitude)` and stays **UNEXPLAINED**. An
undefined `delta %` (master 0, develop non-zero) fails `max_abs_pct` too: the
bound cannot be shown to hold. The shipped HF-24 waivers carry
`expect_sign: '+'` and `max_abs_pct: 5`, against measured rows of +0.19 % to
+1.15 %.

Bounds belong on table waivers only — a cell waiver has no delta to bound, and
`test_waiver_bounds_are_well_formed` fails one that carries them. A waiver that
names `metric`, `key` or `family` **at all**, wildcard included, is a table
waiver and is never read as a cell waiver: `{metric: '*', key: '*', prong: 2}`
names no cell field, so read as one it would waive every finding in the run.

### Reading `comparison.md`

One row per (metric, key), with `master`, `develop`, `delta`, `delta_pct`, the
tolerance, a verdict, a `hotfix` cell and a `candidates` cell:

- **`equivalent`** — within the relative tolerance or under the absolute floor.
  Nothing to do.
- **`explained`** — over tolerance, and a **waiver names this row** with a
  `hotfix:` tag that holds up. The id is in the `hotfix` cell; look it up in the
  ledger.
- **`UNEXPLAINED`** — over tolerance with no such waiver. **This is the verdict
  that matters.** Rows sort `UNEXPLAINED` first, then by `|delta_pct|`
  descending.
- **`MISSING`** — the metric could not be computed at all. A criterion vanishing
  from the table is worse than a difference, so it fails the run too.
- **`one-sided` / `undefined`** — a ratio metric defined on one side or neither.
  Reported, not failed; the additive metric beside it carries the substance.

A run exits non-zero while any `UNEXPLAINED` or `MISSING` row remains.

### How to attribute a difference

Only a waiver can turn `UNEXPLAINED` into `explained`. That is deliberate, and
it is a change from how this started. The registry's `expect` globs used to
grant the verdict directly — but in the metric-name vocabulary they cover **all
13 known metrics** between them, so every over-tolerance row found some claimant
and `UNEXPLAINED` became unreachable. A safety net that catches everything is
not a safety net.

So `expect` is now **advisory**. It fills the `candidates` column: the hot-fix
ids that *claim* they could move this row, offered as a starting point. The
column is unfiltered on purpose — a `ported` or `usa_noop` id appearing there is
worth seeing, because it means the port, or the no-op claim, is the thing to go
and check.

Attributing a difference is therefore a decision with a name on it:

1. Read the row. Note its `candidates`.
2. Work out which change actually produced it — in the ledger, in the diff, by
   rebuilding one stage.
3. Record the answer as a waiver in `tests/equivalence/waivers.yaml` naming the
   row and carrying the id:

   ```yaml
   - interconnect: usa        # optional; omit for "any run"
     prong: 2                 # optional; omit for "any run"
     metric: dispatch_by_carrier
     key: CCGT
     ledger: DL-15
     hotfix: HF-14
     reason: <one line, and the evidence>
   ```

   **Naming the row.** Prefer `metric:` + `key:` — one row, one decision. A
   waiver must name at least one of `metric` / `key` / `family`; one that names
   none is a `compare.py` cell waiver and has nothing to say here.

   `family:` on its own is a **blanket exception over a whole tolerance family**
   — every carrier, every zone, every quantile in it at once — not an ordinary
   way to write a waiver. Reach for it only when the cause genuinely acts on the
   family as a whole (a unit convention, a stack-wide reporting change), say so
   in `reason:`, and expect to be asked why the narrower form would not do. A
   `family:` waiver will keep explaining rows that appear long after it was
   written, including ones nobody has looked at.

   **Scoping the run.** `interconnect:` and `prong:` are optional and mean "any
   run" when absent, but a waiver that names either only applies when it
   matches. That is what keeps a waiver written for the deferred western prong-1
   leg from signing off a whole-USA prong-2 difference it never saw. Scope
   anything whose justification is footprint- or granularity-specific — HF-9 and
   HF-10, for instance, are active on the western leg and inert on USA.

The tag has to hold up: the id must resolve in `hotfixes.yaml`, must not be
`ported: true` (a fix on `master-benchmark` runs on both sides, so it cannot be
why they differ — see §1) and must not be `usa_noop: true` (it did nothing on
this run at all). A rejected tag is reported verbatim in the `hotfix` cell and
the row stays `UNEXPLAINED`.

`expect` is still linted: a pattern that could never name a row this harness
produces is dead configuration and fails the registry test.

One row deserves a direct look every time: the objective. `master` runs pypsa
0.30, which reports `Network.objective` as the solver objective only and carries
the fixed-cost offset in `objective_constant`; `develop` runs pypsa 1.3, which
folds the offset into `objective` and leaves `objective_constant` at 0. Both
sides are normalised to `objective + objective_constant` before comparison. The
table carries `master_raw` and `develop_raw` alongside — they should differ,
visibly, while `delta_pct` sits at zero. That is HF-13's normalisation being
applied rather than assumed; comparing the raw objectives manufactures a
difference exactly the size of the constant.

### Reading the figures

Every PNG in `figures/` has a sibling CSV with exactly the values it plots — a
PNG without its CSV is a bug, and there is a test for it. Plot a number you
cannot trace and you have made a picture, not a result.

Zone maps join on the `reeds_zone` **bus attribute**, never on string surgery
over cluster names: cluster ids are `p{zone}{subcluster} {i}` with no separator
(`"p101 1"` is zone `p10`, sub-cluster 1), so any prefix split produces labels
that match almost nothing and the first report's maps came out mostly grey.

### Findings vs table rows

`findings_<prong>.json` is the fine-grained, per-component comparison: every
frame, every column. It is noisy by design and most of it is waived in
`waivers.yaml`, each waiver naming a `DL-<n>` row in the deltas ledger and,
where one applies, the `HF-<n>` that causes it. The comparison table is the
coarse, decision-grade view. A run passes only when **both** are clean: zero
unwaived findings and zero `UNEXPLAINED` or `MISSING` rows.

## 6. One press: `submit_benchmark.sh`

A harness change is not done until a job is in `squeue --me`. The queue wait
is the review window, not something to earn by reviewing first. To make the
launch one decidable action:

```bash
tests/equivalence/submit_benchmark.sh              # smoke, then full USA run chained afterok
tests/equivalence/submit_benchmark.sh --smoke-only
tests/equivalence/submit_benchmark.sh --full-only  # when the smoke has already passed
tests/equivalence/submit_benchmark.sh --dry-run
```

Stage 1 builds `western` on both sides, prong 2, `EQ_UNTIL=assembled`, on
8 CPUs / 64 GB / 3 h, with `--verdict-exit report`: its comparison verdict is
printed and tabulated but does not fail the job, so a known western difference
cannot cancel the chained run. Stage 2 is the driver with its defaults (whole USA),
released by Slurm only if stage 1 exits 0 (`--dependency=afterok`). Both
jobs carry `--mail-type=END,FAIL`. The script prints `key=value` facts
(job ids, shas, run directories, Slurm log paths) that the brain's `/benchmark`
skill turns into a run-card stub. Every knob is an `EQ_SMOKE_*` / `EQ_FULL_*`
env var; see the header.

Waiting: check `squeue --me` every 20 minutes at most, or let Slurm mail you.
Never end a session with "shall I launch?" when submission is already
permitted: launch, then review while it queues.
