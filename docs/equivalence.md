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

The gate distinguishes two kinds of difference, because they are not the same
thing:

- a **value** difference — both sides carry the key and disagree about it — is
  **fatal**. That is a real disagreement about what to compute. Measured on the
  USA config today there are four, and every one of them is either pinned or
  allowlisted with a reason.
- a **presence-only** difference — one side's default layer supplies a key the
  other's does not — is **recorded** in `run_meta.json`'s
  `config_diff_allowed`, with the reason, and does not abort. There are well
  over a hundred of these (develop's `config.default.yaml` base layer against
  master's `config.{common,plotting,slurm}.yaml`), covering plotting, sector,
  DAC and unused solver option sets. Aborting on them would mean the gate could
  never pass and would simply be switched off, which is strictly worse than
  putting the whole list on the record.

`EQ_STRICT_CONFIG_GATE=1` makes presence-only differences fatal too. That is how
HF-20's owed master-vs-develop replay actually gets discharged: run it once,
triage the list, and either pin the keys in the shared config or move them into
`CONFIG_DIFF_ALLOWLIST` with signed reasons.

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
| `EQ_OPTS` | `3h` | `3h` is the unconstrained twin; `REM-3h` adds the national 1 Mt CO2 cap |
| `EQ_UNTIL` | *(unset)* | `assembled` stops before prepare/solve |
| `EQ_BASELINE_REF` | `master-benchmark` | the baseline branch |
| `EQ_RUN_ID` | minted | names the run directory |
| `EQ_JOBS` | `8` | snakemake `-j` per side |
| `EQ_TIMEOUT` | `43200` | per-side wall-clock cap, seconds |
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
tables/comparison.csv  one row per (metric, key) with a verdict
tables/comparison.md   the same, capped, for reading
figures/*.png          every figure...
figures/*.csv          ...each with the exact values it plots
logs/driver-<job>.log  what the driver did
```

### `run_meta.json`

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

### Reading `comparison.md`

One row per (metric, key), with `master`, `develop`, `delta`, `delta_pct`, the
tolerance, a verdict and a hot-fix id. Three verdicts:

- **`equivalent`** — `|delta_pct|` is within tolerance. Nothing to do.
- **`explained`** — over tolerance, and a live hot-fix id accounts for it. The
  id is in the row; look it up in the ledger.
- **`UNEXPLAINED`** — over tolerance with no hot-fix that accounts for it.
  **This is the only verdict that matters.** Rows are sorted `UNEXPLAINED`
  first, then by `|delta_pct|` descending. A run exits non-zero if there is even
  one, and a refactor change is accepted only when there are none or when each
  one traces to a named, un-ported hot-fix.

A hot-fix marked `ported: true` is **not** accepted as an explanation, for the
reason in §1.

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
unwaived findings and zero `UNEXPLAINED` rows.
