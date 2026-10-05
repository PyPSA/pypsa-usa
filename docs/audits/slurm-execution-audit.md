# Slurm / HPC execution audit — PyPSA-USA on Stanford Sherlock

**Date:** 2026-09-06
**Repo:** `/oak/stanford/groups/iazevedo/kamran/refactor/pypsa-usa`
**Branches inspected:** `feat/equivalence-usa-benchmark` (checked out) vs `origin/develop`
**Method:** static read of tracked + untracked files only. Nothing was executed on the login node
(no `python`, `snakemake`, `pip`, `sinfo`, `sh_part`). Every claim that would require running
something is marked **[UNVERIFIED]** in §4.

---

## 0. Executive summary

The repo ships exactly one supported HPC entry point, `workflow/run_slurm.sh`: a single
`snakemake --cluster "sbatch ..." --cluster-config ...` line. **As shipped it cannot run.** It
references a configfile that does not exist in the repo, a `--cluster-config` file that is not
seeded in this checkout, emits `-A` with an empty value, formats `{resources.walltime}` and
`{resources.mem_mb}` for ~25 rules that declare neither, has no `--cluster-status` (so a
Slurm-killed job hangs the scheduler indefinitely), and runs the multi-day Snakemake scheduler
process on the shared login node.

Meanwhile the *actual* HPC work on this branch is done by three hand-rolled `sbatch` scripts that
bypass `run_slurm.sh` entirely and run Snakemake monolithically inside one big job. That divergence
— documented path broken, real path undocumented and duplicated three times — is the core finding.

A second, orthogonal failure runs underneath all of this: resource *sizing*. Requests are flat
constants or `input.size` byte heuristics that carry no information about model size, and against
the first real USA-scale measurements they are wrong by 26x in the OOM-killing direction for two
rules and by ~56x in the wasteful direction for another. That is §5 (upstream issues #808 / #811).

**Recommendation:** adopt a tracked **Snakemake profile directory** (`workflow/profiles/sherlock/`)
plus a thin **sbatch driver** that runs the scheduler on a compute node. Do it in two phases:
Phase 1 under the currently pinned Snakemake 7.32.4 using `--cluster` + `--cluster-status` driven
entirely from the profile (no `--cluster-config`); Phase 2 bumps to Snakemake 8 +
`snakemake-executor-plugin-slurm`, at which point the profile shape is unchanged and only the
resource *names* move (`walltime` string → `runtime` minutes). Keep a `profiles/sherlock-monolithic/`
as the documented one-node fallback, because the equivalence harness genuinely needs it.

---

## 1. Inventory: everything in the repo that touches Slurm/HPC

### 1.1 The documented launcher — `workflow/run_slurm.sh` (3 lines, identical on `origin/develop`)

```
# SLURM specifications live in config/config.slurm.yaml & the individual rules
# GRB_LICENSE_FILE=/share/software/user/restricted/gurobi/11.0.2/licenses/gurobi.lic
snakemake --cluster "sbatch -A {cluster.account} --mail-type ALL --mail-user {cluster.email}  -p {cluster.partition} -o {cluster.output} -e {cluster.error} -c {threads} --mem {resources.mem_mb} --time {resources.walltime}" --cluster-config config/config.slurm.yaml --jobs 20 --latency-wait 60 --rerun-incomplete --configfile config/CH1/config.tamu.single_horizon.bau.yaml
```

Note the trailing invisible character (U+2060 WORD JOINER) at the end of the commented
`GRB_LICENSE_FILE` line — harmless while commented, a silent breakage if anyone uncomments it.

### 1.2 The scheduler config — `workflow/repo_data/config/config.slurm.yaml` (61 lines, identical on `origin/develop`)

Two unrelated things in one file, by the file's own admission:

* `__default__:` — `account`, `partition`, `email` (all blank in the template), `walltime: 00:30:00`,
  `cpus_per_task: 1`, `chdir:` (blank), `output: logs/{rule}/log-%j.out`,
  `error: logs/{rule}/errlog-%j.err`. Consumed **only** via `--cluster-config`.
  `walltime`, `cpus_per_task` and `chdir` under `__default__` are **dead** — `run_slurm.sh`
  never interpolates `{cluster.walltime}`, `{cluster.cpus_per_task}` or `{cluster.chdir}`.
* `walltime:` — 10 per-rule wall times (`build_renewable_profiles: 04:00:00` …
  `solve_network: 20:00:00`, `solve_network_validation: 09:00:00`).

The same file is **also** loaded as `configfile:` layer #1 in `workflow/Snakefile:105`, which is what
puts `walltime:` into the merged config. Side effect: `__default__` is injected into the merged
`config` dict too, and it is **not** in `workflow/schemas/config.schema.yaml` — it only survives
because the schema's top level is `additionalProperties: true`.

### 1.3 Walltime plumbing

* `workflow/Snakefile:105` loads `repo_data/config/config.slurm.yaml`; `:121-123` conditionally
  loads the per-user overlay `config/config.slurm.yaml` (guarded by `os.path.exists`).
* `workflow/rules/common.smk:72` `config_provider(*keys, default=None)` → `functools.partial` over
  `static_getter`/`dynamic_getter`; rules use `walltime=config_provider("walltime", "<rule>", default="…")`.
* `workflow/schemas/config.schema.yaml:475-477` declares `walltime:` as an open `type: object`
  (no per-rule keys, so a typo'd rule name in the block is silently ignored and the inline
  `default=` wins).
* `docs/source/config-configuration.md:727-742` documents the block via a `literalinclude` bounded
  by the `# docs : WALLTIME` / `# docs :` markers in `config.slurm.yaml`. There is **no**
  `docs/source/configtables/slurm.csv`.

### 1.4 Per-rule resource coverage (60 rules across `rules/*.smk` + `Snakefile`)

| | rules |
|---|---|
| declare `threads:` | 30 |
| declare `mem_mb` | 51 |
| declare `walltime` | 25 |
| declare `benchmark:` | 28 |
| **missing `walltime` but not local** | **~30** — `build_service_demand`, `build_transport_other_demand`, all 6 of `build_sector.smk`, all 7 of `postprocess_sector.smk`, `retrieve_zenodo_databundles`, `retrieve_nrel_efs_data`, `retrieve_eer_demand_data`, `retrieve_cpuc_servm_load`, `retrieve_cpuc_baseline_generators`, `retrieve_sector_databundle`, `retrieve_res_eulp`, `retrieve_com_eulp`, `retrieve_ship_raster` |
| **missing `mem_mb`** | `plot_natural_gas`, `retrieve_sector_databundle`, `retrieve_res_eulp`, `retrieve_com_eulp`, `docs_figures` |

Other directives that interact with the executor:

* `localrules:` (`workflow/Snakefile:68`) contains only `dag, clean`. **All 15 `retrieve_*` rules are
  submitted to Slurm**, each a 30-second download that waits in the queue.
* `group: "prepare"` on `add_extra_components` + `prepare_network`
  (`rules/build_electricity.smk:1096, 1137`) and `add_sectors` (`rules/build_sector.smk:63`).
* `retries:` on `build_fuel_prices` (3) and 7 `retrieve_*` rules (2–3).
* No `--default-resources`, no `resource_scopes`, no `nodes`/`slurm_partition`/`slurm_extra`
  resources anywhere.
* `ATLITE_NPROCESSES = config["atlite"]["nprocesses"]` = **8** (`config.common.yaml:264`), used as
  `threads:` for `build_cutout` and `build_renewable_profiles`.
* `solver_threads(w)` (`rules/common.smk:85`) reads `solving.solver_options.<set>.threads`,
  default 4; `gurobi-default` sets **8**, and `config.equivalence-usa.yaml:248` also sets 8.
* 32 rules hard-code `log: "logs/…"` instead of `LOGS + …`, so their logs land outside `RDIR`.
* `resources/powerplants/powerplants.csv` is a literal path deliberately shared across runs — a
  cross-scenario write race once multiple runs are in flight concurrently.

### 1.5 Pinned Snakemake version and plugin availability

| source | pin |
|---|---|
| `pyproject.toml:60` (`dependencies`) | `snakemake==7.32.4` |
| `pyproject.toml:91` (`[project.optional-dependencies] test`) | `snakemake==7.32.4` |
| `workflow/envs/environment.yaml:52` | `snakemake-minimal==7.32.4` |
| `uv.lock:2250-2251` | `snakemake` `7.32.4` |
| `workflow/Snakefile:3` | `min_version("6.0")` |

`grep` for `executor-plugin` / `executor_plugin` / `snakemake-interface` across `uv.lock`,
`pyproject.toml` and `workflow/envs/environment.yaml` returns **nothing**.
`ls .venv/lib/python3.11/site-packages/ | grep -i 'executor\|slurm'` returns nothing.
`.venv/bin/snakemake` exists.

**So: Snakemake 7, not 8. `--cluster-config` is deprecated-but-functional today, not removed.**
That flips the usual "this is already broken by the version bump" framing into "this will break the
moment anyone bumps", and it means `snakemake-executor-plugin-slurm` is *not currently usable* —
it requires Snakemake ≥8.

**Blocker for the Snakemake 8 bump, found while checking this:** `workflow/Snakefile:6-8` does
`from snakemake.remote.HTTP import RemoteProvider as HTTPRemoteProvider`. The entire
`snakemake.remote` subsystem was removed in Snakemake 8 in favour of storage plugins. Two call
sites: `rules/retrieve.smk:255` and `:278`. Small, but it is real work that must land before or
with the bump.

### 1.6 The real launchers (branch-only, not on `origin/develop`)

`git ls-files` for `*.sh`/`*.sbatch` yields 8 files. The Slurm-relevant ones:

| path | on develop? | what it is |
|---|---|---|
| `workflow/run_slurm.sh` | yes | the documented, broken launcher |
| `tests/equivalence/run_usa.sbatch` (97 ln) | **no — branch only** | monolithic driver for the USA equivalence campaign |
| `tests/equivalence/run_usa_cf.sbatch` (114 ln) | **no — branch only** | ditto, CF-comparison variant |
| `workflow/scripts/nrel_exclusion/build_nrel_artifacts.sbatch` | yes | out-of-band NREL exclusion artifact build |
| `workflow/scripts/nrel_exclusion/compress_godeeep_array.sbatch` | yes | `--array=1-129%16` GODEEEP CF compression |
| `workflow/report_benchmarks.py` (143 ln) | **no — branch only** | turns `benchmarks/*.tsv` into per-rule mem/walltime recommendations |

`workflow/rules/build_electricity.smk` and `build_sector.smk` on this branch add **11 missing
`benchmark:` directives** (commit `f9ac2be`) — that instrumentation is a branch-only asset and is
the empirical input the profile needs.

The two `run_usa*.sbatch` scripts already encode, by hand, every Sherlock lesson this audit would
otherwise have to invent from scratch:

* `-p normal`, `--time=20:00:00`/`24:00:00`, `--cpus-per-task=16`, `--mem=160GB`
* `-o`/`-e` into `/scratch/groups/iazevedo/kamran/eq-usa/` (scratch, not `$HOME`, not Oak)
* `UV_CACHE_DIR`, `PIP_CACHE_DIR`, `XDG_CACHE_HOME` redirected to `$GROUP_SCRATCH`
* `export GRB_LICENSE_FILE=/share/software/user/restricted/gurobi/11.0.2/licenses/gurobi.lic`
* `ml load gcc/12.4.0` + `CFLAGS=-Wno-error=incompatible-pointer-types` (GCC 14 breaks the
  `datrie 0.8.2` source build, which is a *Snakemake* dependency)
* `ml load system git/2.45.1` (system git is 1.8/CentOS 7, predates `git worktree`; this is also
  why `git branch --show-current` fails in this repo)
* `rsync` staging of `data/` and `cutouts/` from Oak → `$GROUP_SCRATCH`, then symlinking
  `workflow/data` and `workflow/cutouts` at the staged copies
* best-effort `snakemake --unlock` sweep of stale `.snakemake/locks` from killed predecessors
* `uv run python -m tests.equivalence.run --jobs 8 --timeout 43200`, i.e. Snakemake runs
  **locally with `-j 8` inside the one big job** — `tests/equivalence/build.py:630-648` builds
  `["uv","run","snakemake",target,"--configfile",cfg,"-j",N,"--scheduler","greedy","--rerun-triggers","mtime"]`
  with no `--cluster` anywhere.

The two `nrel_exclusion` sbatch scripts use a *different* set of site settings again:
`#SBATCH -A iazevedo` and `#SBATCH -p serc` (the group's owner partition), logs to
`logs/run_slurm/` **relative to the submit dir** (i.e. onto Oak), and
`conda activate pypsa-usa` from `/home/groups/iazevedo/miniforge3`. So the repo currently
contains three mutually inconsistent notions of "how we submit to Sherlock": `-p serc -A iazevedo`
+ conda, `-p normal` no account + uv, and `run_slurm.sh`'s `-p {cluster.partition}
-A {cluster.account}` + blank template.

### 1.7 Working tree vs `origin/develop`

`git diff --stat origin/develop` = 69 files. Slurm-relevant deltas, all branch-only:
`tests/equivalence/run_usa.sbatch` (new), `tests/equivalence/run_usa_cf.sbatch` (new),
`workflow/report_benchmarks.py` (new), `+21` lines of `benchmark:` directives in
`build_electricity.smk`, `+3` in `build_sector.smk`. **`run_slurm.sh` and `config.slurm.yaml` are
byte-identical to `origin/develop`** — nothing in this audit's core findings is branch-local.

The rest of the diff is ~470k lines of committed build artifacts (`resources/**`, `results/**`,
`.parquet`, `.pkl`, `.png`) that appear to have been accidentally staged despite `.gitignore:23-29`
listing `resources/`, `benchmarks/`, `logs/`. Out of scope here, but worth a separate look before
this branch is merged.

### 1.8 CI

`.github/workflows/main.yml` never invokes `snakemake` directly — Tier B goes through
`pytest -m integration`. So CI exercises none of the Slurm path and will not catch regressions in it.

### 1.9 Docs

* `docs/source/about-usage.md:55-67` — "Running on HPC Cluster". Tells the user to edit
  `workflow/config/config.slurm.yaml` (account/partition/email/**chdir** — `chdir` is never used),
  to edit the `--configfile` inside `run_slurm.sh`, and to run `bash run_slurm.sh`
  **"within a login node"**. That last instruction is the one that violates Sherlock policy directly.
* `docs/source/about-install.md:24-34` — `init_pypsa_usa.sh` seeds three files including
  `config.slurm.yaml`.
* `docs/source/config-configuration.md:8-20, 727-742` — the layer table and the `walltime` section.
* `docs/source/about-introduction.md:119-134` — directory map.
* `init_pypsa_usa.sh:20,24` — seeds `config.slurm.yaml`; `:44` prints the local-run hint only.

**Observed reality check:** `workflow/config/` in this checkout contains only `config.api.yaml` and
`.gitkeep`. `config.slurm.yaml` was **never seeded here**, so `--cluster-config config/config.slurm.yaml`
would abort on a missing file before Snakemake even parses the DAG. No `workflow/profiles/` exists;
no `~/.config/snakemake` exists.

---

## 2. Defects, ranked

### S1 — blocking: `run_slurm.sh` cannot run as shipped

1. `--configfile config/CH1/config.tamu.single_horizon.bau.yaml` — **no such path in the repo**
   (`config/CH1/` does not exist). Snakemake exits immediately.
2. `--cluster-config config/config.slurm.yaml` — **not present** in `workflow/config/` in this
   checkout; `init_pypsa_usa.sh` is documented to seed it but has not been run, or was run before
   that file was added.
3. `-A {cluster.account}` with `account:` blank in the template → the sbatch line becomes
   `sbatch -A  --mail-type ALL …` → `sbatch: error: invalid option`. Per Sherlock policy there are
   no accounts at all, so the flag should not exist; note the `nrel_exclusion` scripts nonetheless
   pass `-A iazevedo`, which is either tolerated or silently ignored ([UNVERIFIED], §4).
4. Same for `-p {cluster.partition}` with `partition:` blank → `-p ` → invalid.
5. `--mail-user {cluster.email}` blank → `--mail-user ` → invalid. And `--mail-type ALL` on
   **every** job means one BEGIN + one END email per DAG node; a USA run is O(500) jobs, so O(1000)
   emails, which will get the address throttled or blocked.

### S1 — blocking: `{resources.walltime}` / `{resources.mem_mb}` are formatted for rules that do not declare them

`--cluster` string formatting is not tolerant of missing resources. Any rule reached by the DAG that
lacks `walltime` (≈30 rules — all of `build_sector.smk`, all of `postprocess_sector.smk`,
`build_service_demand`, `build_transport_other_demand`, 9 `retrieve_*`) or lacks `mem_mb`
(`plot_natural_gas`, `retrieve_sector_databundle`, `retrieve_res_eulp`, `retrieve_com_eulp`,
`docs_figures`) will fail at submission time with a "failed to format cluster command" style error.
Any sector-coupled (`sector: G` / `E-G`) run therefore cannot get past `build_population_layouts`.

### S1 — blocking: no `--cluster-status`, so a killed job hangs the scheduler forever

With `--cluster` and no `--cluster-status` script, Snakemake 7 detects completion via the jobscript's
own exit-code touch file. If Slurm kills the job — OOM, `TIMEOUT`, `NODE_FAIL`, preemption on an
owner partition — that file is never written and **Snakemake waits indefinitely**. On this repo that
is not hypothetical: `solve_network` requests wall times up to 20 h and memory in the hundreds of GB
(see below), i.e. it is one of the likeliest jobs in the DAG to be killed. There is also no
`--cluster-cancel scancel`, so killing the scheduler leaves every in-flight job orphaned and burning
allocation.

### S1 — policy violation: the scheduler runs on the login node

`docs/source/about-usage.md:61` — *"open a terminal within a login node of your cluster and run the
script"*. A USA-scale DAG keeps the Snakemake process alive for a day or more; `/etc/claude-code/CLAUDE.md`
forbids running Python on the login node, and there is no `nohup`/`tmux`/`setsid` in the script, so
the run also dies with the SSH session. This is the single change that most improves reliability.

### S2 — memory requests do not scale with model size (see §5 for the full treatment)

The two defects below are the scheduler-facing symptoms of upstream issues #808 / #811. §5 quantifies
them against the first USA-scale measurements and proposes the fix; they are listed here so the
severity ranking is complete.

#### S2a — `solve_network` memory request is unbounded and almost certainly unschedulable

`rules/solve_electricity.smk:52`:

```
mem_mb=lambda wildcards, input, attempt: (input.size // 100000) * attempt * 150,
```

`input.size` is the summed byte size of **all** inputs. For the USA `s300_c134a` full-year prepared
network on this branch, an input footprint of ~400 MB yields `4000 × 150 = 600,000 MB ≈ 600 GB` on
the first attempt, `1.2 TB` on the second. Sherlock `normal` nodes are nowhere near that, so the job
either pends forever or is rejected outright. There is no floor, no cap, and no routing to `bigmem`.
Meanwhile `threads: solver_threads` = **8** (`gurobi-default`), so the request is 600 GB against
8 cores — a shape no partition will schedule well. Note also that `bigmem` tops out at 1 day while
`walltime: solve_network: '20:00:00'` leaves only 4 h of slack for queue-time overrun.

#### S2b — `mem_mb` lambdas return floats

Three rules multiply by a float:

* `rules/build_electricity.smk:330` — `… * 2.5`
* `rules/build_electricity.smk:906` (`aggregate_to_substations`) — `… * attempt * 1.5`
* `rules/build_electricity.smk:957` (`cluster_resources`) — `… * attempt * 1.5`

`int // int * int * float` is a `float`, so the `--cluster` template renders `--mem 12345.0`, which
`sbatch` rejects (`Invalid --mem specification`). [UNVERIFIED — depends on whether Snakemake 7 coerces
`resources` to int before formatting; see §4.] Trivially fixed by wrapping in `int(...)`.

### S2 — string `walltime` inside a group job is not aggregatable

`add_extra_components` and `prepare_network` share `group: "prepare"`; `add_sectors` also joins it.
Under `--cluster`, a group becomes **one** Slurm job and Snakemake aggregates resources across its
members. Integer resources sum; a `"00:30:00"` string has no defined aggregation. Best case the
group takes one member's value and under-requests; worst case it raises. The fix is the same one the
Snakemake 8 plugin forces anyway: express time as **integer minutes** (`runtime`), not as an
`HH:MM:SS` string.

### S2 — Slurm log directories are on Oak and are never created

`output: logs/{rule}/log-%j.out` is (a) **relative to the submission cwd**, i.e. `workflow/` on
`$OAK` — job stdout/stderr streaming onto Lustre-backed Oak, which the Sherlock guidance explicitly
tells you not to do; and (b) **not created by Snakemake**. Snakemake creates parent dirs for `log:`
and `benchmark:` outputs, but the `--cluster` sbatch `-o`/`-e` paths are opaque to it. `sbatch` with
a nonexistent output directory submits, then the job fails at launch with no diagnostic anywhere.
The `run_usa*.sbatch` scripts already do the right thing (`/scratch/groups/…/eq-usa/`); the
supported path does not.

### S2 — all 15 `retrieve_*` rules are submitted to Slurm

None are in `localrules:`. Each spawns a Slurm job for a download that takes seconds, so the DAG
front-end becomes queue-bound. Worse, it assumes compute nodes have outbound HTTPS to Zenodo / the
EIA API / NREL, which on Sherlock generally requires a proxy ([UNVERIFIED], §4). Seven of them carry
`retries: 2`/`3`, which multiplies the queue round-trips.

### S3 — `--latency-wait 60` is thin for Lustre at fan-out

60 s is the value in `run_slurm.sh`. It is adequate for a handful of concurrent jobs and marginal
when `build_renewable_profiles` fans out across techs and dozens of jobs land outputs on Lustre at
once. (For Phase 2 this becomes critical: the Snakemake 8 default is 5 s.)

### S3 — Gurobi license and module environment are not provided by the supported path

`GRB_LICENSE_FILE` is present in `run_slurm.sh` only as a **commented-out line with a stray
U+2060**. The working value, proven by both `run_usa*.sbatch`, is
`/share/software/user/restricted/gurobi/11.0.2/licenses/gurobi.lic` — note the license path says
**11.0.2** while `pyproject.toml:37` pins `gurobipy==11.0.3`. Same 11.0.x major/minor, so it should
be accepted, but it is worth confirming rather than assuming ([UNVERIFIED], §4).

Nothing in the supported path loads `gcc/12.4.0` or `system git/2.45.1`, both of which the
equivalence scripts discovered the hard way (datrie source build under GCC 14; `git worktree` under
system git 1.8). Nothing redirects `UV_CACHE_DIR`/`PIP_CACHE_DIR`/`XDG_CACHE_HOME` off the 15 GB
`$HOME`.

### S3 — environment activation inside jobs is implicit and fragile

`run_slurm.sh` calls bare `snakemake`, so it depends on whatever is active in the submitting shell.
Under `--cluster`, Snakemake bakes the *submitting* interpreter path into each generated jobscript;
that happens to work when the venv lives at a stable absolute path on Oak, and breaks if the user
launched via `uv run` into an ephemeral env. Rules declare `conda: "../envs/environment.yaml"` but
nothing passes `--use-conda`, so those declarations are inert; if anyone ever enables it, every
compute node would start solving conda envs concurrently on Lustre.

### S3 — `--rerun-incomplete` is always on and `--jobs 20` is hard-coded

`--rerun-incomplete` unconditionally masks genuinely corrupt intermediates instead of surfacing them.
`--jobs 20` is not tunable without editing the tracked script.

### S4 — dead and mis-scoped config

* `__default__.walltime`, `__default__.cpus_per_task`, `__default__.chdir` are never interpolated.
* `__default__` leaks into the merged `config` dict (because the same file is a `configfile:` layer)
  and is unrepresented in `schemas/config.schema.yaml`.
* `docs/source/about-usage.md:57` still tells users to set `chdir`.
* The `walltime:` block has no schema for its keys, so `walltime: cluster_netowrk:` is accepted
  silently and the inline `default=` quietly wins.

### S4 — three incompatible site conventions coexist

`-p serc -A iazevedo` + conda (`nrel_exclusion/*.sbatch`) vs `-p normal`, no account, + uv
(`run_usa*.sbatch`) vs blank template (`run_slurm.sh`). Whatever design is adopted must collapse
these to one.

### Non-defects worth recording (so they are not "fixed" later)

* **`--mem {resources.mem_mb}` units are correct.** `sbatch --mem` with no suffix is megabytes and
  Snakemake's `mem_mb` is megabytes. No conversion needed.
* **`-c {threads}` is safe.** `threads` always resolves (default 1), unlike `resources.*`.
* **`walltime: '20:00:00'` fits `normal`** (2-day ceiling). Only a >2-day driver needs `--qos=long`.

---

## 3. Recommended design

### 3.1 Options compared

**(a) Snakemake 8 profile + `snakemake-executor-plugin-slurm`, driven from an sbatch driver.**
Per-rule Slurm jobs. The plugin owns `sacct` status polling, cancellation, `--parsable` submission,
retry-on-`TIMEOUT`, and per-rule `slurm_partition`/`runtime`/`cpus_per_task`/`mem_mb`/`slurm_extra`
resources. `set-resources:` in the profile is the site's tuning surface; `default-resources:` gives
every rule a floor so nothing falls into Slurm's 1-CPU/tiny-mem/short-walltime defaults. Fixes S1
(status polling, cancellation), S2 (integer `runtime`, group aggregation), and gives the
per-rule heterogeneity this DAG needs — `solve_network` on `bigmem` with 16 cores, everything else
on `normal` with 2. **Cost:** requires the Snakemake 7 → 8 bump, which also requires replacing
`snakemake.remote.HTTP` (`Snakefile:6-8`, `retrieve.smk:255,278`), re-testing the `scenarios`
machinery, and re-pinning in three places. It also diverges the candidate side of the equivalence
harness from its Snakemake-7 anchor worktree.

**(b) Generic `--cluster` (SM7) / `--executor cluster-generic` (SM8) with a submit script.**
Same architecture — per-rule jobs, profile-driven — but the sbatch line and a `cluster-status`
script are maintained in-repo instead of by the plugin. Works **today**, at 7.32.4, with zero
dependency changes. Costs ~40 lines of shell (`slurm-status.sh` mapping `sacct` state → `running` /
`success` / `failed`). Every resource key and profile entry written for (b) carries over to (a)
verbatim except `walltime` → `runtime`.

**(c) One monolithic sbatch job running `snakemake -j N` locally.**
What the branch actually does today. Virtues: no queue latency between rules, no Lustre
`latency-wait` race, no cluster-status problem, trivially reproducible, and — decisive for the
equivalence harness — a single process whose environment is identical for both the candidate and
the anchor worktree. Vices: the whole DAG is sized to the *largest* rule, so a 160 GB / 16-core
allocation sits mostly idle for the ~55 rules that need 5 GB and 1 core; it cannot exceed one node's
memory, which the current `solve_network` formula already wants to; and the entire multi-hour run is
lost to a single wall-time overrun with no per-rule retry.

### 3.2 Recommendation

**Adopt (b) now and (a) next, sharing one profile directory; keep (c) as an explicitly supported
second profile.**

Rationale specific to this repo:

1. **The bump is not free and should not be on the critical path.** Snakemake is pinned at 7.32.4 in
   three files plus `uv.lock`, and `snakemake.remote.HTTP` must be rewritten first. Option (b)
   delivers every structural fix — profile directory, driver job, `default-resources` floor,
   per-rule `set-resources`, status polling, cancellation, scratch logs — without touching a single
   dependency pin. Option (a) then becomes a mechanical follow-up: same profile, `cluster:` →
   `executor: slurm`, `walltime` → `runtime`.
2. **The DAG is genuinely heterogeneous**, which is the case against (c) as the *only* mode.
   `solve_network` wants bigmem + 8-16 cores + many hours; `build_renewable_profiles` wants 8 cores
   × N techs *concurrently* on different nodes; ~40 rules want one core and a few GB. Sizing all of
   that to one allocation, as `run_usa.sbatch` does at 16 cores / 160 GB / 20 h, wastes most of it.
3. **(c) still has a real job to do.** The equivalence harness runs the candidate and a pinned
   upstream anchor worktree and compares artifacts; introducing Slurm scheduling nondeterminism and
   two different executor code paths into that comparison is a bad trade. Keep it monolithic and
   *name* it, rather than leaving it as two undocumented sbatch scripts.
4. **The empirical inputs already exist.** `workflow/report_benchmarks.py` (branch-only) turns
   `benchmarks/*.tsv` into exactly the `mem`/`time` recommendations a `set-resources:` block wants,
   and commit `f9ac2be` added the 11 missing `benchmark:` directives. The profile is the natural
   destination for that output; today it has nowhere to go.

### 3.3 Specific decisions

**Per-rule walltime — keep the `walltime:` config block in Phase 1, retire it in Phase 2.**

Phase 1 changes nothing in the rules: the cluster command keeps forwarding
`--time {resources.walltime}`, and the ~30 rules that lack it get covered by the profile's
`default-resources: walltime="02:00:00"` — which is precisely the S1 formatting failure, fixed
without editing 30 rules.

Phase 2 deletes the block. `walltime:` is a home-grown reimplementation of `--set-resources`, and
per-rule wall time is a *site* property, not a *scenario* property, so it belongs in the site profile.
Concretely, in Phase 2: drop `walltime:` from `repo_data/config/config.slurm.yaml`, drop the
`walltime:` node from `schemas/config.schema.yaml`, delete the 25 `walltime=config_provider("walltime", …)`
lines from `rules/*.smk`, and express the same numbers as
`set-resources: <rule>: runtime=<int minutes>` in the profile.

**Per-rule memory and threads — keep in the rules, but bound them.**
`mem_mb` lambdas depend on `input.size` and `attempt`, which a static profile cannot express, so they
must stay. Three fixes: wrap every one in `int(...)`; give each a `max(…, floor)`; and cap the
pathological ones — `solve_network` in particular must become something like
`min(int((input.size // 100000) * attempt * 150), 700_000)` with an explicit
`slurm_partition="bigmem"` in the profile, or the formula recalibrated against measured
`max_rss` from `report_benchmarks.py`. `threads:` stays inline; `solve_network`'s
`threads: solver_threads` correctly tracks `solving.solver_options.<set>.threads` and must keep
matching `--cpus-per-task`.

**Per-user overlay — `workflow/config/config.slurm.yaml` stops being a `--cluster-config` file.**
Split its two jobs:

* Site-invariant scheduler settings → the **tracked** `workflow/profiles/sherlock/config.yaml`.
* Genuinely per-user values (partition, email, log root, Gurobi license, cache roots) → **environment
  variables read by the driver script**, not YAML. Slurm natively honours `SBATCH_PARTITION`,
  `SBATCH_ACCOUNT`, `SBATCH_TIMELIMIT`, so the driver can `export SBATCH_PARTITION=serc` and neither
  the profile nor the sbatch template needs a `-p` flag at all. This also removes the blank-value
  failure mode (an unset env var means "Slurm default", not `-p `).
* Seed those exports from a gitignored `workflow/profiles/sherlock/env.sh` written by
  `init_pypsa_usa.sh`, replacing the `config.slurm.yaml` seeding at `init_pypsa_usa.sh:24`.
* In Phase 1, keep `repo_data/config/config.slurm.yaml` for the `walltime:` block only and **delete
  the `__default__:` block** — it is dead, it pollutes the merged config, and it is unschema'd.

**Gurobi license and environment — driver job only, inherited by children.**
`sbatch` defaults to `--export=ALL`, so anything the driver exports reaches every job Snakemake
submits. The driver sets `GRB_LICENSE_FILE`, `ml load gcc/12.4.0`, `ml load system git/2.45.1`,
`UV_CACHE_DIR`/`PIP_CACHE_DIR`/`XDG_CACHE_HOME` under `$GROUP_SCRATCH`, and activates
`.venv` explicitly (`source .venv/bin/activate`, then plain `snakemake` — **not** `uv run snakemake`)
so the interpreter path baked into each jobscript is stable. Pass `--export=ALL` explicitly in the
sbatch template rather than relying on the default.

**Logs.**
Three distinct streams, three destinations:
* Slurm per-job stdout/stderr → `$PYPSA_SLURM_LOGDIR` on `$SCRATCH`/`$GROUP_SCRATCH`, created by the
  driver with `mkdir -p` before Snakemake starts. Never Oak, never `$HOME`.
* Snakemake `log:` files and `benchmark:` TSVs → stay in the repo (`workflow/logs`,
  `workflow/benchmarks`); Snakemake creates their parents, they are small, and
  `report_benchmarks.py` expects them there.
* The driver's own stdout/stderr → `$SCRATCH`, same as the equivalence scripts do today.
Separately: `workflow/resources` and `workflow/results` should be symlinks into `$GROUP_SCRATCH`
(the `run_usa*.sbatch` pattern, extended beyond `data/` and `cutouts/`), with an `rsync` of
`results/` back to Oak at the end of the driver — `$SCRATCH` purges at 90 days of inactivity.

**User-facing command.**

```console
# one-time
bash init_pypsa_usa.sh                       # seeds workflow/profiles/sherlock/env.sh
$EDITOR workflow/profiles/sherlock/env.sh    # partition, email, scratch root

# every run
sbatch workflow/run_slurm.sbatch config/config.my_scenario.yaml
```

`run_slurm.sh` is replaced by `run_slurm.sbatch` (submitted, not sourced), the scenario config
becomes an **argument** instead of a tracked hard-coded path, and nothing runs on the login node.
For the monolithic mode: `sbatch workflow/run_slurm.sbatch --monolithic config/….yaml`, which swaps
`--profile profiles/sherlock` for `--profile profiles/sherlock-monolithic`.

### 3.4 Draft — `workflow/profiles/sherlock/config.yaml` (Phase 1, Snakemake 7.32.4)

```yaml
# PyPSA-USA — Sherlock Slurm profile (Snakemake 7.x, cluster-generic).
#   snakemake --profile profiles/sherlock <target>
# Site-specific values that vary per user (partition, mail, log root) come from
# the environment via profiles/sherlock/env.sh, NOT from this file, so that an
# unset value means "Slurm default" instead of an empty flag.

cluster: >-
  sbatch
  --parsable
  --export=ALL
  --job-name smk-{rule}
  --cpus-per-task {threads}
  --mem {resources.mem_mb}
  --time {resources.walltime}
  --output {resources.slurm_logdir}/{rule}/%j.out
  --error  {resources.slurm_logdir}/{rule}/%j.err
cluster-status: profiles/sherlock/slurm-status.sh
cluster-cancel: scancel
cluster-cancel-nargs: 500

jobs: 60
local-cores: 4
latency-wait: 120
restart-times: 2
max-jobs-per-second: 5
max-status-checks-per-second: 1
keep-going: true
printshellcmds: true
rerun-triggers: [mtime, params, input]
scheduler: greedy

# Floor for every rule. Fixes the ~30 rules with no walltime and the 5 with no
# mem_mb, which today make the --cluster command fail to format.
default-resources:
  - mem_mb=8000
  - walltime="02:00:00"
  - slurm_logdir=os.environ.get("PYPSA_SLURM_LOGDIR", "logs/slurm")

# Site tuning. Numbers here should be refreshed from
#   python workflow/report_benchmarks.py workflow/benchmarks
set-resources:
  - solve_network:mem_mb=480000
  - solve_network:walltime="20:00:00"
  - solve_network:slurm_partition=bigmem
  - solve_network_validation:mem_mb=240000
  - solve_network_validation:walltime="09:00:00"
  - cluster_resources:mem_mb=120000
  - cluster_resources:walltime="05:00:00"
  - cluster_network:mem_mb=96000
  - cluster_network:walltime="04:00:00"
  - add_electricity:mem_mb=64000
  - add_electricity:walltime="04:00:00"
  - build_renewable_profiles:mem_mb=64000
  - build_renewable_profiles:walltime="04:00:00"
  - add_demand:mem_mb=48000
  - add_demand:walltime="02:00:00"
  - aggregate_to_substations:mem_mb=48000
  - aggregate_to_substations:walltime="02:00:00"

set-threads:
  - solve_network=8            # must equal solving.solver_options.<set>.threads
  - build_renewable_profiles=8 # must equal atlite.nprocesses
  - build_cutout=8

# Downloads: seconds of work, no reason to queue for them, and compute-node
# egress may need a proxy. Run them in the driver process instead.
local-rules:
  - retrieve_zenodo_databundles
  - retrieve_sector_databundle
  - retrieve_nrel_efs_data
  - retrieve_eer_demand_data
  - retrieve_cpuc_servm_load
  - retrieve_cpuc_baseline_generators
  - retrieve_gridemissions_data
  - retrieve_res_eulp
  - retrieve_com_eulp
  - retrieve_ship_raster
  - retrieve_cutout
  - retrieve_caiso_data
  - retrieve_pudl
  - retrieve_nrel_exclusion_artifact
  - retrieve_godeeep_cf
  - retrieve_egs
  - retrieve_seismic_risk_mask
```

Phase 2 delta (Snakemake 8 + plugin): delete `cluster`, `cluster-status`, `cluster-cancel*`;
add `executor: slurm`; rename every `walltime="HH:MM:SS"` to `runtime=<int minutes>`; replace
`slurm_logdir` with the plugin's own `--slurm-logdir`/`slurm_extra`; add
`snakemake-executor-plugin-slurm` to `pyproject.toml` and `workflow/envs/environment.yaml`.

### 3.5 Draft — `workflow/profiles/sherlock/slurm-status.sh`

```bash
#!/usr/bin/env bash
# Snakemake 7 --cluster-status hook. Prints exactly one of:
#   success | failed | running
# Without this, a Slurm-killed job (OOM / TIMEOUT / NODE_FAIL) never writes its
# exit-code file and the scheduler blocks forever.
set -uo pipefail
jobid="$1"

state=$(sacct -j "$jobid" --format=State --noheader --parsable2 2>/dev/null \
        | head -n1 | cut -d' ' -f1)

# sacct can lag briefly after submission; fall back to squeue.
if [[ -z "$state" ]]; then
    state=$(squeue -j "$jobid" -h -o %T 2>/dev/null)
fi

case "$state" in
    COMPLETED)                                   echo success ;;
    ""|PENDING|RUNNING|SUSPENDED|COMPLETING|CONFIGURING|REQUEUED|RESIZING)
                                                 echo running ;;
    *)                                           echo failed  ;;   # FAILED CANCELLED TIMEOUT OUT_OF_MEMORY NODE_FAIL PREEMPTED BOOT_FAIL DEADLINE
esac
```

### 3.6 Draft — `workflow/run_slurm.sbatch` (replaces `run_slurm.sh`)

```bash
#!/bin/bash
# PyPSA-USA driver job. The Snakemake scheduler runs HERE, on a compute node —
# never on the login node — and submits one Slurm job per rule via
# profiles/sherlock. Runs for the life of the whole DAG, so it is small and long.
#
#   sbatch workflow/run_slurm.sbatch config/config.my_scenario.yaml [extra snakemake args]
#   sbatch workflow/run_slurm.sbatch --monolithic config/config.my_scenario.yaml
#
#SBATCH --job-name=pypsa-usa
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --open-mode=append
# -p / --mail-user / -o / -e come from SBATCH_PARTITION, SBATCH_MAIL_USER and
# the --output/--error passed on the sbatch command line by scripts/submit.sh,
# or set them here for a fixed site. Do NOT hard-code an empty -A: Sherlock has
# no accounts, and `-A ''` is a submission error.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/workflow/profiles/sherlock/env.sh"   # gitignored, seeded by init_pypsa_usa.sh

PROFILE=profiles/sherlock
if [[ "${1:-}" == "--monolithic" ]]; then
    PROFILE=profiles/sherlock-monolithic
    shift
fi
CONFIGFILE="${1:?usage: sbatch run_slurm.sbatch [--monolithic] <configfile> [snakemake args...]}"
shift

# --- Environment -----------------------------------------------------------
# GCC 14 hard-errors on incompatible-pointer-types, breaking the datrie 0.8.2
# source build (a snakemake dependency). System git is 1.8 and predates
# `git worktree`, which the equivalence harness needs.
ml load gcc/12.4.0
ml load system git/2.45.1
export CFLAGS="${CFLAGS:--Wno-error=incompatible-pointer-types}"

# Keep every cache off the 15 GB NFS $HOME.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$GROUP_SCRATCH/$USER/cache/uv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$GROUP_SCRATCH/$USER/cache/pip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$GROUP_SCRATCH/$USER/cache/xdg}"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

# Slurm job logs on scratch, NOT on Oak and NOT in $HOME. sbatch does not
# create -o/-e directories; a missing one fails the job silently at launch.
export PYPSA_SLURM_LOGDIR="${PYPSA_SLURM_LOGDIR:-$GROUP_SCRATCH/$USER/pypsa-usa/slurm-logs}"
mkdir -p "$PYPSA_SLURM_LOGDIR"
awk '/^rule /{print $2}' "$REPO"/workflow/rules/*.smk | tr -d ':' \
    | xargs -I{} mkdir -p "$PYPSA_SLURM_LOGDIR/{}"

export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-/share/software/user/restricted/gurobi/11.0.2/licenses/gurobi.lic}"
[[ -r "$GRB_LICENSE_FILE" ]] || { echo "ERROR: no readable Gurobi license at $GRB_LICENSE_FILE" >&2; exit 1; }

# Activate the venv explicitly rather than via `uv run`: --cluster bakes the
# submitting interpreter path into every generated jobscript, so it must be a
# stable absolute path.
source "$REPO/.venv/bin/activate"

cd "$REPO/workflow"

# Clear a stale lock left by a killed predecessor job.
if [[ -n "$(ls -A .snakemake/locks 2>/dev/null || true)" ]]; then
    snakemake --unlock --configfile "$CONFIGFILE" || true
fi

# --- Run -------------------------------------------------------------------
# scancel every child job if the driver hits its own wall time.
trap 'scancel --state=PENDING,RUNNING --jobname="smk-*" --me || true' TERM EXIT

snakemake \
    --profile "$PROFILE" \
    --configfile "$CONFIGFILE" \
    "$@"

# --- Persist ---------------------------------------------------------------
# $SCRATCH purges after 90 days of inactivity; results belong on Oak.
if [[ -n "${PYPSA_RESULTS_ARCHIVE:-}" ]]; then
    rsync -a results/ "$PYPSA_RESULTS_ARCHIVE/"
fi
```

### 3.7 Draft — `workflow/profiles/sherlock-monolithic/config.yaml`

```yaml
# One-node mode: no per-rule Slurm submission. This is what the equivalence
# harness (tests/equivalence/run_usa*.sbatch) needs — a single process whose
# environment is bit-identical for the candidate and the pinned anchor
# worktree, with no scheduler nondeterminism in the comparison.
# Submit with:  sbatch --mem=160G -c 16 --time=24:00:00 \
#                   workflow/run_slurm.sbatch --monolithic <configfile>
cores: all
latency-wait: 30
rerun-triggers: [mtime]
scheduler: greedy
keep-going: true
printshellcmds: true
default-resources:
  - mem_mb=8000
  - walltime="02:00:00"
```

### 3.8 Exact change list

**New files**
| path | contents |
|---|---|
| `workflow/profiles/sherlock/config.yaml` | §3.4 |
| `workflow/profiles/sherlock/slurm-status.sh` | §3.5, `chmod +x` |
| `workflow/profiles/sherlock/env.sh.template` | `SBATCH_PARTITION`, `SBATCH_MAIL_USER`, `PYPSA_SLURM_LOGDIR`, `PYPSA_RESULTS_ARCHIVE`, `GRB_LICENSE_FILE` |
| `workflow/profiles/sherlock-monolithic/config.yaml` | §3.7 |
| `workflow/run_slurm.sbatch` | §3.6, `chmod +x` |

**Deleted**
| path | why |
|---|---|
| `workflow/run_slurm.sh` | superseded; keep a one-line stub for a release or two that `echo`s the new command and exits 1 |

**Modified**
| path | change |
|---|---|
| `workflow/repo_data/config/config.slurm.yaml` | delete the whole `__default__:` block (dead + unschema'd); rewrite the header comment (it documents `--cluster-config`, which goes away); keep `walltime:` and its `# docs :` markers for Phase 1 |
| `workflow/rules/build_electricity.smk:330, 906, 957` | wrap the float-producing `mem_mb` lambdas in `int(...)` |
| `workflow/rules/solve_electricity.smk:52` | cap `mem_mb` (`min(..., 700_000)`) and add a `max(..., floor)`; recalibrate the `150` coefficient against measured `max_rss` |
| `workflow/Snakefile:68` | leave `localrules:` alone — the `retrieve_*` list is expressed in the profile's `local-rules:` so non-Sherlock users are unaffected |
| `init_pypsa_usa.sh:20,24,44` | stop seeding `config.slurm.yaml`; seed `workflow/profiles/sherlock/env.sh` from the template instead; print the `sbatch workflow/run_slurm.sbatch …` hint |
| `.gitignore` | add `/workflow/profiles/*/env.sh` |
| `docs/source/about-usage.md:55-67` | rewrite the whole "Running on HPC Cluster" section: profile + driver, `sbatch` not `bash`, **remove the "run it on a login node" instruction**, drop `chdir`, document `--monolithic` |
| `docs/source/about-install.md:24-34` | "three files" → two (`config.default.yaml`, `config.api.yaml`) plus the profile env file |
| `docs/source/config-configuration.md:8-20` | `config.slurm.yaml` row: "per-rule `walltime:` block" only, no scheduler settings |
| `docs/source/config-configuration.md:727-742` | update the `walltime` prose: it is forwarded by the **profile**, not by `run_slurm.sh` |
| `docs/source/about-introduction.md:119-134` | add `profiles/` to the directory map |
| `tests/equivalence/run_usa.sbatch`, `run_usa_cf.sbatch` | keep the data-staging/module/env preamble; replace the inline `uv run python -m tests.equivalence.run` invocation's implicit local execution with `--profile profiles/sherlock-monolithic` so there is one source of truth for `-j`/`latency-wait`/`rerun-triggers` |
| `workflow/scripts/nrel_exclusion/*.sbatch` | drop `-A iazevedo`; take partition from `SBATCH_PARTITION`; move `-o`/`-e` off Oak onto `$SCRATCH` |
| `.github/workflows/main.yml` | add a `snakemake --profile workflow/profiles/sherlock --dry-run` lint step (with a stubbed `PYPSA_SLURM_LOGDIR`) so the profile cannot silently rot — CI currently exercises none of this |
| `workflow/repo_data/config/config.slurm.yaml` | **§5.5(1)** add a `mem:` block mirroring `walltime:`, with `# docs : MEM` / `# docs : end` markers |
| `workflow/schemas/config.schema.yaml` | **§5.5(1)** add a `mem:` node beside the existing `walltime:` node at line 475 |
| `workflow/rules/common.smk:93` | **§5.5(2)** delete the dead `memory(w)`; add `MEM_FLOOR_MB`/`MEM_CAP_MB`/`MEM_HEADROOM`, `INTERCONNECT_SCALE`, `MEM_REF_MB`, `_snapshots()`, `estimate_mem_mb()`, `scheduler_mem()` |
| `workflow/rules/*.smk` (54 sites) | **§5.5(2)** replace all 37 flat `mem_mb=<int>` and 17 `input.size // N * attempt * K` lambdas with `mem_mb=scheduler_mem("<rule>")`; this also subsumes the `int(...)` fix listed above for `build_electricity.smk:330, 906, 957` and the `solve_electricity.smk:52` cap |
| `workflow/rules/postprocess_sector.smk`, `postprocess.smk`, `retrieve.smk`, `validate.smk`, `build_sector.smk` | **§5.5(4)** add the ~28 missing `benchmark:` directives so every ordinary run contributes measurements |
| `workflow/report_benchmarks.py` → `workflow/scripts/report_benchmarks.py` | **§5.5(4)** move out of the `Snakefile`'s directory; extend to divide each peak by the run's size factor and emit a paste-ready `MEM_REF_MB` block |
| `workflow/repo_data/config/config.slurm.yaml` (`walltime:`) | **§5.4** retune against measurements — `add_demand`, `cluster_network`, `build_renewable_profiles`, `add_electricity`, `cluster_resources`, `aggregate_to_substations`, `build_fuel_prices` are all over-requested by 14–750x |
| `workflow/scripts/aggregate_to_substations.py`, `workflow/scripts/cluster_simpl.py` | **§5.5(5)** profile with `memory_profiler` *before* fitting `MEM_REF_MB`; ~27.8 GB peak RSS from ~100 MB of input on a topology-only rule is a probable dense-intermediate bug, not an intrinsic cost |
| `workflow/profiles/sherlock/config.yaml` | **§5.5(3)** `restart-times: 2` (SM7) / `retries: 2` (SM8) is the OOM safety net for the `attempt` multiplier in `scheduler_mem`; effective only together with `slurm-status.sh` (§3.5) |
| `docs/source/config-configuration.md` | **§5.5(1)** add a `mem` section beside the existing `walltime` section (727-742), `literalinclude` bounded by the new `# docs : MEM` markers |

**Phase 2 (separate PR)**
| path | change |
|---|---|
| `workflow/Snakefile:3,6-8`; `workflow/rules/retrieve.smk:255,278` | replace `snakemake.remote.HTTP` with a storage plugin or a plain download script; `min_version("8.0")` |
| `pyproject.toml:60,91`; `workflow/envs/environment.yaml:52`; `uv.lock` | `snakemake==8.x` + `snakemake-executor-plugin-slurm` |
| `workflow/profiles/sherlock/config.yaml` | `executor: slurm`; drop `cluster*`; `walltime="HH:MM:SS"` → `runtime=<minutes>`; delete `slurm-status.sh` |
| `workflow/repo_data/config/config.slurm.yaml`; `workflow/schemas/config.schema.yaml:475-477`; 25 rules in `rules/*.smk` | delete the `walltime:` block, its schema node, and every `walltime=config_provider("walltime", …)` line |
| `docs/source/config-configuration.md:727-742` | delete the `walltime` section |

---

## 4. Not verified, and how to verify it

Everything below needs a compute node — `sh_dev -c 2 --time=00:30:00` — never the login node.

| # | Claim | Command |
|---|---|---|
| 1 | Installed Snakemake really is 7.32.4 (lock/pins say so; the venv was not executed) | `.venv/bin/snakemake --version` |
| 2 | `snakemake-executor-plugin-slurm` is absent (inferred from `ls site-packages` + absent from `uv.lock`) | `.venv/bin/python -c "import snakemake_executor_plugin_slurm"` |
| 3 | Sherlock partition names and ceilings — `normal`, `bigmem`, `dev`, `owners`, and the group's `serc` (used by `nrel_exclusion/*.sbatch`, absent from the org policy list) | `sh_part`; `sinfo -o "%20P %10l %10m %6c %a"` |
| 4 | Max memory per node, which decides whether the recommended `solve_network:mem_mb=480000` and `slurm_partition=bigmem` are even schedulable | `sinfo -p bigmem -o "%P %m %c"`; `node_feat` |
| 5 | Whether `-A <account>` is accepted, ignored, or rejected on Sherlock — org policy says no accounts, but two tracked sbatch scripts pass `-A iazevedo` | `sacctmgr show assoc user=$USER format=Account,Partition`; `scontrol show config \| grep -i AccountingStorageEnforce` |
| 6 | Gurobi 11.0.2 license vs `gurobipy==11.0.3` | `ls -l /share/software/user/restricted/gurobi/11.0.2/licenses/gurobi.lic`; `ml spider gurobi`; `GRB_LICENSE_FILE=… .venv/bin/python -c "import gurobipy; gurobipy.Model()"` |
| 7 | Whether Snakemake 7 coerces float `resources.mem_mb` to int before formatting the `--cluster` string (S2) | `snakemake -n --cluster "echo --mem {resources.mem_mb}" -j1 aggregate_to_substations` on a small config |
| 8 | Whether a rule with no `walltime` really fails `--cluster` formatting (S1) | `snakemake -n --cluster "echo --time {resources.walltime}" -j1 build_population_layouts` |
| 9 | Outbound HTTPS from compute nodes (drives the `local-rules:` decision for `retrieve_*`) | `srun -p normal --time=00:05:00 curl -sI https://zenodo.org` and against the EIA API host |
| 10 | Whether `sacct` is enabled — `slurm-status.sh` and the SM8 plugin both depend on it | `sacct -j 1 --format=State --noheader` (any non-error output is enough) |
| 11 | Actual peak RSS / runtime per rule at USA scale, i.e. the real numbers for `set-resources:` | `python workflow/report_benchmarks.py workflow/benchmarks` on a compute node, using the branch's new `benchmark:` directives |
| 12 | Whether the driver's 2-day `normal` ceiling is enough end to end, or whether `--qos=long` is needed | measure once from #11; `sh_part` for the `long` QOS ceiling |
| 13 | Whether `git worktree` (equivalence anchor provisioning) works under system git | already answered in-repo: it does not — `git branch --show-current` fails here, and commit `d9de77b` added `ml load system git/2.45.1` for exactly this |
| 14 | Whether the §5.2 numbers generalise — every one is a **single observation** from one run on one node, not a distribution, and `max_rss` includes allocator slack | re-run with `benchmark:` on and compare; `seff <jobid>` post-mortem on each child job |
| 15 | The CA-only 0.3 GB figures for `aggregate_to_substations` / `cluster_resources` are quoted from issue #808 and were **not measured in this repo** | run `repo_data/config/config.tutorial.yaml` (CA, simpl=75, clusters=4m) with `benchmark:` on, then read `workflow/benchmarks/` |
| 16 | `INTERCONNECT_SCALE` in §5.5(2) is a placeholder ratio, not a fitted one | on a compute node, `len(pypsa.Network(".../elec_base_network.nc").buses)` for each of `usa`/`eastern`/`western`/`texas` |
| 17 | `solve_network` peak RSS and runtime at **full-year hourly** snapshots — §5.2's 7.4 GB / 11 min is at `opts=3h` (2,920 snapshots), and the campaign has since moved to 8,760 | re-run the equivalence solve without the `3h` opt and read the new benchmark TSV |
| 18 | Whether the ~27.8 GB in `aggregate_to_substations` / `cluster_resources` is intrinsic or a dense-intermediate bug (§5.5(5)) | `mprof run` / `@profile` from `memory_profiler` (already pinned) on `scripts/aggregate_to_substations.py` and `scripts/cluster_simpl.py`, on a compute node |

---

## 5. Memory sizing (issues #808 / #811)

Upstream issues [PyPSA/pypsa-usa#808](https://github.com/PyPSA/pypsa-usa/issues/808) (stale
walltimes) and [#811](https://github.com/PyPSA/pypsa-usa/issues/811) (memory requests that do not
scale with network size) describe the same root cause from two directions: **resource requests are
constants or byte heuristics that carry no information about how large the model actually is.**
This branch is the first place in the repo where that can be checked against measurements, because
commit `f9ac2be` added the 11 missing `benchmark:` directives and the USA campaign then ran with
them on.

### 5.1 What `origin/develop` actually declares

Counted over the eight `rules/*.smk` files as they exist on `origin/develop` (2,329 lines):

| form | count | example |
|---|---|---|
| flat constant `mem_mb=<int>` | **37** | `mem_mb=5000` (`build_shapes`), `mem_mb=30000` (`build_powerplants`) |
| byte heuristic `input.size // N * attempt * K` | **17** | `mem_mb=lambda wildcards, input, attempt: (input.size // 150000) * attempt * 1.5` |
| derived from a config knob | 1 | `mem_mb=ATLITE_NPROCESSES * 5000` (`build_cutout`) |
| **total `mem_mb` declarations** | **55** | |
| rules with **no** `mem_mb` at all | 5 | `plot_natural_gas`, `retrieve_sector_databundle`, `retrieve_res_eulp`, `retrieve_com_eulp`, `docs_figures` |

Two structural gaps:

* **`memory(w)` at `workflow/rules/common.smk:93` is dead code.** It is a PyPSA-Eur–inherited size
  estimator — it parses `{opts}` for `Nh`/`Nseg`, branches on `clusters` ending in `m`/`c`/`all`,
  and multiplies by `len(planning_horizons)` — and `grep -n "memory("` over all of
  `origin/develop:workflow/rules/*.smk` returns exactly one hit: its own `def`. Nothing calls it.
  It is nevertheless the right *skeleton*: it is the only thing in the repo that reasons about model
  size from DAG-time wildcards rather than from bytes on disk.
* **There is no `mem:` config block.** `config.slurm.yaml` has a `walltime:` block, read through
  `config_provider("walltime", "<rule>")`, giving a single documented place to retune wall times
  without editing rules. Memory has no equivalent — `grep '^mem:'` on
  `origin/develop:workflow/repo_data/config/config.slurm.yaml` returns nothing — so the only way to
  retune memory is to edit `rules/*.smk` and rebuild the DAG.

### 5.2 Measurements — USA, `simpl=300`, `clusters=134a`

From the 22 TSVs under `workflow/benchmarks/` on this branch (`equivalence/usa/`,
`equivalence/solve_network/usa/`, `cluster_network/usa/`, `equivalence/`). `max_rss` is MB, `s` is
seconds; each is a single observation from the USA equivalence run of 2026-09-01, `opts=3h`,
one planning horizon, `sector=E`.

| rule (benchmark stem) | `max_rss` MB | `h:m:s` | `mean_load` |
|---|---:|---|---:|
| `aggregate_to_substations` | **27,827.10** | 0:05:05 | 92.97 |
| `cluster_resources_elec_s300` | **27,672.65** | 0:07:49 | 90.65 |
| `power_build_demand_s300` | **10,307.60** | 0:01:36 | 84.09 |
| `solve_network/usa/elec_s300_c134a_ec_lv1.0_3h_E` | 7,372.38 | 0:11:13 | 590.67 |
| `build_renewable_profiles_onwind_2030_s300` | 2,942.38 | 0:00:43 | 71.98 |
| `build_renewable_profiles_solar_2030_s300` | 2,837.43 | 0:00:42 | 77.41 |
| `build_renewable_profiles_solar_s300` | 2,823.85 | 0:00:59 | 117.90 |
| `build_renewable_profiles_onwind_s300` | 2,772.39 | 0:01:49 | 58.17 |
| `elec_s300_add_electricity` | 2,360.36 | 0:02:09 | 68.96 |
| `build_powerplants` | 2,291.33 | 0:00:16 | 126.97 |
| `build_bus_regions` | 2,162.66 | 0:01:24 | 86.90 |
| `build_fuel_prices` | 1,380.80 | 0:01:25 | 26.09 |
| `prepare_network_…_REM-3h` | 690.24 | 0:00:16 | 86.89 |
| `add_extra_components_elec_s300_c134a_ec` | 602.05 | 0:00:18 | 78.88 |
| `build_base_network` | 573.11 | 0:01:15 | 31.44 |
| `prepare_network_…_3h` | 511.67 | 0:00:33 | 13.11 |
| `cluster_network/usa/elec_s300_c134a` | 408.84 | 0:00:37 | 10.21 |
| `build_shapes` | 390.42 | 0:01:03 | 55.62 |
| `add_sectors_…_REM-3h_E` | 379.06 | 0:00:10 | 91.39 |
| `elec_s300_add_demand` | 327.64 | 0:00:09 | 90.78 |
| `build_cost_data_2030` | 247.10 | 0:01:17 | 2.30 |
| `add_sectors_…_3h_E` | 244.04 | 0:00:24 | 5.97 |

Two rules dominate at **~27.8 GB**; one more is at 10.3 GB; **everything else is under 2.5 GB**, and
16 of the 22 are under 1 GB. Issue #808 reports `aggregate_to_substations` and `cluster_resources`
at **0.3 GB on a California-only run** — so the same two rules span **roughly two orders of
magnitude** between the CA and USA footprints. That is the whole of #811 in one line: a request
tuned on CA is off by ~90x on USA, and *no* constant can be right for both.

### 5.3 What the current declarations would have requested

Computed from the on-disk input sizes of this exact run (`resources/equivalence/`), `attempt=1`:

| rule | inputs (bytes) | formula | **requested** | **measured** | error |
|---|---:|---|---:|---:|---|
| `aggregate_to_substations` | `elec_base_network.nc` 99,508,729 + `bus2sub.csv` 5,562,603 + `sub.csv` 1,359,660 = 106,430,992 | `size//150000 * attempt * 1.5` | **1,063 MB** | 27,827 MB | **26x UNDER → certain OOM kill** |
| `cluster_resources` | `elec_b.nc` 49,970,244 + `regions_onshore.geojson` 44,611,929 + `regions_offshore.geojson` 380,315 = 94,962,488 | `size//150000 * attempt * 1.5` | **949 MB** | 27,673 MB | **29x UNDER → certain OOM kill** |
| `cluster_network` | `elec_s300_l_pp.pkl` 1,128,509,305 + `regions_onshore_s300.geojson` 11,958,008 + `regions_offshore_s300.geojson` 306,707 + 8 ReEDS/cost CSVs ≈ 1.145e9 | `size//100000 * attempt * 2` | **~22,900 MB** | 409 MB | **~56x OVER** |
| `solve_network` | prepared `_ec_lv1.0_3h_E.nc` (not retained on disk; O(10^8) B) | `size//100000 * attempt * 150` | **O(300,000 MB)** | 7,372 MB | **~40x OVER, and unschedulable** |

Both failure directions at once, in the same DAG. The reason is structural, and it is worth stating
plainly because it invalidates the whole `input.size` approach rather than just its coefficients:

* `aggregate_to_substations` and `cluster_resources` read a **compact netCDF** and build dense,
  roughly O(n_bus²)-shaped intermediates in memory. Bytes on disk under-predict by design.
* `cluster_network` reads a **1.13 GB pickle** and *contracts* it to 134 buses. Bytes on disk
  over-predict by design.
* `add_electricity` goes the other way: ~60 MB of inputs produce a **1.13 GB** output
  (`elec_s300_l_pp.pkl`), so its `input.size // 400000 * 2` predictor is blind to the thing that
  actually sets its footprint.

`input.size` is not a weak predictor of peak RSS here; for these rules it is anti-correlated with it.

### 5.4 The same data on the walltime half of #808

The `walltime:` block in `config.slurm.yaml` is equally stale — in the safe direction, but stale
enough to cost queue priority on every job:

| rule | `walltime:` configured | measured | over-request |
|---|---|---|---:|
| `add_demand` | `02:00:00` | 0:00:09 | 750x |
| `cluster_network` | `04:00:00` | 0:00:37 | ~390x |
| `build_renewable_profiles` | `04:00:00` | 0:00:43 – 0:01:49 | 130–330x |
| `add_electricity` | `04:00:00` | 0:02:09 | ~110x |
| `solve_network` | `20:00:00` | 0:11:13 | ~110x |
| `cluster_resources` | `05:00:00` | 0:07:49 | ~38x |
| `aggregate_to_substations` | `02:00:00` | 0:05:05 | ~24x |
| `build_fuel_prices` | `00:20:00` | 0:01:25 | ~14x |

`solve_network` is the one defensible margin: this observation is at `opts=3h` (2,920 snapshots) and
the campaign has since moved to full-year hourly snapshots (8,760), which triples the LP and grows
solve time superlinearly. The other seven are not defensible — a 4-hour request for a 37-second job
delays it behind everything else in the queue for no benefit.

### 5.5 Recommendations

#### (1) A `mem:` block mirroring `walltime:`

Give memory the same escape hatch wall time already has, so retuning a single rule for a single site
does not require touching `rules/*.smk`. In `workflow/repo_data/config/config.slurm.yaml`,
immediately after the `walltime:` block:

```yaml
# ====================================================================
# MEM — per-rule scheduler memory requests, in MB
# ====================================================================
# Peak-RSS overrides. This block is an ESCAPE HATCH, not the primary
# mechanism: only rules the size-aware estimator in rules/common.smk gets
# wrong belong here. If a rule is wrong for every interconnect, fix the
# estimator instead, so the fix generalises.
#
# Values are what the job requests BEFORE the `attempt` multiplier, so a
# rule that OOMs is automatically retried at 2x (see `retries` in
# workflow/profiles/sherlock/config.yaml).
#
# Refresh from measurements with:
#   python workflow/report_benchmarks.py workflow/benchmarks
# docs : MEM
mem: {}
  # aggregate_to_substations: 44000   # USA s300: measured 27.8 GB + 1.5x
  # cluster_resources:        44000   # USA s300: measured 27.7 GB + 1.5x
# docs : end
```

and in `workflow/schemas/config.schema.yaml`, beside the existing `walltime:` node (line 475):

```yaml
  mem:
    description: Per-rule scheduler memory requests in MB; owned by config.slurm.yaml.
    type: object
```

#### (2) One size-aware estimator in `rules/common.smk`, replacing 54 ad-hoc declarations

Keep `memory(w)`'s skeleton — wildcard-driven, `{opts}`-aware, horizon-multiplied — refit the
coefficients against §5.2, and generalise it to the three size regimes the DAG actually has. Drop it
in beside `memory(w)` and delete `memory(w)` itself.

```python
# --- Scheduler memory sizing (issues #808 / #811) ---------------------------
# Replaces 37 flat mem_mb constants and 17 `input.size // N` heuristics.
# input.size is the wrong predictor: it under-predicts by ~27x for the
# aggregation rules (compact netCDF in, dense O(n_bus^2) intermediates in RAM)
# and over-predicts by ~56x for cluster_network (1.1 GB pickle in, 134 buses
# out).  Size the request from the DAG-time wildcards instead, which is the one
# thing that IS known before the job runs.

MEM_FLOOR_MB = 2_000
MEM_CAP_MB = 750_000     # nothing may emit an unschedulable request
MEM_HEADROOM = 1.5

# Reference point for every coefficient below:
#   interconnect=usa, simpl=300, clusters=134a, opts=3h, 1 horizon, sector=E
REF_SIMPL, REF_CLUSTERS, REF_SNAPSHOTS = 300, 134, 2_920

# Relative model size per interconnect, normalised to usa=1.0.
# [REFIT: derive from len(n.buses) in each elec_base_network.nc rather than
#  from the placeholder ratios below.]
INTERCONNECT_SCALE = {"usa": 1.00, "eastern": 0.62, "western": 0.28, "texas": 0.10}

# (regime, peak RSS in MB at the reference point).  Regimes:
#   "ic"    pre-clustering            -> scales with interconnect size only
#   "simpl" post-cluster_resources    -> scales with the simpl bus count
#   "lp"    prepare/solve             -> clusters x snapshots x horizons x sector
# Sources: workflow/benchmarks/, 2026-09-01 USA equivalence run.
MEM_REF_MB = {
    "build_shapes":             ("ic",      400),
    "build_base_network":       ("ic",      600),
    "build_bus_regions":        ("ic",    2_200),
    "build_powerplants":        ("ic",    2_300),
    "build_fuel_prices":        ("ic",    1_400),
    "aggregate_to_substations": ("ic",   27_900),   # <-- see the note in 5.5(5)
    "cluster_resources":        ("ic",   27_700),   # <-- see the note in 5.5(5)
    "build_electrical_demand":  ("simpl", 10_400),
    "build_renewable_profiles": ("simpl",  3_000),
    "add_electricity":          ("simpl",  2_400),
    "add_demand":               ("simpl",    400),
    "cluster_network":          ("simpl",    500),
    "add_extra_components":     ("lp",       650),
    "prepare_network":          ("lp",       700),
    "add_sectors":              ("lp",       400),
    "solve_network":            ("lp",     7_400),
    "solve_network_validation": ("lp",     7_400),
}


def _snapshots(wildcards) -> int:
    """Snapshot count implied by an `Nh` token in {opts}; hourly otherwise."""
    for o in getattr(wildcards, "opts", "").split("-"):
        m = re.match(r"^(\d+)h$", o, re.IGNORECASE)
        if m:
            return HOURS_PER_YEAR // int(m.group(1))
    return HOURS_PER_YEAR


def estimate_mem_mb(rule: str, wildcards) -> float:
    """Peak-RSS estimate in MB from DAG-time wildcards."""
    regime, ref = MEM_REF_MB.get(rule, ("ic", 4_000))
    scale = INTERCONNECT_SCALE.get(getattr(wildcards, "interconnect", "usa"), 1.0)

    if regime == "ic":
        return ref * scale

    if regime == "simpl":
        simpl = str(getattr(wildcards, "simpl", "") or REF_SIMPL)
        return ref * scale * (int(simpl) / REF_SIMPL if simpl.isdigit() else 1.0)

    # "lp": the LP itself, so every dimension of the problem multiplies.
    clusters = re.sub(r"\D", "", str(getattr(wildcards, "clusters", ""))) or REF_CLUSTERS
    horizons = len(config_provider("scenario", "planning_horizons")(wildcards))
    sectors = len(str(getattr(wildcards, "sector", "E")).split("-"))
    return (
        ref
        * (int(clusters) / REF_CLUSTERS)
        * (_snapshots(wildcards) / REF_SNAPSHOTS)
        * horizons
        * sectors
    )


def scheduler_mem(rule: str, cap: int = MEM_CAP_MB):
    """Use as:  resources: mem_mb=scheduler_mem("cluster_resources")

    Resolution order:
      1. an explicit `mem: <rule>:` entry in config.slurm.yaml, else
      2. the size-aware estimate above, x MEM_HEADROOM.
    Then x attempt (so a `retries`-driven re-run doubles the request), clamped
    to [MEM_FLOOR_MB, cap] so neither a missing coefficient nor a weird wildcard
    can emit a request no partition will schedule.
    """
    from_config = config_provider("mem", rule, default=None)

    def _mem(wildcards, input, attempt):
        base = from_config(wildcards)
        if base is None:
            base = estimate_mem_mb(rule, wildcards) * MEM_HEADROOM
        return int(min(max(base * attempt, MEM_FLOOR_MB), cap))

    return _mem
```

Rule sites then collapse to one uniform line, which also removes the float-`mem_mb` defect from §2
(the `int(...)` is inside the helper):

```python
    resources:
        mem_mb=scheduler_mem("cluster_resources"),
        walltime=config_provider("walltime", "cluster_resources", default="01:00:00"),
```

#### (3) `retries: 2` in the profile as the OOM safety net

An estimator fitted on one run will be wrong somewhere. Because `scheduler_mem` multiplies by
`attempt`, a retry is a doubling, so first-attempt OOM self-corrects instead of failing the DAG.
Add to `workflow/profiles/sherlock/config.yaml`:

```yaml
retries: 2            # Snakemake 8; the 7.x spelling is `restart-times: 2` (already in §3.4)
```

This is **only** effective in combination with §3.5's `slurm-status.sh`: without a status hook,
Snakemake never learns that Slurm killed the job with `OUT_OF_MEMORY` and blocks forever instead of
retrying (§2, S1). The two changes must land together.

Note that rule-level `retries:` already exists on `build_fuel_prices` (3) and seven `retrieve_*`
rules (2–3) for *network flakiness*; the profile-level setting generalises it to *resource*
failures, and rule-level values continue to win where set.

#### (4) Opportunistic collection, not a dedicated sweep

`workflow/report_benchmarks.py` (143 lines, **branch-only**, not on `origin/develop`) is already the
right tool. It walks `benchmarks/` recursively, reads each TSV, recovers the rule name by matching
path components and stem prefixes against the real rule list parsed out of `rules/*.smk` (so it
handles both the `benchmarks/<run>/<interconnect>/<stem>` and the `benchmarks/<rule>/<interconnect>/<stem>`
layouts, and strips `_s{simpl}`/`_c{clusters}` wildcard debris), aggregates peak `max_rss` and peak
`s` per rule, and prints recommendations:

```
mem  = max(2 GB,   ceil(1.5 x peak max_rss / 500 MB) * 500 MB)
time = max(10 min, ceil(3   x peak runtime / 5 min)  * 5 min)
```

with the asymmetry deliberate — memory overrun is a hard kill, generous wall time only costs queue
priority.

Three changes make it the maintenance loop for §5.5(2):

* **Emit coefficients, not just requests.** Divide each peak by the size factor of the run that
  produced it (`INTERCONNECT_SCALE[ic] × simpl/300` etc.) and print a paste-ready `MEM_REF_MB` block.
  Without this, the output has to be re-derived by hand for every new footprint.
* **Promote it out of `workflow/`.** `workflow/report_benchmarks.py` sits next to the `Snakefile`
  where nothing else lives; `workflow/scripts/report_benchmarks.py` or `tests/report_benchmarks.py`
  is the right home.
* **Keep collection free.** The remaining rules without `benchmark:` — all 7 in
  `postprocess_sector.smk`, 3 in `postprocess.smk`, `build_co2_storage`, `aggregate_egs`,
  `plot_validation_figures`, `benchmark_cpuc_baseline`, and the 15 `retrieve_*` — should get one, so
  every ordinary run contributes data. A dedicated benchmarking sweep is not needed and would not be
  representative: `benchmark:` costs nothing and the equivalence campaign already produces
  USA-scale observations as a side effect.

#### (5) Before fitting: profile the 28 GB, do not enshrine it

`aggregate_to_substations` and `cluster_resources` are **topology-only** rules. The first maps buses
to substations through `bus2sub.csv`; the second runs kmeans down to `simpl=300` and writes a
busmap. Neither touches time series — `elec_s300.nc` is **593 KB** and `elec_b.nc` is 50 MB — yet
both peak at ~27.8 GB from ~100 MB of input, at ~90% single-core load for 5–8 minutes. A ~280x
input-to-RSS blow-up on a pure graph operation is a strong signal of a dense intermediate
(an n×n adjacency or distance matrix over ~17,890 substations, a non-sparse `pd.get_dummies`, or a
repeated concat) rather than an intrinsic cost.

Recommend profiling `scripts/aggregate_to_substations.py` and `scripts/cluster_simpl.py` with
`memory_profiler` (already a pinned dependency, `workflow/envs/environment.yaml`) **before** baking
`27_900` into `MEM_REF_MB`. Fitting first would make a 28 GB request permanent, push both rules onto
`bigmem` for every USA run, and hide the actual bug. If the blow-up is real, the coefficients stand;
if it is a dense intermediate, the fix is worth far more than the resource request. Either way the
two entries should carry a comment pointing back at #811 and at this measurement.
