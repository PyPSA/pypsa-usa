# Running PyPSA-USA on Sherlock

Everything here assumes Stanford's Sherlock cluster and this checkout. Nothing
below runs a scheduler, a solve, or a Python interpreter on a login node.

Design notes and the defect list this replaces:
[`docs/audits/slurm-execution-audit.md`](audits/slurm-execution-audit.md).
The equivalence harness has its own driver and its own rules:
[`docs/equivalence.md`](equivalence.md).

---

## 1. The three run modes

| | interactive | monolithic (one node) | per-rule submission |
|---|---|---|---|
| where the scheduler runs | your `sh_dev` shell | inside one Slurm job | inside one Slurm job |
| where the rules run | same shell | same job | one Slurm job each |
| entry point | `sh_dev` + `snakemake` | `run_slurm.sbatch --local` | `run_slurm.sbatch` |
| per-rule `sacct` numbers | no | no | yes |

### Decision rule

Pick by **where the wall-clock goes**, not by taste. `n_jobs` = rules in the
DAG, `T_longest` = expected wall of the slowest single rule.

| condition | mode | why |
|---|---|---|
| Whole DAG under about 30 min and peak rule memory under about 32 GB (tutorial/test config, 1 interconnect, `24h`/`seg` resolution, 1 horizon) | **interactive** | queueing 60 jobs to save 20 minutes is a loss; you also want a live traceback |
| DAG is small (about 40 jobs or fewer) *and* homogeneous in resource need *and* total wall under about 24 h (single-interconnect benchmark, e.g. `western`/`texas` at coarse `clusters`) | **monolithic** | one queue wait instead of 40 |
| DAG is large or heterogeneous (USA interconnect, multiple `clusters`/`ll`/`opts` in `scenario:`, sector coupling), or any rule needs more than about 180 GB or more than 24 h, or several configs run at once | **per-rule submission** | a 14 GB, 12-min solve should not hold a 160 GB reservation; the fan-out rules parallelise across nodes |

Two overrides that beat the size rule:

* **Equivalence and benchmark comparisons always use monolithic**, even when
  the size rule says per-rule, unless the run is too big to fit one node.
  Equivalence is the project's acceptance criterion and per-rule submission
  adds environment variance between the two arms for no benefit.
* **Memory-profiling runs always use per-rule submission**, because `sacct
  MaxRSS` and the `benchmark:` TSVs are per job — a monolithic run gives you
  one peak number for the whole DAG, which cannot tell you *where* the memory
  went.

---

## 2. Exact commands

### Interactive

```bash
sh_dev -c 8 -m 32GB -t 02:00:00 -p serc          # or: salloc -p serc -c 8 --mem=32G --time=1:00:00
ml load devel uv/0.10.8
ml load system git/2.45.1
ml load gcc/12.4.0                                # AFTER uv: the uv module pulls gcc/14 back in
export UV_CACHE_DIR=$SCRATCH/pypsa-usa/cache/uv XDG_CACHE_HOME=$SCRATCH/pypsa-usa/cache/xdg
cd $REPO && uv sync --frozen --extra dev          # once
cd workflow && uv run snakemake -j8 --configfile repo_data/config/config.tutorial.yaml
```

Never for a solve, and never held while idle.

### Monolithic — one job runs the whole DAG

```bash
mkdir -p logs/controller
sbatch -c 16 --mem=160G --time=24:00:00 \
    workflow/run_slurm.sbatch --local repo_data/config/config.california.yaml
```

`-j` is taken from `$SLURM_CPUS_PER_TASK`, so the one number on the sbatch line
sets both.

### Per-rule submission — the controller submits one job per rule

```bash
mkdir -p logs/controller
sbatch workflow/run_slurm.sbatch config/config.usa.benchmark.yaml
sbatch workflow/run_slurm.sbatch config/config.usa.benchmark.yaml solve_network   # explicit target
sbatch --time=12:00:00 --mem=16G workflow/run_slurm.sbatch config/....yaml        # resize the controller
```

The controller is small and long (`-p serc`, 2 cpu, 8 GB, 4 days by default;
`serc` allows up to 7 days). It must outlive the whole DAG.

### A matrix of configs

```bash
$EDITOR workflow/run_slurm_all.sh          # comment configfiles in and out
bash workflow/run_slurm_all.sh             # one controller per config, concurrent
PYPSA_CHAIN=1  bash workflow/run_slurm_all.sh   # back to back, --dependency=afterok
PYPSA_REPORT=1 bash workflow/run_slurm_all.sh   # + run_report.sbatch afterok on all
PYPSA_DRYRUN=1 bash workflow/run_slurm_all.sh   # print the sbatch lines, submit nothing
```

Concurrent controllers share one working directory, so the launcher passes
`--nolock`. **That is safe only because every config in the array has a
distinct `run.name`**, which the launcher checks before submitting anything;
`run.name` is what puts each config's outputs in its own `resources/` and
`results/` subtree. Never set `PYPSA_NOLOCK=1` by hand to get past a lock left
by a crashed job — the controller already clears stale locks at startup.

### Reporting and statistics

```bash
sbatch workflow/run_report.sbatch                                    # all three groups
PYPSA_REPORT_GROUPS="benchmarks stats" sbatch workflow/run_report.sbatch
bash workflow/collect_solve_statistics.sh                            # login-node safe
uv run python workflow/report_benchmarks.py workflow/benchmarks      # compute node
```

---

## 3. Where things go

| stream | path | filesystem |
|---|---|---|
| controller Slurm capture | `slurm-ctrl-<name>-<jobid>.{out,err}` in the submit dir | wherever you submitted from |
| controller tee'd log | `logs/controller/ctrl-<jobid>.log` | repo (gitignored) |
| per-rule Slurm capture | `$PYPSA_SLURM_LOGDIR/<rule>/<rule>-<jobid>.{out,err}` | `$SCRATCH` |
| snakemake `log:` files | `workflow/logs/...` | repo |
| `benchmark:` TSVs | `workflow/benchmarks/...` | repo |
| results | `workflow/results/...` | repo, or `$PYPSA_RESULTS_ARCHIVE` |

`$SCRATCH` is purged after 90 days without a content write. Anything that has
to survive goes to Oak; set `PYPSA_RESULTS_ARCHIVE` and the controller rsyncs
`results/` there on success.

`sbatch` does not create `-o`/`-e` directories and a missing one kills the job
at launch with no diagnostic. The controller creates one `$PYPSA_SLURM_LOGDIR/<rule>`
per rule before Snakemake starts; you create `logs/controller/` once.

---

## 4. Site settings

One place: `workflow/snakemake_profiles/sherlock/config.yaml`. Not a
`--cluster-config` YAML — that interface is deprecated in Snakemake 7 and gone
in 8, and a blank value in it renders as `-A ''`, which is a submission error
rather than "use the default".

Per-user overrides are environment variables with tracked defaults:

| variable | default | effect |
|---|---|---|
| `PYPSA_SLURM_PARTITION` | `serc` | partition for the child jobs |
| `PYPSA_SLURM_LOGDIR` | `$SCRATCH/pypsa-usa/slurm-logs/<ctrl jobid>` | per-rule Slurm capture |
| `PYPSA_JOB_TAG` | the controller's job id | child jobs are named `smk-<tag>-<rule>`; scopes `scancel` |
| `PYPSA_SBATCH_EXTRA` | empty | extra sbatch flags, e.g. `--mail-type FAIL --mail-user you@stanford.edu` |
| `PYPSA_CACHE_ROOT` | `$SCRATCH/pypsa-usa/cache` | uv / pip / matplotlib caches |
| `GRB_LICENSE_FILE` | the site Gurobi 11.0.2 licence | solver licence |
| `PYPSA_RESULTS_ARCHIVE` | unset | rsync `results/` here on success |

**`-A`/`--account` is not needed on Sherlock.** `sacctmgr show assoc
user=$USER` returns a single association (`iazevedo`), which is therefore the
default account; accounting is enforced
(`AccountingStorageEnforce=associations,limits,qos,safe`) but the default
resolves. Add one through `PYPSA_SBATCH_EXTRA` if your account list ever grows.
Verified 2026-09-14.

Partition facts, from `sh_part` on 2026-09-14: `serc` — not public, 383 nodes,
24–256 cores and 191–2048 GB per node, default job time 2 h, **maximum 7 days**,
QOS `normal,long`. `normal` — max 7 d, 128–384 GB/node. `bigmem` — max 1 d,
384–4096 GB/node. `dev` — max 2 h.

---

## 5. Known costs

* **All 15 `retrieve_*` rules are submitted as their own Slurm jobs** in
  per-rule mode, so seconds-long downloads wait in the queue. Snakemake
  7.32.4 has no `--local-rules` (that is 8.x); the 7.x lever is the
  `localrules:` directive in `workflow/Snakefile`, which is a repo-wide
  statement rather than a site one. Warm the cache once in monolithic or
  interactive mode and this costs nothing afterwards.
* **Three rules have `set-resources:` overrides in the profile**
  (`aggregate_to_substations`, `cluster_resources`, `solve_network`), because
  their declared `mem_mb` formulas are measurably wrong at USA scale — two by
  26–29x *under* (certain OOM) and one by ~40x *over* (unschedulable). An
  override **replaces** the rule's `attempt` multiplier, so `restart-times: 2`
  cannot rescue those three; retune them from measurements instead. The values
  are sized for the USA footprint; comment them out for tutorial/California
  runs. See audit sections 5.2–5.3 and issues #808 / #811.
* **17 of 55 `mem_mb` declarations scale with `attempt`** and therefore
  self-correct after an OOM; 33 are flat constants and 5 rules declare no
  `mem_mb` at all (`plot_natural_gas`, `retrieve_sector_databundle`,
  `retrieve_res_eulp`, `retrieve_com_eulp`, `docs_figures`) and fall back to
  the profile's `default-resources: mem_mb=8000`. A flat rule that OOMs will
  OOM again on the retry — raise it in the profile, or in the rule.

---

## 6. Rules for an agent working in this repository

1. **Never run Python, `pytest`, `snakemake`, or a solve on the login node.**
   Not even "just a quick check". Get a node first:
   `sh_dev -c 4 -m 16GB -t 02:00:00 -p serc`, or submit a job.
   Login-node-safe here: `git`, `squeue`, `sacct`, `scancel`, `sh_part`,
   `bash -n`, and `workflow/collect_solve_statistics.sh` (pure text parsing).
2. **Write a script, then submit it.** Do not compose a long `srun ... bash -c`
   one-liner. The script is the record of what ran, and it is reviewable before
   it burns a node-hour.
3. **Poll `squeue --me` no more often than once every two minutes.** Never in a
   tight loop. `sacct -j <id> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,ReqMem`
   for a post-mortem, `seff <id>` for the efficiency summary.
4. **Never guess a module version.** `ml spider <name>` first. Load order here
   is `devel uv/0.10.8`, then `system git/2.45.1`, then `gcc/12.4.0` — the uv
   module reloads `gcc/14.2.0`, which hard-errors on
   `incompatible-pointer-types` and breaks the `datrie` source build.
5. **Caches and job I/O off `$HOME`.** `UV_CACHE_DIR`, `PIP_CACHE_DIR`,
   `XDG_CACHE_HOME`, `MPLCONFIGDIR` all under `$SCRATCH` or `$GROUP_SCRATCH`.

### Submissions an agent may make without asking first

A single `sbatch` is auto-allowed when **every** one of these holds:

| resource | ceiling |
|---|---|
| memory | 64 GB |
| wall time | 4 h |
| CPUs | 8 |
| GPUs | 0 |
| partition | `serc`, `normal` or `dev` |

Anything above a ceiling — a USA solve, the 160 GB monolithic mode, the 4-day
controller, `bigmem`, `owners`, `--qos=long` — is a submission to ask about
first. Cancelling your own jobs (`scancel` scoped to this run's
`smk-<tag>-` names) never needs asking.
