(usage)=
# Usage

```{note}
**If you have not done so, please follow the [installation instructions](https://pypsa-usa.readthedocs.io/en/latest/about-install.html) [github issues](https://github.com/PyPSA/pypsa-usa/issues)**
```

## Set Configuration

To start, you'll want to set the proper network configuration for your studies purpose. The
default configuration in `config/config.default.yaml` (seeded from the tracked template by
`init_pypsa_usa.sh`) using the `western` interconnect and 33 nodes is a good place to start!

Your config file only needs to carry the keys you actually change: the workflow always
loads `repo_data/config/config.default.yaml` and the other layered files underneath it, and
your `--configfile` is merged on top. So there are two equally valid ways to set up a run —
copy the seeded `config/config.default.yaml` to `config/config.<scenario>.yaml` and edit it,
or write a short file holding only your overrides. Nested mappings merge key by key, but
lists and scalars are **replaced wholesale**, so any list you change must be restated in full.

You can find more information on each configuration setting on the [configurations page](https://pypsa-usa.readthedocs.io/en/latest/config-configuration.html).


## Run workflow

To run the workflow, `cd` into the `workflow` directory and run the `snakemake` from your terminal with your selection of config file:

UV:
```console
uv run snakemake -j1 --configfile config/config.default.yaml --scheduler-ilp-solver GUROBI_CMD
```

mamba:
```console
mamba activate pypsa-usa
snakemake -j1 --configfile config/config.default.yaml
```

### Generate Data Model

To generate the data model only, specify the rule `data_model` in the `snakemake` call. The `data_model` rule builds the network file that is passed into the `solve_network` rule, so it already carries everything `prepare_network` applies — the `{ll}` transmission limit and any `Co2L`/`CH4L`/`Ep` options in the `{opts}` wildcard. What it does **not** carry are the solve-time policy constraints (`RPS`, `REM`, `ERM`, `TCT`), and the network is not solved. It is written to `resources/{run name}/networks/{interconnect}/elec_s{simpl}_c{clusters}_ec_l{ll}_{opts}_{sector}.nc` (the `{run name}/` segment is dropped when `run: shared_resources: true`).

UV:
```console
uv run snakemake data_model -j1 --configfile config/config.default.yaml --scheduler-ilp-solver GUROBI_CMD
```

mamba:
```console
mamba activate pypsa-usa
snakemake data_model -j1 --configfile config/config.default.yaml
```


## Running on HPC Cluster

On a Slurm cluster, `workflow/run_slurm.sh` runs snakemake so that **each rule is submitted as its
own Slurm job** (through `workflow/slurm_submit.sh`). Every argument you give it is passed to
snakemake, so it takes the same targets and `--configfile` as a local run:

```console
cd workflow
bash run_slurm.sh --configfile config/config.my_scenario.yaml
```

Put a target, if you give one, *before* `--configfile`: `--configfile` takes several values and
would otherwise read the target as another config file.

### 1. Tell Slurm about your site, once

The scripts carry no site settings. Set Slurm's own `SBATCH_*` environment variables, which `sbatch`
reads whenever the matching flag is not given, for example in `~/.bashrc`:

| Variable | Meaning | Sherlock (Stanford) | Alliance clusters, e.g. Fir |
|---|---|---|---|
| `SBATCH_ACCOUNT` | allocation to charge | leave unset (no accounts) | `def-<pi>` or `rrg-<pi>`; required |
| `SBATCH_PARTITION` | partition / queue | e.g. `serc`, `normal` | leave unset; jobs are routed by size |
| `SBATCH_MAIL_USER` | job e-mail (failures only, unless `SBATCH_MAIL_TYPE` is set) | optional | optional |

Anything else `man sbatch` lists as an input environment variable (`SBATCH_QOS`,
`SBATCH_CONSTRAINT`, ...) works the same way. Check your site's documentation for the values;
the table above is a starting point, not a guarantee.

`SBATCH_MAIL_USER` and `SBATCH_MAIL_TYPE` are the exception: `sbatch` has no environment variable
for mail, so `workflow/slurm_submit.sh` reads these two and passes them as `--mail-user` and
`--mail-type`. They therefore apply to the rule jobs only, not to an `sbatch` you run yourself.

Load any modules your site needs (a Gurobi license, a newer `git`) before starting the driver: the
rule jobs inherit its environment.

### 2. Run the driver inside a job, not on a login node

`run_slurm.sh` is the snakemake scheduler. It stays alive for the whole workflow, submitting rule
jobs and polling their state, which most sites forbid on login nodes. Submit it as a small,
long job of its own:

```console
cd workflow
mkdir -p logs
sbatch --time=2-00:00:00 -c 2 --mem=8G -o logs/driver-%j.out \
    --wrap "bash run_slurm.sh --configfile config/config.my_scenario.yaml"
```

For a short or interactive run, start it inside an interactive allocation instead (`sh_dev` on
Sherlock, `salloc` elsewhere). The script warns if it is not inside a Slurm job.

### 3. Tune and size the run

Optional environment variables for `run_slurm.sh`:

- `JOBS` — rule jobs in flight at once (default 20).
- `RESTART_TIMES` — retries for a failed rule (default 0). Rules that scale memory with `attempt`
  ask for more on each retry.
- `LATENCY_WAIT` — seconds to wait for outputs to appear on a shared filesystem (default 60).
- `SNAKEMAKE` — snakemake command: an executable, or a launcher such as `uv run snakemake`
  (default: the repository `.venv`, else the one on `PATH`).

Wall times per rule come from the `walltime:` block of `config.slurm.yaml`
({ref}`walltime_cf`); override entries in your copy at `workflow/config/config.slurm.yaml`.
Memory and threads come from the `resources: mem_mb:` and `threads:` fields of each rule.
Rules that declare neither get 8 GB and 2 hours.

### 4. Logs and failures

The driver's log is the `-o` file you gave `sbatch`; each rule job writes
`workflow/logs/slurm/<rule>-<jobid>.out` and `.err`. A rule that Slurm kills (out of memory,
time limit, node failure) is reported as failed through `workflow/slurm_status.sh`, so the driver
stops instead of waiting forever. A job that Slurm requeues after a preemption or node failure
counts as still running. Interrupting the driver with Ctrl-C also cancels its rule jobs;
if you `scancel` the driver job instead, check `squeue` for leftover rule jobs. Re-running the same
command resumes where it stopped.

### Weather-year campaigns

`workflow/run_slurm_weather_years.sh` runs a batch of California weather-year overlays through
`run_slurm.sh`, scheduling them so that overlays sharing input data do not collide. Each overlay is
a small config file in `workflow/config/weather_years/`, and that directory's `manifest.tsv` maps an
overlay filename to its run name and build target (tab-separated, three columns). Both are yours to
create:

```console
bash run_slurm_weather_years.sh ca2040_wy2019_z4.yaml ca2040_wy2019_county.yaml
```

`CONCURRENCY` sets how many overlays run at once. Driver logs land in `workflow/logs/drivers/`.

## Examine Results

### After the run: where outputs land

All outputs are written to `workflow/results/{run name}/{interconnect}/`, where `{run name}`
is the `run: name:` field of your configuration file (`Default` in
`config/config.default.yaml`). Inside you will find three folders: `networks/` holds the
solved network files, `figures/` holds automatically generated maps and plots along with
summary CSVs (capacities, generation, statistics) in its `statistics/` subfolders, and
`configs/` holds a snapshot of the configuration used for the run. A good first stop is the
`figures/` folder to sanity-check the system maps and statistics before diving into the
network file itself.

### Analyzing a solved network

To further analyze the results of a solved network, you can use pypsa to open the
`elec_s{simpl}_c{clusters}_ec_l{ll}_{opts}_{sector}.nc` file in the
`results/{run name}/{interconnect}/networks/` folder. With the default configuration this is
`results/Default/western/networks/elec_s75_c33_ec_lv1.0_REM-3h_E.nc`.

The filename encodes the scenario. Each component is described briefly below; see the
[wildcards page](https://pypsa-usa.readthedocs.io/en/latest/config-wildcards.html) for full
details:

| Component | Example | Meaning |
|-----------|---------|---------|
| `s{simpl}` | `s75` | Number of buses after pre-clustering simplification |
| `c{clusters}` | `c33` | Final number of clustered buses (zones) |
| `ec` | `ec` | Fixed marker: extra components (e.g. storage) have been added |
| `l{ll}` | `lv1.0` | Transmission expansion limit (`v`olume or `c`ost, factor or `opt`) |
| `{opts}` | `REM-3h` | Dash-separated options (here: regional emissions limit, 3-hourly resolution) |
| `{sector}` | `E` | Sectors included (`E` = electricity only, `E-G` adds natural gas) |

(troubleshooting)=
## Troubleshooting:

To force the execution of a portion of the workflow up to a given rule, cd to the `workflow` directory and run:

```console
uv run snakemake -j4 -R build_shapes --until build_base_network --configfile config/config.default.yaml
```
where `build_shapes` is forced to run, and `build_base_network` is the last rule you would like to run.

```{note}
`--configfile` is optional: the Snakefile loads `repo_data/config/config.default.yaml` as
the base layer, so omitting it simply runs the shipped defaults under `run: name: "Default"`.
Pass `--configfile` to select your own scenario and give it its own `run: name:`.
```
