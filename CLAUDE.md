# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PyPSA-USA is a Snakemake-orchestrated PyPSA workflow for capacity expansion, production-cost simulation, and power-flow analysis of the US bulk transmission system. Configuration is layered YAML; intermediate and final artifacts are netCDF/CSV/GeoJSON files produced by rules in `workflow/rules/*.smk` and Python scripts in `workflow/scripts/`.

## Running the workflow

**All `snakemake` invocations run from `workflow/`** — `cd workflow/` first. Snakemake auto-loads `config/config.cluster.yaml`, `config.common.yaml`, `config.plotting.yaml`, `config.api.yaml`, `config.sector.yaml` from `workflow/Snakefile`; the main configfile is passed via `--configfile`.

```bash
cd workflow
uv run snakemake -j1 --configfile config/config.default.yaml --scheduler-ilp-solver GUROBI_CMD
# or under mamba: snakemake -j1 --configfile config/config.default.yaml
```

Useful targets:
- `rule data_model` — build everything up to the assembled-but-unsolved network (no solver).
- `rule all` — full pipeline including solve and figures.
- `--until <rule>` to stop early, `-R <rule>` to force re-execution.
- Tutorial config (`config/config.tutorial.yaml`, CA only, simpl=75, clusters=4m, 2050) is the smallest meaningful end-to-end run.

HPC: edit `config/config.cluster.yaml` (account/partition/email) and `workflow/run_slurm.sh`, then `bash workflow/run_slurm.sh`.

## Tests and lint

- Unit tests live in `workflow/scripts/test/` (covers constraint/helper logic only — `test_land.py`, `test_policy.py`, `test_reserves.py`). They are not end-to-end pipeline tests.
  ```bash
  cd workflow/scripts && pytest test/
  pytest test/test_policy.py::test_name  # single test
  ```
- `conftest.py` provides a `base_network` PyPSA fixture (3-bus network with wind/solar/gas) — extend it rather than building fresh networks per test.
- Pre-commit hooks (snakefmt, ruff, ruff format, blackdoc, pyupgrade, pretty-format-yaml, add-trailing-comma, jupyter-notebook-cleanup) run on `git commit`. Hooks modify files on first run; re-stage and re-commit. **Do not bypass with `--no-verify`.**
- `.github/workflows/main.yml` runs `./test.sh` which **does not exist in the repo** — CI's "Test snakemake workflow" step is effectively a no-op today. Treat green CI accordingly.

## Architecture: the DAG and the resources/ layout

The workflow was recently refactored (the "simplify-early" stack) so that topology aggregation and `{simpl}` kmeans clustering happen **before** the per-bus heavy rules. The clustered network is the input to demand/RE/electricity assembly, not the substation-level network. This is the load-bearing fact about the rule graph:

```
build_base_network → build_bus_regions
  → aggregate_to_substations (elec_b.nc, busmap_b.csv)        # topology only
  → cluster_resources            (elec_s{simpl}.nc, regions_*_s{simpl}.geojson, busmap_s{simpl}.csv)
  → build_renewable_profiles (profile_{tech}_s{simpl}.nc)
  → build_*_demand           (demand outputs keyed to cluster bus)
  → add_demand               (elec_s{simpl}_dem.nc)
  → add_electricity          (elec_s{simpl}_l_pp.pkl)         # dill pickle
  → cluster_network          (elec_s{simpl}_c{clusters}.nc)
  → add_extra_components → prepare_network → solve_network
```

EGS gets a parallel `aggregate_egs` rule (gated on `EGS in extendable_carriers.Generator`) that remaps NREL substation-keyed supply curves through `busmap_s{simpl}.csv` before they reach `add_electricity`. HAC clustering has been removed entirely; only `kmeans` and `modularity` are supported in `cluster_network`.

### Category-first resources/ layout

`workflow/Snakefile` defines category constants — **always use them, never hard-code `RESOURCES + "{interconnect}/foo.nc"`**:

| Constant | Path | What goes here |
|----------|------|----------------|
| `NETWORKS` | `resources/networks/` | `.nc` and `.pkl` networks (`elec_base_network.nc`, `elec_b.nc`, `elec_s{simpl}.nc`, `elec_s{simpl}_dem.nc`, `elec_s{simpl}_l_pp.pkl`, `elec_s{simpl}_c{clusters}.nc`) |
| `BUSMAPS` | `resources/busmaps/` | `bus2sub.csv`, `sub.csv`, `busmap_b.csv`, `busmap_s{simpl}.csv`, `busmap_s{simpl}_{clusters}.csv`, `linemap_*.csv` |
| `PROFILES` | `resources/profiles/` | `profile_{tech}_s{simpl}.nc`, `specs_EGS_s{simpl}.nc`, `profile_EGS_s{simpl}.nc`, NREL mapping cache |
| `GEOSPATIAL` | `resources/geospatial/` | All `*_shapes.geojson`, `regions_*.geojson`, `reeds_shapes.geojson` |
| `COSTS` | `resources/costs/` | `costs_{year}.csv`, `sector_costs_{year}.csv` |
| `PRICES` | `resources/prices/` | Fuel-price CSVs |
| `DEMAND` | `resources/demand/` | Per-end-use demand CSVs and pickles |
| `POWERPLANTS` | `resources/powerplants/` | `powerplants.csv` (literal path, not under `RDIR`, intentionally shared across runs) |
| `HEATING_COP`, `TEMPERATURE`, `POPULATION`, `CO2` | similarly named subdirs | Sector-coupling intermediates |

`RESOURCES` itself = `resources/` + `RDIR` (where `RDIR = run.name + "/"` unless `shared_resources: true`).

### Wildcards

Defined in `workflow/Snakefile`:
- `interconnect`: `usa | texas | western | eastern`
- `simpl`: alphanumeric or `all` — pre-clustering granularity
- `clusters`: integer optionally suffixed with `m`/`a`/`c`, or `all` — final cluster count
- `ll`: `v|c` + number/`opt`/`all` — line-limit scenario
- `opts`: dash/plus-separated options string (transmission, horizon discretization, etc.)
- `sector`: `E`, `G`, or hyphenated combos

### Script conventions

- Scripts in `workflow/scripts/` access inputs/outputs/params/config via the `snakemake` global (`snakemake.input.X`, `snakemake.output.X`, `snakemake.params.X`, `snakemake.config[...]`). Paths are passed in by the rule — scripts are layout-agnostic.
- `_helpers.py` is the shared utility module; many scripts add `..` to `sys.path` and import from it.
- Networks are loaded with `pypsa.Network(path)` for `.nc` and `dill.load(open(path, "rb"))` for `.pkl` (the `add_electricity` output is pickled).

## Configs

- `config/config.default.yaml` — primary user-facing config (Western, default knobs).
- `config/config.tutorial.yaml` — minimal CA-only smoke run.
- `workflow/repo_data/config/` mirrors `config/` and is the source for `docs/source/configtables/` documentation.
- Layered configs in `config/config.{cluster,common,api,plotting,sector}.yaml` are merged automatically by `Snakefile`; the main configfile only overrides what it needs to.

## Things to know before changing rules

- The `{simpl}` wildcard is now load-bearing on most rule outputs downstream of `cluster_resources`. Adding a new per-bus rule? Its inputs and outputs should both carry `_s{simpl}` and consume `NETWORKS + "{interconnect}/elec_s{simpl}.nc"` (or `_dem.nc` / `_l_pp.pkl` depending on stage), not `elec_base_network.nc`.
- When adding precomputed substation-level data (like EGS), pattern after `aggregate_egs`: take the rule's input keyed by `sub_id`, remap through `busmap_s{simpl}.csv` (capacity-weighted means for intensive quantities, sums for extensive), output keyed by cluster bus.
- `bus_strategies` for clustering: `Pd` sums, `reeds_zone`/`county`/`balancing_area` take `first` of the cluster (representative). If new downstream code needs the full many-to-one mapping, read `busmap_*.csv` directly rather than relying on the representative attribute.
- Custom busmaps must now key on `cluster_resources`-output bus IDs, not raw substation IDs.

## Environment

- `pypsa==1.3.0`, `linopy==0.9.1`, `pandas==3.0.5`, `xarray==2026.7.0`, `geopandas==1.1.4`, `atlite==0.3.0`, numpy held at `1.26.0` (rasterio/atlite ABI), Python `>=3.11, <3.12` (conda) / `>=3.11` (uv). Dependency pins in both `pyproject.toml` and `workflow/envs/environment.yaml` — keep them in sync if you bump. Migration notes: `docs/pypsa-v1-migration.md`.
- Gurobi is the default ILP scheduler. `highspy` is in deps as a fallback solver.
