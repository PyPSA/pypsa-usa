# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PyPSA-USA is a Snakemake workflow that builds and solves PyPSA energy-system optimization models of the US bulk power system (and optional sector coupling). Everything runs out of `workflow/` — `cd workflow` before invoking `snakemake`. Python is pinned to 3.11; dependencies are pinned exactly in `pyproject.toml` / `uv.lock`.

## Common commands

All snakemake commands run from `workflow/`. Choose ONE env manager (mamba or uv) — not both.

```bash
# First-time setup: copy config templates into workflow/config/
bash init_pypsa_usa.sh

# Full pipeline (solves + plots)
uv run snakemake -j1 --configfile config/config.default.yaml --scheduler-ilp-solver GUROBI_CMD
# or: mamba activate pypsa-usa && snakemake -j1 --configfile config/config.default.yaml

# Build only the input data model (network .nc, no solve)
uv run snakemake data_model -j1 --configfile config/config.default.yaml

# Force-rerun a subset, stopping at a target rule
snakemake -j4 -R build_shapes --until build_base_network

# DAG of the workflow
snakemake --rulegraph all | sed -n "/digraph/,\$p" | dot -Tjpg -o repo_data/dag.jpg

# Wipe resources/ and results/ (keeps data/)
snakemake clean

# HPC (SLURM)
bash run_slurm.sh   # edit --configfile inside the script first
```

Tests (pytest) live in `workflow/scripts/test/` and use the GLPK solver. Run from that directory or with that as cwd:

```bash
cd workflow/scripts/test && pytest -v
pytest test_reserves.py::test_erm_peak_demand_hour -v
pytest -v -k "storage"
```

Lint/format with ruff (config in `pyproject.toml`, line-length 120, py311 target):

```bash
ruff check --fix workflow/scripts
ruff format workflow/scripts
pre-commit run --all-files
```

## Architecture

### Snakemake is the orchestrator
- `workflow/Snakefile` is the entrypoint. It loads layered config (`config.cluster.yaml` → `config.common.yaml` → `config.plotting.yaml` → `config.api.yaml` → `config.sector.yaml`) and then any user `--configfile`, then `include:`s rule files from `workflow/rules/*.smk`.
- Rule files group the pipeline stages: `retrieve.smk` (downloads), `build_electricity.smk` (network construction), `build_sector.smk` (sector coupling), `solve_electricity.smk`, `postprocess.smk` / `postprocess_sector.smk`, `validate.smk`, and `common.smk` (shared helpers).
- Each rule's `script:` points at a Python file in `workflow/scripts/`. Scripts read inputs/outputs/params via the injected `snakemake` object — they aren't standalone CLIs.
- Two convenience entrypoints: `rule all` (figures, the default) and `rule data_model` (build the prepared network `.nc` without solving).

### Wildcards thread through filenames
Wildcard constraints declared in `Snakefile`:
- `interconnect`: `usa|texas|western|eastern`
- `simpl`: simplification suffix (e.g. number of pre-clustering nodes)
- `clusters`: `[0-9]+m?+a?+c?|all` — `m`/`c` suffixes change memory scaling in `common.smk:memory()`
- `ll`: line-limit code, e.g. `v1.25`, `copt`, `vall`
- `opts`: dash-separated option flags consumed by scripts (e.g. `Co2L-3h-25seg`)
- `sector`: dash-joined letters from `{E, G}` (electricity, natural gas). Auto-prefixed with `E-` if missing.

The canonical output filename pattern is:
`elec_s{simpl}_c{clusters}_ec_l{ll}_{opts}_{sector}.nc` (data model) and similar under `results/{interconnect}/...`.

### Resources layout is category-first
`Snakefile` defines top-level path constants — files are grouped by *what they are*, not by interconnect:
`NETWORKS`, `BUSMAPS`, `PROFILES`, `GEOSPATIAL`, `COSTS`, `PRICES`, `POWERPLANTS`, `DEMAND`, `HEATING_COP`, `TEMPERATURE`, `POPULATION`, `CO2` — each is `RESOURCES + "<name>/"`. When adding a new output, route it through one of these constants (or add a new category) instead of inventing an ad-hoc path. `RESOURCES` itself toggles between shared and per-run based on `run.shared_resources`.

### Config provider pattern (common.smk)
Rules don't read `config[...]` directly. They use `config_provider("key1", "key2", default=...)` from `workflow/rules/common.smk`. If `run.scenarios.enable` is true, this resolves config dynamically per `wildcards.run` via a scenarios file; otherwise it's a static lookup with wildcard interpolation. When adding rule params, follow this pattern so scenario overrides keep working.

### Pipeline stages (high level)
1. **Retrieve** (`retrieve.smk`) — pull Zenodo bundles, EIA, PUDL, NREL exclusion, EGS, gridemissions, etc.
2. **Build base network** — shapes → bus regions → base network → cost data → powerplants → renewable profiles → demand.
3. **Simplify & cluster** — `cluster_simpl.py` then `cluster_network.py` reduce nodes per `simpl`/`clusters` wildcards.
4. **Add components** — `add_electricity.py`, `add_extra_components.py`, `add_demand.py`, `add_sectors.py`.
5. **Prepare & solve** — `prepare_network.py` applies `opts` modifiers, `solve_network.py` calls the solver. Optional add-on constraints live in `workflow/scripts/opts/` (`reserves.py`, `policy.py`, `land.py`, `interchange.py`, `bidirectional_link.py`, `sector.py`).
6. **Postprocess** — `summary*.py`, `plot_*.py` produce figures organized by category (maps / emissions / production / system / validate).

### Solver
Default is Gurobi (pinned `gurobipy==11.0.3`); HiGHS and GLPK also supported. Tests use GLPK. The EIA API key (required by default config for dynamic fuel prices) goes in `config/config.api.yaml`.
