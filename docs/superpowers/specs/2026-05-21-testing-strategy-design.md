# PyPSA-USA Testing Strategy — Design

**Date:** 2026-05-21
**Status:** Approved (brainstorming complete)
**Author:** ktehranchi (with Claude)

## Motivation

The PyPSA-USA workflow recently absorbed a large refactor (the "simplify-early" stack: PRs #7-#12 against `v1-epic`) that restructured rule ordering, repointed every per-bus rule's inputs, and reorganized `resources/` into a category-first layout. Nothing in the repo's current test infrastructure would have caught a regression in any of those changes:

- The unit tests in `workflow/scripts/test/` cover narrow constraint/helper logic (`test_land.py`, `test_policy.py`, `test_reserves.py`) and never exercise the snakemake DAG.
- The CI workflow at `.github/workflows/main.yml` runs `./test.sh`, but `test.sh` does not exist in the repo. The "Test snakemake workflow" CI step is silently a no-op.
- Recent breakage that should have been caught: PRs #9 and #11 were originally merged into their stacked base branches rather than `v1-epic` (the changes never propagated up); the snakemake DAG silently lost the load-bearing memory-win wiring until manually discovered.

This spec defines a tiered test pyramid that runs as a pre-merge gate, designed to catch the classes of breakage that the recent refactor exposed: path/wiring drift, dead config keys, script-level crashes, and silent artifact-shape regressions.

## Scope and non-goals

**In scope:**
- A `tests/` layout, pytest marker scheme, and CI wiring that runs on every PR.
- Static checks: snakemake DAG dry-run, rule input/output path validation, config-key validation.
- A small end-to-end integration build on a new minimal test config, with artifact-shape assertions after each stage.

**Explicitly out of scope:**
- Numerical regression against committed baseline `.nc` files (deferred — see "Out of Scope" below).
- Schema-catalog assertions on artifact columns. These depend on the parallel schema-tracking initiative (`docs/superpowers/specs/2026-05-21-network-schema-tracking-design.md`) and will land as a sixth PR after both this spec and that one are implemented.
- Memory-regression assertions. Defer to a separate per-rule benchmarking effort.
- Replacing or migrating the existing `workflow/scripts/test/*` unit tests — they stay in place, just get collected by the new pytest configuration.

## Tier structure

Two tiers, both required for merge:

| Tier | Test budget | Data deps | What it asserts | Marker |
|------|-------------|-----------|-----------------|--------|
| A — static | <30s | None | DAG resolves on tutorial config; every rule path uses a category constant and resolves to a producer; every `snakemake.config[...]` key in scripts exists in YAML; existing unit tests pass | `fast` |
| B — integration | <5min | `data/`, `cutouts/`, `repo_data/` (cached in CI) | Tutorial-shaped build to `cluster_network` succeeds; every produced artifact has expected bus count, no NaN in load-bearing columns, roundtrips through netCDF, filename wildcards are intact | `integration` |

*"Test budget" measures the pytest invocation only.* CI wall-clock is larger because each job pays for environment setup (Python install for `fast-tests`, micromamba env build for `e2e-tests`) on cold cache.

A third tier (numerical regression against baseline solved networks) is acknowledged as valuable but deferred — see "Out of Scope."

## Layout

```
tests/
├── conftest.py              # shared fixtures (subprocess helpers, repo paths)
├── pytest.ini               # marker registration
├── static/
│   ├── __init__.py
│   ├── test_dag_dryrun.py   # `snakemake -n --until cluster_network|solve_network` exits 0
│   ├── test_paths.py        # walks .smk files via snakemake Python API
│   └── test_config_keys.py  # AST-walks workflow/scripts/*.py
└── integration/
    ├── __init__.py
    ├── conftest.py          # session-scoped fixture: runs snakemake once into temp run dir
    └── test_artifacts.py    # loads each .nc/.pkl, asserts shape

workflow/scripts/test/       # UNCHANGED — collected via pyproject.toml testpaths
```

`pyproject.toml` adds:

```toml
[tool.pytest.ini_options]
testpaths = ["tests", "workflow/scripts/test"]
markers = [
    "fast: tier A — static checks, no data deps, target <30s",
    "integration: tier B — runs snakemake build, target <5min",
]
```

A `tests/conftest.py` adds an auto-marker that tags everything collected from `workflow/scripts/test/` as `fast` (so existing unit tests join Tier A without code changes).

## Tier A — static checks

### `test_dag_dryrun.py`

Parameterized over `(configfile, target_rule)` pairs:

```python
@pytest.mark.fast
@pytest.mark.parametrize(
    "configfile,target",
    [
        ("config/config.tutorial.yaml", "cluster_network"),
        ("config/config.tutorial.yaml", "solve_network"),
        ("config/config.default.yaml", "cluster_network"),
    ],
)
def test_snakemake_dryrun_resolves(configfile, target):
    result = subprocess.run(
        ["snakemake", "-n", "--configfile", configfile, "--until", target],
        cwd="workflow",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
```

Catches: missing inputs, wildcard mismatches, typo'd rule names, syntactically broken `.smk` files.

### `test_paths.py`

Uses the snakemake Python API to load the workflow without executing it, then walks all rules:

```python
@pytest.mark.fast
def test_no_hardcoded_resources_paths():
    """Every input/output that starts with 'resources/' must use a category constant."""
    # Walk .smk files, extract rule input/output assignments via AST
    # Assert no string literal of the form 'resources/{interconnect}/elec_...' appears
    # — must use NETWORKS, BUSMAPS, etc.
    ...


@pytest.mark.fast
def test_every_input_has_producer():
    """Every rule input is either external (data/, repo_data/) or another rule's output."""
    workflow = load_snakemake_workflow("workflow/Snakefile")
    all_outputs = {out for rule in workflow.rules for out in rule.output}
    for rule in workflow.rules:
        for inp in rule.input:
            if inp.startswith(("data/", "repo_data/", "cutouts/")):
                continue
            assert inp in all_outputs, f"Rule {rule.name} input {inp} has no producer"
```

Catches: the exact class of bug from the v1-epic merge conflict (rules pointing at the old `RESOURCES + "{interconnect}/..."` layout after the category-constant migration). Also catches dangling inputs from in-progress refactors.

### `test_config_keys.py`

AST-walks every `.py` file under `workflow/scripts/`, finds all subscript chains rooted at `snakemake.config` or a bare `config` (when assigned from `snakemake.config`), and asserts each accessed key path exists in the merged YAML:

```python
@pytest.mark.fast
def test_all_referenced_config_keys_exist():
    merged_config = load_merged_yaml(
        [
            "config/config.cluster.yaml",
            "config/config.common.yaml",
            "config/config.plotting.yaml",
            "config/config.api.yaml",
            "config/config.sector.yaml",
            "config/config.default.yaml",
        ]
    )
    for py_file in Path("workflow/scripts").rglob("*.py"):
        accesses = extract_config_accesses_via_ast(py_file)
        for key_path in accesses:
            assert key_exists(
                merged_config, key_path
            ), f"{py_file}: config[{'.'.join(key_path)}] is not defined in any YAML"
```

Catches: the exact bug class addressed by PR #10 (`electricity.prm` was accessed in code but never defined in YAML, and vice versa). Initially runs as warn-only on existing violations (file an issue); after a one-time cleanup, becomes a hard failure.

## Tier B — integration build

### `config/config.test.yaml`

A new config explicitly designed for tests (not a user-facing entry point). The smallest config that still routes through every rule on the simplify-early path:

```yaml
run:
  name: "Test"
  shared_resources: false
  shared_cutouts: true

scenario:
  interconnect: [western]
  clusters: [4m]
  simpl: [20]
  opts: [REM-3h]
  ll: [v1.0]
  sector: ""
  planning_horizons: [2030]

foresight: perfect

model_topology:
  transmission_network: 'reeds'
  topological_boundaries: 'reeds_zone'
  include:
    reeds_state: ['CA']
  # smaller than tutorial which also includes adjacent states

snapshots:
  start: "2019-01-01 00:00"
  end: "2019-01-01 23:00"
  inclusive: both

renewable_weather_years: [2019]

# inherit everything else from config.default.yaml via mergeable layered config
```

### Session-scoped fixture

```python
# tests/integration/conftest.py
import os
import subprocess
import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    """Run snakemake --until cluster_network once per session. Yield artifact paths."""
    workflow_dir = Path(__file__).parents[2] / "workflow"
    run_name = f"pytest_{tmp_path_factory.mktemp('run').name}"
    subprocess.run(
        [
            "snakemake",
            "--until",
            "cluster_network",
            "--configfile",
            "config/config.test.yaml",
            "--config",
            f"run={{name: '{run_name}'}}",
            "-j",
            str(os.cpu_count() or 2),
        ],
        cwd=workflow_dir,
        check=True,
    )
    base = workflow_dir / "resources" / run_name
    return SimpleNamespace(
        run_name=run_name,
        elec_b=base / "networks" / "western" / "elec_b.nc",
        elec_s=base / "networks" / "western" / "elec_s20.nc",
        elec_s_dem=base / "networks" / "western" / "elec_s20_dem.nc",
        elec_s_l_pp=base / "networks" / "western" / "elec_s20_l_pp.pkl",
        elec_s_c=base / "networks" / "western" / "elec_s20_c4m.nc",
    )
```

### `test_artifacts.py`

Asserts per-artifact properties without comparing to any baseline:

```python
@pytest.mark.integration
class TestPostAggregate:
    def test_bus_count(self, built):
        n = pypsa.Network(built.elec_b)
        # CA-only Western should yield roughly N_substations buses
        assert 40 < len(n.buses) < 120, f"unexpected substation count {len(n.buses)}"

    def test_no_nan_in_coordinates(self, built):
        n = pypsa.Network(built.elec_b)
        assert not n.buses[["x", "y"]].isna().any().any()


@pytest.mark.integration
class TestPostClusterSimpl:
    def test_bus_count_matches_simpl(self, built):
        n = pypsa.Network(built.elec_s)
        assert len(n.buses) == 20

    def test_busmap_exported(self, built):
        # Phase 3 dependency: busmap_s{simpl}.csv must exist for aggregate_egs
        path = (
            built.elec_s.parent.parent.parent / "busmaps" / "western" / "busmap_s20.csv"
        )
        assert path.exists()


@pytest.mark.integration
class TestPostAddElectricity:
    def test_pickle_loads(self, built):
        with open(built.elec_s_l_pp, "rb") as f:
            n = dill.load(f)
        assert len(n.generators) > 0
        assert len(n.loads) > 0
        assert not n.generators[["p_nom", "bus"]].isna().any().any()

    def test_load_timeseries_no_nan(self, built):
        with open(built.elec_s_l_pp, "rb") as f:
            n = dill.load(f)
        assert not n.loads_t.p_set.isna().any().any()


@pytest.mark.integration
class TestPostClusterNetwork:
    def test_cluster_count(self, built):
        n = pypsa.Network(built.elec_s_c)
        # 4m = at least 4 clusters
        assert len(n.buses) >= 4

    def test_roundtrip(self, built, tmp_path):
        n = pypsa.Network(built.elec_s_c)
        out = tmp_path / "roundtrip.nc"
        n.export_to_netcdf(out)
        n2 = pypsa.Network(out)
        pd.testing.assert_frame_equal(n.buses, n2.buses)
        pd.testing.assert_frame_equal(n.generators, n2.generators)
```

## CI wiring

Modify `.github/workflows/main.yml`:

1. **Replace** the single `./test.sh` step with two jobs, both required for merge.

2. **New `fast-tests` job** — no data download, no micromamba environment:
   ```yaml
   fast-tests:
     runs-on: ubuntu-latest
     steps:
       - uses: actions/checkout@v3
       - uses: actions/setup-python@v5
         with: { python-version: '3.11' }
       - run: pip install -e '.[test]'  # minimal deps: pytest, snakemake, pyyaml
       - run: pytest -m fast --tb=short
   ```
   Targets ~1 min wall-clock.

3. **New `e2e-tests` job** — full micromamba env, reuses the existing data/cutouts cache from the current `build` job:
   ```yaml
   e2e-tests:
     runs-on: ubuntu-latest
     steps:
       - uses: actions/checkout@v3
       - uses: mamba-org/setup-micromamba@v1
         with: { environment-file: workflow/envs/environment.yaml, cache-environment: true }
       - uses: actions/cache@v3
         with: { path: [data, cutouts], key: data-cutouts-${{ env.WEEK }}-${{ env.DATA_CACHE_NUMBER }} }
       - run: pytest -m integration --tb=short
   ```
   Targets ~10 min wall-clock including env + build.

4. **Move** the existing matrix dim `inhouse: master` to a separate weekly cron job — not gating PRs.

5. **Branch protection** on `master` and `v1-epic`: require both `fast-tests` and `e2e-tests` to pass before merge.

## Migration plan

Five PRs, smallest first. Each is independently mergeable and leaves the test suite in a green state:

| PR | Title | Scope | Dependencies |
|----|-------|-------|--------------|
| 1  | Test scaffolding | Add `tests/` skeleton, `pytest.ini`, `pyproject.toml` testpaths, auto-marker for existing unit tests. No new test logic — just plumbing. `pytest -m fast` collects and passes the existing unit tests. | None |
| 2  | Tier A static checks | Add `test_dag_dryrun.py`, `test_paths.py`, `test_config_keys.py`. Includes the fixes for any violations they surface against current master (e.g., `data_model` rule still uses old `RESOURCES + "{interconnect}/elec_..."` paths) — PR leaves `pytest -m fast` exiting 0. | PR 1 |
| 3  | Tier B fixture + smoke test | Add `config/config.test.yaml`, session fixture, single bus-count assertion. Proves the pytest→snakemake plumbing works in CI. | PR 1 |
| 4  | Tier B full assertions | Flesh out `test_artifacts.py` with all per-stage assertions. | PR 3 |
| 5  | CI wiring | Replace `./test.sh` with the two new jobs in `main.yml`. Add branch protection. Move `inhouse: master` matrix to a separate weekly cron. | PRs 2 + 4 |

After the schema-tracking initiative (`docs/superpowers/specs/2026-05-21-network-schema-tracking-design.md`) lands, a sixth PR adds `tests/integration/test_schema.py` that asserts every component column on every artifact appears in the catalog.

## Out of scope (future tiers)

These were considered and explicitly deferred:

- **Numerical regression (Tier C).** Commits a baseline solved network for the test config; future runs diff `n.objective`, per-carrier `p_nom`, per-snapshot dispatch, etc. against baseline. Requires LFS, baseline-refresh workflow when outputs change intentionally, and bumps the CI budget by another ~5-10 min for the solve. Better as an explicit `pytest --baseline` command run after algorithmic changes, not a per-PR gate.
- **Memory regression.** The whole point of the simplify-early refactor was peak-RSS reduction. A useful test would assert peak RSS during `build_renewable_profiles`/`add_electricity` stays below a budget. Needs `psutil` instrumentation in scripts and a stable CI environment for the measurement. Defer to a separate effort.
- **Multi-config matrix.** Currently Tier B exercises a single config. Useful future expansion: parameterize Tier B over `(interconnect, sector, foresight)` triples to catch breakage in code paths the test config doesn't visit. Defer until Tier B is stable.

## Pitfalls and mitigations

1. **Tutorial config too slow.** The current `config.tutorial.yaml` uses `simpl=75, clusters=4m, 2050` and is plausibly >5 min on CI. Mitigation: new `config.test.yaml` with `simpl=20`, 1-day snapshots. Validate the wall-clock during PR 3 — if it overruns, drop snapshots or simpl further.
2. **Subprocess snakemake CWD pitfalls.** Snakemake must be invoked with `cwd=workflow/`; the fixture must be careful with relative paths in `--configfile`. Mitigation: explicit `cwd=` in the subprocess call, use repo-root-relative paths to compute the workflow dir in the fixture.
3. **Existing unit tests use `sys.path.append("..")`.** When collected from a different root, the import may fail. Mitigation: don't collect them from `tests/`; use `testpaths = [..., "workflow/scripts/test"]` so they're collected in their own dir with their existing `conftest.py` providing the path tweak.
4. **`test_paths.py` false positives.** Some inputs are dynamically computed (`lambda w: ...`). Mitigation: only check the leaf string literals that appear in lambda bodies; skip rules whose inputs use unpacking (`unpack(...)`) for now and document as a known gap.
5. **`test_config_keys.py` over-strictness.** Some config keys are accessed only when an optional feature is enabled (`config["enable"]["custom_busmap"]`). They exist in YAML but as `false`. Mitigation: check key existence, not truthiness.
6. **Branch protection lock-out.** Requiring two new jobs before they exist will block all merges. Mitigation: land PRs 1-4 with the jobs configured but not required; flip the branch-protection switch in PR 5 only after both jobs are observed green for several PRs.

## Verification

When the migration is complete:

- `pytest -m fast` exits 0 in <30s on a clean checkout with `pip install -e '.[test]'`.
- `pytest -m integration` exits 0 in <10min on a CI runner with cached data/cutouts.
- `.github/workflows/main.yml` has `fast-tests` and `e2e-tests` jobs; both are required in branch protection on `master` and `v1-epic`.
- A deliberately-broken PR (e.g., one that renames `cluster_simpl` output but doesn't update consumers) is rejected by `test_dag_dryrun.py` or `test_paths.py` before merge.
- A deliberately-introduced regression (e.g., a rule that produces a `.nc` with NaN in `Bus.x`) is rejected by `test_artifacts.py` before merge.
