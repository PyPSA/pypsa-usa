# Testing Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land a tiered pre-merge test pyramid for PyPSA-USA — Tier A static checks (<30s, no data) and Tier B integration build (<5min, cached data) — wired into CI as two parallel required jobs.

**Architecture:** New `tests/` directory at repo root holds the new tests; existing `workflow/scripts/test/*` stays in place and is collected via `pyproject.toml` testpaths. Pytest markers (`fast`, `integration`) gate which tier runs. Tier A invokes `snakemake -n` as a subprocess plus pure-Python AST-based linters. Tier B uses a session-scoped fixture that runs `snakemake --until cluster_network` against a new minimal `config.test.yaml`; subsequent tests load the produced artifacts and assert shape.

**Tech Stack:** pytest 8.x, pyyaml, snakemake 7.32.4 (already pinned), pypsa 0.30.2 (already pinned), dill (already pinned). GitHub Actions for CI.

**Spec:** `docs/superpowers/specs/2026-05-21-testing-strategy-design.md` (commit `a7f014ab` on `v1-epic`).

---

## File Structure

PR 1 (scaffolding):
- Modify: `pyproject.toml` — add `[tool.pytest.ini_options]`, `[project.optional-dependencies] test`
- Create: `tests/__init__.py` (empty)
- Create: `tests/static/__init__.py` (empty)
- Create: `tests/integration/__init__.py` (empty)
- Create: `conftest.py` (repo root — auto-marker for existing unit tests)
- Create: `tests/conftest.py` (repo path helpers)

PR 2 (Tier A):
- Create: `tests/static/test_dag_dryrun.py`
- Create: `tests/static/test_paths.py`
- Create: `tests/static/test_config_keys.py`
- Modify: `workflow/Snakefile` — fix `data_model` rule path to use `NETWORKS` constant (violation surfaced by `test_paths.py`)

PR 3 (Tier B fixture):
- Create: `workflow/config/config.test.yaml`
- Create: `tests/integration/conftest.py`
- Create: `tests/integration/test_artifacts.py` (single placeholder bus-count assertion)

PR 4 (Tier B full):
- Modify: `tests/integration/test_artifacts.py` — flesh out all per-stage assertions

PR 5 (CI wiring):
- Modify: `.github/workflows/main.yml` — replace `./test.sh` step with `fast-tests` and `e2e-tests` jobs

PR 6 (schema assertions, after schema-tracking lands):
- Create: `tests/integration/test_schema.py`

---

# PR 1: Scaffolding

Branch from `v1-epic`. Goal: `pytest -m fast` collects and passes the existing unit tests; nothing else changes.

### Task 1.1: Add pytest configuration to pyproject.toml

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Read the current pyproject.toml structure**

Run: `grep -n "^\[" pyproject.toml`
Note line numbers of `[project.optional-dependencies]` and `[tool.setuptools.package-dir]`.

- [ ] **Step 2: Add `test` optional-dependencies group**

Locate the `[project.optional-dependencies]` section. After the existing `dev = [...]` block, append:

```toml
test = [
    "pytest>=8.0",
    "pyyaml>=6.0",
    "snakemake==7.32.4",
]
```

- [ ] **Step 3: Add pytest configuration block**

After the `[tool.setuptools.package-dir]` block (or at the end of the file if it doesn't exist), append:

```toml
[tool.pytest.ini_options]
testpaths = ["tests", "workflow/scripts/test"]
markers = [
    "fast: tier A — static checks, no data deps, target <30s",
    "integration: tier B — runs snakemake build, target <5min (needs data/, cutouts/, repo_data/)",
]
addopts = "--strict-markers -ra"
```

- [ ] **Step 4: Verify pyproject.toml still parses**

Run: `python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"`
Expected: no output (exit 0).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "Add pytest config and test extras"
```

### Task 1.2: Create the tests/ directory skeleton

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/static/__init__.py`
- Create: `tests/integration/__init__.py`

- [ ] **Step 1: Create empty package files**

Run:
```bash
mkdir -p tests/static tests/integration
touch tests/__init__.py tests/static/__init__.py tests/integration/__init__.py
```

- [ ] **Step 2: Verify directory tree**

Run: `find tests -type f`
Expected:
```
tests/__init__.py
tests/static/__init__.py
tests/integration/__init__.py
```

- [ ] **Step 3: Commit**

```bash
git add tests/
git commit -m "Add tests/ directory skeleton"
```

### Task 1.3: Add a repo-root conftest.py with auto-marker for existing unit tests

**Files:**
- Create: `conftest.py` (repo root)

Why repo-root and not `tests/conftest.py`: pytest only applies a `conftest.py` to items collected under its directory. The auto-marker must reach items under `workflow/scripts/test/`, so the conftest must sit at the rootdir.

- [ ] **Step 1: Write the conftest with the auto-marker hook**

Create `conftest.py`:

```python
"""Repo-root pytest configuration.

Auto-marks every test collected from ``workflow/scripts/test/`` with the
``fast`` marker so the existing unit tests join Tier A without code changes.
Tests under ``tests/`` declare their marker explicitly.
"""

from pathlib import Path

import pytest

_UNIT_TEST_DIR = (Path(__file__).parent / "workflow" / "scripts" / "test").resolve()


def pytest_collection_modifyitems(config, items):
    for item in items:
        try:
            item_path = Path(item.fspath).resolve()
        except (TypeError, ValueError):
            continue
        if _UNIT_TEST_DIR in item_path.parents:
            item.add_marker(pytest.mark.fast)
```

- [ ] **Step 2: Install the test extras**

Run: `pip install -e '.[test]'`
Expected: install succeeds (or "Requirement already satisfied" for everything).

- [ ] **Step 3: Verify pytest collects and tags the existing unit tests**

Run: `pytest -m fast --collect-only -q workflow/scripts/test/`
Expected: collected N items (N > 0, matches `pytest --collect-only -q workflow/scripts/test/` count).

- [ ] **Step 4: Verify pytest -m fast passes**

Run: `pytest -m fast -q`
Expected: all collected items pass. (If any fail today on `v1-epic`, document and skip with `@pytest.mark.skip(reason="pre-existing failure on v1-epic, not introduced by this PR")` — do NOT silently fix unrelated bugs in this PR.)

- [ ] **Step 5: Verify pytest -m integration collects zero items**

Run: `pytest -m integration --collect-only -q`
Expected: `no tests ran` or `0 tests collected`.

- [ ] **Step 6: Commit**

```bash
git add conftest.py
git commit -m "Wire existing unit tests into the 'fast' marker via repo-root conftest"
```

### Task 1.4: Open PR 1

- [ ] **Step 1: Push branch**

```bash
git push -u origin <branch-name>
```

- [ ] **Step 2: Open PR against v1-epic**

```bash
gh pr create --base v1-epic --title "Test scaffolding: pytest config + fast-marker for existing unit tests" --body "$(cat <<'EOF'
## Summary
- Adds `[tool.pytest.ini_options]` with `testpaths = ["tests", "workflow/scripts/test"]` and registers `fast` / `integration` markers.
- Adds `test` optional-dependencies group (`pytest`, `pyyaml`, `snakemake`).
- Adds repo-root `conftest.py` that auto-marks every test under `workflow/scripts/test/` as `fast`.
- Adds empty `tests/` skeleton ready for Tier A and Tier B PRs.

## Test plan
- [x] `pip install -e '.[test]'` succeeds
- [x] `pytest -m fast --collect-only` collects the existing unit tests
- [x] `pytest -m fast` exits 0
- [x] `pytest -m integration --collect-only` collects zero items

## Spec
Part of `docs/superpowers/specs/2026-05-21-testing-strategy-design.md` — PR 1 of 5.
EOF
)"
```

---

# PR 2: Tier A static checks

Branch from `v1-epic` after PR 1 merges. Goal: three new static-check tests, plus the one snakemake fix they surface.

### Task 2.1: Write the failing test for snakemake -n dry-run

**Files:**
- Create: `tests/static/test_dag_dryrun.py`

- [ ] **Step 1: Write the test**

Create `tests/static/test_dag_dryrun.py`:

```python
"""Tier A — verify the snakemake DAG resolves on representative configs.

Catches: missing inputs, typo'd rule names, wildcard mismatches, and
syntactically broken ``.smk`` files. Runs ``snakemake -n`` (dry-run only,
no rule actually executes).
"""

import subprocess
from pathlib import Path

import pytest

WORKFLOW_DIR = Path(__file__).resolve().parents[2] / "workflow"


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
        [
            "snakemake",
            "-n",
            "--configfile",
            configfile,
            "--until",
            target,
            "--quiet",
        ],
        cwd=WORKFLOW_DIR,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"snakemake -n failed for {configfile} --until {target}\n"
        f"stderr:\n{result.stderr}\n"
        f"stdout (last 50 lines):\n" + "\n".join(result.stdout.splitlines()[-50:])
    )
```

- [ ] **Step 2: Run and observe**

Run: `pytest tests/static/test_dag_dryrun.py -v`
Expected: either all parametrized cases pass, or you find pre-existing DAG issues. If a case fails, read the stderr — common causes are missing data files (download via the retrieve rules first) or a broken rule input path (which is what this test exists to catch).

- [ ] **Step 3: Commit**

```bash
git add tests/static/test_dag_dryrun.py
git commit -m "Add Tier A: snakemake -n dry-run resolves on tutorial+default configs"
```

### Task 2.2: Write the path-validator test

**Files:**
- Create: `tests/static/test_paths.py`

- [ ] **Step 1: Write the test**

Create `tests/static/test_paths.py`:

```python
"""Tier A — assert every rule path uses a category constant and resolves.

After PR #12 reorganized ``resources/`` into category-first subfolders
(``NETWORKS``, ``BUSMAPS``, ``PROFILES``, ``GEOSPATIAL``, ``COSTS``,
``PRICES``, ``DEMAND``, ``POWERPLANTS``, ``HEATING_COP``, ``TEMPERATURE``,
``POPULATION``, ``CO2``), rule inputs/outputs must use these constants
rather than hard-coded ``RESOURCES + "{interconnect}/..."`` strings.
This test parses every ``.smk`` file via AST and flags violations.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

WORKFLOW_DIR = Path(__file__).resolve().parents[2] / "workflow"
SMK_FILES = list(WORKFLOW_DIR.glob("Snakefile")) + list(
    WORKFLOW_DIR.glob("rules/*.smk")
)

CATEGORY_CONSTANTS = {
    "NETWORKS",
    "BUSMAPS",
    "PROFILES",
    "GEOSPATIAL",
    "COSTS",
    "PRICES",
    "POWERPLANTS",
    "DEMAND",
    "HEATING_COP",
    "TEMPERATURE",
    "POPULATION",
    "CO2",
}

# Pattern that means "concatenating RESOURCES with an interconnect-keyed
# path" — this is what every category constant replaces. A bare RESOURCES +
# "<other>/..." (no {interconnect} wildcard) is still allowed, e.g.
# RESOURCES + "co2_totals.csv".
HARDCODED_PATTERN = re.compile(r"\{interconnect\}/(?!Geospatial/)")


def _walk_concat(node):
    """Yield every string literal that appears in a chain of Add()s."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        yield node.value
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        yield from _walk_concat(node.left)
        yield from _walk_concat(node.right)


def _left_name(node):
    """Return the leftmost Name in a chain of Add()s, if any."""
    while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        node = node.left
    if isinstance(node, ast.Name):
        return node.id
    return None


@pytest.mark.fast
@pytest.mark.parametrize("smk_path", SMK_FILES, ids=lambda p: p.name)
def test_no_hardcoded_resources_interconnect_paths(smk_path):
    """RESOURCES + "{interconnect}/..." is forbidden — use a category constant."""
    source = smk_path.read_text()
    tree = ast.parse(source)
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Add):
            continue
        leftmost = _left_name(node)
        if leftmost != "RESOURCES":
            continue
        for literal in _walk_concat(node):
            if HARDCODED_PATTERN.search(literal):
                violations.append((node.lineno, literal))
    assert not violations, (
        f"{smk_path.name}: found RESOURCES + '{{interconnect}}/...' literals — "
        f"use a category constant ({', '.join(sorted(CATEGORY_CONSTANTS))}):\n"
        + "\n".join(f"  line {ln}: {lit!r}" for ln, lit in violations)
    )
```

- [ ] **Step 2: Run and observe expected failure on `data_model` rule**

Run: `pytest tests/static/test_paths.py -v`
Expected: `Snakefile` fails — it has `RESOURCES + "{interconnect}/elec_s{simpl}_c{clusters}_ec_l{ll}_{opts}_{sector}.nc"` in the `data_model` rule. The error message will print the line number.

- [ ] **Step 3: Fix the violation in workflow/Snakefile**

Read `workflow/Snakefile` around the `data_model` rule (it's near line 257). The output pattern `elec_s..._ec_l..._{sector}.nc` is the final post-`prepare_network` output. First, find which rule actually produces this file:

Run: `grep -n "elec_s{simpl}_c{clusters}_ec_l{ll}_{opts}_{sector}.nc" workflow/rules/*.smk`

The producer rule's `output:` should use a category constant (likely `NETWORKS`). Use the same constant in `data_model`:

```snakemake
rule data_model:
    input:
        expand(
            NETWORKS
            + "{interconnect}/elec_s{simpl}_c{clusters}_ec_l{ll}_{opts}_{sector}.nc",
            **config["scenario"],
        ),
```

If the producer rule itself uses the wrong constant, fix it too — note all locations in the commit message.

- [ ] **Step 4: Verify the test now passes**

Run: `pytest tests/static/test_paths.py -v`
Expected: all parametrized cases pass.

- [ ] **Step 5: Commit**

```bash
git add tests/static/test_paths.py workflow/Snakefile
git commit -m "Add Tier A: path validator + fix data_model rule to use NETWORKS"
```

### Task 2.3: Write the config-keys test

**Files:**
- Create: `tests/static/test_config_keys.py`

- [ ] **Step 1: Write the test**

Create `tests/static/test_config_keys.py`:

```python
"""Tier A — assert every snakemake.config[...] access in scripts is defined.

AST-walks every script under workflow/scripts/, extracts subscript chains
rooted at ``snakemake.config`` or a bare ``config`` (when locally rebound
from snakemake.config), and asserts each key path exists in the merged
default config. Catches dead/typo'd config keys (the class of bug fixed
in PR #10).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = REPO_ROOT / "workflow"
SCRIPT_DIR = WORKFLOW_DIR / "scripts"
CONFIG_DIR = WORKFLOW_DIR / "config"

DEFAULT_CONFIGFILES = [
    "config.cluster.yaml",
    "config.common.yaml",
    "config.plotting.yaml",
    "config.api.yaml",
    "config.sector.yaml",
    "config.default.yaml",
]


def _merge(a: dict, b: dict) -> dict:
    """Deep-merge b into a; b wins on conflict."""
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_merged_config():
    merged = {}
    for name in DEFAULT_CONFIGFILES:
        path = CONFIG_DIR / name
        with open(path) as f:
            merged = _merge(merged, yaml.safe_load(f) or {})
    return merged


def _key_exists(cfg, keys):
    """Walk a list of keys into nested dicts. True if every key is defined."""
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return False
        node = node[k]
    return True


def _subscript_chain(node):
    """Return list of string keys for chained Subscript nodes, or None."""
    keys = []
    while isinstance(node, ast.Subscript):
        idx = node.slice
        if isinstance(idx, ast.Constant) and isinstance(idx.value, str):
            keys.append(idx.value)
        else:
            return None  # dynamic key — give up
        node = node.value
    keys.reverse()
    return node, keys


def _is_snakemake_config(node) -> bool:
    """True if node is `snakemake.config`."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "config"
        and isinstance(node.value, ast.Name)
        and node.value.id == "snakemake"
    )


def _extract_config_accesses(source: str) -> list[list[str]]:
    """Return key paths accessed via snakemake.config[...][...]."""
    tree = ast.parse(source)
    accesses = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        root, keys = _subscript_chain(node)
        if keys is None:
            continue
        if _is_snakemake_config(root):
            accesses.append(keys)
    return accesses


@pytest.fixture(scope="module")
def merged_config():
    return _load_merged_config()


@pytest.mark.fast
@pytest.mark.parametrize(
    "script",
    sorted(SCRIPT_DIR.glob("*.py")),
    ids=lambda p: p.name,
)
def test_referenced_config_keys_exist(script, merged_config):
    source = script.read_text()
    try:
        accesses = _extract_config_accesses(source)
    except SyntaxError as e:
        pytest.skip(f"{script.name} has syntax that ast cannot parse: {e}")
    missing = [keys for keys in accesses if not _key_exists(merged_config, keys)]
    assert not missing, (
        f"{script.name}: snakemake.config keys not defined in any "
        f"workflow/config/*.yaml:\n"
        + "\n".join(f"  config[{']['.join(repr(k) for k in keys)}]" for keys in missing)
    )
```

- [ ] **Step 2: Run and observe**

Run: `pytest tests/static/test_config_keys.py -v`
Expected: most scripts pass. If any fail, decide per-failure: (a) the key is genuinely missing and the script is buggy — fix the YAML; (b) the key is accessed only when a feature flag is on and is intentionally absent from defaults — add it to the defaults with a comment, OR adjust the test to allow it. Keep this PR focused: fix only blatant typos here, file follow-ups for ambiguous cases.

- [ ] **Step 3: Commit**

```bash
git add tests/static/test_config_keys.py
# add any workflow/config/*.yaml fixes
git commit -m "Add Tier A: config-key existence validator"
```

### Task 2.4: Verify the whole Tier A passes under 30 seconds

- [ ] **Step 1: Time the full Tier A run**

Run: `time pytest -m fast -q`
Expected: all tests pass, wall-clock under 30 seconds. If over, the dry-run cases are the likely culprit — drop the slowest parametrize case or use `--quiet` more aggressively. Note the actual time in the PR description.

- [ ] **Step 2: Open PR 2**

```bash
git push -u origin <branch-name>
gh pr create --base v1-epic --title "Tier A: static checks (DAG dry-run, path validator, config-keys)" --body "..."
```

---

# PR 3: Tier B fixture + smoke test

Branch from `v1-epic` after PRs 1 and 2 merge. Goal: prove the snakemake-from-pytest plumbing works end-to-end in CI; one trivial integration assertion.

### Task 3.1: Create the minimal test config

**Files:**
- Create: `workflow/config/config.test.yaml`

Sizing target: must build to `cluster_network` in under 5 minutes on a CI runner with cached `data/` and `cutouts/`. Tutorial config (`simpl=75`, full year) is too slow; this config uses `simpl=20` and a 1-day snapshot window.

- [ ] **Step 1: Write the config**

Create `workflow/config/config.test.yaml`:

```yaml
# PyPSA-USA test config — used by tests/integration to build to
# cluster_network on a minimal CA-only Western slice in <5 minutes.
# DO NOT use as a user-facing entry point.
run:
  name: "test"
  disable_progressbar: true
  shared_resources: false
  shared_cutouts: true
  validation: false

foresight: 'perfect'

scenario:
  interconnect: [western]
  clusters: [4m]
  simpl: [20]
  opts: [REM-3h]
  ll: [v1.0]
  scope: "total"
  sector: ""
  planning_horizons: [2030]

model_topology:
  transmission_network: 'reeds'
  topological_boundaries: 'reeds_zone'
  interface_transmission_limits: false
  include:
    reeds_state: ['CA']
  aggregate: {}

enable:
  build_cutout: false

renewable_weather_years: [2019]

snapshots:
  start: "2019-01-01 00:00"
  end: "2019-01-01 23:00"
  inclusive: both
```

- [ ] **Step 2: Verify snakemake can parse and dry-run with it**

Run: `cd workflow && snakemake -n --configfile config/config.test.yaml --until cluster_network --quiet; cd ..`
Expected: exit 0 — DAG resolves.

- [ ] **Step 3: Commit**

```bash
git add workflow/config/config.test.yaml
git commit -m "Add minimal CA-only test config for tests/integration"
```

### Task 3.2: Write the session-scoped fixture

**Files:**
- Create: `tests/integration/conftest.py`

- [ ] **Step 1: Write the fixture**

Create `tests/integration/conftest.py`:

```python
"""Tier B — session-scoped fixture that runs `snakemake --until cluster_network`
once per test session and exposes paths to the produced artifacts.

Skipped automatically if required data dirs are missing (lets developers
run `pytest -m fast` locally without setting up the data deps).
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = REPO_ROOT / "workflow"
DATA_DIRS = [
    WORKFLOW_DIR / "data",
    WORKFLOW_DIR / "cutouts",
    WORKFLOW_DIR / "repo_data",
]


@dataclass(frozen=True)
class BuiltArtifacts:
    run_name: str
    base: Path  # resources/{run_name}/
    interconnect: str = "western"
    simpl: str = "20"
    clusters: str = "4m"

    @property
    def elec_b(self) -> Path:
        return self.base / "networks" / self.interconnect / "elec_b.nc"

    @property
    def elec_s(self) -> Path:
        return self.base / "networks" / self.interconnect / f"elec_s{self.simpl}.nc"

    @property
    def elec_s_dem(self) -> Path:
        return self.base / "networks" / self.interconnect / f"elec_s{self.simpl}_dem.nc"

    @property
    def elec_s_l_pp(self) -> Path:
        return (
            self.base / "networks" / self.interconnect / f"elec_s{self.simpl}_l_pp.pkl"
        )

    @property
    def elec_s_c(self) -> Path:
        return (
            self.base
            / "networks"
            / self.interconnect
            / f"elec_s{self.simpl}_c{self.clusters}.nc"
        )

    @property
    def busmap_s(self) -> Path:
        return self.base / "busmaps" / self.interconnect / f"busmap_s{self.simpl}.csv"


@pytest.fixture(scope="session")
def built(tmp_path_factory) -> BuiltArtifacts:
    missing = [d for d in DATA_DIRS if not d.exists()]
    if missing:
        pytest.skip(
            "Integration tests require populated data dirs; missing: "
            + ", ".join(str(m) for m in missing)
        )
    run_name = f"pytest_{tmp_path_factory.mktemp('run').name}"
    cmd = [
        "snakemake",
        "--until",
        "cluster_network",
        "--configfile",
        "config/config.test.yaml",
        "--config",
        f"run={{name: '{run_name}', shared_cutouts: true}}",
        "-j",
        str(os.cpu_count() or 2),
        "--quiet",
    ]
    result = subprocess.run(
        cmd, cwd=WORKFLOW_DIR, capture_output=True, text=True, timeout=600
    )
    if result.returncode != 0:
        pytest.fail(
            f"snakemake build failed (exit {result.returncode}):\n"
            f"stderr (last 100 lines):\n" + "\n".join(result.stderr.splitlines()[-100:])
        )
    return BuiltArtifacts(run_name=run_name, base=WORKFLOW_DIR / "resources" / run_name)
```

- [ ] **Step 2: Write the smoke-test assertion**

Create `tests/integration/test_artifacts.py`:

```python
"""Tier B — load each produced artifact and assert shape.

Initial smoke version: one bus-count assertion to prove the fixture
plumbing works in CI. Later PRs flesh this out with per-stage assertions.
"""

import pypsa
import pytest


@pytest.mark.integration
def test_post_cluster_simpl_bus_count(built):
    n = pypsa.Network(str(built.elec_s))
    assert (
        len(n.buses) == 20
    ), f"cluster_simpl produced {len(n.buses)} buses, expected 20 (config.test.yaml simpl=20)"
```

- [ ] **Step 3: Verify locally (skipped if no data)**

Run: `pytest -m integration -v`
Expected if data is present: snakemake build runs (~3-5 min), then test passes.
Expected if data is missing: test SKIPPED with "Integration tests require populated data dirs" message.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/conftest.py tests/integration/test_artifacts.py
git commit -m "Add Tier B fixture + smoke test (cluster_simpl bus count)"
```

### Task 3.3: Open PR 3

- [ ] **Step 1: Push and open PR**

```bash
git push -u origin <branch-name>
gh pr create --base v1-epic --title "Tier B fixture: snakemake build + smoke test" --body "..."
```

In the PR body, record the local wall-clock of the build (`time pytest -m integration`) so PR 5 has a calibration point.

---

# PR 4: Tier B full assertions

Branch from `v1-epic` after PR 3 merges. Goal: replace the single smoke assertion with comprehensive per-stage shape assertions.

### Task 4.1: Expand test_artifacts.py with per-stage assertions

**Files:**
- Modify: `tests/integration/test_artifacts.py`

- [ ] **Step 1: Replace the file with the full assertion suite**

Overwrite `tests/integration/test_artifacts.py`:

```python
"""Tier B — load each produced artifact and assert shape.

Asserts per-stage: bus counts match wildcards, no NaN in load-bearing
columns, netCDF roundtrips, filename wildcard propagation. No numerical
regression here — that's Tier C (deferred).
"""

from __future__ import annotations

import dill
import pandas as pd
import pypsa
import pytest

pytestmark = pytest.mark.integration


# ----- stage 1: aggregate_to_substations -----------------------------------


class TestPostAggregateToSubstations:
    def test_file_exists(self, built):
        assert built.elec_b.exists(), f"missing {built.elec_b}"

    def test_bus_count_reasonable(self, built):
        n = pypsa.Network(str(built.elec_b))
        # CA-only Western yields O(50) substations after aggregation
        assert (
            10 < len(n.buses) < 200
        ), f"unexpected substation count {len(n.buses)} — config.test.yaml may have drifted"

    def test_no_nan_in_coordinates(self, built):
        n = pypsa.Network(str(built.elec_b))
        assert not n.buses[["x", "y"]].isna().any().any()


# ----- stage 2: cluster_simpl ----------------------------------------------


class TestPostClusterSimpl:
    def test_file_exists(self, built):
        assert built.elec_s.exists(), f"missing {built.elec_s}"

    def test_bus_count_equals_simpl(self, built):
        n = pypsa.Network(str(built.elec_s))
        assert len(n.buses) == int(
            built.simpl
        ), f"cluster_simpl produced {len(n.buses)} buses, expected {built.simpl}"

    def test_busmap_exported(self, built):
        # PR #11 dependency: cluster_simpl now exports busmap_s{simpl}.csv
        # so aggregate_egs can remap substation-keyed supply curves.
        assert built.busmap_s.exists(), f"missing {built.busmap_s}"

    def test_busmap_covers_all_substations(self, built):
        busmap = pd.read_csv(built.busmap_s, index_col=0)
        n_sub = pypsa.Network(str(built.elec_b))
        # Every substation bus must appear in the busmap index
        missing = set(n_sub.buses.index) - set(busmap.index.astype(str))
        assert not missing, f"busmap missing {len(missing)} substations"


# ----- stage 3: add_demand --------------------------------------------------


class TestPostAddDemand:
    def test_file_exists(self, built):
        assert built.elec_s_dem.exists(), f"missing {built.elec_s_dem}"

    def test_loads_attached(self, built):
        n = pypsa.Network(str(built.elec_s_dem))
        assert len(n.loads) > 0
        assert not n.loads_t.p_set.isna().any().any()


# ----- stage 4: add_electricity --------------------------------------------


class TestPostAddElectricity:
    def test_file_exists(self, built):
        assert built.elec_s_l_pp.exists(), f"missing {built.elec_s_l_pp}"

    def test_pickle_loads(self, built):
        with open(built.elec_s_l_pp, "rb") as f:
            n = dill.load(f)
        assert isinstance(n, pypsa.Network)
        assert len(n.generators) > 0

    def test_no_nan_in_load_bearing_columns(self, built):
        with open(built.elec_s_l_pp, "rb") as f:
            n = dill.load(f)
        assert not n.generators[["p_nom", "bus", "carrier"]].isna().any().any()
        assert not n.loads[["bus"]].isna().any().any()
        assert not n.buses[["x", "y", "carrier"]].isna().any().any()


# ----- stage 5: cluster_network --------------------------------------------


class TestPostClusterNetwork:
    def test_file_exists(self, built):
        assert built.elec_s_c.exists(), f"missing {built.elec_s_c}"

    def test_cluster_count_minimum(self, built):
        n = pypsa.Network(str(built.elec_s_c))
        # 4m = at least 4 clusters (the 'm' suffix means minimum)
        assert (
            len(n.buses) >= 4
        ), f"cluster_network produced {len(n.buses)} buses, expected >=4"

    def test_no_nan_in_buses(self, built):
        n = pypsa.Network(str(built.elec_s_c))
        assert not n.buses[["x", "y", "carrier"]].isna().any().any()

    def test_netcdf_roundtrip(self, built, tmp_path):
        n = pypsa.Network(str(built.elec_s_c))
        out = tmp_path / "roundtrip.nc"
        n.export_to_netcdf(str(out))
        n2 = pypsa.Network(str(out))
        # static frames must match exactly after write/read cycle
        pd.testing.assert_frame_equal(
            n.buses.sort_index(), n2.buses.sort_index(), check_dtype=False
        )
        pd.testing.assert_frame_equal(
            n.generators.sort_index(), n2.generators.sort_index(), check_dtype=False
        )

    def test_filename_has_expected_wildcards(self, built):
        # Catches accidental drift in the cluster_network output pattern
        name = built.elec_s_c.name
        assert f"s{built.simpl}" in name
        assert f"c{built.clusters}" in name
```

- [ ] **Step 2: Run the full suite**

Run: `pytest -m integration -v`
Expected if data is present: all tests pass; total wall-clock dominated by the snakemake build, individual test time <2s each.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_artifacts.py
git commit -m "Tier B: per-stage shape assertions for the five build artifacts"
```

### Task 4.2: Open PR 4

```bash
git push -u origin <branch-name>
gh pr create --base v1-epic --title "Tier B: per-stage artifact assertions" --body "..."
```

---

# PR 5: CI wiring

Branch from `v1-epic` after PR 4 merges. Goal: replace the broken `./test.sh` step with two parallel jobs; mark required in branch protection.

### Task 5.1: Replace test.sh step in main.yml

**Files:**
- Modify: `.github/workflows/main.yml`

- [ ] **Step 1: Read the current workflow**

Run: `cat .github/workflows/main.yml`

Locate the existing `build:` job, the `actions/cache@v3` step that caches `data` and `cutouts`, and the `Test snakemake workflow` step that calls `./test.sh`.

- [ ] **Step 2: Rewrite main.yml with two new jobs**

The new shape: keep the existing data-cache wiring (it's the slow part), but split into two top-level jobs that can run in parallel. Replace `.github/workflows/main.yml` with:

```yaml
# SPDX-FileCopyrightText: : 2021-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: CC0-1.0

name: CI

on:
  push:
    branches:
      - master
      - v1-epic
  pull_request:
    branches:
      - master
      - v1-epic
  schedule:
    - cron: "0 5 * * TUE"  # weekly upstream-master regression

env:
  DATA_CACHE_NUMBER: 2

jobs:
  fast-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install test extras
        run: pip install -e '.[test]'
      - name: Run Tier A (static checks)
        run: pytest -m fast --tb=short -ra

  e2e-tests:
    runs-on: ubuntu-latest
    defaults:
      run:
        shell: bash -l {0}
    steps:
      - uses: actions/checkout@v4
      - name: Setup secrets
        run: echo -ne "url: ${CDSAPI_URL}\nkey: ${CDSAPI_TOKEN}\n" > ~/.cdsapirc
      - name: Setup micromamba
        uses: mamba-org/setup-micromamba@v1
        with:
          micromamba-version: latest
          environment-file: workflow/envs/environment.yaml
          log-level: debug
          init-shell: bash
          cache-environment: true
          cache-downloads: true
      - name: Set cache dates
        run: echo "WEEK=$(date +'%Y%U')" >> $GITHUB_ENV
      - name: Cache data and cutouts folders
        uses: actions/cache@v4
        with:
          path: |
            data
            cutouts
          key: data-cutouts-${{ env.WEEK }}-${{ env.DATA_CACHE_NUMBER }}
      - name: Install test extras into the conda env
        run: pip install -e '.[test]'
      - name: Run Tier B (integration build)
        run: pytest -m integration --tb=short -ra
      - name: Upload artifacts on failure
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: failed-resources
          path: |
            workflow/resources
            workflow/logs
          if-no-files-found: warn
          retention-days: 3

  upstream-regression:
    if: github.event_name == 'schedule'
    runs-on: ubuntu-latest
    defaults:
      run:
        shell: bash -l {0}
    steps:
      - uses: actions/checkout@v4
      - uses: mamba-org/setup-micromamba@v1
        with:
          environment-file: workflow/envs/environment.yaml
          cache-environment: true
      - name: Install upstream PyPSA / atlite / linopy from master
        run: |
          pip install \
            git+https://github.com/PyPSA/atlite.git@master \
            git+https://github.com/PyPSA/powerplantmatching.git@master \
            git+https://github.com/PyPSA/linopy.git@master
      - name: Install test extras
        run: pip install -e '.[test]'
      - name: Run all tests
        run: pytest -ra
```

- [ ] **Step 3: Lint the YAML**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/main.yml'))"`
Expected: no output (parses successfully).

- [ ] **Step 4: Commit and push**

```bash
git add .github/workflows/main.yml
git commit -m "Replace broken ./test.sh with fast-tests + e2e-tests CI jobs"
git push -u origin <branch-name>
```

- [ ] **Step 5: Watch the first CI run on the PR**

Open the PR; wait for both `fast-tests` and `e2e-tests` jobs to complete on the PR. If `e2e-tests` exceeds 15 minutes wall-clock, the test config is too big — adjust `config.test.yaml` (drop a snapshot day, lower simpl) in a fixup commit.

- [ ] **Step 6: After merge, mark jobs required in branch protection**

This is a manual step in the GitHub UI:
- Settings → Branches → Branch protection rules → edit rule for `master` and `v1-epic`
- "Require status checks to pass before merging" → add `fast-tests` and `e2e-tests`

Document this in the PR description so the user knows to flip the switch after merge.

### Task 5.2: Open PR 5

```bash
gh pr create --base v1-epic --title "CI: replace ./test.sh with fast-tests + e2e-tests jobs" --body "$(cat <<'EOF'
## Summary
- Splits the existing single CI job into `fast-tests` (no data deps, ~1 min wall-clock) and `e2e-tests` (cached data, ~10 min wall-clock).
- Both jobs run on every push/PR; weekly cron job continues to test against upstream master of PyPSA/atlite/linopy.
- The old `./test.sh` step (file does not exist in repo — silent no-op) is removed.

## Post-merge action required
Manually mark `fast-tests` and `e2e-tests` as required status checks in branch protection settings for `master` and `v1-epic`.

## Test plan
- [x] YAML parses
- [ ] First CI run on this PR shows both jobs green
- [ ] e2e-tests wall-clock < 15 min on cold-cache CI runner
- [ ] e2e-tests wall-clock < 5 min on warm-cache CI runner
EOF
)"
```

---

# PR 6: Schema assertions (deferred)

**Prerequisite:** the schema-tracking initiative (`docs/superpowers/specs/2026-05-21-network-schema-tracking-design.md`) must merge first — specifically the `log_network_schema` helper in `workflow/scripts/_helpers.py` and the `docs/network-schema.md` catalog.

Do not start this PR until both prerequisites are visible on `v1-epic`. Sketch only:

- Create: `tests/integration/test_schema.py`
- For each artifact in `built`, call `log_network_schema(n, stage="<rule>")` to capture the column set.
- Load the catalog from `docs/network-schema.md` (parse the markdown tables or import a `network_schema.SCHEMA` Python object — defer to the schema spec's choice).
- Assert: every component column on the loaded network is registered in the catalog (no orphan columns).
- Assert: no required column in the catalog is missing from the network at the expected stage.

When this PR opens, it will likely surface schema gaps the catalog didn't anticipate — treat those as bugs in the catalog, not the test.

---

## Self-Review

**Spec coverage:** Every section of the spec has a corresponding task — Tier A is PR 2, Tier B fixture+smoke is PR 3, Tier B full assertions is PR 4, CI wiring is PR 5, schema deferral is PR 6. The migration plan in the spec is mirrored 1:1 here.

**Type consistency:** `BuiltArtifacts` dataclass introduced in Task 3.2 is consumed in Task 4.1 — properties (`elec_b`, `elec_s`, `elec_s_dem`, `elec_s_l_pp`, `elec_s_c`, `busmap_s`) are used consistently.

**Placeholders:** No TBDs. The one fuzzy step is "find the producer rule" in Task 2.3 step 3 — but the exact grep command is given.

**Risks not in the spec:**
1. PR 1's `pip install -e '.[test]'` may pull `snakemake==7.32.4` into a venv that doesn't yet have the rest of the heavy deps. That's fine for `pytest -m fast`'s purposes since none of the static tests import pypsa. The `[test]` extras list above includes only `pytest`, `pyyaml`, `snakemake`.
2. The test_paths.py regex `\{interconnect\}/(?!Geospatial/)` deliberately excludes the legacy `Geospatial/` subdirectory which appeared in phase2 paths pre-#12 reorg — confirm this exclusion is no longer needed after the merge by spot-checking. If `Geospatial/` is fully gone, simplify to `\{interconnect\}/`.
3. The session fixture in Task 3.2 uses `tmp_path_factory.mktemp("run").name` for the run name, which yields strings like `run0`. Snakemake's `RDIR` becomes `run0/`. If the test is parallelized with `pytest-xdist`, each worker gets its own session, so each worker rebuilds — undesirable. Keep `-j` out of pytest invocations until parallelization is intentional.
