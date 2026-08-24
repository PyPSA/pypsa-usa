"""Tier B — session-scoped fixture that runs `snakemake --until cluster_network`
once per test session and exposes paths to the produced artifacts.

Skipped automatically if required data dirs are missing (lets developers
run `pytest -m fast` locally without setting up the data deps).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pypsa
import pytest

# Keep network frames on numpy object dtype under pandas 3 (matches _helpers),
# so the artifacts these tests read back match what the workflow produced.
if hasattr(pypsa, "options"):
    pypsa.options.api.legacy_string_dtype = True

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = REPO_ROOT / "workflow"
RUNTIME_CONFIG_DIR = WORKFLOW_DIR / "config"
TEMPLATE_CONFIG_DIR = WORKFLOW_DIR / "repo_data" / "config"
DATA_DIRS = [WORKFLOW_DIR / "data", WORKFLOW_DIR / "cutouts", WORKFLOW_DIR / "repo_data"]


def pytest_collection_modifyitems(config, items):
    """Reject pytest-xdist for integration tests.

    The ``built`` session fixture runs snakemake once per session against the
    shared ``workflow/`` directory. With xdist, each worker is its own
    session — they would race on ``.snakemake/`` locks, repeat the (slow)
    snakemake build, and contend on the seeding step that copies templates
    from ``repo_data/config/``. Until the fixture is refactored to use a
    truly isolated per-worker workflow dir, refuse to run.
    """
    if not any(item.get_closest_marker("integration") for item in items):
        return
    workers = getattr(config.option, "numprocesses", None)
    if workers and workers != 0:
        pytest.exit(
            "tests/integration are not safe under pytest-xdist (see conftest "
            "docstring). Run them without `-n` / `--numprocesses`.",
            returncode=4,
        )


@pytest.fixture(scope="session", autouse=True)
def _seed_runtime_configs():
    """Mirror ``init_pypsa_usa.sh``: copy any missing scenario configs +
    ``policy_constraints/`` from ``workflow/repo_data/config/`` into
    ``workflow/config/`` so snakemake has the inputs it expects.
    Existing tracked files (e.g. ``config.common.yaml``) are left alone.

    Duplicated from ``tests/static/test_dag_dryrun.py`` so Tier B works
    without depending on Tier A collection order.
    """
    if not TEMPLATE_CONFIG_DIR.exists():
        return
    RUNTIME_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for src in TEMPLATE_CONFIG_DIR.iterdir():
        dst = RUNTIME_CONFIG_DIR / src.name
        if dst.exists():
            continue
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


@dataclass(frozen=True)
class BuiltArtifacts:
    """Paths to the per-stage artifacts produced by the Tier B snakemake build.

    The ``interconnect``, ``simpl``, and ``clusters`` defaults MUST match
    ``workflow/repo_data/config/config.test.yaml``'s ``scenario`` section.
    If you change one, change both — the smoke test will fail loudly but
    only after the (slow) build, wasting CI time.
    """

    run_name: str
    base: Path  # resources/{run_name}/
    interconnect: str = "western"
    simpl: str = "20"
    clusters: str = "4m"

    @property
    def elec_b(self) -> Path:
        """Base electrical network (pre-simplification)."""
        return self.base / "networks" / self.interconnect / "elec_b.nc"

    @property
    def elec_s(self) -> Path:
        """Simplified electrical network (after cluster_simpl)."""
        return self.base / "networks" / self.interconnect / f"elec_s{self.simpl}.nc"

    @property
    def elec_s_dem(self) -> Path:
        """Simplified network with demand attached."""
        return self.base / "networks" / self.interconnect / f"elec_s{self.simpl}_dem.nc"

    @property
    def elec_s_l_pp(self) -> Path:
        """Simplified network with line + powerplant attributes added."""
        return self.base / "networks" / self.interconnect / f"elec_s{self.simpl}_l_pp.pkl"

    @property
    def elec_s_c(self) -> Path:
        """Final clustered network (after cluster_network)."""
        return self.base / "networks" / self.interconnect / f"elec_s{self.simpl}_c{self.clusters}.nc"

    @property
    def busmap_s(self) -> Path:
        """Bus mapping from base -> simplified network."""
        return self.base / "busmaps" / self.interconnect / f"busmap_s{self.simpl}.csv"


@pytest.fixture(scope="session")
def built(tmp_path_factory) -> BuiltArtifacts:
    """Run ``snakemake --until cluster_network`` once per session and expose artifact paths.

    Skips the session's integration tests if ``workflow/data``, ``workflow/cutouts``,
    or ``workflow/repo_data`` are missing — keeps Tier A runnable on bare clones.
    """
    missing = [d for d in DATA_DIRS if not d.exists()]
    if missing:
        pytest.skip(
            "Integration tests require populated data dirs; missing: " + ", ".join(str(m) for m in missing),
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
        # Force greedy scheduler to avoid the ILP scheduler's cbc dependency
        # (cbc is shipped non-executable in some envs and causes PermissionError).
        "--scheduler",
        "greedy",
        "--quiet",
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=WORKFLOW_DIR,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired as e:
        raw_stderr = e.stderr
        if isinstance(raw_stderr, bytes):
            raw_stderr = raw_stderr.decode("utf-8", errors="replace")
        stderr_tail = "\n".join(raw_stderr.splitlines()[-100:]) if raw_stderr else "<no stderr captured>"
        pytest.fail(
            f"snakemake build timed out after {e.timeout}s\nstderr (last 100 lines):\n{stderr_tail}",
        )
    if result.returncode != 0:
        pytest.fail(
            f"snakemake build failed (exit {result.returncode}):\n"
            f"stderr (last 100 lines):\n" + "\n".join(result.stderr.splitlines()[-100:]),
        )
    return BuiltArtifacts(
        run_name=run_name,
        base=WORKFLOW_DIR / "resources" / run_name,
    )
