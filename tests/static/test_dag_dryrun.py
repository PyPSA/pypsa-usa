"""Tier A — verify the snakemake DAG resolves on representative configs.

Catches: missing inputs, typo'd rule names, wildcard mismatches, and
syntactically broken ``.smk`` files. Runs ``snakemake -n`` (dry-run only,
no rule actually executes).

Snakemake reads scenario configs from ``workflow/config/``. Only
``config.common.yaml`` is tracked in git; the rest of the configs and
``policy_constraints/`` static inputs are seeded from
``workflow/repo_data/config/`` by ``init_pypsa_usa.sh`` at user-init
time. This test seeds any missing files in a session-scoped fixture so
it runs cleanly in CI without depending on the init script being run
ahead of pytest.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

WORKFLOW_DIR = Path(__file__).resolve().parents[2] / "workflow"
RUNTIME_CONFIG_DIR = WORKFLOW_DIR / "config"
TEMPLATE_CONFIG_DIR = WORKFLOW_DIR / "repo_data" / "config"


@pytest.fixture(scope="session", autouse=True)
def _seed_runtime_configs():
    """Mirror ``init_pypsa_usa.sh``: copy any missing scenario configs +
    ``policy_constraints/`` from ``workflow/repo_data/config/`` into
    ``workflow/config/`` so ``snakemake -n`` has the inputs it expects.
    Existing tracked files (e.g. ``config.common.yaml``) are left alone.
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
