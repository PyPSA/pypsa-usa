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


# County-resolution overrides for the California config. `config.california.yaml`
# ships this same block commented out; snakemake's ``--config`` performs a
# recursive dict update, so only the listed sub-keys are replaced. California has
# 58 counties, and ``simpl: county`` selects the county-FIPS fast path in
# ``cluster_simpl``.
CALIFORNIA_COUNTY_OVERRIDE = [
    "scenario={clusters: [58], simpl: ['county']}",
    "model_topology={topological_boundaries: 'county'}",
]


@pytest.mark.fast
@pytest.mark.parametrize(
    "configfile,target,overrides",
    [
        ("config/config.tutorial.yaml", "cluster_network", []),
        ("config/config.tutorial.yaml", "solve_network", []),
        ("config/config.default.yaml", "cluster_network", []),
        ("config/config.california.yaml", "cluster_network", []),
        ("config/config.california.yaml", "solve_network", []),
        (
            "config/config.california.yaml",
            "cluster_network",
            CALIFORNIA_COUNTY_OVERRIDE,
        ),
    ],
    ids=lambda v: "+".join(v) if isinstance(v, list) else v,
)
def test_snakemake_dryrun_resolves(configfile, target, overrides):
    cmd = [
        "snakemake",
        "-n",
        "--configfile",
        configfile,
        "--until",
        target,
        "--quiet",
    ]
    if overrides:
        cmd += ["--config", *overrides]
    result = subprocess.run(
        cmd,
        cwd=WORKFLOW_DIR,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"snakemake -n failed for {configfile} --until {target} (overrides={overrides})\n"
        f"stderr:\n{result.stderr}\n"
        f"stdout (last 50 lines):\n" + "\n".join(result.stdout.splitlines()[-50:])
    )
