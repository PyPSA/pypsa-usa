"""Tier A — verify the snakemake DAG resolves on representative configs.

Catches: missing inputs, typo'd rule names, wildcard mismatches, and
syntactically broken ``.smk`` files. Runs ``snakemake -n`` (dry-run only,
no rule actually executes).

No seeding step: ``workflow/Snakefile`` reads its whole layered base and the
``policy_constraints/`` static inputs out of the tracked
``workflow/repo_data/config/`` tree, and treats the per-user files under
``workflow/config/`` as optional overlays. A fresh clone resolves without
``init_pypsa_usa.sh`` having been run.
"""

import subprocess
from pathlib import Path

import pytest

WORKFLOW_DIR = Path(__file__).resolve().parents[2] / "workflow"


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
        ("repo_data/config/config.tutorial.yaml", "cluster_network", []),
        ("repo_data/config/config.tutorial.yaml", "solve_network", []),
        ("repo_data/config/config.default.yaml", "cluster_network", []),
        ("repo_data/config/config.test.yaml", "cluster_network", []),
        ("repo_data/config/config.california.yaml", "cluster_network", []),
        ("repo_data/config/config.california.yaml", "solve_network", []),
        (
            "repo_data/config/config.california.yaml",
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
