"""Tier A — the merged config of every shipped config must satisfy the schema.

``workflow/Snakefile`` runs ``snakemake.utils.validate`` on the merged config
at parse time. This test replays the same merge and the same validation
without spawning snakemake, so a schema/config mismatch is caught in
milliseconds rather than at the top of a build.

It also pins the schema's reason for existing: a typo'd key inside one of the
closed subtrees, and an out-of-range enum value, must both fail.
"""

from __future__ import annotations

import copy
from pathlib import Path

import jsonschema
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = REPO_ROOT / "workflow"
CONFIG_DIR = WORKFLOW_DIR / "repo_data" / "config"
SCHEMA_PATH = WORKFLOW_DIR / "schemas" / "config.schema.yaml"

# Must match the unconditional configfile: chain in workflow/Snakefile.
LAYERS = [
    "config.slurm.yaml",
    "config.common.yaml",
    "config.plotting.yaml",
    "config.api.yaml",
    "config.sector.yaml",
    "config.default.yaml",
]

SHIPPED_CONFIGS = [
    "config.default.yaml",
    "config.tutorial.yaml",
    "config.test.yaml",
    "config.equivalence.yaml",
    "config.equivalence-usa.yaml",
]


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text()) or {}


def _merge(a: dict, b: dict) -> dict:
    """Deep-merge b into a; b wins on conflict.

    Kept dependency-light (no snakemake import) like the other Tier A tests.
    It differs from ``snakemake.utils.update_config`` in exactly one corner —
    an empty mapping merged onto a null yields ``{}`` here and ``None`` there —
    which only reaches ``model_topology.include``/``aggregate``, typed
    ``[object, "null"]`` in the schema and therefore valid either way.
    """
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _merged(user_config: str) -> dict:
    merged: dict = {}
    for name in LAYERS:
        merged = _merge(merged, _load(name))
    if user_config not in LAYERS:
        merged = _merge(merged, _load(user_config))
    return merged


@pytest.fixture(scope="module")
def schema() -> dict:
    return yaml.safe_load(SCHEMA_PATH.read_text())


@pytest.mark.fast
def test_schema_is_a_valid_json_schema(schema):
    jsonschema.Draft7Validator.check_schema(schema)


@pytest.mark.fast
@pytest.mark.parametrize("config_name", SHIPPED_CONFIGS)
def test_shipped_configs_validate(config_name, schema):
    jsonschema.validate(_merged(config_name), schema)


@pytest.mark.fast
def test_typo_in_closed_subtree_is_rejected(schema):
    """A misspelled key under `electricity:` must not silently fall through."""
    cfg = copy.deepcopy(_merged("config.default.yaml"))
    cfg["electricity"]["retirment"] = cfg["electricity"].pop("retirement")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(cfg, schema)


@pytest.mark.fast
@pytest.mark.parametrize(
    "path,bad_value",
    [
        (("foresight",), "perfetc"),
        (("electricity", "retirement"), "econmic"),
        (("model_topology", "transmission_network"), "reads"),
        (("clustering", "cluster_network", "algorithm"), "hac"),
        (("solving", "solver", "name"), "gurbi"),
        (("costs", "atb", "scenario"), "moderate"),
        (("renewable_land_access",), "refrence"),
    ],
)
def test_bad_enum_values_are_rejected(path, bad_value, schema):
    cfg = copy.deepcopy(_merged("config.default.yaml"))
    node = cfg
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = bad_value
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(cfg, schema)


@pytest.mark.fast
def test_godeeep_requires_renewable_land_access_key(schema):
    """The godeeep conditional must fire when the key is missing entirely."""
    cfg = copy.deepcopy(_merged("config.default.yaml"))
    assert cfg["renewable"]["dataset"] == "godeeep"
    cfg.pop("renewable_land_access")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(cfg, schema)
