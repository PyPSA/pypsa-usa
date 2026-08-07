"""Tier A — assert every snakemake.config[...] access in scripts is defined.

AST-walks every script under workflow/scripts/, extracts subscript chains
rooted at ``snakemake.config`` and asserts each key path exists in the
merged default config. Catches dead/typo'd config keys (the class of bug
fixed in PR #10).

Canonical config templates live under ``workflow/repo_data/config/`` and
are copied into ``workflow/config/`` by ``init_pypsa_usa.sh``. The latter
also contains the committed ``config.common.yaml`` override. For maximum
coverage we merge the canonical templates and then overlay any tracked
runtime overrides on top.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = REPO_ROOT / "workflow"
SCRIPT_DIR = WORKFLOW_DIR / "scripts"
REPO_DATA_CONFIG_DIR = WORKFLOW_DIR / "repo_data" / "config"
RUNTIME_CONFIG_DIR = WORKFLOW_DIR / "config"

# Merge order matches the configfile chain in workflow/Snakefile plus
# config.default.yaml (the maximal scenario config). config.tutorial.yaml
# is intentionally omitted — it's a minimal opt-in subset used for fast
# DAG smoke tests, not a key-vocabulary source of truth.
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


def _load_merged_config() -> dict:
    merged: dict = {}
    for name in DEFAULT_CONFIGFILES:
        # Prefer the runtime workflow/config copy if present (it may carry
        # tracked overrides), otherwise fall back to the canonical template.
        runtime_path = RUNTIME_CONFIG_DIR / name
        template_path = REPO_DATA_CONFIG_DIR / name
        path = runtime_path if runtime_path.exists() else template_path
        if not path.exists():
            continue
        with open(path) as f:
            merged = _merge(merged, yaml.safe_load(f) or {})
    return merged


def _key_exists(cfg, keys) -> bool:
    """Walk a list of keys into nested dicts. True if every key is defined."""
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return False
        node = node[k]
    return True


def _subscript_chain(node):
    """Return (root_node, [keys]) for chained Subscript nodes, or None."""
    keys: list[str] = []
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
    """True if node is ``snakemake.config``."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "config"
        and isinstance(node.value, ast.Name)
        and node.value.id == "snakemake"
    )


def _extract_config_accesses(source: str) -> list[list[str]]:
    """Return key paths accessed via ``snakemake.config[...][...]``."""
    tree = ast.parse(source)
    accesses: list[list[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        chain = _subscript_chain(node)
        if chain is None:
            continue
        root, keys = chain
        if _is_snakemake_config(root) and keys:
            accesses.append(keys)
    return accesses


@pytest.fixture(scope="module")
def merged_config() -> dict:
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
    # Deduplicate identical key paths in the same script.
    seen: set[tuple[str, ...]] = set()
    unique_missing = []
    for keys in missing:
        t = tuple(keys)
        if t in seen:
            continue
        seen.add(t)
        unique_missing.append(keys)
    assert not unique_missing, (
        f"{script.name}: snakemake.config keys not defined in any "
        f"workflow/config/*.yaml or workflow/repo_data/config/*.yaml:\n"
        + "\n".join(f"  config[{']['.join(repr(k) for k in keys)}]" for keys in unique_missing)
    )
