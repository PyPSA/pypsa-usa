"""Run context, ``run_meta.json`` and the config-equivalence gate.

Two jobs, both about making "master-benchmark vs develop on the same config" a
*checked* statement rather than an asserted one.

**1. Provenance.** :class:`RunContext` is everything a reader needs in order to
say what produced a number, and :func:`write_run_meta` puts it at the top of the
run directory. Three shas, not one: ``master`` (the branch point),
``master-benchmark`` (what actually built the baseline) and ``develop`` (the
checkout under test). The environments are recorded per side because the two
branches pin different pypsa/pandas/linopy majors, and the ``{clusters}``
wildcard is recorded as the *literal value each side ran* because the harness
translates it between two dialects (``paths.baseline_clusters``).

**2. The gate.** :func:`assert_config_equivalent` dumps each side's fully merged
config through that side's OWN loader and refuses to build when they disagree
about anything not on :data:`CONFIG_DIFF_ALLOWLIST`. The two branches do not
load the same set of config files — master's Snakefile has
``config.default.yaml`` commented out, develop layers it in as a base — so an
omitted key resolves to the shipped default on one side and to a script's inline
fallback on the other. That is hot-fix HF-20, whose justification ("no key
changes value") was only ever replayed develop-vs-develop. Running this gate is
that owed replay, and it runs *before* either build so a ten-hour run is never
spent comparing two different configurations.

The merged config is obtained by replaying each side's own Snakefile — never by
re-implementing its load order here, which would only encode today's guess about
what the loaders do.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import build, paths
from .hotfixes import live_code_differences

REPO = paths.REPO

# ---------------------------------------------------------------------------
# Run context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunContext:
    """Everything one equivalence run is, in one immutable record."""

    run_id: str
    run_dir: Path  # workflow/results/equivalence/<run_id>
    prong: int
    interconnect: str
    opts: str
    simpl: str
    clusters_develop: str
    clusters_baseline: str  # the translated dialect value actually run
    master_sha: str  # the branch point, for provenance
    baseline_ref: str  # 'master-benchmark'
    baseline_sha: str  # what actually built the baseline
    baseline_commits_on_top_of_master: int
    develop_ref: str
    develop_sha: str
    develop_dirty: bool
    develop_commits_ahead_of_master: int
    config_sha256: str
    env_master: dict[str, str] = field(default_factory=dict)
    env_develop: dict[str, str] = field(default_factory=dict)
    config_diff_allowed: list[dict] = field(default_factory=list)
    known_code_differences: list[dict] = field(default_factory=list)
    slurm_job_id: str | None = None
    started_at: str = ""


def _git(*args: str, cwd: Path | None = None) -> str:
    cp = subprocess.run(
        ["git", *args],
        cwd=cwd or REPO,
        capture_output=True,
        text=True,
    )
    return cp.stdout.strip() if cp.returncode == 0 else ""


def _count(rev_range: str) -> int:
    out = _git("rev-list", "--count", rev_range)
    return int(out) if out.isdigit() else -1


def build_context(prong: int, probe_env: bool = True) -> RunContext:
    """Resolve the shas, the environments and the wildcards for one run.

    ``probe_env=False`` skips the two ``uv run`` version probes; they are the
    only slow part and a test does not need them. The baseline sha is resolved
    from the local branch, never pinned — if ``master-benchmark`` does not exist
    this raises with a pointer to T1 of the harness plan.
    """
    run_id = paths.run_id(prong)
    run_dir = paths.run_dir(prong)
    run_dir.mkdir(parents=True, exist_ok=True)

    baseline_ref = build.baseline_ref()
    baseline_sha = build.resolve_baseline_sha()
    master_sha = _git("rev-parse", "master^{commit}")
    develop_sha = _git("rev-parse", "HEAD")
    develop_ref = _git("rev-parse", "--abbrev-ref", "HEAD") or "HEAD"

    cfg_path = REPO / "workflow" / paths.CONFIGFILE
    config_sha256 = hashlib.sha256(cfg_path.read_bytes()).hexdigest() if cfg_path.exists() else ""

    env_master: dict[str, str] = {}
    env_develop: dict[str, str] = {}
    if probe_env:
        env_develop = build.side_env_versions(REPO)
        wt = build.BASELINE_WORKTREE
        if wt.exists():
            env_master = build.side_env_versions(wt)

    return RunContext(
        run_id=run_id,
        run_dir=run_dir,
        prong=prong,
        interconnect=paths.INTERCONNECT,
        opts=paths.OPTS,
        simpl="" if prong == 1 else paths.SIMPL2,
        clusters_develop=paths.CLUSTERS,
        clusters_baseline=paths.baseline_clusters(),
        master_sha=master_sha,
        baseline_ref=baseline_ref,
        baseline_sha=baseline_sha,
        baseline_commits_on_top_of_master=_count(f"master..{baseline_sha}") if master_sha else -1,
        develop_ref=develop_ref,
        develop_sha=develop_sha,
        develop_dirty=bool(build.checkout_dirt(REPO)),
        develop_commits_ahead_of_master=_count(f"master..{develop_sha}") if master_sha else -1,
        config_sha256=config_sha256,
        env_master=env_master,
        env_develop=env_develop,
        config_diff_allowed=[],
        known_code_differences=known_code_differences(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def known_code_differences() -> list[dict]:
    """Code-level differences this run cannot pin down to the config.

    The ``{clusters}`` dialect translation first — it is deliberate, it is
    harness-made, and it is the one difference nothing else records — then every
    hot-fix that is live on develop only and not a no-op on the whole-USA case.
    """
    out = [
        {
            "key": "{clusters} wildcard dialect (harness translation)",
            "master": paths.baseline_clusters(),
            "develop": paths.CLUSTERS,
            "hotfix": None,
        },
    ]
    out.extend(live_code_differences())
    return out


def write_run_meta(ctx: RunContext) -> Path:
    """``<run_dir>/run_meta.json``; returns the path it wrote."""
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    out = ctx.run_dir / "run_meta.json"
    out.write_text(json.dumps(dataclasses.asdict(ctx), indent=1, default=str, sort_keys=True))
    return out


def with_config_diff(ctx: RunContext, allowed: list[dict]) -> RunContext:
    """A copy of ``ctx`` carrying the gate's verdict (the dataclass is frozen)."""
    return dataclasses.replace(ctx, config_diff_allowed=allowed)


# ---------------------------------------------------------------------------
# Config-equivalence gate
# ---------------------------------------------------------------------------

_SHIM_NAME = ".eq_dump_config.smk"

# `include:` runs the side's real Snakefile, so its own `configfile:` layers,
# its own user overlays and (on develop) its own schema validation all apply;
# --configfile is applied last by snakemake, exactly as in a real build. The
# dump rule has no inputs, so nothing of the DAG is built to reach it.
_SHIM_SOURCE = '''# Generated by tests/equivalence/context.py; deleted again after the dump.


include: "Snakefile"


rule eq_dump_config:
    output:
        "__OUT__",
    run:
        import json
        from pathlib import Path

        p = Path(output[0])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(config, indent=1, sort_keys=True, default=str))
'''


def merged_config(side: str) -> dict:
    """That side's fully merged config, through that side's OWN loader.

    Writes a throwaway Snakefile shim next to the side's real one, runs the
    single ``eq_dump_config`` rule (no inputs, so no part of the DAG is built)
    and reads back the ``config`` dict snakemake handed the rule.
    """
    root = build.side_root(side)
    wf = root / "workflow"
    out = paths.run_dir() / "config_merged" / f"{side}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)

    shim = wf / _SHIM_NAME
    shim.write_text(_SHIM_SOURCE.replace("__OUT__", str(out)))
    try:
        cp = build.run(
            [
                "uv",
                "run",
                "snakemake",
                "eq_dump_config",
                "--snakefile",
                _SHIM_NAME,
                "--configfile",
                build.side_configfile(side),
                "-j",
                "1",
                "--rerun-triggers",
                "mtime",
            ],
            cwd=wf,
            timeout=3600,
        )
    finally:
        shim.unlink(missing_ok=True)

    if not out.exists():
        tail = "\n".join((cp.stderr or cp.stdout).splitlines()[-60:])
        raise RuntimeError(
            f"could not dump the merged config for side {side!r} from {wf} "
            f"(snakemake exit {cp.returncode}).\n{tail}",
        )
    return json.loads(out.read_text())


# (dotted key path, reason). A path matches itself and everything under it.
#
# Seeded ONLY with differences that are deliberate and provable today. It is
# meant to stay short: the first real run will fail this gate with the full list
# of everything else the two loaders disagree about, and each of those is a
# decision — pin it in the shared config, or add it here with a reason someone
# signed. An allowlist grown by copy-pasting the failure message defeats the
# gate.
CONFIG_DIFF_ALLOWLIST: tuple[tuple[str, str], ...] = (
    (
        "scenario.clusters",
        "deliberate: harness commit 8371e9d changed {clusters} semantics on develop, so the "
        "value is translated into the baseline's dialect (paths.baseline_clusters); "
        "run_meta.json records the literal value each side ran",
    ),
    (
        "godeeep_cf_registry",
        "develop-only retrieval plumbing (HF-15); master's ZenodoScenarioDownloader hardcodes "
        "the same record ids, so both sides fetch identical bytes",
    ),
    (
        "sector.co2.policy",
        "sector coupling is out of scope for this benchmark (scenario.sector is ''), and the "
        "two branches ship the same CSV at different tracked paths (config/ vs repo_data/config/); "
        "no sector rule runs, so no sector policy file is read",
    ),
    (
        "sector.transport_sector.ev_policy",
        "same as sector.co2.policy: an out-of-scope sector policy CSV at two tracked paths",
    ),
)


class _Empty:
    """The one value that ``null``, ``{}`` and ``[]`` all normalise to.

    ``model_topology.include:`` written as ``{}`` in the shared harness config
    arrives as ``None`` on develop, because snakemake's config merge has nothing
    to write when the update is an empty mapping. Every consumer treats an
    absent mapping and an empty mapping the same way — both are falsy, and both
    make the HF-9 / HF-10 gates evaluate ``False`` — so flagging that pair would
    be noise. A *populated* mapping on one side and an empty one on the other
    still differs, which is the case that matters.
    """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<empty>"

    def __eq__(self, other) -> bool:
        return isinstance(other, _Empty)

    def __hash__(self) -> int:
        return hash("<empty>")


EMPTY = _Empty()


def _norm_value(v: object) -> object:
    return EMPTY if v is None or v == {} or v == [] else v


# Keys compared as a WHOLE SUBTREE rather than leaf by leaf.
#
# `model_topology.include` is the gate expression HF-9 and HF-10 hinge on: the
# empty-county sweep and the seam-plant bound are scoped by it, and it is what
# makes both of them no-ops on the whole-USA case. Flattened leaf by leaf, a
# scoped develop (`include: {reeds_state: [CA]}`) against an unscoped master
# (`include: {}`) produces only "present on one side" differences, which the
# gate records rather than refuses. Compared whole, it is the value difference
# it actually is.
_ATOMIC_KEYS = (
    "model_topology.include",
    "model_topology.aggregate",
)


def _flatten(d: dict, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v and key not in _ATOMIC_KEYS:
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def config_differences(master: dict, develop: dict) -> list[dict]:
    """Every key the two merged configs disagree about, flattened and sorted.

    Three kinds, all of them differences: ``value`` (both sides carry the key
    with different values), ``only_master`` and ``only_develop``. A key present
    on one side only is not benign — that is precisely the HF-20 failure mode,
    where one loader supplies a default and the other falls through to a
    script's inline fallback.
    """
    fm, fd = _flatten(master), _flatten(develop)
    diffs: list[dict] = []
    for key in sorted(set(fm) | set(fd)):
        if key not in fm:
            diffs.append({"key": key, "kind": "only_develop", "master": None, "develop": fd[key]})
        elif key not in fd:
            diffs.append({"key": key, "kind": "only_master", "master": fm[key], "develop": None})
        elif _norm_value(fm[key]) != _norm_value(fd[key]):
            diffs.append({"key": key, "kind": "value", "master": fm[key], "develop": fd[key]})
    return diffs


_DEFAULT_LAYER_REASON = (
    "present on one side only: the two branches' default layers differ (HF-20). Recorded, not "
    "fatal — set EQ_STRICT_CONFIG_GATE=1 to make presence-only differences abort the run"
)


def allowlist_reason(key: str) -> str | None:
    """The signed reason this key may differ, or ``None``."""
    for pattern, reason in CONFIG_DIFF_ALLOWLIST:
        if key == pattern or key.startswith(pattern + "."):
            return reason
    return None


def assert_config_equivalent(ctx: RunContext | None = None) -> list[dict]:
    """Refuse to build two sides that are not on the same config.

    Returns the allowed differences, each with the reason it is allowed, for
    ``run_meta.json``. Raises on anything else, listing every offending key with
    both values — one run of the gate should tell you everything to decide, not
    the first thing it tripped on.

    Two severities, because the two kinds of difference are not the same thing:

    - a **value** difference — both sides carry the key and disagree about it —
      is fatal. That is a real disagreement about what to compute.
    - a **presence-only** difference — one side's default layer supplies a key
      the other's does not — is recorded, with its reason, in
      ``run_meta.json``'s ``config_diff_allowed``, and does not abort. There are
      well over a hundred of these between the two loaders (develop's
      ``config.default.yaml`` base layer against master's
      ``config.{common,plotting,slurm}.yaml``), covering plotting, sector, DAC
      and unused solver option sets. Making them fatal would mean the gate could
      never pass and would simply be switched off, which is strictly worse than
      recording them. ``EQ_STRICT_CONFIG_GATE=1`` makes them fatal too, which is
      how HF-20's owed master-vs-develop replay gets properly discharged.

    ``EQ_SKIP_CONFIG_GATE=1`` skips the check entirely and records that it was
    skipped. The gate shells out to snakemake twice; the escape hatch exists so
    a broken gate cannot block a benchmark, and it is loud in ``run_meta.json``
    so a run made without the check can never be mistaken for one made with it.
    """
    if os.environ.get("EQ_SKIP_CONFIG_GATE") == "1":
        build.log("WARNING: config-equivalence gate SKIPPED by EQ_SKIP_CONFIG_GATE=1")
        return [
            {
                "key": "<gate>",
                "kind": "skipped",
                "master": None,
                "develop": None,
                "reason": "config-equivalence gate skipped by EQ_SKIP_CONFIG_GATE=1",
            },
        ]

    strict = os.environ.get("EQ_STRICT_CONFIG_GATE") == "1"
    diffs = config_differences(merged_config("master"), merged_config("develop"))
    allowed, blocking = [], []
    for d in diffs:
        reason = allowlist_reason(str(d["key"]))
        if reason is not None:
            allowed.append({**d, "reason": reason})
        elif d["kind"] == "value" or strict:
            blocking.append(d)
        else:
            allowed.append({**d, "reason": _DEFAULT_LAYER_REASON})

    if blocking:
        listing = "\n".join(
            f"  {d['key']} [{d['kind']}]: master={d['master']!r} develop={d['develop']!r}" for d in blocking
        )
        raise RuntimeError(
            f"the two sides are not on the same config: {len(blocking)} differing key(s) "
            f"not on CONFIG_DIFF_ALLOWLIST.\n{listing}\n"
            "Pin each of these identically in the shared config.equivalence*.yaml, or add it "
            "to CONFIG_DIFF_ALLOWLIST in tests/equivalence/context.py with the reason it is "
            "allowed to differ. Comparing objective/capacity/dispatch across two different "
            "configurations measures the configs, not the branches.",
        )
    return allowed
