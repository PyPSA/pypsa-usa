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
import re
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
    #: ``EQ_UNTIL``: ``'assembled'`` stops the compared pairs (and the built
    #: targets) at the assembled stage, ``''`` runs the whole chain. It decides
    #: WHICH artifacts were compared, so a re-run without it compares a
    #: different set of files while every sha in this record is identical —
    #: which is exactly the kind of silent difference run_meta.json exists to
    #: make impossible.
    until: str = ""
    #: What master's renewable-profile file WAS when the profile metrics read
    #: it. Master builds at substation resolution and develop at s{simpl}, so at
    #: prong 2 the harness rolls master up onto develop's cluster bus space
    #: before any distributional statistic is taken
    #: (``metrics.aggregate_profile_to_clusters``). Recorded because a CF
    #: quantile is meaningless without the bus population it was taken over.
    #: ``build_context`` can only say what is PLANNED (it runs before either
    #: side builds); ``compare.run_comparison`` rewrites this field in the
    #: written record with what the comparison actually did.
    master_profile_stage: str = "nodal"
    #: One entry per prong-2 profile pair: whether the rollup ran and how the
    #: two cluster sets compared (``metrics.cluster_sets``). A cluster on one
    #: side only is reported here and as a ``cluster_set`` row_set finding, and
    #: is excluded from the pooled profile metrics. Filled in after the
    #: comparison, by the same ``update_run_meta`` pass.
    profile_cluster_sets: list[dict] = field(default_factory=list)
    env_master: dict[str, str] = field(default_factory=dict)
    env_develop: dict[str, str] = field(default_factory=dict)
    # 'pending' until the gate has run, then 'passed' / 'failed' / 'skipped'.
    # run_meta.json is written BEFORE the gate, so a run that the gate stops
    # still leaves provenance behind saying which three shas it was about to
    # compare and why it did not.
    config_gate: str = "pending"
    config_gate_error: str = ""
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


def master_profile_stage(prong: int, rolled_up: bool | None = None) -> str:
    """Bus resolution master's profile file was compared AT, for ``run_meta.json``.

    Prong 1 compares the two nodal files bus-for-bus. Prong 2 cannot: master
    builds ``profile_{tech}.nc`` at substation resolution and develop builds
    ``profile_{tech}_s{simpl}.nc`` at cluster resolution, and a capacity-factor
    quantile is a property of its bus population as much as of the weather. So
    master is rolled up first, and the record says which object the numbers came
    from.

    ``rolled_up`` is the OBSERVED outcome, not the intention:

    - ``None`` — nothing has run yet (``build_context`` mints the record before
      either side builds), so the field says the rollup is planned.
    - ``True`` / ``False`` — what ``compare.compare_profiles`` and
      ``plots.collect_metrics`` actually did. The rollup needs develop's
      ``busmap_s{simpl}.csv``; without it the two sides were compared at
      different resolutions, and a reader must not have to infer that from the
      prong number. ``compare.run_comparison`` rewrites the field with this.
    """
    if prong == 1:
        return "nodal"
    if rolled_up is None:
        return f"nodal; rollup to s{paths.SIMPL2} planned, not yet run"
    if rolled_up:
        return f"nodal->s{paths.SIMPL2} (p_nom_max-weighted)"
    return f"nodal (rollup did NOT run: no busmap_s{paths.SIMPL2} or no bus dimension)"


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
        until=paths.UNTIL,
        master_profile_stage=master_profile_stage(prong),
        env_master=env_master,
        env_develop=env_develop,
        config_gate="pending",
        config_gate_error="",
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


def update_run_meta(*, run_dir: Path | None = None, **fields: object) -> Path:
    """Merge ``fields`` into the run's ``run_meta.json`` and rewrite it.

    Provenance that is only knowable AFTER the comparison —
    ``master_profile_stage``, ``profile_cluster_sets`` — has to reach the same
    file as the shas, or a reader has two records to reconcile. The file is
    created if it does not exist yet, so a comparison run standalone (no
    ``run.py``) still leaves the facts behind rather than dropping them.
    """
    out = Path(run_dir or paths.run_dir()) / "run_meta.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    current: dict = {}
    if out.exists():
        try:
            loaded = json.loads(out.read_text())
            if isinstance(loaded, dict):
                current = loaded
        except (OSError, ValueError):
            current = {}
    current.update(fields)
    out.write_text(json.dumps(current, indent=1, default=str, sort_keys=True))
    return out


def with_config_diff(ctx: RunContext, allowed: list[dict], status: str = "passed", error: str = "") -> RunContext:
    """A copy of ``ctx`` carrying the gate's verdict (the dataclass is frozen)."""
    return dataclasses.replace(
        ctx,
        config_gate=status,
        config_gate_error=error,
        config_diff_allowed=allowed,
    )


def run_gate(ctx: RunContext) -> RunContext:
    """Write provenance, run the gate, rewrite provenance; re-raise on failure.

    ``run_meta.json`` exists before the gate and is rewritten after it, so a
    gate failure is not a run with no record: the file names the three shas, the
    config sha and the reason the run stopped. Callers get the exception either
    way.
    """
    write_run_meta(ctx)
    try:
        allowed = assert_config_equivalent(ctx)
    except Exception as exc:
        write_run_meta(with_config_diff(ctx, [], status="failed", error=str(exc)))
        raise
    status = "skipped" if any(d.get("kind") == "skipped" for d in allowed) else "passed"
    ctx = with_config_diff(ctx, allowed, status=status)
    write_run_meta(ctx)
    return ctx


# ---------------------------------------------------------------------------
# Config-equivalence gate
# ---------------------------------------------------------------------------

_SHIM_NAME = ".eq_dump_config.smk"

# `include:` runs the side's real Snakefile, so its own `configfile:` layers,
# its own user overlays and (on develop) its own schema validation all apply;
# --configfile is applied last by snakemake, exactly as in a real build. The
# dump rule has no inputs, so nothing of the DAG is built to reach it.
_SHIM_SOURCE = """# Generated by tests/equivalence/context.py; deleted again after the dump.


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
"""


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


# --- The allowlist ----------------------------------------------------------
#
# (dotted key path, reason). A path matches itself and everything under it.
# Every entry is a signed statement that this key cannot change what either
# side computes on the standing whole-USA benchmark config. The entries below
# were enumerated by actually running the gate against both merged configs on
# 2026-09-14 and checking each one in the two branches' source; the reason says
# what was checked, not that it was checked.
#
# Where safety depends on the VALUE and not just the key, the entry also has a
# guard in _ALLOWLIST_GUARDS. An allowlisted key whose guard fails is not
# allowlisted — that is what stops "develop's ATB defaults happen to match
# master's inline fallback" from silently becoming "develop's ATB defaults are
# allowed to be anything".


def _is_time(v: object) -> bool:
    return isinstance(v, str) and bool(re.fullmatch(r"\d+:\d{2}:\d{2}", v))


def _present_values(diff: dict) -> list:
    """Every side that actually carries a value.

    Both sides, not one: a ``value`` difference has a real value on each side,
    and a ``default_only`` difference has one on exactly one of them. Reading
    only ``master`` when it happens to be non-``None`` was a live bug — master
    ``{}`` against develop ``{enable: true, ...}`` handed the guard the empty
    mapping, which duly reported "shipped enable: false" and allowed an ENABLED
    block through. A guard holds only if it holds for everything present.
    """
    return [v for v in (diff.get("master"), diff.get("develop")) if not _is_empty(v)]


def _every_present(predicate):
    """Lift a value predicate into a guard over both sides of a difference."""

    def guard(diff: dict) -> bool:
        values = _present_values(diff)
        return bool(values) and all(predicate(v) for v in values)

    return guard


_rule_walltime_block = _every_present(
    lambda v: isinstance(v, dict) and set(v) == {"walltime"} and _is_time(v["walltime"]),
)
_walltime_map = _every_present(
    lambda v: isinstance(v, dict) and bool(v) and all(_is_time(x) for x in v.values()),
)
_time_scalar = _every_present(_is_time)


def _not_population(diff: dict) -> bool:
    return "population" not in (diff["master"], diff["develop"])


def _is_inline_atb_fallback(v: object) -> bool:
    """Does this ATB block equal the fallback both branches hard-code.

    Both read it the same way — ``(costs_config or {}).get("atb") or {}`` in
    ``_helpers``, with inline defaults ``scenario="Moderate"``,
    ``model_case="Market"``, ``overrides={}``. Develop's default layer supplies
    exactly those, so the key being absent on master changes nothing. Change any
    of them and it does.
    """
    return (
        isinstance(v, dict)
        and v.get("scenario") == "Moderate"
        and v.get("model_case") == "Market"
        and not v.get("overrides")
    )


def _is_disabled_block(v: object) -> bool:
    """A capability block whose own on-switch is off."""
    return isinstance(v, dict) and not v.get("enable", False) and not v.get("activate", False)


_atb_equals_inline_fallback = _every_present(_is_inline_atb_fallback)
_disabled_block = _every_present(_is_disabled_block)


CONFIG_DIFF_ALLOWLIST: tuple[tuple[str, str], ...] = (
    # --- deliberate, harness-made -------------------------------------------
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
    # --- out of scope for this benchmark ------------------------------------
    (
        "sector.co2.policy",
        "sector coupling is out of scope (scenario.sector is ''), and the two branches ship the "
        "same CSV at different tracked paths (config/ vs repo_data/config/); no sector rule runs",
    ),
    (
        "sector.transport_sector.ev_policy",
        "same as sector.co2.policy: an out-of-scope sector policy CSV at two tracked paths",
    ),
    (
        "dac",
        "direct-air-capture block, sector-coupling only and shipped enable: false; no sector rule "
        "runs at scenario.sector ''",
    ),
    (
        "renewable.EGS",
        "EGS supply-curve settings; EGS is not in electricity.extendable_carriers.Generator for "
        "this config, so aggregate_egs is not in the DAG on either side",
    ),
    (
        "renewable.hydro",
        "master-only atlite hydro-inflow settings (hydrobasins runoff, normalisation, PHS hours). "
        "Both branches attach hydro from the Breakthrough/EIA fleet on this config; the atlite "
        "hydro path is not in either DAG",
    ),
    (
        "plotting",
        "master-only plot axis limits and thresholds; no plotting rule is in the benchmark target "
        "chain on either side",
    ),
    # --- opt-gated, and this run's opts do not select them -------------------
    (
        "electricity.SAFE_reservemargin",
        "read only by the SAFE opt; this run's opts are 3h / REM-3h",
    ),
    (
        "electricity.SAFE_regional_reservemargins",
        "read only by the SAFE opt; this run's opts are 3h / REM-3h",
    ),
    (
        "electricity.erm",
        "read only by the ERM opt; this run's opts are 3h / REM-3h",
    ),
    (
        "electricity.transmission_interface_limits",
        "dead config key: no script reads it on either branch, and develop's solve rule takes the "
        "CSV as a hard-coded rule input. model_topology.interface_transmission_limits is false here",
    ),
    (
        "electricity.demand.scenario.eer_file",
        "read only by the eer demand profile; this run pins electricity.demand.profile: efs",
    ),
    (
        "electricity.demand.scenario.servm_weather_years",
        "read only by the servm demand profile; this run pins electricity.demand.profile: efs",
    ),
    # --- capabilities shipped default-off (HF-21) ----------------------------
    (
        "conventional.ambient_derate",
        "HF-21 default-off capability (enable: false); develop raises if it is ever enabled",
    ),
    (
        "electricity.imports",
        "HF-21 default-off capability (enable: false)",
    ),
    (
        "electricity.exports",
        "HF-21 default-off capability (enable: false)",
    ),
    (
        "electricity.remote_contracted_resources",
        "HF-21 default-off capability (enable: false)",
    ),
    (
        "electricity.operational_reserve",
        "default-off capability (activate: false)",
    ),
    (
        "nrel_caps_reassign",
        "HF-21 default-off capability (enable: false); flag-off output was verified "
        "xr.testing.assert_identical to the pre-change code",
    ),
    (
        "ucap",
        "develop-only unforced-capacity block, shipped enable: false",
    ),
    (
        "run.benchmark_cpuc_horizons",
        "read only when run.benchmark_cpuc is true, which is itself develop-only and false " "(HF-21)",
    ),
    # --- values that match the other side's inline fallback ------------------
    (
        "costs.atb",
        "develop's default layer supplies exactly master's inline fallback — see "
        "_atb_equals_inline_fallback; both branches read it as (costs_config or {}).get('atb') or {}",
    ),
    (
        "clustering.cluster_network.weighting_strategy",
        "the only branch either branch takes on this key is == 'population'; master resolves it to "
        "None and develop to 'demand-capacity', so both take the same gen+load weighting",
    ),
    (
        "clustering.simplify_network.weighting_strategy",
        "same as clustering.cluster_network.weighting_strategy: neither value is 'population'",
    ),
    (
        "offshore_network.enable",
        "dead key: build_base_network reads only offshore_network['bus_spacing'] on both branches; "
        "nothing reads 'enable'",
    ),
    # --- resource declarations, read by the scheduler and never by a script --
    (
        "walltime",
        "develop's per-rule walltime block; a Slurm resource declaration, never read by a script. "
        "HF-20 moved these from nine dead top-level <rule>.walltime keys into one block",
    ),
    (
        "solving.walltime",
        "solve-stage walltime; a Slurm resource declaration, never read by a script",
    ),
    (
        "add_demand",
        "master-only dead top-level <rule>.walltime key (HF-20 removed these on develop)",
    ),
    (
        "add_electricity",
        "master-only dead top-level <rule>.walltime key (HF-20 removed these on develop)",
    ),
    (
        "build_renewable_profiles",
        "master-only dead top-level <rule>.walltime key (HF-20 removed these on develop)",
    ),
    (
        "cluster_network",
        "master-only dead top-level <rule>.walltime key (HF-20 removed these on develop)",
    ),
    (
        "simplify_network",
        "master-only dead top-level <rule>.walltime key (HF-20 removed these on develop)",
    ),
    (
        "solve_network",
        "master-only dead top-level <rule>.walltime key (HF-20 removed these on develop)",
    ),
    (
        "solve_network_validation",
        "master-only dead top-level <rule>.walltime key (HF-20 removed these on develop)",
    ),
    # --- solver option sets this run does not select -------------------------
    (
        "solving.solver_options.cplex-default",
        "alternative solver option set; this run pins solving.solver.options: gurobi-default, "
        "which both sides carry identically",
    ),
    (
        "solving.solver_options.highs-default",
        "alternative solver option set; this run pins solving.solver.options: gurobi-default",
    ),
    (
        "solving.solver_options.gurobi-fallback",
        "alternative solver option set; this run pins solving.solver.options: gurobi-default",
    ),
    (
        "solving.solver_options.gurobi-numeric-focus",
        "alternative solver option set; this run pins solving.solver.options: gurobi-default",
    ),
)

# Guards for the entries whose safety depends on the value, not just the key.
_ALLOWLIST_GUARDS = {
    "costs.atb": _atb_equals_inline_fallback,
    "clustering.cluster_network.weighting_strategy": _not_population,
    "clustering.simplify_network.weighting_strategy": _not_population,
    "conventional.ambient_derate": _disabled_block,
    "dac": _disabled_block,
    "electricity.imports": _disabled_block,
    "electricity.exports": _disabled_block,
    "electricity.remote_contracted_resources": _disabled_block,
    "electricity.operational_reserve": _disabled_block,
    "nrel_caps_reassign": _disabled_block,
    "ucap": _disabled_block,
    "walltime": _walltime_map,
    "solving.walltime": _time_scalar,
    "add_demand": _rule_walltime_block,
    "add_electricity": _rule_walltime_block,
    "build_renewable_profiles": _rule_walltime_block,
    "cluster_network": _rule_walltime_block,
    "simplify_network": _rule_walltime_block,
    "solve_network": _rule_walltime_block,
    "solve_network_validation": _rule_walltime_block,
}


class _Empty:
    """What ``null``, ``{}``, ``[]``, ``""``, ``0`` and ``False`` all mean here.

    ``model_topology.include:`` written as ``{}`` in the shared harness config
    arrives as ``None`` on develop, because snakemake's config merge has nothing
    to write when the update is an empty mapping. Every consumer treats an
    absent mapping and an empty mapping the same way — both are falsy, and both
    make the HF-9 / HF-10 gates evaluate ``False`` — so flagging that pair would
    be noise.

    The same falsiness is what decides whether a key present on ONE side only
    can change that side's behaviour. ``config["electricity"].get(k, {})``
    followed by ``if cfg:`` is the shape all over this workflow; a key whose
    default-layer value is falsy takes the same branch as an absent key, and a
    key whose value is truthy does not. See :func:`_is_empty`.
    """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<empty>"

    def __eq__(self, other) -> bool:
        return isinstance(other, _Empty)

    def __hash__(self) -> int:
        return hash("<empty>")


EMPTY = _Empty()

_EMPTY_LITERALS = (None, {}, [], "", 0, False)


def _is_empty(v: object) -> bool:
    """Say whether a value is indistinguishable from the key being absent.

    ``0`` and ``False`` count, and so do ``""``/``[]``/``{}``: every one of them
    is falsy, so a ``if config.get(key):`` guard takes the absent branch. This
    is the whole basis for treating some present-on-one-side keys as harmless.
    """
    return any(v is lit or (type(v) is type(lit) and v == lit) for lit in _EMPTY_LITERALS)


def _norm_value(v: object) -> object:
    return EMPTY if _is_empty(v) else v


# Keys compared as a WHOLE SUBTREE rather than recursed into.
#
# `model_topology.include` is the gate expression HF-9 and HF-10 hinge on: the
# empty-county sweep and the seam-plant bound are scoped by it, and it is what
# makes both of them no-ops on the whole-USA case. Recursed into, a scoped
# develop (`include: {reeds_state: [CA]}`) against an unscoped master
# (`include: {}`) produces only "present on one side" differences; compared
# whole, it is the value difference it actually is.
_ATOMIC_KEYS = (
    "model_topology.include",
    "model_topology.aggregate",
)


def _flatten(d: dict, prefix: str = "") -> dict[str, object]:
    """Dotted-path view of one config; atomic keys stay whole.

    Used by tests and by anything that wants a flat view. The gate itself walks
    the two configs TOGETHER (:func:`config_differences`), because a populated
    subtree on one side against an empty one on the other is a single value
    difference, not a scatter of present-on-one-side keys.
    """
    out: dict[str, object] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v and key not in _ATOMIC_KEYS:
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


# `value` and `default_only` abort the run; `only_master`/`only_develop` are
# recorded. See assert_config_equivalent for why the split falls here.
FATAL_KINDS = ("value", "default_only")

_FALSY_PRESENCE_REASON = (
    "present on one side only with a FALSY value, so the side without the key takes the same "
    "branch (HF-20's diffuse class). Recorded, not fatal — EQ_STRICT_CONFIG_GATE=1 aborts on "
    "these too"
)


def _presence_kind(side: str, value: object) -> str:
    """``only_<side>`` when the value is falsy, ``default_only`` when it is not.

    A key one side's default layer supplies and the other's does not is only
    harmless if the value is falsy, because that is the branch the side WITHOUT
    the key takes. ``electricity.demand_response: {marginal_cost: 999999,
    shift: 0}`` is the counter-example that motivated this split: it is truthy,
    so develop calls ``add_demand_response``, which adds a ``demand_response``
    Carrier before its own ``shift == 0`` early return, while the baseline's
    ``.get("demand_response", {})`` is falsy and adds nothing. One unwaived
    Carrier row_set finding at three stages, from a key nobody thought could
    matter.
    """
    return f"only_{side}" if _is_empty(value) else "default_only"


def config_differences(master: dict, develop: dict) -> list[dict]:
    """Every key the two merged configs disagree about, walked together.

    Four kinds:

    - ``value`` — both sides carry the key and disagree about it, or one side
      has a populated mapping where the other has an empty one;
    - ``default_only`` — present on one side only, with a TRUTHY value, so the
      two sides can take different branches on it;
    - ``only_master`` / ``only_develop`` — present on one side only with a falsy
      value, which is the same branch the other side takes anyway.

    The first two are the gate's business. The last two are the diffuse HF-20
    class and are recorded rather than refused.
    """
    diffs: list[dict] = []
    _walk(master, develop, "", diffs)
    return diffs


def _walk(master: dict, develop: dict, prefix: str, out: list[dict]) -> None:
    for k in sorted(set(master) | set(develop)):
        key = f"{prefix}{k}"
        in_m, in_d = k in master, k in develop
        mv, dv = master.get(k), develop.get(k)

        if not in_m:
            out.append({"key": key, "kind": _presence_kind("develop", dv), "master": None, "develop": dv})
            continue
        if not in_d:
            out.append({"key": key, "kind": _presence_kind("master", mv), "master": mv, "develop": None})
            continue

        m_sub = isinstance(mv, dict) and bool(mv)
        d_sub = isinstance(dv, dict) and bool(dv)
        if m_sub and d_sub and key not in _ATOMIC_KEYS:
            _walk(mv, dv, key + ".", out)
        elif _norm_value(mv) != _norm_value(dv):
            # Covers a populated subtree against {} / None / a scalar: the whole
            # mapping is the differing value, not a scatter of missing leaves.
            out.append({"key": key, "kind": "value", "master": mv, "develop": dv})


def allowlist_reason(key: str, diff: dict | None = None) -> str | None:
    """The signed reason this key may differ, or ``None``.

    When ``diff`` is supplied and the matching entry has a guard, the guard has
    to agree: an allowlisted key whose value has moved out from under its reason
    is not allowlisted any more.
    """
    for pattern, reason in CONFIG_DIFF_ALLOWLIST:
        if key != pattern and not key.startswith(pattern + "."):
            continue
        guard = _ALLOWLIST_GUARDS.get(pattern)
        if guard is not None and diff is not None and not guard(diff):
            return None
        return reason
    return None


def assert_config_equivalent(ctx: RunContext | None = None) -> list[dict]:
    """Refuse to build two sides that are not on the same config.

    Returns the allowed differences, each with the reason it is allowed, for
    ``run_meta.json``. Raises on anything else, listing every offending key with
    both values — one run of the gate should tell you everything to decide, not
    the first thing it tripped on.

    Two severities, split on whether the two sides can take different branches:

    - **fatal** — a ``value`` difference (both sides carry the key and disagree)
      or a ``default_only`` difference (present on one side only, with a TRUTHY
      value). Both mean the sides can compute different things. Every one of
      these must be pinned in the shared config or allowlisted with a signed
      reason.
    - **recorded** — ``only_master`` / ``only_develop``, i.e. present on one
      side only with a FALSY value. ``config.get(key, {})`` followed by
      ``if cfg:`` takes the same branch either way, so the absent side behaves
      as the present one does. These land in ``run_meta.json``'s
      ``config_diff_allowed`` with their reason.
      ``EQ_STRICT_CONFIG_GATE=1`` makes them fatal too, which is how HF-20's
      owed master-vs-develop replay gets discharged in full.

    The falsy/truthy split is not a nicety. ``electricity.demand_response:
    {marginal_cost: 999999, shift: 0}`` lives in develop's default layer only;
    it is truthy, so develop calls ``add_demand_response``, which adds a
    ``demand_response`` Carrier *before* its own ``shift == 0`` early return,
    while the baseline's ``.get(..., {})`` is falsy and adds nothing — three
    stages of unwaived Carrier ``row_set`` findings from a key that looks inert.

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
        reason = allowlist_reason(str(d["key"]), d)
        if reason is not None:
            allowed.append({**d, "reason": reason})
        elif d["kind"] in FATAL_KINDS or strict:
            blocking.append(d)
        else:
            allowed.append({**d, "reason": _FALSY_PRESENCE_REASON})

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
