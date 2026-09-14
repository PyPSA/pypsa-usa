"""Artifact path map for the Tier C equivalence harness.

Pairs ``develop`` artifacts (category-first resources layout) with ``master``
artifacts (flat ``{interconnect}/`` layout) for one prong of the two-prong
protocol. Paths are relative to each side's ``workflow/`` directory; the run
name is fixed to ``equivalence``.

The baseline is the ``master-benchmark`` branch off ``master`` (see
``memory/plans/harness-master-vs-develop.md`` in the project brain, T1), NOT a
pinned upstream commit patched at build time. ``BASELINE_REF`` is resolved to a
full sha at run time by ``build.resolve_baseline_sha``.

Pairing facts: the two DAGs pass through the same logical states under
different file names. Notably develop's assembled substation network is
``elec_s{simpl}_l_pp.pkl`` (dill) while master's is ``elec_s{simpl}.nc`` (its
simplify_network output); master's own ``elec_base_network_l_pp.pkl`` is nodal
and has no develop counterpart.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

RUN = "equivalence"
INTERCONNECT = os.environ.get("EQ_INTERCONNECT", "western")
UNTIL = os.environ.get("EQ_UNTIL", "")  # 'assembled' = stop pairs at the assembled stage
# The baseline branch, resolved to a sha at run time (never pinned here).
BASELINE_REF = os.environ.get("EQ_BASELINE_REF", "master-benchmark")
_CONFIG_NAME = "config.equivalence.yaml" if INTERCONNECT == "western" else f"config.equivalence-{INTERCONNECT}.yaml"
# The develop side reads the tracked template directly (its Snakefile no longer
# needs a config/ copy). The baseline's Snakefile still expects everything
# under config/, and build.py copies the shared harness config in there.
CONFIGFILE = f"repo_data/config/{_CONFIG_NAME}"
BASELINE_CONFIGFILE = f"config/{_CONFIG_NAME}"
CLUSTERS = os.environ.get("EQ_CLUSTERS", "4")  # reeds transport: must equal the footprint's ReEDS zone count (western/CA slice 4; usa 134)
LL = "v1.0"


def baseline_clusters(wc: str = CLUSTERS) -> str:
    """Translate a develop ``{clusters}`` value into the baseline's dialect.

    Harness-branch commit ``8371e9d`` (2026-09-06) changed ``{clusters}``
    semantics on develop: a plain integer now means "aggregate only
    conventional carriers" (master's ``m`` suffix) and a trailing ``s`` means
    "aggregate every carrier" (master's plain integer). ``master`` does not
    carry that commit, so without this translation the two sides would build
    *different generator sets* and the comparison would be meaningless.

    Hot-fix-ledger open item 5 asks whether ``8371e9d`` belongs on develop with
    its own HF row — a separate decision. The harness needs the translation
    either way, and ``run_meta.json`` records the literal value each side ran.
    """
    if wc == "all":
        return wc
    if wc.endswith("s"):
        return wc[:-1]
    if wc[-1].isdigit():
        return wc + "m"
    return wc


BASELINE_CLUSTERS = baseline_clusters()
OPTS = os.environ.get("EQ_OPTS", "REM-3h")
SIMPL2 = os.environ.get("EQ_SIMPL", "20")  # prong-2 simpl granularity (prong 1 is always pass-through '')
SECTOR = "E"
HORIZON = "2030"  # godeeep planning-horizon subdir for profiles (future scenarios only)


def _profile_horizon_dir() -> str:
    """Horizon path segment for profile artifacts, from the shared config.

    Both branches emit profiles under a ``{planning_horizon}/`` subdir only
    for GODEEEP *future* scenarios (``godeeep_planning_horizon`` in each
    side's build_electricity.smk); historical runs emit flat paths. The
    scenario is pinned in the shared harness config, so read it from there
    rather than duplicating the choice here.
    """
    import yaml

    cfg_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "workflow", CONFIGFILE
    )
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    scenarios = cfg.get("renewable_scenarios") or ["rcp85cooler"]
    return "" if scenarios[0] == "historical" else f"{HORIZON}/"

EQ = f"resources/{RUN}"
RES = f"results/{RUN}"


@dataclass(frozen=True)
class ArtifactPair:
    """One comparable artifact across the two sides."""

    stage: str  # short stage label used in findings/report
    develop: str  # path relative to the develop checkout's workflow/
    master: str  # path relative to the baseline worktree's workflow/
    kind: str  # loader: network | network_pkl_vs_nc | profile | demand_total
    solve_stage: bool = False  # apply D7 tolerances instead of D2


def prong_pairs(prong: int) -> list[ArtifactPair]:
    """Comparable artifacts for prong 1 (simpl='') or prong 2 (simpl=SIMPL2)."""
    s = "" if prong == 1 else SIMPL2
    ic = INTERCONNECT
    hdir = _profile_horizon_dir()  # "" for historical, f"{HORIZON}/" for future scenarios
    pairs = [
        # NOTE: these two CSVs are keyed at different granularities (master is
        # NODAL, pre-aggregation raw bus ids; develop is substation-keyed),
        # so only the clustering-invariant system total is compared. Per-bus
        # demand equivalence is covered by the assembled substation network's
        # Load_t.p_set comparison.
        ArtifactPair(
            stage="demand",
            develop=f"{EQ}/demand/{ic}/power_electricity_s{s}.csv",
            master=f"{EQ}/{ic}/demand/power_electricity.csv",
            kind="demand_total",
        ),
        ArtifactPair(
            stage="profile_onwind",
            develop=f"{EQ}/profiles/{ic}/{hdir}profile_onwind_s{s}.nc",
            master=f"{EQ}/{ic}/{hdir}profile_onwind.nc",
            kind="profile",
        ),
        ArtifactPair(
            stage="profile_solar",
            develop=f"{EQ}/profiles/{ic}/{hdir}profile_solar_s{s}.nc",
            master=f"{EQ}/{ic}/{hdir}profile_solar.nc",
            kind="profile",
        ),
    ]
    if prong == 1:
        # Substation-granularity assembled network exists on both sides only
        # under pass-through simpl. (Prong 2's pre-cluster networks differ by
        # design — different simpl-stage kmeans — so they are not compared.)
        pairs.append(
            ArtifactPair(
                stage="assembled_substation_network",
                develop=f"{EQ}/networks/{ic}/elec_s_l_pp.pkl",
                master=f"{EQ}/{ic}/elec_s.nc",
                kind="network_pkl_vs_nc",
            ),
        )
    if UNTIL == "assembled":
        return pairs
    core = f"elec_s{s}_c{CLUSTERS}"
    prepared = f"{core}_ec_l{LL}_{OPTS}"
    mcore = f"elec_s{s}_c{BASELINE_CLUSTERS}"
    mprepared = f"{mcore}_ec_l{LL}_{OPTS}"
    pairs += [
        ArtifactPair(
            stage="clustered_network",
            develop=f"{EQ}/networks/{ic}/{core}.nc",
            master=f"{EQ}/{ic}/{mcore}.nc",
            kind="network",
        ),
        ArtifactPair(
            stage="extra_components",
            develop=f"{EQ}/networks/{ic}/{core}_ec.nc",
            master=f"{EQ}/{ic}/{mcore}_ec.nc",
            kind="network",
        ),
        ArtifactPair(
            stage="prepared_network",
            develop=f"{EQ}/networks/{ic}/{prepared}.nc",
            master=f"{EQ}/{ic}/{mprepared}.nc",
            kind="network",
        ),
        ArtifactPair(
            stage="sectored_network",
            develop=f"{EQ}/networks/{ic}/{prepared}_{SECTOR}.nc",
            master=f"{EQ}/{ic}/{mprepared}_{SECTOR}.nc",
            kind="network",
        ),
        ArtifactPair(
            stage="solved_network",
            develop=f"{RES}/{ic}/networks/{prepared}_{SECTOR}.nc",
            master=f"{RES}/{ic}/networks/{mprepared}_{SECTOR}.nc",
            kind="network",
            solve_stage=True,
        ),
    ]
    return pairs


def final_target(prong: int, solve: bool = True) -> str:
    """The snakemake target that forces the whole prong's chain (develop side)."""
    s = "" if prong == 1 else SIMPL2
    prepared = f"elec_s{s}_c{CLUSTERS}_ec_l{LL}_{OPTS}_{SECTOR}"
    if solve:
        return f"{RES}/{INTERCONNECT}/networks/{prepared}.nc"
    return f"{EQ}/networks/{INTERCONNECT}/{prepared}.nc"


def baseline_final_target(prong: int, solve: bool = True) -> str:
    """The same target in the baseline's flat layout and {clusters} dialect."""
    s = "" if prong == 1 else SIMPL2
    prepared = f"elec_s{s}_c{BASELINE_CLUSTERS}_ec_l{LL}_{OPTS}_{SECTOR}"
    if solve:
        return f"{RES}/{INTERCONNECT}/networks/{prepared}.nc"
    return f"{EQ}/{INTERCONNECT}/{prepared}.nc"


def assembled_target(prong: int = 1) -> str:
    """Develop assembled-stage target (add_electricity output)."""
    s = "" if prong == 1 else SIMPL2
    return f"{EQ}/networks/{INTERCONNECT}/elec_s{s}_l_pp.pkl"


def baseline_assembled_target(prong: int = 1) -> str:
    """Baseline assembled-stage target (its simplify_network output)."""
    s = "" if prong == 1 else SIMPL2
    return f"{EQ}/{INTERCONNECT}/elec_s{s}.nc"
