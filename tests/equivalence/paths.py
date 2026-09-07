"""Artifact path map for the Tier C equivalence harness.

Pairs candidate (v1-epic, category-first resources layout) artifacts with
anchor (upstream/develop e7f8bd70, flat {interconnect}/ layout) artifacts for
one prong of the two-prong protocol. Paths are relative to each side's
``workflow/`` directory; the run name is fixed to ``equivalence``.

Pairing facts come from the 2026-08-07 research workflow (see
docs/superpowers/plans/2026-08-07-ca-equivalence-harness.md): the DAGs pass
through the same logical states under different file names. Notably the
candidate's assembled substation network is ``elec_s{simpl}_l_pp.pkl`` (dill)
while the anchor's is ``elec_s{simpl}.nc`` (its simplify_network output);
the anchor's own ``elec_base_network_l_pp.pkl`` is nodal and has no
candidate counterpart.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

RUN = "equivalence"
INTERCONNECT = os.environ.get("EQ_INTERCONNECT", "western")
UNTIL = os.environ.get("EQ_UNTIL", "")  # 'assembled' = stop pairs at the assembled stage
_CONFIG_NAME = "config.equivalence.yaml" if INTERCONNECT == "western" else f"config.equivalence-{INTERCONNECT}.yaml"
# Candidate side reads the tracked template directly (its Snakefile no longer
# needs a config/ copy). The anchor is a pinned upstream checkout whose
# Snakefile still expects everything under config/, and build.py copies the
# shared harness config in there.
CONFIGFILE = f"repo_data/config/{_CONFIG_NAME}"
ANCHOR_CONFIGFILE = f"config/{_CONFIG_NAME}"
CLUSTERS = os.environ.get("EQ_CLUSTERS", "4")  # reeds transport: must equal the footprint's ReEDS zone count (western/CA slice 4; usa 134)
LL = "v1.0"


def anchor_clusters(wc: str = CLUSTERS) -> str:
    """Translate a candidate ``{clusters}`` value into the anchor's dialect.

    Since 2026-09-06 the candidate's plain integer means "aggregate only
    conventional carriers" (the anchor's ``m``) and ``s`` means "aggregate
    every carrier" (the anchor's plain integer). The pinned anchor keeps the
    old semantics, so its targets and config use the translated value.
    """
    if wc == "all":
        return wc
    if wc.endswith("s"):
        return wc[:-1]
    if wc[-1].isdigit():
        return wc + "m"
    return wc


ANCHOR_CLUSTERS = anchor_clusters()
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
    candidate: str  # path relative to candidate workflow/
    anchor: str  # path relative to anchor worktree workflow/
    kind: str  # loader: network | network_pkl_vs_nc | profile | demand_total
    solve_stage: bool = False  # apply D7 tolerances instead of D2


def prong_pairs(prong: int) -> list[ArtifactPair]:
    """Comparable artifacts for prong 1 (simpl='') or prong 2 (simpl=SIMPL2)."""
    s = "" if prong == 1 else SIMPL2
    ic = INTERCONNECT
    hdir = _profile_horizon_dir()  # "" for historical, f"{HORIZON}/" for future scenarios
    pairs = [
        # NOTE: these two CSVs are keyed at different granularities (anchor is
        # NODAL, pre-aggregation raw bus ids; candidate is substation-keyed),
        # so only the clustering-invariant system total is compared. Per-bus
        # demand equivalence is covered by the assembled substation network's
        # Load_t.p_set comparison.
        ArtifactPair(
            stage="demand",
            candidate=f"{EQ}/demand/{ic}/power_electricity_s{s}.csv",
            anchor=f"{EQ}/{ic}/demand/power_electricity.csv",
            kind="demand_total",
        ),
        ArtifactPair(
            stage="profile_onwind",
            candidate=f"{EQ}/profiles/{ic}/{hdir}profile_onwind_s{s}.nc",
            anchor=f"{EQ}/{ic}/{hdir}profile_onwind.nc",
            kind="profile",
        ),
        ArtifactPair(
            stage="profile_solar",
            candidate=f"{EQ}/profiles/{ic}/{hdir}profile_solar_s{s}.nc",
            anchor=f"{EQ}/{ic}/{hdir}profile_solar.nc",
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
                candidate=f"{EQ}/networks/{ic}/elec_s_l_pp.pkl",
                anchor=f"{EQ}/{ic}/elec_s.nc",
                kind="network_pkl_vs_nc",
            ),
        )
    if UNTIL == "assembled":
        return pairs
    core = f"elec_s{s}_c{CLUSTERS}"
    prepared = f"{core}_ec_l{LL}_{OPTS}"
    acore = f"elec_s{s}_c{ANCHOR_CLUSTERS}"
    aprepared = f"{acore}_ec_l{LL}_{OPTS}"
    pairs += [
        ArtifactPair(
            stage="clustered_network",
            candidate=f"{EQ}/networks/{ic}/{core}.nc",
            anchor=f"{EQ}/{ic}/{acore}.nc",
            kind="network",
        ),
        ArtifactPair(
            stage="extra_components",
            candidate=f"{EQ}/networks/{ic}/{core}_ec.nc",
            anchor=f"{EQ}/{ic}/{acore}_ec.nc",
            kind="network",
        ),
        ArtifactPair(
            stage="prepared_network",
            candidate=f"{EQ}/networks/{ic}/{prepared}.nc",
            anchor=f"{EQ}/{ic}/{aprepared}.nc",
            kind="network",
        ),
        ArtifactPair(
            stage="sectored_network",
            candidate=f"{EQ}/networks/{ic}/{prepared}_{SECTOR}.nc",
            anchor=f"{EQ}/{ic}/{aprepared}_{SECTOR}.nc",
            kind="network",
        ),
        ArtifactPair(
            stage="solved_network",
            candidate=f"{RES}/{ic}/networks/{prepared}_{SECTOR}.nc",
            anchor=f"{RES}/{ic}/networks/{aprepared}_{SECTOR}.nc",
            kind="network",
            solve_stage=True,
        ),
    ]
    return pairs


def final_target(prong: int, solve: bool = True) -> str:
    """The snakemake target that forces the whole prong's chain."""
    s = "" if prong == 1 else SIMPL2
    prepared = f"elec_s{s}_c{CLUSTERS}_ec_l{LL}_{OPTS}_{SECTOR}"
    if solve:
        return f"{RES}/{INTERCONNECT}/networks/{prepared}.nc"
    return f"{EQ}/networks/{INTERCONNECT}/{prepared}.nc"


def anchor_final_target(prong: int, solve: bool = True) -> str:
    s = "" if prong == 1 else SIMPL2
    prepared = f"elec_s{s}_c{ANCHOR_CLUSTERS}_ec_l{LL}_{OPTS}_{SECTOR}"
    if solve:
        return f"{RES}/{INTERCONNECT}/networks/{prepared}.nc"
    return f"{EQ}/{INTERCONNECT}/{prepared}.nc"


def assembled_target(prong: int = 1) -> str:
    """Candidate assembled-stage target (add_electricity output)."""
    s = "" if prong == 1 else SIMPL2
    return f"{EQ}/networks/{INTERCONNECT}/elec_s{s}_l_pp.pkl"


def anchor_assembled_target(prong: int = 1) -> str:
    """Anchor assembled-stage target (its simplify_network output)."""
    s = "" if prong == 1 else SIMPL2
    return f"{EQ}/{INTERCONNECT}/elec_s{s}.nc"
