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
CONFIGFILE = (
    "config/config.equivalence.yaml" if INTERCONNECT == "western" else f"config/config.equivalence-{INTERCONNECT}.yaml"
)
CLUSTERS = "4"
LL = "v1.0"
OPTS = "REM-3h"
SECTOR = "E"
HORIZON = "2030"  # godeeep planning-horizon subdir for profiles

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
    """Comparable artifacts for prong 1 (simpl='') or prong 2 (simpl=20)."""
    s = "" if prong == 1 else "20"
    ic = INTERCONNECT
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
            candidate=f"{EQ}/profiles/{ic}/{HORIZON}/profile_onwind_s{s}.nc",
            anchor=f"{EQ}/{ic}/{HORIZON}/profile_onwind.nc",
            kind="profile",
        ),
        ArtifactPair(
            stage="profile_solar",
            candidate=f"{EQ}/profiles/{ic}/{HORIZON}/profile_solar_s{s}.nc",
            anchor=f"{EQ}/{ic}/{HORIZON}/profile_solar.nc",
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
    pairs += [
        ArtifactPair(
            stage="clustered_network",
            candidate=f"{EQ}/networks/{ic}/{core}.nc",
            anchor=f"{EQ}/{ic}/{core}.nc",
            kind="network",
        ),
        ArtifactPair(
            stage="extra_components",
            candidate=f"{EQ}/networks/{ic}/{core}_ec.nc",
            anchor=f"{EQ}/{ic}/{core}_ec.nc",
            kind="network",
        ),
        ArtifactPair(
            stage="prepared_network",
            candidate=f"{EQ}/networks/{ic}/{prepared}.nc",
            anchor=f"{EQ}/{ic}/{prepared}.nc",
            kind="network",
        ),
        ArtifactPair(
            stage="sectored_network",
            candidate=f"{EQ}/networks/{ic}/{prepared}_{SECTOR}.nc",
            anchor=f"{EQ}/{ic}/{prepared}_{SECTOR}.nc",
            kind="network",
        ),
        ArtifactPair(
            stage="solved_network",
            candidate=f"{RES}/{ic}/networks/{prepared}_{SECTOR}.nc",
            anchor=f"{RES}/{ic}/networks/{prepared}_{SECTOR}.nc",
            kind="network",
            solve_stage=True,
        ),
    ]
    return pairs


def final_target(prong: int, solve: bool = True) -> str:
    """The snakemake target that forces the whole prong's chain."""
    s = "" if prong == 1 else "20"
    prepared = f"elec_s{s}_c{CLUSTERS}_ec_l{LL}_{OPTS}_{SECTOR}"
    if solve:
        return f"{RES}/{INTERCONNECT}/networks/{prepared}.nc"
    return f"{EQ}/networks/{INTERCONNECT}/{prepared}.nc"


def anchor_final_target(prong: int, solve: bool = True) -> str:
    s = "" if prong == 1 else "20"
    prepared = f"elec_s{s}_c{CLUSTERS}_ec_l{LL}_{OPTS}_{SECTOR}"
    if solve:
        return f"{RES}/{INTERCONNECT}/networks/{prepared}.nc"
    return f"{EQ}/{INTERCONNECT}/{prepared}.nc"


def assembled_target() -> str:
    """Candidate assembled-stage target (prong 1)."""
    return f"{EQ}/networks/{INTERCONNECT}/elec_s_l_pp.pkl"


def anchor_assembled_target() -> str:
    """Anchor assembled-stage target (its simplify output, prong 1)."""
    return f"{EQ}/{INTERCONNECT}/elec_s.nc"
