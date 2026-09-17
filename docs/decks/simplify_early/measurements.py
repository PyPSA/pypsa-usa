"""Measured quantities behind the "simplify-early" deck.

Every number in this module was read off an artifact on Sherlock on
2026-09-17. Nothing here is estimated, interpolated or invented: each record
carries the file or command it came from. If a quantity was never measured it
is absent from this module and appears on the slide as "not measured".

Provenance vocabulary
---------------------
``bench``   a snakemake ``benchmark:`` TSV, one row per rule execution. Columns
            are ``s`` (wall seconds) and ``max_rss`` (MiB), among others.
``fs``      a file size in bytes from the filesystem.
``busmap``  a row / unique-value count in a busmap CSV emitted by the workflow.
``sacct``   Slurm accounting for a finished job.
``slurm``   a line in the job's stdout log.

Repository roots
----------------
DEVELOP_ROOT  the main checkout, branch ``feat/equivalence-usa-benchmark``
MASTER_ROOT   the ``master-benchmark`` worktree, the baseline that actually runs

Both are read-only from this module's point of view; the deck never rebuilds
them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Roots (recorded so a reader can re-check any number by hand)
# --------------------------------------------------------------------------

DEVELOP_ROOT = "code/pypsa-usa"
MASTER_ROOT = "code/pypsa-usa/.worktrees/master-benchmark"

DEVELOP_BENCH = f"{DEVELOP_ROOT}/workflow/benchmarks"
MASTER_BENCH = f"{MASTER_ROOT}/workflow/benchmarks"

DEVELOP_COMMIT = "d773abcb1a16c6c6672f1fce6d9d26e46848b4e5"
MASTER_COMMIT = "ca73014f62077b53e02ba027516f91400fe9cf61"

CASE = "usa / config.equivalence-usa.yaml / simpl=300 / clusters=134 / 3h / horizon 2030"
WESTERN_CASE = "western / config.equivalence.yaml / reeds_state=[CA] / simpl=20 / 3h"

MACHINE = "Sherlock, partition serc, 16 cores (USA) / 8 cores (western smoke)"

# The benchmark TSVs were not all written by the same Slurm job: the harness
# reuses finished rules across jobs via snakemake's mtime rerun triggers. Each
# record below therefore carries the mtime of the TSV, and the job whose window
# contains that mtime. This is the single most important caveat in the deck.
JOB_WINDOWS = [
    # (job id, name, start, end, elapsed, state, max_rss_kib, cores)
    ("43605139", "eq-usa", "2026-09-15T14:53:13", "2026-09-15T15:59:50", "01:06:37", "CANCELLED", None, 16),
    ("43616413", "eq-usa", "2026-09-15T16:00:31", "2026-09-15T17:29:26", "01:28:55", "FAILED", None, 16),
    ("43640844", "eq-usa", "2026-09-15T18:10:01", "2026-09-15T21:12:15", "03:02:14", "FAILED", 33_161_644, 16),
    ("43671762", "eq-smoke-western", "2026-09-15T21:31:35", "2026-09-15T21:39:24", "00:07:49", "COMPLETED", 1_820_492, 8),
    ("43671766", "eq-usa", "2026-09-15T21:43:04", "2026-09-15T23:22:57", "01:39:53", "FAILED", 34_994_120, 16),
]
JOB_WINDOWS_SOURCE = "sacct -j 43605139,43616413,43640844,43671762,43671766 --format=JobID,JobName,Start,End,Elapsed,State,MaxRSS"

# NOTE on `State`: FAILED on the eq-usa jobs is the *comparison verdict* of the
# equivalence harness (prong 2 reported UNEXPLAINED rows), not a crash of the
# build. The networks were built on both sides; see experiments/runs/
# eq-usa-80abb3bc125f/CARD.md.


# --------------------------------------------------------------------------
# Per-rule benchmarks
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RuleBench:
    """One rule, measured on one or both branches.

    ``master_*`` / ``develop_*`` are ``None`` where that branch has no
    ``benchmark:`` directive for the rule, i.e. the quantity was never
    measured. Such rules are still listed, so the gap is visible rather than
    silently dropped.
    """

    stage: str  # human label used on slides
    master_file: str | None
    develop_file: str | None
    master_s: float | None
    master_rss_mib: float | None
    develop_s: float | None
    develop_rss_mib: float | None
    master_mtime: str | None
    develop_mtime: str | None
    heavy: bool = False  # a per-bus "heavy" rule, the target of the refactor
    note: str = ""

    @property
    def measured_both(self) -> bool:
        return self.master_s is not None and self.develop_s is not None


# USA case. master files live under MASTER_BENCH, develop files under
# DEVELOP_BENCH; the paths below are relative to those two roots.
USA_RULES: list[RuleBench] = [
    RuleBench(
        stage="build_fuel_prices",
        master_file="equivalence/usa/build_fuel_prices",
        develop_file="equivalence/usa/build_fuel_prices",
        master_s=89.5549,
        master_rss_mib=1327.43,
        develop_s=36.7162,
        develop_rss_mib=2017.84,
        master_mtime="2026-09-15 15:01:34",
        develop_mtime="2026-09-15 16:04:57",
        heavy=False,
        note="not a per-bus rule; included as a control that is unaffected by the reordering",
    ),
    RuleBench(
        stage="build_renewable\nprofiles (onwind)",
        master_file="equivalence/usa/build_renewable_profiles_onwind",
        develop_file="equivalence/usa/build_renewable_profiles_onwind_s300",
        master_s=69.0955,
        master_rss_mib=5013.57,
        develop_s=107.7344,
        develop_rss_mib=3033.98,
        master_mtime="2026-09-15 15:34:40",
        develop_mtime="2026-09-15 16:18:49",
        heavy=True,
        note="develop still reads the NODAL regions for land availability (input regions_nodal) "
        "before carrying cells to the cluster through busmap_s300, so wall time does not drop; memory does",
    ),
    RuleBench(
        stage="build_renewable\nprofiles (solar)",
        master_file="equivalence/usa/build_renewable_profiles_solar",
        develop_file="equivalence/usa/build_renewable_profiles_solar_s300",
        master_s=242.9569,
        master_rss_mib=5459.79,
        develop_s=48.0828,
        develop_rss_mib=2966.12,
        master_mtime="2026-09-15 15:33:31",
        develop_mtime="2026-09-15 16:19:37",
        heavy=True,
    ),
    RuleBench(
        stage="build_electrical\ndemand",
        master_file="equivalence/usa/power_build_demand",
        develop_file="equivalence/usa/power_build_demand_s300",
        master_s=619.6149,
        master_rss_mib=10604.30,
        develop_s=86.4283,
        develop_rss_mib=11076.70,
        master_mtime="2026-09-15 15:12:51",
        develop_mtime="2026-09-15 16:17:02",
        heavy=True,
        note="time falls 7.2x; peak RSS is flat because the rule still reads the same zonal source data",
    ),
    RuleBench(
        stage="add_demand",
        master_file="equivalence/usa/add_demand",
        develop_file="equivalence/usa/elec_s300_add_demand",
        master_s=996.2917,
        master_rss_mib=14785.31,
        develop_s=10.4258,
        develop_rss_mib=348.05,
        master_mtime="2026-09-15 15:29:28",
        develop_mtime="2026-09-15 16:19:48",
        heavy=True,
    ),
    RuleBench(
        stage="add_electricity",
        master_file="equivalence/usa/add_electricity",
        develop_file="equivalence/usa/elec_s300_add_electricity",
        master_s=2135.2832,
        master_rss_mib=8357.12,
        develop_s=103.4288,
        develop_rss_mib=2372.67,
        master_mtime="2026-09-15 18:51:58",
        develop_mtime="2026-09-15 21:50:21",
        heavy=True,
    ),
    RuleBench(
        stage="cluster_network",
        master_file="cluster_network/usa/elec_s300_c134m",
        develop_file="cluster_network/usa/elec_s300_c134",
        master_s=118.9298,
        master_rss_mib=671.02,
        develop_s=231.2703,
        develop_rss_mib=2171.80,
        master_mtime="2026-09-15 19:35:52",
        develop_mtime="2026-09-15 21:54:12",
        heavy=False,
        note="master's wildcard is c134m, develop's is c134; per workflow docs a plain integer and "
        "'m' select the same aggregation, so the two are comparable. On develop this rule "
        "consumes the pickled add_electricity output, so it now carries the attachment work",
    ),
    RuleBench(
        stage="solve_network",
        master_file="equivalence/solve_network/usa/elec_s300_c134m_ec_lv1.0_3h_E",
        develop_file="equivalence/solve_network/usa/elec_s300_c134_ec_lv1.0_3h_E",
        master_s=3436.5054,
        master_rss_mib=17535.61,
        develop_s=3157.4397,
        develop_rss_mib=25973.46,
        master_mtime="2026-09-15 20:34:34",
        develop_mtime="2026-09-15 22:53:43",
        heavy=False,
        note="downstream of the reordering; both solve a 134-cluster 3h network. The two sides do "
        "not yet agree on the solution (see the equivalence run card), so this pair compares cost "
        "of solving two slightly different problems, not a like-for-like speedup",
    ),
]

# Rules that exist on develop only, or that only develop instruments. These are
# the *cost* of the reordering and must be shown next to the savings.
USA_DEVELOP_ONLY: list[RuleBench] = [
    RuleBench(
        stage="aggregate_to_substations",
        master_file=None,
        develop_file="equivalence/usa/aggregate_to_substations",
        master_s=None,
        master_rss_mib=None,
        develop_s=190.5641,
        develop_rss_mib=29449.73,
        master_mtime=None,
        develop_mtime="2026-09-15 16:09:39",
        note="new rule: the topology half of master's simplify_network, moved to the front",
    ),
    RuleBench(
        stage="cluster_resources",
        master_file=None,
        develop_file="equivalence/usa/cluster_resources_elec_s300",
        master_s=None,
        master_rss_mib=None,
        develop_s=355.3665,
        develop_rss_mib=13961.55,
        master_mtime=None,
        develop_mtime="2026-09-15 16:15:35",
        note="new rule: the {simpl} kmeans half of master's simplify_network, moved to the front",
    ),
]

# The counterpart on master. It has NO `benchmark:` directive on the
# master-benchmark branch, so no TSV exists and the quantity is unmeasured.
MASTER_UNMEASURED = {
    "rule": "simplify_network",
    "why": (
        "master's simplify_network does both jobs that develop splits into "
        "aggregate_to_substations and cluster_resources, but it carries no "
        "`benchmark:` directive on master-benchmark, so no per-rule wall time "
        "or peak RSS was recorded for it."
    ),
    "consequence": (
        "master's side of the reduction step is missing from the per-rule "
        "charts. Every comparison that includes develop's two new rules is "
        "therefore conservative: adding master's unmeasured cost would only "
        "move the result further in develop's favour."
    ),
}

# Rules instrumented on develop but not on master (build_base_network,
# build_bus_regions, build_shapes, add_extra_components, prepare_network,
# add_sectors). Recorded for completeness; not charted, because a one-sided
# bar invites a comparison that cannot be made.
USA_DEVELOP_ONLY_INSTRUMENTED = {
    "build_shapes": (51.5649, 460.67),
    "build_base_network": (77.0148, 717.77),
    "build_bus_regions": (74.4242, 928.05),
    "add_extra_components (c134_ec)": (178.0367, 3511.53),
    "prepare_network (c134_ec_lv1.0_3h)": (178.2384, 3693.61),
    "add_sectors (c134_ec_lv1.0_3h_E)": (57.0725, 1621.25),
}

# --------------------------------------------------------------------------
# Western smoke case (California only) - the counter-example
# --------------------------------------------------------------------------

WESTERN_RULES: list[RuleBench] = [
    RuleBench(
        stage="build_renewable\nprofiles (onwind)",
        master_file="equivalence/western/build_renewable_profiles_onwind_2030",
        develop_file="equivalence/western/build_renewable_profiles_onwind_2030_s20",
        master_s=23.4855,
        master_rss_mib=2957.97,
        develop_s=21.5045,
        develop_rss_mib=2981.86,
        master_mtime="2026-09-15 11:43:00",
        develop_mtime="2026-09-15 14:28:55",
        heavy=True,
    ),
    RuleBench(
        stage="build_renewable\nprofiles (solar)",
        master_file="equivalence/western/build_renewable_profiles_solar_2030",
        develop_file="equivalence/western/build_renewable_profiles_solar_2030_s20",
        master_s=220.5375,
        master_rss_mib=2328.17,
        develop_s=19.7585,
        develop_rss_mib=3097.70,
        master_mtime="2026-09-15 11:35:27",
        develop_mtime="2026-09-15 14:29:15",
        heavy=True,
    ),
    RuleBench(
        stage="build_electrical\ndemand",
        master_file="equivalence/western/power_build_demand",
        develop_file="equivalence/western/power_build_demand_s20",
        master_s=81.7739,
        master_rss_mib=6839.58,
        develop_s=80.6491,
        develop_rss_mib=8667.99,
        master_mtime="2026-09-15 11:31:47",
        develop_mtime="2026-09-15 11:26:36",
        heavy=True,
    ),
    RuleBench(
        stage="add_demand",
        master_file="equivalence/western/add_demand",
        develop_file="equivalence/western/elec_s20_add_demand",
        master_s=3.1629,
        master_rss_mib=259.29,
        develop_s=3.0016,
        develop_rss_mib=270.11,
        master_mtime="2026-09-15 11:43:03",
        develop_mtime="2026-09-15 11:26:39",
        heavy=True,
    ),
    RuleBench(
        stage="add_electricity",
        master_file="equivalence/western/add_electricity",
        develop_file="equivalence/western/elec_s20_add_electricity",
        master_s=10.7408,
        master_rss_mib=810.70,
        develop_s=32.6510,
        develop_rss_mib=265.70,
        master_mtime="2026-09-15 11:43:14",
        develop_mtime="2026-09-15 21:38:58",
        heavy=True,
    ),
]

# --------------------------------------------------------------------------
# Resolution ladder - how many buses each stage sees
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Resolution:
    label: str
    buses: int
    source: str


# Both branches share build_base_network, so the nodal count is common.
USA_LADDER = [
    Resolution(
        "nodal\n(elec_base_network)",
        82_549,
        "rows in resources/equivalence/busmaps/usa/bus2sub.csv (develop) - one row per base-network bus",
    ),
    Resolution(
        "substations\n(elec_b)",
        41_012,
        "unique sub_id in resources/equivalence/busmaps/usa/busmap_b.csv",
    ),
    Resolution(
        "{simpl} clusters\n(elec_s300)",
        300,
        "unique cluster_bus in resources/equivalence/busmaps/usa/busmap_s300.csv",
    ),
    Resolution(
        "final clusters\n(elec_s300_c134)",
        134,
        "unique target in resources/equivalence/busmaps/usa/busmap_s300_134.csv",
    ),
]

WESTERN_LADDER = [
    Resolution("nodal", 4_249, "rows in busmaps/western/bus2sub.csv"),
    Resolution("substations", 1_976, "unique sub_id in busmaps/western/busmap_b.csv"),
    Resolution("{simpl} clusters", 20, "unique cluster_bus in busmaps/western/busmap_s20.csv"),
]

# Resolution each branch's heavy per-bus rules actually run at.
HEAVY_RULE_RESOLUTION = {
    "master": 82_549,  # regions_onshore.geojson / elec_base_network.nc, nodal
    "develop": 300,  # regions_onshore_s300.geojson / elec_s300.nc
}
REDUCTION_FACTOR = HEAVY_RULE_RESOLUTION["master"] / HEAVY_RULE_RESOLUTION["develop"]

# --------------------------------------------------------------------------
# Artifact sizes - the arrays the heavy rules write
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Artifact:
    label: str
    produced_by: str
    master_path: str
    develop_path: str
    master_bytes: int
    develop_bytes: int
    note: str = ""


USA_ARTIFACTS = [
    Artifact(
        "onwind profile",
        "build_renewable_profiles",
        f"{MASTER_ROOT}/workflow/resources/equivalence/usa/profile_onwind.nc",
        f"{DEVELOP_ROOT}/workflow/resources/equivalence/profiles/usa/profile_onwind_s300.nc",
        622_450_598,
        10_330_358,
    ),
    Artifact(
        "solar profile",
        "build_renewable_profiles",
        f"{MASTER_ROOT}/workflow/resources/equivalence/usa/profile_solar.nc",
        f"{DEVELOP_ROOT}/workflow/resources/equivalence/profiles/usa/profile_solar_s300.nc",
        746_175_398,
        10_575_778,
    ),
    Artifact(
        "electrical demand\n(CSV)",
        "build_electrical_demand",
        f"{MASTER_ROOT}/workflow/resources/equivalence/usa/demand/power_electricity.csv",
        f"{DEVELOP_ROOT}/workflow/resources/equivalence/demand/usa/power_electricity_s300.csv",
        2_424_183_225,
        24_843_782,
        note="develop additionally writes power_zonal_components_s300.parquet (186,302,484 B) "
        "which master has no counterpart for; it is excluded from this bar",
    ),
    Artifact(
        "network after\nadd_demand",
        "add_demand",
        f"{MASTER_ROOT}/workflow/resources/equivalence/usa/elec_base_network_dem.nc",
        f"{DEVELOP_ROOT}/workflow/resources/equivalence/networks/usa/elec_s300_dem.nc",
        2_758_570_331,
        17_262_481,
    ),
    Artifact(
        "network after\nadd_electricity",
        "add_electricity",
        f"{MASTER_ROOT}/workflow/resources/equivalence/usa/elec_base_network_l_pp.pkl",
        f"{DEVELOP_ROOT}/workflow/resources/equivalence/networks/usa/elec_s300_l_pp.pkl",
        6_498_564_020,
        1_128_509_306,
        note="smallest ratio of the set: this pickle carries one entry per attached generator, and "
        "the generator count does not fall with the bus count as steeply as the time series do",
    ),
]
ARTIFACT_SOURCE = "ls -la on the 2026-09-15 run artifacts, both worktrees, read 2026-09-17"

# --------------------------------------------------------------------------
# Whole-job numbers
# --------------------------------------------------------------------------

BUILD_TIMES = {
    # (job, side, seconds, whether the side started from a cold cache)
    ("43640844", "master"): (8297, False, "master build OK in 8297s - slurm-eq-43640844.out"),
    ("43640844", "develop"): (4, True, "develop build OK in 4s - fully cached, not a real build"),
    ("43671766", "develop"): (3910, False, "develop build OK in 3910s - slurm-eq-43671766.out"),
    ("43671766", "master"): (117, True, "master build OK in 117s - reused the finished side"),
    ("43671762", "develop"): (37, True, "western smoke, cached"),
    ("43671762", "master"): (3, True, "western smoke, cached"),
}

TOTAL_PIPELINE_CAVEAT = (
    "No single Slurm job built both branches from a cold cache. The harness "
    "reuses finished rules across jobs through snakemake's mtime rerun "
    "triggers, so the two 'build OK in Ns' lines in any one job are not a "
    "like-for-like total. End-to-end cold total pipeline wall time, master vs "
    "develop, is NOT MEASURED. The per-rule benchmark TSVs are per-rule cold "
    "measurements and are what the deck compares."
)

# --------------------------------------------------------------------------
# Equivalence harness
# --------------------------------------------------------------------------

HARNESS = {
    "location": "code/pypsa-usa/tests/equivalence/",
    "docs": "code/pypsa-usa/docs/equivalence.md",
    "baseline": "master-benchmark branch (a sha, not a patched checkout)",
    "test_files": 13,  # test_*.py under tests/equivalence/
    "modules": [
        "build.py - provisions and builds both sides",
        "metrics.py - the compared quantities",
        "compare.py / tables.py - the verdict table",
        "plots.py - the PNG + CSV figure set",
        "hotfixes.yaml - machine-readable ledger of documented differences",
        "waivers.yaml - bounded, per-component waivers",
    ],
    "verdicts": ["equivalent", "explained (a ledger hot-fix)", "UNEXPLAINED (a failure)"],
    "compared": [
        "capacity factor profiles (p_max_pu)",
        "resource availability (p_nom_max)",
        "existing capacity (p_nom) by zone and carrier",
        "demand by zone",
        "solved objective, capacity and dispatch",
    ],
}


def heavy_totals(rules: list[RuleBench]) -> dict[str, float]:
    """Sum wall time over rules measured on BOTH branches and flagged heavy."""
    pairs = [r for r in rules if r.heavy and r.measured_both]
    return {
        "master_s": sum(r.master_s for r in pairs),
        "develop_s": sum(r.develop_s for r in pairs),
        "master_peak_rss": max(r.master_rss_mib for r in pairs),
        "develop_peak_rss": max(r.develop_rss_mib for r in pairs),
        "n_rules": len(pairs),
    }
