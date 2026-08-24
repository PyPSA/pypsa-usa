"""Build one side (candidate or anchor) of an equivalence run.

Candidate builds run in the main checkout. Anchor builds run in a dedicated
git worktree of the pinned anchor SHA, provisioned per the plan:

- symlink ``workflow/data`` and ``workflow/cutouts`` from the main checkout
  (both untracked/gitignored on both branches);
- seed the gitignored layered configs (``config.cluster/plotting/api/sector``)
  from the anchor's own ``workflow/repo_data/config/`` (they are
  ``configfile:``-loaded unconditionally) — the tracked ``config.common.yaml``
  comes from git;
- copy the shared ``config.equivalence.yaml`` in;
- apply the documented BUILD-INFRA patch to ``rules/common.smk`` (the
  upstream #764 ``constants`` source-cache import bug; same fix as v1-epic
  commit e43fa927). Infra-only: cannot affect numbers;
- apply the documented ADOPTED-FIX patches: ``build_bus_regions`` (DL-11:
  footprint-scoped empty-county sweep), ``build_powerplants`` (DL-12:
  pre-aggregate EIA-860 history before the LEFT JOINs) and ``add_electricity``
  (DL-13: bound the must-add seam-plant fallback to the model footprint).
  Results-affecting BY DESIGN and applied to BOTH sides by user decision so
  the harness keeps comparing like-for-like; see the deltas ledger;
- ``touch`` retrieve_caiso_data's output if present so a fresh-checkout mtime
  on its tracked input xlsx does not retrigger a re-download into shared
  ``data/``.

Instrumentation: after a build, ``write_manifest`` records git SHA, config
hash, per-rule benchmark rows (wall time, max_rss) and output file sizes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path

from .paths import CONFIGFILE

REPO = Path(__file__).resolve().parents[2]
ANCHOR_SHA = "e7f8bd70"
ANCHOR_WORKTREE = REPO / ".worktrees" / "anchor-e7f8bd70"
INFRA_PATCH_MARK = "materialize sibling modules"  # idempotence marker
ADOPTED_FIX_MARK = "restricting empty-county sweep"  # idempotence marker (DL-11)
# DL-12 sentinel: a CTE name that exists only in the candidate's
# build_powerplants.py query. Guards the file-adoption patch against silently
# copying a wrong/stale source file over the anchor's script.
POWERPLANTS_FIX_MARK = "ges_latest"
POWERPLANTS_SCRIPT = "workflow/scripts/build_powerplants.py"
# DL-13 sentinel: the seam-bound helper's name. Present in the candidate's
# add_electricity.py only after the seam fix, and never in the pristine anchor.
SEAM_FIX_MARK = "_drop_distant_seam_plants"
ADD_ELECTRICITY_SCRIPT = "workflow/scripts/add_electricity.py"
FORCE_RERUN_MARKER = ".eq-force-rerun"  # rules to -R once after a newly applied patch

LAYERED_CONFIGS = [
    "config.cluster.yaml",
    "config.plotting.yaml",
    "config.api.yaml",
    "config.sector.yaml",
]


def log(msg: str) -> None:
    print(f"[equivalence] {msg}", flush=True)


def run(cmd: list[str], cwd: Path, timeout: int = 7200) -> subprocess.CompletedProcess:
    log(f"$ {' '.join(cmd)}  (cwd={cwd})")
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def provision_anchor_worktree() -> Path:
    """Create/refresh the anchor worktree; idempotent."""
    wt = ANCHOR_WORKTREE
    if not wt.exists():
        wt.parent.mkdir(exist_ok=True)
        cp = run(
            ["git", "worktree", "add", "--detach", str(wt), ANCHOR_SHA],
            cwd=REPO,
        )
        if cp.returncode != 0:
            raise RuntimeError(f"git worktree add failed:\n{cp.stderr[-2000:]}")
    wf = wt / "workflow"

    # Shared data dirs via symlink (never copied — 13G).
    for name in ("data", "cutouts"):
        link = wf / name
        target = REPO / "workflow" / name
        if link.is_symlink():
            continue
        if link.exists():
            raise RuntimeError(f"{link} exists and is not a symlink; refusing")
        link.symlink_to(target)

    # Seed every gitignored config entry from the anchor's own templates
    # (mirrors tests/integration/conftest.py::_seed_runtime_configs and
    # init_pypsa_usa.sh): layered yaml files AND directories such as
    # config/policy_constraints/ that rules reference as inputs.
    src_root = wf / "repo_data" / "config"
    dst_root = wf / "config"
    for entry in src_root.iterdir():
        dst = dst_root / entry.name
        if dst.exists():
            continue
        if entry.is_dir():
            shutil.copytree(entry, dst)
        else:
            shutil.copy2(entry, dst)

    # Shared harness configs (kept in sync from the candidate repo_data copies).
    for cfg in (REPO / "workflow" / "repo_data" / "config").glob("config.equivalence*.yaml"):
        shutil.copy2(cfg, wf / "config" / cfg.name)

    apply_infra_patches(wt)
    apply_adopted_fix_patches(wt)

    # Fresh-checkout mtime on the tracked caiso xlsx must not retrigger a
    # re-download into the SHARED data/ tree.
    caiso_out = wf / "data" / "costs" / "caiso_ng_power_prices.csv"
    if caiso_out.exists():
        os.utime(caiso_out)

    return wt


def apply_infra_patches(wt: Path) -> None:
    """Documented build-infrastructure patches to the anchor worktree.

    Rule: a patch may make the anchor RUNNABLE but must not be able to change
    numbers. Every patch added here needs a matching entry in
    docs/CHANGELOG-v1-epic.md.
    """
    common = wt / "workflow" / "rules" / "common.smk"
    text = common.read_text()
    if INFRA_PATCH_MARK not in text:
        needle = 'path = workflow.source_path("../scripts/_helpers.py")\n'
        if needle not in text:
            raise RuntimeError("anchor common.smk shape unexpected; refusing to patch")
        patched = text.replace(
            needle,
            needle
            + "# EQUIVALENCE-HARNESS INFRA PATCH (documented in "
            + "docs/CHANGELOG-v1-epic.md):\n"
            + "# materialize sibling modules _helpers.py imports into the "
            + "source cache.\n"
            + 'workflow.source_path("../scripts/constants.py")\n',
        )
        common.write_text(patched)
        log("applied infra patch: common.smk constants source-cache fix")


def mark_force_rerun(wt: Path, rules: Iterable[str]) -> None:
    """Record rules to ``-R`` on the next build of this side (merge-safe).

    The harness runs snakemake with ``--rerun-triggers mtime``, under which
    code/rule changes NEVER invalidate existing outputs — a freshly patched
    worktree with pre-patch artifacts would silently keep them and the harness
    would compare fixed-vs-unfixed (2026-08-23 adversarial-review blocker).
    Deleting the rule's outputs is NOT enough either: when the final target is
    otherwise up-to-date, snakemake never revisits missing intermediates
    (observed 2026-08-23). ``build_side`` turns this marker into
    ``-R <rules>`` and clears it on success.

    Merges with any pending marker so a second patch in the same provision
    (or a patch applied while an earlier marker is still pending) cannot drop
    the earlier rules.
    """
    marker = wt / FORCE_RERUN_MARKER
    existing = marker.read_text().split() if marker.exists() else []
    merged = list(dict.fromkeys([*existing, *rules]))
    if not merged:
        return
    marker.write_text("\n".join(merged) + "\n")
    log(f"one-shot forced-rerun marker now: {merged}")


def snakemake_attrs(text: str) -> set[str]:
    """Names a script pulls off ``snakemake.input/params/output``."""
    return set(re.findall(r"snakemake\.(?:input|params|output)\.(\w+)", text))


def apply_adopted_fix_patches(wt: Path) -> None:
    """Adopted results-affecting fixes, mirrored onto the anchor by decision.

    Unlike ``apply_infra_patches`` these CAN change numbers — that is the
    point. Each entry is a fix the user countersigned into the ledger with an
    explicit decision to patch BOTH sides so the harness keeps comparing
    like-for-like instead of freezing a known-wrong anchor behavior.

    DL-11 (countersigned 2026-08-23): scope build_bus_regions' empty-county
    sweep to the model_topology.include footprint. Same change as v1-epic
    commit ccfe4b77; without it a CA-scoped run's regions cover ~7x the
    state and attach the whole interconnect fleet.

    DL-12 (2026-08-23): adopt the candidate's build_powerplants.py wholesale.
    v1-epic pre-aggregates EIA-860 history in DuckDB CTEs (``ges_latest`` /
    ``plants_latest`` / ``yg_latest``) BEFORE its LEFT JOINs; the anchor
    aggregates only after the join fan-out, so ~24 years of ``report_date``
    duplicate rows reweight the ``mean()`` that produces heat_rate /
    fuel_cost / efficiency. Left unpatched this is the dominant prong-1
    residual (solved objective rel 2.34%, CCGT/OCGT p_nom_opt split).

    DL-13 (countersigned 2026-08-23): bound ``add_electricity``'s "must add"
    seam-plant fallback to SEAM_PLANT_MAX_KM of the model footprint in
    footprint-scoped runs. Same change as v1-epic commit d98cb93f. Without it
    a CA-scoped run attaches 23 out-of-footprint plants / 1,887.4 MW (nearest
    890 km away) to California buses. Gated on ``model_topology.include``, so
    it is a no-op for unfiltered interconnect/usa runs on both sides.
    """
    applied_rules: list[str] = []
    smk = wt / "workflow" / "rules" / "build_electricity.smk"
    text = smk.read_text()
    if ADOPTED_FIX_MARK not in text:
        needle = (
            "rule build_bus_regions:\n"
            "    params:\n"
            "        topological_boundaries=config_provider(\n"
            '            "model_topology", "topological_boundaries"\n'
            "        ),\n"
        )
        if needle not in text:
            raise RuntimeError("anchor build_electricity.smk shape unexpected; refusing to patch")
        smk.write_text(
            text.replace(
                needle,
                needle
                + "        # EQUIVALENCE-HARNESS ADOPTED-FIX PATCH DL-11: "
                + "restricting empty-county sweep\n"
                + '        model_topology_include=config_provider("model_topology", "include", default=None),\n',
            ),
        )
        log("applied adopted-fix patch DL-11: build_electricity.smk include param")
        applied_rules.append("build_bus_regions")

    script = wt / "workflow" / "scripts" / "build_bus_regions.py"
    text = script.read_text()
    if ADOPTED_FIX_MARK not in text:
        needle_params = "    # Params\n    topological_boundaries = snakemake.params.topological_boundaries\n"
        needle_sweep = (
            "    # Identify empty counties WITHIN the interconnect's BA shapes total footprint "
            "(using reeds BA shapes for a cleaner shape)\n"
            "    combined_bus_regions = gpd_reeds.geometry.union_all()\n"
        )
        if needle_params not in text or needle_sweep not in text:
            raise RuntimeError("anchor build_bus_regions.py shape unexpected; refusing to patch")
        text = text.replace(
            needle_params,
            needle_params + "    include_filter = snakemake.params.model_topology_include\n",
        )
        text = text.replace(
            needle_sweep,
            "    # EQUIVALENCE-HARNESS ADOPTED-FIX PATCH DL-11 (same as v1-epic ccfe4b77):\n"
            "    # scope the empty-county sweep to the include-filtered footprint.\n"
            "    if include_filter:\n"
            "        gpd_reeds = gpd_reeds.loc[gpd_reeds.index.isin(n.buses.reeds_zone.unique())]\n"
            "        logger.info(\n"
            '            "model_topology.include set: restricting empty-county sweep to %d ReEDS zones present in the filtered network.",\n'
            "            len(gpd_reeds),\n"
            "        )\n"
            "    combined_bus_regions = gpd_reeds.geometry.union_all()\n",
        )
        script.write_text(text)
        log("applied adopted-fix patch DL-11: build_bus_regions.py footprint-scoped sweep")
        applied_rules.append("build_bus_regions")

    apply_powerplants_adoption(wt, applied_rules)
    apply_seam_adoption(wt, applied_rules)

    if applied_rules:
        mark_force_rerun(wt, applied_rules)


def apply_powerplants_adoption(wt: Path, applied_rules: list[str]) -> None:
    """DL-12: adopt the candidate's build_powerplants.py onto the anchor.

    Dynamic file adoption rather than textual surgery: the candidate file is
    read from the live REPO checkout so the patch self-maintains if v1-epic's
    query evolves, and any drift re-triggers the forced rerun. The two rule
    definitions are byte-identical apart from the output path, and the script
    is layout-agnostic (it only ever touches ``snakemake.output.powerplants``),
    so the anchor's flat ``resources/powerplants.csv`` layout is preserved.

    Safety rails, all of which raise rather than guess:
      * the candidate file must carry the DL-12 sentinel CTE;
      * every ``snakemake.input/params/output`` name the candidate file reads
        must also be read by the anchor's PRISTINE script (fetched from git,
        not from the possibly already-patched worktree), i.e. the candidate
        cannot demand a rule key the anchor's rule does not define.
    """
    cand_src = REPO / "workflow" / "scripts" / "build_powerplants.py"
    if not cand_src.exists():
        raise RuntimeError(f"candidate {POWERPLANTS_SCRIPT} missing; refusing to patch")
    cand_text = cand_src.read_text()
    if POWERPLANTS_FIX_MARK not in cand_text:
        raise RuntimeError(
            f"candidate {POWERPLANTS_SCRIPT} lacks the DL-12 sentinel "
            f"{POWERPLANTS_FIX_MARK!r}; refusing to adopt an unexpected file",
        )

    cp = run(["git", "show", f"{ANCHOR_SHA}:{POWERPLANTS_SCRIPT}"], cwd=wt)
    if cp.returncode != 0:
        raise RuntimeError(f"cannot read pristine anchor {POWERPLANTS_SCRIPT}:\n{cp.stderr[-2000:]}")
    orig_text = cp.stdout
    if POWERPLANTS_FIX_MARK in orig_text:
        raise RuntimeError(
            f"pristine anchor {POWERPLANTS_SCRIPT} already contains "
            f"{POWERPLANTS_FIX_MARK!r}; DL-12 premise is wrong — refusing to patch",
        )
    missing = snakemake_attrs(cand_text) - snakemake_attrs(orig_text)
    if missing:
        raise RuntimeError(
            f"candidate {POWERPLANTS_SCRIPT} reads snakemake keys the anchor's "
            f"build_powerplants rule does not provide: {sorted(missing)}; "
            "refusing wholesale adoption (do targeted query surgery instead)",
        )

    dst = wt / POWERPLANTS_SCRIPT
    if dst.read_text() == cand_text:
        return
    dst.write_text(cand_text)
    log("applied adopted-fix patch DL-12: build_powerplants.py adopted from candidate")
    applied_rules.append("build_powerplants")


def apply_seam_adoption(wt: Path, applied_rules: list[str]) -> None:
    """DL-13: mirror the seam-plant bound onto the anchor's add_electricity.py.

    Wholesale file adoption (the DL-12 mechanism) is NOT available here:
    v1-epic's ``add_electricity.py`` legitimately differs from the anchor's in
    the simplify-early bus2sub/sub_id removals, the ``length_factor=1.0``
    decision (DL-1/DL-2) and the schema-logging calls. Copying it over would
    smuggle those unrelated deltas onto the anchor. So this is targeted string
    surgery instead.

    What it introduces, matching the candidate's semantics exactly:
      * the ``SEAM_PLANT_MAX_KM`` module constant;
      * the ``_drop_distant_seam_plants`` helper;
      * a ``footprint_scoped: bool = False`` parameter on
        ``filter_plants_by_region``, applied right after the
        ``plants_must_add.set_index`` that closes the fallback's construction;
      * ``main()`` wiring that reads ``model_topology.include`` off
        ``snakemake.config`` and passes ``footprint_scoped=bool(include)``.

    Difference from the candidate: none in the code that runs. The whole
    ``filter_plants_by_region`` body is byte-identical between e7f8bd70 and
    v1-epic (verified 2026-08-24), so the anchor takes the same
    ``footprint_scoped`` parameter plumbing rather than the inlined-config
    variant that a divergent anchor shape would have forced. Only the comment
    banners differ, marking the lines as harness patches.

    The constant block and the helper body are sliced out of the LIVE candidate
    file rather than duplicated here, so the numeric logic the two sides run is
    the same text and any drift in v1-epic's helper re-triggers the forced
    rerun. The four wiring edits are hardcoded because they are the part that
    must adapt to the anchor's own shape.

    Safety rails, all of which raise rather than guess:
      * the candidate file must carry the DL-13 sentinel and yield both slices;
      * every needle is verified against the PRISTINE anchor file fetched from
        git, not the possibly already-patched worktree;
      * the pristine anchor must NOT already contain the sentinel, else the
        DL-13 premise is wrong;
      * the assembled result must carry the sentinel exactly three times and be
        wired end to end before anything is written.

    Idempotent by CONTENT (like DL-12), not by the sentinel, so that an edit to
    v1-epic's helper propagates here instead of being skipped as "already
    patched" — the anchor must never run a stale copy of the candidate's logic.
    """
    dst = wt / ADD_ELECTRICITY_SCRIPT
    cand_src = REPO / "workflow" / "scripts" / "add_electricity.py"
    if not cand_src.exists():
        raise RuntimeError(f"candidate {ADD_ELECTRICITY_SCRIPT} missing; refusing to patch")
    cand_text = cand_src.read_text()
    if SEAM_FIX_MARK not in cand_text:
        raise RuntimeError(
            f"candidate {ADD_ELECTRICITY_SCRIPT} lacks the DL-13 sentinel "
            f"{SEAM_FIX_MARK!r}; refusing to mirror an unexpected file",
        )

    # Slice the constant block and the helper out of the candidate.
    try:
        const_start = cand_text.index("# Maximum distance from the model footprint")
        const_end = cand_text.index("SEAM_PLANT_MAX_KM = 100.0") + len("SEAM_PLANT_MAX_KM = 100.0")
        helper_start = cand_text.index(f"def {SEAM_FIX_MARK}(")
        helper_end = cand_text.index("def filter_plants_by_region(")
    except ValueError as exc:
        raise RuntimeError(
            f"cannot slice the DL-13 constant/helper out of the candidate "
            f"{ADD_ELECTRICITY_SCRIPT}; its shape changed: {exc}",
        ) from None
    if not (const_start < const_end < helper_start < helper_end):
        raise RuntimeError(
            f"candidate {ADD_ELECTRICITY_SCRIPT} DL-13 slices are out of order; refusing to patch",
        )
    const_block = cand_text[const_start:const_end]
    helper_block = cand_text[helper_start:helper_end]

    cp = run(["git", "show", f"{ANCHOR_SHA}:{ADD_ELECTRICITY_SCRIPT}"], cwd=wt)
    if cp.returncode != 0:
        raise RuntimeError(f"cannot read pristine anchor {ADD_ELECTRICITY_SCRIPT}:\n{cp.stderr[-2000:]}")
    orig_text = cp.stdout
    if SEAM_FIX_MARK in orig_text:
        raise RuntimeError(
            f"pristine anchor {ADD_ELECTRICITY_SCRIPT} already contains "
            f"{SEAM_FIX_MARK!r}; DL-13 premise is wrong — refusing to patch",
        )

    banner = "    # EQUIVALENCE-HARNESS ADOPTED-FIX PATCH DL-13 (same as v1-epic d98cb93f):\n"
    needle_logger = "logger = logging.getLogger(__name__)\n"
    needle_signature = (
        "def filter_plants_by_region(\n"
        "    plants: pd.DataFrame,\n"
        "    regions_onshore: gpd.GeoDataFrame,\n"
        "    regions_offshore: gpd.GeoDataFrame,\n"
        "    reeds_shapes: gpd.GeoDataFrame,\n"
        "    all_reeds_shapes: gpd.GeoDataFrame,\n"
        "    reeds_memberships: pd.DataFrame,\n"
        ") -> pd.DataFrame:\n"
    )
    needle_set_index = '        plants_must_add.set_index("generator_name", inplace=True)\n'
    needle_main_call = (
        "    plants = filter_plants_by_region(\n"
        "        plants,\n"
        "        regions_onshore,\n"
        "        regions_offshore,\n"
        "        reeds_shapes,\n"
        "        all_reeds_shapes,\n"
        "        reeds_memberships,\n"
        "    )\n"
    )
    needles = {
        "logger": needle_logger,
        "signature": needle_signature,
        "set_index": needle_set_index,
        "main_call": needle_main_call,
    }
    bad = {name: orig_text.count(n) for name, n in needles.items() if orig_text.count(n) != 1}
    if bad:
        raise RuntimeError(
            f"anchor {ADD_ELECTRICITY_SCRIPT} shape unexpected; refusing to patch. "
            f"Needles not found exactly once: {bad}",
        )

    text = orig_text
    # 1. module constant, right after the logger. The slice is inserted verbatim;
    #    a module-level banner above it marks it as a harness patch.
    text = text.replace(
        needle_logger,
        needle_logger
        + "\n"
        + "# EQUIVALENCE-HARNESS ADOPTED-FIX PATCH DL-13 (same as v1-epic d98cb93f):\n"
        + const_block
        + "\n",
    )
    # 2. helper, immediately above filter_plants_by_region (its only caller).
    text = text.replace(needle_signature, helper_block + needle_signature)
    # 3. the gated parameter on the signature.
    text = text.replace(
        needle_signature,
        needle_signature.replace(
            "    reeds_memberships: pd.DataFrame,\n",
            "    reeds_memberships: pd.DataFrame,\n" + banner + "    footprint_scoped: bool = False,\n",
        ),
    )
    # 4. the gated call, right after plants_must_add is finished being built.
    text = text.replace(
        needle_set_index,
        needle_set_index
        + "\n"
        + banner.replace("    #", "        #")
        + "        # The regions layers only tile the model footprint when the run is\n"
        + "        # scoped with model_topology.include, so the unconditional add-back\n"
        + "        # above leaks far-away plants into the model. Bound it — but only for\n"
        + "        # scoped runs, so unfiltered interconnect/usa runs stay byte-identical.\n"
        + "        if footprint_scoped:\n"
        + f"            plants_must_add = {SEAM_FIX_MARK}(\n"
        + "                plants_must_add,\n"
        + "                regions_onshore,\n"
        + "                regions_offshore,\n"
        + "            )\n",
    )
    # 5. main() wiring off snakemake.config.
    text = text.replace(
        needle_main_call,
        banner
        + "    # A run scoped with model_topology.include tiles regions over the footprint\n"
        + "    # only; the seam-plant fallback must then be distance-bounded.\n"
        + '    include_filter = snakemake.config.get("model_topology", {}).get("include") or {}\n'
        + needle_main_call.replace(
            "        reeds_memberships,\n    )\n",
            "        reeds_memberships,\n        footprint_scoped=bool(include_filter),\n    )\n",
        ),
    )

    if text.count(SEAM_FIX_MARK) != 3:  # def, gated call, constant comment
        raise RuntimeError(
            f"DL-13 patch produced {text.count(SEAM_FIX_MARK)} sentinel occurrences "
            "in the anchor (expected 3); refusing to write a half-applied patch",
        )
    if "footprint_scoped=bool(include_filter)" not in text or "if footprint_scoped:" not in text:
        raise RuntimeError("DL-13 patch did not wire footprint_scoped end to end; refusing to write")

    # Idempotence is by CONTENT, not by the sentinel: the patched text is always
    # rebuilt from the pristine anchor plus the live candidate slices, so a later
    # edit to v1-epic's helper re-applies here and re-arms the forced rerun
    # instead of leaving the anchor running a stale copy of it.
    if dst.read_text() == text:
        return
    dst.write_text(text)
    log("applied adopted-fix patch DL-13: add_electricity.py seam-plant bound")
    applied_rules.append("add_electricity")


def snakemake_cmd(target: str, jobs: int = 4) -> list[str]:
    return [
        "uv",
        "run",
        "snakemake",
        target,
        "--configfile",
        CONFIGFILE,
        "-j",
        str(jobs),
        "--scheduler",
        "greedy",
        "--rerun-triggers",
        "mtime",
    ]


def build_side(side: str, target: str, jobs: int = 4, timeout: int = 10800) -> dict:
    """Run snakemake for one side; returns manifest dict (also written)."""
    assert side in ("candidate", "anchor")
    wt = provision_anchor_worktree() if side == "anchor" else REPO
    wf = wt / "workflow"
    cmd = snakemake_cmd(target, jobs)
    marker = wt / FORCE_RERUN_MARKER
    forced = marker.read_text().split() if marker.exists() else []
    if forced:
        cmd += ["-R", *forced]
        log(f"{side}: forcing rerun of {forced} (newly applied patch)")
    t0 = time.time()
    cp = run(cmd, cwd=wf, timeout=timeout)
    wall = time.time() - t0
    ok = cp.returncode == 0
    log(f"{side} build {'OK' if ok else 'FAILED'} in {wall:.0f}s")
    if ok and forced:
        marker.unlink()
    if not ok:
        tail = "\n".join((cp.stderr or cp.stdout).splitlines()[-120:])
        raise RuntimeError(f"{side} snakemake failed (exit {cp.returncode}):\n{tail}")
    return write_manifest(side, wt, target, wall)


def write_manifest(side: str, wt: Path, target: str, wall: float) -> dict:
    wf = wt / "workflow"
    sha = run(["git", "rev-parse", "HEAD"], cwd=wt).stdout.strip()
    cfg = (wf / CONFIGFILE).read_bytes()
    manifest = {
        "side": side,
        "sha": sha,
        "target": target,
        "wall_s": round(wall, 1),
        "config_sha256": hashlib.sha256(cfg).hexdigest(),
        "benchmarks": collect_benchmarks(wf),
        "file_sizes": collect_sizes(wf),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out = REPO / "workflow" / "results" / "equivalence"
    out.mkdir(parents=True, exist_ok=True)
    from .paths import INTERCONNECT

    suffix = "" if INTERCONNECT == "western" else f"_{INTERCONNECT}"
    path = out / f"manifest_{side}{suffix}.json"
    path.write_text(json.dumps(manifest, indent=1))
    log(f"manifest -> {path}")
    return manifest


def collect_benchmarks(wf: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for root in (wf / "benchmarks" / "equivalence", wf / "benchmarks" / "cluster_network"):
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            try:
                header, data = p.read_text().splitlines()[:2]
                rows[str(p.relative_to(wf / "benchmarks"))] = dict(
                    zip(header.split("\t"), data.split("\t")),
                )
            except (ValueError, IndexError):
                continue
    return rows


def collect_sizes(wf: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for root in (wf / "resources" / "equivalence", wf / "results" / "equivalence"):
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file():
                sizes[str(p.relative_to(wf))] = p.stat().st_size
    return sizes


if __name__ == "__main__":
    side, target = sys.argv[1], sys.argv[2]
    build_side(side, target)
