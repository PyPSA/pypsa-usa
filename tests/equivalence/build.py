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
- apply the documented ADOPTED-FIX patch to ``build_bus_regions`` (DL-11:
  footprint-scoped empty-county sweep). Results-affecting BY DESIGN and
  applied to BOTH sides by user decision (2026-08-23) so the harness keeps
  comparing like-for-like; see the deltas ledger;
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
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .paths import CONFIGFILE

REPO = Path(__file__).resolve().parents[2]
ANCHOR_SHA = "e7f8bd70"
ANCHOR_WORKTREE = REPO / ".worktrees" / "anchor-e7f8bd70"
INFRA_PATCH_MARK = "materialize sibling modules"  # idempotence marker
ADOPTED_FIX_MARK = "restricting empty-county sweep"  # idempotence marker (DL-11)
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
    """
    applied = False
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
        applied = True

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
        applied = True

    if applied:
        # The harness runs snakemake with ``--rerun-triggers mtime``, under
        # which code/rule changes NEVER invalidate existing outputs — a
        # freshly patched worktree with pre-patch artifacts would silently
        # keep them and the harness would compare fixed-vs-unfixed
        # (2026-08-23 adversarial-review blocker). Deleting the rule's
        # outputs is NOT enough either: when the final target is otherwise
        # up-to-date, snakemake never revisits missing intermediates
        # (observed 2026-08-23). Instead, record a one-shot forced-rerun
        # marker that build_side turns into ``-R build_bus_regions`` on the
        # next build and clears on success. One-time: patch application is
        # marker-gated.
        (wt / FORCE_RERUN_MARKER).write_text("build_bus_regions\n")
        log("wrote one-shot forced-rerun marker: build_bus_regions")


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
