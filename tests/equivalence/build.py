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

REPO = Path(__file__).resolve().parents[2]
ANCHOR_SHA = "e7f8bd70"
ANCHOR_WORKTREE = REPO / ".worktrees" / "anchor-e7f8bd70"
INFRA_PATCH_MARK = "materialize sibling modules"  # idempotence marker

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

    # Shared harness config (kept in sync from the candidate repo_data copy).
    shutil.copy2(
        REPO / "workflow" / "repo_data" / "config" / "config.equivalence.yaml",
        wf / "config" / "config.equivalence.yaml",
    )

    apply_infra_patches(wt)

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


def snakemake_cmd(
    target: str,
    jobs: int = 4,
    configfile: str = "config/config.equivalence.yaml",
) -> list[str]:
    return [
        "uv",
        "run",
        "snakemake",
        target,
        "--configfile",
        configfile,
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
    t0 = time.time()
    cp = run(snakemake_cmd(target, jobs), cwd=wf, timeout=timeout)
    wall = time.time() - t0
    ok = cp.returncode == 0
    log(f"{side} build {'OK' if ok else 'FAILED'} in {wall:.0f}s")
    if not ok:
        tail = "\n".join((cp.stderr or cp.stdout).splitlines()[-120:])
        raise RuntimeError(f"{side} snakemake failed (exit {cp.returncode}):\n{tail}")
    return write_manifest(side, wt, target, wall)


def write_manifest(side: str, wt: Path, target: str, wall: float) -> dict:
    wf = wt / "workflow"
    sha = run(["git", "rev-parse", "HEAD"], cwd=wt).stdout.strip()
    cfg = (wf / "config" / "config.equivalence.yaml").read_bytes()
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
    path = out / f"manifest_{side}.json"
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
