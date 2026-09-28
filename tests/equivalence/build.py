"""Build one side (``master`` or ``develop``) of an equivalence run.

The develop side builds in the main checkout. The baseline builds in a
dedicated git worktree registered at the resolved ``master-benchmark`` sha —
a commit anyone can check out, diff and review. Whether that worktree is
attached to the branch or detached does not matter and is deliberately not
forced: the branch is still being built in it.

**There is no build-time patching.** Everything the baseline needs in order to
run and to be compared lives as ordinary commits on ``master-benchmark``
(``BENCHMARK-BRANCH.md`` on that branch is its manifest; the project brain's
``memory/plans/harness-master-vs-develop.md`` T1 is the policy). The old
sentinel/string-surgery machinery and its ``.eq-force-rerun`` marker are gone:
the baseline that actually ran is now a sha, not a mutated working tree.

Provisioning does, in order:

- resolve ``BASELINE_REF`` to a full sha;
- ``git worktree prune``, then **refuse** a ``.worktrees/master-benchmark``
  directory that git does not list as a registered worktree (a leftover from a
  previous checkout location carries a ``.git`` file pointing at another
  repository's gitdir; adopting it would run git against the wrong repo);
- make sure the worktree is at the resolved sha: ``git worktree add --detach``
  when it does not exist, ``git -C <wt> checkout --detach`` when it exists at a
  *different* sha, and nothing at all when it is already there (it may be
  attached to ``master-benchmark``; that is left alone on purpose);
- symlink ``workflow/data`` and ``workflow/cutouts`` from the main checkout
  (untracked on both branches; master's ``.gitignore`` covers ``data/`` but
  not ``cutouts``, which is why the clean-tree check below ignores untracked
  files);
- seed the gitignored layered configs from the baseline's own
  ``workflow/repo_data/config/`` (they are ``configfile:``-loaded
  unconditionally);
- copy the shared ``config.equivalence*.yaml`` in, with ``{clusters}``
  translated into the baseline's dialect (``paths.baseline_clusters``);
- sync the harness-only policy CSVs and the per-user API keys;
- ``touch`` retrieve_caiso_data's output if present so a fresh-checkout mtime
  on its tracked input xlsx does not retrigger a re-download into shared
  ``data/``.

Before either side builds, ``assert_clean_checkout`` refuses a checkout whose
TRACKED files differ from its commit — otherwise the sha the manifest records
would not describe the code that ran. ``EQ_ALLOW_DIRTY=1`` builds anyway and
the manifest carries ``dirty: true`` plus the offending paths.

Instrumentation: after a build, ``write_manifest`` records git SHA, config
hash, per-rule benchmark rows (wall time, max_rss) and output file sizes, into
this run's directory (``paths.run_dir()``).
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
from pathlib import Path

from .paths import BASELINE_CONFIGFILE, BASELINE_REF, CONFIGFILE, baseline_clusters, run_dir

REPO = Path(__file__).resolve().parents[2]
SIDES = ("master", "develop")
BASELINE_WORKTREE = REPO / ".worktrees" / "master-benchmark"

# Packages whose resolved version is recorded per side (the two branches pin
# different pypsa/pandas/linopy majors, so this is load-bearing provenance).
ENV_PACKAGES = ("pypsa", "pandas", "linopy", "numpy")
_VERSION_PROBE = (
    "import json, platform, importlib.metadata as md\n"
    "out = {'python': platform.python_version()}\n"
    f"for pkg in {ENV_PACKAGES!r}:\n"
    "    try:\n"
    "        out[pkg] = md.version(pkg)\n"
    "    except Exception:\n"
    "        out[pkg] = ''\n"
    "print(json.dumps(out))\n"
)


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


def baseline_ref() -> str:
    """The ref the baseline is built from, read at call time.

    ``EQ_BASELINE_REF`` wins over the module default so a long-lived process
    (and a test) can change it without re-importing ``paths``.
    """
    return os.environ.get("EQ_BASELINE_REF") or BASELINE_REF


def resolve_baseline_sha() -> str:
    """Full sha of the baseline ref, resolved at run time.

    Raises with a pointer to T1 of the harness plan when the branch does not
    exist locally — the baseline is a branch someone has to build, not a pin
    this module can invent.
    """
    ref = baseline_ref()
    cp = run(["git", "rev-parse", f"{ref}^{{commit}}"], cwd=REPO, timeout=60)
    sha = cp.stdout.strip()
    if cp.returncode != 0 or len(sha) != 40:
        raise RuntimeError(
            f"cannot resolve baseline ref {ref!r} in {REPO}.\n"
            "The baseline is the master-benchmark branch, not a pinned commit: "
            "create it first (T1 of memory/plans/harness-master-vs-develop.md — "
            "`git switch -c master-benchmark master`, then port the documented "
            "commits), or set EQ_BASELINE_REF to an existing ref.\n"
            f"git said: {(cp.stderr or cp.stdout).strip()[-500:]}"
        )
    return sha


def _registered_worktrees() -> dict[str, str]:
    """``{realpath: HEAD sha}`` for every worktree git currently lists."""
    cp = run(["git", "worktree", "list", "--porcelain"], cwd=REPO, timeout=60)
    if cp.returncode != 0:
        raise RuntimeError(f"git worktree list failed:\n{cp.stderr[-2000:]}")
    out: dict[str, str] = {}
    path = None
    for line in cp.stdout.splitlines():
        if line.startswith("worktree "):
            path = os.path.realpath(line[len("worktree ") :])
            out[path] = ""
        elif line.startswith("HEAD ") and path is not None:
            out[path] = line[len("HEAD ") :].strip()
    return out


def _foreign_gitdir(wt: Path) -> str:
    """What a directory that is not a registered worktree claims to be."""
    dot_git = wt / ".git"
    if dot_git.is_file():
        text = dot_git.read_text().strip()
        for line in text.splitlines():
            if line.startswith("gitdir:"):
                return line.split("gitdir:", 1)[1].strip()
        return f"a .git file with no gitdir: line ({text[:200]!r})"
    if dot_git.is_dir():
        return f"a nested git repository at {dot_git}"
    return f"no .git entry at all ({dot_git})"


def provision_baseline_worktree() -> Path:
    """Create/refresh the baseline worktree at the resolved sha; idempotent."""
    sha = resolve_baseline_sha()
    wt = BASELINE_WORKTREE
    wt.parent.mkdir(parents=True, exist_ok=True)

    # Drop records of worktrees whose directory is gone, so the registration
    # check below reflects reality rather than stale bookkeeping.
    run(["git", "worktree", "prune"], cwd=REPO, timeout=120)
    registered = _registered_worktrees()
    key = os.path.realpath(wt)

    if wt.exists() and key not in registered:
        raise RuntimeError(
            f"{wt} exists but git does not list it as a worktree of {REPO}; "
            f"it points at {_foreign_gitdir(wt)}. This is a leftover from a "
            "previous checkout location — running git against it would target "
            "another repository. Remove or rename it, then re-run; refusing to "
            "adopt it."
        )

    if key in registered:
        if registered[key] != sha:
            cp = run(["git", "-C", str(wt), "checkout", "--detach", sha], cwd=REPO, timeout=600)
            if cp.returncode != 0:
                raise RuntimeError(f"git checkout --detach {sha} failed:\n{cp.stderr[-2000:]}")
            log(f"baseline worktree repointed {registered[key][:12]} -> {sha[:12]}")
    else:
        cp = run(["git", "worktree", "add", "--detach", str(wt), sha], cwd=REPO, timeout=600)
        if cp.returncode != 0:
            raise RuntimeError(f"git worktree add failed:\n{cp.stderr[-2000:]}")
        log(f"baseline worktree created at {sha[:12]} ({baseline_ref()})")

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

    # Seed every gitignored config entry from the baseline's own templates
    # (mirrors tests/integration/conftest.py::_seed_runtime_configs and
    # init_pypsa_usa.sh): layered yaml files AND directories such as
    # config/policy_constraints/ that rules reference as inputs.
    src_root = wf / "repo_data" / "config"
    dst_root = wf / "config"
    # A fresh worktree has no workflow/config/ at all: master's .gitignore has a
    # bare `config/` rule, so nothing under it is checked out.
    dst_root.mkdir(parents=True, exist_ok=True)
    for entry in src_root.iterdir():
        dst = dst_root / entry.name
        if dst.exists():
            continue
        if entry.is_dir():
            shutil.copytree(entry, dst)
        else:
            shutil.copy2(entry, dst)

    # Shared harness configs (kept in sync from the develop repo_data copies).
    # master keeps the pre-2026-09-06 {clusters} suffix semantics, so
    # scenario.clusters is translated on the way in (paths.baseline_clusters).
    for cfg in (REPO / "workflow" / "repo_data" / "config").glob("config.equivalence*.yaml"):
        text = re.sub(
            r"^(\s*clusters:\s*\[)([^\]]*)(\])",
            lambda m: m.group(1) + ", ".join(baseline_clusters(v.strip()) for v in m.group(2).split(",")) + m.group(3),
            cfg.read_text(),
            count=1,
            flags=re.MULTILINE,
        )
        (wf / "config" / cfg.name).write_text(text)

    # Harness-specific policy CSVs (e.g. the USA national CO2 cap) are
    # referenced by repo_data-relative paths in the shared configs; master does
    # not ship them, so sync them from the develop checkout.
    pc_src = REPO / "workflow" / "repo_data" / "config" / "policy_constraints"
    pc_dst = wf / "repo_data" / "config" / "policy_constraints"
    for csv in pc_src.glob("*equivalence*.csv"):
        shutil.copy2(csv, pc_dst / csv.name)
        shutil.copy2(csv, wf / "config" / "policy_constraints" / csv.name)

    # Per-user API keys live in the develop checkout's untracked config/
    # overlay (never in the tracked repo_data templates). Mirror them onto the
    # baseline so both sides run with the same credentials. Infra-only:
    # identical key, cannot move numbers.
    api_overlay = REPO / "workflow" / "config" / "config.api.yaml"
    if api_overlay.exists():
        shutil.copy2(api_overlay, wf / "config" / "config.api.yaml")

    # Fresh-checkout mtime on the tracked caiso xlsx must not retrigger a
    # re-download into the SHARED data/ tree.
    caiso_out = wf / "data" / "costs" / "caiso_ng_power_prices.csv"
    if caiso_out.exists():
        os.utime(caiso_out)

    return wt


def checkout_dirt(root: Path) -> list[str]:
    """Modified or staged TRACKED files in a checkout, one porcelain line each.

    ``--untracked-files=no`` is deliberate. The harness itself creates untracked
    files in both checkouts — the ``data``/``cutouts`` symlinks, the seeded
    ``workflow/config/`` overlay, ``resources/``, ``results/``, ``benchmarks/``
    — and master's ``.gitignore`` does not cover all of them. What invalidates
    a run is a *tracked* file that differs from the commit whose sha the
    manifest records.
    """
    cp = run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, timeout=120)
    if cp.returncode != 0:
        raise RuntimeError(f"git status failed in {root}:\n{cp.stderr[-2000:]}")
    return [line for line in cp.stdout.splitlines() if line.strip()]


def assert_clean_checkout(side: str, root: Path) -> list[str]:
    """Refuse to build a side whose tracked files differ from its commit.

    ``EQ_ALLOW_DIRTY=1`` builds anyway and the manifest then carries
    ``dirty: true`` with the offending paths, so the numbers are never read as
    coming from the recorded sha. That escape hatch is what keeps plan D2 ("a
    dirty develop side is recorded and warned about, not refused") reachable
    while the default stays a refusal.
    """
    dirt = checkout_dirt(root)
    if not dirt:
        return dirt
    listing = "\n".join(dirt[:20])
    more = f"\n... and {len(dirt) - 20} more" if len(dirt) > 20 else ""
    if os.environ.get("EQ_ALLOW_DIRTY") == "1":
        log(
            f"WARNING: {side} checkout {root} has uncommitted changes to tracked files; "
            f"building anyway because EQ_ALLOW_DIRTY=1:\n{listing}{more}"
        )
        return dirt
    raise RuntimeError(
        f"{side} checkout {root} has uncommitted changes to tracked files, so the sha "
        f"its manifest records would not describe the code that ran:\n{listing}{more}\n"
        "Commit or stash them, or set EQ_ALLOW_DIRTY=1 to build anyway and have the "
        "manifest record dirty: true."
    )


def side_root(side: str) -> Path:
    """Repository root a side builds in (provisions the baseline on demand)."""
    if side not in SIDES:
        raise ValueError(f"unknown side {side!r}; expected one of {SIDES}")
    return provision_baseline_worktree() if side == "master" else REPO


def side_configfile(side: str) -> str:
    """Where each side's copy of the shared harness config lives.

    develop loads it from the tracked ``repo_data/config/`` templates; master's
    Snakefile only looks in ``config/``, and ``provision_baseline_worktree``
    copies the file there.
    """
    if side not in SIDES:
        raise ValueError(f"unknown side {side!r}; expected one of {SIDES}")
    return BASELINE_CONFIGFILE if side == "master" else CONFIGFILE


def side_env_versions(wt: Path) -> dict[str, str]:
    """Resolved ``{'python','pypsa','pandas','linopy','numpy'}`` for one side.

    Read from that side's OWN venv: ``uv run`` inside a worktree resolves that
    worktree's ``pyproject.toml``/``uv.lock``, and the two branches pin
    different pypsa/pandas/linopy majors. A probe failure is logged and
    returned as empty strings rather than raised — provenance must not abort a
    ten-hour build.
    """
    blank = dict.fromkeys(("python", *ENV_PACKAGES), "")
    try:
        cp = run(["uv", "run", "python", "-c", _VERSION_PROBE], cwd=wt, timeout=900)
    except (OSError, subprocess.SubprocessError) as exc:
        log(f"env probe failed in {wt}: {exc}")
        return blank
    if cp.returncode != 0:
        log(f"env probe failed in {wt} (exit {cp.returncode}): {cp.stderr[-500:]}")
        return blank
    try:
        return json.loads(cp.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        log(f"env probe produced unparseable output in {wt}: {cp.stdout[-500:]!r}")
        return blank


def snakemake_cmd(
    target: str,
    jobs: int = 4,
    configfile: str = CONFIGFILE,
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
        # A predecessor job killed mid-rule (the driver's walltime, an OOM)
        # leaves snakemake metadata marking those outputs incomplete, and the
        # next run refuses to start until someone intervenes. Re-run them
        # instead: the whole point of the run directory is that a resumed build
        # picks up where the dead one stopped.
        "--rerun-incomplete",
    ]


def mirror_godeeep_cf_for_master(data_dir: Path) -> list[Path]:
    """Give master the GODEEEP CF files develop retrieved, as hard links.

    develop's ``retrieve_godeeep_cf`` places CF files under
    ``data/godeeep/<scenario>/``; master's ``build_renewable_profiles`` calls
    its own ``ZenodoScenarioDownloader``, which looks under
    ``data/zenodo/<scenario>/`` and, when the file is absent, fetches it from
    Zenodo — or returns ``None`` on any metadata miss, which xarray then reports
    as "did not find a match in any of xarray's currently installed IO
    backends" (smoke job 43580538, 2026-09-15). Both directories live in the
    shared cache; the historical files were hand-linked on 2026-09-01, which
    is why the USA leg never hit this.

    Linking every file develop retrieved does three things at once: master
    reads byte-identical inputs, master never touches the network inside the
    benchmark, and a scenario the cache has never seen works on the first
    run. Hard link first (same filesystem, no duplication); symlink if the
    link crosses filesystems. Existing regular files are left alone.
    Returns the paths created.
    """
    src_root = data_dir / "godeeep"
    dst_root = data_dir / "zenodo"
    created: list[Path] = []
    if not src_root.is_dir():
        return created
    for src in sorted(src_root.glob("*/*.nc")):
        if not src.is_file():
            continue
        dst = dst_root / src.parent.name / src.name
        if dst.exists() or dst.is_symlink():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src.resolve(), dst)
        except OSError:
            dst.symlink_to(src.resolve())
        created.append(dst)
    if created:
        log(f"master: mirrored {len(created)} GODEEEP CF file(s) from data/godeeep/ into data/zenodo/")
    return created


def build_side(side: str, target: str, jobs: int = 4, timeout: int = 10800) -> dict:
    """Run snakemake for one side; returns manifest dict (also written)."""
    if side not in SIDES:
        raise ValueError(f"unknown side {side!r}; expected one of {SIDES}")
    wt = side_root(side)
    assert_clean_checkout(side, wt)
    wf = wt / "workflow"
    if side == "master":
        mirror_godeeep_cf_for_master(wf / "data")
    cmd = snakemake_cmd(target, jobs, side_configfile(side))
    t0 = time.time()
    cp = run(cmd, cwd=wf, timeout=timeout)
    wall = time.time() - t0
    ok = cp.returncode == 0
    log(f"{side} build {'OK' if ok else 'FAILED'} in {wall:.0f}s")
    if not ok:
        tail = "\n".join((cp.stderr or cp.stdout).splitlines()[-120:])
        raise RuntimeError(f"{side} snakemake failed (exit {cp.returncode}):\n{tail}")
    return write_manifest(side, wt, target, wall)


def write_manifest(side: str, wt: Path, target: str, wall: float) -> dict:
    wf = wt / "workflow"
    sha = run(["git", "rev-parse", "HEAD"], cwd=wt, timeout=60).stdout.strip()
    dirt = checkout_dirt(wt)
    cfg = (wf / side_configfile(side)).read_bytes()
    manifest = {
        "side": side,
        "ref": baseline_ref() if side == "master" else "HEAD",
        "sha": sha,
        # True only under EQ_ALLOW_DIRTY=1; build_side refuses otherwise.
        "dirty": bool(dirt),
        "dirty_paths": dirt[:50],
        "target": target,
        "wall_s": round(wall, 1),
        "config_sha256": hashlib.sha256(cfg).hexdigest(),
        "benchmarks": collect_benchmarks(wf),
        "file_sizes": collect_sizes(wf),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    # One run directory per run (plan D5): both sides' manifests land next to
    # run_meta.json, the findings and the figures, so a run is one directory
    # rather than a set of files that have to be matched up by suffix.
    out = run_dir()
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
