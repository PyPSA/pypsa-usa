"""Policy checks on the ``master-benchmark`` baseline branch (harness plan T1).

Tier A (``fast``): a handful of ``git log`` / ``git show`` calls against the
object database this repository already has. No data, no network, no build.

The baseline the harness builds is a branch, not a patched checkout, so the
branch itself is the artifact that has to be reviewable:

- every commit declares a ``Category:`` and where it was ``Ported-from:``;
- every ported item names the hot-fix ledger row it closes, so a difference the
  baseline removes can never be read as a ``develop`` effect;
- a ``Category: resources`` commit may touch nothing but resource/benchmark
  declarations — that is what makes "resource tweaks cannot change numbers" a
  checked statement rather than an asserted one;
- ``BENCHMARK-BRANCH.md`` lists exactly the ported commits, so the manifest
  cannot drift from the branch.

Everything is skipped with a clear reason when the branch or its worktree is
absent, so a checkout that has not run T1 yet stays green.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from tests.equivalence import build

pytestmark = pytest.mark.fast

REPO = Path(__file__).resolve().parents[2]
BRANCH = "master-benchmark"
BASE = "master"
MANIFEST = "BENCHMARK-BRANCH.md"
HOTFIXES = Path(__file__).parent / "hotfixes.yaml"

CATEGORIES = {"runnability", "comparability", "resources"}

# A `resources` commit may only move these knobs, and only in these files.
_RESOURCE_FILES = re.compile(
    r"^workflow/rules/[^/]+\.smk$|^workflow/(repo_data/)?config/config\.cluster\.yaml$",
)
_RESOURCE_LINE = re.compile(r"^[+-]\s*(mem_mb|walltime|threads|runtime|benchmark)\b")
_RESOURCE_BLOCK = re.compile(r"^[+-]\s*(resources|benchmark)\s*:")

_UNIT_SEP = "\x1f"
_REC_SEP = "\x1e"
_SHA_TOKEN = re.compile(r"`([0-9a-f]{7,40})`")


def _git(*args: str, cwd: Path = REPO) -> str:
    cp = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if cp.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed in {cwd}:\n{cp.stderr.strip()}")
    return cp.stdout


def _ref_exists(ref: str) -> bool:
    return (
        subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
            cwd=REPO,
            capture_output=True,
        ).returncode
        == 0
    )


@pytest.fixture(scope="module")
def branch() -> str:
    """Skip the module unless the branch AND its worktree are both in place."""
    wt = build.BASELINE_WORKTREE
    if not wt.exists():
        pytest.skip(
            f"{wt} does not exist — run T1 of memory/plans/harness-master-vs-develop.md "
            f"(create {BRANCH} and its worktree) before these checks mean anything",
        )
    if os.path.realpath(wt) not in build._registered_worktrees():
        pytest.skip(f"{wt} exists but git does not list it as a worktree of {REPO}")
    for ref in (BASE, BRANCH):
        if not _ref_exists(ref):
            pytest.skip(f"ref {ref!r} does not exist locally; T1 has not run here")
    return BRANCH


def _trailers(block: str) -> dict[str, str]:
    """Parse git's own trailer block (``%(trailers:only)``) into ``{key: value}``.

    Git decides what a trailer is; this only splits the block it hands back, so
    a ``Category:`` written in the middle of the body is not mistaken for one.
    """
    out: dict[str, str] = {}
    for line in block.splitlines():
        m = re.match(r"^([A-Za-z][A-Za-z-]*):\s*(.*?)\s*$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


@pytest.fixture(scope="module")
def commits(branch: str) -> list[dict]:
    """``master..master-benchmark``, oldest first, with trailers and files."""
    raw = _git(
        "log",
        "--reverse",
        f"--format=%H{_UNIT_SEP}%s{_UNIT_SEP}%(trailers:only=true,unfold=true){_REC_SEP}",
        f"{BASE}..{branch}",
    )
    out: list[dict] = []
    for record in raw.split(_REC_SEP):
        record = record.strip("\n")
        if not record.strip():
            continue
        sha, subject, trailer_block = record.split(_UNIT_SEP, 2)
        sha = sha.strip()
        files = [f for f in _git("show", "--name-only", "--format=", sha).splitlines() if f.strip()]
        tr = _trailers(trailer_block)
        out.append(
            {
                "sha": sha,
                "subject": subject.strip(),
                "trailers": tr,
                "files": files,
                # The manifest commit cannot carry its own sha in its own table,
                # so it is the one commit excused from the Hot-fix requirement.
                "is_manifest": "Hot-fix" not in tr and files == [MANIFEST],
            },
        )
    return out


def test_branch_point(branch):
    """The branch is master plus commits, never master minus any."""
    assert (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", BASE, branch],
            cwd=REPO,
            capture_output=True,
        ).returncode
        == 0
    ), f"{BASE} is not an ancestor of {branch}; rebase it instead of merging"
    behind = _git("rev-list", "--count", f"{branch}..{BASE}").strip()
    assert behind == "0", f"{branch} is missing {behind} commit(s) present on {BASE}"


def test_branch_is_not_empty(commits):
    assert commits, f"{BASE}..{BRANCH} is empty; T1 ported nothing"


def test_commit_trailers(commits):
    """Every commit declares its category and its provenance."""
    bad: list[str] = []
    for c in commits:
        tr = c["trailers"]
        cat = tr.get("Category")
        if cat not in CATEGORIES:
            bad.append(f"{c['sha'][:7]} {c['subject']!r}: Category={cat!r}, expected one of {sorted(CATEGORIES)}")
        if not tr.get("Ported-from"):
            bad.append(f"{c['sha'][:7]} {c['subject']!r}: missing a Ported-from: trailer")
    assert not bad, "commit trailer policy violations:\n" + "\n".join(bad)


def test_ported_commits_name_a_hotfix(commits):
    """Every ported item cites its ledger row; only the manifest commit is excused."""
    bad: list[str] = []
    for c in commits:
        if c["is_manifest"]:
            continue
        hf = c["trailers"].get("Hot-fix", "")
        if not re.fullmatch(r"HF-\d+", hf):
            bad.append(
                f"{c['sha'][:7]} {c['subject']!r}: Hot-fix={hf!r}, expected 'HF-<n>' "
                "(bare integer, not zero-padded)",
            )
    assert not bad, "hot-fix trailer violations:\n" + "\n".join(bad)


def test_exactly_one_manifest_commit(commits):
    manifest = [c for c in commits if c["is_manifest"]]
    assert len(manifest) == 1, (
        f"expected exactly one commit adding/updating {MANIFEST} with no Hot-fix trailer, "
        f"found {len(manifest)}: {[c['sha'][:7] for c in manifest]}"
    )


def test_hotfix_ids_resolve(commits):
    """Every id cited on the branch has a row in the registry.

    The registry landed with T4, so this no longer skips: an id with no row is
    a port nobody can trace back to a ledger entry.
    """
    assert HOTFIXES.exists(), (
        f"{HOTFIXES.name} is missing; it is the machine-readable extract of "
        "memory/plans/hotfix-ledger.md and every ported commit is checked against it"
    )
    rows = yaml.safe_load(HOTFIXES.read_text()) or []
    known = {str(r.get("id")) for r in rows}
    missing = sorted(
        {c["trailers"]["Hot-fix"] for c in commits if c["trailers"].get("Hot-fix")} - known,
    )
    assert not missing, f"Hot-fix ids on {BRANCH} with no row in {HOTFIXES.name}: {missing}"


def test_ported_commits_are_marked_ported_in_the_registry(commits):
    """A fix on the baseline must read ``ported: true`` in the registry.

    This is the join that keeps a neutralised difference out of the explanation
    set. If the branch carries HF-8 but hotfixes.yaml still says
    ``ported: false``, the comparison table will happily accept HF-8 as the
    reason two sides that BOTH run it disagree.
    """
    rows = yaml.safe_load(HOTFIXES.read_text()) or []
    registry = {str(r.get("id")): r for r in rows}
    unmarked = sorted(
        hid
        for hid in {c["trailers"]["Hot-fix"] for c in commits if c["trailers"].get("Hot-fix")}
        if not registry.get(hid, {}).get("ported")
    )
    assert not unmarked, (
        f"{BRANCH} carries {unmarked} but {HOTFIXES.name} does not mark them ported: true — "
        "a ported hot-fix runs on BOTH sides and must not stay a candidate explanation"
    )


def test_resources_commits_are_inert(commits):
    """A `resources` commit may not touch anything that can move a number.

    Vacuous while no such commit exists — that is the point: it is armed for
    the first one.
    """
    bad: list[str] = []
    for c in commits:
        if c["trailers"].get("Category") != "resources":
            continue
        for path in c["files"]:
            if not _RESOURCE_FILES.match(path):
                bad.append(f"{c['sha'][:7]}: resources commit touches {path}")
        diff = _git("show", "--unified=0", "--format=", c["sha"])
        for line in diff.splitlines():
            if not line.startswith(("+", "-")) or line.startswith(("+++", "---")):
                continue
            if _RESOURCE_LINE.match(line) or _RESOURCE_BLOCK.match(line):
                continue
            bad.append(f"{c['sha'][:7]}: resources commit changes a non-resource line: {line.strip()!r}")
    assert not bad, "resources commits must be numerically inert:\n" + "\n".join(bad)


def _manifest_table_shas(branch: str) -> list[str]:
    """Backticked shas in the ``## Commits`` table of BENCHMARK-BRANCH.md."""
    text = _git("show", f"{branch}:{MANIFEST}")
    start = text.index("## Commits")
    end = text.find("\n## ", start + 1)
    table = text[start:end] if end != -1 else text[start:]
    return [m.group(1) for line in table.splitlines() if line.lstrip().startswith("|") for m in _SHA_TOKEN.finditer(line)]


def test_branch_manifest_matches(branch, commits):
    """BENCHMARK-BRANCH.md lists exactly the ported commits, in order."""
    listed = _manifest_table_shas(branch)
    ported = [c["sha"] for c in commits if not c["is_manifest"]]

    unlisted = [s[:7] for s in ported if not any(s.startswith(t) for t in listed)]
    assert not unlisted, (
        f"{MANIFEST} does not list ported commit(s) {unlisted}; "
        "regenerate its table (branch-point sha included) after every port"
    )

    stale = [t for t in listed if not any(s.startswith(t) for s in ported)]
    assert not stale, f"{MANIFEST} lists sha(s) {stale} that are not commits on {BRANCH}"

    order = [next(s for s in ported if s.startswith(t)) for t in listed]
    assert order == ported, (
        f"{MANIFEST} lists the commits in a different order than the branch.\n"
        f"manifest: {[s[:7] for s in order]}\nbranch:   {[s[:7] for s in ported]}"
    )
