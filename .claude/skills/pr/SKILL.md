---
name: pr
description: Open a pull request to PyPSA/pypsa-usa following repo conventions
disable-model-invocation: true
---

# Open a PR to PyPSA/pypsa-usa

Invoking `/pr` is standing permission to push the branch and open the PR — no
further confirmation for those two actions. Ask the user only where a step
below says to ask.

## 1. Pick the base branch

The base is `develop` on `PyPSA/pypsa-usa`. Never `master` — if the user asks
for `master`, stop and remind them changes land on `develop` and flow to
`master` in releases.

Integration-branch exception: when the branch descends from an active
integration branch (a long-lived upstream branch such as `v1-epic`), target
that branch instead. Detect it:

```bash
git fetch upstream develop v1-epic
git merge-base HEAD upstream/develop
git merge-base HEAD upstream/v1-epic
```

If the merge-base with the integration branch is a descendant of the
merge-base with `develop`, the branch was cut from the integration branch —
base there.

## 2. Pick the push remote

```bash
gh api repos/PyPSA/pypsa-usa --jq .permissions.push
```

`true` → push the head branch to `upstream` (the `PyPSA/pypsa-usa` remote).
Otherwise push to the user's fork (`origin`) and open a cross-repo PR.

## 3. Gauntlet — every gate green before pushing

1. Working tree clean (`git status`); commit or stash anything loose with the
   user's direction.
2. Branch merges cleanly with the base; rebase or merge the base in if behind.
   Leave history otherwise untouched — no squashing. Conventional-commit
   titles (`fix(rps): ...`) are welcome, not required.
3. `pre-commit run --files <changed files>` passes.
4. `pytest -m fast` passes — Tier A, the same selection CI's `fast-tests` job
   runs, so a local pass predicts a green check.
5. Size the change. It is **larger** when it touches `workflow/scripts/` or
   `workflow/rules/` in a way that can alter produced network artifacts, or
   spans >~300 changed lines / >10 files. For a larger change, ask the user
   whether to also run Tier B (`pytest -m integration`, needs `data/` and
   `cutouts/`) — encourage it, never require it.

## 4. Draft the body

Fill `.github/pull_request_template.md`, then add:

- **Testing** — which tiers ran and their verbatim results (pass/fail counts).
- **Equivalence** (larger changes only) — ask the user whether an
  equivalence-harness run (`tests/equivalence/`) exists for this branch. If
  yes: paste the report summary and list any deltas-ledger rows (DL-N in
  `docs/superpowers/specs/2026-08-07-deltas-ledger.md`) this PR adds or
  touches. If no: write "No equivalence run for this branch." — stated,
  never omitted.
- Ask the user whether the PR closes an issue; if so, `Closes #N` at the top.
- Attribution footer:

  ```
  🤖 Generated with [Claude Code](https://claude.com/claude-code)
  ```

## 5. Open the PR

Open as **draft** when the change is larger and Tier B / equivalence evidence
is missing; otherwise ready-for-review. Mark ready once the evidence lands.

```bash
git push <remote> HEAD:<head-branch>
gh pr create --repo PyPSA/pypsa-usa --base <base> [--draft] ...
```

## 6. Watch CI

Wait for the `fast-tests` check (`gh pr checks <num> --repo PyPSA/pypsa-usa
--watch`), fixing failures in-session and pushing until it is green.
`e2e-tests` runs longer — report its status, don't wait on it.

Done when: PR URL reported to the user, `fast-tests` green (or its failure
explained), and any skipped evidence (Tier B, equivalence) called out
explicitly in the PR body.
