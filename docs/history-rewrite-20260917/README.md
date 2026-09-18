# History rewrite, 2026-09-17

`commit-map.txt` maps every commit sha this repository had before 2026-09-17 to
the sha it has now. Keep it: run cards, the hot-fix ledger and
`tests/equivalence/hotfixes.yaml` all cite shas, and without this table an old
citation cannot be resolved to anything.

## What happened

The cluster policy that hosts this work denies `rm -rf`, so a cleanup on
2026-09-01 **renamed** three directories of finished equivalence output aside
instead of deleting them:

```
resources/equivalence-2wk-rcp85cooler/  ->  .trash-20260901/resources-equivalence-2wk-rcp85cooler/
results/equivalence-usa-2wk/            ->  .trash-20260901/results-equivalence-usa-2wk/
```

`.gitignore` ignores `resources/` and `results/` **by path**. After the rename
those 662 MB no longer matched any rule, a `git add -A` swept 46 files into
commit `a7304ce9`, and one of them —
`demand/usa/power_zonal_components_s300.parquet`, 177.67 MB — exceeded GitHub's
hard 100 MB limit.

Nobody noticed for two weeks, because the repository had no remote configured
at the time. It surfaced on 2026-09-17, when the first push attempt rejected
every branch that descended from `a7304ce9`.

## The rewrite

```bash
git clone --no-local --mirror <repo> clean.git
cd clean.git
git filter-repo --path .trash-20260901 --invert-paths \
    --refs a7304ce9^..feat/equivalence-usa-benchmark \
           a7304ce9^..wip/<each>                      --force
```

The `--refs` scoping is load-bearing. An **unscoped** `filter-repo` run was
tried first and rejected: it strips the `gpgsig` header from every commit it
rewrites, which changed `master` (`fbe5883f`) and `develop` (`34b643ff`) even
though neither ever contained the trash directory. That would have destroyed
the merge base with `PyPSA/pypsa-usa` — no PRs, no rebases, no shared history.
Anchoring the ranges at `a7304ce9^` left every upstream-shared commit
byte-identical, signatures intact.

**60 commits rewritten, all of them ours.** Repository 318 MB → 215 MB.

Invariants checked afterwards: `master` and `develop` shas unchanged, `gpgsig`
still present on `master`, zero commits touching `.trash-20260901`, no blob over
100 MB.

## What stops it recurring

1. `.githooks/pre-commit` — tracked, armed by `init_pypsa_usa.sh` via
   `core.hooksPath`, refuses any staged file over 10 MB. The pre-existing
   `check-added-large-files` hook in `.pre-commit-config.yaml` did not fire
   because `.git/hooks/` was empty: nothing had ever run `pre-commit install`.
   A guard that needs a manual install is off by default on every clone.
2. `.github/workflows/large-files.yml` — the same limit server-side, on **every**
   branch, so `--no-verify` and unarmed clones cannot get around it. The main CI
   workflow only triggers on `master`/`develop`/`v1-epic` and would never have
   seen the branch where this happened.
3. `.gitignore` now ignores by shape as well as path — `.trash*/`,
   `slurm-*.{out,err}`, and the bulk data formats outside `workflow/repo_data/`
   and `workflow/geodata_repo/`.
4. The repository has a remote and is pushed daily. That is the real fix; the
   other three only shorten the time to discovery.
