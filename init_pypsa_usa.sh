#!/bin/bash
# One-time setup: seed the per-user configuration files.
#
# Everything else - config.common.yaml, config.plotting.yaml,
# config.sector.yaml, config.default.yaml and the
# policy_constraints/ CSVs - is loaded by workflow/Snakefile straight out of
# the tracked templates directory, so there is nothing to copy and nothing to
# keep in sync. Only the files below are genuinely user-owned.
#
# Safe to re-run: existing files are left untouched, missing ones are created.

set -euo pipefail

templates="workflow/repo_data/config"
destination="workflow/config"

# Files copied into $destination for the user to edit.
#   config.default.yaml  starting point for your own scenario config
#   config.api.yaml      API keys (EIA); can be replaced by $EIA_API_KEY
#   config.slurm.yaml    HPC/SLURM account, partition, email
user_files=(
    "config.default.yaml"
    "config.api.yaml"
    "config.slurm.yaml"
)

mkdir -p "$destination"

created=0
for f in "${user_files[@]}"; do
    if [ -e "$destination/$f" ]; then
        echo "keeping existing $destination/$f"
    else
        cp "$templates/$f" "$destination/$f"
        echo "created  $destination/$f"
        created=$((created + 1))
    fi
done

# Arm the tracked git hooks. `.pre-commit-config.yaml` has declared a
# check-added-large-files guard for a long time, but it never ran: nothing
# installed it into .git/hooks, so every fresh clone and every `git worktree
# add` started unguarded. That is how 662 MB of renamed, regenerable output
# (incl. a 177.67 MB parquet) reached the history on 2026-09-01 and blocked
# every push for two weeks. core.hooksPath points at a TRACKED directory, so
# the guard travels with the repo instead of living in untracked local state.
if git rev-parse --git-dir >/dev/null 2>&1; then
    git config core.hooksPath .githooks
    echo "armed    .githooks (core.hooksPath) - commits over 10 MB will be refused"
    if command -v pre-commit >/dev/null 2>&1; then
        pre-commit install --install-hooks >/dev/null 2>&1 \
            && echo "armed    pre-commit framework hooks"
    else
        echo "note     pre-commit not on PATH; the size guard still runs, lint hooks do not"
    fi
fi

echo
echo "Done ($created file(s) created)."
echo "Edit $destination/config.default.yaml (or copy it to"
echo "$destination/config.<scenario>.yaml) and run the workflow with:"
echo "    cd workflow && snakemake -j1 --configfile config/config.default.yaml"
