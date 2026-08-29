#!/bin/bash
# One-time setup: seed the per-user configuration files.
#
# Everything else - config.common.yaml, config.plotting.yaml,
# config.sector.yaml, config.default.yaml, config.cluster.yaml and the
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
#   config.cluster.yaml  HPC/SLURM account, partition, email
user_files=(
    "config.default.yaml"
    "config.api.yaml"
    "config.cluster.yaml"
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

echo
echo "Done ($created file(s) created)."
echo "Edit $destination/config.default.yaml (or copy it to"
echo "$destination/config.<scenario>.yaml) and run the workflow with:"
echo "    cd workflow && snakemake -j1 --configfile config/config.default.yaml"
