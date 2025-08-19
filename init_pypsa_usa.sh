#!/bin/bash

templates="workflow/repo_data/config"
destination="workflow/config"
existing_files=$(ls "$destination" | grep -v ".gitkeep")

if [ -z "$existing_files" ]; then
    echo "Copying config files from '$templates' to '$destination'..."
    cp -r "$templates"/* "$destination"
else
    echo "Existing config files found in '$destination'. Delete the following files and rerun."
    echo "$existing_files"
fi
