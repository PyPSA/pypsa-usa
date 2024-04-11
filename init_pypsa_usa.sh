#!/bin/bash

templates="workflow/repo_data/config"
destination="workflow/config"
exisitng_config_files=$(ls "$destination" | grep -v ".gitkeep")

if [ -z "$files" ]; then
    echo "Copying config files from '$templates' to '$destination'..."
    cp -r "$templates"/* "$destination"
else
    echo "Exisitng config files found in '$destination'. Delete the following files and rerun."
    echo "$files"
fi