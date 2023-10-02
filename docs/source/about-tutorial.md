(tutorial)=
# Tutorial 

We are working on this!

(execution)=
# Execution 
To execute the workflow, go into the `workflow` directory and execute `snakemake` from your terminal. 

```bash
snakemake -j6
```

where 6 indicates the number of used cores, you may change it to your preferred number. This will run the workflow defined in the `Snakefile`.

Note: The `build_renewable_profiles` rule will take ~10-15 minutes to run the first time you run the workflow. After that, changing the number of clusters, load, or generator configurations will not require rebuilding the renewable profiles. Changes to `renewables` configuration will cause re-run of `build_renewable_profiles`.

(troubleshooting)=
# Troubleshooting:

To force the execution of a portion of the workflow up to a given rule, cd to the `workflow` directory and run:

```bash
snakemake -j4 -R build_shapes  --until build_base_network
```
where `build_shapes` is forced to run, and `build_base_network` is the last rule you would like to run.
