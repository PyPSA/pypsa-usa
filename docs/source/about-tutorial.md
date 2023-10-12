(tutorial)=
# Tutorial 

```{note}
** If you have not done so, please follow the [installation instructions](https://pypsa-usa.readthedocs.io/en/latest/about-install.html) [github issues](https://github.com/PyPSA/pypsa-usa/issues) **
```

## Set Configuration

To start, you'll want to set the proper network configuration for your studies purpose. You can copy the default `config/config.default.yaml` to `config/config.yaml`. The default configuration using the `western` interconnect and 30 nodes is a good place to start!

You can find more information on each configuration setting on the [configurations page](https://pypsa-usa.readthedocs.io/en/latest/config-configuration.html).


## Run workflow

To run the workflow, `cd` into the `workflow` directory and run the `snakemake` from your terminal.

```bash
snakemake -j4
```

where 4 indicates the number of used cores.

The `build_renewable_profiles` rule will take ~10-15 minutes to run the first time you run the workflow. After that, changing the number of clusters, load, or generator configurations will not require rebuilding the renewable profiles. Changes to `renewables` configuration will cause re-run of `build_renewable_profiles`.

(troubleshooting)=
# Troubleshooting:

To force the execution of a portion of the workflow up to a given rule, cd to the `workflow` directory and run:

```bash
snakemake -j4 -R build_shapes  --until build_base_network
```
where `build_shapes` is forced to run, and `build_base_network` is the last rule you would like to run.
