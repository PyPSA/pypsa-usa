(config)=
# Configuration

(run_cf)=
## `run`

It is common conduct to analyse energy system optimisation models for **multiple scenarios** for a variety of reasons,
e.g. assessing their sensitivity towards changing the temporal and/or geographical resolution or investigating how
investment changes as more ambitious greenhouse-gas emission reduction targets are applied.

The `run` section is used for running and storing scenarios with different configurations which are not covered by [wildcards](#wildcards). It determines the path at which resources, networks and results are stored. Therefore the user can run different configurations within the same directory.

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-at: run:
   :end-before: # docs :

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/run.csv
```


(scenario_cf)=
## `scenario`

The `scenario` section is used for setting the wildcards and defining planning horizon settings. All configurations within this section are described in [wildcards](#wildcards) with the exception of planning_horizons and foresight.

Planning horizons determines which year(s) of future demand forecast to use for your planning model. To build a multi-investment period model set multiple `planning_horizons:` years. The `foresight:` option specifies whether perfect foresight or myopoic foresight optimization model is developed. In perfect foresight, a monolithic model is developed where all `planning_horizons` specified are optimized at once, e.g. future horizon values of costs and demand are incorporated into decisions made in earlier planning horizons. Myopic optimization solves each planning horizon sequentially, and passes the results forward. Currently only `perfect` foresight is implemented.

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-at: scenario:
   :end-before: # docs :

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/scenario.csv
```

(snapshots_cf)=
## `snapshots`

Specifies the temporal range to build an energy system model for as arguments to [`pandas.date_range`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html)

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-at: snapshots:
   :end-before: # docs :

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/snapshots.csv
```

(atlite_cf)=
## `atlite`

Define and specify the `atlite.Cutout` used for calculating renewable potentials and time-series. All options except for `features` are directly used as [`cutout parameters`](https://atlite.readthedocs.io/en/latest/ref_api.html#cutout)

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.common.yaml
   :language: yaml
   :start-at: atlite:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/atlite.csv
```

(electricity_cf)=
## `electricity`

Specifies the types of generators that are included in the network, which are extendable, and the CO2 base for which the optimized reduction is relative to.

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-at: electricity:
   :end-before: # docs :

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/electricity.csv
```

```{note}
See [here](./config-co2-base.md) for information on interconnect level base emission values.
```

(renewable_cf)=
## `renewable`

### `solar`
```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.common.yaml
   :language: yaml
   :start-at: solar:
   :end-before: hydro:

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/solar.csv
```

### `onwind`
```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.common.yaml
   :language: yaml
   :start-at: onwind:
   :end-before: offwind:

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/onwind.csv
```

### `Offshore wind`
```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.common.yaml
   :language: yaml
   :start-at: offwind:
   :end-before: solar:
```

(lines_cf)=
## `lines`
```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-at: lines:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/lines.csv
```

(links_cf)=
## `links`

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-at: links:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/links.csv
```

<!-- (load_cf)=
## `load`

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-after: # p_nom_max:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/load.csv
``` -->

(costs_cf)=
## `costs`

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-at: costs:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/costs.csv
```

(sector_cf)=
## `sector`
<!-- ```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-at: sector:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/sector.csv
``` -->

```{warning}
Sector coupling studies are all under active development
```

```yaml
sector:
  co2_sequestration_potential: 0
  natural_gas:
    allow_imports_exports: true # false to be implemented
    cyclic_storage: false
  heating:
    heat_pump_sink_T: 55.
  demand:
    profile:
      residential: eulp # efs, eulp
      commercial: eulp # efs, eulp
      transport: efs # efs
      industry: efs # efs
    scale:
      residential: aeo # efs, aeo
      commercial: aeo # efs, aeo
      transport: aeo # efs, aeo
      industry: aeo # efs, aeo
    disaggregation:
      residential: pop # pop
      commercial: pop # pop
      transport: pop # pop
      industry: pop # pop
    scenarios:
      aeo: reference
```

```{eval-rst}
.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/sector.csv
```


(clustering_cf)=
## `clustering`

When clustering `aggregation_zones` defines the region boundaries which will be respected through the clustering process; State boarders, balancing authority regions, or REeDs shapes. This feature is important for imposing constraints (`opts`) which are defined over specific regions. For example, the data included in the model on interface transfer capacities are prepared for REeDs shapes but not states and BA regions. Moving forward we plan to use REeDs shapes as our default however we will maintain States and BA regions as well.

Each clustering and interconnection option will have a different number of minimum nodes which can be clustered to, an error will be thrown in `cluster_network` notifying you of that number if you have selected a value too low.

Cleaned and labeled REeDs Shapes are pulled from this github repository: https://github.com/pandaanson/NYU-law-work

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-at: clustering:
   :end-before: # docs :

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/clustering.csv
```


```{note}
`feature:` in `simplify_network:` are only relevant if `hac` were chosen in `algorithm`.
```

```{tip}
use `min` in `p_nom_max:` for more conservative assumptions.
```

(solving_cf)=
## `solving`

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-at: solving:

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/solving.csv
```

(plotting_cf)=
## `plotting`

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.plotting.yaml
   :language: yaml

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/plotting.csv
```
