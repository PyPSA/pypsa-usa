(config)=
# Configuration

(run_cf)=
## `run`

It is common conduct to analyse energy system optimisation models for **multiple scenarios** for a variety of reasons,
e.g. assessing their sensitivity towards changing the temporal and/or geographical resolution or investigating how
investment changes as more ambitious greenhouse-gas emission reduction targets are applied.

The `run` section is used for running and storing scenarios with different configurations which are not covered by [wildcards](#wildcards). It determines the path at which resources, networks and results are stored. Therefore the user can run different configurations within the same directory.

```{eval-rst}
.. literalinclude:: ../../workflow/config/config.default.yaml
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

Planning horizons determines which year of future demand forecast to use for your planning model. If you leave `planning_horizons:` empty, historical demand will be set according to `snapshots`.

```{eval-rst}
.. literalinclude:: ../../workflow/config/config.default.yaml
   :language: yaml
   :start-at: scenario:
   :end-before: # docs :
```

(snapshots_cf)=
## `snapshots`

Specifies the temporal range to build an energy system model for as arguments to `(pandas.date_range)[https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html]`

```{eval-rst}
.. literalinclude:: ../../workflow/config/config.default.yaml
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
.. literalinclude:: ../../workflow/config/config.common.yaml
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
.. literalinclude:: ../../workflow/config/config.default.yaml
   :language: yaml
   :start-at: electricity:
   :end-before: # docs :

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/electricity.csv
```

If using the `{opts}` wildcard to reduce emissions, the user must put in a `co2base` value. Provided below are historical yearly CO2 emission values for both the power sector and all sectors at an interconnect level. This data can be used as a starting point for users. **Note the units in this table are Million Metric Tons (MMT).** This data originates from the [EIA State Level CO2 database](https://www.eia.gov/opendata/browser/co2-emissions/co2-emissions-aggregates?frequency=annual&data=value;&sortColumn=period;&sortDirection=desc;), and is compiled by the script `workflow/notebooks/historical_emissions.ipynb`

```{eval-rst}
.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/emissions.csv
```
(renewable_cf)=
## `renewable`

### `solar`
```{eval-rst}
.. literalinclude:: ../../workflow/config/config.common.yaml
   :language: yaml
   :start-at: solar:
   :end-before: # docs :

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/solar.csv
```

### `onwind`
```{eval-rst}
.. literalinclude:: ../../workflow/config/config.common.yaml
   :language: yaml
   :start-at: onwind:
   :end-before: # docs :

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/onwind.csv
```

(lines_cf)=
## `lines`
```{eval-rst}
.. literalinclude:: ../../workflow/config/config.default.yaml
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
.. literalinclude:: ../../workflow/config/config.default.yaml
   :language: yaml
   :start-at: links:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/links.csv
```

(load_cf)=
## `load`

```{eval-rst}
.. literalinclude:: ../../workflow/config/config.default.yaml
   :language: yaml
   :start-after: load:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/load.csv
```

(costs_cf)=
## `costs`

```{eval-rst}
.. literalinclude:: ../../workflow/config/config.default.yaml
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
```{eval-rst}
.. literalinclude:: ../../workflow/config/config.default.yaml
   :language: yaml
   :start-at: sector:
   :end-before: # docs

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
.. literalinclude:: ../../workflow/config/config.default.yaml
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

- Use `focus_weights` to specify the proportion of cluster nodes to be attributed to a given zone given by the `aggregation_zone` configuration.
```

```{tip}
use `min` in `p_nom_max:` for more conservative assumptions.
```

(solving_cf)=
## `solving`

```{eval-rst}
.. literalinclude:: ../../workflow/config/config.default.yaml
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
.. literalinclude:: ../../workflow/config/config.plotting.yaml
   :language: yaml

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/plotting.csv
```
