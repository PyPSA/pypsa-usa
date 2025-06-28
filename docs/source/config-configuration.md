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

Planning horizons determines which year(s) of future demand forecast to use for your planning model. To build a multi-investment period model set multiple `planning_horizons:` years. The `foresight:` option specifies whether perfect foresight or myopic foresight optimization model is developed. In perfect foresight, a monolithic model is developed where all `planning_horizons` specified are optimized at once, e.g. future horizon values of costs and demand are incorporated into decisions made in earlier planning horizons. Myopic optimization solves each planning horizon sequentially, and passes the results forward.

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

(co2_cf)=
## `co2`

The `co2` section specifies whether the model may use underground storage to sequester captured CO2 or not. In case underground storage is specified, each node (composing the network) has a specific storage potential and a cost associated with it. The storage potential (in tonnes) is calculated by aggregating all the underlying storage potentials of the U.S. counties encompassed in the node's geographical area. Counties that are only partially covered by the node's geographical area have their potential fractionated accordingly. The storage cost (in $/tonne) is calculated by weighting the potential with the cost of each county encompassed. The dataset containing information about underground CO2 storage potentials and costs at a county level (and used in PyPSA-USA) was provided by Edna Calzado at The University of Texas (Austin), which was derived from the Roads to Removal project (https://roads2removal.org).

In addition, the section specifies whether the model may transport captured CO2 between nodes or not. In case transportation is specified, a network of CO2 pipelines is built based on the electricity grid layout represented in PyPSA-USA to determine where/how to build pipelines to connect nodes.

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-at: co2:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/co2.csv
```

(dac_cf)=
## `dac`

The `dac` section specifies whether the model should use Direct Air Capture (DAC) or not to capture emitted CO2. DAC may operate under different granularities (or scopes) and whether the model is based on sectors or not.

For a sector-less model:

   - When `granularity` is set to `node`, each node (composing the network) has a specific "air atmosphere" into which all the processes belonging to the node emit CO2. For each node, DAC is built to capture CO2 from its "air atmosphere".

   - When `granularity` is set to `state`, each U.S. state (represented in the network) has a specific "air atmosphere" into which all the processes of the nodes belonging to the state emit CO2. For each node, DAC is built to capture CO2 from the state "air atmosphere" it belongs to.

   - When `granularity` is set to `nation`, the model only has one single "air atmosphere" into which all the processes of all the nodes emit CO2. For each node, DAC is built to capture CO2 from this "air atmosphere".

For a sector-based model:

   - When `granularity` is set to `node`, each sector/node pair has a specific "air atmosphere" into which all the processes belonging to the sector/node pair emit CO2. For each sector/node pair, DAC is built to capture CO2 from its "air atmosphere".

   - When `granularity` is set to `state`, each sector/U.S. state pair has a specific "air atmosphere" into which all the processes of the nodes belonging to the state emit CO2. For each sector/node pair, DAC is built to capture CO2 from the sector/state pair's "air atmosphere" it belongs to.

   - Given that a `granularity` set to `nation` does not make sense in a sector-based model, it defaults to `node` in this case.

```{eval-rst}
.. literalinclude:: ../../workflow/repo_data/config/config.default.yaml
   :language: yaml
   :start-at: dac:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/dac.csv

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
Sector coupling studies are all under active development. More info to come!
```


(clustering_cf)=
## `clustering`


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
