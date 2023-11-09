(config)=
# Configuration 

**This workflow is currently only being tested for the `western` interconnection wildcard.**

(network_cf)=
## Pre-set Configuration Options

## ``network_configuration``

The `network_configuration` option accepts 3 values: `pypsa-usa` , `ads2032`, and `breakthrough`. Each cooresponds to a different combiation of input datasources for the generators, demand data, and generation timeseries for renewable generators. The public version of the WECC ADS PCM does not include data on the transmission network, but does provide detailed information on generators. For this reason the WECC ADS generators are superimposed on the TAMU/BE network.

| Configuration Options: | PyPSA-USA | ADS2032(lite) | Breakthrough |
|:----------:|:----------:|:----------:|:----------:|
| Transmission | TAMU/BE | TAMU/BE | TAMU/BE |
| Thermal Generators | EIA860 | WECC-ADS | BE |
| Renewable Time-Series | Atlite | WECC-ADS | Atlite |
| Hydro Time-Series | Breakthrough (temp) | WECC-ADS | Breakthrough |
| Demand | EIA930 | WECC-ADS | Breakthrough |
| Years Supported | 2019 (soon 2017-2023) | 2032 | 2016 |
| Interconnections Supported | WECC (soon US) | WECC | WECC (soon US)|
| Purpose[^+] | CEM, PCS | PCS | PCS |

[^+]: CEM = Capacity Expansion Model, PCS = Production Cost Simulation


(run_cf)=
## ``run``

It is common conduct to analyse energy system optimisation models for **multiple scenarios** for a variety of reasons,
e.g. assessing their sensitivity towards changing the temporal and/or geographical resolution or investigating how
investment changes as more ambitious greenhouse-gas emission reduction targets are applied.

The ``run`` section is used for running and storing scenarios with different configurations which are not covered by :ref:`wildcards`. It determines the path at which resources, networks and results are stored. Therefore the user can run different configurations within the same directory.

```{eval-rst}  
.. literalinclude:: ../../workflow/config/config.default.yaml
   :language: yaml
   :start-at: run:
   :end-before: enable:


.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/run.csv
```

(snapshots_cf)=
## ``snapshots``

Specifies the temporal range to build an energy system model for as arguments to `(pandas.date_range)[https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html]`

```{eval-rst}  
.. literalinclude:: ../../workflow/config/config.default.yaml
   :language: yaml
   :start-at: snapshots:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/snapshots.csv
```


(electricity_cf)=
## ``electricity``

Specifies the types of generators that are included in the network, which are extendable, and the CO2 base for which the optimized reduction is relative to.

```{eval-rst}  
.. literalinclude:: ../../workflow/config/config.default.yaml
   :language: yaml
   :start-at: electricity:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/electricity.csv
```

(renewable_cf)=
## ``renewable``
### ``solar``
```{eval-rst}  
.. literalinclude:: ../../workflow/config/config.default.yaml
   :language: yaml
   :start-at: solar:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/solar.csv
```

### ``onwind``
```{eval-rst}  
.. literalinclude:: ../../workflow/config/config.default.yaml
   :language: yaml
   :start-at: onwind:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/onwind.csv
```

(clustering_cf)=
## ``clustering``

Minimum Number of clusters:
```
Eastern: TBD
Western: 30
Texas: TBD
```

Maximum Number of clusters:
```
Eastern: 35047
Western: 4786
Texas: 1250
```

```{eval-rst}  
.. literalinclude:: ../../workflow/config/config.default.yaml
   :language: yaml
   :start-at: clustering:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/clustering.csv

.. note::
   ``feature:`` in ``simplify_network:`` are only relevant if ``hac`` were chosen in ``algorithm``.
   
   - Use `focus_weights` to specify the proportion of cluster nodes to be attributed to a given zone given by the `aggregation_zone` configuration.

.. tip::
   use ``min`` in ``p_nom_max:`` for more `
   conservative assumptions.
```


(solving_cf)=
## ``solving``

```{eval-rst}  
.. literalinclude:: ../../workflow/config/config.default.yaml
   :language: yaml
   :start-at: solving:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/solving.csv
```


