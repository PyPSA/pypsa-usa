(config)=
# Configuration 

**This workflow is currently only being tested for the `western` interconnection wildcard.**

(network_cf)=
## Pre-set Configuration Options

The `network_configuration` option in the `config.yaml` file accepts 3 values: `pypsa-usa` , `ads2032`, and `breakthrough`. Each cooresponds to a different combiation of input datasources for the generators, demand data, and generation timeseries for renewable generators. The public version of the WECC ADS PCM does not include data on the transmission network, but does provide detailed information on generators. For this reason the WECC ADS generators are superimposed on the TAMU/BE network.

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

(clustering_cf)=
## Clustering

There have been issues in running operations-only simulations with clusters >50 for the WECC. Issue is currently being addressed.

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


<!-- .. literalinclude:: ../config/config.default.yaml
   :language: yaml
   :start-at: clustering:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/clustering.csv

.. note::
   ``feature:`` in ``simplify_network:``
   are only relevant if ``hac`` were chosen in ``algorithm``.

.. tip::
   use ``min`` in ``p_nom_max:`` for more `
   conservative assumptions. -->

(run_cf)=
## ``run``

It is common conduct to analyse energy system optimisation models for **multiple scenarios** for a variety of reasons,
e.g. assessing their sensitivity towards changing the temporal and/or geographical resolution or investigating how
investment changes as more ambitious greenhouse-gas emission reduction targets are applied.

The ``run`` section is used for running and storing scenarios with different configurations which are not covered by :ref:`wildcards`. It determines the path at which resources, networks and results are stored. Therefore the user can run different configurations within the same directory.

<!-- .. literalinclude:: ../config/config.default.yaml
   :language: yaml
   :start-at: run:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/run.csv -->

<!-- (scenario)=
## ``scenario`` -->

<!-- The ``scenario`` section is an extraordinary section of the config file
that is strongly connected to the :ref:`wildcards` and is designed to
facilitate running multiple scenarios through a single command

.. code:: bash

   # for electricity-only studies
   snakemake -call solve_elec_networks

   # for sector-coupling studies
   snakemake -call solve_sector_networks

For each wildcard, a **list of values** is provided. The rule
``solve_all_elec_networks`` will trigger the rules for creating
``results/networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc`` for **all
combinations** of the provided wildcard values as defined by Python's
`itertools.product(...)
<https://docs.python.org/2/library/itertools.html#itertools.product>`_ function
that snakemake's `expand(...) function
<https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#targets>`_
uses.

An exemplary dependency graph (starting from the simplification rules) then looks like this:

.. image:: img/scenarios.png

.. literalinclude:: ../config/config.default.yaml
   :language: yaml
   :start-at: scenario:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/scenario.csv -->


<!-- (snapshots_cf)=
## ``snapshots`` -->

<!-- Specifies the temporal range to build an energy system model for as arguments to `pandas.date_range <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html>`_

.. literalinclude:: ../config/config.default.yaml
   :language: yaml
   :start-at: snapshots:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/snapshots.csv -->