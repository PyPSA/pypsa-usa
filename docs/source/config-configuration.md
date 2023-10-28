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

<!-- It is common conduct to analyse energy system optimisation models for **multiple scenarios** for a variety of reasons,
e.g. assessing their sensitivity towards changing the temporal and/or geographical resolution or investigating how
investment changes as more ambitious greenhouse-gas emission reduction targets are applied.

The ``run`` section is used for running and storing scenarios with different configurations which are not covered by :ref:`wildcards`. It determines the path at which resources, networks and results are stored. Therefore the user can run different configurations within the same directory. If a run with a non-empty name should use cutouts shared across runs, set ``shared_cutouts`` to `true`.

.. literalinclude:: ../config/config.default.yaml
   :language: yaml
   :start-at: run:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/run.csv -->


(foresight_cf)=
## ``foresight``

<!-- .. literalinclude:: ../config/config.default.yaml
   :language: yaml
   :start-at: foresight:
   :end-at: foresight:

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/foresight.csv

.. note::
    If you use myopic or perfect foresight, the planning horizon in
    :ref:`planning_horizons` in scenario has to be set. -->

(scenario)=
## ``scenario``

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

.. _countries:

(snapshots_cf)=
## ``snapshots``

<!-- Specifies the temporal range to build an energy system model for as arguments to `pandas.date_range <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html>`_

.. literalinclude:: ../config/config.default.yaml
   :language: yaml
   :start-at: snapshots:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/snapshots.csv -->

(enable_cf)=
## ``enable``

<!-- Switches for some rules and optional features.

.. literalinclude:: ../config/config.default.yaml
   :language: yaml
   :start-at: enable:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/enable.csv -->

(CO2_budget_cf)=
## ``co2 budget``

<!-- .. literalinclude:: ../config/config.default.yaml
   :language: yaml
   :start-at: co2_budget:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/co2_budget.csv

.. note::
    this parameter is over-ridden if ``CO2Lx`` or ``cb`` is set in
    sector_opts. -->

(electricity_cf)=
## ``electricity``

<!-- .. literalinclude:: ../config/config.default.yaml
   :language: yaml
   :start-at: electricity:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/electricity.csv -->

(atlite_cf)=
## ``atlite``

<!-- Define and specify the ``atlite.Cutout`` used for calculating renewable potentials and time-series. All options except for ``features`` are directly used as `cutout parameters <https://atlite.readthedocs.io/en/latest/ref_api.html#cutout>`_.

.. literalinclude:: ../config/config.default.yaml
   :language: yaml
   :start-at: atlite:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/atlite.csv -->

(renewable_cf)=
## ``renewable``

<!-- ``onwind``
----------

.. literalinclude:: ../config/config.default.yaml
   :language: yaml
   :start-at: renewable:
   :end-before:   offwind-ac:

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/onwind.csv

.. note::
   Notes on ``capacity_per_sqkm``. ScholzPhd Tab 4.3.1: 10MW/km^2 and assuming 30% fraction of the already restricted
   area is available for installation of wind generators due to competing land use and likely public
   acceptance issues.

.. note::
   The default choice for corine ``grid_codes`` was based on Scholz, Y. (2012). Renewable energy based electricity supply at low costs
   development of the REMix model and application for Europe. ( p.42 / p.28)

``offwind-ac``
--------------

.. literalinclude:: ../config/config.default.yaml
   :language: yaml
   :start-at:   offwind-ac:
   :end-before:   offwind-dc:

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/offwind-ac.csv

.. note::
   Notes on ``capacity_per_sqkm``. ScholzPhd Tab 4.3.1: 10MW/km^2 and assuming 20% fraction of the already restricted
   area is available for installation of wind generators due to competing land use and likely public
   acceptance issues.

.. note::
   Notes on ``correction_factor``. Correction due to proxy for wake losses
   from 10.1016/j.energy.2018.08.153
   until done more rigorously in #153

``offwind-dc``
---------------

.. literalinclude:: ../config/config.default.yaml
   :language: yaml
   :start-at:   offwind-dc:
   :end-before:   solar:

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/offwind-dc.csv

.. note::
   both ``offwind-ac`` and ``offwind-dc`` have the same assumption on
   ``capacity_per_sqkm`` and ``correction_factor``.

``solar``
---------------

.. literalinclude:: ../config/config.default.yaml
   :language: yaml
   :start-at:   solar:
   :end-before:   hydro:

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/solar.csv

.. note::
   Notes on ``capacity_per_sqkm``. ScholzPhd Tab 4.3.1: 170 MW/km^2 and assuming 1% of the area can be used for solar PV panels.
   Correction factor determined by comparing uncorrected area-weighted full-load hours to those
   published in Supplementary Data to Pietzcker, Robert Carl, et al. "Using the sun to decarbonize the power
   sector -- The economic potential of photovoltaics and concentrating solar
   power." Applied Energy 135 (2014): 704-720.
   This correction factor of 0.854337 may be in order if using reanalysis data.
   for discussion refer to this <issue https://github.com/PyPSA/pypsa-eur/issues/285>

``hydro``
---------------

.. literalinclude:: ../config/config.default.yaml
   :language: yaml
   :start-at:   hydro:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/hydro.csv -->

(conventional_cf)=
## ``conventional``

<!-- Define additional generator attribute for conventional carrier types. If a
scalar value is given it is applied to all generators. However if a string
starting with "data/" is given, the value is interpreted as a path to a csv file
with country specific values. Then, the values are read in and applied to all
generators of the given carrier in the given country. Note that the value(s)
overwrite the existing values.

.. literalinclude:: ../config/config.default.yaml
   :language: yaml
   :start-at:   conventional:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/conventional.csv -->

(lines_cf)=
## ``lines``

<!-- .. literalinclude:: ../config/config.default.yaml
   :language: yaml
   :start-at: lines:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/lines.csv -->

(links_cf)=
## ``links``

<!-- .. literalinclude:: ../config/config.default.yaml
   :language: yaml
   :start-at: links:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/links.csv -->

(transformers_cf)=
## ``transformers``

<!-- .. literalinclude:: ../config/config.default.yaml
   :language: yaml
   :start-at: transformers:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/transformers.csv -->

(load_cf)=
## ``load``

<!-- .. literalinclude:: ../config/config.default.yaml
   :language: yaml
   :start-after:   type:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/load.csv -->

(energy_cf)=
## ``energy``

<!-- .. note::
   Only used for sector-coupling studies.

.. literalinclude:: ../config/config.default.yaml
   :language: yaml
   :start-at: energy:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/energy.csv -->

(biomass_cf)=
## ``biomass``

<!-- .. note::
   Only used for sector-coupling studies.

.. literalinclude:: ../config/config.default.yaml
   :language: yaml
   :start-at: biomass:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/biomass.csv

The list of available biomass is given by the category in `ENSPRESO_BIOMASS <https://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/ENSPRESO/ENSPRESO_BIOMASS.xlsx>`_, namely:

- Agricultural waste
- Manure solid, liquid
- Residues from landscape care
- Bioethanol barley, wheat, grain maize, oats, other cereals and rye
- Sugar from sugar beet
- Miscanthus, switchgrass, RCG
- Willow
- Poplar
- Sunflower, soya seed
- Rape seed
- Fuelwood residues
- FuelwoodRW
- C&P_RW
- Secondary Forestry residues - woodchips
- Sawdust
- Municipal waste
- Sludge -->

(solar_thermal_cf)=
## ``solar_thermal``

<!-- .. note::
   Only used for sector-coupling studies.

.. literalinclude:: ../config/config.default.yaml
   :language: yaml
   :start-at: solar_thermal:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/solar-thermal.csv -->

(existing_caopacities_cf)=
## ``existing_capacities``

<!-- .. note::
   Only used for sector-coupling studies. The value for grouping years are only used in myopic or perfect foresight scenarios.

.. literalinclude:: ../config/config.default.yaml
   :language: yaml
   :start-at: existing_capacities:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/existing_capacities.csv -->


(sector_cf)=
##``sector``
<!-- .. note::
   Only used for sector-coupling studies.

.. literalinclude:: ../config/config.default.yaml
   :language: yaml
   :start-at: sector:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/sector.csv -->

(industry_cf)=
## ``industry``

<!-- .. note::
   Only used for sector-coupling studies.

.. literalinclude:: ../config/config.default.yaml
   :language: yaml
   :start-at: industry:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/industry.csv -->

(costs_cf)=
## ``costs``

<!-- .. literalinclude:: ../config/config.default.yaml
   :language: yaml
   :start-at: costs:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/costs.csv

.. note::
   ``rooftop_share:`` are based on the potentials, assuming
   (0.1 kW/m2 and 10 m2/person) -->

(solving_cf)=
## ``solving``

<!-- .. literalinclude:: ../config/config.default.yaml
   :language: yaml
   :start-at: solving:
   :end-before: # docs

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/solving.csv -->

(plotting_cf)=
## ``plotting``

<!-- .. warning::
   More comprehensive documentation for this segment will be released soon.

.. literalinclude:: ../config/config.default.yaml
   :language: yaml
   :start-at: plotting:

.. csv-table::
   :header-rows: 1
   :widths: 22,7,22,33
   :file: configtables/plotting.csv -->
