(solve-network)
# Solve Network

(prepare)=
## Rule `prepare_network`
```{eval-rst}
Prepare PyPSA network for solving according to :ref:`opts` and :ref:`ll`, such
as.

- adding an annual **limit** of carbon-dioxide emissions,
- adding an exogenous **price** per tonne emissions of carbon-dioxide (or other kinds),
- setting an **N-1 security margin** factor for transmission line capacities,
- specifying an expansion limit on the **cost** of transmission expansion,
- specifying an expansion limit on the **volume** of transmission expansion, and
- reducing the **temporal** resolution by averaging over multiple hours or segmenting time series into chunks of varying lengths using ``tsam``.

**Relevant Settings**

.. code:: yaml

    costs:
        year:
        version:
        fill_values:
        emission_prices:
        marginal_cost:
        capital_cost:

    electricity:
        co2limit:
        max_hours:

.. seealso::
    Documentation of the configuration file ``config/config.yaml`` at
    :ref:`costs_cf`, :ref:`electricity_cf`

**Inputs**

- ``resources/costs.csv``: The database of cost assumptions for all included technologies for specific years from various sources; e.g. discount rate, lifetime, investment (CAPEX), fixed operation and maintenance (FOM), variable operation and maintenance (VOM), fuel costs, efficiency, carbon-dioxide intensity.
- ``networks/elec_s{simpl}_{clusters}.nc``: confer :ref:`cluster`

**Outputs**

- ``networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc``: Complete PyPSA network that will be handed to the ``solve_network`` rule.

**Description**

.. tip::
    The rule :mod:`prepare_elec_networks` runs
    for all ``scenario`` s in the configuration file
    the rule :mod:`prepare_network`.
```

(solve)=
## Rule `solve_network`
**Directly uses PyPSA-Eur Implementation**

```{eval-rst}
Solves optimal operation and capacity for a network with the option to
iteratively optimize while updating line reactances.

This script is used for optimizing the electrical network as well as the
sector coupled network.

**Description**

Total annual system costs are minimised with PyPSA. The full formulation of the
linear optimal power flow (plus investment planning is provided in the)
`documentation of PyPSA <https://pypsa.readthedocs.io/en/latest/optimal_power_flow.html#linear-optimal-power-flow>`_.

The optimization is based on the :func:`network.optimize` function.
Additionally, some extra constraints specified in :mod:`solve_network` are added.

.. note::

    The rules ``solve_elec_networks`` and ``solve_sector_networks`` run
    the workflow for all scenarios in the configuration file (``scenario:``)
    based on the rule :mod:`solve_network`.
```
