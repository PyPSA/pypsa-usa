(data-policies)=
# State and Federal Policy

```{note}
This page describes the policy data shipped with PyPSA-USA. The mathematical formulation of
each policy constraint is documented on the [model constraints](./model-constraints.md) page.
```

## Policy Constraints

### Integration with ReEDS

PyPSA-USA integrates with the ReEDS capacity expansion model developed by NREL to incorporate data on regional and federal policies. This integration allows for the modeling of various policy-driven constraints that guide the decarbonization process.

### Policy Data

Default policy inputs live in `workflow/config/policy_constraints/`; each file can be used
as shipped or replaced with custom entries to explore new policy pathways.

- **Renewable Portfolio Standards (RPS)**: State RPS compliance trajectories derived from the
  NREL ReEDS model inputs (`reeds/rps_fraction.csv`), covering roughly 30 states with annual
  fractions from 2010 through 2050, including solar- and wind-specific carve-outs. Custom
  region/carrier targets can be added through `portfolio_standards.csv`; compliance is grouped
  by REC trading zones.
- **Clean Energy Standards (CES)**: State CES trajectories from the ReEDS inputs
  (`reeds/ces_fraction.csv`) for 16 states (CA, CO, CT, IL, MA, ME, MI, MN, NC, NE, NM, NV,
  NY, OR, VA, WA), with annual fractions through 2050.
- **Energy Reserve Margins (ERM / SAFE)**: Annual planning reserve margins from the ReEDS
  inputs (`reeds/prm_annual.csv`) for ten NERC-style assessment regions (ERCOT, MISO, NPCC
  New England and New York, PJM, SERC, SPP, and the WECC CA/NWPP/SRSG regions) over 2020-2050,
  in static and ramped variants. A model-wide `SAFE_reservemargin` scalar and a per-region
  template (`SAFE_regional_prm.csv`) are also available.
- **Technology Capacity Targets (TCT)**: Minimum/maximum capacity targets by carrier, region,
  and planning horizon (`technology_capacity_targets.csv`). The shipped entries are derived
  from ReEDS: state nuclear no-build lists, forced retirements, and state storage mandates.
- **Emissions Limits**: Annual regional CO2 caps (`regional_Co2_limits.csv`) covering
  California (AB 32 / 2022 CARB Scoping Plan), Washington, Oregon, and the eleven RGGI states,
  with optional user-defined emissions factors to penalize imports.

### Flexible Policy Horizons and Geographic Scope Enforcements

Each of these constraints can be defined for different investment horizons (e.g., 2030, 2040, 2050) and applied uniquely across various geographical levels:

- **State-Level**
- **Balancing Areas (BAs)**
- **Interconnects**
- **National Level**

Users have the flexibility to apply the policy constraints defined by ReEDS or to implement custom policy constraints, allowing for the exploration of new policy pathways and scenarios.


### Data
```{eval-rst}
.. csv-table::
   :header-rows: 1
   :widths: 22,22,22,22
   :file: datatables/policies.csv
```
