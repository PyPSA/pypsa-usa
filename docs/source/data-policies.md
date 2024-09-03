(data-policies)=
# State and Federal Policy

## Policy Constraints

### Integration with ReEDS

PyPSA-USA integrates with the ReEDS capacity expansion model developed by NREL to incorporate data on regional and federal policies. This integration allows for the modeling of various policy-driven constraints that guide the decarbonization process.

### Implemented Policy Constraints

PyPSA-USA currently supports several key policy constraints, including:

- **Planning Reserve Margins**: Constrains capacity to meet a reserve margin above peak demand.
- **Clean Energy Standards (CES)**: Mandates the proportion of electricity generation that must come from clean energy sources.
- **Renewable Portfolio Standards (RPS)**: Requires a specific percentage of electricity generation to come from renewable sources.
- **Technology Capacity Targets**: Sets specific capacity expansion or retirement goals for certain technologies, such as wind, solar, or nuclear.
- **Emissions Constraints**: Limits the total emissions allowed within a region, with options to penalize imports by user-defined emissions factors.

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
