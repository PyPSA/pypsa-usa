(data-policies)=
# State and Federal Policy

## Policy Constraints

The rapid decarbonization of the electricity system has been accelerated by declining costs of renewable and clean technologies, as well as by state, regional, and national regulations. These regulations drive investments in decarbonization, ensuring resource adequacy and reliability across overlapping regions governed by local, state, and federal authorities.

### Integration with ReEDS

PyPSA-USA integrates with the ReEDS capacity expansion model maintained by NREL to incorporate data on regional and federal policies. This integration allows for the modeling of various policy-driven constraints that guide the decarbonization process.

### Implemented Policy Constraints

PyPSA-USA currently supports several key policy constraints, including:

- **Planning Reserve Margins**: Ensures sufficient capacity to meet peak demand, considering regional reliability standards.
- **Clean Energy Standards (CES)**: Mandates the proportion of electricity generation that must come from clean energy sources.
- **Renewable Portfolio Standards (RPS)**: Requires a specific percentage of electricity generation to come from renewable sources.
- **Technology Capacity Targets**: Sets specific capacity goals for certain technologies, such as wind, solar, or nuclear.
- **Emissions Constraints**: Limits the total emissions allowed within a region, supporting the transition to lower-carbon energy systems.

### Flexible Policy Horizons and Customization

Each of these constraints can be defined for different investment horizons (e.g., 2030, 2040, 2050) and applied uniquely across various geographical levels:

- **State-Level**
- **Balancing Areas (BAs)**
- **Interconnects**
- **National Level**

Users have the flexibility to apply the policy constraints defined by ReEDS or to implement custom policy constraints, allowing for the exploration of new policy pathways and scenarios.
