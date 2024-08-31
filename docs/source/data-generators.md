(data-generators)=
# Generators

PyPSA-USA utilizes the Public Utility Data Liberation (PUDL) project database as the core source for generator and storage device data. The PUDL database aggregates and cleans data from various agencies, including the Energy Information Agency (EIA), Federal Energy Regulatory Commission (FERC), and the National Renewable Energy Laboratory (NREL). This integration supports reproducibility and ensures continuity as new reports are released.

### Generator Data Integration

PyPSA-USA integrates unit-level generator data from PUDL, which includes:

- **Heat Rates**
- **Plant Fuel Costs**
- **Seasonal Derating**
- **Power and Energy Capacities**
- **Generator Characteristics**
- **Fuel Types**

### Thermal Unit Commitment and Ramping Constraints

To model thermal unit commitment and ramping constraints, data from the WECC Anchor Data Set (ADS) is incorporated. This dataset is used by transmission and system planners across the WECC region and includes:

- **Start-up and Shut-down Costs**
- **Minimum Up and Down Time**
- **Ramp Limits**
- **Piecewise-linear Heat-rate Curves** (for matched thermal plants in the WECC)

For plants outside the WECC, and for internal plants missing data, PyPSA-USA imputes values using capacity-weighted averages by technology type. Although PyPSA can integrate part-load efficiency reductions, the current implementation uses single-point efficiencies, with plans to explore part-load impacts in future work.

### Generator Clustering

As part of the network clustering algorithm (see Section 2.1), generators are clustered by technology type at each bus. All technology types are imported from the EIA860 dataset via PUDL, and the plants are categorized into 11 types:

- Combined Cycle Generation Turbines
- Open Cycle Generation Turbines
- Coal
- Oil
- Geothermal
- Onshore Wind
- Fixed-bottom Offshore Wind
- Floating Offshore Wind
- Solar
- Biomass
- Battery Energy Storage Systems (BESS)

The data sources for generator and storage data are summarized in Table X.
