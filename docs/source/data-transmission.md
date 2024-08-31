(data-transmission)=
# Transmission

## Transmission Networks

PyPSA-USA offers a unique capability by integrating two distinct transmission networks: the ReEDS NARIS-derived zonal network and the Breakthrough Energy - Texas A&M University (BE-TAMU) synthetic nodal network. These networks can be used independently or together, providing flexibility in how transmission constraints and network dynamics are modeled.

### ReEDS NARIS-derived Zonal Network

The **ReEDS NARIS-derived zonal network** is based on the ReEDS capacity expansion model, which divides the continental US into 137 zones. This network is designed to respect state boundaries and can be mapped to balancing authorities, NERC regions, and RTOs/ISOs. Key features of this network include:

- **Zonal Representation**: Focuses on larger, aggregated zones rather than detailed, node-level data.
- **Interface Transmission Limits (ITLs)**: Enforces interregional zonal transmission constraints, which are critical for modeling power flows between different regions.
- **Suitable for Capacity Expansion**: The zonal network's lower spatial resolution is well-suited for capacity expansion planning, as it simplifies computational requirements.

However, the zonal network's lower resolution can be a limitation when studying the impacts of intra-regional transmission constraints, especially in areas where transmission-constrained renewable resources are concentrated within a single zone.

### BE-TAMU Synthetic Nodal Network

The **BE-TAMU synthetic nodal network** offers a high-resolution representation of the US power system, specifically designed for operational simulations. This network includes:

- **High Spatial Resolution**: Comprising 82,549 buses, 41,561 substations, 83,497 AC lines, and 17 HVDC lines, it provides a detailed view of the transmission grid.
- **Operational Detail**: Captures the dynamics of power flows and network operations with a finer granularity than the zonal network.
- **Not Geographically or Topologically Precise**: While representative, the BE-TAMU network is synthetic and not precisely aligned with the actual US transmission network.

Due to its high resolution, the BE-TAMU network is computationally intensive, making it less suitable for capacity expansion planning without adjustments.

### Combined Network Configuration

To leverage the strengths of both networks, PyPSA-USA offers a combined transmission configuration. In this setup:

- **Zonal Constraints on Nodal Topology**: The zonal ITLs from the ReEDS network are superimposed on the clustered BE-TAMU nodal network, allowing for detailed nodal analysis while respecting interregional constraints.
- **Flexible Node Clustering**: Users can select the number of nodes in the clustered BE-TAMU network, enabling a balance between computational efficiency and network detail.
- **DC Power Flow Model**: The combined network supports a DC power flow (DC PF) model, providing a realistic simulation of power flows within the high-resolution nodal network while enforcing the zonal ITLs.

This dual-network approach in PyPSA-USA allows for a more comprehensive analysis, combining the broad, strategic insights from the zonal network with the detailed operational dynamics captured by the nodal network.
