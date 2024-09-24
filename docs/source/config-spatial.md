# Spatial Configuration

## Configuring Spatial Scope

PyPSA-USA allows for flexible configuration of the spatial scope of your energy model, enabling you to define the geographical area and level of detail for your simulations.

- The **spatial scope** is determined by the `{interconnect}` wildcard and the `model_topology` configuration setting.
- You can use `{interconnect}` to model entire interconnections independently.
- Alternatively, the `model_topology: include:` option allows you to filter individual states or balancing authorities to be included in your model.

### Example: Modeling California

To create a model that includes only California, you can specify the relevant ReEDS zone IDs (p8-11) as shown below. This will limit the spatial scope to the specified regions within California.

```yaml
model_topology:
   include:
      reeds_zone: ['p8', 'p9', 'p10', 'p11']
```

Alternatively, you can use the reeds_state: 'CA' option to achieve the same result by specifying the entire state.

<!-- ### Custom Aggregation Example
If you'd like to create custom nodal aggregations, you can use the model_topology: aggregate: option. For instance, to cluster Arizona, Nevada, and New Mexico into a single region (WECC_SW), you would configure the model as follows:


```yaml
model_topology:
   aggregate:
      reeds_state:
         WECC_SW : ['AZ', 'NV', 'NM']
```

This configuration will dissolve the borders between these states, effectively creating a copper-plate network where resources can be clustered and shared across the region. -->

## Configuring Spatial Resolution

PyPSA-USA gives you control over the spatial resolution of your transmission and resource networks, allowing for detailed or aggregated views depending on your modeling needs.

### Transmission Network Selection
You can specify the transmission network you want to use by setting the model_topology: transmission_network: option. There are two available options:

- 'tamu': The synthetic BE-TAMU nodal network.
- 'reeds': The ReEDS zonal network.

### Configuring Node Clusters
When using the ReEDS network, you need to specify the number of clusters (nodes) for your modeled interconnection. The number of nodes for each zone is **detailed in the table below**. If you're working with custom configurations, PyPSA-USA will notify you during the cluster_network stage, indicating the correct number of nodes to set in the clusters configuration.

### Resource Group Resolution
PyPSA-USA allows you to independently configure the resolution of resource zones from the transmission network. You can control this using the simpl and clusters parameters in the configuration file.

For example, if you want a transmission network with 10 nodes and a resource model with 100 nodes, you would configure it as follows:

```yaml
scenario:
   clusters: [10m]
   simpl: [100]
```
This setup results in a model with 10 transmission nodes and 100 distinct resource zones, allowing for more granular modeling of renewable resource distribution while keeping the transmission network simplified.


```{eval-rst}
.. csv-table::
   :header-rows: 1
   :widths: 22,22,33
   :file: datatables/transmission_nodes.csv
```
