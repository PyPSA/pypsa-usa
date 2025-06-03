(spatial)=
# Spatial Configuration

## Configuring Spatial Scope

PyPSA-USA allows for flexible configuration of the spatial scope of your energy model, enabling you to define the geographical area and level of detail for your simulations. The **spatial scope** is determined by the `interconnect`, `model_topology` configuration settings.

- `{interconnect}` is used to select which of the three asynchronous interconnection to model. You can select `western`, `eastern`, `texas`, or `usa` to model the entire US.

```{eval-rst}
.. image:: _static/zones/interconnects.png
   :width: 600px
   :align: center
   :alt: Map of US Interconnections
```

- After selecting your `{interconnect}`, you can specify `model_topology: include:` to filter individual states or balancing authorities to be selected in your model.

### Example: Modeling California

To create a model that includes only California, you can specify the relevant ReEDS zone IDs (p8-11) as shown below. This will limit the spatial scope to the specified regions within California. Your interconnect could be set to `western` or `usa`.

```{eval-rst}
.. image:: _static/zones/reeds_zones.png
   :width: 600px
   :align: center
   :alt: Map of ReEDS Zones
```

```yaml
model_topology:
   include:
      reeds_zone: ['p8', 'p9', 'p10', 'p11']
```

Alternatively, you can use the code reeds_state: 'CA' option to achieve the same result by specifying the entire state.

In addition to filtering by `reeds_zone` and `reeds_state`, you can filter by `reeds_ba`, `trans_reg`, and `nerc_reg` shown graphically below.


```{eval-rst}
.. image:: _static/zones/reeds_trans_reg.png
   :width: 600px
   :align: center
   :alt: Map of ReEDS Transmission Regions
```

```{eval-rst}
.. image:: _static/zones/reeds_nerc_reg.png
   :width: 600px
   :align: center
   :alt: Map of NERC Regions
```

```{eval-rst}
.. image:: _static/zones/reeds_ba.png
   :width: 600px
   :align: center
   :alt: Map of Balancing Authorities
```

## Configuring Resource Resolution
PyPSA-USA allows you to independently configure the resolution of resource zones from the transmission network. You can control this using the simpl and clusters parameters in the configuration file.

For example, if you want a transmission network with 10 nodes and a resource model with 100 nodes, you would configure it as follows:

```yaml
scenario:
   clusters: [10m]
   simpl: [100]
```

This setup, using an `m` after the `clusters` wildcard, results in a model with 10 transmission nodes and 100 distinct renewable resource zones, allowing for more granular modeling of renewable resource distribution while keeping the transmission network simplified. If you use a `c` after the `clusters` wildcard, all conventional resources from the `simpl` step will not be clustered. If you input an `a` after the `clusters` wildcard, all resources will not be clustered beyond the `simpl` level.

## Configuring Transmission Resolution

### Transmission Network Selection
You can specify the transmission network you want to use by setting the `model_topology: transmission_network:` option. There are two available options:

- 'reeds': The ReEDS NARIS networks.
- 'tamu': The synthetic BE-TAMU nodal network.

When selecting between the three ReEDS NARIS networks, you will need to also specify the `model_topology: topological_boundaries:`. Currently you can set either `county` or `reeds_zone`. To use the FERC 1000 regions, you will need to use the custom network topologies described in the example below.

### Transmission Network Resolution

IF you are using the TAMU/BE network, you can flexibly set an arbitrary number of clusters between the min and max number of nodes. If using a ReEDS NARIS network, you need to specify the minimum number of clusters (nodes) for your modeled interconnection. The number of nodes for each zone is **detailed in the table below**.

If you're working with custom configurations, PyPSA-USA will notify you during the cluster_network stage, indicating the correct number of nodes to set in the clusters configuration.

```{eval-rst}
.. csv-table::
   :header-rows: 1
   :widths: 22,22,33
   :file: datatables/transmission_nodes.csv
```

#### Example: Meshed ReEDS NARIS WECC Topology

If you would like to mesh the three ReEDS NARIS networks you can do so by using the `model_topology: aggregate:` option. For instance, to create a model where California is represented at a county level resolution, but Non-CA WECC areas are represented at the FERC 1000 level, you would configure the model as follows:


```yaml
scenario:
   interconnect: [western]
   clusters: [87]
   simpl: [380] # can be set differently based on number of resource zones you'd like to keep


model_topology:
  transmission_network: 'reeds'
  topological_boundaries: 'county'
  interface_transmission_limits: false
  include:
    # nothing specified here since we are modeling the entire WECC
  aggregate:
    trans_grp: ['NorthernGrid_South', 'NorthernGrid_West', 'NorthernGrid_East', 'WestConnect_North','WestConnect_South']
```

This configuration will copper plate the Non-CA regions listed under `trans_grp`, effectively creating a copper-plate network where resources can be clustered and shared across the region. Using these custom aggregation requires information on the region memberships which you can find in `workflow/repo_data/ReEDS_Constraints/membership.csv`.
