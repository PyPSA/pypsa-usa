(model-network-schema)=
# Network Schema

PyPSA-USA scripts annotate PyPSA components with custom columns — geographic
memberships, allocation factors, dataset identifiers — that downstream rules, policy
constraints, and plots rely on. The catalog below records, for every custom column:
its dtype, which script adds it, which scripts consume it, how it must be aggregated
when the network is clustered, and whether NaN values are legal and what they mean.

Two reading notes:

- PyPSA's built-in attributes are documented in the
  [PyPSA components reference](https://pypsa.readthedocs.io/en/latest/user-guide/components.html)
  and are not repeated here.
- The **aggregation strategy** column is load-bearing: a custom column that reaches a
  clustering step without a registered strategy falls back to PyPSA's `consense`,
  which raises if values disagree within a cluster. If you add a new column upstream
  of `aggregate_to_substations` or `cluster_network`, register it and record it here.

```{include} ../network-schema.md
:start-after: are not repeated here.
```
