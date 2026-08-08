# Deltas Ledger — v1-epic vs anchor (upstream/develop e7f8bd70)

One row per accepted result-difference between the candidate and the anchor
under the Tier C equivalence harness (CA prong 1, `config.equivalence.yaml`).
Every row must have a matching machine-readable waiver in
`tests/equivalence/waivers.yaml`, and vice versa. **Sign-off column:**
entries marked *provisional* were adjudicated autonomously during the
2026-08-07 harness bringup under the user's standing "keep moving" mandate
and await the user's countersignature.

| ID | Stage(s) | Delta | Root cause | Why accepted | Sign-off |
|----|----------|-------|------------|--------------|----------|
| DL-1 | assembled_substation_network | `Line.capital_cost` differs ≤0.2% on 2810/2811 lines (max $19/MW-yr) | Stage-ordering artifact: anchor prices lines pre-aggregation on base-network sum-of-segment haversine lengths (1.00112× endpoint); candidate prices post-aggregation on endpoint lengths. Same formula, same $/MW-km, same factor (after the double-`length_factor` fix, commit bd69126e). | Clustered-stage transmission (ITL links) is recomputed identically on both branches — verified equal to full precision. The assembled-stage residue never reaches results in any config: line-preserving configs re-derive costs downstream of lengths that themselves agree. | provisional (Claude, 2026-08-07) |
| DL-2 | assembled_substation_network | `Link.capital_cost` differs on the 2 DC links (15_fwd/15_rev: 10894.9 vs 9244.1) | Same stage-ordering artifact as DL-1: anchor cost computed at base stage where the DC link length was the shorter pre-aggregation value. | Same masking argument as DL-1; clustered ITL links identical. | provisional (Claude, 2026-08-07) |
| DL-3 | all network stages | `Carrier.color` differs (onwind, 4hr_battery_storage) | v1-epic updated plotting palette entries. | Pure plotting metadata; no solver input. | provisional (Claude, 2026-08-07) |
| DL-4 | all network stages | `Bus.control`/`Bus.generator` slack assignment differs (slack at p9 vs p10); `Generator.control` PQ/Slack/'' bookkeeping differs | pypsa assigns slack per sub-network from the first generator encountered; generator ordering differs with the DAG reorder. | Power-flow bookkeeping only; LOPF/expansion solves ignore `control`. | provisional (Claude, 2026-08-07) |

| DL-5 | assembled + clustered | Pebbly Beach Generating Station Hybrid (EIA 6704, Catalina Island; 11.3 MW oil + 1.0 MW battery) assigned to bus 36973 (candidate) vs 37317 (anchor) | `match_plant_to_bus` nearest-bus matching now runs against 1,975 substation coordinates instead of 4,248 nodal coordinates; the offshore island plant's nearest neighbor flips. | Inherent to substation granularity; 12.3 MW on an island interconnection; zone-level totals unaffected (both buses in p11). Waiver lands after the hydro-fix rebuild isolates this as the sole [bus,carrier] residual. | provisional (Claude, 2026-08-07) |
| DL-6 | assembled + clustered | Generator/StorageUnit element NAMES differ (candidate per-plant ids at assembled, anchor pre-aggregated `{bus} {carrier}`); anchor-only `Generator_t.p_max_pu` columns for aggregate names | Stage-ordering: anchor aggregates one-ports/generators at simplify (pre-comparison stage); candidate keeps per-plant granularity until cluster_network. Attach code byte-identical. | Content is guarded by the UNWAIVED `Generator[bus,carrier]` / StorageUnit aggregate comparisons (battery 21,254.0 MW, PHS 3,978.4 MW match exactly). Names are not results. | provisional (Claude, 2026-08-07) |
| DL-7 | clustered stages | Small conventional-generator parameter diffs at p8–p11: `fuel_cost` ≤ 2.55 $/MWh (n=17), `efficiency` ≤ 0.0028 (n≤9), `marginal_cost` ≤ 2.55 $/MWh | Non-composable aggregation strategies (`mean` for fuel_cost/vom): anchor aggregates plants→sub→zone (mean of means), candidate plants→zone (single mean). Same plants, same strategy config, different composition order. | Inherent to the single-step aggregation of the refactor; bounded and small; capacity totals exact. Flagged for the user: switching those strategies to capacity-weighted means would make aggregation composable and eliminate the delta class. | provisional (Claude, 2026-08-07) |

Open findings NOT yet waived: demand total −6.3% (fix in progress:
`remove_transformers` Pd/LAF transfer), hydro attachment (fix in progress:
busmap remap in `attach_breakthrough_renewable_plants`), solved objective
0.48% (expected to close once demand+hydro fixes land), DL-5/DL-6 waiver
entries pending the post-fix rebuild.
