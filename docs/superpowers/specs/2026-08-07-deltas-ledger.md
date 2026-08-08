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

Open findings NOT waived (require fixes or further adjudication before the
harness can go green): demand total −6.3% / 95 missing load buses (candidate),
hydro generator attachment (candidate ~129 MW vs anchor 12,977 MW solved),
StorageUnit set differences, generator plant-id vs bus-carrier naming
convention, solved objective 0.48%.
