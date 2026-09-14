# Original MMReD fixed-checkpoint pilot

All400 validation and600 test rows were retained for each final arm. Scores require complete typed JSON and native EOS. These are exploratory historical-test results.

Promising mass-specific pilot heuristic: **False**. This label does not authorize another experiment or establish five-point noninferiority.

| Test panel | Ordinary | Normalized | Mass | Mass − ordinary, pp (95% CI) |
|---|---:|---:|---:|---:|
| N8 primary macro | 87.0% | 45.0% | 52.0% | -35.0 [-46.0, -24.0] |
| N16 primary macro | 72.0% | 42.0% | 35.0% | -37.0 [-49.0, -25.0] |
| N32 primary macro | 61.0% | 32.0% | 28.0% | -33.0 [-46.0, -19.0] |

Individual prospective conditions:

- ordinary_val_retrieval_ge_80_of_100: True
- ordinary_val_spend_together_ge_50_of_100: True
- ordinary_val_where_spend_ge_50_of_100: True
- mass_N32_primary_ge_60_percent: False
- mass_N32_primary_gain_ordinary_ge_10pp: False
- mass_N32_primary_gain_normalized_ge_5pp: False
- primary_mass_ordinary_CI_lower_positive: False
- N32_spend_together_mass_ordinary_nonnegative: False
- N32_where_spend_mass_ordinary_nonnegative: False
- N8_primary_mass_ordinary_drop_le_5pp: False
- N16_primary_mass_ordinary_drop_le_5pp: False
- N32_char_at_frame_mass_ordinary_drop_le_5pp: False
- N32_steps_in_room_mass_ordinary_drop_le_5pp: False

All30 task/macro analyses, every arm/contrast interval, answer-prior/question strata and unrounded replicate files are bound in analysis.json and artifacts.json.
The ordinary validation count-control50/100 reference is descriptive: True.
Native cold timings include measurement/capture/publication overhead. Shared feature construction is counted once in the experiment and once per standalone arm; timings are not production latency or FLOPs.
The pilot uses one fit seed and historically used tests. New-world/two-seed confirmation, relevant-load controls and plain/reasoning replication remain required.

The original resource-failed array444052 is preserved. Each arm combines its three unchanged original outcomes with997 continuation outcomes. Original allocations totaled98 GPU-seconds; new and cumulative allocations are disclosed separately in costs.json. Statistical calculations and thresholds are the frozen originals.
