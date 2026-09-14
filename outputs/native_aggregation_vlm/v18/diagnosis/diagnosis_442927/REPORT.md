# V18 saved-state diagnosis

All four fixed models and all272 fresh scenes per model are retained. Geometry uses only the original empty prefix. No model, head, fitting, or intervention was run.

| Model | N | Whole answers | First token | Closed positive gates | Open negative gates | Mean attenuation | Mean leakage | Mean projected P | Mean projected Z | Unfused→fused first-token transitions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| native_gate_s20 | 32 | 132/136 | 136/136 | 0/1088 | 0/3264 | 0.494407 | 0 | 358.929 | 0 | {'1->1': 8, '0->1': 128} |
| native_gate_s20 | 64 | 131/136 | 136/136 | 0/1088 | 0/7616 | 0.493932 | 0 | 358.952 | 0 | {'1->1': 8, '0->1': 128} |
| native_gate_s21 | 32 | 132/136 | 136/136 | 0/1088 | 0/3264 | 0.494407 | 0 | 402.091 | 0 | {'1->1': 8, '0->1': 128} |
| native_gate_s21 | 64 | 131/136 | 136/136 | 0/1088 | 0/7616 | 0.493932 | 0 | 402.116 | 0 | {'1->1': 8, '0->1': 128} |
| all_open_s20 | 32 | 0/136 | 6/136 | 0/1088 | 0/3264 | 0.494407 | 0 | 119.065 | 297.656 | {'1->0': 8, '0->0': 122, '0->1': 6} |
| all_open_s20 | 64 | 0/136 | 0/136 | 0/1088 | 0/7616 | 0.493932 | 0 | 119.049 | 694.514 | {'1->0': 8, '0->0': 128} |
| all_open_s21 | 32 | 0/136 | 20/136 | 0/1088 | 0/3264 | 0.494407 | 0 | 128.134 | 271.42 | {'1->0': 8, '0->0': 108, '0->1': 20} |
| all_open_s21 | 64 | 0/136 | 8/136 | 0/1088 | 0/7616 | 0.493932 | 0 | 128.134 | 633.313 | {'1->0': 8, '0->0': 120, '0->1': 8} |

Selection columns describe the original native gates in both arms. Vector columns use gates actually applied by that arm.

All544 family comparisons retain semantic parent mappings and changed Step/image identities. Maximum FP64 identity residuals: `{'selection': 0.0, 'message_partition': 0.0, 'paired_product': 0.0, 'paired_roundoff_reconciliation': 0.0, 'paired_query_parent_added_negative': 1.9184653865522705e-13}`.

Exact identities concern FP64 algebra. The separate saved FP32 multiplication/reduction residuals are descriptive.

If negative gates are closed, their branch messages are absent. Remaining errors can involve positive payloads or native decoding; these norms alone do not identify which intervention would repair them.

The original primary/practical decisions are copied unchanged. This analysis establishes neither task transfer nor composition with reasoning.

[All scenes](scenes.json) · [All paired families](families.json) · [Per-N/K and mandatory partition summaries](analysis.json) · [Input identities](input_bindings.json) · [Sources](source_hashes.json).
