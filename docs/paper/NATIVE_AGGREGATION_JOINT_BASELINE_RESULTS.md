# Frozen joint-image reference

The unadapted Qwen joint-image baseline also fails the fresh N16 identity join. Each input uses all16 canonical MMReD images in one native sequence, the exact same name question, and ordinary greedy generation capped at4 tokens.

| Model | A: familiar groups | B: new groups | C: new groups and questions |
|---|---:|---:|---:|
| Frozen joint |42/108|36/108|18/54|
| Fitted product |64/108|14/108|7/54|
| Fitted additive |49/108|14/108|7/54|

Entries are exact complete names including EOS on identical scenes. Frozen joint solves only1/36 complete A triples and0 complete B/C triples. The fitted models improve familiar-group answers but lose on new groups. None meets the predefined criteria. No general aggregation benefit is established.

The baseline is unadapted; it does not isolate training from architecture. N+1 parallel streams and one joint sequence also have different computation despite similar forward-call accounting. Profile+main used280GPU-seconds; six profile selected-query CPU head replays had exact argmaxes and maximum total variation3.62e-7. All270 main trajectories and639 native calls were independently audited. Total-job timing also includes different setup and capture costs, so it is not a clean architectural speed comparison.

The failed factor N16 screen remains closed before N32/N64. A new prospectively registered room-swap training-coverage study uses separate initialization and confirmation data.

[Independent report](../../outputs/native_aggregation_vlm/identity_join_joint_baseline/report_443643/REPORT.md) · [Full analysis](../../outputs/native_aggregation_vlm/identity_join_joint_baseline/report_443643/analysis.json)
