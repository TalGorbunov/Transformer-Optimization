# Fresh composition fails after perfect native training

Both fixed step6000 visual factor checkpoints fail their registered N16 fresh-data criteria. [Independent report443611](../../outputs/native_aggregation_vlm/identity_join_factor_fresh/evaluation/report_443611/REPORT.md) passed all540 trajectory and1240 emitted-prefix integrity/functional audits. The two GPU runs used340seconds each; the report used162CPU-seconds. No extra native heads were evaluated.

| Model | Familiar groups A | New groups B | New groups/questions C |
| --- | --- | --- | --- |
| Product |64/108 answers;15/36 triples|14/108;0/36|7/54;0/18|
| Additive |49/108 answers;11/36 triples|14/108;0/36|7/54;0/18|

The orientation split clarifies A: product scores51/54 on the original training orientation and13/54 on the opposite orientation; additive scores41/54 and8/54. Thus fresh realizations of fitted configurations transfer much better than reversed role assignments or new person groups. This is consistent with an orientation/configuration-dependent learned rule; it does not uniquely identify the internal shortcut.

Whole-answer and first-token correctness coincide. Every trajectory terminates, and all outputs are canonical names. The error is semantic answer choice, not truncation or an inability to emit the required name format. The report's shorter-than-target EOS statistic also counts wrong canonical names with fewer subword tokens; it should not be described as a separate decoding-format failure.

Neither checkpoint qualifies on A and B. N32/N64 evaluation of these checkpoints is stopped. The already rendered longer scenes remain untouched by model predictions. These failed fresh panels cannot serve as independent confirmation for a later model. A product advantage confined to panel A is not evidence of general aggregation or wider representational bandwidth.

The next study should test whether training support teaches a reusable rule while retaining the same feature-map/SUM/readout architecture. The current108 scenes comprise only18 families and a single orientation for each person-group/question cell. More varied training and balanced counterfactual assignments are distinct possible interventions; their data, coverage and compute must be specified before fitting. Expanded support changes which groups/questions are unseen and requires a new confirmation set. The separately prepared frozen ordinary joint-image reference remains useful for interpretation, but cannot reopen the failed length gate or isolate architecture from task adaptation.
