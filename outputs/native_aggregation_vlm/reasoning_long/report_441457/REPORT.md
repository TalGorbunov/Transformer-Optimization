# Cosmos: exploratory4096-token reasoning extension

The existing direct32 and original512 outcomes remain fixed. New128/512/4096 endpoints share one4096-cap trace per scene.

| Policy | N16 exact | N32/N64 exact | OOD completed | OOD parsed | OOD truncated | OOD MAE (parsed n) |
|---|---:|---:|---:|---:|---:|---:|
| direct32 | 4/18 | 4/36 | 36/36 | 36/36 | 0/36 | 5.139 (36) |
| original_reason512 | 4/18 | 1/36 | 8/36 | 8/36 | 28/36 | 7.000 (8) |
| extended_reason128 | 0/18 | 0/36 | 0/36 | 0/36 | 36/36 | NA |
| extended_reason512 | 4/18 | 1/36 | 8/36 | 8/36 | 28/36 | 7.000 (8) |
| extended_reason4096 | 5/18 | 6/36 | 32/36 | 32/36 | 4/36 | 6.781 (32) |

Primary descriptive difference: +5.56pp, paired-anchor interval[+0.00,+13.89].5pp screen: True. N16 difference: +5.56pp.
Prefix mismatches against original512: 0/54. No scenes were removed or rerun.
Measured generation 1705.8s, total 1755.3s including load; 66888 generated tokens. No separate128/512 latency is inferred.

## Interpretation limits

- Adaptive budget extension on already inspected scenes, prompted by original Cosmos512 truncation28/36 OOD. This is exploratory and not fresh confirmation.
- Only the reasoning output cap changes to4096. Direct32 predictions are reused from the verified original frozen Cosmos run; no direct rerun, prompt search, model selection or new data.
- All128/512/4096 new endpoints come from one4096-cap trajectory. Earlier endpoints require native EOS by that budget and have no separately measured latency.
- Original512 outcomes remain reported. Raw-prefix disagreements are disclosed without dropping, rerunning or choosing examples; unequal prefixes limit interpretation as continuation of the old run.
- The official Cosmos guide recommends4096 or more output tokens. Native EOS can still arrive later than this cap; the assay retains NF4/bf16, greedy decoding and repetition_penalty1 deviations from the official BF16/sampling example.
- Whole-trace think/answer grammar and native EOS are mandatory. Every truncated/malformed output stays wrong; MAE uses its explicit parsed denominator.
- Within-Cosmos reasoning-policy minus existing direct exact is descriptive. System policy and output budget differ; this is not an isolated causal effect of token count or evidence of an improved aggregation method.
- Bootstrap resamples18 complete anchors, retaining N32/N64 pairs,10000 draws seed20260917. The5pp effect screen is descriptive; no seed-population or cross-model causal inference.
- A natural profile may stop before4096. Report the actually exercised cache length; declaring a4096 cap alone does not demonstrate4096 decode steps.
