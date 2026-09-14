# Fixed-uniform identity-join result

**The fixed-uniform control failed both registered diagnostic criteria.** Independent report 443271 passed its computational, source, checkpoint, loss, native-output, and resource audits. This is a valid failure of the fixed training recipe.

| Partition | First token | Whole name + EOS | Complete families | Required |
|---|---:|---:|---:|---|
| Training | 36/108 | 36/108 | 0/18 six-context families | At least 103/108 for both scores and 16/18 families |
| Development | 18/54 | 18/54 | 0/18 three-context families | At least 49/54 for both scores and 16/18 families |

All 162 native answers completed and parsed, with zero truncations. Every planned answer and failure remained in its denominator. The evaluated endpoint was fixed at update 600; development neither selected nor extended training. See the [independent report](../../outputs/native_aggregation_vlm/identity_join_uniform/report_443271/REPORT.md) and [all outcomes](../../outputs/native_aggregation_vlm/identity_join_uniform/report_443271/outcomes.json).

The control reused the pilot's exact 108 training contexts, 54 intact N8/N16 pairs, initial tensors, persistent pair order, seed24, and native sequence-balanced CE. It fixed the sigmoid selector weights to zero and bias to 0.5, leaving 1,041,600 trainable coordinates and 97 frozen selector coordinates. No local labels, fitted parent checkpoint, or additional supervision entered optimization.

The independent audit confirmed exact 0.5 gates on valid items, zero padding, unchanged selector tensors, and all six initial trainable-gradient hashes matching the original CE-only initialization. All 600 training steps and all 324 native evaluation prefixes satisfied the fixed-gate contract; no first-query or later-query aggregate had every gate closed. The terminal 16-scene batch had sequence CE 0.528400; this is a batch loss, not a full-training-set score.

Fixing the gates prevented closure but did not make this recipe learn the association task. Gate closure is therefore unnecessary for failure here. This does not show that learned selection never helps, or identify whether payload representation, the native readout, or optimization is the remaining obstacle. Repeated question components and background accumulation remain possible contributors. A person/room probe error is not an information ceiling.

The prospectively conditional [readout diagnostic](NATIVE_AGGREGATION_READOUT_DIAGNOSTIC_PROPOSAL.md) is now eligible for separate review: a deliberately ideal post-SiLU answer code tests readout trainability, while a training-only geometry audit describes the uniform payload. This result does not itself release that diagnostic or another architecture. No fresh test, length extrapolation, general aggregation benefit, or reasoning composition was evaluated.

Allocated GPU cost was **293 seconds**: profile 443261 used **85**, and main 443264 used **208**; maximum concurrency was one, within the 990-second campaign cap. [Verified analysis](../../outputs/native_aggregation_vlm/identity_join_uniform/report_443271/analysis.json) SHA256: `c2176613b6b1ade6eb4c5a13e7ddfe23cb523b7d7709907a312f834bb5a07056`. [Report summary](../../outputs/native_aggregation_vlm/identity_join_uniform/report_443271/summary.json) SHA256: `1be7e5c8e5c598efd71d996035985e6f8f797dd749e977f4ac7817fc0df424f0`. [Execution record](../../outputs/native_aggregation_vlm/identity_join_uniform/execution.json).
