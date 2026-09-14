# V7 synthetic response surface

216 fixed synthetic first-token points; no fitting and no VLM forward. This is a post-hoc diagnostic, not a new efficacy evaluation.

| Seed | Question | Synthetic N16 first-token correct | Synthetic N64 first-token correct | Raw-zero padding |
|---:|---|---:|---:|---|
| 8 | How many frames show Daniel in the Bathroom? | 27/27 | 8/27 | True |
| 8 | How many frames show Daniel in the Bedroom? | 27/27 | 8/27 | True |
| 9 | How many frames show Daniel in the Bathroom? | 26/27 | 1/27 | True |
| 9 | How many frames show Daniel in the Bedroom? | 27/27 | 3/27 | True |

All 216 native FP16 raw-logit rows, messages, genuine training sums and pre/post-RMS carries are archived. Full-vocabulary predictions are separate from restricted numeral argmax diagnostics.

The native FP16 norm/head tensor hashes match the earlier actual-model capture. The four-row CPU-versus-GPU replay is descriptive and retained in analysis.json.

A failure on a fixed-K negative extension can be localized to the added negative message direction. If different counts survive in z but collapse in the native carry or acquire incorrect numeral ordering, the failure is downstream of that synthetic statistic. Neither observation proves an intrinsic model capacity limit.

Successful synthetic N64 decoding does not establish real N64 generalization: genuine later Step labels and new local features are absent here.

[Complete diagnostic and input provenance](analysis.json).
