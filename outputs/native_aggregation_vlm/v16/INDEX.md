# V16: rendered Step labels affect native aggregation

**Verified diagnostic; the research objective remains unmet.** On 17 previously evaluated paired families, changing only actual-image Step labels to cyclic 1–16 raises centered exact-bank N64 accuracy from 3/17 to 14/17 and from 5/17 to 14/17. Both N32 exact-bank models reach 17/17. Learned-mean models remain at 9/17 and 10/17 at N64; offset controls reach 3/17.

All 288 trajectories and 668 native queries passed independent rescoring. The 16 original-image sentinels have identical tokens and logits. Pixel staging retained all 1,632 occurrences with only the registered footer edits. Repeated visible ordinals change the original numbering grammar. The extra 24 reference streams remain; this is neither fresh confirmation nor a reasoning result.

- [Independent answers and transitions](step_wrap_study/report_442774/REPORT.md): CPU 442774 passed in 55 seconds.
- [Paired first-query message analysis](first_prefix_decomposition/decomposition_442775/REPORT.md): CPU 442775 passed in 64 seconds; all 668 current and 637 original prefixes audited.
- [Figure](figure_442776/native_vision_v16.png) and [PDF](figure_442776/native_vision_v16.pdf): CPU 442776 passed in 2 seconds; root visually inspected the PNG.
- [Execution ledger](execution.json), [pixel staging](data_staging/stage_442744/summary.json), and [retry plan](step_wrap_study/check_442755/summary.json).
- [Infrastructure failure and preserved source](infrastructure_442749/README.md).

Total cost: **2,888 allocated GPU-seconds**, comprising the failed node attempt (1,090) and healthy retry (1,798). Maximum concurrency was four GPUs. The resource cap was explicitly amended from 4,200 to 5,400 seconds before retry; the scientific protocol was unchanged. All 36 prepared archive hashes match the first CPU preparation. No model was fitted in V16.
