# Independent native local batching: software oracle passed

Ordinary independent Qwen batch rows preserved the frozen teacher's local binary judgments through the tested batch size of64. CPU441679 froze the inputs and sources; GPU441680 passed the unchanged computational and numerical checks. This establishes a usable local-computation building block on these inputs. It produces separate local outputs, not one aggregate count, and does not establish reasoning composition. The [canonical report](../../outputs/native_aggregation_vlm/parallel_local/run_441680/REPORT.md), [raw157 rows](../../outputs/native_aggregation_vlm/parallel_local/run_441680/rows.jsonl), and [summary](../../outputs/native_aggregation_vlm/parallel_local/run_441680/summary.json) retain all observations.

The frozen selection contains64 training image/question pairs: the first16 sorted teacher pair IDs in each of positive, character-only, room-only and neither, interleaved in that order. Each row uses the original complete N1 count/chat prompt, unchanged rendered image,392px processing and frozen Qwen2.5-VL7B NF4/bfloat16/SDPA. Four fresh serial references cover one pair per category. Nested batches1,8,16,64 and reverse64 add153 responses, giving157 total across nine native VLM/visual calls. There is no learned branch, prompt search or outcome filtering.

Every conditional0/1 prediction matched the archived teacher. The maximum absolute conditional-p1 difference was0.01479939 and maximum numeric-mass difference0.000127340, both below the registered0.02 limits. All157 also matched the training audit truth, a descriptive observation rather than an extra eligibility criterion. Fresh serial0/1 logits and log normalizers matched their archived values exactly; the17 overlapping batch-versus-fresh-serial comparisons passed the same descriptive bounds. Forward64 and reverse64 matched exactly for all64 pairs on the stored0/1 logits, log normalizer, three grouped probabilities and top1 ID.

| Case | Responses | Forward seconds | Maximum conditional-p1 drift | Maximum numeric-mass drift |
|---|---:|---:|---:|---:|
| Batch1 |1|0.0827|0|0|
| Batch8 |8|0.2219|0.006093|0.000029754|
| Batch16 |16|0.3803|0.008322|0.000018399|
| Batch64 |64|1.3058|0.014799|0.000127340|
| Reverse64 |64|1.2836|0.014799|0.000127340|

CPU checks compared packed tensors with the ordinary batched processor. GPU execution verified unpadded token IDs, processed-image/grid ordering and all three actual native mRoPE axes, with one VLM/visual invocation per case and caching disabled. All selected prompts happened to contain265 tokens, so these GPU cases exercised **no nonzero left padding**. Variable-length padding remains covered by synthetic packing checks and implementation inspection, not this empirical result.

Batching uses substantial computation:64 rows took1.3058seconds, or20.4ms per row, versus a warm serial mean of84.7ms over three measured references. The first serial call took584ms. This is within-harness throughput evidence, not a matched64-example serial benchmark or constant-cost aggregation. Peak allocated GPU memory rose from5.79GiB for a serial case to8.19GiB for64 rows on a B200. The process recorded12.84seconds model loading and18.68seconds total; forward timing includes the position-capture hook, while image preparation was substantially staged on CPU.

An independent read-only audit recomputed all157 scalar comparisons and17 fresh-reference comparisons and verified source, teacher, plan, raw-row and input-identity JSON bindings. Actual pixel/tensor equivalence was checked by the completed CPU/GPU jobs; the read-only audit did not reload the large prepared tensor archive or run another model. Verified summary SHA256: `d069e1e06a7c03e611479e265750200bbcb068d56c6dfb68c9a24240ba6326de`.

The supported claim is conditional0/1 and numeric-mass fidelity on64 fixed training pairs. Hidden states and full-vocabulary logits were not saved or shown equal; grouped output agreement does not imply their equality. No common-scene fusion, aggregate-answer training, extrapolation, generated reasoning, broadcast-token update or cached decoding was tested. Those remain separate architectural and empirical obligations.
