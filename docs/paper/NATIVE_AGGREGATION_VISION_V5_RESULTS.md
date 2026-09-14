# V5: extensive independent visual messages did not improve extrapolation

V5 failed its registered SUM-versus-MEAN screen in both seeds. SUM was worse on both familiar-length extrapolation and N16 accuracy; its seed-5 run collapsed to unparseable outputs at N64. Neither arm established useful unseen-count extrapolation. This does not support the proposed extensive-message method, and there is no reasoning-composition result.

The canonical [report](../../outputs/native_aggregation_vlm/v5/REPORT.md), [verified analysis](../../outputs/native_aggregation_vlm/v5/analysis.json), and [figure](../../outputs/native_aggregation_vlm/v5/comparison.png) were produced by CPU job441547 (18 seconds). These are exploratory reused V2 scenes, not untouched confirmation data.

## Registered comparison

Both arms read the same ordinary, independently encoded Qwen visual features and inject a native residual at layer14. At fixed weights and query, SUM equals the number of visible images times MEAN. This changes scale and optimization; it does not add information or establish a new attention family. Both use the V4 refresh schedule:1,620 presentations of1,540 distinct scenes,405 updates, nine72-example development evaluations,1,041,504 branch parameters and720,896 upper-LoRA parameters. Development alone selected the checkpoints.

| Arm / seed | N16 /108 | N32 /108 | N64 /108 | Familiar OOD /216 | Unseen K9–16 /128 | Selected block |
|---|---:|---:|---:|---:|---:|---:|
| SUM /4 |33 (30.6%)|17 (15.7%)|13 (12.0%)|30 (13.9%)|0|6|
| SUM /5 |17 (15.7%)|12 (11.1%)|0|12 (5.6%)|0|6|
| MEAN /4 |42 (38.9%)|25 (23.1%)|22 (20.4%)|47 (21.8%)|0|7|
| MEAN /5 |45 (41.7%)|17 (15.7%)|16 (14.8%)|33 (15.3%)|1 (0.8%)|8|

The registered requirement was at least+5 percentage points of familiar N32/N64 exact accuracy, with at most5 points lost on N16, **in each seed**. SUM-minus-MEAN OOD differences were−7.87pp for seed4 (paired-anchor95% interval[−17.59,+1.39]) and−9.72pp for seed5 ([−14.35,−5.56]). N16 differences were−8.33pp and−25.93pp. The pooled fixed-seed difference was−8.80pp ([−15.28,−2.78]); it is descriptive and cannot rescue either failed seed. Intervals resample whole anchor families, retaining paired lengths and fixed seeds; they do not estimate variation over a population of training seeds.

## Failures and optimization limits

SUM/5 parsed98/108 N32 answers, all as zero, and0/108 N64 answers. Frequent N64 strings included `8/8/` and `7/8/`. Its familiar OOD parse rate was98/216 (45.4%); its unseen-count parse rate was42/128 (32.8%), with all42 parsed answers equal to zero and0/64 parsed at unseen-count N64. All other runs parsed every answer. Unparseable outputs remain incorrect in every exact-accuracy denominator.

Familiar OOD MAE was2.421,3.980,2.171 and2.699 for SUM/4, SUM/5, MEAN/4 and MEAN/5. SUM/5's3.980 is conditional on the98 parsed N32 answers; it omits the entirely unparseable N64 cell and must not be read as a full-length error measure. Unseen-count exact accuracy was0/128 for three runs and1/128 for MEAN/5; all four had0/64 at unseen-count N64. Conditional unseen-count MAE ranged from6.672 to12.095.

Equal initial parameters, samples, order, nominal learning rates and parameter counts did **not** equalize updates. SUM clipped405/405 updates in each seed, compared with401/405 and402/405 for MEAN. Mean preclip branch norms were12.177/13.522 for SUM versus5.901/4.935 for MEAN. Shared global clipping also rescales the LoRA gradients. The observed disadvantage therefore belongs to the registered scale-plus-optimization treatment; it cannot be attributed solely to the aggregation algebra, RMSNorm or insufficient representational capacity. Initial branch-gradient hashes matched4/5 tensors, but four branch gradients are zero at zero-U initialization; this is not evidence of identical nonzero branch updates. Some first LoRA gradient hashes also differed. Gradient tensors and Adam moments were not retained for exact replay.

Both arms use independent visual memory. Their comparison cannot identify whether that memory source helps over ordinary hidden-state adaptation. Historical V4 results use different seeds and are not the missing contemporaneous control.

## Numerical history and cost

Earlier absolute and native-referenced raw-logit gates failed. Localization found exact image memory, image positions and rotary embeddings, with small upstream hidden-state differences and near-exact fixed-input branch replay. Most raw N64 logit discrepancy was a common offset. A later centered/probability gate still failed on the unused MEAN/N64 software case: total variation0.0118166 exceeded the frozen0.01 cap; its disabled-branch native reference was0.0148787.

A fixed-checkpoint kernel intervention identified native four-bit dispatch as a contributor: forcing the existing dequantize/matmul route reduced those distances to0.0077706 and0.0097143 while full-prefill outputs remained exact across routes. It did not eliminate numerical differences. Production inference retained the native NF4/bf16/SDPA/bitsandbytes backend. An explicit pre-efficacy engineering exception authorized exploratory training; **the failed pre-training strict gate remains failed**, separately from computational-integrity acceptance.

Post-training audit441517 checked all four development-selected checkpoints, with five cache comparisons and ten software evaluations per checkpoint. Computational integrity passed for all four, and all20 observations passed the unchanged strict numerical criterion. This does not retrospectively change earlier failures or prove general/long-reasoning cache equivalence. Audit outcomes did not select or replace checkpoints or main predictions.

The complete V5 GPU expenditure was**7,436 GPU-seconds (2.06556 GPU-hours)**:6,868 seconds for the four main runs,413 for earlier V5 profiles/diagnostics/completion, and155 for the final four-checkpoint audit. This is the V5 block, not the cumulative V1–V4 or separate reasoning-baseline research cost. Main jobs were441499,441501,441500 and441498 for SUM/4, SUM/5, MEAN/4 and MEAN/5. CPU analysis jobs add no GPU time. Measured training time per run was772–781 seconds; familiar OOD inference was1.09–1.13 model seconds/example and1.59–1.63 seconds including preprocessing. These timings include the shared GPU diagnostic instrumentation and exclude subsequent CPU tensor serialization/hashing.

## Separate conditional readability analysis

The preregistered conditional [readability audit](../../outputs/native_aggregation_vlm/v5/local_readability/run_441550/REPORT.md) completed as CPU job441550 (5 seconds). It fitted one fixed class-balanced ridge probe per checkpoint on54 familiar N16 families (864 frames), with fit-only standardization,96 message features,lambda0.001 and threshold0. It evaluated54 disjoint families at each length and all128 unseen-count contexts separately. These external labels never entered native model training or inference.

| Checkpoint | Held N16 AUROC | Held N32 AUROC | Held N64 AUROC | N64 TPR / FPR at0 |
|---|---:|---:|---:|---:|
| SUM /4 |0.6269|0.5321|0.5108|0.2663 /0.2309|
| SUM /5 |0.6232|0.4598|0.4425|0 /0|
| MEAN /4 |0.6107|0.5938|0.5391|0 /0|
| MEAN /5 |0.6917|0.5858|0.5348|0.4070 /0.3365|

Each held N64 cell contains3,456 frames:199 positives and3,257 negatives. SUM/5 and MEAN/4 predict every OOD frame negative at the fixed threshold; their zero false-positive rates therefore do not indicate a useful detector. Unseen-count AUROCs are0.5512,0.5602,0.6271 and0.6210, respectively, on a distinct secondary distribution.

This fixed linear probe does not provide strong evidence of reliably readable conjunction messages at extrapolated lengths. It weakens the specific account that the messages are already cleanly readable and only numeral output is failing. It does **not** prove information absence, failure of every nonlinear decoder, or the mechanism of V5's count errors. Pooled frame metrics can reflect contextual prevalence cues; per-N/K and negative-class results remain available. Frames share contexts/families, and these descriptive scores have no new significance or pass/fail claim.

The separate message-decomposition job441549 initially failed an artifact-schema check: prediction rows do not serialize the anchor/parent position lists. After a registered diagnostic-only repair and CPU self-test441556, [decomposition441558](../../outputs/native_aggregation_vlm/v5/message_decomposition/decomposition_441558/REPORT.md) completed in6CPU-seconds over all864 comparisons. The correction uses the hash-bound manifest maps and rejects conflicting optional prediction fields; failed441549 remains preserved.

Added-negative message norms substantially exceed mapped-original-frame drift: SUM seed4 N16→N64 means624.86 versus10.55, seed5 means673.87 versus8.93. The saved branch/native-attention-output norm ratio approximately doubles with each doubling ofN (SUM4:11.10→22.62→46.36; SUM5:8.42→17.01→34.75). MEAN stays roughly1.65–1.79, and its added-message/normalization terms largely cancel. These are descriptive vector observations, not proof of harmful evidence or normalization-induced information loss: the denominator is attention output, not the full residual stream. Only150/3456 mapped image pairs are byte-identical;3306 have renumbered Step pixels, which confound a causal interpretation of mapped-frame drift. No native prediction is changed by this analysis.

Verified analysis SHA256: `897421f71839039007c8398cf132cd21db25b1b46f1516ce542e683b0cd9bc91`. Readability summary SHA256: `346e8fb051a2e6247752cb03b44878157ef4c50d044b1af5e6c0262d51b8560c`.
