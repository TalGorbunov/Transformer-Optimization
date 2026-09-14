# Factor binding: pairing matters, but neither model fits the diagnostic

Independent [report 443383](../../outputs/native_aggregation_vlm/identity_join_factor_binding/report_443383/REPORT.md) passed its computation and provenance audits. **Both fixed training screens failed**, so this comparison does not release native whole-answer evaluation or further fitting.

These are cached first-token predictions on the same 108 training contexts: 18 complete families, each containing three answer-changing variants at N8 and N16. The registered screen required at least 103/108 correct first tokens and 16/18 entirely correct families. All contexts remain in the denominator.

| Model / final intervention | Correct first tokens /108 | Complete families /18 | Mean first-token NLL | Paired screen |
|---|---:|---:|---:|---|
| Product, original pairing |79|5|0.868483|FAIL|
| Additive, original pairing |58|1|1.055851|FAIL|
| Product, fixed cyclic permutation |38|0|1.402758|Descriptive only|

Both models used identical fresh seed 24 parameter bytes, 1,385,760 trainable parameters, 97 frozen selector coordinates, the same fixed global conditioning statistics, and 600 updates over the original 4800 pair presentations. The objective was mean-scene, mean-token **full-name-plus-EOS CE**, with residual consistency coefficient 0. Each fit consumed 21,334 target positions. Only the fixed step 600 checkpoint was evaluated after reset and restricted reload. Final minibatch CE was 0.414147 for product and 0.485151 for additive; these minibatch losses are distinct from the all-context first-token NLL above.

The product combined two learned 96-dimensional tanh factors coordinate-wise; the matched control averaged them. Both then used the same fixed half-weight SUM and SiLU readout, communicating 96 coordinates. Product corrected 24 cases that additive missed while losing 3 cases additive got right. This is a descriptive difference from one matched seed, not evidence of reliable task mastery.

The single predetermined permutation rolled the second factor by one valid item within each scene. It preserved the original factors, their marginal multisets, query, weights and masks; exact saved-factor/query equality was verified. It changed 48 correct product predictions to errors and 7 errors to correct predictions. Thus the fitted product uses within-item pairing, but the intervention does not identify either factor as a person or room representation, establish disentanglement, or explain every error. Additive pooled invariance passed all 7 CPU checks in FP64; FP32 reassociation differences remain descriptive.

All 33 saved training/final capture batches passed the prescribed functional arithmetic audit. CPU native-head replay covered 21 batches / 324 rows, including the product permutation. All 324 argmaxes matched. Maximum full-vocabulary TV was 0.003827726 for product and 0.003777661 for additive, below the fixed 0.02 gate. This validates replay from cached native states; it does not establish equivalence to complete VLM generation. No development, fresh test, full-name completion, EOS-generation or reasoning outcome was evaluated.

Product job 443375 used 28 allocated GPU-seconds; additive 443376 used 27: **55 total**, maximum 2 concurrent GPUs. The additive/product runs used 607/614 norm calls and 607/614 head calls respectively: 1,221 of each and 42,992 rows overall. There were zero VLM or vision calls. Independent CPU reporting was a separate job.

This remains established multiplicative feature machinery: [bilinear CNNs](https://openaccess.thecvf.com/content_iccv_2015/html/Lin_Bilinear_CNN_Models_ICCV_2015_paper.html) pool local outer products, and [Hadamard low-rank bilinear pooling](https://arxiv.org/abs/1610.04325) uses projected coordinate-wise products. Our result neither introduces a new attention primitive nor demonstrates increased communication capacity or greater expressiveness than a nonlinear vector map. The registered comparison ends with failed trainability screens. A separately reviewed geometry analysis may describe the retained captures; no new fit follows from this result.

Provenance: [proposal](NATIVE_AGGREGATION_FACTOR_BINDING_PROPOSAL.md), [frozen CPU plan](../../outputs/native_aggregation_vlm/identity_join_factor_binding/check_443373/plan.json), [analysis](../../outputs/native_aggregation_vlm/identity_join_factor_binding/report_443383/analysis.json), [summary](../../outputs/native_aggregation_vlm/identity_join_factor_binding/report_443383/summary.json), [reporter](../../scripts/report_native_identity_join_factor_binding.py), [core](../../gnnformer/parallel_local_factor_binding.py). The report binds 99 source files, all inputs, both fixed endpoints and raw captures.

Summary SHA256: `de38a1498115bc835955cf264a609ead1a095bacb0c3faa99eba10a1ae962238`.
Analysis SHA256: `326db6392f648a517e169b206ca2c46b25b28f1d7f2faf3da31a314afcb74f94`.
