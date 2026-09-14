# Query placement in the product aggregator

Status: one fresh-initialization ablation, prepared after geometry443401 and the failed factor-binding comparison443383. No fitting before source review and CPU verification. This is the final architecture variant in this trainability branch.

The existing product achieved79/108 training first tokens and5/18 complete families; its registered screen failed. The fixed factor-pairing intervention reduced it to38/108 and0/18 families. Independently audited saved-state geometry then showed that removing the explicit global-query term lowered saturation and raised the local payload derivatives on all216 final scene/factor rows, across all6 questions and both lengths. Final product-factor saturation changes from94.5/94.4% to72.6/76.6%; mean-absolute payload partials increase from.0112/.0117 to.0881/.0710. This supports a particular placement test, not a causal explanation of errors. Early sampled states at steps1–2 were largely unsaturated; this concerns learned operating states. Substantial saturation remains in the counterfactual.

Make exactly one change to the frozen product architecture:

```
x = (RMS(h_i) - frozen_global_mean) / frozen_scale
q = Wq RMS(g)
u_i = tanh(A x + a)          # remove explicit +q here
v_i = tanh(B x + b)          # remove explicit +q here
z = SUM_valid_items 0.5 * (u_i * v_i)
Delta = U SiLU(Wagg z + q + bagg)
```

The local hidden state h already depends on the complete current question and prefix. This is not a question-independent image encoder; it removes one conditioning route while retaining the query-dependent readout. Global and local representations are different, so the removed term is not algebraically redundant. A small subclass routes zero to the inherited factor encoder and leaves the actual q in the inherited post-SUM readout. Product dimensions, fixed half-weight selection, all parameter shapes and counts remain unchanged:1,385,857 retained,1,385,760 trainable,97 frozen selector coordinates. No activation, statistics, optimizer, loss or factor variant is added.

Start from the exact **unfitted** seed24 parameter packet saved by factor check443373, shared with the previous product and additive runs. Never use either fitted endpoint. Confirm the new constructor reproduces those bytes. The complete initial output remains unchanged because U=0 in both models. Internal factors/aggregates, the first U gradient and subsequent optimization paths differ; equal parameter bytes do not make those gradients equal.

Reuse the exact original108 training contexts,54 N8/N16 pairs,18 families, existing global statistics and4800 pair-presentation order. Run600 full-name-plus-EOS CE updates with the existing mean-scene/mean-token reduction,8 pairs/update,AdamW lr.001,warmup50,cosine to1e-5,weight decay0,clip1,betas(.9,.999),epsilon1e-8. Consistency coefficient remains0. Actual frozen Qwen FP16 norm/head consume g+Delta.half(); the core staysFP32 withTF32 disabled. Training has21334 target rows. Save all600 logs and6 pre-update captures at1,2,32,128,300,600, then reset/restricted-reload only step600.

Evaluate the same108 paired first queries in7 batches. Preserve the same predetermined within-scene valid-only cyclic factor-B permutation diagnostic in7 additional batches. The run has614 core/norm/head calls and21550 head rows; zero VLM/vision calls. Save all raw full-vocabulary logits, factor captures, fixed checkpoint and actual Python placement/pairing state. Independent CPU replay covers14 final batches/216 rows; training captures require functional arithmetic and FP64 NLL checks without extra heads. Retain atol/rtol1e-4 for FP32 functional captures,2e-6 for FP64-reference NLL, and TV<=.02 plus exact argmax for native FP16 head replay. Verify the actual captured query remains unchanged and enters the post-SUM readout.

Compare descriptively with the already audited original product run443375, binding its raw predictions, initial packet, statistics, order and source. That model is not refitted or selected from a sweep. The new fixed screen remains at least103/108 paired first tokens and16/18 complete families. The permutation diagnostic cannot alter the screen or select a checkpoint. The one-seed comparison tests query placement under this recipe, not generic architectural superiority.

Resources: one CPU check90s/4cores/16G; one B200 GPU attempt90s, at most1 project GPU; one CPU report300s/4cores/16G. Preserve failures and allocation accounting. No profile, extension, continuation, alternative checkpoint or additional architecture variant is released. Source closure is101 geometry ancestors plus7 new files. A passed screen releases preparation of separately audited native full-name/EOS evaluation and fresh-family validation of the unchanged checkpoint. Failure ends this architecture branch. Neither outcome by itself establishes N32/N64 extrapolation, wider communication capacity, a new attention primitive, or reasoning composition. Bilinear pooling and nonlinear DeepSets remain the relevant architectural precedents identified in the factor-binding proposal.
