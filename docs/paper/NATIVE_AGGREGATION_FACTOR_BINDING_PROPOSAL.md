# Does an explicit within-image product improve learning a joint code?

Status: implementation preparation after independently verified conditioning failure443351. No fitting until source review and a fresh CPU check pass. This is a fixed training diagnostic, not an accepted method or extrapolation study.

The identity-join contrasts preserve person counts, room counts, backgrounds and physical slots while changing which person appears in both requested rooms. Separate marginals cannot solve them. Our existing nonlinear vector map can encode conjunctions, yet no tested learned join variant has fit the complete contrast families. We test one concrete hypothesis: an explicit multiplicative interaction before pooling makes the relevant joint code easier to learn under the same optimization recipe.

For each original frozen local hidden state h and global state g, use the already fixed global conditioning statistics from conditioningV2 check443335:

```
x = (RMS(h) - global_mean_FP32) / shared_scale_FP32
q = Wq RMS(g)
u = tanh(A x + q + a)
v = tanh(B x + q + b)
m_product  = u * v
m_additive = 0.5 * (u + v)
z = SUM_valid_items 0.5 * m
Delta = U SiLU(Wagg z + q + bagg)
```

Products are coordinate-wise. Both maps have96 output coordinates, shared query projection, identical stored tensors and1,385,760 trainable parameters. The97 legacy selector parameters remain frozen at w=0,b=.5; retained count1,385,857. The local projection is one192-by3584 matrix split into two96-dimensional outputs. No factor labels, question lookup, per-image target, chosen semantic axes, new normalizer or trainable statistics enter either model. The normal path uses paired factors. Both compute the same projections and nonlinearities; the additive control places tanh before combining, avoiding a redundant linear factorization. Both payloads are bounded in[-1,1], without claiming matched activation variance or identical gradient geometry.

Initialize one fresh CPU model with seed24 and copy exactly the same parameter bytes to both arms. Use the existing module initialization procedure: construct the inherited core, replace its local projection with a default-initialized192-output linear layer and zero192-vector bias; all other inherited parameters retain their initialization, including zeroU. This is a new matched initialization, not the old96-output uniform initialization. The exact initialized state is saved before fitting. Both arms reuse unchanged global statistics, rows, frozen states, targets and the4800 pair-presentation order from check443335. No statistics are recomputed or chosen from outcomes.

Fit exactly two arms, product and additive, once each. Use108 original training contexts,54 N8/N16 pairs,18 families, seed24,600 AdamW updates,8 pairs/update,lr.001,50-step warmup, cosine to1e-5,weight decay0,clip1,betas(.9,.999),epsilon1e-8. Optimize only the unchanged mean-scene mean-all-name-plus-EOS native CE. Residual consistency is recorded with coefficient0. Actual frozen Qwen FP16 norm/head consume g+Delta.half(); projection arithmetic isFP32 withTF32 disabled. Training has21334 head rows, maximum44 positions/call, with zero vision/backbone calls. Save all600 training records and6 pre-update captures at1,2,32,128,300,600 with corresponding weights. Use only fixed step600 after reset and restricted checkpoint reload.

Evaluate all108 cached first queries in7 batches. The registered resource screen remains at least103 correct first tokens and16/18 complete six-context families. Store raw full-vocabulary FP16 logits, argmax IDs and FP64-reference NLL for every context. This screen establishes neither native generation nor full-name/EOS success. A passing arm releases preparation of native evaluation of that exact checkpoint; failed arms stop without refits, added updates, checkpoint selection or another normalization variant.

A single predetermined diagnostic asks whether the product uses within-image pairing. At the final product checkpoint, cyclically roll v by one position among valid items independently for every scene/query; keep u, query, factors' marginal multisets, mask and all weights fixed. Recompute all108 first-query outputs in7 additional batches and save raw vectors and captures. This permutation is fixed before fitting and independent of gold labels. It is an intervention outside training, not a semantic factor swap or proof of disentanglement. Report paired/permuted accuracy, families, NLL and residual/aggregate changes descriptively. On additive captures independently verify permutation invariance of the pooled function, retaining FP32 rounding differences without extra head execution. Neither permuted accuracy nor change selects checkpoints or changes the primary screen.

The additive arm has607 core/norm/head calls and21442 head rows. Product has614 calls and21550 rows including the diagnostic; both have600 training updates and identical paired evaluation. No VLM/vision calls occur. Save paired and permuted factor preactivations, u, original v, used v, payload, scores/gates, messages, aggregate, readout preactivation,Delta, original/conditioned local inputs, fused FP16 states, native normalized states and logits. Preserve all failures and allocation accounting.

CPU check:90s,4cores,16G, no pretrained forward. Two parallel B200 fits:90s each, maximum2 project GPUs,total180 allocated GPU-seconds including failures, no profiling/retry extension. One independent CPU report:300s,4cores,16G. Audit source/input/init/statistics bindings, order, all600 loss reductions, fixed selectors, checkpoint roundtrip and all captured functional arithmetic. Retain FP32 functional capture tolerances atol/rtol1e-4; FP64-reference NLL atol/rtol2e-6; native FP16 head replay TV<=.02 and exact argmax. CPU head replay covers7 paired batches per arm plus7 product-permuted batches:21 calls/324 rows. No additional head gate on initial training captures.

Product alone fitting plus sensitivity to permutation would support the usefulness of multiplicative pairing under this recipe. Both fitting would implicate the richer nonlinear encoder without establishing product-specific benefit. Neither fitting ends this comparison. Any successful result still needs native full-answer validation, fresh families and N32/N64 extrapolation, matched stronger controls and a separate reasoning-composition study.

This is established mathematical machinery. [Bilinear CNNs](https://openaccess.thecvf.com/content_iccv_2015/html/Lin_Bilinear_CNN_Models_ICCV_2015_paper.html) pool local outer products; [Hadamard low-rank bilinear pooling](https://arxiv.org/abs/1610.04325) uses learned projections and coordinate-wise products for visual question answering. Our nonlinear product is a two-factor feature map within a DeepSets sum. It is not a new attention primitive, does not enlarge the96-coordinate communication state, and does not prove greater expressiveness than nonlinear vector messages. Potential contribution must come from measured aggregation and extrapolation behavior under native language-model supervision, not renaming bilinear pooling. The factors cannot be called person/room codes without evidence.
