# Held: oracle readout and uniform-payload diagnosis

**Design only; no implementation or jobs released.** Proceed only after a completed, independently valid report shows that the [uniform control](NATIVE_AGGREGATION_UNIFORM_JOIN_CONTROL.md) fails its registered **training** criterion. An invalid run, incomplete run, or development-only failure does not trigger this diagnostic. This proposal precedes uniform outcomes; profile 443261 was running when it was requested. Preserve the original criterion and every result.

The question is whether the frozen native readout can learn a deliberately ideal answer code under a fixed recipe. This separates one readout positive control from descriptive evidence about the uniform model's upstream representation. The [feature-accessibility probe](NATIVE_AGGREGATION_FEATURE_ACCESSIBILITY_RESULTS.md) does not establish an information ceiling.

Reuse only the exact 108 training contexts: 54 N8/N16 pairs, six questions, three person trios and 18 complete six-context families. No development/test predictions, new images, prompts, local supervision, or backbone execution enter this diagnostic.

For answer class $y$, define $c_y=\sqrt{96}\,e_y\in\mathbb R^{96}$, with one active coordinate among the first nine and 87 unused coordinates. Class order is the frozen manifest order: Sandra, Mary, Michael, John, Daniel, Laura, Peter, Emma, Noah. Inject this fixed **post-SiLU code directly into U**, replacing the entire usual SiLU output:

\[
\delta_s=Uc_{y_s},\qquad
\ell_s=\operatorname{Head}_{FP16}\!\left(\operatorname{Norm}_{FP16}
 [g_s+\operatorname{cast}_{FP16}(\delta_s)]\right).
\]

The code has RMS one over 96 coordinates; its fixed amplitude avoids an arbitrarily small unit-one-hot RMS without calibration from outcomes. **SiLU is bypassed**, with no question addition or factor one half. The cached original global empty-prefix state $g_s$, actual native norm/head, FP16 arithmetic and source/weight identities remain frozen. Use full vocabulary logits without a mask. This cached head route makes no new claim of equality to a full native batch.

Initialize $U\in\mathbb R^{3584\times96}$ to exact FP32 zero; it is the only trainable tensor, with **344,064 stored/trainable coordinates** and no bias. Nine active columns provide **32,256 effective degrees of freedom**; the other 87 remain zero. This intentionally easy oracle is not capacity-matched to the uniform model.

Run exactly **600 updates**, seed 24, using the identical persistent pair order and eight intact pairs/16 scenes per update. Retain AdamW lr .001, 50-step warmup, cosine decay to 1e-5, weight decay zero, clipping at one, and the bound implementation's optimizer defaults. For canonical name-plus-EOS target length $L_s$, optimize

\[
\mathcal L=\frac1{16}\sum_s\frac{\operatorname{CE}(\ell_s,y_{s,0})}{L_s}.
\]

This preserves each original first-token loss contribution, without renormalizing retained tokens: Sandra and Noah have $L_s=3$, the other names $L_s=2$. Verify these lengths and nine distinct first-token IDs from frozen tokenizer metadata before fitting. No continuation, EOS or consistency loss is optimized.

Use only final step 600, verified by reset and restricted checkpoint reload. Record unweighted first-token CE, weighted loss, full-vocabulary argmax and failures for all 108 contexts, names, questions, lengths, all 54 pairs and all 18 families. The prospectively fixed readout-fit screen is **103/108 correct first tokens and 16/18 families with all six first tokens correct**. It is not the original whole-answer criterion and supports no EOS or name-completion claim. Both training and this evaluation supply the answer code: neither measures generalization nor a deployable method.

The separate CPU geometry audit uses every uniform final-training **first-query** capture, its final checkpoint and exact input bindings. Compute the existing FP32 RMS local states $x_i$ with epsilon 1e-6; define each question's mean $\mu_q$ over all its captured training item occurrences, counting each context once—**1,296 occurrences total**. This explicit occurrence weighting includes the extra N16 negatives. Decompose

\[
a_i=W_lx_i+q_s+b_l
   =W_l(x_i-\mu_q)+(W_l\mu_q+q_s+b_l),\quad p_i=\tanh(a_i).
\]

Retain common/deviation norms, per-coordinate variation, $1-p_i^2$, and saturation fractions at the fixed derivative threshold .01. Report all three pairwise variant contrasts per family and length, before pooling and after $z=.5\sum_i p_i$, using original slot alignment; retain all 54 length-pair contrasts and their insertion provenance. Use FP64 descriptive reductions, with CPU-versus-captured FP32 differences reported separately. No centered-input intervention is performed. Norms or saturation alone cannot identify the cause of failure.

Allow one source-check CPU job up to 60 seconds; one geometry/validation CPU job up to 300 seconds, four cores/16 GB; and **one head-only GPU job up to 150 allocated seconds**, including loading, publication and failures. The GPU inventory is 600 training plus seven final evaluation norm/head batch calls, maximum batch 16: **zero VLM and zero vision calls**. Source checks cover code placement/RMS, loss weights and frozen-parameter gradients. Bind inputs, labels, order, sources and native weights before fitting; retain final raw logits, every update log and failures. Save U under the checkpoint root and derived data under the data root. No automatic retry, tuning or additional software campaign.

Oracle success establishes readout trainability with this ideal code, not an upstream capacity limit or the cause of uniform failure. Oracle failure leaves this readout/optimization recipe unresolved, rather than proving impossibility. Neither outcome releases centering, another architecture, runtime gold access, or a broader aggregation claim.
