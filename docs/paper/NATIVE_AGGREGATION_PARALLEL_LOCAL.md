# Reusing full local computation in one model forward

Prospective fallback design,2026-09-10, while V6 fresh evaluation is running. The registered first step is an independent software parity oracle, with no aggregation result or model training. Its input selection, thresholds and0.1GPUh envelope are in PREREG_AGG.md. No full architecture experiment is registered by this note.

The V6 teacher establishes competence of the complete frozen isolated-image VLM. It does not establish that a randomly initialized rank96 cross-attention reader, a copied late FFN, or a truncated decoder tail can perform the same judgment. If V6 fails local transfer, reusing the demonstrated computation is better motivated than continuing to adjust its readout.

A direct construction uses a native batch of N+1 streams. The first N rows contain complete single-image question/chat prompts and their actual isolated three-axis positions. The final row contains the global question and count instructions for the actual N, without image placeholders. Ordinary batch separation supplies the independent attention computation. All rows use the same frozen model weights and one visual/model invocation.

Each local final query state becomes available only after the last decoder layer. Fuse there, immediately before the global final RMSNorm and vocabulary head:

$$\widetilde h_g=h_g+U\sum_i\phi(h_i,h_g).$$

A shared nonlinear elementwise map is enough to define the initial candidate. An explicit0/1 decoder or external arithmetic is unnecessary. Only the global row emits user-visible tokens. Keeping the entire backbone frozen preserves the local computation and leaves a small trainable aggregation branch. A globally attached LoRA would also alter local rows; global-only LoRA needs explicit row routing and is not part of this minimal design.

This is a different input/computation arrangement. Zero initialization recovers the text-only global baseline, not the original joint-image VLM. Preserving the original joint-image path is possible, but requires both isolated and joint image decoder states. Shared visual embeddings avoid repeated vision encoding; they do not avoid the duplicated language-decoder FFN work or the joint path's quadratic attention.

The initial oracle compares native local batches against the archived V6 isolated teacher, with fresh serial references. It freezes complete token/pixel/grid/position identities and numerical thresholds, varies batch size/order, records all failures, and measures actual cost. It does not test a global stream, cached generation, or aggregate answers. Those are subsequent implementation gates.

For later generation, choose the global output token and broadcast it to every local stream at the next step. Each row keeps its own cache and positions. This permits another parallel local computation for each reasoning token, but makes no claim that it improves reasoning. Full-depth local states cannot depend on the current late global state during the same layer traversal. Fusion at the final layer also does not directly rewrite earlier-layer caches; subsequent coordination occurs through the shared generated text.

Local attention costs scale with N times the square of local prompt length, while FFNs still process every local prompt. A text-only global row avoids duplicated image-token FFNs. During decoding, however, every global token requires N+1 full-depth token computations, and local caches grow with N times generated length. Wall time, memory and FLOPs must accompany any one-forward claim.

A frozen backbone may allow local/global features to be cached for training the final aggregation branch, provided the complete original prompts, teacher-forced target positions, row layout and numerical backend are preserved. That is only a potential training optimization. Final evaluation must establish equivalence to the actual single-forward model, and no feature caching can substitute for a claimed native inference result.

[FiD](https://aclanthology.org/2021.eacl-main.74/) is a close precedent for independent question-conditioned evidence processing followed by generative fusion. Conditional nonlinear map-and-sum aggregation is already covered by [Deep Sets](https://arxiv.org/abs/1703.06114). The repository's earlier fenced replicas are another direct ancestor. The candidate contribution is therefore a measured transfer of pretrained local competence into precise native aggregate answers, with honest compute accounting and later tests of reasoning composition; no new attention-family claim follows from packing the streams.

## Final-layer readout constraint identified before training

The linear injection above is not a universal native readout. Fix the question and N, and suppose ideal neutral evidence gives an aggregate K v. Absorbing frozen final RMS gains into the bias-free vocabulary matrix M, predicted classes maximize a_j+K b_j, with a=Mh_g and b=MUv. The common positive RMS denominator cannot create additional argmax boundaries. To predict adjacent numeric classes k−1 and k correctly at their respective integer inputs requires Δb_k>0 and (k−1)Δb_k<a_{k−1}−a_k<kΔb_k. In particular the frozen numeric intercepts must already decrease in the required order. Equal intercepts make one class win for every positive K. At K0 the exactly neutral branch cannot correct an initially wrong text-only answer.

This conditional counterexample does not describe arbitrary learned local features, which can retain nonneutral contributions or content-dependent structure. It does show why a faithful local teacher plus a linear final write can fail for reasons unrelated to local capacity. A learned global offset and/or post-aggregation map rho(h_g,sum phi) is the more defensible candidate before training. Such a map must be compared with a suitable affine readout control, including intercept freedom. It is the standard second stage of Deep Sets, not an additional universality claim for the frozen native head.
