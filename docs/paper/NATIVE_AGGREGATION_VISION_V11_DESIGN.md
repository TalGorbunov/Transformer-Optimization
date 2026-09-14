# V11: common penultimate reads, different native write locations

**Protocol registered; efficacy execution held.** The native software profile passed, but the strong V10 background-direction diagnostic now takes priority. No V11 cache corpus or training run has been launched. This document specifies one comparison; it authorizes no training, data generation or GPU jobs. The question is whether letting the final pretrained decoder block process and retain an aggregation write improves supported-answer length extrapolation. It does not introduce a new pooling operator or establish reasoning composition.

## Comparison and decision

Use the same unchanged SUM/SiLU core, rank 96, hidden size 3584 and 1,041,600 FP32 parameters. Both conditions read local and global states immediately after block 26, before the final block 27, in the 28-block Qwen model. `pre_last` writes the resulting residual there; `post_last` holds that same computed residual and writes immediately before the final norm. Both use native `hidden + delta.to(hidden.dtype)`, the real FP16 norm/head and frozen NF4 backbone. Placement is the sole treatment; earlier V10 final-layer-read models are historical context, not its matched control.

Use seeds 16/17, matched initial parameter bytes and presentation order within seed. Both optimize native full-answer CE plus coefficient 1 residual consistency on the same pairs and strict prefixes. Zero-U outputs should agree subject to the registered numerical gate; initial gradients need not agree, because the write locations have different Jacobians. Record gradient hashes without requiring equality.

Proposed primary criterion, in **each** seed: `pre_last` gains at least 7/136 N64 correct answers over `post_last`, while losing no more than 6/136 at N32. Proposed practical criterion additionally requires `pre_last` N32≥123/136, N64≥109/136 and N64 K9–16≥52/64. These retain V10's thresholds; pooling seeds cannot rescue failure. Report paired-family uncertainty with 10,000 K-stratified resamples shared across conditions, lengths and fitted seeds. Freeze a new bootstrap seed before execution.

## Data and training budget

Reuse the exact audited V10 training/development manifests, schedule and pairing. There are 1782 unique training scenes, 918 pairs and 1836 weighted scene slots per epoch: K0–8 pairs N8 with N16, K9–15 pairs distinct N16 scenes, and 54 K16 pairs repeat the same saturated N16 SID. Retain these identity pairs with zero consistency in the denominator. The historical N8/K8 and N16/K16 reuse exceptions remain disclosed. Development is the same 64 N16 scenes, four each for K0–15; no purported fresh saturated K16 development example.

Forty epochs give 73,440 scene presentations, 177,120 valid target positions and 4590 batch 16 updates. Use the same persistent `random.Random(seed)` pair permutations, carry batches across epochs, and preserve pair-side order. Retain AdamW lr 0.001, weight decay 0, clip 1, 50-step warmup and cosine decay to 0.00001. Evaluate development at 918/1836/2754/3672/4590; select whole-answer exact, then raw first-token NLL, then earliest. Retain all five checkpoints. First-token NLL remains only a tie-break for shared first digits.

Generate 272 new test contexts: eight families per K0–16, with each N32 parent extended to N64 by the unchanged negative-frame insertion law. Freeze the seed and complete-context exclusions before rendering. The expected ancestor inventory is V10's 25 excluded manifests plus V10 balanced and fresh manifests; verify the actual complete ledger, including any subsequently introduced samples. All new tests exclude current train/dev. This tests longer contexts with supported answer values, not unseen integers or new visual primitives. Report K0–8, K9–15 and saturated K16 separately.

## New frozen features and exact layouts

The V10 cache reads immediately before final norm and cannot be reused as penultimate evidence. Reuse its immutable inventory/tokenization utilities, but publish a separate cache root and descriptor identifying the exact model, block 26 output, source hashes and extraction route.

| Record | Required contents |
|---|---|
| Local query feature | Native FP16 penultimate state; key(imageSHA, exact question, strict prefix IDs); input/grid/position hashes |
| Global query feature | Native FP16 penultimate state; key(exact question, strict prefix IDs); text-only input/position hashes |
| Full global prompt | All penultimate token states and native IDs/mask/positions for each of 54 questions, keyed by exact prompt/model/read boundary |
| Scene | Ordered local feature IDs [N][T], global IDs [T], full native numeral+EOS, prefix lists and prompt record |

Gold, N and total answer length must not enter feature keys. The 18 distinct strict prefixes still imply 972 global query states; the local inventory is recomputed and audited. Capture the full prompt and its empty-prefix query from the same forward, so prompt[-1] exactly equals that query feature. Store valid unpadded prompt states, not unidentified padding. Bind native dtypes, masks, three/four position axes, model/shard metadata and actual installed backend sources.

## Differentiable final-block training

For a scene with prompt length P and target length T, reconstruct prompt hidden states followed by the cached last-query states for prefixes y[:1] through y[:T−1]. Inputs are prompt+y[:-1]; the T loss/query positions are P−1+t. The last prompt token predicts the first numeral. No target EOS or future answer token enters an earlier query. Validate the reconstruction against exact native input IDs and prefix metadata.

Pack complete global sequences with explicit padding and per-scene query indices. Compute all local-to-global residuals in FP32, keeping every position live. Invoke the **actual** final block once on the complete global batch, with native rotary embeddings, causal mask, `past_key_values=None` and `use_cache=False`. In `pre_last`, scatter all residuals before that block; in `post_last`, scatter them afterward. Apply the real norm/head at valid query positions. Do not split prefixes into independent cached one-token training calls or detach earlier residuals: either would remove the proposed temporal gradient path.

CE averages valid tokens within each scene, then scenes; consistency averages corresponding strict prefixes within each pair, then pairs. Its denominator is the unchanged fixed squared norm of the shared frozen penultimate global query plus 1e−6. Those global states remain identical within pairs because they precede any write and have the same text-only prompt/prefix. Saturated identity penalties must be exactly zero. Retain per-position/per-pair loss components, clipping, finite-gradient checks and frozen-backbone audits. Ordinary frozen tensors must be cloned outside inference mode before autograd replay.

## Software gates and extraction-route uncertainty

The new [replay helper](../../scripts/native_vision_v11_last_block.py) and [native controller](../../gnnformer/parallel_local_memory.py) are software prototypes, not efficacy evidence. Required gates include later-token CE reaching an earlier independent delta only in `pre_last`, zero future-delta gradients, native cast-before-add, ragged masks, zero/nonzero-write replay, restoration and frozen weights. Full-prefix native replay must apply **all historical assistant-query writes**. Native cached `pre_last` retains fused final-layer K/V; `post_last` retains unfused K/V. Earlier layers and every local stream remain unchanged.

Separate three numerical comparisons: replay from the exact same captured full penultimate batch; reconstruction from independently harvested prompt/query states; and actual native N+1-stream deployment. Passing the first does not imply the other two. Feature harvest batches, teacher-prefix lengths and deployment batches differ; native quantized kernels and SDPA can change rounding. Freeze one common extraction route and cache for both conditions. Preserve hidden-state errors, centered/full-vocabulary differences, top1 and TV against fixed native references; report every violation, with thresholds fixed before outputs. No arm-specific feature correction or successful-example subset is allowed.

Native evaluation remains one N+1-row VLM invocation per generated token, one visual prefill, ordinary caches and greedy generation up to 4 tokens with native EOS. Every generated token is broadcast unchanged; no forced digits, local labels or external tally. Whole-answer correctness requires a complete ASCII integer and EOS. Malformed/truncated answers remain incorrect. Save raw token IDs and losslessly promoted native logits for independent rescoring.

## Source and resource release

Add new V11 files for feature inventory/harvest, paired training, native evaluation integration, release, report and selected-checkpoint audit; leave V10 sources/data/results frozen. Reuse parameterized immutable helpers where their contracts still apply. Each attempt snapshots sources before validation. CPU plans bind the V10 ancestor audit, common penultimate cache, fresh manifest, both software gates and exact backbone/last-block identity. Reports independently reconstruct targets, order, loss reductions, checkpoint selection, predictions and both-seed criteria.

Profile both 32-update training conditions on training-only cases, including strict prefixes of 0/9/10/16. NF4 input backward dequantizes frozen matrices: head-only V10 timings are ineligible. Measure model/cache load, full last-block forward/backward, peak memory, preprocessing and new native four-token N16/N64 generation bounds. Run the final block in both conditions; record that its backward work differs. With shared maxima, project `load + 1.25*(4590*step + 320*T16 + 272*T64) + 120`. Freeze campaign and per-job limits from those measurements before any main; no assumption that earlier 45-minute allocations suffice. All heavy CPU and GPU work uses Slurm and the prescribed data/checkpoint roots.

An improvement would support the complete write-location intervention. It could reflect richer within-query pretrained computation as well as attention to earlier writes. First-token gains cannot depend on earlier assistant writes. The temporal-gradient test proves a possible path, not its causal contribution to accuracy; free reasoning and a memory-specific intervention remain separate experiments.
