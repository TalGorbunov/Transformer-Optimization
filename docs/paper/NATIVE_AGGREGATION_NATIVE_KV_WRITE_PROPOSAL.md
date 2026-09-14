# A native KV path for aggregation writes

**Held mechanism proposal; no release or change to current fits or controls.** Injecting before an existing decoder block provides a continuous path from an earlier aggregation write to later predictions. It avoids requiring thought-token supervision merely to give earlier writes any gradient. It does not establish that this path will improve reasoning.

## One precise boundary

Start with the smallest case: read immediately before the final decoder block and write there, using the same question-conditioned learned set core. In the 28-block Qwen software prototype this means reading block 26's output and writing before block 27, using zero-based indices. Cosmos requires its own verified architecture and boundary. Do not search over layers.

For global token position t, let h_t be the frozen lower-stack state and d_t the aggregation residual. Supply `h_t + d_t.to(h_t.dtype)` to the final block, before its input normalization and Q/K/V projections. Local rows receive no write. Its current global K_t and V_t therefore depend on d_t. Subsequent global queries can attend to those cached entries:

`d_t → normalized block input → K_t,V_t → attention at u>t → native logits_u → answer CE_u`.

The block's post-attention/MLP output is not retroactively stored in that same block's KV. With additional upper blocks, their KV also incorporates upstream processing. A hook after block j cannot change KV_j already computed; its first affected KV is in block j+1. A final-norm-only hook changes no KV. These distinctions require explicit hook placement, not a generic “intermediate residual” label.

At a fixed token history, all lower-layer caches and all local caches remain unchanged. Only global KV at and above the injection boundary may change. The core's lower-layer query itself does not read that upper-layer continuous memory; the upper decoder performs the temporal integration. The residual need not survive verbatim: frozen normalization and projections transform it.

## Training without detaching the proposed memory

Teacher forcing can process all target positions in parallel with the native causal mask. For target tokens y, input `prompt + y[:-1]`; inject at prediction-query positions `P-1+t`, including the last prompt position. Compute all d_t and scatter them before the upper stack. Later losses then backpropagate through attention into earlier d_t. Future writes must have zero influence on earlier losses.

Cache frozen lower-stack local query states and the **complete** global prompt/prefix lower states, with exact native IDs, padding and position axes. Run the actual upper block(s), norm and head on complete global sequences. Frozen parameters still require activation autograd: do not use inference mode/no_grad around this replay, detach earlier writes, or train independent one-token upper replays with detached KV. Quantized input backward, casts and backend support need direct testing; frozen weights do not make backward free.

Native generation remains one N+1-stream model call per token. Each current global write enters ordinary native KV; broadcast emitted tokens exactly as before. There is no external memory, additional VLM call or attention operator. Teacher-forced full-sequence replay and cached generation must agree within a frozen numerical contract on identical histories, including every historical write.

## Minimal discriminating checks

1. **Identity and ownership:** zero U reproduces native outputs; nonzero writes change only permitted global states/KV. Check current versus old cache entries, local/lower invariance, padding, positions, hooks and restoration.
2. **Temporal derivative:** isolate independent d_t leaves. Later answer loss has a finite nonzero earlier-write gradient before the block, zero for the matched final-norm placement; future/other-scene gradients are zero. Test a nonzero-output fixture and the actual frozen quantized block.
3. **Persistence intervention:** change one earlier write, then force identical subsequent tokens and restore identical later writes. Later logits may still differ through upper KV. Restoring the affected historical KV must remove that difference. This distinguishes continuous persistence from the token-mediated route of final-norm fusion.
4. **Replay and cost:** compare actual same-captured-state upper replay first, then independently cached features versus deployment. Preserve all discrepancies. Profile forward/backward, long-sequence activation memory, cache storage and native latency before a fit.

Use common lower reads, core initialization, data and CE for a pre-block versus final-norm comparison. First-token gains reflect changed current-query processing, not earlier-write memory. Temporal tests establish a path; an efficacy attribution requires the persistence intervention at fitted checkpoints.

Long reasoning still costs N+1 streams and upper-stack activation storage/recomputation; native KV capacity is unchanged, not free. Answer-only CE can credit continuous writes along a fixed thought history but cannot differentiate the discrete choice of those thoughts. This does not solve trace quality, exposure bias, nuisance accumulation or insufficient local semantics.

The existing [V11 replay prototype](../../scripts/native_vision_v11_last_block.py) already implements the one-block principle. [FiD](https://aclanthology.org/2021.eacl-main.74/) supplies a relevant independent-encoding/generative-fusion precedent. Here the proposed difference is the tested placement and training of a compact write into existing decoder memory, not a claimed new pooling or cross-attention family.
