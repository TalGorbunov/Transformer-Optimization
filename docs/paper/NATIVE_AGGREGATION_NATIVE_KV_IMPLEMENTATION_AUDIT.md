# Native KV write: minimal implementation audit

**Held engineering note; source/JSON inspection only.** No model, tensor, test or Slurm execution occurred for this audit. The immutable six-fit experiment and [next-controls plan](NATIVE_AGGREGATION_LEARNED_SELECTION_NEXT_CONTROLS.md) remain unchanged. Implement one before-final-block boundary, not a layer sweep; this note does not release that implementation or predict efficacy.

## Installed contract and causal boundary

Both pinned mirrors—the Qwen snapshot `cc594898137f460bfe9f0759e9844b3ce807cfb5` and Cosmos compatibility mirror `cosmos_reason1_compat_440953`—declare `Qwen2_5_VLForConditionalGeneration`, 28 decoder blocks, H=3584, 28 query heads, four KV heads and head dimension 128. Their config bytes have the same SHA256 `77d9ec7321cc572e3579e2c84799c9cadaded63c49ce93b101733349fc330c43`. This is architecture compatibility, not weight, tokenizer, prompt or numerical equivalence. Bind each actual loaded identity separately; config says BF16 while prior native norm/head proofs measured FP16.

In installed Transformers 4.57.6, `model.model.language_model.layers[27]` receives block 26's output. `Qwen2_5_VLDecoderLayer.forward` applies input RMSNorm, then attention Q/K/V projections. Attention applies mRoPE to Q/K, calls `past_key_values.update(..., layer_idx=27, ...)`, then SDPA. `DynamicLayer.update` appends K/V along the sequence axis. Injecting before that block therefore changes current global K/V; injecting before final norm does not. The final block's attention/MLP output is not itself stored in its own KV.

Choose common block-26 reads for both placements. At fixed tokens, lower layers 0–26 and every local row remain unchanged; only global layer-27 KV after the first write differs. Earlier cached entries must remain intact as new positions append. The core's lower global query does not itself read this upper memory; the final block integrates it.

## Reuse and necessary new code

| Existing asset | Reuse boundary |
|---|---|
| [Learned-selection core](../../gnnformer/parallel_local_learned_selection.py) | Keep all 1,041,697 parameters, current-prefix scores, bounded values and padding-only mask; no new gate supervision. |
| [V11 controller](../../gnnformer/parallel_local_memory.py) | Reuse two-hook lifecycle, explicit query indices and native cast-before-add. It already accepts the learned core's base class, but lacks its mode lock, score/payload capture and finite native-write checks. Add these in a new controller file. |
| [V11 replay](../../scripts/native_vision_v11_last_block.py) | Reuse `reconstruct_global_sequence` and short-sequence `replay_last_block`. Run outside active controllers. Preserve the actual frozen block, norm/head and rotary modules. |
| [V11 batch helper](../../scripts/native_vision_v11_batches.py) | Reuse reconstruction/left-padding logic only. Its old branch/path objective and fixed counting bindings are ineligible. |
| [Current native runtime](../../scripts/native_learned_selection_runtime.py) | Reuse safe input views, broadcast, global-logit streaming, source/weight checks and exception cleanup. Replace the last-query final-norm controller; its four-token assertion cannot silently become a reasoning budget. |

Proposed new modules: `gnnformer/parallel_local_learned_memory.py` for the controller, `scripts/native_learned_memory_runtime.py` for native execution, and `scripts/native_learned_memory_training.py` for reconstruction/replay. Separate tests and a bounded software profiler precede any corpus cache or fit. No monkeypatching frozen ancestors.

Controller API: `(read_layer, final_norm, core, n_local_rows, placement, query_indices, stream_positions, capture)`, with immutable selection mode. Full-prefix execution writes **every** query `W-1+t` for observed prefixes, not merely the final query. Cached generation writes current tensor index zero, absolute packed position `W+t-1`; initial prefill writes `W-1`. Preserve tuple extras and change only the global row. Compute the core once per forward over the supplied queries, retaining its actual intermediate tensors.

## New cache and full causal training

Existing final-norm feature tensors are at the wrong boundary. Reuse the core architecture, not current fitted weights; common-boundary weights require new matched training. Harvest new block-26 states from unchanged native image/question/prefix inputs. Keep ordered local states `[N,T,H]`, global query states `[T,H]`, and full global prompt states `[P,H]`. Store IDs, masks, native position axes, image/prompt/prefix identities and extraction-route hashes. Capture prompt[-1] and the empty query together and require exact identity. No target length, answer or N in feature keys.

For targets y including terminal EOS, reconstruct `prompt+y[:-1]`; query t is `P-1+t`. For a training batch, concatenate valid queries across scenes, gather local `[Nmax,sum(T),H]` and global `[sum(T),H]`, explicitly mask padded items, and compute all live FP32 deltas. Left-pad complete global sequences to `[B,L,H]`; scatter deltas at each scene's offset query indices using `h+delta.to(h.dtype)`.

Call the actual final block with `past_key_values=None`, `use_cache=False`, `output_attentions=False`. Native text-model inputs can carry `[4,B,L]` positions: axis zero is text position, the remaining three are mRoPE. Reproduce the installed split, `create_causal_mask`, `cache_position=arange(L)` and actual rotary embeddings. For unpadded stored layouts, native three-axis positions are sufficient; never substitute packed offsets for logical positions. Capture the actual mask/cos/sin for same-state reference replay.

Frozen parameters remain eval with `requires_grad=False`; activation autograd stays enabled through the final block. Clone cached tensors outside inference mode. Installed bitsandbytes `MatMul4Bit.backward` implements input gradients by multiplying through dequantized frozen weights, supporting the intended path in source. Recheck actual-device finite gradients and unchanged weights; do not assume cost equivalence to head-only training.

For long reasoning, separate **write_indices** from **loss_indices** in a new replay wrapper: all historical writes and the full causal block remain live, but only supervised answer positions need vocabulary logits. The old helper projects every write into 152,064 logits, which is unnecessarily large for answer-only CE. This optimization must preserve gradients to earlier writes; sampling independent prefix losses or detaching historical KV would not.

## What is already proved, and the next gates

[V11 profile 442259](../../outputs/native_aggregation_vlm/v11/memory_software/profile_442259/summary.json) passed 39 native forwards, 16 standalone block replays and 39 head replays, including 12 same-batch block comparisons. Its later numeral loss had earlier-delta gradient norm 0.0789695 before the block versus zero at final norm; future gradients were zero. It preserved two descriptive cached/full local-row TV failures, approximately 0.0202965. It did not train this learned-selection core, validate Cosmos upper-block backward, establish long-trace feasibility or demonstrate useful memory.

Next gates, in order: tiny causal/ragged/scatter tests; actual Qwen and Cosmos zero/active same-state replay with frozen identities; cached/full histories with all writes and special tokens; then a same-token intervention changing one historical write, followed by restoration of the corresponding layer-27 global KV. Restoring that slot must remove later effects with identical later writes. Keep captured-batch replay binding; separately report independently harvested/global-only numerical differences without filtering failures.

Profile complete upper-block backward, native long decoding, feature storage, head projections and I/O. Native KV remains `2 * 28 * (N+1) * 4 * S * 128 * bytes_per_element`; no new memory slots does not mean low memory. Answer-only CE credits continuous writes conditional on a fixed thought history, not the discrete choice of that history. First-token improvements and temporal gradients alone cannot establish beneficial reasoning composition.

Inspected installed sources: `transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` SHA `f5051523300145f66de080765c06bd8080350c15692c096d2c38e73fe9d03099`; `cache_utils.py` SHA `f4e11caf14c75bdefa5e3003b93ef0e8b97fb6103375705ad944b0b344d7da8a`; `masking_utils.py` SHA `7a963feed8173b8265298dd24350c166222ce6b4719442fce48ab414d0478a6a`; `bitsandbytes/autograd/_functions.py` SHA `f827d38aa68d400d32ca420569d2003add15aa7dc1fcd08af6c959997829dfad`. Future software plans must bind these and their actual imported dependencies anew.
