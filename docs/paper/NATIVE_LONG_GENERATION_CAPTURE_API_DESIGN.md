# Shared cold native generation and bounded capture

Design only. This note proposes one new inference loop for the Qwen2.5-VL family, including the pinned Cosmos-Reason1 backbone. It does not modify a frozen runtime, authorize execution, select checkpoints, or establish a resource budget. Dataset parsing, training, feature harvesting and accuracy reporting remain outside the loop.

## Minimal interface

```text
prepare_ordinary_prefix(model, bound_case, bound_features) -> ColdPrefix
prepare_memory_prefix(model, bound_text, bound_memory) -> ColdPrefix

generate_native_cold(model, tokenizer, prefix: ColdPrefix,
                     generation: GenerationPolicy,
                     capture: CapturePolicy,
                     sink: EvidenceSink) -> TraceDescriptor
```

The two prefix builders adapt existing `ordinary_packet` and `compose_embeddings` mechanics. They produce the same structure: actual first-call `inputs_embeds`, prompt width W, actual four-axis prefill positions, an affine continuation-position rule, and immutable provenance. Ordinary input preserves its native mRoPE positions and continuation offsets `[0, delta, delta, delta]`; memory uses sequential positions and zero offsets. The loop never reconstructs an image layout from memory-slot placeholder IDs. It consumes already verified features or memory; it calls neither the vision tower nor the memory core.

`ColdPrefix` binds the base/checkpoint identity, active adapter and memory state, processor/tokenizer/template, exact prompt policy and token boundary, case SID, feature/coordinate descriptors, and actual prefix embedding identity. Describe the weight-hash coverage actually available; a revision string or architecture match is not a checksum of weight contents. Feature ownership must match the current backbone. Prefix embeddings use the current input embedding weights; memory records its original and native-cast identities. The prefix builders retain the complete native prefill positions once, or reference their already-bound prepared artifact. No target, gold answer or answer parser enters this API.

`GenerationPolicy` freezes maximum new tokens, total-context limit, both native EOS IDs, greedy tie semantics, and all generation overrides. The initial implementation is batch-one unrestricted greedy generation: no sampling, beams, repetition transform, vocabulary mask, minimum length or answer-tag stop. Both direct and reasoning Cosmos policies have the same 4,096-token ceiling and stop naturally at EOS. The native contract supplies hidden/vocabulary widths, layer/KV layout and actual dtypes; these are not silently inherited from the Qwen experiment.

`EvidenceSink` has an exclusive run directory, a small append-only step journal, unique selected-head packets, and final/failure manifests. It receives observations during actual calls. It never regenerates a token, duplicates a completed trajectory, or holds a full trajectory's logits in a Python list. Publish the successful completion marker last; preserve flushed partial records if execution fails. Forced process termination may leave the current unfinished record absent, which must be reported rather than synthesized.

## Invariants at every actual call

Number emitted tokens from t=1. The first call consumes the complete cold prefix with no prior cache. Later calls consume exactly the immediately preceding generated ID. Before a later call the cache length is W+t−2; after every call it is W+t−1. Therefore the final generated token, including EOS, has not itself been fed back. Record actual lengths for every decoder layer and require consistent batch/head layout. The cache object belongs to this trajectory; do not impose byte immutability on keys/values that native decoding legitimately appends.

Compare observed positions against the supplied native prefill positions or affine single-query continuation. `cache_position` is `[0,W)` initially and the single current index thereafter. The actual attention mask must have the expected dtype and shape `[1,W+t−1]`, with every entry one. Validate this at the call boundary, then record its canonical all-ones descriptor and validation result. Do not copy, hash or serialize a growing mask at every step. Persist the first position tensor once and four observed integer positions per subsequent token; do not accumulate duplicate expected-position tensors.

Observe one model, language, final norm and selected-row head call per emitted token, all decoder-layer call counts, and zero vision calls. Retain actual norm/head shapes, dtypes and finiteness checks. The selected normalized row must equal the actual head input; actual returned logits must equal that head's row. Derive the next ID from the actual unmodified logits, with deterministic lowest-index handling of ties. Save that ID, its logit, second-highest logit, tie count and EOS status, alongside the equality/finite checks. Reductions and comparisons may run on-device; unsampled rows need no full-vocabulary CPU copy. These scalar observations are runtime evidence, not an independently replayable proof of every unsampled argmax.

A trajectory must stop at its first generated native EOS or the fixed token ceiling. Parser completion never changes execution. Verify evaluation/frozen-gradient mode, active adapter configuration and parameter identities/versions before and after; restore the original mutable rope state and remove hooks even on exceptions. Cold prefixes have no independent-question KV reuse claim.

## Prospective head schedule and storage

Use the same deterministic schedule for both prefix types and both instruction policies. For each selected trajectory retain complete native head evidence at emitted steps 1–8, every multiple of 128, and the final completed step, deduplicated and intersected with actual emitted steps. At most 40 rows are retained for a 4,096-token trajectory. Final-step inclusion is a predeclared rule, including an early EOS or the last completed call before failure; it is not a correctness-based choice. A one-row device-resident rolling buffer permits terminal retention without copying every unsampled head to CPU or keeping previous heads; its clone/comparison cost remains billed. Selected records contain pre-norm input, normalized row, actual head input and actual native-dtype full-vocabulary output, each saved once. An FP32 view needed by a CPU checker is derived from that saved output, not saved as another actual head invocation.

`CapturePolicy` must contain a frozen set of trajectory identities, this step rule, a whole-run maximum replay-row count, and the source/policy hashes. Software profiles should select all their predeclared trajectories. A benchmark may select a fixed SID subset, chosen by source order within declared strata before predictions and shared across arms/policies; its exact membership and resulting CPU/storage cap require a separate release. This API does not authorize multiplying 40 rows by an arbitrary benchmark size. All trajectories retain every emitted ID and the compact runtime journal, including those outside the numerical sample. Report eligible, emitted, captured and actually replayed counts separately. Never backfill short completed traces or replace selected failures with other examples.

With the current H=3,584, V=152,064 and FP16 head evidence, 40 rows of three hidden vectors plus one head vector occupy about 13.03 MB before container overhead. Persistent trace size is O(W + T·layers + K·(V+H)), with K≤40, rather than O(TV) duplicated logits plus O(T²) masks. This is a storage claim: constructing/checking native masks and attention still costs work. Cache growth also remains native; for 28 layers, four KV heads and head width128, FP16 KV adds 57,344 bytes per cached token (about224 MiB for4,096), before prefix/cache implementation overhead.

## Software proof and release boundary

Before replacing any caller, compare old and new loops on identical fixed short prefixes: ordinary native positions, zero-delta memory, early EOS, first-token EOS and a length-censored trace. Require exact same-backend IDs and logits at every tested step, no additional forwards, and exact source/state ownership. Exercise the head schedule, terminal deduplication, partial publication, incorrect incoming ID, mask/position/cache mismatch and restoration-on-failure. A separate explicitly labeled capacity fixture may advance a cache to the long boundary; its forced continuation is not a natural efficacy trace.

Then measure at least one actual long-cache path, selected-head capture/CPU replay, journal/publication cost and peak memory under the proposed policy. The final replay tolerance and accounting caps must be fixed before those outcomes; sampled replay does not authorize a claim of exhaustive head fidelity. The shared loop returns IDs, completion and artifact references. Direct-JSON and think/answer typed parsers remain separate pure reporting functions, so adding a reasoning format does not fork the decoder pipeline.
