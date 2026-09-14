# Frozen ordinary joint-image reference on fresh N16 joins

This is held source preparation, not an execution release. Evaluate the unadapted Qwen backbone on the exact270 published fresh N16 scenes used by the fixed product/additive models: A108, B108 and C54. This is an input/backbone-matched frozen reference. It does not match the trained models' adaptation budget, establish equal compute, or replace a future matched joint-adapter control.

## Native inputs and fixed outputs

Require the passed fresh render proof and its bound complete V2 native training report. Freeze the same manifest, ordered panel/family/variant rows, image/QA hashes and canonical target IDs. Each model input contains only SID, N, question and image descriptors. Preserve every published image in manifest order, including its Step label, RGB pixels and the existing392×392 resize. Form one ordinary user message: all16 image entries, then the **exact unchanged question** asking for the person's name; apply the original processor's generation chat suffix. No count wrapper, binary prompt, candidate list, inferred relevance, factor adapter, separate global row, global-token broadcast or external summary is supplied. The V7 joint helper's image packing is useful precedent, but its integer-answer text wrapper is inapplicable here.

Use the identical immutable Qwen snapshot, processor and installed native API, NF4/double quantization/BF16 backbone, native FP16 final norm/head, SDPA and bound precision settings. Freeze every model parameter. Invoke ordinary native greedy generation with `logits_to_keep=1`, maximum4 tokens, native EOS IDs151645/151643, and no vocabulary mask, penalty or additional logits processor. Exact success requires the complete canonical name target IDs including EOS151645. Preserve first-token results, noncanonical outputs, early EOS and truncation separately; none replaces exact scoring.

Audit the single joint sequence's original IDs/mask, image ownership, four-axis text/mRoPE positions, actual cached raw-token history and cache lengths. One vision prefill processes all16 images; subsequent calls use the ordinary native cache. Restore hooks/rope state on exceptions and retain partial observations before failure. No exhaustive all-layer KV identity claim is made.

## Shape and cost release

One CPU preparation job,300seconds/four cores/16GiB, verifies all270 native joint bundles and freezes three profile cases: the largest actual padded joint prompt width within each A/B/C panel, choosing the first manifest row on ties. Selection uses inputs only and precedes predictions. Profile outcomes never determine eligibility or prompt choice.

GPU launch verifies the complete source/plan ledger and every non-bundle input upfront. Only prepared image-bundle hashing is deferred to immediately before each consumed trajectory, inside its measured preparation time: three bundles for the profile and all270 for the main. CPU preparation and independent reports retain full270-bundle verification; no model, native identity or other input hash is skipped.

One120-second GPU profile executes those three natural trajectories, at most12 model/language/norm/head calls and three vision prefills. Nonmutating hooks record actual norm input/output and head input/output shapes. The installed decoder normalizes its full `[1,W,H]` prefill, then slices before the head when `logits_to_keep=1`; do not infer head coverage from that source alone. Require observed head input/output `[1,1,3584]`/`[1,1,152064]` at every call. Preserve unexpected shapes and stop rather than silently increasing replay scope.

Save the pre-norm last-query state, actual normalized head input and original full-vocabulary FP16 head output. The independent300-second CPU release replays the frozen per-token norm and actual one-query head: at most12 head calls/12 rows, requiring probability TV≤0.02 and exact argmax on every row. This covers each executed head query, not every prefill norm position. No extra GPU head is run.

Measure setup from driver entry through loading/verification. For each profile scene use `T=preparation + work*4/emitted_tokens`, including native execution, observations, raw publication, hashing and validation. Missing/nonpositive/nonfinite measurements fail. Require the **joint-only** forecast `setup + 1.25*270*max(T_A,T_B,T_C) +60 <=1200seconds` before the main. Borrow no parallel-stream timing bound. The forecast does not guarantee physical runtime.

One1200-second main evaluates all270 scenes once, at most1080 model/norm/head calls,1080 native head rows and270 vision prefills. The entire joint stage is capped at1320 GPU-seconds, including failed/cancelled/zero-allocation attempts, one GPU concurrently, no implicit retry. CPU release reserves the full1200 main allowance. Root alone submits. Prepared data/raw tensors and models follow the existing designated data/checkpoint roots; immutable source and input identities accompany every artifact.

## Independent result and limits

A300-second CPU report verifies every raw logit argmax, target/EOS outcome, input/history/position/counter record and unchanged native weights. It reuses the profile head proof without new head calls. Report all A/B/C whole-answer and triple counts, orientations, names and questions. Retain literal A/B98/108 plus33/36 criteria (effective99 correct), and separate C49/54 plus16/18. Baseline scores do not select a parallel checkpoint or override the registered parallel N16-first stop rule.

Only a passed parallel fresh report with the identical manifest/cohort permits descriptive paired comparisons against both fixed endpoints; missing results remain pending, never dropped. There is no new comparative significance or superiority threshold. N32/N64 remain outside this release.

[V7's software profile](../../outputs/native_aggregation_vlm/v7/runtime/profile_441845/REPORT.md) and [independent result](../../outputs/native_aggregation_vlm/v7/report_441910/REPORT.md) demonstrate historical joint-image execution at16/32/64 frames with counting prompts. They do not validate this name prompt or its resource bound. The image-free global stream is not this ordinary joint-image baseline; a frozen-reference gain would still combine adaptation and evidence-routing differences.
