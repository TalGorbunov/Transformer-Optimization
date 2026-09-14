# Learned aggregation in native decoder memory: software results

**The independent software audit passes for Qwen and Cosmos.** An earlier aggregate written before the final decoder block changes later native predictions through that block's global KV state. Restoring the affected KV slot removes the effect. The corresponding final-norm write has no persistent effect when subsequent tokens and writes are held fixed. This establishes an execution and gradient path; no adapter was fitted and no answer accuracy was evaluated.

[Verified report](../../outputs/native_aggregation_vlm/learned_memory/software/report_precision_443156/summary.json) · [Analysis and provenance](../../outputs/native_aggregation_vlm/learned_memory/software/report_precision_443156/analysis.json) · [Frozen CPU plan](../../outputs/native_aggregation_vlm/learned_memory/software/check_443132/plan.json).

Both placements read the same block-26 states. The `pre_last` condition writes before block 27; `post_last` writes before final normalization. The rank-96 learned-selection core has 1,041,697 FP32 parameters. Its fixed zero and active states use seed 20261124, with no optimization or model selection. Tests use two training-derived N16/N64 image contexts, native FP16 residual cast-before-add, separately bound Qwen/Cosmos weights, and fixed continuation tokens.

Each model passes 12 zero-output identity checks and 24 common-read/all-layer KV contrasts. Lower-layer KV, all local-row KV and earlier prompt entries remain unchanged. Active `pre_last` changes both current global K and V in the final block; `post_last` leaves native KV unchanged. These all-layer digest checks cover the fixed-token calls. Natural generation additionally checks masks, cache lengths, current queries and token history.

In the N16 persistence test, the first global write is perturbed while the next token and next applied write are kept identical. The next global distribution changes under `pre_last`: TV is 0.00001435 for Qwen and 0.00039407 for Cosmos. Neither changes its argmax. Restoring only that first global position's final-layer K/V recovers every baseline cache byte and all next logits exactly. The analogous `post_last` perturbation already yields identical next logits without restoration. Local-row logits remain identical throughout.

The actual frozen, quantized final block also carries loss gradients through earlier writes. A fixed middle-position colon-token loss uses independent local, earlier, current and future delta leaves. The saved L2 norms are:

| Model | Placement | Earlier delta | Current delta | Local / future delta |
|---|---|---:|---:|---:|
| Qwen | pre_last | 0.0184598771 | 0.2991451740 | 0 / 0 |
| Qwen | post_last | 0 | 0.3073334093 | 0 / 0 |
| Cosmos | pre_last | 0.0360465915 | 0.4388442336 | 0 / 0 |
| Cosmos | post_last | 0 | 0.4234370403 | 0 / 0 |

These are derivative measurements, not fitted learning gains. Backbone parameters remain frozen. Full historical writes stay live while only the selected loss row reaches the vocabulary head. All 3,524 native-shaped head comparison rows pass with TV 0 and exact top1, including the four standalone final-block replays.

Cached versus full-prefix execution remains a separate, descriptive comparison. Qwen retains 14/328 numerical failures, maximum TV 0.0290202; Cosmos retains 74/328, maximum TV 0.1090844, including two argmax mismatches. None was filtered or converted into a pass.

The first independent report 443139 failed after 36 allocated CPU-seconds because exact dictionary equality rejected tiny `centered_rms` reduction differences. Diagnostic 443153 used 6 CPU-seconds and identified six one-ULP differences: two Qwen rows, maximum absolute 3.47e-18, and four Cosmos rows, maximum 6.94e-18. A separately frozen CPU reporter permits at most two FP64 ULPs only for that descriptive RMS field. Every other field, TV≤0.02, top1 and pass flag remains exact; the original source, failed report and diagnostic remain bound. [Retained failure](../../outputs/native_aggregation_vlm/learned_memory/software/report_443139/failure.json) · [Diagnosis](../../outputs/native_aggregation_vlm/learned_memory/software/report_diagnosis_443153/summary.json).

Qwen executed 47 VLM calls, 22 vision calls and 51 extra head projections; Cosmos executed 49, 22 and 53. Together this is 96 VLM calls, 44 vision calls, four standalone decoder-block replays and 104 extra head projections. Each GPU allocation lasted 131 seconds: **262 allocated GPU-seconds**, with two concurrent GPUs. Four public native-generation trajectories produced only 10 tokens in total. The public `forward_native` API was not executed; fixed forwards used the profile harness. No long-reasoning feasibility, beneficial memory, fitted composition or efficacy claim follows. The final analysis SHA256 is `f25a497da5eb91642cd4c25907088199784e4106c3f0a90f2a185648aa8cdaf4`.
