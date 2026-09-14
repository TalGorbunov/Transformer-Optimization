# Native semantic aggregation during reasoning: bounded software check

The four-prefill Cosmos audit passed its fixed direct-local feasibility criterion on the two old software scenes. All80 local native judgments were correct, all71 negative gates were exactly zero, and positive gates ranged from0.997967 to0.999606. Minimum full-vocabulary digit mass was0.999998809. The reasoning-local prompt had negligible digit mass and did not supply a usable semantic gate. All164 native replay rows had zero total variation. These observations support checking direct local prompts with a reasoning global stream; they do not establish aggregate accuracy. [Probe report](../../outputs/native_aggregation_vlm/reasoning_semantic_gate/report_442926/analysis.json).

Use exactly the completed probe's direct-local N16/K3 and N64/K6 prepared bundles and the same Cosmos model/processor/native identity. At each length generate three trajectories: bare, zero-U semantic branch, and active semantic branch. The private seed20261118 fixes one untrained rank96 core; the active version changes only U to Normal(0,0.001). No V18 fitted checkpoint is transferred. Generation is native unmasked greedy decoding with a maximum of eight tokens and ordinary native EOS.

Each zero/active trajectory computes the original local probability-gap gate once and reuses it at every later position. Current local/global states still produce new bounded vector payloads. Every actual global token, including reasoning and special tokens, is broadcast verbatim to the local rows. Local instructions request a direct answer, so those later continuations are a known distribution mismatch. Software validity alone cannot show that the learned payload will be useful there.

Let G be the number of generated tokens across all six trajectories, and A the number across the two active trajectories. Replay every active prefix, including the empty prefix, in a full forward using its bound original artifact; no replay may recompute a gate. The exact upper inventory is:

| Operation | Inventory |
|---|---:|
| Native VLM, ordinary norm/head | G+A, at most64 |
| Vision | 6+A, at most22 |
| Original extra norm/head/probability probes | Exactly4 |
| Same-state standalone native-shaped norm/head replays | G+A, at most64 |
| Total extra norm/head calls | At most68 |
| Origin rows compared with the same-run bare prefill | Exactly164 |
| Cached/full active global comparisons | A, at most16 |
| Native decode transitions with all-layer old-KV checks | G-6, at most42 |

Require exact bare/zero-U token and raw-logit equality, exact original hidden/normalized probe identity where registered, head total variation at most0.02 and identical top1 for each hard replay comparison. Cached/full-prefix numerical differences remain descriptive and all are retained. Compare KV only within the same actual history. Every prior KV prefix at all28 layers must remain unchanged; final-norm fusion changes only the current global query with the native FP16 cast-before-add. Verify four-axis native generation positions, three-axis full replay positions, masks, query ownership, all-coordinate gating and closed coordinates, original gate reuse, and exception-safe restoration of hooks and caller rope state.

The new generation wrapper uses the existing immutable semantic controller/core and Cosmos global-only recorder. Retain bounded global vocabulary vectors, necessary origin artifacts and query captures. Do not enable long-trace Hugging Face all-row logit retention. Full-prefix replays happen after generation completes, without resuming a disturbed cache.

CPU preparation and independent report use Slurm, four cores,16GB and at most300seconds each. One B200 GPU allocation is capped at180seconds; the entire software campaign is capped at540GPU-seconds including failed and zero allocations. Reserve the full next allocation before loading the model and retain the complete accounting ledger. Maximum one campaign GPU and four project GPUs; exclude the previously failing node131-255. Data use the requested data root, initial core artifacts the requested checkpoint root, and source/proofs the repository output directory.

Freeze independently reviewed source and prepared inputs before execution. Preserve all partial failures. No fit, generated-answer accuracy, persistent-memory efficacy, or long-reasoning experiment is released by this check. A passed result would only establish that the native semantic branch can execute through real reasoning prefixes without breaking the model contract.
