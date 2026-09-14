# V18: protect reliable local judgments during vector aggregation

V17 found that the frozen model correctly distinguishes relevant from irrelevant images in the canonical MMReD Vision scenes where native aggregate answers fail. V16 separately showed that irrelevant rendered Step labels can change learned background contributions substantially. V18 tests a simpler interface: preserve the native local decision while learning the vector payload and aggregate readout. [V17 evidence](NATIVE_AGGREGATION_VISION_V17_RESULTS.md), [V16 evidence](NATIVE_AGGREGATION_VISION_V16_RESULTS.md).

At the last token of the original one-image prompt, apply the model's existing native norm and vocabulary head. Compute full-vocabulary probabilities and cache

`g_i = stopgrad(max(0, p_i(1) - p_i(0)))`.

At each generated position, the trainable branch computes

`q = Wq RMS(h_global)`

`m_i = g_i * tanh(Wlocal RMS(h_i) + q + b_local)`

`delta = U SiLU(Wagg SUM_i(m_i) + q + b_agg)`.

The residual is added to the current global query before the native final norm and head. Every payload coordinate passes through the gate. Later positions use their current states and the unchanged original gates. The model still has to learn a complete integer answer and EOS; no scalar tally is supplied.

The control replaces the applied gates with ones on real image rows. It performs the same original probe, probability calculation, gate caching and dense payload work. Padding remains zero in both arms. Both have 1,041,600 trainable FP32 parameters, a rank-96 payload, and zero-initialized U. Canonical pixels, prompts, backbone and ordinary generation are identical. There is no reference bank, estimated background mean, extra predictor or image augmentation.

For T generated tokens, execution uses T batched VLM forwards over N image streams plus one text-only global stream, one vision prefill, T ordinary norm/head calls and one extra original-query norm/head probe. The extra head computation and repeated local prompts have a real cost. A successful result would concern aggregation within each forward; it would not establish a free increase in model capacity or reasoning composition.

| Item | Fixed choice |
|---|---|
| Conditions | Native gate; matched all-open control |
| Seeds | 20 and 21, matched initialization and presentation order |
| Training | Original V10 corpus, N≤16, answer counts 0–16 |
| Objective | Complete native sequence CE + paired residual consistency, coefficient 1 |
| Schedule | 40 epochs, 4,590 updates, 16 scenes per update |
| Endpoint | Final update only; one descriptive 64-scene N16 development sweep |
| Fresh test | 136 paired N32/N64 families, eight per count 0–16; 272 scenes total |
| Freshness | Exclude all 30 prior manifests at complete context/question level; visual atoms may recur |
| Decoding | Raw global greedy argmax, four-token cap, native EOS, no vocabulary mask |
| Correctness | Clean ASCII integer body with terminal EOS; invalid/truncated outputs remain failures |

The primary comparison must pass in both seeds: at least 7/136 additional N64 exact answers, with at most 6/136 lost at N32. Practical success additionally requires both gated models to reach at least 123/136 at N32, 109/136 at N64, and 52/64 on N64 counts 9–16. These thresholds concern length extrapolation with trained answer support. They define a vision milestone, not the complete research objective.

Source and native software checks precede training. The native profile verifies original probe logits, zero-U identity, global-only writes, all-layer fixed-history KV preservation and bound gate reuse. Captured-state head replay is a hard numerical check; previously documented cached-versus-full-prefix kernel differences are retained separately. The training profiles verify gradients, full-sequence loss arithmetic, gate isolation, checkpoint restoration and native readout on fixed old examples without test scoring.

Four full fits require a passed independent data/source audit and a measured resource release. Each full GPU allocation is capped at 2,700 seconds; the whole V18 campaign is capped at 14,400 GPU-seconds including failures. At most four project GPUs run concurrently. All model and heavy CPU work uses Slurm, with data and checkpoints in the user-specified roots. [Execution record](../../outputs/native_aggregation_vlm/v18/execution.json), [append-only protocol](../../PREREG_AGG.md).

The gate remains an explicit counting prior. A false negative closes an image permanently for that trajectory; later reasoning cannot reopen it. Positive weights and payload directions can vary, and the native decoder can still fail. Gating and vector pooling are familiar components. The scientific question is whether this particular preservation of an already reliable native judgment solves the observed aggregation failure under a matched test. Broader tasks, models and reasoning benefits require further experiments.
