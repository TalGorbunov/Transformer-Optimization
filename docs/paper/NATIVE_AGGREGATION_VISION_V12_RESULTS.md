# Dynamic reference centering improves complete native answers, with a remaining multi-digit failure

Subtracting a training-estimated irrelevant contribution at each actual generated prefix substantially improved all four frozen V10 models on the fixed exploratory subset. The model still generated its own integer and EOS, using one batched native forward per token. No parameters were fitted in this study.

| Frozen core | N32 base → centered | N64 base → centered | N64 centered, K10–16 |
|---|---:|---:|---:|
| CE, seed14 | 2 → 14 /17 | 1 → 10 /17 | 1/7 |
| CE, seed15 | 3 → 14 /17 | 0 → 12 /17 | 2/7 |
| Consistency, seed14 | 5 → 14 /17 | 3 → 12 /17 | 2/7 |
| Consistency, seed15 | 6 → 16 /17 | 4 → 12 /17 | 2/7 |

The norm-matched orthogonal control reached only1/17,1/17,0/17,0/17 atN64. All corrected outputs were parseable integers and completed with native EOS. Across the four fitted models, corrected N64 counts0–9 were39/40 correct; counts10–16 were7/28. Those pooled figures describe the failure pattern; they do not create additional independent seeds. Most higher-count errors underestimated by one or two, but CE14 also generated130 for15 and140 for16. Format completion alone does not ensure numerical precision.

The reference bank contains the24 image occurrences from each question's N8K0 andN16K0 training scenes. Each native forward processes Nactual streams,24reference streams and one global stream. Every stream receives only the actual global generated prefix. The correction is `(N-16)*Wagg*mean_reference_message` before the unchanged SiLU readout, native residual addition, norm and vocabulary head. Allthree modes use the same augmented inputs and compute the reference messages. Thus the extra width is controlled in this diagnostic, but it is not free and the labeled bank remains question-specific.

The independent report verified all408 trajectories, raw argmaxes, exact FP16→FP32 logit preservation, EOS/integer scoring, reference inputs, frozen checkpoints, source hashes and calls. Study jobs used1162GPU-seconds,1006model forwards and408visual prefills. Software verification used130GPU-seconds,46model/24vision calls and46same-captured head replays. All software bindings and780descriptive cache/full comparisons passed. TotalV12 was1292GPU-seconds. The initial conservative600-second/model timing gate failed and was prospectively extended to1200; each actual study job finished in290–291seconds. The source-only dtype assertion was repaired before selected inference, with the exact native promotion independently checked on CPU.

These are17 previously inspected paired families, not fresh confirmation. The result supports a correctable accumulated background component in this setting. It does not establish a new attention operator, a bank-free method, successful high-count extrapolation or reasoning composition. V10's failed practical criteria remain unchanged.

The next method hypothesis is to learn the conditional null-message mean and train directly on `sum(messages)-N*predicted_null`, removing theN16 anchor and reference streams. Its matched control should have the same predictor and auxiliary examples, using that predictor as an ordinary once-per-query offset. The incomplete frozen-reference procedure will not receive a large fresh confirmation campaign before that comparison is designed. See the [learned-null proposal](NATIVE_AGGREGATION_LEARNED_NULL_PROPOSAL.md) and [literature positioning](NATIVE_AGGREGATION_NULL_REFERENCE_POSITIONING.md).

[Independent report and all raw results](../../outputs/native_aggregation_vlm/v12/reference_study/report_442359/REPORT.md) · [Execution ledger](../../outputs/native_aggregation_vlm/v12/INDEX.md).

[Verified complete-answer figure](../../outputs/native_aggregation_vlm/v12/figure_442365/native_vision_v12.pdf) (CPU442365; all36 plotted values bound to the independent report).
