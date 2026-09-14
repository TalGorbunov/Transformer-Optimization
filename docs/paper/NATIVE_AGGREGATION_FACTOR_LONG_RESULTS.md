# Both locked visual factor models fit with the calibrated schedule

Product and additive each achieve **108/108 correct training first tokens and 18/18 complete families** after 6,000 updates. Independent report 443543 passed. This establishes training fit on cached native features; native complete-answer generation and fresh-data extrapolation are the next tests.

| Model | Step 600 | Step 2,000 | Step 6,000 | Final mean first-token NLL |
|---|---:|---:|---:|---:|
| product | 78/108 | 108/108 | 108/108 | 0.0002168 |
| additive | 36/108 | 99/108 | 108/108 | 0.0046020 |

Both runs started from the exact original unfitted factor checkpoint, with the same 108 training scenes, fixed global conditioning, frozen native features and head, seed, full-name-plus-EOS CE and optimizer. The intervention changed both the optimization horizon and its cosine annealing schedule. Earlier short-run failures remain recorded; the comparison does not isolate additional steps alone. Only the fixed step-6,000 checkpoint determined acceptance.

The product model learned this training task earlier than the additive control. Because both ultimately fit, these results do not establish that multiplication is necessary, that the learned factors represent explicit semantic attributes, or that either model generalizes. The architectures use known nonlinear set-pooling ingredients. A matched ordinary joint-image baseline and a same-backbone reasoning comparison are still needed for the broader research objective.

The independent audit reconstructed the original inputs, all 12,000 CE updates and 58 saved computations. All 648 endpoint rows passed exact top-1 replay with maximum total-variation distance 0.003901. Intermediate observations preserved the optimizer and model state. Both fits used 159 allocated GPU-seconds in total, with at most two GPUs. The audit used 39 CPU allocation seconds. There were no VLM or vision calls in these cached fits.

Next, both fixed checkpoints undergo natural image-to-answer generation on all training contexts. If native trainability passes, the predeclared fresh panels test new scene realizations, new person combinations, held questions and N16→N32→N64 length scaling. Neither fresh scores nor intermediate checkpoints select a model.

[Independent report](../../outputs/native_aggregation_vlm/identity_join_factor_long/report_443543/REPORT.md) · [Frozen protocol](NATIVE_AGGREGATION_FACTOR_LONG_PROPOSAL_V2.md)
