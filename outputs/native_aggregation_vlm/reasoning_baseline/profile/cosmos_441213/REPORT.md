# Frozen reasoning baseline software profile

The primary contrast is within each fixed model: reasoning512 minus direct32 on paired N32/N64 scenes. All malformed/truncated outputs remain wrong.

## cosmos

| Model | Endpoint | N16 exact | N32/N64 exact | OOD parsed | OOD truncated | Parsed OOD MAE |
|---|---|---:|---:|---:|---:|---:|
| cosmos | direct32 | 0/2 | 0/2 | 2/2 | 0/2 | 4.000 (2) |
| cosmos | reason128 | 0/2 | 0/2 | 0/2 | 2/2 | — |
| cosmos | reason512 | 1/2 | 0/2 | 0/2 | 2/2 | — |

cosmos: primary difference +0.00 pp (paired-anchor interval [+0.00, +0.00]); 5 pp descriptive screen: False. N16 difference +50.00 pp.
Measured total 70.7s; generation 53.7s; 1925 generated tokens.
direct: 4.5s generation, 36 tokens (9.0/scene), mean prompt 8009.0 tokens.
reason: 49.2s generation, 1889 tokens (472.2/scene), mean prompt 8006.0 tokens.
Profile cost projection for all54 main scenes at both caps: 1697.4s; within26minutes: False.

## Interpretation limits

- Frozen-model baseline assay only; no improved aggregation method, external tally, oracle prefix, training, or composition experiment.
- Primary contrasts compare policies within each fixed model. Cosmos and Qwen differ in pretraining/post-training; their difference is not a causal reasoning effect.
- System instruction and output budget change together. This tests a prompted-reasoning policy, not the isolated causal effect of adding a token.
- The official Cosmos guide recommends at least 4096 output tokens and reports BF16 testing. This bounded assay uses 512, NF4/bf16, greedy decoding and repetition penalty1; negative or truncated outcomes do not establish general reasoning failure.
- The 128-token endpoint is censored from the same 512-cap trace, with no independently measured 128 latency. Later normal EOS is incorrect at128, even if a prefix already contains an answer.
- Whole-trace grammar and normal EOS are required. Every malformed/truncated output remains in exact denominators. MAE is conditional on parsed, completed outputs.
- Intervals resample18 entire paired anchors and are descriptive for these fixed models/scenes, not seed-population inference. No prompt/model/budget selection is performed.
