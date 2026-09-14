# Frozen reasoning baseline on fresh MMReD Vision scenes

The primary contrast is within each fixed model: reasoning512 minus direct32 on paired N32/N64 scenes. All malformed/truncated outputs remain wrong.

## qwen

| Model | Endpoint | N16 exact | N32/N64 exact | OOD parsed | OOD truncated | Parsed OOD MAE |
|---|---|---:|---:|---:|---:|---:|
| qwen | direct32 | 0/18 | 0/36 | 0/36 | 0/36 | — |
| qwen | reason128 | 0/18 | 0/36 | 0/36 | 0/36 | — |
| qwen | reason512 | 0/18 | 0/36 | 0/36 | 0/36 | — |

qwen: primary difference +0.00 pp (paired-anchor interval [+0.00, +0.00]); 5 pp descriptive screen: False. N16 difference +0.00 pp.
Measured total 136.3s; generation 86.1s; 219 generated tokens.
direct: 43.9s generation, 111 tokens (2.1/scene), mean prompt 7481.0 tokens.
reason: 42.2s generation, 108 tokens (2.0/scene), mean prompt 7478.0 tokens.

## Interpretation limits

- Frozen-model baseline assay only; no improved aggregation method, external tally, oracle prefix, training, or composition experiment.
- Primary contrasts compare policies within each fixed model. Cosmos and Qwen differ in pretraining/post-training; their difference is not a causal reasoning effect.
- System instruction and output budget change together. This tests a prompted-reasoning policy, not the isolated causal effect of adding a token.
- The official Cosmos guide recommends at least 4096 output tokens and reports BF16 testing. This bounded assay uses 512, NF4/bf16, greedy decoding and repetition penalty1; negative or truncated outcomes do not establish general reasoning failure.
- The 128-token endpoint is censored from the same 512-cap trace, with no independently measured 128 latency. Later normal EOS is incorrect at128, even if a prefix already contains an answer.
- Whole-trace grammar and normal EOS are required. Every malformed/truncated output remains in exact denominators. MAE is conditional on parsed, completed outputs.
- Intervals resample18 entire paired anchors and are descriptive for these fixed models/scenes, not seed-population inference. No prompt/model/budget selection is performed.
