Saved-state query geometry only. All26 paired captures and every final scene were retained.

| Arm/factor | Actual saturated | Query removed | Payload partial mean, actual | Query removed |
|---|---:|---:|---:|---:|
| product/a | 0.9451 | 0.7259 | 0.0112149 | 0.0880957 |
| product/b | 0.9442 | 0.7657 | 0.0116811 | 0.0709504 |
| additive/a | 0.9324 | 0.7479 | 0.0076976 | 0.0588388 |
| additive/b | 0.9314 | 0.7597 | 0.00770993 | 0.0515858 |

Actual and query-removed saturation use consistent FP64 tanh of FP32 preactivations; the latter independently recomputes Wlocal*x+b. Captured-factor derivatives and subtraction roundoff remain descriptive. Payload partials exclude the outer half-weight and are not loss gradients. No downstream prediction was recomputed. Large query norms alone do not identify pathology. Training batches mix name and EOS positions and differ in inputs and weights; their summaries are not a causal learning trajectory.
