# Synthetic correct-count latent: native first-token capacity

The correct K is supplied to the readout. These are oracle scores, not MMReD vision accuracy.

| Readout | Fit /72 | New questions /72 | New lengths /72 | Both /72 | Fit capacity |
|---|---:|---:|---:|---:|---|
| identity | 13 | 13 | 21 | 21 | FAIL |
| silu | 62 | 41 | 30 | 17 | FAIL |

Correct K is supplied explicitly as z=K e1; this is not vision aggregation accuracy.
Only first-token0..8 capacity is tested; no EOS, whole answer, multi-digit, cache update or reasoning.
Held oracle cells are descriptive and did not select checkpoints or hyperparameters.
Failure of this fixed optimizer is not an architecture impossibility proof.
