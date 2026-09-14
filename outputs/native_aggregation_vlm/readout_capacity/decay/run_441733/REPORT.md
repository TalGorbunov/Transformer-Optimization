# Synthetic correct-count latent: native first-token capacity

The correct K is supplied to the readout. These are oracle scores, not MMReD vision accuracy.

| Readout | Fit /72 | New questions /72 | New lengths /72 | Both /72 | Fit capacity |
|---|---:|---:|---:|---:|---|
| identity | 72 | 67 | 49 | 45 | PASS |
| silu | 72 | 62 | 49 | 37 | PASS |

Correct K is supplied explicitly as z=K e1; this is not vision aggregation accuracy.
Only first-token0..8 capacity is tested; no EOS, whole answer, multi-digit, cache update or reasoning.
Held oracle cells were already observed in initial441720 and are explicitly reused descriptively.
Duration and schedule jointly change; this does not isolate their individual effect.
Failure of this fixed optimizer is not an architecture impossibility proof.
