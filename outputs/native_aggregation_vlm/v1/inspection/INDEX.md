# V1 checkpoint scale inspection

Canonical run: [20260910_154026_440563](20260910_154026_440563/inspection.json). Slurm440563 completed140s.

Two fixed profile examples at N16 and two at N64 per model. Algebraic description only; contextual hidden states already contain evidence.

| Arm | Branch / ordinary N16 | Branch / ordinary N64 | Zero-read / ordinary N64 | Read-dependent / ordinary N64 |
|---|---:|---:|---:|---:|
| sum | 37.57 | 47.67 | 171.79 | 124.24 |
| mean | 1.50 | 1.40 | 1.43 | 0.20 |
| post_sum | 9.32 | 4.13 | 77.10 | 77.24 |
| post_mean | 1.16 | 1.13 | 0.95 | 0.28 |
| global | 1.17 | 1.17 | 0.93 | 0.26 |
| hierarchical | 1.25 | 1.25 | 0.93 | 0.33 |

The sum branch is much larger than ordinary attention. Both sum-scaled arms show large zero-read and opposing read-dependent components at N64. Normalized branches remain around ordinary-attention scale. These observations identify a scale/cancellation concern, not a demonstrated causal explanation. The zero-read term is not an evidence-free term.

All observations used raw first-token forward outputs; these are different from V1 generation-policy NLL. They neither replace exact-answer results nor measure reasoning composition. Model weights, inputs, and hidden states were not intervened on.
