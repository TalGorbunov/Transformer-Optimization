# V11 software: placing aggregation before the final transformer block

Software preparation only. No efficacy training or new tests released yet. The comparator reads the same local/global states after the penultimate block; pre_last writes before the final frozen block, post_last writes before final normalization. Both retain the native SUM/SiLU core and one batched model invocation per output token.

The earlier write can enter final-layer global K/V and receive gradients from later-token losses under fixed teacher-forced inputs. Accuracy gains alone would not distinguish this persistent path from the added frozen nonlinear computation.

CPU source/layout/causal-gradient checks precede one GPU software profile up to5minutes. Total software envelope0.15GPUh(540seconds), including failures. No main training, new architecture-width/layer search or reasoning efficacy release. [Prospective protocol](../../../PREREG_AGG.md).

- [CPU causal-memory and actual-Qwen tests, 442222](memory_software/selftest_442222/summary.json): all 21 passed; no GPU or efficacy work.

- [Matched write-placement design](../../../docs/paper/NATIVE_AGGREGATION_VISION_V11_DESIGN.md): protocol now preregistered; main training still requires measured release.

- Native software [profile442259](memory_software/profile_442259/summary.json) passed;114GPU-seconds including the preserved logging failure442246. Full V11 efficacy is held while the null-reference follow-up is tested.
