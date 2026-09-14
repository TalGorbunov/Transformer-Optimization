# Independent mixed-oracle CPU audit

Audit: PASS. Original numerical gate: FAIL.

All92state files,492logit vectors and246comparisons verified. Original failures retained:1.
Maximum reproduced metric difference:7.22e-15; audit arithmetic tolerance:1e-10.

No new model forward, fitting, threshold adjustment or original artifact mutation.
Full-vocabulary metrics were independently recomputed from all saved native logits on CPU.
Actual internal causal-mask tensors and K/V values were not saved; original exact assertions remain source-bound evidence.
Saved logical positions, key masks, cache positions, all cache geometries and assertion bindings were independently verified.
This audit does not make the original failed numerical gate pass or establish aggregation/reasoning efficacy.
