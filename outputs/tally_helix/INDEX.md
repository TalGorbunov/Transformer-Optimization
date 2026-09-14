# outputs/tally_helix — in-model tally feasibility (PREREG_AGG 2026-09-07 S2 H1–H3)

Script `outputs/_scratch/dbg/tally_helix.py`; frozen Qwen2.5-7B-Instruct; number vectors v_k = mean answer-row residual over 6 copy templates
(k = 0 … 64; template copy accuracy 1.00); patch = replace the answer row of a plain MMRED prompt at layer L on every greedy step.
Canonical run: `Qwen2.5-7B-Instruct/20260907_111230_1973535` (job 428235).

| layer | emitted = k, k ≤ 8 | 9–32 | 33–64 | R² helix+linear / linear |
|---|---|---|---|---|
| 8 | .22 | .00 | .00 | .698 / .383 |
| 12 | .16 | .00 | .00 | .718 / .441 |
| 16 | .16 | .00 | .00 | .674 / .410 |
| 20 | .17 | .03 | .00 | .663 / .409 |
| 24 | 1.00 | .26 | .01 | .401 / .128 |

H3 (k = 33 … 64 at layer 24): 0.00 from the true vector, the helix fit on k ≤ 32, and the linear fit. Frozen no-patch exact on the same
prompts .40 / .30 / .45 (N = 64 / 128 / 256, n = 20). One patched row carries one token under digit-by-digit tokenization.
