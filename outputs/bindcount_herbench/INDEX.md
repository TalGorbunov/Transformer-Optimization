# outputs/bindcount_herbench — bind, then count on HERBench Action Counting (real egocentric video)

Data: `data/herbench_ac/ev_fill{16,32}` (HERBench AC questions, HD-EPIC frames at t+0.3 s + fillers; split by video). Script `outputs/_scratch/dbg/train_bindcount_vlm.py --dataset herbench`. Test n = 150 (51 held-out videos).

| arm | run | N=16 exact (verdict acc / AUC) | N=32 exact (acc / AUC) |
|---|---|---|---|
| fenced carriers + sum read | `sum/20260903_194022_2677497` | .12 (.683 / .774) | .04 (.685 / .788) |
| carriers, no fence, + sum read | `sum_nofence/20260903_194022_2677495` | .153 (.749 / .791) | .127 (.817 / .799) |
| frozen VLM, full prompt | (in the same logs) | .12 (MAE 3.6) | .047 (MAE 4.7) |
| frozen VLM, oracle frames only | (in the same logs) | .02 | .027 |

Reading: perception ceiling (per-frame AUC ≈ .78); the fence hurts on real frames (H3 falsified); recorded in PREREG_AGG.md.
