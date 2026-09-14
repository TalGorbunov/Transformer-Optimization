# outputs/fence_only — control: fence + reset only, frozen model answers the count itself (2026-09-08, not pre-registered)

Script `outputs/_scratch/dbg/fence_only.py`. Same layouts as the trained/training-free bind-then-count runs, no per-block question, no
carriers/heads; greedy digits at the tail. Canonical runs: `text/20260908_104326_139582/` (job 432240), `vision/20260908_104326_139580/` (job 432241).

| modality | config | exact by N |
|---|---|---|
| text (Qwen2.5-7B) | fence + reset, model answers | .16 / .07 / .07 / .07 / .07 / .10 / .03 / .05 (N = 8 … 1024) |
| text | plain causal, same prompt | .72 / .57 / .53 / .43 / .27 / .30 / .23 / .24 |
| vision (Qwen2.5-VL-7B) | fence + reset, model answers | .14 / .10 / .07 / .14 (N = 8 … 64) |
| vision | plain causal, same prompt | .16 / .18 / .19 / .19 |

Reference: fence + per-block question + count of yeses (outputs/judge_fenced, judge_fenced_vlm): text 1.00 at every N, vision .98 / .97 / .99 / .98.
