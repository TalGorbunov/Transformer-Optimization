# outputs/judge_fenced — training-free fenced judge, TEXT (PREREG_AGG 2026-09-07 A2/A3/B1 + fence control)

Script `outputs/_scratch/dbg/judge_fenced.py`; frozen Qwen2.5-7B-Instruct; no trained parameters; verdict = logit(Yes) − logit(No) at the
block's " Answer:" token; count = #(verdict > 0). n = 100 per cell. Canonical run: `Qwen2.5-7B-Instruct/20260907_111006_4003053` (job 428210).

| config | MMRED exact N = 8 / 16 / 32 / 64 / 128 / 256 / 512 / 1024 | MMRED per-unit | needles HC exact N = 64 / 128 / 256 | needles per-unit |
|---|---|---|---|---|
| direct cue, fence | 1.00 ×8 | 1.000 | .89 / .92 / .91 | .995 / .995 / .998 |
| meta cue, fence | 1.00 / .79 / .98 / .99 / .97 / .99 / 1.00 / 1.00 | .987–1.000 | .19 / .09 / .10 | .897 / .858 / .926 |
| no cue (prefix only), fence | .49 / .16 / .07 / .22 / .41 / .66 / .66 / .80 | .914 / .795 / .894 / .958 / .985 / .998 / .998 / .9996 | .55 / .47 / .43 | .978 / .970 / .984 |
| direct cue, NO fence | .65 / .14 / .18 / .01 / .00 / .00 / .00 / .00 | .94 / .86 / .87 / .80 / .80 / .71 / .61 / .54 | .04 / .01 / .00 | .67 / .62 / .63 |

AUC of the verdict is ≥ .998 in every fenced cell (ranking perfect; the cue calibrates the threshold); without the fence AUC falls to .88 (MMRED
N = 64) and .71 (needles N = 256). Trained-head reference (outputs/bindcount): 1.00 on all of these cells.
