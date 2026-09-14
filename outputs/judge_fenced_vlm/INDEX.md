# outputs/judge_fenced_vlm — training-free fenced judge, VISION (PREREG_AGG 2026-09-07 A1/A3 + fence control)

Script `outputs/_scratch/dbg/judge_fenced_vlm.py`; frozen 4-bit Qwen2.5-VL-7B; per-frame cue "Is {who} in the {room} in this image? Answer:";
verdict = logit(Yes) − logit(No) at the ":" token; count = #(verdict > 0); rendered MMRED (data/mmred_vfiltered), n = 100 per cell.
Canonical run: `20260907_111236_4004643` (job 428234).

| config | exact N = 8 / 16 / 32 / 64 | per-frame verdict acc | AUC |
|---|---|---|---|
| direct cue, fence + M-RoPE reset | .98 / .97 / .99 / .98 | .9975 / .9963 / .9994 / .9997 | 1.00 |
| meta cue, fence | .74 / .62 / .69 / .67 | .951 / .947 / .974 / .988 | .982–.998 |
| direct cue, NO fence (causal, native positions) | .33 / .41 / .31 / .31 | .766 / .919 / .955 / .974 | .918–.992 |

References on the same data: trained bind-then-count .99 / .99 / 1.00 / .99 (outputs/bindcount_vlm); frozen model full prompt .20–.40; frozen
model given only the gold frames .65–.80 (outputs/mmred_vfrozen).
