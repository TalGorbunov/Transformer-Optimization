# outputs/loramech/ — INDEX (experiment → canonical run → headline)

Campaign: LORAMECH, 2026-08-26→28 (brief `CAMPAIGN_BRIEF.md`, full log `STATE.md`,
RESULTS.md entries [2026-08-26→27] P0+C1 / [2026-08-27→28] P1-family / [2026-08-27→28]
mechanism). Exam files: `examdirs/` (150/cell, class-balanced, contamination-proof).

| experiment | canonical run | headline |
|---|---|---|
| P0 ff_le8 (frames-first SFT, train ≤8) | `p0_ff_le8/20260826_161912_lora/` (job 137467) | in-length N=8 **0.540** (Q-first anchor ~1.0) |
| P0 ff_le32 (frames-first SFT, ≤32) | `p0_ff_le32/20260826_171532_lora/` (137465, h200) | **H0 REFUTED**: 0.600/0.480/0.313 @8/16/32 |
| P0 zero-shot exams | `zs_ff_le8/20260826_173123_lora/`, `zs_ff_le32/` (137504/137593) | extremes relapse: ff_le32 @128 mid-range 0.026 |
| C1 Q-first pipeline control | `c1_qfirst_le8/` (137706) | template = the factor: **0.867 @8** (+0.33 within-pipeline) |
| C2 15-ep convergence bound | — (137707 CANCELLED, 3× h200 preemption) | closed by budget-parity + C1 |
| P1 fenced-SFT 5-ep | `p1_fenced_le16/20260827_131154_fenced/` (137744) | 0.993/0.847 in-length, live curves |
| **P1b fenced-SFT 10-ep (CANONICAL)** | `p1b_fenced_ep10/20260827_192346_fenced/` (137778) | **1.000/0.893 in-length; 0.867 @32 ZERO-SHOT** |
| P1b zero-shot 64/128 | `p1b_zs_64_128/` (137862) | wall @4×/8×: 0.340/0.230, lawful undercount |
| P2 fenced on MMReD-HF train | `p2_fenced_hf/20260827_193851_fenced/` (137799) | HF test 0.900/0.660/0.300 (floors 0.62/0.38/0.16) |
| N4 park→HF cross-domain | `n4_park_on_hf/` (137898) | **HF seq8 test 1.000** zero HF training; P2 gap = data |
| L1 Hahn α on P0 adapters | `l1_hahn/{ff_le32,ff_le8}_N*/` + `_fig/` (137821/22) | GAIN: in-window α 0.71/0.73 ≈ frozen 0.72, amp ×3–6 |
| N2 read-vs-supply (p1fence arm) | `n2_hahn/{p1fence_ep10,p1fence_frozen}_N*/` (137901/02) | verdict α **+0.009** flat; read α **+0.80** |
| N3 eval-logN | `n3_logn_fenced/`, `n3_logn_plain/` (137899/900) | fenced +0.13 @4× (0.470); plain H3 NULL |
| P3 trained-with-logN | `p3_fenced_logn/` (137937) | in-window 1.000/1.000; 2× collapses to 0.420 |
| P3 zero-shot 64/128 | `p3_zs_64_128/` (138006) | 0.310/0.170 — below eval-only logN everywhere |

Adapters promoted: `checkpoints/sft_ff_le{8,32}_adapter/`, `sft_fenced_le16_adapter/`,
`sft_fenced_le16_ep10_adapter/` (canonical fenced), `sft_fenced_hf_adapter/`,
`sft_fenced_logn_adapter/` (eval-contract: `--attn-logn-sref 3398`).
