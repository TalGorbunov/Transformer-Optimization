# outputs/treefold/ — INDEX (hand-maintained; experiment → canonical run → headline)

Campaign brief: [CAMPAIGN_BRIEF.md](CAMPAIGN_BRIEF.md) · log: [STATE.md](STATE.md)
Model everywhere: Qwen2.5-VL-7B-Instruct frozen nf4 (one model for Ask/leaf/merge/answer).
Primary Ask mode: **few-shot** (Tal, 2026-08-24 checkpoint — armC-matched convention);
zero-shot = T5 control. All samples/scoring shared with recagg armC/armB (byte-identical
stride, norm/EM).

| experiment | canonical run | headline |
|---|---|---|
| P0 Ask (768 q × 2 modes) | `p0_ask/20260824_203137_{zs,fs}/` | parse/combinable: zs 0.990/0.990, fs 0.997/0.997 (band ≥0.95 MET); template-echo 0 after words-not-templates fix; zs schemas often not restriction-shaped (no accumulators) → Tal set fs primary |
| T1 oracle-leaf fan-2 tree (H1) | `t1_oracle/20260824_fs/` | **H1 NOT MET**: ALL 0.23/0.20/0.18/0.21 (N=16→128; band ≥0.85; majority 0.36–0.39; armC v3 same samples 0.89–0.92). Post-close diagnosis: **conditional merge fidelity 0.995** (fold is sound); failure = MAP predicate eval 0.56–0.91/frame on GT text + ANSWER read-off (5/8 correct roots still answered wrong); 14B control (leaf+answer-targeted) proposed, needs OK |
| T2 fan-in law k∈{2,4,8,16,N} (H2) | `t2_fan/20260824_fs_k{4,8,16,N}/` (+T1 as k=2) | **H2 NOT SUPPORTED**: EM flat in k (0.16–0.21 @N=128, all ≈ executor-noise floor); k=N has ~0 parse deaths yet same EM — depth mortality vs breadth squashing net out; mechanism claim reverts to in-forward measurements (armor) |
| T3 captions (targeted VLM leaves) | `t3_leaves/20260824_fs/` | 23,008 frames @512, 1:35; JSON parse 0.990; **count-like per-frame leaf acc 0.921 < 0.98 band** (≈ Arm B full-state 0.913/char) — question-conditioning does not buy note-level perception; note-emission cost vs 0.995+ verdict-bit record |
| T3 end-to-end tree (H3) | `t3_tree/20260824_fs/` | **H3 NOT MET**: ALL 0.31/0.23/0.18/0.16, declining in N, < majority from N=32; merge-parse deaths grow with depth (1/9/22/36 per 96); single-frame lookups survive (first_at_room 0.75–1.00) |
| T4 Arm-B-caption leaves (H4, executor swap) | `t4_fullstate/20260824_fs/` | **H4 NOT MET**: ALL 0.25/0.24/0.18/0.23 vs Arm B (Python, same records) 0.79/0.81/0.72/0.72 — model-merge costs ~0.5 EM on identical inputs |
| T5 zero-shot control (H5) | `t5_zeroshot/20260824_zs/` | **H5 NOT MET @128**: zs 0.208/0.073 vs fs 0.229/0.208 (N=16/128); zs merge-parse 70/192 @128 — program quality depends on demonstrations |
| Figures F-A..F-D | `figs_20260825/` | F-A fan-in flat-below-majority; F-C per-level p_merge decay ~N-independent at fixed level (corruption tracks composition DEPTH, not length) |
| Conditional-merge diagnosis (post-close) | (analysis on `t1_oracle/` artifacts, STATE 2026-08-25) | **conditional merge fidelity 0.995** (2005/2016) — fold sound; leaks = leaf predicate + answer read-off |
| Leaf-prompt ablation, 7B (2×2 with 14B twin) | `leaf_ablate/20260825{,_14b}/` | **binding failure = margin×clutter**: 7B canon 0.500 / 7B minimal 0.984 / 14B 1.000 both (FP\|room-occupied 0.86→0.02) |
| ctrl-C: answer-swap alone (7B+7B+14B) | `ctrl14b/20260825_ctrlC_ans14b/` | no change vs T1 (±0.02) — roots already leaf-corrupted, as predicted |
| ctrl-A/B: 14B leaves (+14B answers) | `ctrl14b/20260825_ctrlA_tree/`, `…_ctrlB_ans14b/` | ctrl-B steps_in_room **1.00/1.00/0.88/0.50** vs pre-registered merge-noise ceiling 0.93/0.86/0.73/0.53 — composition law PREDICTIVE; ALL still ~majority (UNSEEN ask quality) |
| T1-minimal: 7B-only, clutter-stripped prompts | `t1_minimal/20260825_fs/` (v1 = Nobody-clause read-off bug, preserved) | **7B-only tree works at short N**: steps_in_room 1.00/0.75/0.38/0.50 (was 0/0/0/0), rooms_visited 1.00/0.75/0.75/1.00; ALL 0.36/0.29/0.28/0.22 |
| **armC-7B: the campaign 7B as compiler + exact executor** | `armc7b_compile/20260826/` | **LENGTH-FLAT ALL 0.64/0.60/0.61/0.63**; same 7B 3× better as one-call compiler than as 2N-call executor at 1/180th the gens; 7B→14B gap (0.62 vs 0.89) = constant compile quality, not length-coupled |
