# DIAG — STATE (append-only, newest last)

## [2026-09-22] Campaign created on branch `theory-2` from the approved plan (Claude; plan approved by Tal 2026-09-22) — code being written; nothing run
- Source: `docs/DIAGNOSTICS_2026-09-22.md` (§2 cells D1–D8 with predictions / arms / qtype cells /
  costs; §5 verification). Brief = `CAMPAIGN_BRIEF.md` (bands H-D1…H-D6 fixed there, §3); index =
  `INDEX.md` (every cell family "not run").
- Code state on this branch: `experiments/probe_hahn.py` (port; arms plain/fenced/gated,
  `steps_in_room` flips only) and `experiments/evaluate.py` exist. Being added: the `qfirst` arm,
  needle flips (`char_at_frame`, `n_char_at_frame`), `--qtypes`, `--attn-sharpen` /
  `--sharpen-from-layer`, `--attn-logn-sref`. Not yet written: `experiments/probe_attention.py`
  (photograph + (s, C) NLS fit + headscan), `experiments/diag_fit.py`, `experiments/figs/diag_fig.py`,
  `sbatch/probe_attention.sbatch`, the D4 slot-capture entrypoint. Port gates (`outputs/port/INDEX.md`):
  B1 login-node smoke only; C1 / C2 / B3 / C3 not run. No DIAG cell runs before C1 reproduces 0.533@8.
- Data facts verified today (they shape the cells; brief §1.1): official test split = exactly 50 rows
  per qtype per N (N ∈ 8/16/32/64/128). `steps_in_room` gold at N=8 = 31/50 zeros; at N=128 median
  gold 18, 60 % > 16 ⇒ flip response uses ALL golds (pairs ≤ 50/cell, typically 40–50); the digit
  margin is valid only for gold+1 ≤ 9 and is reported on that subset; per-gold strata need ≥ 5 pairs
  and are thin at N ≥ 64 (headfit rows pooled in, labelled). Headfit rows exist as
  `seq_len_{32,64,128}_headfit` (600/length = 25/qtype; label "headfit (official generator, our
  seed)"). Room names have distinct first tokens ⇒ first-token margin for `char_at_frame`; digits 0–9
  are single tokens. N=128 fenced/gated ≈ 41.8k tokens (324 image tokens/frame) ⇒ h200 (140 GB) or
  the batched-block path; `evaluate.py` refuses the dense fenced path above 24k tokens by default;
  plain-arm N=128 fits 40–48 GB.
- Data regimes to label on every number: official test / official headfit / legacy park.
- Order fixed: C1 → D4 → D3 → D1/D2 (N ≤ 64 before N=128) → D5/D6 → Tier 1 after B3.
- Decisions carried from the plan's §6: (1) DIAG code lives on `theory-2` (this campaign);
  (2) Tier 1 waits for the B3 adapter — DIAG ships frozen-only first (the "deployed model" question);
  (3) D1–D3 use the 3 qtypes + the `where_spend` control; all 24 qtypes go into D4 only.
- No job submitted. No GPU hour spent. No RESULTS.md entry.

## [2026-09-22 15:20] First submissions (branch theory-2)
- Code landed: experiments/_diag_common.py (pairs, vocab margins, QKCapture, row_attention, tau/log-N knobs),
  experiments/probe_hahn.py (generalized: qtypes steps_in_room/char_at_frame/n_char_at_frame; arms plain/qfirst/fenced/gated;
  --attn-sharpen/--attn-logn-sref), experiments/probe_attention.py (photograph + headscan + first-token margins),
  evaluate.py (+ tau/log-N knobs), sbatch/probe_attention.sbatch, probe_hahn.sbatch (rewritten), diag_smoke.sbatch,
  tests/test_diag_common.py (4/4). CPU builders running: experiments/diag_fit.py, experiments/figs/diag_fig.py.
- Submitted: 156674 diag_smoke (a100-public 2h_2g 01:30; all instruments, limit 3, seq_len_8_test -> outputs/_scratch/diag_smoke);
  156675 C1 port check (evaluate PORT_CHECK=arm-a seq_len_8_test, a100-public 12h_4g 03:00; target 640/1200 = 0.533);
  156676 C1 faithful (evaluate --frozen LAYOUT=paper seq_len_8_test, same QOS; the NEW frozen baseline row);
  156677 B1 verify (prepare_data VERIFY_N=50 seq_len_8_val, 4h_0g CPU).

## [2026-09-22 15:50] CPU pieces landed; D4 instrument added
- experiments/diag_fit.py (alpha/margins/sharelaw/headscan/gamma/mechform/summary; tests/test_diag_fit.py 7/7) reproduces the
  legacy anchors from the real run dirs: ARMOR-A plain L20 alpha 0.719 [0.63,0.77] (logged 0.72 [0.63,0.78]); S10 P1b L20
  s 0.266, C 7.8, R2 0.970, alpha_pred(k=4) 0.727 (logged 0.30/8/0.969; aggregation = head-mean then sample-mean);
  S11 gamma L20 1.194 [1.17,1.22] with C=5 (= record; the default C grid picks C=4, gamma 1.10 - quote gamma with its C).
- experiments/figs/diag_fig.py: F8 (per-qtype heatmap, legacy composed-linear readout vs legacy frozen grid at 392 px)
  and F5 (mass under sharpening, legacy S10b CSV) drawn to outputs/diag/fig/; F1-F4/F7/F10 wait for fits.
- D4 instrument: experiments/gate_capture.py (slot states under plain / qfirst / fenced_qlast / fenced_qfirst) +
  experiments/gate_fit.py (LR held out by sample; per qtype x N; cross-N transfer); sbatch/gate_capture.sbatch; synthetic check OK.
- Chain wrappers: sbatch/diag_d3.sbatch (photo + eval over CONDS at one N), sbatch/diag_d2.sbatch (photo over ARMS at one N).
- probe_hahn render canary now compares decoded pixels (PNG bytes differ only in matplotlib's Software chunk).
- Smoke 156674: probe_hahn leg PASSED (3 qtypes x 4 arms + tau; replay floors exactly 0; margins written); photo/eval legs running.
  Gate-capture smoke 156679 running.

## [2026-09-22 15:35] Wave 1 submitted (sbatch/diag_submit_tier0.sh wave1; ids in outputs/diag/jobs.tsv)
- D4 slot captures: seq_len_{8,16}_train x {plain, qfirst, fenced_qlast, fenced_qfirst}, all 24 qtypes x 50 rows (a100-public 4d_1g).
- D1 flip ladder: N=8/16/32, qtypes steps_in_room char_at_frame n_char_at_frame, arms plain qfirst fenced gated, 50 pairs, 12 controls (a100 24h_1g).
- D3 tau/log-N chain at N=32: conds base tau1.5 tau2 tau3 tau4 logn3000; photo (qfirst) + exact-match eval (frozen, question-first) over
  char_at_frame steps_in_room where_spend, every official row (a100 12h_4g). logn3000 ~ the seq_len_8 prompt length in tokens.
- D2 photograph ladder: N=8/16/32, arms plain qfirst fenced gated, qtypes char_at_frame steps_in_room n_char_at_frame where_spend (l40s 12h_4g).
- Faithful frozen grid (paper prompt, 512 px, all 24 qtypes): N=16, 32 (l40s 24h_1g) -> outputs/diag/eval/grid/N*; N=8 = C1 faithful (156676).
- Smoke 156674 still running its photo/eval legs; C1 port check 156675 at 675/1200 acc 0.513 (target 0.533), C1 faithful 156676 at 400/1200 acc 0.477.

- [15:42] 156704 (grid N16, l40s n314) FAILED at start: "No CUDA GPUs are available" (node GRES hiccup; 156705 on the same node runs). Resubmitted as 156709 on 2h_2g. D2 chains 156701-703 moved from 12h_4g (cap 3, held by the C1 evals) to 24h_4g. B1 verify 156677 PASSED: JSON bytes EQUAL, 400/400 frames pixel-identical, gold parity 1200/1200.
- [15:47] 156705 (grid N32) FAILED the same way on n314 -> n314 is not exposing GPUs to my jobs today; D2 chains 156701-703 moved to a100-public; grid N32 resubmitted as 156710 (a100 24h_1g).
- [15:55] C1 PORT CHECK PASSED: 637/1200 = 0.531 [0.502, 0.560] vs anchor 640/1200 = 0.533; per-qtype identical to the legacy grid (char_at_frame 41/50, steps_in_room 28/50, final_app 37/50, where_spend 18/50, crowd_count 14/50). Run outputs/port/evaluate/seq_len_8_test/20260922_151901_156675/. C1 faithful (156676) at 750/1200 acc 0.495.
- [16:00] C1 FAITHFUL (the new frozen baseline, paper prompt, images-then-question, 512 px, upstream parser): 618/1200 = 0.515 [0.487, 0.543], parse_fail 0; vs 0.531 under the 392-px preset (within CI; 50-row cells move <=0.12). Run outputs/port/evaluate/seq_len_8_test/20260922_151900_156676/20260922_151917_faithful. Grid N=16..128 faithful rows are wave-1/2 jobs (156709, 156710, +wave2).

## [2026-09-22 16:40] First DIAG cells landed (official test rows, frozen model)
- D1 N=8 (156697, outputs/diag/hahn/multi/N8/): 149 pairs (49 steps_in_room, 50 char_at_frame, 50 n_char_at_frame); render canary
  1192/1192 pixel-identical; replay floors exactly 0 (all arms/loci); fenced perm floor 0.57-0.69 at L20 final.
  Median ||dh|| at L20 final, needle char_at_frame: plain 19.6 (ctrl edit elsewhere 9.4) | qfirst 14.3 (6.3) | fenced 4.7 (3.8) | gated 27.9 (0).
  Flip changes the predicted room: plain 0.64, qfirst 0.76, FENCED 0.08, gated 1.00; fenced slot_t response 8.9 (the block carries the edit,
  the frozen fenced read does not cash it -> the frozen model's evidence routing lives in joint cross-frame attention; the fence needs the trained read).
  steps_in_room k->k+1: plain 16.8, qfirst 8.8, fenced 3.5, gated 38.5 (slot 16.1).
- D2 N=8 + N=16 (156701/156702, outputs/diag/photo/multi/): per-frame mass at L20 halves 8->16 (needle, plain 0.083->0.040); needle:non-evidence
  per-frame ratio ~2 in plain/qfirst (1.98/1.70; 2.29/2.51), count evidence ratio 1.1-1.4, fenced 1.0-1.2; prompt+sink share plain 0.60-0.79,
  qfirst 0.41-0.46, fenced 0.18-0.28. Gated headscan AUC is trivially 1.0 (hidden blocks have zero mass) - use ungated arms only.
- Wave 2 submitted (156736-156743). CPU collection job 156735 running (gate_fit over the four N=8 captures; first pooled cell fenced_qfirst L12 0.875/0.956).
- Lesson: wave-1 D2/D3 chains ran with NOBLOCK=1 (no block_mass.csv) -> no within-evidence concentration for N<=32/D3 N32; removed for wave2/h200.
- [16:50] 156740 (D1 N64) OOM: mask-free arms were forced onto EFFICIENT/MATH (46 GB score matrix at 21k tokens). Fixed in probe_hahn/probe_attention/gate_capture: sdpa_kernel(FENCED_SDPA) only when a mask is injected; unfenced forwards use the default FLASH path (as evaluate.py). Resubmitted as 156751. Running N<=32 cells are unaffected (MATH fits at 10k tokens; slower only).
- [17:05] D4 N=8 train captures (all 24 qtypes, 50 rows each, held out BY SAMPLE, pooled LR; job 156735 partial): fenced_qfirst slot L12 0.875/AUC 0.956, L20 0.970/0.9965, L24 0.960/0.993; fenced_qlast (question-blind block) L12 0.502/0.499, L20 0.486/0.479 = chance. Selection requires conditioning, on the official benchmark. Plain/qfirst arms + per-qtype tables pending in the same job.
- [17:05] Faithful frozen grid N=16 (156709): 498/1200 = 0.415 [0.387, 0.443] (legacy 392-px grid 0.422); char_at_frame 0.64, steps_in_room 0.42, first_app 0.80, crowd_count 0.14, rooms_visited 0.04. Run outputs/diag/eval/grid/N16/20260922_153655_faithful.

## [2026-09-22 17:40] D4 (N=8 train, 24 qtypes x 50 rows, LR on the <|vision_end|> slot, held out BY SAMPLE) — H-D4 MET
Pooled acc/AUC: plain L12 .476/.479 L20 .513/.525 L24 .492/.487 | qfirst .763/.878 .836/.931 .839/.925 |
fenced_qlast .502/.499 .486/.479 .478/.471 | fenced_qfirst .875/.956 **.970/.996** .960/.993 (job 156735; outputs/diag/fits/gate_d4_train.json).
Per-qtype at L20 (fenced_qfirst / qfirst / fenced_qlast): char_at_frame 1.00/1.00/0.45(AUC) · n_char_at_frame 1.00/1.00/0.49 · room_at_frame 1.00/1.00/0.56 ·
steps_in_room 1.00/0.94/0.60 · who_spend 1.00/0.81/0.52 · crowd_count 0.99/0.77/0.96 · crowded_room 1.00/0.79/0.97 · first_at_room 1.00/0.93/0.63 ·
last_at_room 1.00/0.93/0.71 · char_on_char_first_app 0.89/0.58/0.55 · room_on_char_first_app 0.95/0.70/0.45 · n_room_on_char_first_app 0.98/0.65/0.55 (AUCs).
Reading: the slot carries the question-conditioned local fact only when the question PRECEDES the frames (qfirst 0.84 pooled; fenced_qfirst 0.97);
question-blind blocks are at chance except for question-independent visual facts (crowd present: qlast AUC 0.96-0.97; first/final frame: 1.00);
the "when B first/last appeared" family is partly global and reads 0.89-0.95 (the readout's job, as the evidence rule states).
D1 N=16 landed (156698); CPU fits over N=8+16 resubmitted (156773).
- [17:45] h200 wave submitted (all N=128 fenced cells: D1, D2, D3, D4 x2) — queued behind the 24h_1g/4d_1g caps.
- [17:50] h200 chains re-queued off the full 24h_1g: D1 N128 + D3 N128 -> 24h_4g, D2 N128 -> 12h_4g (time 12:00).
- [17:55] N=128 split: mask-free arms to a100-public (D1 plain+qfirst 12h_4g; D2 plain+qfirst 12h_4g; D3 qfirst chain moved to a100 24h_4g), fenced+gated arms stay on h200 (D1 24h_4g, D2 72h_8g, D4 x2 4d_1g). 156774/156775 cancelled and replaced.
- [18:20] Faithful grid N=32 (156710): 418/1200 = 0.348 [0.322, 0.374] (legacy 392-px 0.361); char_at_frame 0.54, n_char_at_frame 0.62, steps_in_room 0.14, where_spend 0.16, crowd_count 0.02, first_app 0.80. D4 pooled over N=8+16 train (7144 held-out frames, 156773): fenced_qfirst L12 0.892/0.965, L20 0.9745/0.9978, L24 0.969/0.996.

## [17:35] D3 N=32 complete (156700; frozen, question-first, tau on decoder modules >= 12, log-N sref=3000; 50 official rows per qtype)
L20, head-mean then sample-mean. evid / competitor / sink mass; edge = evid-per-frame : nonevid-per-frame; EM = exact match (greedy, upstream parser).
| cond | needle char_at_frame: evid comp sink edge EM (margin) | count steps_in_room: evid comp sink edge EM | where_spend EM |
| base | .046 .535 .419 x2.67 0.60 (+0.30) | .122 .474 .405 x1.10 0.22 | 0.24 |
| tau1.5 | .042 .416 .542 x3.17 0.54 (+0.29) | .109 .425 .467 x1.11 0.18 | 0.16 |
| tau2 | .036 .319 .645 x3.50 0.44 (-0.20) | .089 .364 .547 x1.05 0.16 | 0.14 |
| tau3 | .030 .265 .705 x3.53 0.34 (-0.77) | .061 .258 .681 x1.05 0.02 | 0.10 |
| tau4 | .026 .281 .694 x2.86 0.22 (-1.70) | .051 .227 .722 x1.10 0.12 | 0.04 |
| logn3000 | .044 .513 .443 x2.68 0.54 (+0.02) | .121 .465 .414 x1.14 0.18 | 0.26 |
Verdict: H-D3 accuracy clause REFUTED for a global eval-time temperature on the frozen read: the needle's RELATIVE edge rises with tau
(x2.7 -> x3.5, the k=1 half of the two-constraint argument) and the count's stays flat, but the freed competitor mass flows to the
prompt/sink, the needle's absolute mass falls monotonically, and exact match falls for BOTH classes; log-N is a null at this N.
The dissociation exists in the attention weights, not at the answer. (Tier 1 repeats on the trained read; N=128 chain queued.)
Runs: outputs/diag/photo/d3/qfirst/N32*/ and outputs/diag/eval/d3/qfirst/N32*/.

## [17:50] D4 cross-N transfer (156797): LR gate fitted on N<=16 train slot states (28,800 frames), tested on N=32 test (22,400 frames, 14 qtypes)
fenced_qfirst: L12 0.895/AUC 0.963 | **L20 0.982 / 0.9987, recall 0.988, specificity 0.975** | L24 0.979/0.998.
Per type at L20: char_at_frame 0.997 (AUC 1.000) · first_at_room 0.993 · last_at_room 0.994 · steps_in_room 0.978 (0.996) · who_spend 0.983 ·
crowd_count 0.964 · crowded_room 0.952 · char_on_char_final_app 0.963 · char_on_char_first_app 0.927; all-frames-evidence types trivially 1.000.
fenced_qlast (question-blind): 0.47-0.49 at every layer = chance. Per-frame error ~1.8% pooled at N=32 -> compounds at exact match
((0.982)^32 ~ 0.56 all-correct); needle/first-last types are >=0.99/frame. N=64 captures (156738/39) landed; transfer to 64 in the next CPU pass.

## [18:05] D1 N=32 landed (156699, 2h44) -> first three-point exponents on official rows (frozen; L20 answer row; pooled pairs; log-log OLS on medians)
| edit | plain (deployed) | qfirst | fenced (frozen) | gated |
| needle char_at_frame | 19.6/14.1/10.8 a~0.43 | 14.3/10.7/5.0 a~0.77 | 4.7/2.9/1.7 (ctrl floor 1.3 @32) | 27.9/28.0/27.0 a~0.02 |
| needle n_char_at_frame | 11.3/10.9/7.8 a~0.26 | 14.4/11.8/8.2 a~0.41 | 4.1/2.0/1.3 | 18.8/16.5/19.5 a~-0.03 |
| count steps_in_room k->k+1 | 16.8/7.8/4.1 a~1.01 | 8.8/4.6/1.8 a~1.14 | 3.5/2.0/1.2 | 38.5/12.8/5.1 (k-composition: base k ~ N/6; stratify) |
| fenced slot_t (own frame) | - | - | 8.9/9.9/9.2 a~-0.02 (needle); 16.0/14.1/15.0 a~0.05 (count) | same |
Floors @32 L20 final: replay 0 (all arms), perm 0.74, ctrl plain 2.4 / qfirst 1.4 / fenced 1.3 / gated 0.
Reading: the gated (evidence-only) needle read is N-invariant on the FROZEN model; the per-frame slot is N-invariant; the deployed needle read
decays with a sub-unit exponent (a large sink term C: prompt+sink holds 0.60 of the mass at every N -> law predicts ~0.5 over this range);
the count read decays faster in the pooled protocol (k grows with N). CIs, by-gold and by-position fits: CPU pass 156856 (fits + figures).
- [18:35] Faithful grid N=64 (156742): OVERALL acc 342/1200 = 0.285  [0.261, 0.310];  char_at_frame: 15/50 = 0.300 crowd_count: 0/50 = 0.000 first_app: 29/50 = 0.580 steps_in_room: 2/50 = 0.040 
- [19:00] diag_fit fixes: needle flips (flip_kind=needle) were excluded by an ==evid filter (alpha/gamma/margins); run_dirs now ignores unfinished runs (no report.txt/meta.json); diag_fig relative-path bug. Tests 7/7. CPU pass 156888 resubmitted (156856 results superseded for needle cells).

## [19:25] Corrected fits (156888): alpha ladder with CIs; share-law identifiability; long-N needle edge
ALPHA (L20 answer row, N=8/16/32, pooled pairs, bootstrap CI): needle char_at_frame plain 0.43 [0.20,0.61] · qfirst 0.77 [0.48,0.96] ·
fenced 0.73 [0.62,0.86] (floor-limited: ctrl 2.3, perm 0.64) · GATED 0.02 [-0.03,0.09] (27.9/28.0/27.0). n_char_at_frame plain 0.26 [0.12,0.48] ·
qfirst 0.41 [0.22,0.64] · fenced 0.81 · gated -0.03 [-0.13,0.10]. steps_in_room pooled: plain 1.01 [0.67,1.36] · qfirst 1.14 [0.94,1.31] ·
fenced 0.76 · gated 1.46 [0.94,1.83] BUT at fixed base gold 0: plain 0.75, qfirst 0.89, fenced 0.74, gated 0.03 -> the gated count's
pooled slope is k-composition (base k ~ N/6). Slot (fenced, own frame): -0.02 [-0.08,0.07] / 0.05 [0.00,0.12] / 0.05 [-0.02,0.10] -> H-D1 slot clause MET.
By flip position (terciles, ~17 pairs each): needle plain first 0.57 / mid 0.22 / last 0.34 — no monotone distance effect; steps plain
first 0.95 / mid 1.22 / last 0.08 (last-third count flips decay much less under the deployed layout: recency; n small, check at N=64).
HEADSCAN (frozen, ungated): best single head AUC 0.86 (N=8) -> 0.78 (128) fenced, 0.76 -> 0.69 plain; NO head >= 0.98 at any N — the
evidence-selector head of the record is trained-in (Tier 1), consistent with the legacy frozen best 0.848.
SHARE LAW: the 2-parameter fit on the evidence SHARE is not identifiable on this prompt (returns s=-0.34, C=0.9 for the deployed needle
whose measured per-frame edge is x2): the prompt+sink takes a constant FRACTION of the mass (0.60 deployed / 0.42 qfirst / 0.25->0.08
fenced) at every N, not a constant number of frame-equivalents; the evidence share is then (1 - sink) * k e^s/(k e^s + (N-k)). The
per-frame edge e^s (evid/frame : nonevid/frame, L20): deployed needle 1.98/1.70/1.99/0.78/0.69 at N=8..128 (COLLAPSES between 32 and 64,
where char_at_frame EM falls 0.54 -> 0.30); qfirst needle 2.29/2.51/2.68/1.37/1.52; count 1.1-1.4 (deployed) / 1.1 (qfirst); fenced
1.15/1.17/1.24/1.08/1.03. -> fitter to be extended with the frame-normalized law + sink fraction per N.
- [19:40] Frame-only decomposition added to diag_fit.sharelaw (frame_law: edge, s_frame, sink per N). Deployed needle (L20, head-mean, mean of per-sample
  ratios): edge 2.20/1.84/2.22/2.13/1.63 at N=8/16/32/64/128, sink 0.63/0.60/0.60/0.60/0.57 -> s_frame 0.69, alpha_pred(frame share, C=0) 0.89;
  qfirst edge 2.41/2.64/2.85/2.61/3.13, sink 0.43/0.42/0.42/0.41/0.39, s_frame 1.0. The per-(sample,head) MEDIAN edge at N>=64 is <1 (0.78/0.69):
  a minority of heads carries the needle edge. Measured answer-row alpha (0.43 deployed) is SHALLOWER than the needle share's own decay (~0.9):
  the response is not proportional to the share alone (ctrl edit elsewhere already moves the state 5.8). Next CPU pass will carry frame_law.
- [20:00] D1 N=128 fenced+gated (156782, h200, 2h55; canary 19200/19200; replay 0; perm floor 1.37): GATED needle 26.6 (N=8 27.9, 16 28.0, 32 27.0 -> flat over 16x, frozen); gated n_char 16.7; gated count 2.1 with base k median ~18 (the k-wall, not N); frozen fenced at floor (1.49/1.57/1.55); slot_t 9.4 / 7.4 / 13.9 (flat vs N=8: 8.9 / 7.7 / 16.0). Run outputs/diag/hahn/multi/N128_fenced/.
- [20:15] D1 N=64 (156751, 3h26; canary 9600/9600): L20 answer row medians — needle char_at_frame plain 7.56 / qfirst 2.48 / fenced 1.26 / gated 28.8; n_char plain 4.90 / qfirst 3.20 / gated 17.1; count plain 1.92 / qfirst 0.89 / gated 2.95 (pooled k ~ N/6). Needle deployed ladder 19.6/14.1/10.8/7.6 (8..64).
- [20:35] D1 N=128 plain+qfirst (156781, a100, 3h32; canary 19200/19200; replay 0): needle plain 5.35 (ladder 19.6/14.1/10.8/7.6/5.3, slope ~0.47 over 8..128) / qfirst 1.96 (14.3/10.7/5.0/2.5/2.0, ~0.75); n_char plain 3.62 / qfirst 1.71; count plain 1.30 / qfirst 0.79; ctrl floors 0.88 / 0.69. Gated needle across the same range: 27.9/28.0/27.0/28.8/26.6.
- [21:55] Faithful grid N=128 (156743, 5h15): OVERALL acc 264/1200 = 0.220  [0.197, 0.244];  char_at_frame: 12/50 = 0.240 crowd_count: 0/50 = 0.000 first_app: 25/50 = 0.500 steps_in_room: 0/50 = 0.000  -> the faithful frozen ladder is 0.515/0.415/0.348/0.285/0.220 at N=8..128 (legacy 392-px: 0.533/0.422/0.361/0.304/0.247).
- [22:05] Figure script fixed (agent) and F1/F8 patched: F1 = four bars per qtype (deployed / question-first / fenced fact / gated), count bars from the fixed-count 0->1 stratum (deployed 0.75, qfirst 0.89, fenced 0.03, gated 0.03; needle deployed 0.46, qfirst 0.79, fact -0.02, gated 0.01); F8 right panel now the faithful 512-px grid (steps_in_room 0.56/0.42/0.14/0.04/0.00; char_at_frame 0.86/0.64/0.54/0.30/0.24); F2 frame-law; F3 reads sharelaw_d3; F7 headscan groups; F10 margins. diag_fit margins accuracy mapped index->label; collector includes the port N=8 faithful run. Final figure regeneration after D3 N128 + the last CPU pass.
- [22:20] D4 transfer, gate fitted at N<=16 (156963): fenced_qfirst L20 0.9821 @N32 (22,400 frames) / 0.9800 @64 (44,800) / 0.9725 @128 (89,600); per type @128: char_at_frame 0.986, steps_in_room 0.976, first_at_room 0.986, crowd_count 0.970, who_spend 0.971; fenced_qlast 0.49-0.50 at every N (chance). Frame-law edges (L20, head-mean): deployed needle 2.2/1.8/2.2/2.1/1.6, deployed count 1.65/1.66/1.24/1.29/1.09, qfirst needle 2.4/2.6/2.9/2.6/3.1, fenced ~1.0-1.2; sink deployed 0.63->0.57, qfirst 0.43->0.39, fenced 0.25->0.08-0.14. All 8 figures regenerated.
- [22:30] Figures regenerated from the complete Tier-0 data (F1/F2/F3/F4/F5/F7/F8/F10 in outputs/diag/fig/). F10 (first-token margins, D1 logits): needle deployed +4.85 -> +2.0 -> +0.7 -> -1.1 -> -1.5 (crosses 0 near N~40, where char_at_frame EM falls 0.54->0.30); qfirst +3.2 -> -1.05; GATED +5.5/+5.9/+4.6/+5.1/+4.5 (flat); count: deployed +0.4 -> -2.1, gated +0.2 -> -4.5 (the frozen gated count read is uncalibrated: answers the number of visible frames, = the legacy S1-control effect). F3 N=128 row awaits the D3 chain (156776: 4/6 photo, 3/6 evals).

## [2026-09-23 00:20] D3 N=128 landed (156776, 7h36) — Tier 0 COMPLETE (50 jobs; ledger outputs/diag/jobs.tsv)
N=128, frozen, question-first, L20 (evid | competitor | sink | edge | EM): needle base .0145 | .595 | .391 | x3.09 | 0.24; tau1.5 .0115 | .492 | .497 | x2.98 | 0.26;
tau2 .0096 | .388 | .603 | x3.16 | 0.24; tau3 .0079 | .282 | .710 | x3.56 | 0.14; tau4 .0050 | .261 | .734 | x2.44 | 0.10; logn3000 .0147 | .547 | .438 | x3.42 | 0.22.
count steps_in_room: edge x1.00-1.05, EM 0.00 under every condition; where_spend EM 0.18 / 0.20 / 0.28 / 0.04 / 0.00 / 0.30.
Same verdict as N=32: sharpening raises the needle's relative edge, the sink absorbs the freed mass, accuracy falls; log-N null. Final CPU pass 157208 (fits + all figures).
- [2026-09-23 02:05] Final CPU pass 157208 done: all 8 figures regenerated from the complete Tier-0 data; F3 footer wording fixed and redrawn. Tier 0 closed: 58.1 GPU-hours over 37 completed GPU jobs (+0.1 h on 5 failed/replaced). All 8 CPU test files green. RESULTS.md untouched (awaiting "log this").

## [2026-09-23 10:36] Tier 1 started (Tal: "run tier 1 ... do the comparison table"; + short-length rows N=1/2/4 for the "underpolation" question)
- train.py gained --layout paper, --no-fence (plain LoRA arm) and --attn-logn-sref (log-N training prior); evaluate.py inherits the adapter log-N contract; sbatch/train.sbatch NOFENCE/LOGN knobs; new sbatch/diag_tier1_eval.sbatch (N-chain per arm) + sbatch/diag_submit_tier1.sh (train | eval | probes).
- Smokes: 157946 fenced B3 (limit 40, 1 ep), 157967 plain paper-layout; renders of seq_len_{1,2,4}_test 157958-60. l40s-shared/public in admin_maint (draining) -> everything moved to a100-public.
- Literature check on sink x temperature x NIAH running (workflow sink-temperature-lit).
- [11:05] Smokes landed: plain 157967 (80 train, 1 ep, 546 s: loss 0.365, val 0.40 vs majority 0.10, adapter saved); fenced 157946 (707 s: loss 0.375, val 0.10 = majority after 80 samples — path runs end to end, adapter saved). Both print the torch "None of the inputs have requires_grad" checkpoint warning once (vision tower); harmless.
- [11:10] Tier-1 trainers launched then RESUBMITTED: 158002/158003/158007/158008/158009 (10 h) cancelled after <10 min — 1200 train rows/epoch x 5 epochs at 6-9 s/row needs 11-15 h and `scontrol update TimeLimit` is denied for users. Resubmitted with --time=22:00:00: A plain/paper 158011, B plain/qfirst 158012, C fenced 158013, D gated-oracle 158014 (24h_1g), E fenced+logN3000 158015 (24h_4g); all on a100-public (l40s draining). diag_submit_tier1.sh train now uses 22 h.
- [11:20] Frozen eval chains (no adapter needed) submitted: 158025 frozen deployed (paper layout), 158026 frozen question-first; N = 1 2 4 8 16 32 64 128, qtypes steps_in_room/char_at_frame/where_spend, 4d_1g. Adapter chains + probes follow when the trainers save.
- [11:25] experiments/diag_table.py written (comparison table: arms x N per qtype, steps_in_room split gold<=16/>16; latest finished run per cell; CSV carries run dirs); smoke on the Tier-0 D3 dirs OK.
- [11:40] Literature check landed (17 agents, 48 candidates, 12 verified): NO paper says temperature fails on NIAH because the sink absorbs the freed mass; three report the failure without the mechanism (Gao 2605.10828 "consistently degrades" -> calibration; Velickovic 2410.01104 inference-only "does not work well" -> tokenisation; DySCO 2602.22175 uniform tau helps only x1.03-1.11). All theories are two-party (qTTT, SSMax, Gao drop the "other tokens" term). Our D3 mass accounting (sink 0.42->0.71 @32, 0.39->0.73 @128 under tau) is unclaimed. Must-read before the novelty claim: Barbero 2025 over-mixing, Gu 2024, VAR (visual sink). Doc: docs/PRIOR_ART_SINK_TEMPERATURE_2026-09-23.md.
- [13:05] Frozen deployed chain 158025 done (1h31, N=1..128, 50/qtype/N; outputs/diag/eval/tier1/frozen_plain/): char_at_frame 1.00/.98/.90/.86/.64/.54/.30/.24; steps_in_room 1.00/.86/.66/.56/.42/.14/.04/.00 (gold<=16 stratum .15@32 (48) / .05@64 (37) / .00@128 (20); gold>16 = 0 everywhere); where_spend 1.00/.98/.86/.36/.32/.16/.26/.26. The count read is already sliding at N=2 and N=4, i.e. below every training length. Frozen question-first 158026 at N=128; its N=32 row (0.60/0.22/0.24) reproduces the Tier-0 D3 base cell exactly. Table: outputs/diag/tier1_table.md (experiments/diag_table.py).
- [13:15] Frozen question-first chain 158026 done (1h38; outputs/diag/eval/tier1/frozen_qfirst/): char_at_frame 1.00/.92/.88/.88/.78/.60/.34/.24; steps_in_room .80/.56/.46/.52/.34/.22/.04/.00; where_spend 1.00/.96/.86/.34/.38/.24/.20/.18. N=32 and N=128 reproduce the Tier-0 D3 base cells exactly. Question-first costs the count read at N<=4 (.80/.56/.46 vs 1.00/.86/.66 deployed) and buys the needle at N=16-64. Both frozen rows of the comparison table are final; trainers 158011-158015 at 1h40, first epoch pending.
