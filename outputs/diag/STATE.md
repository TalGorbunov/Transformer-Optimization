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
