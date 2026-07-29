# Stage-4 STATE (overwrite at every transition)

Updated: 2026-07-21 ~15:10 (HEADLINE CELL LANDED: le16@N=128 = 0.087, band refuted; fixed-criterion arms training)

## Landed (all in draft + INDEX; numbers traceable)

| item | result |
|---|---|
| E-D no-harm (124508) | MME −0.2 / POPE −1.4 pts — **GO** |
| E-C strong form (124492) | d′ 2.40 (27% of teacher 8.89), tally 0.508 — NO-GO |
| **E-C(b)** Q-first+at-end (124514) | d′ 10.27, **tally 0.999 = interleaved — layout-freedom GO (qualified: question must lead)** |
| E-E seeds (124509/10 + 124482) | TF-count **1.000 × 3 seeds**; tf-exact 0.963±0.007 |
| Rooms decode-gap diag (124527) | all errors = missing-room verdicts; counting exact; detection recall is the residual |
| E-B SFT control (124484) | train-length 0.9984; **long-N BLOCKED — script never saves adapter** (BLOCKED.md) |
| E-A le16@N=32 zero-shot (124522) | **0.280** (in-range 0.394, pf 0.007) — digit 0.092 « A3 0.215 « le16 0.280 « A4-in-length 0.447 |
| E-A le64@N=32 held-out (124571) | **0.733** (pf 0.000, MAE 0.43, g32 9/9) — tally+in-length data is the working recipe |

## Running / queued

| job | what | note |
|---|---|---|
| 124596 | **le16@N=128 headline cell** (h200, LIMIT 34, dec 280) | RUNNING, ~5-7h |
| 124597 | le64@N=128 (2× extrap from ≤64) | queued behind it |
| 124586 | le64@N=64 held-out (52, dec280) | running ~7h |
| 124578 | le16@N=64 zero-shot (60, dec200) | running (degenerate-repetition mode visible) |

Ops lessons encoded: N=128 ≈ 14-20 min/sample → LIMIT 34/dec 280/11h wall; exams launch only
after trainers COMPLETE; report writes only at end (never let walltime kill an eval).
A3/A4 N=128 rows dropped for budget (story already told at N=32).

## New cells landed (draft + table rows 8/9/20-24 updated)

| cell | number |
|---|---|
| le16@N=64 zero-shot (124578) | 0.150 (pf 0.133) — decay 1.000→0.280(2×)→0.150(4×) |
| le64@N=64 held-out (124586) | **0.558** (in-range 0.821; g48/64 cap-truncated by design) |

## BUDGET OVERRIDE (Tal, permanent): GPU-hours unconstrained; etiquette rules unchanged.

## Active jobs

| job | what | ETA |
|---|---|---|
| 124755/56/57/58 | E-F v2 exams N=32/48/64 held-out + N=128 (h200) | ~8-12h |
| 124701 | E-G train (matched n=8772 ✓) ep0 done | ~05:30 |
| (124697 done: 0.287 — cell 0.284±0.004 over 2 seeds) | | |
| 124727 | L12-ckpt N=32 exam | ~09:00 |
| 124736 | le16@N=128 HEADLINE (resubmitted after external cancel of 124606/07) | h200 queue |
| 124729/30 | main session's carrier_layer jobs — MONITOR ONLY, not mine | — |

## Landed since last update

- **E-B long-N: SFT control N=16 0.480 / N=32 0.350** — no full collapse; bimodal
  extreme-count heuristic, DEAD MID-RANGE (the theory's supply ceiling located); adapter
  saved; N=64 cell OOM'd (log entry has the full reading + caveats).
- **L12 full-data arm: TF 1.000 @ep2, tf-exact 0.991** (fastest fidelity of any arm) →
  `carrier_tally_le16_L12/20260720_192738_L12_r8/`; N=32 exam = 124727.

Parse-sensitivity preview logged (first-vs-last-match ≤1 sample per cell). E-F data in
`data/mmred_longN_park2/` (312 N32 + 210 N48, seed 7).

## Next

1. Collect 124698 → its N=32 exam; 124697 → seed error bar; 124682 → held-out N=32/64 +
   N=128 exams (LIMIT34/dec280/11h); 124696 → E-B verdict (collapse band N≥32).
2. N=128 pair on h200 whenever the node frees; then final table + Phase-4 summary.
3. Further follow-ups (pre-register first): more long-N iterations toward N=128; rooms
   detection recall (supply-side); more seeds.

## MAIN-SESSION UPDATE (2026-07-20 ~22:00): N=128 unblocked from the h200

- carrier_layer_lora.py eval path now prefers EFFICIENT attention (EFF_SDPA) — N=128 evals fit
  a 40GB A100 (verified: 2-sample smoke, job 124720, no OOM, ~20 min/sample). The "N=128 =
  h200-only" ops lesson is RETIRED for evals.
- h200 jobs 124606/124607 CANCELLED (queue was hostage to a 5-day 8-GPU h200-dds job);
  resubmitted on a100: le16@N=128 and le64@N=128, LIMIT=34 dec=280, ~11h each. Agent: collect
  THESE instead; do not resubmit to h200; future N=128 exams (E-F/E-G) go to a100 with the
  same flags.


## Late additions (2026-07-21 afternoon)

- **HEADLINE: le16@N=128 0.087** (first-match 0.130; recovered from 23/34 dumps after
  TIMEOUT — the dump safety net worked). Band ≥0.80 REFUTED; ladder story logged.
- L12@N=32 zero-shot **0.443** (GO ≥0.40) → L12 = new default; l12v2 arm (124773, 5 ep,
  FIXED save criterion (acc, tf-exact)) = best shot at the composition.
- E-G 4-ep undertrained (0.923/tf 0.720) → pcouple8 rerun (124774, 8 ep, fixed criterion).
- v2 exams confounded by old save criterion (N=32 0.607, N=64 0.365) — logged as exhibit;
  v2-N48 (124756) + v2-N128 (124758) still running/queued.
- Seed cell: N=32 zero-shot 0.284±0.004 (2 seeds).
- E-B SFT long-N: N=16 0.480 / N=32 0.350 — no collapse; dead mid-range (supply ceiling).
- Successor: when 124773/124774 land → exams on THEIR eval splits (dirs-files) at
  N=32/48/64 + N=128 (LIMIT 34/dec 280/**12h wall — 11h timed out**, dumps=40); compare
  l12v2 vs v2 (data lever, clean ckpts) and pcouple8 vs v2 (E-G GO = beats at every OOD
  length, pf~0).


## E-H L* sweep launched (2026-07-21 evening, Tal-approved)

- Arms: L8=124917, L10=124918, L14=124919 (4d_1g) · L20=124920 (12h_4g) — headline ≤16
  recipe, only --l-open varies; fixed save criterion. EPOCHS=4, ~4-6h each + queue.
- When each lands: zero-shot exams N=32 (LIMIT 300, dec 240) + N=64 (LIMIT 60, dec 200);
  assemble 6-point curve with L12 (0.443) and L17 (0.280) reference cells (recipe-matched,
  criterion caveat logged); winner → 2 seeds + N=128 (h200 — a100 can't fit seq 23k math
  attention; deviation from Tal's 'a100' noted).
- Also running: l12v2 exams (124904-907), pcouple8 final epoch (124774).


## 2026-07-22 early-morning state

Landed overnight (all logged + INDEXed where final):
- **l12v2 ladder: N=32 held-out 0.953 (pf 0) · N=48 0.878 cap-adj · N=64 0.678 cap-adj** —
  campaign-best in-model long-N readout. N=128 = 124907 (h200 queue).
- E-G/pcouple8: in-dist ceiling 0.955/0.832@ep5 (8 ep); **N=32 held-out 0.527 — losing
  cell 1 of the GO** (l12v2 0.953 on identical dirs). N=64=124923, N=128=124924 pending.
- E-H arms all landed (L8/L10/L14/L20, table in draft); 8 zero-shot exams launched
  (124965-124972: per arm N=32 LIMIT300 + N=64 LIMIT60).

Running/queued: 124907/124924 (N=128 h200 queue), 124922 done, 124923 (E-G N64),
124904-906 done, 124965-72 (E-H exams), main session's 124729/30.

Next: assemble the 6-point L* curve when 124965-72 land → winner gets 2 seeds + N=128;
collect E-G N64/N128 → final E-G verdict vs matched cells; l12v2-N128 → the best-arm
N=128 cell; then FULL-THESIS TABLE + Phase-4 summary refresh.
