# outputs/diag/ — INDEX (experiment → canonical run → headline)

Campaign: DIAG, created 2026-09-22 on `theory-2` (brief `CAMPAIGN_BRIEF.md`, full log `STATE.md`,
plan `docs/DIAGNOSTICS_2026-09-22.md`). Diagnostic-only, frozen model first; Tier 1 after the
rewrite's B3 adapter. Run dirs: `hahn/`, `photo/`, `eval/`, `slot/`, `fits/`, `fig/`. Every row
carries its data regime (official test / official headfit / legacy park), arm and model.
Nothing below is cited until the row has a run dir and a band verdict in `STATE.md`.

## Tier 0 — frozen model

| experiment | canonical run | headline | status |
|---|---|---|---|
| D4 fence × question 2×2 at the `<\|vision_end\|>` slot (24 qtypes @ N 8/16; 9 DC + 5 NIAH @ N 32–128) | — | — | not run |
| D3 τ / log-N dissociation — photograph (char_at_frame vs steps_in_room; N 32/128; τ ∈ {1,1.5,2,3,4} + log-N) | — | — | not run |
| D3 τ / log-N dissociation — accuracy + margin (same grid) | — | — | not run |
| D3 fenced ceiling @ N 32/128 (needle + count) + where_spend control | — | — | not run |
| D1 flip ladder — plain / qfirst / fenced, N ≤ 64 (3 qtypes) | — | — | not run |
| D1 flip ladder — N=128 (h200) | — | — | not run |
| D2 attention photograph + share-law (s, C) fit + headscan (3 qtypes × 3 arms × 5 N) | — | — | not run |
| D5 mechanistic-form fit (G, s, C), frozen + legacy re-fit (CPU) | — | — | not run (fitter synthetic-recovery test first) |
| D6 flip position strata (re-analysis of D1) + frame permutation | — | — | not run |

## Tier 1 — method model (needs B3)

| experiment | canonical run | headline | status |
|---|---|---|---|
| D1b / D2b flip ladder + photograph with the fenced-SFT adapter and the oracle gate | — | — | not run (waits for B3) |
| D3b τ / log-N on the trained reader | — | — | not run (waits for B3) |
| D8 headscan, trained vs frozen, per class | — | — | not run (waits for B3) |

## Optional

| experiment | canonical run | headline | status |
|---|---|---|---|
| D7 unquantized bf16 control @ N 8/32 (nf4 floor) | — | — | not run (optional) |

## Figures (`fig/`; regenerated from CSVs by `experiments/figs/diag_fig.py`)

| figure | claim-title | source cells | status |
|---|---|---|---|
| F1 | one frame's influence: the read dilutes, the fact does not, the gate flattens | D1 (+D1b) | not drawn |
| F2 | attention obeys the share law; the gate removes N from it | D2 (+D2b) | not drawn |
| **F3** | one temperature rescues a needle and breaks a count | D3 | not drawn |
| F4 | single-frame influence vs N, three lines (+ flip-position inset) | D1, D6 | not drawn |
| F5 | where the mass goes when you sharpen | D3 (legacy CSV drawable today) | not drawn |
| F6 | fine-tuning moves the gain, the gate moves C | D5 | not drawn |
| F7 | the model ranks the evidence and still cannot count it | D2 / D8 | not drawn |
| F8 | per-frame readout is flat, the answer collapses — across the benchmark | D4 (legacy CSVs drawable today) | not drawn |
| F9 | after the knockout the decay is in k | D1b k-chain | not drawn (later) |
| F10 | the decision margin crosses zero where accuracy dies | D1 | not drawn |
