# outputs/diag/ — INDEX (experiment → canonical run → headline)

Campaign: DIAG, created 2026-09-22 on `theory-2` (brief `CAMPAIGN_BRIEF.md`, full log `STATE.md`,
plan `docs/DIAGNOSTICS_2026-09-22.md`). Diagnostic-only, frozen model first; Tier 1 after the
rewrite's B3 adapter. Run dirs: `hahn/`, `photo/`, `eval/`, `slot/`, `fits/`, `fig/`. Every row
carries its data regime (official test / official headfit / legacy park), arm and model.
Nothing below is cited until the row has a run dir and a band verdict in `STATE.md`.

## Tier 0 — frozen model (official MMReD test rows, 50 per qtype per N; all runs 2026-09-22)

| experiment | canonical run | headline | status |
|---|---|---|---|
| Port gates B1 / C1 | `outputs/port/` (156677, 156675, 156676) | B1: 400/400 frames pixel-identical; C1 port 637/1200 = 0.531 vs 0.533; **faithful frozen N=8 = 0.515 [0.487, 0.543]** | PASS |
| Faithful frozen grid (paper prompt, 512 px, 24 qtypes) N=16/32/64 | `eval/grid/N{16,32,64}/*_faithful` (156709, 156710, 156742) | 0.415 / 0.348 / 0.285 (steps_in_room 0.42/0.14/0.04, char_at_frame 0.64/0.54/0.30, first_app 0.80/0.80/0.58) | DONE: 0.515 / 0.415 / 0.348 / 0.285 / **0.220** (156743) |
| D4 fence × question 2×2 at the `<\|vision_end\|>` slot, 24 qtypes, N=8+16 train, LR held out by sample | `gate/N{8,16}_train/{plain,qfirst,fenced_qlast,fenced_qfirst}/` (156689–96); fits `fits/gate_d4_train.json` | **fenced_qfirst L20 0.975 / AUC 0.998**; qfirst 0.84; fenced_qlast 0.49; plain 0.51 (chance). Per type: needles + counts ≥ 0.96, "when B first appeared" family 0.82–0.92 | H-D4 MET |
| D4 transfer: gate trained N ≤ 16 → tested N=32 (14 qtypes, 22,400 frames) | `gate/N32_test/fenced_q*/` (156736/37); `fits/gate_d4_transfer_*.json` | fenced_qfirst L20 **0.982 / 0.9987** (char_at_frame 0.997, steps_in_room 0.978); qlast 0.49 | DONE; N=64/128 captures in (156738/39, 156777/78), transfer in the next CPU pass |
| D3 τ / log-N dissociation at N=32 — photograph + exact match (qfirst; char_at_frame / steps_in_room / where_spend; τ ∈ {1,1.5,2,3,4} on L ≥ 12; log-N sref 3000) | `photo/d3/qfirst/N32*/`, `eval/d3/qfirst/N32*/` (156700) | needle edge ×2.67 → ×3.53 (τ 1→3) but sink 0.42 → 0.71 and needle EM 0.60 → 0.34 → 0.22 (τ4); count edge flat ~1.1, EM 0.22 → 0.02; log-N: EM 0.54, edge unchanged | **H-D3 accuracy clause REFUTED** at N=32 and N=128 (156776: needle EM 0.24→0.10 over τ, sink 0.39→0.73, count 0.00 throughout, log-N null); dissociation in the weights, not at the answer |
| D1 flip ladder, 3 qtypes × {plain, qfirst, fenced, gated}, N=8…128 | `hahn/multi/N{8,16,32,64}/`, `N128_unfenced/`, `N128_fenced/` (156697/98/99, 156751, 156781, 156782); `fits/alpha.json` | needle (char_at_frame) L20 answer row: plain 19.6/14.1/10.8/7.6/5.3 (α 0.43 [0.20,0.61] over 8–32; ≈0.47 to 128); qfirst 14.3/10.7/5.0/2.5/2.0 (0.77 [0.48,0.96]); fenced frozen at floor; **gated 27.9/28.0/27.0/28.8/26.6 (α 0.02 [−0.03,0.09])**; slot flat (CI ∋ 0); count pooled plain 1.01, at fixed gold 0.75; replay floors 0, perm 0.7 | H-D1 slot clause MET; plain α below the [0.5,1.0] band for the needle (sink share 0.60) |
| D2 photograph + share law, 4 qtypes × {plain, qfirst, fenced, gated}, N=8…128 | `photo/multi/{arm}/N*/` (156701/02/03, 156741, 156801, 156783, 156784); `fits/sharelaw.json` | per-frame mass ≈ 1/N with a constant edge: needle ×2 (plain) / ×2.4–3.1 (qfirst), count ×1.1–1.4, fenced ≈1.0–1.2; sink fraction constant in N (0.60 plain / 0.42 qfirst); gated needle mass flat 0.477 → 0.473; 2-parameter (s, C) fit not identifiable on this prompt → frame-only law + sink fraction (frame_law) | DONE; headscan: no frozen head ≥ 0.98 (best 0.86 → 0.78) |
| D5 mechanistic-form fit (G, s, C) | `fits/mechform.json` | pending the frame-law pass | CPU |
| D6 flip position strata (re-analysis of D1) | `fits/alpha.json` by_position | needle plain first/mid/last 0.57/0.22/0.34 (no monotone distance effect, ~17 pairs each); count plain 0.95/1.22/0.08 (recency on the last third; n small) | descriptive |

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
| F5 | where the mass goes when you sharpen | D3 (legacy CSV) | drawn `fig/F5_mass_when_sharpen.png` |
| F6 | fine-tuning moves the gain, the gate moves C | D5 | not drawn |
| F7 | the model ranks the evidence and still cannot count it | D2 / D8 | not drawn |
| F8 | per-frame readout is flat, the answer collapses — across the benchmark | legacy composed-linear readout + legacy 392-px grid | drawn `fig/F8_perqtype_heatmap.png` (right panel to be redrawn from `eval/grid`) |
| F9 | after the knockout the decay is in k | D1b k-chain | not drawn (later) |
| F10 | the decision margin crosses zero where accuracy dies | D1 | not drawn |
