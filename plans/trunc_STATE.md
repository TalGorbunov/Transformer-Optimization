# TRUNC campaign STATE (overwrite at every transition)

Updated: 2026-07-25 ~05:40 (E2-indist FAIL 0.047 → E4 retrain LAUNCHED; E5 structural PASS)

## Success screen (fills in as cells land)

| item | status / number | run dir(s) |
|---|---|---|
| **E1 exactness verdict + speedup** | **CLOSED — READS-FRAMES (band 3): kvdrop identical 1/20, answer-equal 1/20 (div at first verdict token); fast≡mask 18/20; decode speedup 16.2× (mix) / 98.9× (N=64: 657.1→6.6 s/sample); keep 103/12775 (124×)** | `trunc_kvdrop/e1{a,b}/20260725_*_evalonly/` (125554/55) |
| **E2 truncation verdict (Δ vs 1.000/0.987/0.981)** | **COMPLETE — FAIL ×3: 0.047 / 0.040 / 0.019** (Δ ≈ −0.95; pf 0 everywhere — format survives, evidence content gone). Not a saturation claim (E1 mechanism); E4 discriminates | `trunc_at12/{indist150,N32,N64}/20260725_*/` (125562/63/64) |
| **E3 saturation curve L∈{12,14,16,20,24} @N=32** | L12 = 125563; L14 125565 (running); L16 125571 + L20 125572 (l40s 12h_4g); L24 queued next slot | `trunc_sweep/L*/` |
| **E4 retrain (deploy-matched)** | caption: trainer TF-count 1.000 @ep5 in 1h34 (14×/21× efficiency, split gate PASSED) but **in-dist EXAM 0.133 = FAIL** — greedy all-or-nothing (per-slot carrier verdict unreliable even TF: tf-exact 0.165); N32/N64 caption exams running (125605/06); **E4b scan retry RUNNING (125609)** — presence-only verdicts | `trunc_retrain/{carrier_caption_trunc12,exam_indist150}/` |
| **E5 chunked prefill** | **STRUCTURAL PASS**: L0 delta exactly 0; carriers ≤0.21 abs everywhere; "dq=34" resolved = bf16 noise on header attention-sink dims (|h|≈5k, ~0.7% rel); tail Δ real (prereg'd). Behavioral bar re-runs on E4 ckpt | `trunc_bench/chunkverify/20260725_030842_*/` (125566, 125568/69 dbg) |
| **E6 benchmark table** | machinery ready + validated; runs post-E4 | — |
| **E7 N=256 supply + 0.88-prediction** | gated on E4 GO | — |

## The campaign story so far (one paragraph)

Masks never hid frames from tail/decode rows (P0.1); the trainer teacher-forced targets
WITH that access; so the winner's decode reads frames (E1: identity 1/20) and eval-only
truncation fails (E2: 0.047/0.040/0.019). Deploy-matched retrain (E4-caption) closes the
TEACHER-FORCED channel completely (TF-count 1.000, 14× faster, 21× smaller cache) but
GREEDY readout collapses all-or-nothing (in-dist 0.133): per-slot carrier verdicts are
unreliable even under TF (tf-exact 0.165) although the info is linearly present in
carriers (gate→tally 0.99). The wall is the LM's per-slot ADDRESSING at readout, not
carrier content. E4b (scan: presence-only verdicts) separates identity-vs-addressing.
Fallback GO-route if E4b fails: truncated carriers + EXTERNAL gate→tally readout (one
cache run + CPU probe) — "one token/frame inference with linear readout" cell.

## Live jobs

| job | what | QOS/partition |
|---|---|---|
| 125563 | E2 N=32 trunc@12 (150 dirs, DEC=320) | 24h_1g h200 |
| 125564 | E2 N=64 trunc@12 (52 dirs, DEC=620) | 24h_1g h200 |
| 125565 | E3 L14 @N=32 | 4d_1g h200 |
| 125570 | **E4 truncated retrain** | 12h_4g h200 |
| 125571/72 | E3 L16 / L20 @N=32 | 12h_4g l40s |
| 125573 | E3 L24 @N=32 | 4d_1g l40s |

Monitors: wave-1 completion + E4 log, persistent.

## SATURATION DEPTH MEASURED (2026-07-25 ~10:30) — the campaign's mechanistic headline

Non-truncated probe curve (150× N=32, z-scored logistic, 5 seeds): gate err
**0.339@L12 → 0.216@L14 → 0.173@L16 → 0.0082@L20 → 0.0051@L24**; external gate→tally
**0.909±0.016 @L24**. Truncated@12 stays ~0.35 at all layers → the gate is WRITTEN in
layers 12-19 via the retained carrier→own-frame edges (the brief's "hypothesis under
test", answered). Best L_trunc = 20, not 12. → **E4c running (125628): l_open=20 +
truncate_at=20**, caption verbatim; chunked prefill stays valid. Figure:
`hybrid_dump_N32_notrunc/…/saturation_curve.png`.

## Next actions

1. E4b scan trainer (125609) lands → tf-exact tells the story (caption plateaued 0.165);
   if promising → exams ×3; if greedy fails too → the addressing-wall verdict is final
   for in-model readout; then run the FALLBACK cell: truncated-carrier cache (N=32 dirs)
   + external gate→tally probe (CPU) = the compression GO-route.
2. Caption exams 125605/06 + E3 L16/L20/L24 land → log (curve expected flat ~0.03-0.05).
3. E6 benchmark runs REGARDLESS of GO (engineering numbers): exactness-check+FAST runs
   at N=8/32/64/128 on a100/h200 + chunked arm; assemble table + KV-bytes math.
4. E7 stays gated per PREREG (needs E2/E4 GO) — record as gated-out if E4b fails.

## Coexistence

p0p2 on a100 (untouched): 125507 24h_1g + 125498/502/504/519/567 4d_1g. My QOS use:
24h_1g 2/4 · 4d_1g 2/8 (→1 when E1b exits) · 12h_4g 3/3 (mine only) · 2h_2g 0. h200
5/8 busy, l40s n314 2/8.

## Blockers

None.
