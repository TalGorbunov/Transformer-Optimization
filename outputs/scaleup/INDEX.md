# outputs/scaleup/ — INDEX (experiment → canonical run → headline)

Campaign: SCALEUP, created 2026-09-23 on `theory-2` (brief `CAMPAIGN_BRIEF.md`, log `STATE.md`,
plan `docs/SCALEUP_2026-09-23.md`). Run dirs: `<dataset>/<backbone>/<stage>/<cell>/<stamp>_<job>/`.
Every row carries dataset, protocol (planted / uniform / mmred), N, arm and backbone.
Nothing below is cited until the row has a run dir and a verdict in `STATE.md`.

## S0 — data + refactor proof

| experiment | canonical run | headline | status |
|---|---|---|---|
| HERBench lite_v2 prep + verify | — | — | not run |
| MINERVA prep + verify (20 hand-checked traces) | — | — | not run |
| Refactor proof: C1 faithful N=8 through the new seams | `mmred/qwen2.5-vl-7b/s0_proof/c1_faithful_N8/20260923_200832_faithful` (159293) | **618/1200 = 0.515 [0.487, 0.543]** = job 156676 exactly (`refactor_proof_diff.txt`) | PASS |
| Refactor proof: C2 p2 port check through the new seams (fenced path) | `mmred/qwen2.5-vl-7b/s0_proof/c2_p2_N8/20260923_201335_pc-p2` (159297) | **44/50 = 0.880** vs 45/50; 1 row moved (0004694), 49/50 identical | PASS (tolerance) |
| Refactor proof: D4 fenced_qfirst N=16 capture + fit through the new seams | capture `mmred/qwen2.5-vl-7b/s0_proof/d4_fenced_qfirst_N16/20260923_205942_159361` (159361); fits `…/s0_proof/d4_fit/` (164391 refit, 164392 reference) | slot states cosine 0.9998 @L20 vs the 09-22 capture, labels identical; refit **L20 acc 0.977 / AUC 0.998** vs reference 0.975 / 0.998 (anchor 0.975 / 0.998) | PASS |

## S1–S3 — Qwen2.5-VL-7B on the video datasets

| experiment | canonical run | headline | status |
|---|---|---|---|
| S1 HERBench lite_v2 comparability row: Qwen2.5-VL-7B (4-bit), uniform N=16, frames then question, official prompt + letter parser, all rows | `outputs/scaleup/herbench/qwen2.5-vl-7b/s1_eval/uniform_N16/plain/20260928_212557_faithful` (164387) | **909/1971 = 0.461 [0.440, 0.484]**, parse_fail 0; families (task-mean) R&T 0.762, GC&V 0.489, TR&C 0.432, MEA&N 0.357; letter-prior baseline 0.259; coverage 0.755. Published: 0.359 on the FULL set; lite_v2 third-party rows for other 8B models 0.46–0.48 | DONE |
| S1 HERBench frozen ladder, Qwen, `planted`, 50 rows/task (482 @N8, 532 above) | `outputs/scaleup/herbench/qwen2.5-vl-7b/s1_eval/planted_N*/{plain,qfirst}/` (164375–78) | plain **0.598 / 0.583 / 0.555 / 0.555 / 0.538**; qfirst **0.529 / 0.462 / 0.344 / 0.267 / 0.173** at N=8/16/32/64/128. qfirst @128: 58 % answer 'C', 26 % free-text captions, every task at or below chance — the question is lost | DONE (interval tasks re-running under `isolated`) |
| S2 per-block evidence classification, Qwen, `planted`, 5-fold by video, train N=8+16, test every N (131,536 frames) | captures `outputs/scaleup/herbench/qwen2.5-vl-7b/s2_gate/planted_N*/<arm>/20260928_*`; fits `outputs/scaleup/herbench/qwen2.5-vl-7b/s2_fit/` (164557, 164558; control 164565) | fenced_qfirst L12/L20/L24 AUC **0.63 / 0.74 / 0.74** (bal 0.59 / 0.66 / 0.66); qfirst 0.76 / 0.81 / 0.62±0.09. Per task, fenced L20: TSO 0.91, SVA 0.91, ASII 0.73, FAM 0.71, FOM 0.70, AC 0.70, AG* 0.53–0.58, MEGL/MPDR 0.52–0.54; flat in N for every task. **Interval tasks (AG*, MEGL, MPDR) are mis-specified under `planted`** (fillers inside the evidence interval) → re-running under `isolated` (164567–78) | PARTIAL: control + position baseline + isolated re-run pending |
| S3 oracle gate (fenced + evidence labels), Qwen, `planted` | `outputs/scaleup/herbench/qwen2.5-vl-7b/s3_oracle/planted_N*/gated/` (164385, 164390) | **0.467 / 0.491 / 0.491 / 0.487 / 0.492** at N=8…128: flat in N; below plain (0.54–0.60), far above qfirst at long N (0.17 @128) | DONE (interval tasks re-running under `isolated`) |

## S4 — backbone ports on MMReD

| experiment | canonical run | headline | status |
|---|---|---|---|
| InternVL3.5-8B (Qwen3-8B LM, non-thinking) on MMReD: frozen grid, paper layout | `mmred/internvl3.5-8b/s4_grid/N{8,16}/plain/` (164954, 164955) | **0.561 / 0.478** at N=8 / 16 (1,200 rows each; Qwen2.5-VL-7B 0.515 / 0.415) | N=8,16 DONE; N>=32 pending |
| InternVL3.5-8B on MMReD: per-block evidence probe (D4 protocol, held out by question, N=8+16 train rows) | captures `mmred/internvl3.5-8b/s4_gate/`; fits `…/s4_fit/` (164967, 164968) | fenced_qfirst **L21 acc 0.928 / AUC 0.982** (L11 0.946, L15 0.955, L26 0.975, L31 0.956 AUC); question-blind fenced_qlast = chance at every layer (AUC 0.49-0.50). Qwen anchor: L20 0.977 / 0.998, blind chance | DONE |
| Batched fenced path on InternVL3.5: parity with the dense path | `mmred/internvl3.5-8b/s0_proof/fastpath/` (164952, 164953) + `tests/test_fastpath.py` | tiny model float32: exact (2e-8 / 3e-7, same tokens); GPU 120 rows: 0.467 dense vs 0.442 batched, 117/120 same correctness (bf16 drift) | PASS |
| Gemma-3-12B-it (4-bit, bfloat16; masked dense path) on MMReD: frozen grid, paper layout | `mmred/gemma-3-12b/s4_grid/N8/plain/` (164989); N=16 = 164990 | **0.472 [0.443, 0.500]** at N=8 (Qwen 0.515, InternVL 0.561). Different regime: 40 of 48 layers see 1,024 tokens back | N=8 DONE; N=16 running |
| Gemma-3-12B on MMReD: per-block evidence probe (D4 protocol, held out by question) | captures `mmred/gemma-3-12b/s4_gate/`; fits `…/s4_fit/` (165042, 165043) | fenced_qfirst **L27 acc 0.921 / AUC 0.980** (L14 0.922, L21 0.954, L34 0.971, L41 0.968 AUC); question-blind fenced_qlast = chance at every layer (0.49-0.51). L41 has one dimension beyond float16 in the stored captures (read as saturated) | DONE |

## S1u — UNIT baseline: how well does the frozen model perceive one evidence unit by itself? (plan 2026-10-06)

Three unfenced regimes through the model's own output, on the pinned 582-row HERBench set
(`sbatch/lib/splits/qids_herbench_native582.txt`) and the MMReD anchor; run dirs
`<dataset>/<backbone>/s1u_unit/{T,E_<unit>_<res>,D_<unit>_<res>}/`. Plan: `docs/UNIT_BASELINE_2026-10-06.md`.

| experiment | canonical run | headline | status |
|---|---|---|---|
| W0 units: HERBench `units_native/` for the 582 rows (9-frame grids + hard + easy negatives), verify | `outputs/prepare/herbench/20261006_151722_170561` (170561; smoke 170555); `data/herbench_v2/verify_report_units.txt` | **532 questions, 1,516 positive units, 1,384 hard + 1,513 easy negatives, 12.35 GB, 7 min**; verify clean (0 missing, 0 hard inside interval + margin, 0 easy < 5 s); trailer shots = 3-frame grids (TSO 171/200, SVA 181/200) | DONE |
| T text-only, HERBench × Qwen (582 rows) | `herbench/qwen2.5-vl-7b/s1u_unit/T/uniform_N8/20261006_161034_faithful_T` (170568) | **0.388 [0.349, 0.428]**; AC 0.04 · AGAR 0.72 · AGBI 0.84 · AGLT 0.82 · ASII 0.68 · FAM 0.44 · FOM 0.30 · MEGL 0.16 · MPDR 0.28 · RLPC 0.06 · SVA 0.30 · TSO 0.14 | DONE |
| E evidence-only, HERBench × Qwen: frame@lo, frame@hi, clip3_d1 / clip5_d1 / clip5_d2 / clip9_d2 @hi | `…/s1u_unit/E_<unit>_<res>/20261006_*` (170569, 170570, 170571); table `…/s1u_unit/TABLE.md` | pooled 0.624 / **0.633** / 0.620 / 0.581 / 0.620 / 0.594 (frame lo / frame hi / clip3 / clip5_d1 / clip5_d2 / clip9) vs T 0.419; P* per task: AC 0.68 frame · TSO 0.74 frame · FOM 0.84 clip9 · FAM 0.82 clip3 · ASII 0.86 · AGBI 0.96; walls: SVA 0.36, MEGL 0.24, MPDR 0.28, AGLT 0.82 = T. Clips hurt AC / TSO, help FOM / FAM. Fence cost vs the planted oracle cells: TSO 0.72 vs 0.30, FAM 0.72 vs 0.46, FOM 0.64 vs 0.42, pooled 0.624 vs 0.476 | DONE |
| D per-unit yes/no, HERBench × Qwen: the same six units, pos vs hard vs easy | `…/s1u_unit/D_<unit>_<res>/20261006_*` (170572–170575), 4,277 units each | AUC vs hard, frame@hi: TSO 0.87 · SVA 0.90 · AGBI 0.79 · AC 0.67 · FAM 0.63 · FOM 0.61 · ASII 0.60 (all 0.711; vs easy 0.784); clip3_d1: AC 0.76 · FAM 0.67 · FOM 0.67 (actions up, TSO down to 0.64); recall@spec99 <= 0.34, yes-rate on evidence 0.04–0.36 → the model's own vote cannot gate at long N | DONE |
| MMReD anchor × Qwen: T, E(frame), D(frame) at seq_len 8 and 16 (50 rows / qtype) | `mmred/qwen2.5-vl-7b/s1u_unit/{T,E_frame_default,D_frame_default}/…` (170607/8, 170609/10, 170578/81); tables `TABLE_seq{8,16}.md` | T 0.189 / 0.154 → E(evidence frames) **0.640 / 0.605** (official 0.515 / 0.415); needles 1.00, crowd_count 0.00, steps_in_room 0.32; D per-frame AUC 1.00 on positional / trigger qtypes, 0.84 crowd_count: perception fine, aggregation is the wall | DONE |
| W2 MINERVA × Qwen (489 pinned rows, `sbatch/lib/splits/qids_minerva_50.txt`; regex evidence = lower bound): T, E frame@lo + frame/clip3_d1/clip9_d2 @hi, D frame@lo + frame/clip3_d1 @hi | `minerva/qwen2.5-vl-7b/s1u_unit/…` (170859–170863); table `…/s1u_unit/TABLE.md` | T 0.186 → E 0.354 / 0.362 / 0.405 / **0.421** (official N=32 row 0.284); clips help (+0.06); walls counterfactual / numerical / counting / goal; D near chance (hard AUC 0.56–0.61) | DONE |
| W3 InternVL3.5-8B (non-thinking) + Gemma-3-12B × HERBench (532 rows) + MMReD anchors (seq 8 / 16) | `herbench/{internvl3.5-8b,gemma-3-12b}/s1u_unit/…` (170864–170872, 170873–170877), `mmred/{…}/s1u_unit/…` (170869/70, 170878/79); tables `…/s1u_unit/TABLE*.md` | HERBench T → E frame@hi: InternVL 0.513 → 0.643 (wall 0.130), Gemma 0.335 → 0.496 (0.162); shared walls MEGL / MPDR, shared targets FOM / FAM / AC / TSO; InternVL detection with clip3: TSO 0.95, SVA 0.92; MMReD E 0.671 / 0.602 (IV), 0.599 / 0.575 (Gemma) vs official 0.561 / 0.478, 0.472 / 0.388; crowd_count 0.00 on both | DONE |

## S5 / S6 — cross cells, method

| experiment | canonical run | headline | status |
|---|---|---|---|
| S5 (S1–S3) × {internvl3.5-8b, gemma-3-12b} × {herbench, minerva} | — | — | not run |
| S6 gate + read (training source per plan §6) | — | — | not run |

## MINERVA (Qwen2.5-VL-7B, 4-bit)

| experiment | canonical run | headline | status |
|---|---|---|---|
| S1 reference row: uniform N=32 (the lmms-eval default), frames then question, native frames (360p / 480p), all rows, Listening excluded | `outputs/scaleup/minerva/qwen2.5-vl-7b/s1_eval/native/uniform_N32/plain/20260929_214137_faithful` (164965) | **386/1188 = 0.325 [0.299, 0.351]**, parse_fail 0, 13,024 tokens mean; chance 0.20, letter prior 0.215; coverage (>= 1 parsed-evidence frame in view) 0.63. Per skill: state changes 0.47, object recognition 0.40, reading 0.39, counting 0.29, temporal 0.29, counterfactual 0.13. Third-party Qwen2.5-VL-7B-based rows at 32 frames: 0.29-0.32 | DONE |
