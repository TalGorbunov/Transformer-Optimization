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

## S1u — VNBench (Qwen2.5-VL-7B, 4-bit), per-needle perception baseline (plan `docs/VNBENCH_2026-10-09.md`)

| experiment | canonical run | headline | status |
|---|---|---|---|
| VNBench landing: zip -> videos -> probe -> pool_native (256) -> units (try-0 rows, 1,350) | `outputs/prepare/vnbench/` (175136 frames; 175150 verify + units after the `stage_verify` fix) | 1,350 mp4, 150 per subtask; every needle inside its decoded duration (121 end past the json `length`); pool_native 27 GB; units_native 9,034 units (3,929 positive), 5.5 GB | DONE |
| T: question alone, 1,350 try-0 rows | `vnbench/qwen2.5-vl-7b/s1u_unit/T/uniform_N8/20261010_*` (175208) | **0.250 [0.229, 0.273]** = chance on every subtask (0.18–0.33); nothing is text-answerable | DONE |
| E frame@lo / frame@hi / clip5_d1@hi: evidence units only, official layout | `…/s1u_unit/E_frame_lo/…` (175209), `…/E_frame_hi/…`, `…/E_clip5_d1_hi/…` (175210) | **0.880 / 0.878 / 0.807**; ret 0.95–1.00, ord 0.99, cnt_edit1 1.00, cnt_insert1 0.81, cnt_edit2 0.18–0.21 (the one wall); by k 0.97 / 0.83 / 0.81; resolution gain −0.002; the 3-frame clip hurts counting (duplicate views inflate the count) | DONE |
| D frame@lo / frame@hi: per-unit yes/no vs hard + easy negatives | `…/s1u_unit/D_frame_lo/…` (175216), `…/D_frame_hi/…` (175217); 9,034 units | per-task AUROC vs hard **0.92–1.00** (ret 1.00 / 1.00 / 0.95, cnt 0.99, ord 0.99 / 0.98 / 0.92), hard ≈ easy; pooled 0.94, rec@spec99 0.49 (one global threshold; yes-rate on positives 0.00–0.24 except ret 0.99) | DONE |
| UNIT table (seven columns + gaps) | `…/s1u_unit/TABLE.md` | pooled wall 0.627 [0.599, 0.655]; fence cost (E frame hi − planted oracle gate) ret 0.00–0.03, ord 0.71–0.79, cnt 0.44–0.77; retrieval room vs uniform N=8 0.48 | DONE |
| Frozen curve in N: `uniform` N=8…256, plain, lo res (max_pixels 151,200), all 1,350 try-0 rows | `vnbench/qwen2.5-vl-7b/s1_eval/native_mp151200/uniform_N*/plain/` (175153, 175154; N=256 175240) | **0.40 / 0.49 / 0.58 / 0.67 / 0.68 / 0.65** at N=8…256: rises while needles come into view, turns down at N=256 where every row has every needle sampled (exact check 1,350/1,350); the every-needle-in-view rows fall 0.93 → 0.65 with the planted ladder's slope; at N=256 ret 1.00 / 0.99 / 0.87, ord 0.80 / 0.51 / 0.43, cnt 0.60 / 0.30 / 0.32 | DONE |
| Planted ladder, plain: needles in view + fillers, N=8…128 (N=256 impossible: pool = 256) | `…/s1_eval/native_mp151200/planted_N*/plain/` (175155, 175156) | **0.84 / 0.79 / 0.76 / 0.72 / 0.69**: retrieval flat ≈ 1.0, ordering 0.86–0.97 → 0.48–0.83, counting 0.78–0.80 → 0.39–0.58; by k: k=1 0.98→0.93, k=2 0.96→0.73, k=3–4 0.73→0.53 (the share-law collapse on real video) | DONE |
| Planted ladder, oracle gate: fenced + evidence labels, batched path, N=8…128 | `…/s3_oracle/native_mp151200/planted_N*/gated/` (175157, 175158) | **0.53 / 0.49 / 0.49 / 0.49 / 0.53**: flat by construction; ret 1.00 / 0.99 / 0.93, ordering = chance, counting 0.18–0.37 (by k 0.94 / 0.43 / 0.25 / 0.00) | DONE |
| Fence-cost decomposition on `planted` N=8…64: question-first unfenced (175236), fence only with the question last and all blocks kept (175250, dense masked path), fenced question-first all kept (175237; N=64 rerun 175482 after a TIMEOUT at 600/1,344 rows) | `…/s1_eval/native_mp151200/planted_N*/{qfirst,fenced_qlast,fenced_qfirst}/` | pooled N=8/16/32/64: plain 0.84/0.79/0.76/0.72 · qfirst 0.59/0.48/0.37/0.36 · fence-only 0.50/0.46/0.44/0.37 · fenced q-first 0.47/0.32/0.25/0.21 · + oracle gate 0.53/0.49/0.49/0.49; the reset alone puts ordering at chance and counting ≤ 0.35 at every N; ungated, the fence makes retrieval N-sensitive too (k=1 rows 0.91→0.67; fenced q-first 0.81→0.39), which the oracle gate repairs (ret 1.00/0.99/0.93, flat) | DONE |
| Figure | `…/fig/F_vnbench_collapse.{png,pdf,csv}` (`experiments/figs/vnbench_fig.py`) | retrieval / ordering / counting / by-k panels: planted plain vs uniform plain vs planted oracle gate | DONE (re-rendered 09:30 with N=256) |

## S2/S3 — VNBench week 1 (2026-10-11; program table in `docs/VNBENCH_2026-10-09.md`; STATE entry 2026-10-11 15:45)

| experiment | canonical run | headline | status |
|---|---|---|---|
| Cell 1 self-gate: label-free within-question rules on the D regime's stored scores (`experiments/self_gate.py`) | `vnbench/qwen2.5-vl-7b/s1u_unit/SELF_GATE_{lo,hi}.md` (CPU on 175216 / 175217) | literal "yes" keeps 13% of needles pooled; max-2 log-odds rule recall 0.95 / rejection 0.90 / exact 0.78; top-k ceiling exact 0.94; no single margin fits every task | DONE |
| Cell 3 capture: planted N=16 + 64, fenced question-first, L12/20/24 slot states | `…/s2_gate/native_mp151200/planted_N{16,64}/fenced_qfirst/` (175646) | 21,600 + 86,016 frames, pos rate 0.18 / 0.045 | DONE |
| Cell 3 fits: 2-fold by haystack (train N=16), leave-one-task-out, N16->N64 transfer, saved fold gates | `…/s2_gate/fits/{fold2_trainN16,loto_N16,transfer_N16_to_N64,heldout_fit{A,B}}.json`, `gate_L20_fit{A,B}.npz` (175897-175901) | **L20 per-frame 0.989 / AUC 0.999 held out by haystack, identical at N=16 and 64**; leave-one-task-out AUC 1.000 on 7/9 unseen tasks (picture tasks 0.96 / 0.99 at L12); fold gates 10,800 frames each, held-out 0.995 | DONE |
| Cell 3 gated ladder, fitted gate on the held-out haystack fold, N=16..128, batched path | `…/s3_model/native_mp151200/planted_N*/gated_fold{A,B}/` (175911, 175912, 175966, 175967), pooled table `…/s3_model/GATED_MODEL.md` | **0.48 / 0.48 / 0.48 / 0.51** (1,350 rows) vs oracle 0.49 / 0.49 / 0.49 / 0.53 vs plain 0.79 → 0.69; by k 0.93 / 0.43 / 0.25 / 0.01; gate exact set 0.94 → 0.90, false blocks 0.1 → 1.0 per question (linear in N), missed 0.01 flat | DONE |
| Cell 4 Gemma-3-12B: plain and oracle-gated planted ladders, N=8/16/64, `--res lo`, dense path | `vnbench/gemma-3-12b/{s1_eval/…/plain,s3_oracle/…/gated}/` (175663, 175664; N=64 standalone 175910, 175914), `…/CURVES.md` | plain **0.713 / 0.660 / 0.579**, gated **0.505 / 0.469 / 0.469** (parse_fail ≤ 5%): the Qwen shape; by k plain 0.96 / 0.86 / 0.53 → 0.92 / 0.63 / 0.36 | DONE 10-11 |
| Cell 4 InternVL3.5-8B: same ladders | `vnbench/internvl3.5-8b/…` (175665; 175666 cancelled at N=64) | plain 0.408 / 0.323 / 0.445, gated 0.418 / 0.384 / — with **parse_fail 0.34-0.58** (prose openings under the VNBench suffix) → not a ladder; readout decision pending | SUPERSEDED by the week-2 letter-readout ladders (below) |
| Cell 5 official circular score, all 4 option orders, uniform N=16 / 64 (`vnbench4`, `experiments/vnbench_circular.py`) | `outputs/scaleup/vnbench4/qwen2.5-vl-7b/CIRCULAR_N{16,64}.md` (175657, 175658) | **0.363** (ret 0.569 / ord 0.291 / cnt 0.229) at N=16, **0.556** (0.844 / 0.518 / 0.304) at N=64 vs published Qwen2-VL-7B 33.8 (69.1 / 12.4 / 20.0); circular costs 11-13 points vs try 0 | DONE |
| Cells 2 + 6 (code): `evaluate.py --gate-npz`, `gate_fit.py --save-gate / --leave-one-task-out`, `--qids-file` skips rows a planted split cannot build, inside-interval `uniform_labels`, `curve_table.py` pools `*` cells | `tests/test_gate_model.py`, `tests/test_data_videoqa.py` | 15 test files green | DONE |

## S4/S5 — VNBench week 2 (2026-10-11/12; items 1-3 of the week-2 list; STATE entry 2026-10-12 09:30)

| experiment | canonical run | headline | status |
|---|---|---|---|
| Item 1 two-pass frozen method: fenced scan + fitted L20 gate (the week-1 fold gates), kept frames read unfenced in the official layout, held-out haystack fold, planted N=8..128 | `vnbench/qwen2.5-vl-7b/s4_method/native_mp151200/planted_N*/twopass_fold{A,B}/` (177259, 177260), pooled `…/s4_method/TWOPASS.md` | **0.881 / 0.873 / 0.871 / 0.866 / 0.880** = E (0.880) at every N vs plain 0.84 → 0.69 and the fenced gated read 0.48; by k at N=128 0.96 / 0.98 / 0.82 / 0.79; rows with the exact gate set 0.89 at every N, false frames only 0.69–0.75, a missed needle 0.41–0.47 | DONE |
| Item 1 two-pass oracle gate (labels), same read | `…/s4_method/native_mp151200/planted_N*/twopass_oracle/` (177353) | **0.885 / 0.877 / 0.878 / 0.877 / 0.891**: the two-pass ceiling, flat in N; the fitted gate is within 0.4–1.1 points | DONE |
| Item 2 attention photograph, plain planted read, 20 rows × 9 tasks per N, layers 12/16/20/24/27 | `…/s5_photo/native_mp151200/planted_N*/plain/` (177467) | L20 needle mass **0.030 → 0.005** (N=8 → 128), filler mass 0.032 → 0.091, prompt + sink 0.90–0.94; per-frame needle / filler ratio ≈ 2 at L20 at every N (share law e^s ≈ 2), 2 → 7 at L27 | DONE |
| Item 2 photograph under the oracle-gated fence, N=16 / 128 | `…/s5_photo/native_mp151200/planted_N{16,128}/gated/` (177538) | filler mass exactly 0; L20 needle mass **0.371 / 0.369** (identical at both N; 19× / 72× the plain read's); first-token ord / cnt at chance, ret ≥ 0.9 | DONE |
| Item 2 MMReD anchor parity of the ported probe (char_at_frame seq_len_8, rows 0003834 / 0003841) | `outputs/_scratch/week2_smoke/photo_mmred_N8_plain/20261011_231149_177349` vs `outputs/diag/photo/multi/plain/N8/20260922_153238_156701` | 280/280 (qid, layer, head) rows, max abs diff 0.022, mean 0.004, seq_tokens 2756 in both | DONE (smoke) |
| Item 3 Qwen letters control: `--readout letters` on the dense plain path, N=8 / 16 | `…/s1_eval/native_mp151200/planted_N{8,16}/plain_letters/` (177360), `…/s1_eval/LETTERS_AGREEMENT.md` | **0.838 / 0.796** vs generated 0.837 / 0.795; row-level agreement 0.996 / 0.993 | DONE |
| Item 3 InternVL3.5-8B plain + oracle-gated ladders with the letter readout, N=8/16/64, `--res lo` | `vnbench/internvl3.5-8b/{s1_eval/native/planted_N*/plain_letters,s3_oracle/native/planted_N*/gated_letters}/` (177465, 177544), `…/CURVES_LETTERS.md` | plain **0.803 / 0.781 / 0.734**, gated **0.496 / 0.460 / 0.463**, parse_fail 0: the Qwen / Gemma shape; replaces the week-1 generated rows | DONE |

## S4 — VNBench week 3 (2026-10-12; STATE entry 2026-10-13 12:30)

| experiment | canonical run | headline | status |
|---|---|---|---|
| Cell 1 share-law fit on the week-2 photograph (`experiments/share_law_fit.py`) | `vnbench/qwen2.5-vl-7b/s5_photo/SHARE_LAW.md` (CPU on 177467 / 177538) | three-group law with a constant prompt+sink weight FAILS (C ≈ 660–2,540, sink keeps 0.90–0.94 at every N); frame-share law k e^s / (k e^s + N − k) fits L20 with **e^s = 1.94** within 0.02 at every N; L27 decays ~N^-0.5 | DONE |
| Cell 2a two-pass (fitted gate) under the uniform protocol, try 0, N=16/64/128/256, held-out folds | `…/s4_method/native_mp151200/uniform_N*/twopass_fold{A,B}/` (178004, 178005), pooled `…/s4_method/TWOPASS_UNIFORM.md` | **0.508 / 0.737 / 0.823 / 0.836** vs plain 0.49 / 0.67 / 0.68 / 0.65; keeps rising where plain turns down; gate keeps needle RUNS at high fps (7.3 / 14.5 / 28.9 blocks), counting pays (cnt_edit1 0.84 vs 0.98 planted) | DONE |
| Cell 2b circular four-try score, two-pass, uniform N=64, folds pooled (5,400 rows) | `outputs/scaleup/vnbench4/qwen2.5-vl-7b/s4_method/native_mp151200/uniform_N64/twopass_fold{A,B}/` + `twopass_folds_pooled_eval.csv` (178006, 178007), `…/vnbench4/qwen2.5-vl-7b/CIRCULAR_TWOPASS_N64.md` | **0.677** = ret 0.849 / ord 0.744 / cnt 0.438 vs plain 0.556 (0.844 / 0.518 / 0.304) vs published Qwen2-VL-7B 33.8 | DONE |
| Cell 2b at N=256 (every needle in view) | — | not run (~26 GPU-h); pending the run-merge decision | PENDING |
| Cell 3 variant 2 of the read: kept frames at their full-prompt positions (`--two-pass-positions original`), planted N=8..128, held-out folds | `…/s4_method/native_mp151200/planted_N*/twopass_original_fold{A,B}/` (178013, 178014; 177993 / 177994 cancelled, memory leak), `…/s4_method/TWOPASS_POSITIONS.md` | **0.880 / 0.862 / 0.842 / 0.820 / 0.819** vs packed 0.88 flat; k=5-6 **0.74 → 0.17** (packed 0.79, plain 0.27): the counting collapse rides on positional spread, not only filler mass | DONE |

