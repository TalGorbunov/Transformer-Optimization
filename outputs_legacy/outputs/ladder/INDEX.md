# outputs/ladder/ — aggregation-ladder experiment group (plan 2026-07-08, workstream A)

| experiment | canonical run | headline |
|---|---|---|
| text-MMRED behavior (steps, N=8, n=250) | `text_mmred/behavior/20260708_200014/steps_in_room/20260708_200402/` | acc **0.196**, bias −1.95 (vs image 0.21 — extraction removed, wall unchanged) |
| text-MMRED locmap+cache (n=400, L4–24, off9/13, job 119518) | `text_mmred/locmap_cache/20260708_200751/count/` | model 0.165; dtc@L16 room+char **0.514**; per-frame AUROC 0.94; evidence distributed over many question tokens |
| text-MMRED parity (joint) | `text_mmred/parity/<ts>/` | room d′_w 1.65–1.88 (≈ image, NOT high); pred 0.27–0.30 vs ridge 0.27–0.41 (distributed channels) |
| text-MMRED parity (fenced, job 119525) | `text_mmred/parity_fenced/<ts>/` + `locmap_cache_fenced/20260708_202818/` | **fence NULL on text** (d′ ≈ joint; per-frame AUROC 0.91 vs 0.94) |
| text-MMRED native axis + law closure | `text_mmred/native_axis/20260708_200014/{20260708_200631,compare}/` | cos(nat,w*)=−0.004, d′_nat 0.80@L16 → **pred 0.163 vs measured 0.165** |
| text-CWE battery N=8/16/32 (job 119522, OOM at N=64) + N=64 rerun (119529) | `text_cwe/locmap_cache/20260708_201605/N{8,16,32,64}/text_cwe/` | behavior **0.793/0.547/0.540/0.473**; per-frame AUROC ≈ 1.0; carrier = quoted word (off11) |
| text-CWE parity | `text_cwe/parity/<ts>/` | word@L16 N=8: d′_w 5.1, pred 0.69 vs ridge 0.73 (model 0.79 ≥ single-carrier ceiling) |
| text-CWE native axis | `text_cwe/native_axis/20260708_201605/{20260708_205647,compare}/` | d′_nat 1.4@L16, pred@nat 0.30 ≪ model 0.79 — single-axis account fails on CWE (content-addressed reading) |
| text-MMRED longN caches N=16/24/40 (job 119532) | `text_mmred/longN/20260708_214148/` | joint d′_AUC plateaus at 1.39–1.40 for N≥16 (from 1.91 @N=8); law tracks the collapse (RESULTS [2026-07-08e]) |
| text-MMRED longN parity | `text_mmred/parity_longN/<ts>/` | pred 0.141/0.116/0.078 vs ridge 0.139/0.083/0.033 (N=16/24/40) |
| C-range N=40 fact-interface test (job 119531) | `text_mmred/c_range_n40/20260708_213957/` | base 0.027 vs fact-written counts **0.987/1.000 over 0–40** (all two-digit values verbalize) |

Smokes for this group live in `outputs/_scratch/` (b0a_res_smoke → RESULTS [2026-07-08]; c1_token_interface → RESULTS [2026-07-08b]); broken runs archived in `outputs/_scratch/broken_runs/`.

## 2026-07-10/11 campaign additions (RESULTS [2026-07-10]–[2026-07-11b])

| experiment | canonical run | headline |
|---|---|---|
| **A1-fu1 text multipass cache** (job 119991) | `text_mmred/multipass_cache/20260710_211249/count/` | perception 0.995; mp-sum solution **0.965** |
| **A1-fu1 multipass block read** (120010) | `text_mmred/block_read_multipass/20260710_215350/` | **d′ 7.9 @L16 (joint 2.45) — NOT write-capped; cap is joint-context** |
| **A1-fu2 easy-text cache + block read** (119992/120011) | `text_mmred_easy/{locmap_cache/20260710_211249,block_read/20260710_215350}/` | **d′ 2.89, model 0.215 — binding-format account REFUTED** |
| **B1 caches** joint/fenced/multipass × N∈{8,16,32,64,128} @392px (120013–22,120004) | `image_longN/{joint,fenced,multipass}/N*/2026071*/` | dirs with CORRUPT.marker = quota-killed wave-1 partials |
| **B1 d′ vs N + Fig B1** (120065) | `image_longN/dprime_vs_n/20260711_003548/` | **joint FLAT ~2.0 ∀N ≪ 6.3; multipass 7.2–8.1 ∀N; fenced ≤ joint** |
| **B2 gate calibration + Fig B2** (120059) | `image_longN/gate_calibration/20260710_231635/` | raw FN 0.09→0.99; mass-norm ÷10 drift but FP over-counts; bias law exact |
| **B3 behavior vs N** (120030–32) | `image_longN/behavior/N{8,16,32,64,128}/` | 0.173→0.020; **emitted range clamped ~3 ∀N**, ordinal corr 0.75 to N=64 |
| evidence-only behavior (120029) | `evidence_only_behavior/20260710_221821/` | 0.00 @N≥6 with ALL frames evidence |
| **A4 natural cell caches** (120042/46/47/48) | `natural/<cell>/2026071*/herbench_ac/` | model 0.657/0.580/0.567/0.607; dtc 0.94–1.00 |
| **A4 d′ dial + Fig A2** (120056–58) | `natural/dprime_cells/20260710_231039_off10/` | **6.21/5.41/4.30/4.28 — dial works; model ≈ law (no-binding rung)** |
| A3 MLVU-AC behavior N=32 (120069) | `mlvu_ac/behavior_N32/<ts>/` | MCQ 0.282 (chance .25), open 0.112 — **sampling-limited** (judge: 35% of Qs zero visible evidence) |
| A3 MLVU-AC judge labels N=32 (120070) | `data/mlvu_ac/*/lookagain_N32.json` | ~0.37 visible frames per gold instance |
| A3 MLVU-AC behavior N=128 (120071) | `mlvu_ac/behavior_N128/<ts>/` | in flight |

Readout-group runs (C1b grid, C2/C3 codebook table incl. **token-interface-necessary**) live in
`outputs/readout/` — see `outputs/readout/INDEX.md`. Figures for the living report:
`outputs/ladder_report/fig_{a1_ladder,a2_natural_dial,b1_final,b2_final,c1_routes}.png`.

## Continuation campaign additions (2026-07-11, RESULTS [2026-07-11d]–[2026-07-11k])
| experiment | canonical run | headline |
|---|---|---|
| **E4 sweep** natural cells + long-N joint (120130–38) | `natural/block_read/20260711_125059_*`, `image_longN/block_read/20260711_125059_N*` | adequacy tracks evidence DIVERSITY (dist PASS / ident FAIL); long-N kurt +0.5→+25 |
| **MLVU d′/parity** (120139/120150) | `mlvu_ac/{msgcache_n32judge/20260711_125240,block_read/20260711_130831}/` | action-token carrier, block d′ 2.80; E4 std-ratio 6.6–7.9 (judge noise); model at its delivered-evidence ceiling |
| **law+clamp closure (fig_b3_clamp)** | `image_longN/law_clamp/<ts>/` | near-exact zero-param account of B3 (err 0.013 vs plain law 0.13); transfers directionally |
| **P1f attn-renorm patch** (120145/46/49/54) | `image_longN/renorm/{N8,N32}/` + `renorm/dprime_N*` | **mass competition REFUTED** (renorm ≈ joint; ~1–8% of gap) |
| **P1e chunk sweep** (120143/120166) | `image_longN/chunk_sweep/{k*,dprime_20260711_134108}/` | **d′ 8.08→3.37 at k=2** → 1.98 at k=32; tax onsets at first companion |
| native axis vs N (120144, partial) | `image_longN/native_axis/N{8,16}/` | axis STABLE: |cos(a16,a8)| 0.82–0.86; N=32/64 legs pending H200 |
| **dedup semantics** (120147) | `natural/dedup_semantics/<ts>/` | question wording selects Σ vs distinct-count (ident dissociation 0.83 exact-vs-unique) |
| **VNBench port** (120155/58/59/60/70) | `data/vnbench_cnt*`, `vnbench/{behavior_N32,behavior_N64,msgcache_n32exact,block_read}/` | E4 var-ratio ✓; d′ 2.50 = MMRED level (✗ CWE-like); carrier flat (✗); clamp reappears |
| **e2e pipeline (chunked)** (120151) | `outputs/pipeline/e2e_tally_chunked/` | 0.600/0.280/0.193 @N=8/16/32 (2–5 fwd) ≫ frozen; dies ≥64 (k=8 d′ ≈ joint) |
| **e2e pipeline (retrieve)** (120162/63/64) | `outputs/pipeline/e2e_retrieve_N*/` | **0.880/0.807 @N=32/64** (frozen 0.053; mp-sum 0.680/0.580); N=128 in flight |

## 2026-07-13 additions

| experiment | canonical run | headline |
|---|---|---|
| **B1 small-N caches** joint N∈{1,2,4,6} @392px (jobs 120778–81, a100) | `image_longN/joint/N{1,2,4,6}/20260713_1614*/count/` | frames-first deployed-carrier caches below N=8 (n=100/200/300/300) |
| **B1 small-N d′ vs N** (CPU, on the above + N8/20260710_215405) | `image_longN/dprime_vs_n_smallN/20260713_smallN/` | joint L16 d′ flat ~2.0 from N=2 (1.93/1.95/2.20/1.97 @N=2/4/6/8); model 1.00/.875/.497/.303 @N=1/2/4/6 — model ≥ single-carrier law-pred at N≤4, falls behind from N≈6; N=1 d′ unmeasurable (probe saturated) |
| **small-N v2** gen (120800–03: `data/mmred_smallN_park/`, ~1200/len) + joint caches (120804–07) + multipass caches (120811–13) | `image_longN/{joint,multipass}/N{1,2,4,6}/20260713_17*/count/` | 1,300-sample frames-first caches, both arms, N∈{1,2,4,6} |
| **small-N v2 d′ vs N** (CPU) | `image_longN/dprime_vs_n_smallN/20260713_v2_1300/` | joint L16 d′ 2.71/2.52/2.48 @N=2/4/6 (v1 small-cache values were estimator-limited); multipass 6.72/7.96/8.44 (AUC-cap caveat); joint model 0.997/.889/.480/.338 @N=1/2/4/6; mp tally .993/.974/.953 |
| **repair-2×2 missing cell** joint-q × un-mixed-v (job 120864, a100 160G, 7.6 min) | `image_longN/qkv_2x2/20260712_n500/jointq_cell/` | **1.72±0.05** @L16 (vs joint 2.09, clean-q ceiling 6.33) — value repair buys nothing behind a joint query; cross-checks reproduce logged 3.82/6.17/2.09 exactly |
| **N=1 cache rebuild** (gold-0 builder fix in probe_frame_to_carrier_message.py; job 120903) | `image_longN/joint/N1/20260713_211210/count/` + CPU analysis (console) | n=1200 balanced (600/600); joint@N=1 d′_w **6.40±0.16** (AUC-cap 5.26), bal-acc 0.998, law-pred(N=1) 0.999, model acc 0.994 — a frame alone in its forward reads at multipass quality; complete 0..1 prior (N≥2 caches keep gold≥1 convention) |
| **trained-query ceiling (Exp 2)** shared q* trained on joint k/v, logistic proxy, 3 inits (jobs 120908 v1 + 120915 v2 traj, a100 ~13/40 min) | `image_longN/qkv_2x2/20260712_n500/trained_query/` (v1 at `trained_query_v1_notraj/`) | anchors reproduce exactly (2.09/2.33, 3.82/4.47, PASS); trained q* eval d′ **0.36–0.48**, trajectory max over ALL epochs/inits 0.51 (cherry-picked-on-eval bound) — a fixed shared query can't even match the sample-specific joint query (2.09), 10× below the ≥4 GO bar → slot-head NO-GO; per-frame addressing must be architectural |
| **replica carriers** masked + unmasked interleaved (job 121431) | `image_longN/replica_carrier/20260714_214534/` + `replica_carrier_nomask/20260714_221634/` | **unmasked interleave d′ 3.56 @L16 (+81% over joint 1.97, one forward, zero training)**; masked 2.52 (per-copy ladder = value contamination); in-run off−9 anchor invalid; RESULTS [2026-07-14] |

## 2026-07-15 additions (oneforward brief: fence / compose / gate)

| experiment | canonical run | headline |
|---|---|---|
| **Exp A: replicas + FULL frame fencing** (`replica_carrier_probe.py --fence-frames`, job 121925; smoke 121917) | `image_longN/replica_fence/20260715_203619/` | **d′ 4.07±0.08 @L16 — best one-forward supply** (masked 2.52 / unmasked 3.56); per-copy NOT flat (3.81→~2.0–2.9) → residual gap vs mp 6.3–7.3 ≈ RoPE/position term; PARTIAL band |
| **Exp B1: un-mixer retrained + weights SAVED** (`encoding_unmixer.py --save-dir`, job 121919) | `image_longN/unmixer_saved/20260715_194450/` (weights: `unmixer_saved/weights/unmixer_L16.pt`) | recovers 84% of encoding gap offline: 3.82→5.94 (ceiling 6.33), mp query; prior run's 93%/6.17 ≈ retrain variance |
| **Exp B2: deployed composition** replica-q × unmix-v via k/v_proj hooks @L16 (`--no-mask --unmix-dir`, job 121928; smoke 121926) | `image_longN/replica_unmix/20260715_204158/` | **NO TRANSFER — destructive: 3.56→1.44 @L16**; frame 0 alone improves (3.73→4.17, in-distribution); L14 control unchanged (2.70 vs 2.72); off-distribution (question-conditioned) k/v is the mechanism |
| **Exp C: CoGNN broadcast gate** additive per-token logit offset (`broadcast_gate_probe.py`, job 121918) | `image_longN/broadcast_gate/20260715_194451/` | anchors PASS (2.09/3.82); **content arm 1.80 eval (traj max 2.13) = floor → routing NOT repairable from content**; qcond arm 30.69 INVALID (q_pad feature leak) |
| **Exp C follow-up: q_pad leak probe** (`qpad_leak_probe.py`, job 121927) | `image_longN/broadcast_gate/qpad_leak/20260715_201750/` | q_pad itself d′ 8.63 eval (q_mp 10.32) ≥ mp×mp ceiling — leak proven; content features k/v-mean ~0.65 — content arm had nothing to exploit |

## 2026-07-17 additions (position reset)

| experiment | canonical run | headline |
|---|---|---|
| **Exp A2: replicas + fence + per-block M-RoPE reset** (`replica_carrier_probe.py --fence-frames --reset-positions`, job 122744; smoke 122739) | `image_longN/replica_posreset/20260717_181630/` | **d′ 4.66±0.05 @L16 — new one-forward best** (fenced 4.07); per-copy FLATTENED (3.70 3.57 3.17 3.00 3.09 2.30 2.84 3.49) — position term real; residual vs mp band now shared by all copies incl. frame 0 |
| **Exp A2 N=1 anchor control** same script/flags on `mmred_smallN_park/seq_len_1` (job 122764) | `image_longN/replica_posreset_N1anchor/<ts>/` | pending — truly-alone frame through identical machinery; splits protocol/estimator from real one-forward residual |
| **Exp A3: full block-diagonal fence** (`--fence-blocks`, marker leak sealed; job 122809) | `image_longN/replica_blockfence/20260717_190158/` | **GO: d′ 6.34±0.11 @L16 (AUC-cap caveat), per-copy FLAT 3.6–4.6 — multipass-in-one-forward confirmed** (pre-reg band ≥5.5); vision markers were the last cross-frame channel |
| **N=1 solo anchors, same instrument** (jobs 122782 smallN n=1200; 122810 parkimg n=200; 122764 NaN=single-class lesson) | `image_longN/replica_posreset_N1anchor/20260717_184440/` + `_parkimg/20260717_190153/` | solo 6.01±0.26 (n=1200) / 4.55 (n=200, deflated); subsample-to-300 = 4.76 → per-copy scale is estimator-deflated, pooled is the valid scale |
| **Exp A4: Q-first + blockfence + posreset** (job 122888) | `image_longN/replica_blockfence_qfirst/20260717_195501/` | **d′ 9.24±0.33 @L16 (AUC-cap caveat) — Q-first is a supply amplifier** (question-conditioned frame encoding); gate→tally **0.991±0.008** exact @N=8 (A3 layout: 0.960) |
| **N-sweep: blockfence+posreset N=16/32/64/128** (jobs 122889–92) | `image_longN/replica_blockfence_N{16,32,64,128}/20260717_*/` | **supply FLAT: d′ 7.62 / 7.81 / 7.55 / 7.24 (= mp band) to N=128**, per-copy flat, no OOM @seq≈28k |
| **Gate→tally vs N** (CPU on the sweep caches) | (analysis in `plans/oneforward_DRAFT_RESULTS.md` A4) | **one-forward exact-count 0.960/0.976/0.960/0.952/0.876 @N=8–128** — beats retrieve-then-verify (0.79@128) and multipass tally (0.42@128) at every N; 44× frozen |
| **Q-first long-N** (jobs 123027/123028, N=32/128) | `image_longN/replica_blockfence_qfirst_N{32,128}/<ts>/` | pending — Q-first amplifier at scale |
| **Q-first long-N** (jobs 123027/28) | `image_longN/replica_blockfence_qfirst_N{32,128}/20260717_2105*/` | d′ 12.67 / 11.57; **gate→tally exact 1.000 @N=32, 0.984 @N=128 — one forward, 50× frozen** |
| **Learned carrier token** proxy+distill arms (jobs 122938/39) | `image_longN/carrier_token/20260717_201919_{proxy,distill}_room_k1/` | 1 trained embedding (3.6k params) replaces the 20-token replica: **distill eval d′ 8.35 (93% of teacher 8.95), full-n ~9.0**; untrained warm-start already 5.23; trained-query floor 0.4 crushed 20× |
| **Carrier length-gen** eval-only N=32/128 (jobs 123128/29) | `image_longN/carrier_token_lengen_N{32,128}/20260718_*/` | N=8-trained carrier zero-shot: **d′ 11.40 / 9.71**; full N=8-trained stack (carrier+gate) zero-shot exact **0.917 / 0.860**; per-N gate refit 1.000 / 0.988 |
| **Carrier ablations** random-init / k=2 / k=4 (jobs 123124–26) | `image_longN/carrier_token/20260718_0058*_distill_*/` | random init → same endpoint (8.25 vs 8.35) — mechanism learned, not init; k=1 suffices (8.14–8.38 flat); all arms 92% of teacher |
| **Natural images Q-first blockfence** dist_far/near (jobs 123136/37) | `image_longN/replica_natural_{dist_far,dist_near}/20260718_*/` | real photos: d′ 6.22 / 5.69 (in-run joint 3.12 / 3.61); gate→tally 0.920 / 0.760 vs model-alone 0.58/0.61 (n=50, wide bars) |
| **Cooc Q-first blockfence N=8** (job 123132) | `image_longN/replica_cooc_qfirst/20260718_011918/` | relational predicate: d′ 6.36 (joint anchor 4.40); **gate→tally 0.973 exact — new cooc record** (prev 0.766); steps→cooc gate zero-shot only 0.460 (gate is per-task) |
| **Cross-task carrier** steps-e_c on cooc (job 123145) | `image_longN/carrier_token_crosstask_cooc/20260718_020945/` | zero-shot d′ 5.58 (~88% of cooc teacher 6.36); + per-task gate → 0.880 exact — carrier is task-general, gate is per-task |
| **Cooc long-N Q-first blockfence** (jobs 123133/34) | `image_longN/replica_cooc_qfirst_N{32,128}/20260718_*/` | d′ 8.45 / 7.58 flat; gate→tally 0.932 @N=32, 0.680 @N=128 |

## 2026-07-18 additions (stage-2 session)

| experiment | canonical run | headline |
|---|---|---|
| **Cross-DOMAIN carrier** steps-e_c on natural dist_far, eval-only (job 123208) | `image_longN/carrier_token_crosstask_natural/20260718_122538_proxy_room_k1/` | zero-shot d′ 3.19±0.55 (~51% of the cell's replica teacher 6.22); fresh gate tally 0.432 < frozen model 0.58 — carrier transfer is task-general but partly domain-bound (n=50, wide bars) |
| **Frozen baseline FULL prior N=8** LIMIT=900 (job 123225) | `image_longN/frozen_baseline/20260718_125303/` | acc **0.219** MAE 1.86 (gold uniform 0..8; g4+ near-zero = undercount clamp). Retires the truncated-prior 0.513 (123205) |
| **Q-first blockfence probe FULL prior N=8** LIMIT=900 (jobs 123232+123236) | `image_longN/replica_blockfence_qfirst_full900/20260718_130546/` (+`gate_tally/`) | d′ 13.54±0.27 @n900 (matched-n300 10.5-10.9 vs truncated 9.24); **gate→tally 0.998±0.001 exact, MAE 0.00** vs majority 0.111 — headline N=8 scaffold on the full prior |
| **Stage-2 30-ep convergence ref** truncated prior (job 123206) | `image_longN/carrier_layer/20260718_122503_L17_r8/` | best 0.840 @ep12, loss→0 by ep14, flat to ep30 — undertraining refuted; gap is data-limited |
| **Carrier-token distill FULL prior N=8** 450/450 (job 123233) | `image_longN/carrier_token/20260718_130545_distill_room_k1/` | eval d′ **11.45 @ep9 = 96% of scale-matched teacher 11.94** (teacher anchor 13.54 = probe run exactly); converged ep5 |
| **Stage-2 FULL prior steps-only** 450/450, 40ep (job 123235) | `image_longN/carrier_layer/20260718_131157_L17_r8/` | best emitted **0.678 @ep30** (scaffold 0.998, frozen 0.219); per-count uniform incl. g8 — clamp dead; train loss→0 = memorizes 450 → data-starved |
| **Stage-2 steps+cooc mixture** 30ep (job 123237) | `image_longN/carrier_layer_mixture/20260718_133821_L17_r8/` | best **0.693 @ep25** (cooc 0.796 / steps 0.560) — one carrier + one LoRA, no task interference; data-starved |
| **Stage-2 3-task +rooms** 30ep (job 123240) | `image_longN/carrier_layer_mixture3/20260718_134209_L17_r8/` | best 0.509 @ep12; rooms 0.50 (frozen 0.087, pipeline 0.993) — cross-carrier set-union PARTIALLY learned; data-starved |
| **E3: steps ckpt @N=32** eval-only (job 123742) | `image_longN/carrier_layer_eval_N32/20260718_182252_L17_r8_evalonly/` | RAW 0.138 / restricted 0.126, in-range 0.181 — N=8-only-trained LoRA does NOT length-extrapolate (supply does); motivates pooled variable-N P1 |
| **E4a: steps ckpt →cooc zero-shot** eval-only (job 123743) | `image_longN/carrier_layer_eval_cooc0shot/20260718_182247_L17_r8_evalonly/` | 0.179 (chance 0.111, mixture 0.796) — in-model zero-shot task transfer absent; task-generality requires mixture training |
| **Stage-2 pooled GO** P1 3-task 6k (job 123741, agent) + cached 2-task 5.1k frozen-e_c (job 123937) | `image_longN/carrier_layer/…` + `carrier_layer_cached/20260718_192428_L17_r8/` | **all-in-model emitted answer 0.963@ep5 (climbing) / 0.980@ep10** vs scaffold 0.998, frozen 0.219; clamp dead, per-count uniform; frozen e_c suffices; cached trainer 4× faster (934s/ep) |
| **P1-CACHED 2-task frozen-e_c** 15ep (job 123937) | `image_longN/carrier_layer_cached/20260718_192428_L17_r8/` | **0.980 @ep10** (steps 0.987 / cooc 0.946) MAE 0.02 — GO; frozen e_c matches trainable; 934s/ep workhorse |
| **Exam: cached ckpt @N=32** (job 124275) | `image_longN/cached_eval_N32/20260719_000556_L17_r8_evalonly/` | **0.097** — collapse to "0" at unseen length; readout binds to trained N range |
| **Exam: cached ckpt →rooms zero-shot** (job 124276) | `image_longN/cached_eval_rooms0shot/20260719_000556_L17_r8_evalonly/` | 0.153 (chance 0.111) — no zero-shot task transfer (2nd task pair) |
| **Exam: LoRA drift on plain prompt** (job 124277) | `image_longN/frozen_baseline_driftlora/20260719_000556/` | 0.313 vs frozen 0.219 — adapter safe left on; slightly HELPS plain prompting |
| **TRACK B: InternVL solo-QFIRST probe** (job 124280) | `outputs/frame_axis/internvl/multipass_qfirst/20260719_004112/` | d′ **6.31/5.11 @L16/20** vs joint 1.79/1.90 (**ports, 3.5×**); vs plain solo 6.38/6.56 → **Q-first amplifier does NOT port** |
| **C1/C2 ablation battery** 7 arms, steps8+cooc 900-train 8ep (jobs 124300-06) | `image_longN/cached_ablations/{base,L12,L22,r4,r16,noqfirst,noposreset}/` | ranking: **L12 0.941** > r16 0.731 > base 0.698 ≈ r4 0.694 > noposreset 0.669 > L22 0.513 > **noqfirst 0.378** — Qfirst most load-bearing; posreset mild; earlier opening ≫ |
| **P1 pooled 3-task stage-2 (trainable e_c)** 6k, 12ep (job 123741) | `image_longN/carrier_layer_pooled/20260718_182248_L17_r8/` | **0.999 @ep12** (rooms 108/108, cooc 161/162, steps 630/630) — in-model ≥ scaffold 0.998; data curve 450→0.678, 6k→0.999 |
| **A3 scratchpad+jitter** 5.8k, ckpt ep1 (job 124282, cancelled post-convergence) | `image_longN/carrier_layer_scratchpad/20260719_005342_L17_r8/` | TF-count 1.000 @ep1 (tf-exact 0.990 @ep3); **in-dist GREEDY 0.953, parse-fail 0.000** (124314) |
| **A3 exams: NIAH-0shot / union-0shot** (124316/124335) | `image_longN/scratchpad_eval_{niah0shot,union0shot}/` | 0.087 / **0.321** — alien question type dead, shared-format untrained reduction partially composes |
| **5-task scratchpad mixture** (+NIAH+union), ckpt ep1 (job 124336) | `image_longN/carrier_layer_scratchpad5/20260719_031356_L17_r8/` | TF 1.000 all 5 @ep1; **in-dist greedy 0.966** (which 0.992, union 0.910, steps 0.997, cooc 0.944, rooms 0.842) (124349) |
| **P1-ckpt exams** N32/NIAH/union (124353/54/55) | `image_longN/pooled_eval_{N32,niah0shot,union0shot}/` | 0.092 / 0.117 / 0.150 — variable-N digit readout: no length, no task, no composition transfer |
| **A3 exam N=32 zero-shot** (124315) | `image_longN/scratchpad_eval_N32/20260719_024313_*/` | **0.215** (in-range 0.311), parse-fail 0 — 2.2× digit collapse but below band → A4 |
| **A4 long-N scratchpad** (+N16 all, +N32 first-200; grad-ckpt) TF 0.997 @ep3 (job 124362) | `image_longN/carrier_layer_scratchpad_longN/20260719_054023_L17_r8/` | held-out N=32 greedy **0.447** (in-range 0.626, parse-fail 0, MAE 1.44) (124376) — 2.1× the zero-shot 0.215; g>8 unsolved |
| **E-D no-harm MME+POPE** base vs le16-LoRA-on (job 124508) | `image_longN/noharm_bench/20260719_203833/` | MME 0.862→0.860 (−0.2), POPE 0.862→0.848 (−1.4) — **no-harm GO**, adapter safe always-on |
| **E-C carriers-at-end distill** (job 124492) | `image_longN/carrier_atend/20260719_192758_distill_room_k1/` | **d′ 2.40 vs teacher 8.89 (27%) — layout-freedom strong form NO-GO**; E-C(b) isolates cause (124514) |
| **E-C(b) Q-first + carriers-at-end** (job 124514) | `image_longN/carrier_atend_qfirst/20260719_205916_distill_room_k1/` | d′ 10.27 (75% teacher), **tally 0.999 = interleaved** — carriers placement-free IF question leads; no-qfirst variant d′ 2.40/tally 0.508 |
| **E-E seeds 1/2 of headline tally arm** (124509/10) | `image_longN/carrier_tally_le16_seed1/20260719_203924_L17_r8/` · `image_longN/carrier_tally_le16_seed2/20260719_203925_L17_r8/` | TF-count **1.000 both** (tf-exact 0.962/0.966) — count cell 1.000 ± 0.000 across 3 seeds |
| **Posreset necessity** no-reset probes N=8/64 (jobs 124713/14) | `image_longN/noreset_N{8,64}/20260720_*/` | N=8: 7.74 (minor vs 9.24); N=64: 7.54 pooled but per-copy decays 6.4→3.0 — reset's value grows with N; keep, re-justified |
| **L12 full-data tally arm + N=32 zero-shot exam** (124698/124727) | `image_longN/carrier_tally_le16_L12/20260720_192738_L12_r8/` · `tallyL12_eval_N32/…/` | TF 1.000@ep2 (tf-exact 0.991); **N=32 zero-shot 0.443** vs L17 0.280 (pf 0, MAE 1.15) — **L12 = new default** |
| **N=128 cells** (124736 recovered, 124758) | `tally16_eval_N128/` · `tallyV2_eval_N128/` | le16 8×-0shot **0.087** (headline band REFUTED); v2 (≤64-trained) **0.118** — extrapolation wall robust; E-G/l12v2 = live candidates |
| **E-B SFT long-N** (124696) | `sft_control_le8_v2/20260720_191541_lora/` | N=16 0.480 / N=32 0.350, dead mid-range — no collapse but no aggregation; extremes-heuristic |
| **L12@N=32 zero-shot** (124727) | `tallyL12_eval_N32/` | **0.443** vs L17 0.280 — L12 default |
| **l12v2 (L12 + longN data + fixed ckpt criterion)** train+exams (124773, 124904/05/06) | `image_longN/carrier_tally_l12v2/20260721_071710_L12_r8/` + `tallyL12v2_eval_N{32,48,64}heldout/` | TF 1.000/tf-ex 0.976@ep5; **N=32 0.953 (pf 0) · N=48 0.878 cap-adj · N=64 0.678 cap-adj** — campaign-best long-N readout |
| **E-G pcouple8** train+first exam (124774, 124922) | `image_longN/carrier_tally_pcouple8/20260721_071710_L17_r8/` + `tallyPC8_eval_N32heldout/` | in-dist 0.955/0.832 (coupling fit cost persistent); **N=32 0.527 vs l12v2 0.953 same dirs** — GO failing at cell 1 |
| **E-G verdict** (124922/124923) | `tallyPC8_eval_N{32,64}heldout/` | coupled 0.527/0.212 (pf 0.04/0.37) vs uncoupled l12v2 **0.953/0.615** same dirs — **position coupling REFUTED** (in-dist cost + OOD format breakdown) |
| **FMT sweep: arm A (l12v2) in-dist + rooms controls** (125107/08) | `image_longN/tallyL12v2_eval_indist150/` · `image_longN/tallyL12v2_eval_rooms100/` | **in-dist-150 1.000 (all 5 tasks 30/30) · rooms-100 1.000**, pf 0, dec≤100 — l12v2 has NO rooms decode gap (0.84 was the L17 5-task ckpt); FORMAT-sweep controls |
| **FMT sweep trainers B/C/D + in-dist/rooms cells** (125104-06, 125183/84, 125190-93) | `image_longN/carrier_fmt_{scan,caption,chunked}/20260722_*/` + `fmt{B,C,D}_eval_{indist150,rooms100}/` | TF-fit: B 0.996 · C 0.994 · D 0.904 (A 0.976); **in-dist: B 1.000 · C 1.000 · D 0.987 (A 1.000); rooms: B/C 1.000 · D 0.920 (A 1.000)** — sanity bands met ×4; C≡B in-dist parity |
| **E-H L\* curve COMPLETE** (124965-972 collected, p0p2 P0.2) | `image_longN/tallyL{8,10,14,20}_eval_N{32,64}/` | N=32 0-shot: L8 0.277 · L10 0.373 · **L12 0.443 (peak, stands)** · L14 0.330 · L17 0.280 · L20 0.273 — inverted-U confirmed; L20 pf 0.107 |
| **FMT sweep length cells (partial)** B/C N=32 (125194/95) + D all (125185-87) | `image_longN/fmt{B,C,D}_eval_N{32,48,64}heldout/` | **B scan N=32 1.000 (pf 0)** > C 0.987 > A 0.953 > D 0.907; D N=48 0.679 / N=64 0.615 (pf 0 everywhere); B/C N=48/64 = 125196-99 running |
| **P1.2 measured before-ceiling** (125259, CPU on existing joint caches) | `image_longN/measured_ceiling/20260723_222428/` | measured best-linear-on-sum **0.317/0.281/0.189/0.183/0.122** @N=8-128 ≈ law-pred (0.307/0.246/0.175/0.137/0.096); MLP−linear ≤0.006; fig `_scratch/figs/pre_stage1_squashed_readout_measured.png` |
| **P1.3 E-B SFT N=64 cell** (125267, EFF/FLASH eval path) | `image_longN/sft_control_le8_v2_evalN64/20260723_225940_lora/` | **0.220** (pf 0, MAE 3.46, n=100; test_iid re-anchor 1.0000) — extremes heuristic, dead mid-range; SFT ladder 0.480/0.350/0.220 @N=16/32/64 |
| **FMT sweep DECIDED** (B/C N=64 landed: 125198/99) | `fmtB_eval_N64heldout/20260723_122357_*/` · `fmtC_eval_N64heldout/20260723_122649_*/` | **B N=64 0.942 (cap-adj 0.956) → scan GO; C N=64 0.981 pf 0 → WINNER = C (caption)**, ckpt `carrier_fmt_caption/20260722_222032_L12_r8/`; D chunking NO; N=48 cells pending (secondary) |
| **FMT sweep FINAL** (N=48 pair landed: 125196/97) | `fmt{B,C}_eval_N48heldout/20260723_122357_*/` | B N=48 **0.982** · C N=48 0.972 (pf 0 both); final length-mean B 0.975 / C 0.980 / A 0.786 / D 0.734 — **winner C (caption)**; sweep closed |
| **P2b MLVU-AC zero-shot carrier cell (32f)** (125350) | `mlvu_ac/carrier_eval_N32/20260724_064754_L12_r8_evalonly/` | open exact **0.000** (pf 0.214; 161/206 emit "0"); **MCQ nearest-option 0.107 ≤ frozen 0.282 → domain gap measured (prereg NO-transfer band)**; format survives, evidence detector domain-bound |
| **P3b InternVL scaffold gate→tally** (CPU on 124280's cache) | `outputs/frame_axis/internvl/gate_tally/20260724_165356/` | **0.938±0.031 exact @L16** (gate 0.991, majority 0.160; L20 0.892) — band ≥0.90 MET, scaffold ports; multipass-isolated label; digit-readout ref 0.586 |
| **P3a L1 natural supply** (125486) | `natural_mm/replica_supply_dist_far/20260724_180046/` | L16 replica d′_w **27.3** vs joint 7.75 (**3.5×**, per-copy flat 8.5-13.7); band MET; AUC-cap caveat; v1 n=50 was deflated |
| **P3a L2 natural gate→tally** (CPU) | `natural_mm/replica_supply_dist_far/20260724_180046/gate_tally/` | **0.980±0.012 exact** @N=8 (gate err 0.0025, majority 0.187) — band ≥0.85 MET, park-level |
| **P4.3 SFT-adapter no-harm** (125499) | `image_longN/noharm_bench_sft/20260724_190307/` | **MME −0.6 / POPE +1.2 pts → GO** (band ≤2; carrier ref −0.2/−1.4); fail dumps = clean yes/no, no digit leakage |
| **P3a natural ladder COMPLETE** (125486/88/92-95 + CPU) | `outputs/ladder/natural_mm/{frozen_baseline,replica_supply_dist_far,carrier_caption_nat,nat_eval_*}/` | supply **GO** d′ 27.3 · scaffold **GO** 0.980±0.012 · **in-model NO-GO 0.145-0.289 (below frozen 0.31-0.56), pure-extremes anatomy** — failure localized to the in-model rung (park e_c / LoRA integration) |
| **TRUNC E1 exactness** (125554/55) | `image_longN/trunc_kvdrop/e1{a,b}/` | **decode READS frames: kvdrop identical 1/16, answers collapse**; fast cached decode ≡ mask 18/20, **16×(mix)–100×(N=64) decode speedup**; baseline re-score 1.000/16 | 
| **TRUNC E2 in-dist trunc@12** (125562) | `image_longN/trunc_at12/indist150/20260725_030649_*/` | **0.047** vs ref 1.000 (pf 0, MAE 4.73; all-yes lowercase degeneration) — FAIL band, E4 retrain triggered |
| **TRUNC E2 N=64 trunc@12** (125564) | `image_longN/trunc_at12/N64/20260725_030642_*/` | **0.019** vs ref 0.981 (pf 0, MAE 15.73) — FAIL band, consistent with E1 mechanism |
| **TRUNC E2 verdict (3 cells)** (125562/63/64) | `image_longN/trunc_at12/{indist150,N32,N64}/` | **eval-only truncation FAIL: 0.047/0.040/0.019 vs 1.000/0.987/0.981** (pf 0 everywhere) → E4 retrain is the fix under test |
| **TRUNC E4 truncated retrain** (125570) | `image_longN/trunc_retrain/carrier_caption_trunc12/20260725_032236_L12_r8/` | TF-count **1.000 @ep5** in 1h34 (14× faster epochs, 21× smaller cache); split gate passed; exams 125604-06 running |
| **TRUNC E3 L14 @N=32** (125565) | `image_longN/trunc_sweep/L14/…/` | 0.033 (pf 0) — flat at the kvdrop floor; eval-only sweep = decode-confound reading, see draft |
| **TRUNC E4-caption exam in-dist** (125604) | `image_longN/trunc_retrain/exam_indist150/…/` | **0.133** (band ≥0.99 FAIL; all-or-nothing transcripts; TF-count 1.000 vs greedy collapse) → E4b scan retry 125609 |
| **P4.2 carrier+digit in-length** (125498 + 125610/11) | `image_longN/carrier_digit_inlength/20260724_202048_L12_r8/` + `p42digit_eval_N{32,64}/` | in-dist 0.863; **N=32 0.333 · N=64 0.140, dead ≥g3-g4 → theory-confirmed band** (caption same-carriers ref 0.987/0.981); split rebuilt (gold>9 prep-skip drift) |
| **P1.1 caption seeds ×3** (125347/48 + exams 125519/507) | `carrier_fmt_caption_seed{1,2}/` + `fmtCseed{1,2}_eval_N32heldout/` | **N=32 held-out 0.982 ± 0.007** (0.987/0.987/0.973, pf 0 ×3, identical dirs; poslist ref 0.953) — seed-robust |
| **TRUNC E3 L16 @N=32** (125571) | `image_longN/trunc_sweep/L16/…/` | 0.107 (curve: L12 0.040 · L14 0.033 · L16 0.107; L20/24 pending) |
| **TRUNC E4b scan retrain + exam** (125609/125613) | `image_longN/trunc_retrain/{carrier_scan_trunc12,exam_scan_indist150}/` | tf-exact plateau **0.165 = caption's**; in-dist exam **0.093** — greedy failure format-independent; in-model NO-GO |
| **TRUNC E4-caption exam row** (125604/05/06) | `image_longN/trunc_retrain/exam_{indist150,N32,N64}/` | **0.133 / 0.073 / 0.096** vs 1.000/0.987/0.981 — deploy-matched retrain NO-GO (TF-count 1.000 though; see hybrid-probe supply-timing finding) |
| **P4.1 SFT in-length** (125567 trainer h200 + 125620 exams) | `image_longN/sft_inlength_p41/20260725_031153_lora/` + `sft_inlength_p41_exams/` | **N=32 0.967 (pf 0) → "simple-fix-wins" band, logged honestly; N=64 extrap 0.787**; retires the ≤8-trained SFT ladder as strongest baseline; surviving contrasts: P4.2 0.333 asymmetry, 5-task generality, h200-only training cost |
| **TRUNC E3 sweep complete** (125563/565/571/572/573) | `image_longN/trunc_sweep/L{14,16,20,24}/` + `trunc_at12/N32/` | eval-only curve **0.040/0.033/0.107/0.073/0.073** @L12-24 — flat at kvdrop floor (decode confound); probe curve supersedes |
| **TRUNC saturation-depth probe** (125615/125621 + CPU) | `image_longN/trunc_retrain/hybrid_dump_N32_notrunc/…/` | gate err **0.34@L12 → 0.0082@L20 → 0.0051@L24**; external tally **0.909±0.016 @L24, N=32**; truncated@12 stays ~0.35 → consolidation lives in L12-19 own-frame edges; **best L_trunc = 20** → E4c (l_open=20) 125628 |
| **TRUNC E6 run-A benchmark** (125616-19) | `image_longN/trunc_bench/runA_N*/` | decode **1.9/32.3/95.4/311.4×** @N=8/32/64/128 (base 3547s → fast 11.4s at N=128); fast≡mask 12/12 |
