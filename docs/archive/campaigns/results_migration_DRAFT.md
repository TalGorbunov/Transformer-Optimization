# RESULTS.md migration draft — compiled 2026-07-23 (p0p2 campaign P0.3). Entries formatted for direct paste into RESULTS.md experiment log. NEWEST FIRST. Tal logs these — this file is staging only.

> Compiled from `plans/carrier_stage2_DRAFT_RESULTS.md`, `plans/oneforward_DRAFT_RESULTS.md`,
> `outputs/ladder/INDEX.md`, `plans/carrier_stage{3,4}_STATE.md`. Everything after RESULTS.md's
> newest entry ([2026-07-18] STAGE-2 CARRIER LAYER), plus two 2026-07-15 oneforward negatives
> that never made it in. Every number traces to the cited run dir / job id.
>
> **Still pending — the campaign will append these as they land:** format-sweep winner (B/C
> N=48/64 cells, jobs 125196–99), L12 extra seeds, LOTO (P2a), MLVU leg, and the N=128 full-34
> reports (l12v2 job 124907, E-G job 124924 — both currently PARTIAL, recovered from dumps).

---

## [2026-07-25] ✅📊 P4 — READOUT-SIMPLICITY CONTROLS (3 pre-registered cells): the ≤8-trained SFT baseline was a DATA artifact — plain LoRA SFT trained in-length reads N=32 at 0.967 ("simple-fix-wins" band, logged honestly); but carriers+digit with the same data collapse (0.333/0.140, theory-confirmed), and the SFT path needed an h200 to train at all

> Prereg `plans/p4_PREREG.md` (bands fixed pre-trainer; P4.1 amended to trained-≤32 after 4
> documented OOM/skip-cascade attempts). Runs: P4.1 trainer 125567 (h200, 0 skips) →
> `sft_inlength_p41/20260725_031153_lora/` + exams 125620 (`sft_inlength_p41_exams/`); P4.2 trainer
> 125498 → `carrier_digit_inlength/20260724_202048_L12_r8/` + exams 125610/11
> (`p42digit_eval_N{32,64}/`); P4.3 = 125499 (`noharm_bench_sft/20260724_190307/`).

| cell | N=32 | N=64 | verdict vs prereg band |
|---|---|---|---|
| P4.1 plain SFT, in-length (≤32) | **0.967** (pf 0, per-count uniform) | 0.787 (extrapolation — N=64 training fits no available GPU) | **≥0.90 → simple-fix-wins, logged honestly** |
| P4.2 carriers + digit, in-length | **0.333** (dead ≥g4) | **0.140** (dead ≥g3) | **≤0.50 + dead mid-range → theory-confirmed** |
| caption winner (ref) | 0.987 (seeds 0.982±0.007) | 0.981 (in-length) | — |
| P4.3 SFT no-harm | MME −0.6 / POPE +1.2 pts | — | **GO** (≤2 pts; digit-on-yes/no failure refuted) |

- **Reframing (per the prereg's own terms):** the old "SFT ladder 0.480/0.350/0.220" strongest-baseline
  row is RETIRED — it measured missing in-length data, not a readout limit. What survives as the
  thesis contribution: (1) the P4.2 asymmetry — a 2M-param carrier LoRA with a single-token readout
  cannot use the same data (readout expressivity is the separator); (2) one caption model serves 5
  tasks + partial LOTO transfer vs single-task SFT (script-limited by design); (3) measured training
  cost — SFT@N=32 trains only on a 140GB h200 (5 attempts documented: masked-forward→MATH 45.6GiB,
  ckpt-recompute OOM, skip-cascade silent data loss), while the cached carrier trainer does ≤64 on a
  40GB a100; (4) N=64: caption 0.981 in-length vs SFT 0.787 structurally-extrapolated.
- Split-drift discipline (new standing lesson): BOTH P4 trainers redraw splits (gold>9 prep-skip /
  declare_splits) — every exam dirs-file was contamination-checked and rebuilt from provably-unseen
  dirs where needed (`eval_dirs_p42_N{32,64}.txt`, `eval_dirs_p41_N32.txt`); the arm-A N=64 file is
  clean for P4.1 (nothing trained at 64).

## [2026-07-24] ✅📊 P3a — NATURAL-IMAGES MMRED FULL LADDER: supply GO (one-forward d′ 27.3 = 3.5× joint) · scaffold GO (linear gate→tally 0.980 ± 0.012) · IN-MODEL NO-GO (0.145–0.289, below the frozen floor, pure-extremes anatomy) — the cross-domain failure is localized to the in-model readout rung, not the mechanism

> Prereg `plans/p3a_natural_PREREG.md`. Data: `data/mmred_natural_mm` composed from the judge-gated
> `mmred_natural_v2` pools (global image-half split — train/eval image-disjoint; builder
> `natural_compose_mmred.py`, BUILD_INFO in the root). Runs: L0 125488 (`natural_mm/frozen_baseline/`),
> L1 125486 (`natural_mm/replica_supply_dist_far/20260724_180046/`), L2 CPU (`…/gate_tally/`),
> L3 trainer 125487 (`natural_mm/carrier_caption_nat/20260724_180058_L12_r8/`) + exams 125492–95
> (`natural_mm/nat_eval_N{8,16}{far,near}/`).

| rung | natural | park ref | band | verdict |
|---|---|---|---|---|
| L0 frozen | 0.563/0.407 @N=8 · 0.422/0.311 @N=16 (g≤8, digit protocol) | 0.219 | ref | — |
| L1 supply | d′_w 27.3 (joint anchor 7.75; per-copy flat 8.5–13.7; AUC-cap caveat) | 13.5 | ≥4.0 & ≥2× joint | **GO** |
| L2 scaffold | **0.980 ± 0.012** (gate err 0.0025, majority 0.187, n=300) | 0.998 | ≥0.85 | **GO** |
| L3 in-model | 0.289/0.259 @N=8 · 0.155/0.145 @N=16 (pooled 0.218, pf 0) | 0.987–1.000 | GO ≥0.80 / NO-GO ≤L0+0.10 | **NO-GO** |

- L3 anatomy is PURE EXTREMES (g0 perfect, gmax mostly right, every intermediate 0) — the bimodal
  signature; the trainer's TF-count 0.996 rode the gold tally prefix (tf-exact 0.187 was the tell).
- Reading: natural per-frame evidence is EASY (high frozen floor, huge d′) and the GNN mechanism is
  fully intact off-domain — what fails is the in-model rung (park-distilled carrier e_c and/or
  LoRA-through-frozen-layers), consistent with the ~51% cross-domain carrier and the MLVU 0.107 cell.
  Successor experiment (not launched): natural-distilled e_c to split e_c from the integration.

## [2026-07-24] ✅📊 P4.3 — NO-HARM ON THE PLAIN SFT ADAPTER: MME −0.6 / POPE +1.2 pts (band |Δ|≤2 → GO); predicted digit-on-yes/no failure REFUTED — both adapter families are safe always-on

> Job 125499 → `outputs/ladder/image_longN/noharm_bench_sft/20260724_190307/`; same 500+500
> MME/POPE protocol/seed as the carrier cell (124508, ref −0.2/−1.4); new `--peft-adapter` arm +
> ≤20 failure dumps in `logs/p43_noharm-125499.out`.

- MME 0.862→0.856, POPE 0.862→0.874. Small-n category texture: OCR −20 / celebrity −13.3 vs
  count +16.7 / commonsense +12.5. Fail dumps emit clean yes/no words — no digit contamination.

## [2026-07-24] ✅📊 P3b — InternVL2.5-8B SCAFFOLD-LEVEL GATE→TALLY: 0.938 ± 0.031 exact @N=8 (L16; per-frame gate 0.991, majority 0.160) — the GNN scaffold (per-frame messages + linear gate + sum) PORTS across model families, at the multipass-isolated supply level

> CPU fit (`experiments/glstm/internvl_gate_tally.py`) on the existing cache of job 124280
> (`outputs/frame_axis/internvl/multipass_qfirst/20260719_004112/bench_cache.pt`) →
> `outputs/frame_axis/internvl/gate_tally/20260724_165356/`; logistic gate, sample-disjoint 60/40,
> seeds 0–2, tally = Σ verdicts. L20: 0.892 ± 0.012.

- Band ≥0.90 MET at L16 → scaffold ports. **Honest label: multipass-isolated Q-first supply** (each
  frame solo in its own forward), one rung below Qwen's one-forward blockfence cell.
- Cross-family replication of the readout-misalignment signature: InternVL's own per-frame digit
  readout is 0.586, while the linear gate on its carrier messages reads 0.991.

## [2026-07-24] ✅📊 P2b — MLVU-AC ZERO-SHOT CARRIER CELL (32-frame arm): MCQ nearest-option 0.107 ≤ frozen 0.282 — the pre-registered "domain gap measured" outcome; the readout FORMAT transfers, the trained evidence detector does not fire outside the MMRED render domain

> Job 125350 → `outputs/ladder/mlvu_ac/carrier_eval_N32/20260724_064754_L12_r8_evalonly/` (+
> `mcq_mapping.txt`); winner ckpt (C caption) on all 206 MLVU-AC questions, 32f @392px. Dense N=128
> ruled prohibitive pre-launch (~20 min/sample × 206 > 60h) → prereg 32f fallback with the
> evidence-delivery caveat ([2026-07-11c]: ~0.37 visible frames per gold instance at N=32).

- **Open emitted count 0.000 exact** (pf 0.214, MAE 2.99); 161/206 emit "0", 44 parse-fail, 1×"1".
  **MCQ nearest-option (prereg rule, parse-fail = wrong): 22/206 = 0.107** (g1 12/37 · g2 8/52 ·
  g3 2/45 · g4 0/33 · g5 0/39). Band: ≤ frozen 0.282 → **transfer NO, domain gap measured**.
  Protocol note (by design): this cell never sees the MCQ options; the frozen 0.282 had them
  in-prompt — chance structure differs.
- Failure anatomy (206 dumped transcripts): scan/caption structure emitted and mostly well-formed;
  content collapses to all-negative verdicts, and the parse-fails are fluent refusals that correctly
  DESCRIBE the frames ("the image shows a zebra … no 'making jewelry' action"). Perception works;
  the trained evidence-detection channel is domain-bound — consistent with the cross-domain carrier
  result (~51% of teacher, [2026-07-18]) and the delivered-evidence ceiling.

## [2026-07-23] ✅📊 P1.3 — E-B SFT BASELINE N=64 CELL (the OOM'd leg, rerun): 0.220 — the bimodal extremes heuristic extends to 8× training length; the SFT ladder is complete at N=16 0.480 / N=32 0.350 / N=64 0.220 vs the carrier readout's 0.953/0.878/0.678

> Job 125267 → `outputs/ladder/image_longN/sft_control_le8_v2_evalN64/20260723_225940_lora/`;
> eval-only restore of `sft_control_le8_v2/20260720_191541_lora/adapter`, LIMIT 100, same
> stratified-prefix sampling as the N=16/32 cells. Adapter-restore sanity: test_iid 1.0000 = the
> original run byte-exact.

- **N=64: 0.220, parse-fail 0.000, MAE 3.46**; per-count g0 4/7 · g1 5/7 · g64 6/6, mid-range
  g4–g48 ≈ 0 — no collapse, no aggregation, the dead-mid-range supply ceiling unchanged at 8×.
- Ops lesson (new): the 124696 OOM = mask-None generate → `enable_gqa=True` → mem-efficient sdpa
  ineligible (num_heads mismatch on dense inputs) → MATH materializes 17GB. **FLASH handles
  GQA+causal at 8.3 GiB peak @seq 12.7k** (smoke 125263). `lora_sft_baseline.py` eval path now
  [FLASH, EFFICIENT, MATH] + `--eval-only-adapter`.

## [2026-07-23] ✅📊 P1.2 — MEASURED BEFORE-CEILING: the best sample-disjoint linear readout of the summed joint-carrier messages lands within 0.01–0.05 of the zero-parameter law prediction at every N — the "squashed readout" curve is now measured, not just predicted

> Job 125259 (CPU, `probe_dprime_parity.py --carrier-caches`, deployed locus L16/off9, existing joint
> caches `image_longN/joint/N{8,16,32,64,128}/20260710_2154*/count/`, 60/40 sample-disjoint split,
> seeds 0–2) → `outputs/ladder/image_longN/measured_ceiling/20260723_222428/`.

| N | law-pred (iid) | MEASURED best linear | frozen model (same caches) |
|---|---|---|---|
| 8 | 0.307 | **0.317** (ridge, ±0.036) | 0.207 |
| 16 | 0.246 | **0.281** (logit) | 0.127 |
| 32 | 0.175 | **0.189** (logit) | 0.053 |
| 64 | 0.137 | **0.183** (logit) | 0.040 |
| 128 | 0.096 | **0.122** (logit) | 0.013 |

- Slightly ABOVE the law at N≥64 — the logistic readout exploits the non-Gaussian tail the law ignores
  (adequacy kurtosis +1.8→+14.0, as in [2026-07-11e/n]); MLP−linear ≤0.006 (E3 sufficiency); d′_w flat
  ~2.0 to N=64 (1.6 @128), replicating B1. Thesis reading: frozen < measured-linear < trained scaffold
  (0.95–1.00 @N=32) — readout misalignment vs supply repair, both gaps now measured.
- Fig: `outputs/_scratch/figs/pre_stage1_squashed_readout_measured.png`. Caveat: gold≥1 convention at
  N≥2; n = 300/300/300/200/150 → wider bars at N=128 (±0.031).

## [2026-07-24] ✅📊 FORMAT SWEEP COMPLETE (arms A–D) — the gold scratchpad TEXT alone is worth +0.19–0.37 at held-out lengths: full-scan formats crush the positive-list (B scan N=32 **1.000**, N=48 **0.982**; C caption N=64 **0.981**, pf 0.000 everywhere); WINNER = C (caption); chunking NO

> Prereg `plans/scratchpad_format_PREREG.md` (bands fixed pre-GPU); arms differ ONLY in gold
> scratchpad text (`--scratchpad-format`), l12v2 recipe/data/split verbatim — eval/train dirs-files
> byte-identical across arms. Trainers 125104/05/06 →
> `outputs/ladder/image_longN/carrier_fmt_{scan,caption,chunked}/20260722_2220*_L12_r8/`; exams
> 125107/08 + 125183–99 on arm A's dirs-files (`fmt{B,C,D}_eval_*` + `tallyL12v2_eval_*` siblings).

| cell (identical dirs) | A poslist | B scan | C caption (WINNER) | D chunked |
|---|---|---|---|---|
| TF-fit (acc / tf-exact) | 1.000 / 0.976 | 0.999 / 0.996 | 0.999 / 0.994 | 1.000 / 0.904 |
| in-dist-150 | 1.000 | 1.000 | 1.000 | 0.987 |
| rooms-100 | 1.000 | 1.000 | 1.000 | 0.920 |
| N=32 held-out (150) | 0.953 | **1.000** (125194) | 0.987 (125195) | 0.907 (125185) |
| N=48 held-out (109) | 0.789 / 0.878 cap-adj | **0.982** (125196) | 0.972 (125197) | 0.679 (125186) |
| N=64 held-out (52) | 0.615 / 0.711 cap-adj | 0.942 / 0.956 cap-adj (125198) | **0.981** (125199) | 0.615 (125187) |
| worst parse-fail | 0.135 (N=64) | 0.019 | **0.000** | 0.000 |

- **All five prereg bands decided:** in-dist sanity ≥0.90 MET ×4 · **scan GO** (B N=64 cap-adj 0.956
  ≥ 0.761) · **agnostic-caption GO** (C≡B: in-dist/rooms exact parity, N=32 −0.013, N=64 +0.039 in
  C's favor) · **chunking NO** (D 0.615 < max(A,B)) · rooms-ordering parity-only (control A also
  1.000 — the 0.84 gap was the L17 5-task ckpt, not reproducible here).
- **Winner = C (caption)**, ckpt `carrier_fmt_caption/20260722_222032_L12_r8/carrier_layer_best.pt`:
  takes the hardest cell (0.981 vs B 0.942) with pf 0 at every length; B within noise overall
  (length-cell mean 0.975 vs 0.980) — C chosen on the primary N=64 cell + parse robustness + the
  agnosticism property (attribute words in the scratchpad). Mechanism reading: a slot per frame
  converts the long-N search burden into a deterministic frame-order scan; D's subtotals kill
  truncation (dec-means 55/75/101, pf 0) but cost accuracy at every length.
- Cost note: scan-family decode ≈ 2.4× poslist in-dist (~47 vs ~20 tok); N=48/64 exams ≈ 18h each
  on l40s (~21 min/sample @dec620) — a100 ≈ 2× faster.

## [2026-07-23] ✅📊 E-H SEPARATOR-LAYER (L*) CURVE COMPLETE — inverted-U confirmed, peak at L_OPEN=12: zero-shot N=32 0.277/0.373/**0.443**/0.330/0.280/0.273 for L*=8/10/12/14/17/20 — ~12 fenced supply layers suffice, every remaining layer is wanted for trained integration; L12 STANDS as the default

> Four new arms (ONLY `--l-open` varies), headline ≤16 running-tally recipe, fixed (acc, tf-exact)
> save criterion. Trainers: L8 124917 / L10 124918 / L14 124919 / L20 124920; zero-shot exams
> jobs 124965–972 → `outputs/ladder/image_longN/tallyL{8,10,14,20}_eval_N{32,64}/`. Reference
> cells: L12 (124698/124727), L17 (124482/124522) — recipe-matched, but predate the fixed save
> criterion (caveat logged).

| L* (open layer) | in-dist TF / tf-exact | N=32 zero-shot (n=300) | pf | N=64 zero-shot (n=60) | pf |
|---|---|---|---|---|---|
| 8 | 1.000 / 0.978 | 0.277 | 0.017 | 0.133 | 0.183 |
| 10 | 0.999 / 0.927 | 0.373 | 0.000 | 0.217 | 0.150 |
| **12** | 1.000 / 0.991 | **0.443** | 0.000 | (not measured, le16 recipe) | — |
| 14 | 1.000 / 0.997 | 0.330 | 0.000 | 0.217 | 0.150 |
| 17 (ref) | 1.000 / 0.977 | 0.280 | 0.007 | 0.150 | 0.133 |
| 20 | 1.000 / 0.996 | 0.273 | 0.107 | 0.183 | 0.300 |

- In-dist TF saturates for ALL L* (pre-registered: not the verdict metric). L20's parse-fail is
  the worst of any arm and concentrates at high counts (N=32: g24 7/23, g32 17/23 fails) — late
  opening leaves too few integration layers to keep the format coherent at long N.
- **Decision rule (LOTO/p0p2 briefs): no arm beats L12's 0.443 by >0.05 with pf ≤0.02 →
  L_OPEN=12 stands** for the seed retrains and LOTO. L12's N=64 cell and 2 extra seeds ON HOLD
  per Tal's no-new-launches instruction.

## [2026-07-22] ✅📊 l12v2 — L12 + in-length data + FIXED ckpt criterion = the campaign-best in-model long-N readout: held-out **N=32 0.953 · N=48 0.878 cap-adj · N=64 0.678 cap-adj**; N=128 PARTIAL 0.286 with format fully intact (best 128 reading so far)

> Two measured levers composed: (1) **L12 depth** — le16-recipe L12 arm (124698,
> `carrier_tally_le16_L12/20260720_192738_L12_r8/`) reads N=32 zero-shot **0.443** (pf 0, MAE 1.15)
> vs L17's 0.280 (job 124727, `tallyL12_eval_N32/20260720_223517_L12_r8_evalonly/`; band ≥0.40 GO →
> L12 = new default); (2) **in-length data** (le64v2 roots + `mmred_longN_park2` N=32×312 + N=48×210).
> Trainer 124773 → `carrier_tally_l12v2/20260721_071710_L12_r8/`: TF 1.000 / tf-exact 0.976 @ep5,
> the strongest ckpt of the campaign. Exams 124904/05/06 → `tallyL12v2_eval_N{32,48,64}heldout/`.

- **N=32 held-out (150 dirs): 0.953, pf 0.000, MAE 0.05** — band ≥0.85 MET, thesis-grade; per-count
  near-uniform incl. multi-digit (g12 4/4 · g16 8/9 · g24 10/12 · g32 9/10). Trained-at-length
  progression: A4 scratchpad 0.447 → le64 tally 0.733 → **l12v2 0.953**.
- **N=48 (109): 0.789 raw / 0.878 cap-adjusted** (all pf = g48 cap truncations). **N=64 (52):
  0.615 raw / 0.678 cap-adjusted** (excl. g48/64, unmeasurable under dec 280) — met vs the v2 band
  ≥0.65 cap-adjusted, just under the stricter l12v2 prereg ≥0.70.
- **N=128 (2× beyond max trained): PARTIAL 0.286 (4/14), parse-fail 0.000** — job preempted by
  h200-dds mid-run, recovered from dumps (stratified-order prefix, unbiased); best N=128 of the
  campaign (le16 0.087 · v2 0.118) and the FIRST with intact format at 128; misses are undercounts
  at g≥5, not derails. Full-34 report = requeued job 124907.
- Caveat / confound exhibit: the earlier v2 arm (124682, old acc-only save criterion, ckpt ep3
  tf 0.891) read 0.607/0.514/0.365/0.118 at N=32/48/64/128 (124755/56/57/58) — kept as the
  ckpt-selection confound exhibit; trainer save criterion is now (TF-count, tf-exact) lexicographic.

## [2026-07-22] ❌📊 E-G POSITION-COUPLED TALLY REFUTED — coupling decoded-token positions to carrier anchors costs in-distribution fit (persistent over 8 ep) and BREAKS the format at 4× length: N=32 0.527 / N=64 0.212 (pf 0.365) vs uncoupled l12v2 0.953 / 0.615 on IDENTICAL dirs

> `couple_offsets` in `carrier_layer_lora.py` + `--pos-couple` (one rule drives teacher-forced AND
> online decode positions; CPU-verified anchor rule + couple-debug — this is a MECHANISM negative,
> not an implementation bug). First train 124701 (`carrier_tally_pcouple/`) TF-undertrained at 4 ep
> (0.923/0.720, still climbing); converged rerun 124774 → `carrier_tally_pcouple8/20260721_071710_L17_r8/`,
> in-dist ceiling **0.955 / tf-exact 0.832 @ep5** (oscillatory) vs uncoupled 1.000/0.976 — ~2× the
> epochs for the same TF level. Exams 124922/23/24 → `tallyPC8_eval_N{32,64,128}heldout/`.

| cell (identical dirs) | E-G (coupled) | l12v2 (uncoupled best) | v2 (uncoupled same-L17) |
|---|---|---|---|
| in-dist TF / tf-exact | 0.955 / 0.832 | 1.000 / 0.976 | 1.000 / 0.916 |
| N=32 held-out | 0.527 (pf 0.040) | **0.953 (pf 0.000)** | 0.607 |
| N=64 held-out | 0.212 (pf 0.365, MAE 44) | **0.615 (pf 0.135)** | 0.365 |
| N=128 | PARTIAL 0.154 (4/26, pf 0.500) | PARTIAL 0.286 (pf 0.000) | 0.118 |

- **Pre-registered GO ("beats uncoupled at EVERY OOD length, pf ~0") refuted at every testable
  cell**; N=64 transcripts show token-salad degeneration ("28 (10 (10), 32 (11)…") — the opposite
  of the design goal (it was built against the tally-index confusion seen at 8×). Forcing decoded
  positions to ride carrier anchors degrades the LM's own sequential coherence more than it helps
  length binding. N=128 full cell (requeued 124924) cannot change the verdict.

## [2026-07-20→21] ⚠️📊 E-A LENGTH LADDER + THE N=128 WALL — running-tally readout decays smoothly zero-shot (1× 1.000 → 2× 0.280 → 4× 0.150 → **8× 0.087: headline band ≥0.80 REFUTED**); in-length training is the working lever (le64: N=32 **0.733**, N=64 0.558/0.821-in-range) — the honest claim is a LADDER in trained-length coverage, not extrapolation

> Arms: le16 (124482, `carrier_tally_le16/20260719_184950_L17_r8/`, 11 roots + longN_16,
> running-tally + jitter 16, TF-count 1.000 @ep2) · le64 (124483, `carrier_tally_le64/…/`,
> +longN_32/64, TF 1.000 @ep4). Exams: 124522/578/736 (le16), 124571/586 (le64), 124758 (v2→128).

| ckpt \ exam | N=32 | N=64 | N=128 |
|---|---|---|---|
| le16 (trained ≤16), zero-shot | 0.280 (pf 0.007; in-range 0.394) | 0.150 (pf 0.133) | **0.087** (first-match 0.130, pf 0.087) |
| le64 (trained ≤64), held-out | **0.733** (pf 0, MAE 0.43, g32 9/9) | 0.558 (in-range 0.821) | 0.118 (v2 ckpt, 124758) |

- **The headline cell** (`tally16_eval_N128/`, job 124736, TIMEOUT @23/34 — recovered from the
  --dump-decodes safety net, per-sample gold/parsed in `logs/cl_eval-124736.out`): train-≤16 →
  emit-at-128 (8×) does NOT hold. Transcripts show tally-index confusion (verdict indices leak
  into tally slots: "frames 30 (11), 31 (112)…") and unterminated chains. Trained-to-≤64 → 128
  (2×) also fails the ≥0.35 band (0.118; ckpt-selection confound caveat, gap to band large).
- le64's N=64 g48/g64 cells are cap-truncations (280-token decode < the 48–64-verdict tally) —
  annotated as unmeasurable-under-cap, not model failures. le16 N=64 failure mode = repetition
  loops. Parse sensitivity: first-vs-last-match differs ≤1 sample/cell (last-match stays primary).
- Zero-shot N=32 stable across seeds: **0.284 ± 0.004** (124522 headline jitter-16 0.280;
  124697 seed1 jitter-12 0.287). Successor arms (l12v2 0.953 in-length / 0.286@128, E-G refuted)
  logged above.

## [2026-07-20] ✅📊 POSRESET NECESSITY (Tal's challenge) — minor at N=8 (no-reset d′ 7.74 vs 9.24) but the per-copy position-tax gradient RETURNS at N=64 (pooled 7.54, per-copy decays 6.4→3.0): KEEP posreset, re-justified as N-scaling-critical; the historical "+0.59" justification retired

- Runs: `outputs/ladder/image_longN/noreset_N{8,64}/20260720_*/` (jobs 124713/124714) — Q-first
  blockfence WITHOUT `--reset-positions`; comparators with-reset 9.24 (N=8) and ~12 @N=64
  (bracketed 12.67@32 / 11.57@128).
- **N=8: 7.74 ± 0.18** — minor cost, and > fence-level 6.34 (Q-first partially substitutes for
  reset). **N=64: pooled 7.54 is carried by early frames — per-copy decays 6.4 → ~3.0** with frame
  index (the A2 position-tax fingerprint); extrapolated, late-frame supply at N=128 approaches
  joint level.
- Verdict (pre-registered intermediate band): keep — free at inference, increasingly load-bearing
  with N, and required infrastructure for jitter + position-coupling experiments.

## [2026-07-20] ✅📊 E-B SFT CONTROL — a 23.8M-param plain LoRA (12× our budget, q/k/v/o+MLP all layers, trained N≤8) matches in-distribution (0.998–1.000) but does NOT aggregate at length: N=16 **0.480** / N=32 **0.350** with a DEAD MID-RANGE — the theory's joint-supply ceiling located behaviorally

- First run 124484 (`sft_control_le8/20260719_185022_lora/`): test_iid **0.9984** at N≤8 (best ep1)
  — but the long-N cell was BLOCKED (script never saved the adapter; `plans/carrier_stage4_BLOCKED.md`).
  Rerun with adapter-saving patch: job 124696 → `sft_control_le8_v2/20260720_191541_lora/`
  (best ep3 val 0.983, test_iid 1.000).
- **N=16 0.480 / N=32 0.350 (pf 0.000 both)** — more nuanced than the pre-registered "collapse":
  the LoRA rides EXTREME-count anchors (g1 8/8, g32 7/7 but g6–g8 0/27 at N=16; mid-range ~0 at
  N=32; MAE 1.83) — a bimodal low-end/saturation heuristic. The dead mid-range is exactly where
  per-frame aggregation is needed (joint supply d′≈2); extremes are solvable from global gist.
- Honest packaging: the carrier method's long-N edge is the in-length-trained cell (0.733 vs
  0.350 @N=32) and uniform per-count coverage; the zero-shot tally cell (0.280) does NOT beat the
  SFT control. N=64 cell OOM'd during generate (adapter saved; low priority).

## [2026-07-20] 📊 ROOMS DECODE-GAP DIAGNOSTIC — every error in 40/40 held-out transcripts is a MISSING-ROOM verdict; emitted count ALWAYS equals emitted list length: the readout's counting is exact, the residual is per-frame DETECTION RECALL (supply-side), not the readout

- Run: `outputs/ladder/image_longN/rooms_gap_diag/…_evalonly/` (job 124527; 40 held-out rooms
  samples, 5-task L17 ckpt, full transcripts): acc 0.825, parse-fail 0.000, MAE 0.17. No format
  derail, no reordering, no count-list mismatch (e.g. gold 6 → "Bathroom, Bedroom, Garden,
  Office, Park -> 5").
- Same signature as the long-N misses → the future lever is carrier content, not the readout.
- Follow-up caveat (2026-07-22): the l12v2 ckpt reads rooms-100 at **1.000** (job 125108) — the
  0.84–0.85 gap was a property of the L17 5-task ckpt family, not of the method.

## [2026-07-20] ✅ E-E SEEDS — headline running-tally recipe at 3 seeds: in-dist TF-count **1.000 ± 0.000** (tf-exact 0.963 ± 0.007); the zero-shot N=32 length cell 0.284 ± 0.004 over 2 seeds — the recipe is seed-stable

- Runs: headline 124482 (seed/shuffle 0, jitter 16, 1.000 @ep2) ·
  `carrier_tally_le16_seed1/20260719_203924_L17_r8/` (124509, jitter 12, 1.000 @ep3) ·
  `carrier_tally_le16_seed2/20260719_203925_L17_r8/` (124510, jitter 12, 1.000 @ep3).
- Caveat: seed arms ran jitter 12 vs headline 16 (seed+jitter-dose bundled) — indistinguishable
  both in-dist and at the N=32 cell (0.287 vs 0.280, job 124697), so jitter dose 12-vs-16 is a
  no-op here.

## [2026-07-19] ✅📊 E-C/E-C(b) LAYOUT FREEDOM (qualified GO) — carrier tokens can sit as a PURE SUFFIX after the question (d′ 10.27, tally **0.999** = the interleaved stack), but the strong form fails: with no leading question, at-end carriers read d′ 2.40 / tally 0.508 — the binding requirement is QUESTION-FIRST, not carrier placement

> E-C strong form (job 124492, `carrier_atend/20260719_192758_distill_room_k1/`, n=900,
> frames-first teacher anchor 8.89 ± 0.14): student d′ **2.40** (27% of teacher), tally 0.508 —
> band ≥5 missed; the at-end carrier reproduces messages in bulk (MSE converged) but not the
> discriminative direction, despite an IDENTICAL allowed-key set (mask-debug: 223 keys, same).
> E-C(b) restores ONE thing — the leading question (`--atend-qfirst`, job 124514,
> `carrier_atend_qfirst/20260719_205916_distill_room_k1/`, Q-first teacher anchor 13.70).

| layout (fence+posreset, distill, N=8 full prior) | eval d′ | fresh-logistic tally |
|---|---|---|
| Q-first, carriers INTERLEAVED (123233, reference) | 11.45 (96% of teacher) | 0.999 ± 0.001 |
| **Q-first, carriers AT END (124514)** | **10.27 @ep2 (75%)** | **0.999 ± 0.001** |
| no leading question, carriers at end (124492) | 2.40 (27%) | 0.508 ± 0.017 |

- Both E-C(b) bands met (d′ ≥5 ✓, tally = interleaved ✓): the E-C failure was missing
  question-conditioned frame ENCODING (the same +3 d′ Q-first term as the C2 ablation), adjacency
  innocent. Method statement: prompt = [question][frames][question][carrier suffix] — carriers
  never interrupt user content (the deployment-friendly form).

## [2026-07-19] ✅ E-D NO-HARM — the carrier-layer LoRA left permanently ON costs nothing on general benchmarks: MME −0.2 pts, POPE −1.4 pts (band ≤2 = GO); with the drift row (plain-MMRED 0.313 vs frozen 0.219) the adapter is deployment-safe always-on

- Run: `outputs/ladder/image_longN/noharm_bench/20260719_203833/` (job 124508; 500 MME + 500 POPE
  items, identical samples both arms, Yes/No logit-argmax, le16 running-tally ckpt LoRA).

| benchmark | base | LoRA-on | Δ | band |
|---|---|---|---|---|
| MME (acc) | 0.862 | 0.860 | **−0.2 pts** | GO (≤2) |
| POPE (acc / F1) | 0.862 / 0.839 | 0.848 / 0.819 | **−1.4 pts** | GO (≤2) |

- Per-subtask deltas ~0 across 12/14 MME cells (existence −4.5 / landmark −2.4 are small-cell
  noise n≈20–40, celebrity +2.7); all POPE splits −1.1..−1.7.

## [2026-07-19] ✅📊 SCRATCHPAD READOUT (A3/A4/5-task) — verdict-scratchpad targets fit in ONE epoch (TF-count 1.000), in-dist greedy 0.953–0.966 with parse-fail 0.000; a 5-task mixture (adds NIAH 0.992 + OR-union 0.910) runs on ONE carrier + ONE LoRA; untrained OR-union composes PARTIALLY (0.321 = 2.1× digit); zero-shot length 0.215 → +200 in-length samples 0.447

- **A3 train** (job 124282, `carrier_layer_scratchpad/20260719_005342_L17_r8/`, 5.8k, jitter 12):
  TF-count **1.000 @ep1**; **in-dist greedy 0.953, pf 0.000, MAE 0.05** (124314; steps 0.980 /
  cooc 0.906 / rooms 0.850). **NIAH which-frame zero-shot 0.087** ≈ chance 0.125 (124316).
- **Composition (OR-union, never trained): 0.321** (pf 0.021, MAE 1.34; hits over the FULL count
  range — not mode collapse) vs digit-ckpt 0.150 → the shared VERDICT FORMAT is what transfers
  (NIAH, fully alien, gets 0.087). Run `scratchpad_eval_union0shot/20260719_030349_…/` (124335).
- **5-task mixture** (124336, `carrier_layer_scratchpad5/20260719_031356_L17_r8/`, +NIAH 720
  +union 540): TF-count 1.000 on all 5 @ep1; **in-dist greedy 0.966** (steps 0.997 · which 0.992 ·
  cooc 0.944 · union 0.910 · rooms 0.842) (124349). Bands NIAH ≥0.90 ✓ ("easy once in mixture"),
  union ≥0.85 ✓, no regression ✓.
- **Length:** A3 N=32 zero-shot **0.215** (in-range 0.311, pf 0.000 — format fully survives where
  digit ckpts collapse to "0") (124315) → A4 fallback triggered. **A4** (+longN_16 all 330 +
  longN_32 first-200; 124362, TF 0.997 @ep3) → **N=32 held-out complement 0.447 (in-range 0.626,
  pf 0.000, MAE 1.44)** (124376, `scratchpadLN_eval_N32heldout/20260719_092517_…/`) — band
  partial (<0.80); errors are verdict undercounts, never format collapse; long-N data curve
  still steep. (A3/A4 N=128 rows dropped for budget; the wall was later measured on the tally
  arms — see the N=128 entry.)

## [2026-07-19] ✅ STAGE-2 POOLED GO — data was the whole gap: the in-model carrier layer at 6k pooled samples reads **0.999** (steps 630/630 · rooms 108/108 · cooc 161/162) = the 0.998 scaffold, one architecture, three tasks incl. the provably-nonlinear rooms set-union; frozen-e_c cached trainer hits 0.980 at 4× speed

- **P1 pooled 3-task** (job 123741, `carrier_layer_pooled/20260718_182248_L17_r8/`; steps N=2..8
  4200 + cooc 1080 + rooms 720, train 5100/eval 900 stratified per (task,N), trainable e_c
  warm-started from the full-prior distill, L17 r8, 12 ep): **BEST 0.999 @ep12, MAE 0.00.**
  Trajectory 0.176 → 0.669(ep1) → 0.963(ep5) → 0.997(ep8) → 0.763(ep11, transient optimizer
  blip) → 0.999(ep12).
- **P1-CACHED** (job 123937, `carrier_layer_cached/20260718_192428_L17_r8/`; e_c FROZEN, layers
  ≤L16 run once and cached, steps 2–8 + cooc, ~5.1k, 934 s/ep ≈ 4× faster): **0.980 @ep10**
  (steps 0.987 / cooc 0.946), MAE 0.02 — frozen e_c matches trainable; validated workhorse.
- **Data-starvation diagnosis vindicated: 450→0.678 · 5.1k→0.980 · 6k→0.999** (scaffold 0.998,
  frozen 0.219, chance 0.111). Caveat: all trained-on-clean synthetic MMRED; scaffold ceiling
  comparison is same-data.

## [2026-07-19] ❌📊 DIGIT-READOUT EXAMS — length collapse and zero-shot task transfer are both robust NULLS: N=32 0.092–0.097 (collapse to "0" regardless of training-N diversity); five unseen-task pairs all ≈ chance (0.087–0.179); the LoRA left on plain prompts is SAFE (0.313 vs frozen 0.219)

- **Length:** cached ckpt (0.980 @N≤8) @N=32 → **0.097** (g0 24/24, else ~0)
  (`cached_eval_N32/20260719_000556_…/`, 124275); pooled ckpt (0.999, variable-N 2..8 trained)
  @N=32 → **0.092** (`pooled_eval_N32/20260719_051622_…/`, 124353) — training-N diversity buys
  nothing; the digit readout binds to the trained carrier-position range. Steps-450 ckpt: 0.138
  (`carrier_layer_eval_N32/20260718_182252_…/`, 123742).
- **Task transfer (zero-shot, all ≈ chance 0.111–0.125):** steps→cooc 0.179 (123743) ·
  cached→rooms 0.153 (124276) · A3→NIAH 0.087 (124316) · pooled→NIAH 0.117 (124354) ·
  pooled→union 0.150 (124355). Task coverage must be trained — and the mixture rows show it
  costs nothing (no interference).
- **Drift** (`frozen_baseline_driftlora/20260719_000556/`, 124277): plain prompt + LoRA hooks ON
  = **0.313 vs 0.219** frozen (n-mismatch caveat: 300 vs 900) — slightly HELPS; no gating needed.

## [2026-07-19] ✅📊 TRACK B — InternVL2.5-8B PORT: solo-Q-first carrier d′ **6.31/5.11 @L16/L20 vs joint 1.79/1.90 — the supply mechanism ports (3.5×)**; but vs plain solo 6.38/6.56 the Q-first AMPLIFIER does NOT port — fence/isolation is the portable piece, Q-first is Qwen-specific

- Run: `outputs/frame_axis/internvl/multipass_qfirst/20260719_004112/` (job 124280; n=200, 1600
  solo passes, same seed/data/estimator as the 118996 record — sample-matched; `--qfirst` flag in
  `experiments/internvl/multipass_bench.py`; per-frame perception acc 0.586 unchanged).
- Verdicts vs pre-registered bands: mechanism-ports GO (≥2× joint ✓ at 3.5×); amplifier band
  (≥ +20% over plain solo) FAILED — 6.31 vs 6.38 flat at L16, 5.11 vs 6.56 negative at L20.
  Honest thesis scope note: question-conditioned frame encoding is family-dependent.
- Method note: solo forwards = the fence/multipass-equivalent supply measurement (fence ≡
  multipass identity established on Qwen); no mask surgery in InternVL remote code.

## [2026-07-19] ✅📊 C1/C2 ABLATION BATTERY (7 arms, 900-train starved regime, RANKING is the deliverable) — Q-first is the single most load-bearing piece (−46%); earlier opening ≫ (L12 0.941 vs L17 0.698 vs L22 0.513); posreset mild (−4%); LoRA rank flat

> Cached digit trainer, steps8+cooc (n=1800, train 900), 8 ep, frozen e_c, one change per arm
> (jobs 124300–06) → `outputs/ladder/image_longN/cached_ablations/{base,L12,L22,r4,r16,noqfirst,noposreset}/`.
> ABSOLUTE numbers are data-starved by design — never cross-compare to the 5–6k runs.

| arm | BEST acc | arm | BEST acc |
|---|---|---|---|
| **L_OPEN=12** | **0.941 @ep8** | rank=4 | 0.694 @ep7 |
| rank=16 | 0.731 @ep7 | no-posreset | 0.669 @ep8 |
| base (L17 r8) | 0.698 @ep8 | L_OPEN=22 | 0.513 @ep8 |
| | | **no-Q-first** | **0.378 @ep8** |

- Q-first −46% matches its +3 d′ supply price (and Track B: this piece is Qwen-specific);
  posreset −4% consistent with +0.6 d′. L12 ≫ L17 ≫ L22 — caveat: LoRA params scale with open
  depth (16/11/6 layers), depth and capacity confounded; at full data L17 already reaches
  0.98–0.99, so L12 is the better default for small data (later confirmed OOD — see E-H).

## [2026-07-18] ✅📊 E1 FULL-PRIOR RECALIBRATION (N=8, LIMIT=900, gold uniform 0..8) — frozen baseline **0.219** (retires the truncated-prior 0.513); Q-first blockfence probe d′ 13.54 @n900 with gate→tally **0.998**; carrier-token distill 11.45 = 96% of teacher, carrier-stack tally 0.999 — the scaffold ceiling for all stage-2 comparisons

- **Audit caveat driving this entry:** the old N=8 caches were e-sorted → n=300 meant gold∈{0,1,2}
  only. Fix: stratified `iter_sample_dirs_shuffled` (+`--shuffle-dirs`, gold-hist prints) wired
  into probe/distill/trainer/baseline scripts; smokes 123222/123226 (`outputs/_scratch/st2_smoke*/`).
- **Frozen baseline** (123225, `frozen_baseline/20260718_125303/`): **0.219, MAE 1.86** — the
  undercount clamp in full view (g4+ ≈ 0). Truncated-prior 0.513 (123205) RETIRED (it rewarded
  the low-count bias).
- **Probe** (123232, `replica_blockfence_qfirst_full900/20260718_130546/`): **d′ 13.54 ± 0.27**
  @n900 (joint anchor 5.95; per-copy flat 8.4–9.2); matched-n300 subsample 10.50–10.89 vs the
  truncated band 9.24 ± 0.33 — supply is prior-free; d′ estimator scales with n (caveat for all
  cross-n comparisons). **Gate→tally (123236): exact 0.998 ± 0.001, MAE 0.00** vs majority 0.111.
- **Distill** (123233, `carrier_token/20260718_130545_distill_room_k1/`): in-run teacher anchor
  reproduces 13.54 exactly; **carrier eval d′ 11.45 @ep9 = 96% of the scale-matched teacher
  11.94**; carrier-stack fresh-logistic tally **0.999 ± 0.001** (CPU job 123243).

## [2026-07-18] ⚠️📊 STAGE-2 AT 450 TRAIN SAMPLES — convergence hypothesis REFUTED (30-ep ref flat at 0.840 from ep12), full-prior steps 0.678 with the clamp DEAD, mixtures 0.693 (2-task) / 0.509 (3-task, rooms 0.50): a DATA-STARVATION ceiling, not architecture — sets up the pooled P1; cross-DOMAIN carrier transfer is only ~51% of teacher

- **30-ep truncated-prior ref** (123206, `carrier_layer/20260718_122503_L17_r8/`): BEST 0.840
  @ep12, loss→1e-4 by ep14, eval FLAT to ep30 — "undertrained, still climbing" ([2026-07-18]
  entry) is REFUTED; the gap is data/generalization, not optimization.
- **Full-prior steps-only 40ep** (123235, `carrier_layer/20260718_131157_L17_r8/`, train 450/eval
  450): **0.678 @ep30**, per-count uniform incl. g8 (clamp dead); train loss→0 = memorizes 450.
- **Mixtures** (warm-start distilled e_c, 30 ep): steps+cooc **0.693 @ep25** (cooc 0.796 / steps
  0.560 — no task interference; 123237, `carrier_layer_mixture/20260718_133821_L17_r8/`);
  +rooms 3-task 0.509 @ep12, rooms 0.50 vs frozen 0.087 / pipeline 0.993 — cross-carrier
  set-union PARTIALLY learned (123240, `carrier_layer_mixture3/20260718_134209_L17_r8/`).
- **Cross-DOMAIN carrier** (123208, `carrier_token_crosstask_natural/20260718_122538_proxy_room_k1/`,
  n=50 — wide bars): steps-distilled e_c on natural dist_far zero-shot d′ **3.19 ± 0.55** = ~51%
  of the cell's replica teacher (6.22); fresh-gate tally 0.432 UNDERPERFORMS the frozen model
  (0.58) — the carrier is partly domain-bound (vs 88% cross-task within synthetic MMRED).

## [2026-07-15] ❌📊 ONE-FORWARD REPAIR ARMS (oneforward Exp B/C, never logged) — the offline encoding un-mixer does NOT transfer to the replica forward (3.56 → **1.44**, destructive), and a CoGNN-style content-side broadcast gate cannot repair routing (1.80 ≈ the 2.09 floor); the "qcond GO" is INVALID by q_pad feature leak (q_pad itself carries d′ 8.63)

- **B1** un-mixer retrained + weights SAVED (121919, `unmixer_saved/20260715_194450/`, weights
  `unmixer_saved/weights/unmixer_L16.pt`): offline mp-q × un-mixed-kv 5.94 vs joint-kv 3.82 /
  ceiling 6.33 = **84% of the encoding gap** (prior 93% ≈ retrain variance).
- **B2 deployed composition** (121928, `replica_unmix/20260715_204158/`; g_k/g_v hooked on
  frame-token k/v_proj @L16 in the unmasked replica forward): **3.56 → 1.44 @L16 — NO TRANSFER,
  actively destructive.** Frame 0 alone improves (3.73→4.17, in-distribution); L14 control
  unchanged (2.70 vs 2.72) — the replica layout's question-conditioned k/v are off the
  un-mixer's training distribution. (Moot after A3 blockfence: prevention beats repair.)
- **C broadcast gate** (121918, `broadcast_gate/20260715_194451/`; in-run anchors PASS 2.09/3.82):
  content arm ([k_j,v_j]) eval **1.80** (trajectory max 2.13 ≈ the 2.09 floor) — pre-registered
  "routing NOT repairable from content" outcome, closing addressing from a third direction
  (trained shared query NO-GO · content gate NO-GO · only architectural frame identity works).
  **qcond arm 30.69 = INVALID (do NOT cite):** q_pad is captured after the question attends the
  frame — the leak probe (121927, `broadcast_gate/qpad_leak/20260715_201750/`) reads q_pad alone
  at d′ **8.63** (q_mp 10.32) ≥ the mp×mp ceiling, while pooled content k/v-means carry ~0.65 —
  the gate broadcasts label information, and computing q_pad at inference = multipass anyway.
  The script's auto-VERDICT "GO" line is superseded by this reading.

## [2026-07-25] TRUNC campaign E1 (candidate for RESULTS.md)

- **Finding**: the caption/L12 winner's generation is NOT carrier-only at decode time —
  masking frame columns from decoded rows (KV-drop semantics) changes 15/16 transcripts,
  starting at the first evidence-verdict token, and collapses answers (kvdrop
  answer-equal 1/16). Cause chain: build_block_mask leaves tail/decode rows causal over
  frames (code truth, executable smoke) + the cached trainer teacher-forces target rows
  with that visibility → the LoRA learned decode-time frame reads. Carrier aggregation
  (prefill) and decode readout (frames) are SEPARATE channels in the current ckpt.
- **Engineering**: cached incremental decode over [question]+[carriers]+[tail] is exact
  vs the mask-only arm 18/20 (2 near-tie numeric flips) and cuts per-sample decode
  59.7s→3.7s (mix) / 608.9s→6.0s (N=64) on h200; keep=103 of 12775 tokens at N=64.
- Runs: `outputs/ladder/image_longN/trunc_kvdrop/e1{a,b}/20260725_*_evalonly/report.txt`
  (jobs 125554/125555); PREREG + amendment in `plans/trunc_PREREG.md`.
