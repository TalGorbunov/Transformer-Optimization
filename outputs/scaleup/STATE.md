# SCALEUP — STATE (append-only, newest last)

## [2026-09-23] Campaign created on branch `theory-2` from the approved plan (Claude; approved by Tal 2026-09-23) — no code, nothing run
- Source: `docs/SCALEUP_2026-09-23.md` (live copy of `~/.claude/plans/sorry-i-forgot-to-valiant-lake.md`).
  Brief = `CAMPAIGN_BRIEF.md`; index = `INDEX.md` (every cell family "not run").
- Facts verified today (they shape the design): HERBench lite_v2 = 1,971 Q / 68 videos / 12 tasks,
  explicit evidence timestamps in `metadata_json` for 11 tasks (RLPC = "All"), three timestamp formats,
  AC true_count = 1 for 115/144, letter prior C 26 %; videos 34.3 GB in 4 tar parts, no train split.
  MINERVA = 1,285 (DeepMind) / 1,341 (lmms) Q, ~190 videos, 5-way MCQ, timestamps only inside
  `reasoning` (99.6 %, ~4 per Q), Listening (131) needs ASR, lmms Lance mirror 8.85 GB / 197 videos.
  transformers 4.57.6 has native InternVL / Gemma3 / Mistral3 / Qwen3VL classes; `.venv` has no video
  decoder and no lance; only Qwen2.5-VL-7B weights on disk; `/rg` 138 G free (97 %); home 140/300 G.
  July HERBench-lite AC work (144 Q) is in `docs/archive/RESULTS_pre_fencing.md:2342-2430`; its videos are gone.
- Decisions (Tal): backbones InternVL3.5-8B-HF + Gemma-3-12B-it; no closed-model rows; `av` + `pylance`
  installs; videos kept on `/rg`; 50 rows/task; stage 2 (per-block evidence classification) before the method.
- Execution order: data seam + Qwen backbone spec (no behaviour change) → GPU refactor proof →
  `prepare_video.py` + videoqa/mcq → S1–S3 on Qwen (HERBench, then MINERVA) → InternVL S4 → Gemma S4 → S5 → S6.
- No job submitted. No GPU hour spent. No RESULTS.md entry.

## [2026-09-23] S0 code: step 1 drafted (seams + evaluate.py), CPU tests green; refactor-proof jobs submitted (Tal: "run jobs while I review")
- Code (uncommitted, Tal reviewing): `core/data/`, `core/backbones/`, `core/prompt.py` (ARMS, system_prompt=), `core/model.py`
  (wrappers over the Qwen spec; `get_rope_index_fn` walks PEFT/`.model` wrappers), `core/fence.py` (`reset_positions` over
  any leading shape; `greedy_decode(passthrough_keys=)`), `core/constants.py` (TURN_END), `experiments/evaluate.py`
  (`--dataset --backbone --protocol --n-frames --arm`). Tests: 11 files green incl. legacy/v1 parity; new
  `tests/test_data.py` (7), `tests/test_backbones.py` (7), `test_fence.py::test_reset_positions_leading_shapes`.
- 159294 (C2 p2) FAILED: `get_rope_index` looked up on the PEFT-wrapped model (PeftModel.model = the OUTER model);
  fixed in `core/model.py` + evaluate passes the pre-wrap model; regression test added. Resubmitted as 159297.
- **159297 C2 p2 port check = 44/50 = 0.880 [0.780, 0.960]** vs anchor 45/50: one row moved (qid 0004694, gold 5,
  legacy 5 -> now 6); 49/50 rows identical -> PASS within the gate's 1-2 row tolerance (same class as C1 port's 3 rows).
  First C2 gate run ever; the fenced path (`bb.fenced_setup` + `greedy_decode`) is exercised through the seams.
  Run: `outputs/scaleup/mmred/qwen2.5-vl-7b/s0_proof/c2_p2_N8/20260923_201335_pc-p2/` (+ `port_check_diff.txt`).
- 159293 C1 faithful N=8 (target 618/1200 = 0.515): running on n314.
- Local symlink `checkpoints/sft_fenced_hf_adapter` was missing (README row exists); recreated (relative link).
- **159293 C1 faithful N=8 through the seams = 618/1200 = 0.515 [0.487, 0.543]** — the anchor (156676) exactly at the count; per row 16 flips (8 up / 8 down) and 25 raw-text diffs of 1200 = bf16 drift between the two runs (different nodes/GPU types; the plain arm is model.generate, untouched by the seams). Evaluate-side refactor proof PASS (C1 + C2). D4 half waits for the gate_capture port.

## [2026-09-23 evening] Video seam + prep drafted; HERBench data landing; D4 proof submitted (Tal: pip allowed, continue while reviewing)
- Installed into `.venv`: `av` 18.1.0, `pylance` 12.0.0 (torch 2.8.0 / transformers 4.57.6 untouched; all tests re-run green).
- Code (uncommitted): `core/data/mcq.py` (verbatim ports of HERBench `_format_prompt`/`extract_answer_choice` and lmms-eval MINERVA
  `doc_to_text`/`_extract_choice_letter`), `core/data/videoqa.py` (parsers for every HERBench timestamp format + the MINERVA trace regex,
  POOL=256 official grid nested across N, `planted` / `uniform` composition, `VideoMCQ` + `HERBench` + `Minerva` specs, registered),
  `core/prompt.py` (`system_prompt=None` = no system turn; `answer_text=`), `experiments/prepare_video.py` (rows / videos / frames / tar /
  verify), `sbatch/prepare_video.sbatch`, `sbatch/lib/common.sh:stage_video`, `experiments/gate_capture.py` + `gate_fit.py` ported to the
  seams (`--dataset/--backbone`, `group` array, `--group-by group` default, balanced accuracy), `sbatch/{evaluate,gate_capture}.sbatch`
  gain DATASET/BACKBONE/PROTOCOL/N/ARM knobs. Tests: `tests/test_data_videoqa.py` (11) green; 12 CPU test files green.
- Data: `data/herbench_v2 -> /rg/shocher_prj/lab_data/herbench/lite_v2`, `data/minerva -> /rg/shocher_prj/lab_data/minerva/lmms_v1`.
  HERBench rows: 1,971 / 68 videos; per task k = AC 1.80 (max 11), ASII 5, FAM/FOM 4, SVA/TSO 4, MEGL 2.03, MPDR 2.82, AG* 1, RLPC = all.
  MINERVA rows: 1,207 kept (Listening 134 dropped), 197 videos; evidence parsed for all but 4 rows (kind none); counting k mean 4–5.
  HERBench tar parts 00–03 (34.3 GB) downloaded to /rg in 7 min; job 159358 (4h_0g) = sha256 + stream-extract + 256-frame pools +
  evidence frames + tar + verify. MINERVA blobs (8.85 GB) fetching on the login node (`lmms_v1/fetch_videos.log`).
- D4 half of the refactor proof: gate_capture fenced_qfirst seq_len_16 train (50/qtype) through the seams, to be compared per frame
  with `outputs/diag/gate/N16_train/fenced_qfirst/` (2026-09-22) and refit with the N8 capture (anchor L20 0.975 / AUC 0.998).

## [2026-09-28] Proof closed on the capture side; HERBench wave on Qwen submitted (Tal: "do all those")
- D4 capture through the seams (159361): 19,200 frames, same (qid, frame) keys as the 09-22 capture; cosine mean 0.99996 / 0.99980 /
  0.99981 at L12 / L20 / L24 (min 0.9938), labels identical. Refit (N8 old + N16 new, by qid) = job 164391; reference fit = 164392.
- HERBench prep: 159358 TIMEOUT after OOM kills (decoder held all frames of a video in memory; 8 workers x 1408^2 > 16G) — fixed by
  streaming frames to disk; resume job 160327 finished in 2:35. `verify_report.txt`: planted samples 1,653 / 1,820 / 1,823 / 1,823 / 1,823
  at N = 8 / 16 / 32 / 64 / 128 (skipped: RLPC kind=all 148; k > N/2 170 at N=8, 3 at N=16).
- Smokes (1 row per task): s1 qfirst 4/10 @N8, 6/11 @N16; s2 fenced_qfirst 80 / 176 frames captured. Both clean.
- Wave (13 jobs, 50 rows/task, planted; comparability row = uniform N=16 plain on ALL rows): 164375-164379, 164381-164383, 164385,
  164387-164390. Four jobs of the first submission (164374, 164380, 164384, 164386) were cancelled before start: they came from the
  wave script before its patch (per-task cap on the comparability row; 48G instead of 96G on the h200) and were resubmitted.
- The wave jobs read frames from /rg (NFS), not the NVMe: the cell wrapper's STAGE knob collided with common.sh's staging switch.
  Results unaffected; wrapper fixed (STAGE_DATA) for later waves.
- MINERVA: 58/197 videos on disk; the hf:// blob fetch was ~6 min/video and the local table download broke at 248 MB. A retrying
  chain is running from the login node (`~/hf_fetch.log`). InternVL3.5-8B (2.9/17 GB) and Gemma-3-12B (3.7/24 GB) follow in the same chain.
- New: `sbatch/gate_fit.sbatch`, `experiments/backbone_probe.py` + `sbatch/backbone_probe.sbatch` (job 164393).
- **D4 refit PASS (164391 vs 164392):** L12 0.8945 / 0.9666, **L20 0.9768 / 0.9979**, L24 0.9686 / 0.9965 (acc / AUC, held out by qid, n=7,144) against the reference fit on the two 09-22 captures 0.8922 / 0.9651, 0.9745 / 0.9978, 0.9692 / 0.9964. The refactor proof is closed on all three cells.
- **S1 comparability row (164387):** HERBench lite_v2, uniform N=16, plain, all 1,971 rows = **0.461 [0.440, 0.484]**, parse_fail 0. Per task: AGBI 0.874, ASII 0.779, AGLT 0.725, AGAR 0.688, AC 0.597, FOM 0.560, FAM 0.557, SVA 0.351, MPDR 0.327, RLPC 0.257, MEGL 0.216, TSO 0.191. Predicted-letter distribution follows the gold one (no letter collapse). Not comparable to the paper's 0.359 (full set, different difficulty); in the band of the lite_v2 third-party rows (0.46–0.48 for other 8B models).
- MINERVA prep done (164445): 194/197 videos; 3 truncated blobs in the mirror (19 rows) skipped as no_pool. Planted N=8 usable rows 952, uniform 1,188.
- Backbone specs written (`core/backbones/{internvl,gemma3}.py`) from the probe; fence extended (separator-aware blocks, bidirectional image squares, per-layer-type masks with the sliding window in reset-position space). Smokes v1: InternVL runs but thinks (7/12 rows cut inside <think> at 24 tokens) -> `assistant_prefix` = empty think block (PROTOCOL CHOICE, awaiting Tal's confirmation); Gemma failed on the slow tokenizer (sentencepiece) -> default processor. Smokes v2 = 164452-164455.
- **Backbone smokes, MMReD seq_len_8 test, 24 rows (pipeline checks, not results):** InternVL3.5-8B with the non-thinking prefix: plain 7/24, gated 11/24, parse_ok 24/24 (164452/53). Gemma-3-12B: the spec load left the non-quantized modules in float16 -> NaN logits, pads, 0/24 (164454); diagnosis 164458 (bf16: finite logits, 3/3 correct, 11.7 GB) -> `torch_dtype="bfloat16"` in the spec; after the fix plain 7/24 (164460), gated 12/24 (164461). Also fixed: `greedy_decode` pads per-token inputs (token_type_ids) as the sequence grows. 4-bit Gemma is usable; no fallback model needed.
- Gotcha: jobs of the same user on one node share `$TMPDIR/mmred`; a finishing job's cleanup removed the staged frames under a running one (164457). Use STAGE=0 for short jobs until stage_split uses a per-job directory.

## [2026-09-29 00:40] HERBench x Qwen wave complete (`planted`); first per-block evidence numbers; protocol correction `isolated`
- S1 frozen ladder (50 rows/task): plain 0.598 / 0.583 / 0.555 / 0.555 / 0.538; qfirst 0.529 / 0.462 / 0.344 / 0.267 / 0.173 (N=8…128).
  qfirst @128 per-row: pred 'C' 311/532, empty-parse 138/532 (free-text captions such as "The man Y walks past the camera."), all tasks <= chance.
- S3 oracle gate: 0.467 / 0.491 / 0.491 / 0.487 / 0.492 — N-invariant. Level below plain: open (fence cost on a frozen model vs tasks that need
  context beyond the annotated frames); read per task after the isolated re-run.
- S2 fits (gate_fit --folds 5 --group-by group --train-n 8 16; test = held-out videos at every N): fenced_qfirst AUC 0.628 / 0.743 / 0.741 at
  L12 / L20 / L24 (bal 0.59 / 0.66 / 0.66); qfirst 0.757 / 0.809 / 0.620 (sd 0.09 at L24). MMReD anchor at L20: 0.977 / 0.998.
  Per task (fenced L20 AUC, N=8 -> 128): TSO 0.93 -> 0.91, SVA 0.91 -> 0.91, ASII 0.72 -> 0.73, FAM 0.69 -> 0.71, FOM 0.75 -> 0.70, AC 0.74 -> 0.69,
  AGLT 0.60 -> 0.57, AGAR 0.58 -> 0.55, AGBI 0.55 -> 0.53, MEGL 0.55 -> 0.55, MPDR 0.55 -> 0.52. qfirst: ASII 0.92, FOM 0.91, FAM 0.89 (position
  confound: evidence moments cluster in time), TSO 0.87 -> 0.65, SVA 0.90 -> 0.59 (decays with N).
- DESIGN FLAW FOUND (mine): under `planted` the fillers keep their margin from the planted frame's TIME, not from the evidence INTERVAL; for
  presence-interval evidence (AG*, MEGL, MPDR) fillers inside the interval show the person and are labelled 0. New protocol `isolated` (fillers
  outside every evidence interval +- margin); identical frames for point evidence. Usable rows under isolated: 1,607 @N8 … 1,668 @N128
  (too-few-fillers skips 46 … 155). Re-run for AGAR AGBI AGLT MEGL MPDR TSO SVA = jobs 164567–164578 (STAGE_DATA=0).
- gate_fit: + `--folds`, `--train-n`, position-only baseline (pseudo-layer POS: cubic in relative frame index, per task) — fit 164579.
  A first fenced fit (164556) was cancelled before output: its glob also matched the 09-25 smoke captures.

## [2026-09-29 evening] Resolution error found; protocol corrected; batched fenced path built and proven (Tal: "do all those")
- ERROR (mine): frames were stored at 512 px long side; the benchmark's frame-based code feeds the video's own resolution (Qwen: 2,691
  tokens for a 1080p frame vs our 180). Every HERBench number of 09-28/29 is a "512 px" row: not the paper's protocol, not evidence about
  perception. Details + decisions: `docs/SCALEUP_2026-09-23.md`, addendum 2026-09-29.
- Isolated-protocol fits (164624-26), 512 px frames: person tasks stay near chance (fenced L20 AUC AGAR 0.57, AGBI 0.60, AGLT 0.57; blind 0.52-0.53;
  POS 0.58-0.68) while oracle-gated QA on them is 0.72-0.90. Facts behind it: all person tasks are WildTrack (1920x1080 crowd scene, 14 videos,
  one location); at 512x288 a described person is ~15x40 px; 31-50 evidence frames per task vs 3,584 features. Ordering tasks lose accuracy under
  the fence (TSO 0.62 -> 0.30 at N=8): the position reset removes temporal order from the read.
- Native frames extracted (frame set `native`): HERBench 68 videos (164899), MINERVA 194 (164900; mirror videos are 360p/480p).
- `core/fastpath.py` (batched fenced path) + `--fast` in evaluate / gate_capture; `--frame-set`, `--max-pixels`; loader default = uniform + native.
  CPU parity (tiny random Qwen2.5-VL, float32): slots 1e-8, logits 3e-7, identical tokens (gated / ungated, both layouts, chunked, known-gate skip).
  GPU parity on MMReD N=8 gated, 120 rows: dense 0.592 = batched 0.592, correctness identical 120/120, 5.7 min vs 15 min; captures N=16 cosine
  0.99986 at L20 (min 0.994) — the same drift as dense-vs-dense across GPU types.
- Native smokes 164935-37 (1 row per task), then the wave `sbatch/scaleup_submit_native.sh herbench qwen2.5-vl-7b all` (13 jobs).
- Batched path on InternVL3.5 (non-thinking; Tal confirmed the mode 2026-09-29): GPU, MMReD N=8 gated, 120 rows: dense 0.467 vs batched 0.442, raw identical 115/120, correctness identical 117/120. Exact parity on a tiny random InternVL in float32 (3 configurations: gated, ungated chunked, paper layout with the last frame hidden): slot states 2e-8, logits 3e-7, identical tokens. The GPU differences are numerical (bf16, different SDPA kernels with and without a mask), not structural.
- Native wave submitted 2026-09-29 (HERBench x Qwen 164939-51; InternVL x MMReD 164952-58; MINERVA reference row 164965 + smoke 164966).
- **InternVL3.5-8B on MMReD (non-thinking):** frozen paper layout 0.561 @N=8, 0.478 @N=16 (Qwen 0.515 / 0.415). Per-block evidence probe, D4 protocol: fenced_qfirst peaks at L21 (depth 0.58): acc 0.928, AUC 0.982; question-blind fenced_qlast at chance on all five layers. The MMReD pattern (evidence readable at the block's end token only when the block sees the question) reproduces on a second backbone with a different vision tower, a different language model generation and 1-D positions; the readout is somewhat weaker than Qwen's 0.977 / 0.998.
- **MINERVA reference row (164965):** Qwen, uniform N=32, plain, native, 1,188 rows = **0.325 [0.299, 0.351]**, parse_fail 0. Per skill: object recognition 0.397, reading 0.390, cause and effect 0.367, situational 0.367, numerical 0.314, event occurrence 0.296, counting 0.290, temporal 0.286, spatial 0.255, counterfactual 0.129 (n=31), state changes 0.467 (n=15), goal 0.286 (n=14). Predicted letters skew to E (304) and away from A (141) while gold is near-uniform. In the band of the published 7B rows at 32 frames (0.29-0.32, all RL fine-tunes of the same backbone).
- HERBench native, first complete cell: plain uniform N=8 uncapped 312/582 = 0.536 (caps: 1 MP 0.522, 151,200 px 0.519).
- **Gemma-3-12B on MMReD:** frozen paper layout N=8 0.472. Per-block evidence probe (D4 protocol): fenced_qfirst peaks at L27 (depth 0.56): acc 0.921, AUC 0.980; question-blind control at chance on all five layers. Third backbone, first non-Qwen language model: same pattern. First fit attempt crashed at L41 (`Input X contains NaN`): one Gemma dimension exceeds float16 in the stored capture (17,416 / 19,200 frames at N=16); gate_fit now reads inf as +-65504. TODO after the running wave: clip in gate_capture before the float16 cast.
- **Three-backbone table, MMReD, fenced question-first, best layer (acc / AUC):** Qwen2.5-VL-7B L20 0.977 / 0.998; InternVL3.5-8B L21 0.928 / 0.982; Gemma-3-12B L27 0.921 / 0.980. Question-blind: chance on all three. Peak depth 0.71 / 0.58 / 0.56 of the stack.

## [2026-09-30] Native-resolution probe at uniform N=16 (fits 165141-44): person tasks still near chance; two measurement problems found
- Gemma MMReD N=16 plain 0.388. Three-backbone frozen ladder N=8 -> 16: InternVL 0.561 -> 0.478, Qwen 0.515 -> 0.415, Gemma 0.472 -> 0.388 (drop 0.08-0.10 each).
- HERBench, Qwen, uniform N=16, native uncapped, fenced, L20, 5-fold by video (816 frames per task, 50 questions):
  AGAR 0.623 (blind 0.581, POS 0.464) · AGBI 0.580 (0.524, 0.293) · AGLT 0.554 (0.527, 0.481) · MEGL 0.587 (0.592, 0.470) · MPDR 0.568 (0.561, 0.398) ·
  TSO 0.576 (0.519, 0.539) · SVA 0.568 (0.514, 0.534). 1 MP cap: same picture. Resolution did not rescue the person tasks (512 px isolated: 0.57-0.60).
- PROBLEM 1 (label rule): under uniform sampling at N=16 frames are ~9 s apart and `uniform_labels` accepts +-min(2 s, half spacing); for 1 s trailer shots
  most labelled frames show a different shot -> TSO/SVA fall from 0.91 (planted, exact frames) to 0.57. Fix after the wave: inside-the-interval only.
- PROBLEM 2 (power): point-evidence tasks have 1-2.4 % positives at N=16 (8-20 frames per task) -> ASII / FAM / FOM / AC rows unreadable; pooled "all"
  AUC (0.83; POS 0.87; blind 0.79) only reflects per-task base rates (RLPC 100 %, MPDR 53 %, AC 1 %).
- Clean resolution A/B submitted: isolated protocol, native uncapped, N=16, batched captures, interval tasks, up to 200 rows per task, both arms.
- Unfenced interim (582 rows): plain N=8 native 0.536 / 1 MP 0.522 / 151,200 px 0.519; qfirst small cap 0.455 -> 0.292 -> 0.163 -> 0.220 at N=8..64.
  Oracle gate batched: native 0.409 / 0.411 / 0.448 (84 rows too_long at N=32); 1 MP 0.405 / 0.414 / 0.423 / 0.456 (78 too_long at N=64).
- **Oracle gate, batched, uniform, 1 MP cap (164951), 415 rows common to every N:** 0.427 / 0.436 / 0.429 / 0.455 / 0.467 / 0.506 at N=8…256; mean kept frames
  0.6 / 1.2 / 2.3 / 3.9 / 6.3 / 10.6; rows with no evidence in view 226 / 163 / 114 / 58 / 20 / 5; read tokens 870 … 12,400. Per task N=8 -> 256:
  AC 0.08 -> 0.48, FAM 0.36 -> 0.56, FOM 0.32 -> 0.48; AGAR / AGBI / AGLT flat at 0.74-0.94; ASII 0.70 -> 0.66; SVA 0.38 -> 0.28; TSO 0.16 -> 0.26.
- **TEXT-ONLY ANSWERABILITY (from the same run, N=8, rows where the gate kept no frame):** AGAR 1.00 (n=5), AGBI 0.92 (12), ASII 0.68 (44), AGLT 0.67 (12),
  FAM 0.38 (45), FOM 0.29 (42), SVA 0.20 (5), TSO 0.15 (13), AC 0.04 (46). The person tasks and ASII are largely answerable from the question + options
  alone by this model: they are not needle tasks in practice, which explains the near-chance evidence probe on them. Evidence-dependent tasks = AC, FAM, FOM
  (and, by the planted probe, TSO / SVA). TODO after the wave: a no-frame arm on ALL rows (needs a `--gate blind` option in evaluate.py).
- **Clean resolution A/B (165286/87): isolated protocol, N=16, fenced, L20, 5-fold by video, native vs 512 px** — TSO 0.931 vs 0.888, SVA 0.895 vs 0.903,
  AGAR 0.541 vs 0.539 (n=496 frames), AGBI 0.626 vs 0.595, AGLT 0.646 vs 0.570, MEGL 0.558 vs 0.478, MPDR 0.564 vs 0.517. Question-blind at native 0.53-0.57.
  Resolution adds 0-8 points on the person / multi-person tasks and leaves them near chance; the trailer shot tasks are ~0.9 at both. Combined with the
  text-only answerability of AGAR / AGBI / AGLT, the per-block probe's weakness on those tasks is not a resolution or perception failure to chase.
- Oracle gate native uncapped (164950) completed: 0.409 / 0.411 / 0.448 / 0.473 / 0.480 / 0.510 on 582 / 582 / 498 / 450 / 406 / 363 rows — same shape and end
  point as the 1 MP cap; full resolution buys nothing here and drops more rows at the read limit -> 1 MP is the operating point on HERBench.
- Unfenced 1 MP ladders: plain 0.522 / 0.531 / 0.517, qfirst 0.393 / 0.402 / 0.419 at N=8 / 16 / 32 (32 = the context limit at 1 MP).

## [2026-09-30 11:00] Wave collected; readout comparison (stage 2b) built and submitted (Tal: "yeah do that")
- Native wave complete except the 1 MP capture chains (164947/164949, on their last cell N=256, ~96 s/row). New numbers:
  THE PAPER'S ROW at full resolution (164939) = **938/1971 = 0.476 [0.453, 0.499]** (512 px row was 0.461); per task AGBI 0.943, AGAR 0.938,
  ASII 0.831, AGLT 0.791, FOM 0.597, FAM 0.582, AC 0.514, MPDR 0.327, SVA 0.317, MEGL 0.242, TSO 0.237, RLPC 0.223; evidence in view 75.5 %.
  Small-cap ladders (582 rows, N=8..256): plain 0.519 / 0.500 / 0.510 / 0.512 / 0.521 / 0.510 (flat to 256); qfirst 0.455 / 0.292 / 0.163 / 0.220 / 0.172 / 0.141.
  Matched 415-row table (1 MP, unfenced N=8/16/32 vs oracle gate N=8..256) computed from the eval csvs; AC unfenced 0.74 / 0.60 / 0.42 is the "1" prior.
- Results artifact (tables, no codenames): https://claude.ai/artifact/HaqnXheV3rABZGVzG6u9CD (Tal: do not edit for now).
- Verdict given to Tal: the method can work on AC / FAM / FOM (evidence tasks) and cannot on the text-answerable, ordering and multi-person tasks;
  the decisive unknown is a gate good enough at N=256 (1-4 % positives) — the linear slot probe (AUC ~0.7) is not.
- Stage 2b (readout comparison) built: `core/fastpath.py` gains `pool_layers` (mean of the block's image-token states) and a per-block
  yes/no JUDGE (`judge_ids` + `judge_vocab`: [prefix + block t] + instruction + assistant opener, batched over blocks; logits of the first
  answer token at the yes/no ids). Parity test added (`tests/test_fastpath.py::test_judge_and_pooled_states_parity`: judge == the dense
  one-frame prompt question/frame/instruction for every block; pooled == dense; chunked == unchunked; the read is untouched) — 12/12 green.
  `gate_capture.py --pool --judge TEXT` stores P and J; `gate_fit.py --features slot|pool|both`, `--per-task-fold`, JUDGE pseudo-layer
  (training-free, scored on the same held-out folds), and operating-point metrics precision / recall@spec95 / recall@spec99 in the CSV.
  Wrapper knobs POOL / JUDGE in `sbatch/scaleup_cells.sbatch` (run dir suffix `_pool_judge`).
- Judge instruction (after the frame, same user turn): "Does this frame contain evidence needed to answer the question above? Answer yes or no."
  Smoke 165370 (1 row/task, N=16): 80 frames, J finite, all scores < 0 (the judge says "no" to every frame; the ranking is what the fit reads).
- Submitted (HERBench, Qwen, uniform, native frames, 1 MP, batched, fenced_qfirst, slot + pooled + judge in ONE pass):
  165371 N=64 AC FAM FOM TSO SVA x150 (12h_4g) · 165372 N=256 AC FAM FOM x150 (24h_1g, ~13 h) · 165373 N=256 TSO SVA x100 (12h_4g) ·
  165374 isolated N=64 AC FAM FOM x150 (exact evidence frames; 24h_1g). Fits on the finished full-res captures N=16+64 (164946/48): 165375-77.

## [2026-10-06] UNIT baseline planned and built (Tal approved the plan; branch `theory-2`): per-unit perception before more datasets / backbones
- Question (Tal): before scaling the method up, where is the baseline of how well the model perceives ONE evidence unit by itself, given
  that resolution may be too low and an action is temporal (one frame is not enough)? Plan: `docs/UNIT_BASELINE_2026-10-06.md`
  (live copy of `~/.claude/plans/ok-lets-take-a-stateful-axolotl.md`). Decisions (Tal, AskUserQuestion): first wave = Qwen x HERBench
  + the MMReD anchor; UNFENCED only (clip-blocks in the fence = the first method cell afterwards, not here); video-mode input deferred.
- Three regimes, all through the model's own output (task- and model-generic: question + gold + evidence timestamps; generation /
  first-token logits only): **T** text-only (no frames) · **E(U)** evidence-only (the k annotated units, no fillers, official layout,
  accuracy by k) · **D(U)** per-unit yes/no (one unit + bare question + the stage-2b judge line; hard negatives = the grid 2 s after the
  interval, easy = a seeded grid >= 5 s from every interval; position-free). Unit sweep U = {frame, clip3_d1, clip5_d1, clip5_d2,
  clip9_d2} x {lo, hi} where lo / hi are the backbone's OWN resolution modes (Qwen max_pixels 151,200 / 1,003,520; InternVL one tile /
  dynamic tiling; Gemma one tile / pan-and-scan). Derived per task: perception wall = max_U E - T; resolution gain; temporal gain;
  fence cost = E(frame hi) - oracle gate (exists); retrieval room = max_U E - official row (exists). Stop rule for SCALEUP replaces the
  S2-AUC one: max_U E - T not above the CI -> perception wall, no method cell; text-answerable tasks (T >= 0.6) excluded as such.
- **Stage-2b readouts of 2026-09-30, never written up until now** (`s2_fit/readouts_*.csv`; Qwen, fenced_qfirst, 1 MP, 5-fold by video;
  JUDGE = the model's own yes/no on the single frame in-fence, POS = position only, slot = linear probe L24):
  uniform N=64 (pos rate 1-10 %): AC JUDGE 0.786 / slot 0.772 / POS 0.536 · FAM 0.685 / 0.692 / **POS 0.811** · FOM 0.625 / 0.682 / **POS 0.767** ·
  TSO 0.700 / 0.634 / 0.528 · SVA 0.677 / 0.618 / 0.513 (AUC). JUDGE recall@spec95 / precision@spec95: AC 0.325 / 0.067, FAM 0.177 / 0.070,
  FOM 0.178 / 0.075, TSO 0.285 / 0.382, SVA 0.262 / 0.364. uniform N=256 trailers: JUDGE TSO 0.810, SVA 0.809 (slot 0.755 / 0.753; POS 0.49-0.50);
  recall@spec99 0.236 / 0.182. isolated N=64 (exact evidence frames): JUDGE AC 0.836 (rec@spec95 0.403, prec 0.189), FAM 0.747, FOM 0.664;
  POS FAM 0.853, FOM 0.803 (evidence clusters in time). Pooled image-token features are below the slot everywhere; 165372 (N=256 AC/FAM/FOM)
  FAILED on a missing pool frame after 200 rows (no capture). Reading: the single 1 MP frame is a weak unit for the HD-EPIC action tasks
  even for the model's own judgement, and FAM / FOM are position-confounded under uniform sampling -> the per-unit baseline must be
  position-free (one unit alone) and must vary the unit (clips) — exactly what this campaign measures.
- Code (uncommitted, Tal reviews): `core/data/videoqa.py` (unit grid `unit_grid` / `negative_centres` / `plan_units`, protocol
  `evidence_<unit>`, `units()`, MAX_UNIT_FRAMES = 48 with fallback clip9_d2 -> clip5_d2 -> clip3_d1 -> frame), `core/data/base.py`
  (`evidence_only`, `units`, `bare_question`, `evidence_strata` = accuracy by k), `core/data/mmred.py` (strata), `core/backbones/base.py`
  (`BackboneSpec.res_modes`, `set_resolution`) + the three specs, `experiments/prepare_video.py --stage units verify_units --qids-file`,
  `experiments/evaluate.py --no-frames | --unit | --res`, NEW `experiments/unit_judge.py`, `sbatch/unit_baseline.sbatch` (REGIME=T|E|D,
  UNITS chained) + `sbatch/unit_submit.sh`, `sbatch/lib/splits/qids_herbench_native582.txt` (the 582 rows of the 09-29 native wave,
  identical across its S1 / S3 cells). Tests: 15 files green incl. new `tests/test_unit_judge.py` (tiny random Qwen) and the unit-grid /
  evidence-protocol / zero-frame / resolution-mode tests. Fence and fastpath untouched. The stale +-2 s `uniform_labels` rule is NOT fixed here.
- Unit plan simulated on the 582 rows with the real pool durations (CPU, no decode): 532 rows with evidence, 1,516 positive units,
  36,237 frames to decode (~11 GB native). Per task pos / hard / easy: AC 92 / 88 / 92, ASII 250 / 179 / 250, FAM 200 / 163 / 200,
  FOM 200 / 165 / 200, TSO 200 / 200 / 200, SVA 200 / 200 / 200, MEGL 99 / 77 / 81, MPDR 143 / 93 / 119, AGAR 32 / 31 / 29, AGBI 50 / 50 / 49,
  AGLT 50 / 49 / 46 (hard negatives drop when occurrences are < 6 s apart; easy ones when a presence interval covers the clip).
  Trailer shots are ~1 s: their grids hold 3 frames (TSO 171/200, SVA 181/200), so clip5 / clip9 = clip3 there. Fallbacks at 48 frames:
  only AC rows with k >= 6 (clip9_d2 -> clip5_d2 on 6 rows; 2 rows to clip3_d1).
- Pre-registered bands (plan §expectations): T ~ 1.0 / 0.9 / 0.7 / 0.7 on AGAR / AGBI / AGLT / ASII, <= 0.2 on AC k=1, FAM, FOM, TSO, SVA;
  E(frame hi) AC k=1 >= 0.6, FAM / FOM 0.5-0.7, TSO / SVA >= 0.7 and far above the oracle gate; clips +0.03-0.10 on HD-EPIC actions, ~0 on
  trailers; lo -> hi <= 0.05; D(frame hi) AUROC ~0.8 on AC, clips ~0.9, hard << easy. MMReD: wall = aggregation, not perception.
- W0 smoke submitted: job 170555 (`prepare_video.sbatch`, 4h_0g, `--stage units verify_units --limit 1 --qids-file ...`, one video).
  Full extraction follows the smoke; every GPU cell (T / E / D on HERBench, the MMReD anchor) waits for Tal's OK (`sbatch/unit_submit.sh`).
- W0 smoke 170555 COMPLETED (one 720p trailer: 10 questions, 560 frames in 10 s, verify clean: 0 missing, 0 hard inside interval+margin,
  0 easy < 5 s; 3-frame grids on the 1 s shots as designed). Full extraction for the 582 rows = job 170561 (4h_0g, 04:00:00, resumable).
- **W0 DONE (170561, 7 min 19 s on 4h_0g):** `data/herbench_v2/units_native/` = 532 questions, 1,516 positive units, 1,384 hard + 1,513 easy
  negatives, 12.35 GB (frames 1408x1408 HD-EPIC / 1920x1080 WildTrack + trailers, ~95-400 KB each); `verify_report_units.txt` clean and every
  per-task count equals the CPU simulation above. Loader check: `evidence_frame_test` 532 rows (1.0 frame per unit), `evidence_clip5_d1_test`
  (mean frames/row by task as planned), `evidence_clip9_d2_test` falls back only on the k >= 6 AC rows. The wave is ready:
  `bash sbatch/unit_submit.sh herbench qwen2.5-vl-7b all` + `... mmred ... all` — awaiting Tal's OK.
- **Tal (2026-10-06): "i approve everything you can work autonomously for now."** Wave submitted the same hour: smokes 170566 (HERBench
  T/E/D, 1 row per task, frame + clip9_d2 @hi, 2h_2g, -> outputs/_scratch/unit_smoke/) and 170567 (MMReD T/E/D, 1 row per qtype); HERBench
  170568 T · 170569 E frame@lo · 170570 E frame/clip3_d1/clip5_d1 @hi · 170571 E clip5_d2/clip9_d2 @hi · 170572 D frame@lo · 170573 D
  frame/clip3_d1 @hi · 170574 D clip5_d1/clip5_d2 @hi · 170575 D clip9_d2 @hi; MMReD 170576-78 (T/E/D seq_len_8) · 170579-81 (seq_len_16).
  Every GPU was busy at submission (all pending on Priority). Incident: the wave script's QOS popper ran in a $(subshell), so 8 jobs landed on
  12h_4g and 6 on 24h_4g (caps 3); moved while pending with scontrol update (12h_4g 3 · 24h_1g 4 · 24h_4g 3 · 4d_1g 4); script fixed.
  evaluate.py now counts an unfenced CUDA OOM row as too_long (the 45-frame clip9_d2 prompts are the longest unfenced prompts yet).
- Collection script `experiments/unit_table.py` (CPU): newest run per cell -> the plan's per-task table (T, E per unit, P* with its unit,
  O(N) / F(N) from the existing oracle-gate and official rows on the SAME qids, wall = P* - T with a paired bootstrap CI, resolution /
  temporal gains, fence cost, retrieval room, D(U*) AUC vs hard / easy, recall@spec99) + E by k and the unit fallbacks -> `s1u_unit/TABLE.md`
  (+ .csv). Checked on a synthetic run tree and on the real partial state (O / F columns only, until the cells land). Gotcha: run_logged
  drops `runner-*.log` into the cell dir, so a newest-run glob must require eval.csv / metrics.csv.
- **First cells back (2026-10-06 16:20).** HERBench **T (170568, 582 rows) = 226/582 = 0.388 [0.349, 0.428]**, per task: AC 0.040 · AGAR 0.719 ·
  AGBI 0.840 · AGLT 0.820 · ASII 0.680 · FAM 0.440 · FOM 0.300 · MEGL 0.160 · MPDR 0.280 · RLPC 0.060 · SVA 0.300 · TSO 0.140 (chance 0.2).
  Inside the pre-registered bands except FAM (0.44 from text alone, band <= 0.2) and SVA at 0.30. Run dir
  `s1u_unit/T/uniform_N8/20261006_161034_faithful_T`. Command lines of 170569 / 170572-74 / 170576 verified (PASS).
- INCIDENT (mine): the wrapper's generation budget defaulted to 8 tokens for every dataset; MMReD answers are JSON (~10 tokens), so the MMReD
  smoke's T had parse_fail 24/24 and E 9/22 (170567; D unaffected: no generation). Cancelled 170576 / 170577 / 170579 / 170580 (MMReD T / E),
  wrapper now defaults MAX_NEW to 24 on MMReD (8 on the MCQ datasets), resubmitted as 170607 / 170608 (T seq 8 / 16) and 170609 / 170610
  (E seq 8 / 16). The MMReD smoke's D row (192 units, AUC 0.662, yes-rate on evidence frames 0.18) stands as a pipeline check only.
- Throughput: E frame@lo 532 rows in ~4 min; D frame@lo ~4.6 units/s (yes on 11/200 evidence units at lo in the first 50 rows).
- **Smoke 170612 PASS (30 min on an A100 40 GB, off n315):** T, E frame@hi (tokens_mean 3,251 -> the hi mode is applied), E clip9_d2@hi
  (tokens_mean 24,096, max row 45 frames, too_long 0, parse_fail 0), D frame + clip9_d2 @hi (82 units each, finite). The 09-30 smoke on n315
  (170566) had sat > 15 min in model load on the saturated H200 node and was cancelled.
- **E frame@lo (170569, 532 rows) = 332/532 = 0.624 [0.583, 0.665]**, tokens_mean 710, no fallbacks; per task AC 0.740 · AGAR 0.781 ·
  FAM 0.720 · FOM 0.640 · SVA 0.300 · TSO 0.720 (others in the run dir); by k: k = 1 0.851 (n 174), k = 2-4 0.478 (n 301). AC k = 1 rows
  0.895 (38) vs k >= 2 rows 0.25: the frozen model counts one handed-over frame and little more (the aggregation wall, not perception).
  Run dir `s1u_unit/E_frame_lo/20261006_162701_faithful_E-frame_reslo`.
- **D frame@lo (170572, 4,277 units, 34 min):** the model's own yes/no on ONE low-resolution frame. AUC vs hard / easy negatives:
  SVA 0.888 / 0.920 · TSO 0.858 / 0.897 · AGLT 0.754 / 0.717 · AGBI 0.748 / 0.668 · AGAR 0.653 / 0.623 · MPDR 0.640 / 0.606 · AC 0.617 / 0.808 ·
  MEGL 0.616 / 0.593 · ASII 0.599 / 0.741 · FAM 0.596 / 0.727 · FOM 0.577 / 0.640 · all 0.685 / 0.735. Yes-rate on true evidence units is low
  everywhere (AC 0.03, FAM 0.07, FOM 0.10, MPDR 0.00, SVA 0.34): the judge answers "no" to almost every single frame; the ranking carries
  the signal. Precision@spec95 is 0.44-0.93 here because pos:neg is ~1:1 by construction (not the 1-4 % of uniform sampling).
  Run dir `s1u_unit/D_frame_lo/20261006_*_D-frame_reslo`.
- **MMReD T (170607 / 170608, 1,200 rows each, parse_fail 0 at 24 tokens): seq_len_8 0.189 [0.167, 0.211], seq_len_16 0.154 [0.134, 0.175]**
  (steps_in_room 0.36 / 0.18, n_empty 0.26 / 0.12, crowd_count 0.18 / 0.18, positional and where_spend 0.00). The MMReD text floor.

## [2026-10-07] UNIT baseline wave 1 COMPLETE (Qwen2.5-VL-7B x HERBench 582 pinned rows + MMReD anchor): perception is NOT the wall on the action / ordering tasks; the FENCE is
Tables: `outputs/scaleup/herbench/qwen2.5-vl-7b/s1u_unit/TABLE.md` (oracle column = uniform cells), `TABLE_planted_oracle.md` /
`TABLE_isolated_oracle.md` (oracle column = the 09-28/29 exact-evidence cells at 512 px: the matched "fence cost"), MMReD
`outputs/scaleup/mmred/qwen2.5-vl-7b/s1u_unit/TABLE_seq{8,16}.md`; all from `experiments/unit_table.py`. 13 wave jobs + 1 smoke, 0 failures,
~16 GPU-h. Every E cell: parse_fail 0, too_long 0 (clip9_d2 @hi = 25.6 K tokens mean, 45 frames max, on 40 GB A100s / the H200).

**HERBench, 532 rows with evidence units (T on all 582). E at hi (1 MP) unless noted; P* = smallest unit within 1 point of the best E@hi:**
| task | T | E frame lo | E frame hi | clip3_d1 | clip5_d1 | clip5_d2 | clip9_d2 | P* (U*) | wall P*-T [paired CI] |
| AC | 0.04 | 0.74 | 0.68 | 0.46 | 0.30 | 0.36 | 0.30 | 0.68 frame | 0.64 [0.50, 0.78] |
| TSO | 0.14 | 0.72 | 0.74 | 0.72 | 0.56 | 0.70 | 0.54 | 0.74 frame | 0.60 [0.46, 0.74] |
| FOM | 0.30 | 0.64 | 0.74 | 0.76 | 0.80 | 0.80 | 0.84 | 0.84 clip9_d2 | 0.54 [0.40, 0.68] |
| FAM | 0.44 | 0.72 | 0.78 | 0.82 | 0.78 | 0.82 | 0.80 | 0.82 clip3_d1 | 0.38 [0.22, 0.54] |
| ASII | 0.68 | 0.80 | 0.84 | 0.82 | 0.84 | 0.86 | 0.84 | 0.86 clip5_d2 | 0.18 [0.08, 0.30] |
| AGBI | 0.84 | 0.92 | 0.96 | 0.92 | 0.92 | 0.90 | 0.92 | 0.96 frame | 0.12 [0.04, 0.22] |
| AGAR | 0.72 | 0.78 | 0.72 | 0.84 | 0.84 | 0.84 | 0.81 | 0.84 clip3_d1 | 0.125 [-0.03, 0.28] |
| MEGL | 0.16 | 0.22 | 0.18 | 0.22 | 0.20 | 0.24 | 0.18 | 0.24 | 0.08 [-0.04, 0.22] |
| SVA | 0.30 | 0.30 | 0.30 | 0.30 | 0.32 | 0.32 | 0.36 | 0.36 | 0.06 [-0.12, 0.24] |
| AGLT | 0.82 | 0.84 | 0.82 | 0.78 | 0.72 | 0.78 | 0.78 | 0.82 frame | 0.00 [-0.10, 0.12] |
| MPDR | 0.28 | 0.24 | 0.24 | 0.26 | 0.20 | 0.28 | 0.24 | 0.28 | 0.00 [-0.14, 0.14] |
| ALL (532) | 0.419 | 0.624 | 0.633 | 0.620 | 0.581 | 0.620 | 0.594 | 0.633 frame | 0.214 [0.169, 0.263] |
By k (pooled, frame@hi): k = 1 rows 0.828 (n 174), k = 2-4 0.502 (n 301); AC k = 1 rows ~0.9, AC k >= 2 rows ~0.25 at every unit.
- **Perception walls (rule 1: P* - T not above the CI): SVA, MEGL, MPDR, AGLT, AGAR.** AGAR / AGLT (and AGBI at T 0.84) are text-answerable;
  SVA ("distorted" shots), MEGL, MPDR do not move even with the exact annotated units handed over -> no method cell on them with this backbone.
- **Method targets: TSO, FOM, FAM, AC (k >= 2 is the counting wall), ASII.**
- **Resolution is not the wall:** frame hi - lo = -0.06 (AC) ... +0.10 (FOM), pooled +0.009. (Band "<= 0.05": met except FOM 0.10.)
- **The unit is task-dependent:** clips add +0.10 FOM, +0.125 AGAR, +0.04 FAM, +0.02 ASII, but HURT the two tasks whose answer is read off the
  number / order of units: AC 0.68 -> 0.30-0.46 and TSO 0.74 -> 0.54-0.72 (pooled: no gain). U* per task = frame (AC, TSO, AGBI, AGLT),
  clip3_d1 (FAM, AGAR), clip5_d2 (ASII), clip9_d2 (FOM). (Band "clips +0.03-0.10 on HD-EPIC actions": met for FOM / FAM; "~0 on trailers": wrong, negative.)
- **FENCE COST is the big number.** The oracle-gated FENCED read of the same units (09-28 planted cells, 512 px, block-0 positions, no
  cross-block attention; common rows 450, AC only 18 at N = 8) vs E frame@lo (the matched resolution): TSO 0.30 vs 0.72 · FAM 0.46 vs 0.72 ·
  FOM 0.42 vs 0.64 · AC 0.44 / 0.52 (N 8 / 16) vs 0.74 · ASII 0.68 (N 16) vs 0.80 · AGLT 0.84 = 0.84 · AGBI 0.90 vs 0.92 · AGAR 0.81 vs 0.78 ·
  MEGL 0.16 vs 0.22 · MPDR 0.28 vs 0.24 · SVA 0.24 vs 0.30; pooled 0.476 / 0.495 vs 0.624. Isolated-protocol cells: same picture (pooled 0.516 / 0.529;
  AGLT 0.75, MEGL 0.08-0.18, MPDR 0.19). Ordering (TSO), "which did NOT happen" (FAM / FOM) and counting (AC) are exactly the operations
  that need the kept units read JOINTLY and in order; the fence as built (isolated blocks, reset positions, read = prefix + kept blocks)
  loses 0.26-0.42 on them. Caveat: 512 px vs 1 MP and the 09-28 protocol; a matched fenced-on-U* cell (hi, same units) is the follow-up.
- **Retrieval room (P* - official uniform row at 1 MP, N 8 / 16):** TSO +0.58 / +0.46 · FAM +0.28 / +0.26 · FOM +0.28 / +0.28 · ASII +0.10 ·
  AGBI +0.06 · AC -0.06 / +0.08 (the official N = 8 row is the "answer 1" prior: 38 of 50 golds are 1) · SVA -0.14 / -0.08 (more frames than
  the annotated evidence HELP: the annotation is not the sufficient evidence for SVA).
- **D, the model's own yes/no on one unit (pos vs hard negatives, AUC; easy in parentheses):** frame@hi AC 0.67 (0.87) · FAM 0.63 (0.79) ·
  FOM 0.61 (0.71) · ASII 0.60 (0.79) · TSO 0.87 (0.91) · SVA 0.90 (0.92) · AGBI 0.79 · MPDR 0.68 · all 0.711 (0.784). Clips help detection on the
  HD-EPIC actions: AC 0.76 (0.90) at clip3_d1, FAM 0.67 (0.85), FOM 0.67 (0.80), ASII 0.63 (0.83) — and hurt the trailers (TSO 0.64 at clip3,
  shot-boundary frames). lo -> hi: AC 0.62 -> 0.67, all 0.685 -> 0.711. Recall at 99 % specificity <= 0.34 everywhere; yes-rate on true
  evidence units 0.04-0.36: the model's own vote cannot gate at N = 256 (1-4 % positives); a TRAINED gate stays necessary (rule 3).
  (Band "D(frame hi) AUROC ~0.8 on AC": met vs easy 0.87, not vs hard 0.67; "clips ~0.9": vs easy 0.90, hard 0.76.)
- Pre-registered T bands: met except FAM (0.44 from text; band <= 0.2) and SVA 0.30; E(frame hi) FAM / FOM 0.78 / 0.74 exceed the 0.5-0.7 band;
  SVA E 0.30 refutes "distinct shots >= 0.7": SVA is not a shot-recognition task for this model.

**MMReD anchor (Qwen, 50 rows / qtype, one 512 px frame = one unit):** T 0.189 (seq 8) / 0.154 (seq 16) -> E(evidence frames only) **0.640**
(n 1,142; k = 1 0.831, k = 2-4 0.742) / **0.605** (n 1,172) vs the official rows 0.515 / 0.415: retrieval room +0.125 / +0.19 at N <= 16.
Per qtype (seq 8): positional needles 1.00 (char_at_frame, first_app), room_on_char_first_app 0.86, last_at_room 0.675, n_empty 0.56,
who_spend 0.48, where_spend 0.36, steps_in_room 0.32, rooms_visited 0.30, **crowd_count 0.00** (k evidence frames shown, the answer is k:
the model never counts them). D(frame) per-frame judge: char_at_frame / first_at_room / n_empty / steps_in_room AUC 1.00, who_spend 0.99,
crowd_count 0.84 (pooled 0.71 only because calibration differs by qtype). **The anchor contrast: on MMReD the per-frame fact is read
perfectly and the wall is aggregation (counting); on HERBench actions the unit is read at 0.7-0.85 but detected weakly (hard AUC 0.6-0.75),
and the fence costs 0.26-0.42 on top.**

**What this changes for SCALEUP:** (1) method cells on HERBench = TSO, FOM, FAM, AC, ASII with per-task units; SVA / MEGL / MPDR excluded as
perception walls, AGAR / AGLT / AGBI as text-answerable; (2) the read must see the kept units jointly and in order (a second unfenced pass
over the kept units, or kept positions) — the deferred "fenced on U*" cell is now the first method cell, with the 0.26-0.42 fence cost as its
target; (3) resolution stays at 1 MP; (4) the gate must be trained (the model's own vote is too weak at long N); (5) W2 MINERVA and W3
InternVL / Gemma run the same T / E(frame, clip3_d1, clip9_d2 @hi) / D(frame, clip3_d1 @hi) subset. RESULTS.md untouched (no "log this").
- **W2 prep DONE (170855, 3 min):** MINERVA `units_native/` for the pinned 489 rows (`sbatch/lib/splits/qids_minerva_50.txt`: up to 50 per skill with
  regex-parsed evidence and a decodable pool; counterfactual 31, situational 30, state_changes 15, goal 14): 1,443 positive units, 1,236 hard +
  1,440 easy negatives, 3.02 GB (640x360 / 854x480 mirror videos), verify clean.
- **W3 smokes (1 row per task, HERBench units, frame + clip3_d1 @hi):** InternVL3.5-8B (170856, 13 min) runs end to end; its hi mode = dynamic
  tiling IS active (E frame tokens_mean 7,194 ~ 10 tiles x 256 per frame; clip3 19.5 K). Gemma-3-12B (170857, 12 min) runs end to end but its
  hi mode did NOT activate: E frame tokens_mean 909 = one 256-token tile per frame on 16:9 frames (pan-and-scan should crop at ratio >= 1.2).
  Diagnosing the processor path before any Gemma hi cell; Gemma at its default (one tile) is still a valid "lo" row.
- **INCIDENT + FIX (resolution modes on the tile / crop backbones):** `set_resolution` set the image-processor attributes, but the
  processor's chat-template path merges its CLASS-LEVEL defaults over them (Gemma3 `do_pan_and_scan=False`, InternVL `crop_to_patches=True`):
  Gemma's hi mode stayed one tile (smoke 170857) and InternVL's lo mode would have tiled. Fix: `set_resolution` stores the mode's kwargs on the
  processor and `bb.encode` forwards them with every `apply_chat_template` call; `tests/test_backbones.py::test_resolution_modes` now pins the
  image-token counts on a 1920x1080 frame through encode: Qwen 180 / 1,222, InternVL 256 / 2,304, Gemma 256 / 768 (lo / hi). Qwen rows are
  unaffected (max_pixels was always applied). The InternVL lo cells 170865 / 170867 were cancelled before start and resubmitted (170871 / 170872).
  Note for the record: InternVL's default through the chat template IS dynamic tiling (9 tiles on 16:9, 1 tile on the square 512 px MMReD
  renders), so the earlier InternVL MMReD rows are unaffected.
- **W2 + W3 submitted 2026-10-07 (reduced unit set: E frame@lo + {frame, clip3_d1, clip9_d2}@hi — InternVL without clip9_d2, 12 tiles x 45 frames
  does not fit —, D frame@lo + {frame, clip3_d1}@hi; MMReD anchors chained T + E + D per split):** MINERVA x Qwen 170859-170863; HERBench x
  InternVL3.5-8B 170864, 170866, 170868, 170871 (E lo), 170872 (D lo); MMReD x InternVL 170869 / 170870; HERBench x Gemma-3-12B 170873-170877;
  MMReD x Gemma 170878 / 170879. Collect: `experiments/unit_table.py --dataset minerva` and `--backbone internvl3.5-8b | gemma-3-12b`.

## [2026-10-07 09:30] UNIT baseline W2 + W3 COMPLETE (19 jobs, 0 failures, ~14 GPU-h): the W1 picture holds on MINERVA and on two more backbones
Tables: `outputs/scaleup/minerva/qwen2.5-vl-7b/s1u_unit/TABLE.md` (official column = the uniform N=32 reference row on the same rows),
`outputs/scaleup/herbench/{internvl3.5-8b,gemma-3-12b}/s1u_unit/TABLE.md`, `outputs/scaleup/mmred/{internvl3.5-8b,gemma-3-12b}/s1u_unit/TABLE_seq{8,16}.md`.
Reduced unit set (E frame@lo + frame / clip3_d1 / clip9_d2 @hi; InternVL without clip9_d2; D frame@lo + frame / clip3_d1 @hi).

**MINERVA x Qwen (489 pinned rows, regex evidence = lower bound; T on the same rows):** T 0.186 -> E frame lo 0.354 / hi 0.362 / clip3 0.405 /
**clip9 0.421** (official uniform N=32 row on these rows 0.284). Clips HELP here (+0.06 pooled): event_occurrence 0.30 -> 0.52, spatial 0.28 -> 0.42,
temporal 0.28 -> 0.42, object_recognition 0.46 -> 0.52, cause_and_effect 0.35 -> 0.47; reading 0.62 at the frame; counting 0.32 at the frame (clips
0.22-0.26). Walls (P* - T, CI through 0): counterfactual 0.03, numerical 0.06, counting 0.12, goal 0.14; targets: reading 0.44, event_occurrence
0.40, situational 0.37, object_recognition 0.36, state_changes 0.33, spatial 0.30, temporal 0.24, cause_and_effect 0.22. D (own yes/no, hard
negatives) is near chance on MINERVA: pooled 0.56 (frame) -> 0.61 (clip3); reading 0.67; yes-rate on evidence units 0.04-0.26 — the regex
timestamps make hard negatives 2 s from a possibly-approximate time, and the model's own detection cannot be trusted here at all.

**HERBench x InternVL3.5-8B (non-thinking; 532 rows):** T **0.513** (AC 0.58, AGLT 0.90, AGBI 0.94, ASII 0.74 from text) -> E frame lo 0.643 =
hi 0.643, clip3 0.622; pooled wall 0.130 [0.086, 0.173]. Walls: AGLT -0.04, MPDR 0.00, MEGL -0.02, ASII 0.02 (clip3 0.50!), AGBI 0.04, AGAR 0.09;
targets: TSO 0.46 (E 0.78), FOM 0.42 (clip3 0.74), FAM 0.28 (clip3 0.74), SVA 0.20 (T 0.14 -> E 0.34), AC 0.18 (E 0.76-0.80, k = 1 prior).
Resolution gain 0.00 pooled (tiling vs one tile). D: clips lift detection on EVERY task here: hard AUC frame hi -> clip3: TSO 0.82 -> **0.95**,
SVA 0.77 -> 0.92, AC 0.65 -> 0.69, FAM 0.59 -> 0.69, FOM 0.59 -> 0.68, ASII 0.58 -> 0.63; pooled 0.60 -> 0.67.

**HERBench x Gemma-3-12B (532 rows):** T 0.335 -> E frame lo 0.479 / hi 0.496 / clip3 0.492 / clip9 0.496; pooled wall 0.162 [0.116, 0.209]. The
weakest reader of the three (E 0.50 vs 0.63 / 0.64): TSO only 0.28-0.32 with the shots handed over (T 0.04), FAM 0.56, FOM 0.58, AC 0.76 (k = 1 prior),
AGAR 0.91 at clip9, AGLT 0.64 (T 0.48: less text-answerable than Qwen / InternVL). Walls: AGBI 0.06, MEGL 0.08, MPDR -0.02; targets AC 0.50, FOM 0.30,
AGAR 0.28, FAM 0.26, SVA 0.24, TSO 0.24, ASII 0.20, AGLT 0.16. Resolution (pan-and-scan) +0.017 pooled. D: pooled hard 0.58 (frame hi) -> 0.62
(clip3); SVA 0.81, TSO 0.69, AC 0.62-0.65, FAM 0.53-0.60, FOM 0.49-0.56 (chance).

**MMReD anchors on the new backbones (50 rows / qtype):** InternVL T 0.198 / 0.200 -> E(evidence frames) **0.671 / 0.602** (official 0.561 / 0.478);
Gemma T 0.142 / 0.144 -> E **0.599 / 0.575** (official 0.472 / 0.388). crowd_count 0.00 on both at both lengths, steps_in_room 0.42 / 0.32 (IV),
0.21 / 0.19 (Gemma), char_at_frame 0.98 / 0.94 (IV), 1.00 / 0.96 (Gemma). D per-frame judge: steps_in_room AUC 1.00 on both, char_at_frame 0.82-0.93
(IV) / 1.00 (Gemma), crowd_count 0.78-0.84. The anchor contrast holds on all three backbones: per-frame fact readable, counting is the wall.

**Cross-backbone, HERBench (pooled T -> E frame@hi; wall):** Qwen 0.419 -> 0.633 (0.214) · InternVL 0.513 -> 0.643 (0.130) · Gemma 0.335 -> 0.496 (0.162).
Shared walls: MEGL, MPDR (every backbone), AGBI / AGLT (text-answerable on Qwen and InternVL); shared targets: FOM, FAM, AC (k = 1), TSO (Qwen,
InternVL; Gemma cannot order shots at all). SVA is a wall on Qwen only (its T is 0.30; InternVL / Gemma start at 0.06-0.14 and reach 0.30-0.34).
Clips: task-dependent on Qwen (hurt AC / TSO), neutral-to-negative on InternVL's E (ASII collapses to 0.50 at clip3) but strongly positive for
InternVL's detection, positive on MINERVA. The per-task unit U* therefore has to be chosen per (task, backbone) — the plan's rule 2 as written.
Resolution: 0.00 / +0.017 / +0.009 pooled on the three backbones — settled: it is not the wall anywhere.
- Results page (tables, minimal text; W1-W3 incl. fence cost and the per-task verdict matrix): https://claude.ai/artifact/XuHyW2HzEW4niNKXLsVCQk (Tal asked 2026-10-07; private until shared).

## [2026-10-10 21:30] VNBench adopted as a main benchmark: data landed, adapter + prep written, UNIT baseline wave to run overnight (Tal: "research autonomously, clean minimal code, report back")
Plan `docs/VNBENCH_2026-10-09.md` (= `~/.claude/plans/i-think-i-want-abstract-sparrow.md`, approved 2026-10-09). Decisions: all 9 subtasks; try-0 rows
only for the baseline (the official all-four-tries circular score = a later comparability cell); Claude writes every piece.
**Dataset facts verified first-hand** (json in the HF cache + the July 448 px frames of `data/vnbench_cnt`): 5,400 rows = 1,350 videos x 4 option
orders; 9 subtasks x 150 videos; `needle_time` = integer start seconds in the mp4's FINAL timeline; a needle is on screen EXACTLY 1 s from its start
(subtitle visible at t+0.23..t+0.77, absent at t-0.03 and t+1.04 — the July prep's 1.5 s window over-labelled); insert needles lengthen the mp4 by 1 s
each, so the json `length` (the haystack's) is shorter than an insert video; the json `type` has ret_insert1/2 swapped relative to the filenames (the
upstream scorer swaps them back at report time) -> qtype = filename suffix (ret_insert1 = fruit, ret_insert2 = landmark), raw type in meta; official
prompt = the MINERVA template with 4 options (byte-equal, pinned); official letter rule = cut at the first '.', then 'A' in pred / elif 'B' ... (fixed order).
For a 1 s range the stored unit grid keeps offsets -1/0/+1 only: the five unit names collapse to `frame` (midpoint t+0.5) and the 3-frame `clip5_d1`.
**Built (uncommitted on theory-2; 15/15 CPU test files green):** `core/data/videoqa.py` (VNBENCH_TASKS, NEEDLE_DUR, parse_vnbench_evidence ->
ranges [t, t+1], `VideoMCQ.group_of` hook, `VNBench` with group = haystack), `core/data/mcq.py` (vnbench_prompt / vnbench_extract), the registry,
`experiments/prepare_video.py` (rows_vnbench with meta.tries / gt_rule / raw_type; stage_videos_vnbench = unzip + decoded-duration probe asserting every
needle fits; `--tries`), `sbatch/{unit_baseline,prepare_video}.sbatch`, `sbatch/unit_submit.sh`, `sbatch/lib/splits/qids_vnbench_try0.txt`, `data/README.md`.
**Data:** `data/vnbench -> /rg/shocher_prj/lab_data/vnbench/official`: json + the 10.9 GB Drive zip (gdown, 1,350 mp4, 11.0 GB) in `hf/`; rows 1,350 /
150 per qtype, answers A 340 / B 328 / C 349 / D 333; k mean 3.6 (cnt_edit1 / cnt_insert1), 4 (ord*, cnt_edit2), 1 (ret*). Prep job **175136**
(4h_0g: videos -> probe -> pool_native -> verify -> units -> verify_units).
**Pre-registered bands (plan §7):** T ≈ 0.25 everywhere (no text prior; counting options maybe slightly above) -> nothing text-answerable. E(frame@hi):
ret_edit1 ≥ 0.9, ret_insert1 (fruit) ≥ 0.8, ret_insert2 (landmark) 0.6–0.8, ord* ≥ 0.8 (needles shown in order), cnt k = 1 ≈ 1.0 falling past k = 2–3,
cnt_edit2 0.3–0.5. Resolution ≈ 0 except possibly ret_edit1 at lo. Clips ≈ 0 (static needles). D(frame) AUROC ≈ 1.0 vs hard and easy. If the bands
hold, VNBench = "perception perfect, aggregation wall" on real footage; the wall is in k, not N. Try-0 accuracy is an upper bound of the official score.

## [2026-10-11 01:40] VNBench night run COMPLETE (Qwen2.5-VL-7B 4-bit, 1,350 try-0 rows): perception is perfect, the frozen read aggregates only with frames-first + full attention, and aggregation collapses with N even when every needle is in view
Autonomous overnight run (Tal: "research autonomously, clean minimal code, report in the morning"). Plan `docs/VNBENCH_2026-10-09.md`.
Tables: `outputs/scaleup/vnbench/qwen2.5-vl-7b/s1u_unit/TABLE.md` (UNIT seven columns + gaps + by-k), `…/CURVES.md` (accuracy vs N per
subtask / per true k / exact needle-in-view split, every arm; `experiments/curve_table.py`), figure `…/fig/F_vnbench_collapse.{png,pdf,csv}`
(`experiments/figs/vnbench_fig.py`). Ledger `outputs/scaleup/jobs.tsv` (175136-175250); ~14 GPU-h; 0 failures except prep 175136
(the `verify` stage looped over the `evidence` protocol — pre-existing bug since 10-06, fixed, frames were already complete).
**Data (verified):** 1,350 mp4 (320x240: 513, 640x480: 315, 640x360: 180, 1280x720: 72; 10.0–175.6 s), decoded durations confirm `needle_time`
in the mp4's own timeline (121 rows end past the json `length`, 0 past the decoded duration); pool_native 27 GB; units_native 5.5 GB,
3 frames per positive unit (t, t+0.5, t+1: the subtitle is visible at t and t+0.5 and GONE at t+1.0, eyeballed). Negatives dropped on the
dense short videos (cnt/ord hard present 336–371 of 541–600; ret 145–147 of 150). `uniform` all N; `planted` N=8 1,246 (k>N/2) … N=128 1,198
(too few fillers); N=256 impossible (pool = 256).
**UNIT baseline (regimes T / E / D; TABLE.md):** T **0.250** [0.229, 0.273] = chance on every subtask (0.18–0.33) → nothing text-answerable.
E frame lo / hi **0.880 / 0.878** (tokens 515 / 1,082): ret_edit1 1.00, ret_insert1 0.99–1.00, ret_insert2 0.95; ord 0.99 / 0.99 / 0.99;
cnt_edit1 **1.00 (k up to 6)**, cnt_insert1 0.81, cnt_edit2 0.18–0.21 = the only perception wall (count objects inside 4 images, sum 4–16);
by k 0.97 / 0.83 / 0.81. Resolution gain −0.002 (settled, as on the other datasets). E clip5_d1@hi 0.807: the 3-frame unit HURTS counting
(cnt_edit1 0.99→0.79, cnt_insert1 0.81→0.48, k≥5 0.78→0.38) — duplicate views inflate the count (HERBench AC/TSO again). D frame (9,034 units)
per-task AUROC vs hard: ret 1.00 / 1.00 / 0.95, cnt 0.99 / 0.99 / 0.99, ord 0.99 / 0.98 / 0.92; hard = easy (synthetic needles); pooled 0.94,
rec@spec99 0.49 — the model's raw "yes" is rare on order/count questions (yes-rate 0.00–0.24 vs 0.99 on ret), so one global threshold is weak
while the within-task ranking is near-perfect. Derived: wall P*−T 0.57–0.81 everywhere but cnt_edit2 (−0.12); fence cost (E frame hi − planted
oracle gate N=8) ret 0.00–0.03, ord 0.71–0.79, cnt 0.44–0.77; retrieval room vs uniform N=8 0.48 pooled.
**Collapse ladders (lo res, all rows; CURVES.md, figure):**
- `planted` plain (every needle in view): ALL **0.84 / 0.79 / 0.76 / 0.72 / 0.69** at N=8/16/32/64/128. ret flat (1.00 / 1.00 / 0.94→0.85);
  ord 0.97→0.83, 0.93→0.68, 0.86→0.48; cnt_edit1 0.80→0.58, cnt_insert1 0.78→0.39; cnt_edit2 0.23 flat. By k: k=1 0.98→0.93, k=2 0.96→0.73,
  k=3–4 0.73→0.53, k=5–6 0.46 (N=16)→0.27. = the share-law collapse on real footage: error grows with N at fixed k, faster for larger k.
- `uniform` plain (official sampling): ALL **0.40 / 0.49 / 0.58 / 0.67 / 0.68 / 0.65** (N=8…256; 175153 / 175154 / 175240): rises because needles come into view
  (exact in-view split: none 0.25 = chance; every needle 0.93 (236 rows @N=8) → 0.69 (1,301 @N=128) → 0.65 (all 1,350 @N=256)); ordering peaks at N=64 and falls at
  128 (ord_insert2 0.59→0.43); k=5–6 ≤ 0.28 at every N. The report's "coverage" column uses the ±δ label rule and over-counts; use the exact split.
- fenced + oracle gate (`planted`, batched path): **0.53 / 0.49 / 0.49 / 0.49 / 0.53** — flat by construction (the same k blocks at every N) and
  low: ret 1.00 / 0.99 / 0.93, ord = chance (0.20–0.30), cnt 0.18–0.37; by k 0.94 / 0.43 / 0.25 / 0.00.
**Fence-cost decomposition (`planted`, pooled; N=8 / 16 / 32 / 64):** plain (frames first, no fence) 0.837 / 0.795 / 0.755 / 0.717 ·
question-first unfenced 0.589 / 0.482 / 0.374 / 0.359 · fence only (question LAST, all blocks kept, dense masked) 0.499 / 0.459 / 0.436 / 0.372 ·
fenced question-first all kept 0.465 / 0.323 / 0.250 / 0.205 · + oracle gate 0.529 / 0.489 / 0.489 / 0.491. Per task at N=8 (ret | ord | cnt):
plain 1.00/1.00/0.94 | 0.97/0.93/0.86 | 0.80/0.23/0.78; qfirst 0.99/0.99/0.71 | 0.59/0.38/0.39 | 0.33/0.30/0.51; fence-only 1.00/0.99/0.75 |
0.25/0.23/0.23 | 0.35/0.21/0.40; gated 1.00/0.99/0.93 | 0.29/0.20/0.28 | 0.23/0.32/0.37. → the per-block position reset ERASES block order and
count on its own (ordering at chance with the question last); the question-first layout is a second, independent loss that grows with N; the
gate restores retrieval and holds the fenced read flat where the ungated fence collapses (0.47→0.32→0.25). A FROZEN reader cannot aggregate kept
blocks under either; the method's READ step must be trained (on MMReD the LoRA readers do this; VNBench has no train split).
**Pre-registered bands (STATE 2026-10-10):** T ≈ 0.25 ✓; E ret_edit1 ≥ 0.9 ✓ (1.00), fruit ≥ 0.8 ✓ (1.00), landmark 0.6–0.8 ✗ higher (0.95);
ord ≥ 0.8 ✓ (0.99); cnt k=1 ≈ 1.0 ✓ but "falling past k=2–3" ✗ — handed the needle frames, counting subtitles is perfect to k=6 (the MMReD
"never above 2" wall is a haystack effect, not a count-of-frames limit); cnt_edit2 0.3–0.5 ✗ lower (0.2 = wall); resolution ≈ 0 ✓; clips ≈ 0 ✗
(negative on counting); D ≈ 1.0 ✓ per task. Verdict as pre-registered: VNBench = "perception perfect, aggregation wall" on real footage, and the
wall is in k AND in N once the needles are in view.
**All cells landed** (last two: uniform N=256 175240, 5.4 h; fenced question-first N=64 175482, 4.1 h — addenda below).
**Not run / proposed:** the official all-four-tries circular score (rows carry `meta.tries`; `--tries 0 1 2 3` re-prep + uniform N=16/64; 4x
cost; published Qwen2-VL-7B 1 fps 224 px: ret 69.1 / ord 12.4 / cnt 20.0 vs our try-0 N=64: ret 0.89 / ord 0.73 / cnt 0.39 on Qwen2.5-VL);
D on the clip unit; gate_capture per-block probe (needs the fence; the D regime already gives per-task AUC ≥ 0.92 from the model's own vote).
**Slope of error vs log2 N (least squares over N=8…128; CPU, from the eval.csv files):** `planted` plain: k=1 +0.012 / doubling, k=2 +0.061,
k=3–4 +0.050, k=5–6 +0.063; retrieval +0.008, ordering +0.064, counting (edit1, insert1) +0.078. `uniform` plain restricted to the rows with
EVERY needle in view (exact grid check): k=1 +0.010, k=2 +0.084, k=3–4 +0.042, k=5–6 +0.051; retrieval +0.008, ordering +0.055, counting +0.094.
The two protocols agree: once the needles are visible, aggregation error grows ≈ linearly in log N (0.05–0.09 per doubling) while retrieval is
flat (0.01) — the haystack hurts the joint read, not the find.

**Addendum 04:30 — decomposition arms at N=32/64 landed** (175236 ✓, 175250 ✓; 175237 TIMEOUT at N=64 after 600/1,344 rows — rows are
interleaved across subtasks, running acc 0.18; rerun as 175482). Pooled N=8/16/32/64: question-first unfenced 0.589 / 0.482 / 0.374 / 0.359;
fence only (question last, all kept) 0.499 / 0.459 / 0.436 / 0.372; fenced question-first all kept 0.465 / 0.323 / 0.250 / 0.205 (175482). The new fact:
WITHOUT a gate the fence makes even retrieval N-sensitive. k=1 rows: plain 0.98→0.93 (N=8→128); fence-only 0.91 / 0.88 / 0.81 / 0.67; fenced
q-first 0.81 / 0.53 / 0.39 / 0.31; q-first unfenced 0.87 / 0.75 / 0.56 / 0.52. Per task under fence-only: ret_edit1 1.00 at every N,
ret_insert1 0.99→0.76, ret_insert2 0.75→0.25 (the landmark needle, the hardest to read, goes first). Ordering stays at chance (0.18–0.30) and
counting ≤ 0.35 under every fenced arm at every N; k=5–6 0.00–0.02. The oracle gate (same fence, same layout, only the k needle blocks kept)
holds retrieval at 1.00 / 0.99 / 0.93 at every N (ret_insert2 0.93 vs 0.25–0.34 ungated at N=32–64): the ungated fenced collapse is a share
effect of the N−k distractor blocks, which is exactly what the gate removes. So on VNBench the gate's entire contribution is retrieval under
dilution; it adds nothing to ordering or counting because the per-block reset has already erased them. Error per doubling of N on k=1 rows:
plain +0.01, fence-only +0.11 (N=16→64), q-first +0.12 (16→64), fenced q-first +0.17 (8→64).

**Addendum 09:30 — the last two cells.** `uniform` plain N=256 (175240, 5.4 h): **0.646** [0.621, 0.671]; per task ret 1.00 / 0.99 / 0.87,
ord 0.80 / 0.51 / 0.43, cnt 0.60 / 0.30 / 0.32; by k 0.94 / 0.71 / 0.50 / 0.24. At N=256 the official grid samples every needle of every row
(exact check: 1,350 / 1,350), so the official-sampling curve becomes a pure read curve — and it TURNS DOWN: 0.68 (N=128) → 0.65; ordering
0.89→0.80 and 0.69→0.51, cnt_edit1 0.63→0.60, retrieval flat. The every-needle-in-view rows now run 0.93 / 0.83 / 0.79 / 0.75 / 0.69 / 0.65
(N=8→256), the same slope as the planted ladder (0.84→0.69 at N=128): the frozen read's collapse in N is reproduced under the benchmark's own
sampling, with no planting. Family means at N=256 (try-0, single try): ret 0.95 / ord 0.58 / cnt 0.41 — an upper bound of the circular 4-try
score (R2); the published Qwen2-VL-7B row (1 fps, 224 px) is 69.1 / 12.4 / 20.0. Fenced question-first all kept N=64 (175482): **0.205**
[0.184, 0.227]; ordering 0.11–0.16 (below chance = the letter prior), retrieval 0.39 / 0.17 / 0.27, k=1 0.31. Ladder complete:
0.465 / 0.323 / 0.250 / 0.205 vs the oracle-gated 0.529 / 0.489 / 0.489 / 0.491 on the same fence and layout.

## [2026-10-11 15:45] VNBench week 1 COMPLETE except two Gemma N=64 cells (cells 1-6 of the 12-cell program; Tal: "lets start with everything in week 1, i want it done today"): the frozen model ranks its needles but cannot threshold them; a linear gate fitted on half the haystacks at N=16 holds its per-frame accuracy at N=64 and on 7 of 9 unseen tasks; its read equals the oracle's and stays blind to order and count; Gemma shows the same fence cost; InternVL's readout is broken on VNBench
Program table `docs/VNBENCH_2026-10-09.md` ("The program after the morning review") = `~/.claude/plans/i-think-i-want-abstract-sparrow.md`. Ledger `outputs/scaleup/jobs.tsv` (175646-175967); ~41 GPU-h over 17 GPU jobs + 5 CPU fits (two insurance jobs and one cancelled job included, see cell 4). Cells 7-9 (synthetic needles on HERBench / MINERVA footage, the method row, variable N) deferred by Tal; 10-12 week 3 / pending. RESULTS.md untouched.
**Cell 1, self-gate (CPU; `experiments/self_gate.py` on the D regime's stored scores, 9,034 units; `s1u_unit/SELF_GATE_{lo,hi}.md`):** the literal "yes" (one global threshold) keeps 13% of the needles pooled (ret 0.99, ord and cnt recall 0.00-0.24). A within-question relative rule (keep every unit within 2 log-odds of the question's best) keeps 95% and drops 90% of the stored negatives, exact set on 78% of questions; the ranking ceiling (top-k with the true k) is 94% exact. No single margin fits every task (ord_insert2 exact 0.41, cnt_edit2 0.49 at max-2). Ranking is near-perfect; how many to keep is the gap: the label-free gate fails on calibration, not ranking.
**Cell 2 (code):** `evaluate.py --gate-npz` runs a fitted linear gate on the batched fenced path through the existing `keep` callable of `core/fastpath.py` (one forward: the slot state at the gate layer is z-scored, dotted with w, block kept if > 0; per row `n_evid`, `gate_fp`, `gate_fn`; report line `gate: exact set ...`); `gate_fit.py --save-gate / --save-layer / --train-groups-file / --leave-one-task-out`; `tests/test_gate_model.py` pins the rule to the legacy S8/S9 scoring (module = legacy layer - 1, bf16 capture). `--qids-file` now skips (with a `[rows]` line) qids absent from a planted / isolated split, which drops rows it cannot build at large N; strict everywhere else. 15 test files green.
**Cell 3, trained gate (capture 175646: planted N=16 21,600 frames + N=64 86,016 frames, fenced question-first, L12/20/24 slot states, 1h51; fits 175897-175901, 1-2 min each on `4h_0g`; `s2_gate/fits/`):**
- 2-fold by haystack, train N=16 only, test N=16 and 64 (`fold2_trainN16.json`): L20 per-frame acc **0.989**, recall 0.994, spec 0.989, AUC 0.999 (L12 0.974, L24 0.989). Per task the accuracy at N=64 equals the accuracy at N=16 to the third decimal (pooled 0.990 vs 0.989): the gate does not degrade with N; what grows is its false blocks per question (spec 0.97 on the two picture tasks).
- Leave-one-task-out at N=16 (`loto_N16.json`): L12 acc 0.972 / AUC 0.988, AUC 1.000 on 7 of 9 unseen tasks; the picture tasks are the exceptions (cnt_edit2 0.96, ret_insert2 0.99 at L12; at L20 cnt_edit2 recall 0.43, ord_insert2 spec 0.66): the ranking transfers across tasks, the threshold not fully (cell 1's calibration gap, smaller).
- Transfer N16 -> N64 on the same haystacks 0.999 / AUC 1.000 (footage leaks: by-N only). Saved gates `gate_L20_fit{A,B}.npz` (10,800 frames / 75 haystacks each, train acc 1.000; by-group held-out 0.995 / 0.999); L20 kept (legacy deployment layer; L24 ties; L12 = the task-transfer layer).
- **Gated ladder on the held-out fold** (fold A read with fit B, fold B with fit A; 675 rows each; N=16/32/64/128; batched path; `s3_model/native_mp151200/planted_N{N}/gated_fold{A,B}`, pooled by `curve_table.py` with `gated_fold*` -> `s3_model/GATED_MODEL.md`; jobs 175911/175912 for N=16 (+ N=32 fold A), 175966/175967 for the rest after the `--qids-file` fix): pooled **0.48 / 0.48 / 0.48 / 0.51** (1,350 / 1,349 / 1,344 / 1,198 rows) vs oracle 0.49 / 0.49 / 0.49 / 0.53 vs plain 0.79 / 0.76 / 0.72 / 0.69. By k: 0.93 -> 0.91 (k=1), 0.43, 0.25, 0.01 -> 0.00 flat (oracle 0.94 / 0.43 / 0.25 / 0.00). Ordering and counting within 1-2 points of the oracle at every N; retrieval pays for the false blocks: ret_insert1 0.99 -> 0.97 (oracle 0.99), ret_insert2 0.91 -> 0.87 (oracle 0.93), ret_edit1 1.00. Gate stats fold A / B: exact set 0.938 / 0.942 (N=16), 0.923 / 0.926 (32), 0.911 / 0.909 (64), 0.896 / 0.909 (128); mean false blocks 0.16 / 0.11 -> 0.36 / 0.26 -> 0.75 / 0.55 -> 1.18 / 0.81 (~1% of the blocks, linear in N, as spec 0.989 predicts); missed blocks 0.01-0.02 at every N; kept 3.0 -> 3.9 blocks for 2.9 true. The knockout stands on a real gate: fitted once on half the haystacks it reproduces the oracle ladder on the other half, and the fenced read behind either gate is blind to block order and count. Empty run dirs of the failed first attempt (no eval.csv) remain next to the real ones; `curve_table` takes the newest dir that has one.
**Cell 4, backbones (planted N=8/16/64, `--res lo` = one tile per frame, dense path; `vnbench/{gemma-3-12b,internvl3.5-8b}/{s1_eval,s3_oracle}/native/planted_N*/`, CURVES.md per backbone):** Gemma-3-12B parses (parse_fail 0.2-0.4% plain, 5% gated): plain **0.713 / 0.660 / 0.579** (N=8 n=1,246 / N=16 / N=64 n=1,344; by k: k=1 0.96 -> 0.92, k=2 0.86 -> 0.63, k=3-4 0.53 -> 0.36), oracle-gated **0.505 / 0.469 / 0.469** (n=1,344 at N=64, parse_fail 4.8%; by k: k=1 0.97 -> 0.97, k=2 0.35 -> 0.35, k=3-4 0.19 -> 0.19) = the Qwen shape (plain falls with N, the fenced gated read is flat and lower). N=64: 175663 TIMEOUT at 5 h with ~1,075 / 1,344 rows (eval.csv is written at the end; `scontrol` time extension denied) -> standalone 175910 done 16:02; 175664 projected past its 7 h -> standalone 175914 done 17:56 (05:42:57). InternVL3.5-8B is NOT interpretable: plain 0.408 (n=1,246) / 0.323 / 0.445 with parse_fail **0.49 / 0.58 / 0.37**, gated 0.418 / 0.384 (N=8/16) with 0.34 / 0.39: its answers open with prose ("Okay, let's figure out the order", "The image shows a coconut, which is") cut at 8 new tokens, under the non-thinking prefix, whereas on HERBench single-frame units it parsed 100%. The VNBench suffix does not hold it to a letter once there are many frames. Its N=64 gated cell (175666) was cancelled at 12:18 (16 s/row -> ~18:00 vs a 17:11 limit) and not resubmitted. Decision for Tal: a forced-choice readout (letter logits at the first answer token; model-agnostic, parse-free; new `evaluate.py` flag) or an InternVL answer prefix through `assistant_prefix` (core/), then ~8 GPU-h to redo its ladder.
**Cell 5, official circular score (`vnbench4` = all 4 option orders, 5,400 rows; `experiments/vnbench_circular.py`; `outputs/scaleup/vnbench4/qwen2.5-vl-7b/CIRCULAR_N{16,64}.md`):** uniform N=16 (175657, 1h26) overall **0.363** = retrieval 0.569 / ordering 0.291 / counting 0.229 (try-0 only 0.494 = the night's 0.49); uniform N=64 (175658, 5h22) overall **0.556** = 0.844 / 0.518 / 0.304 (try-0 0.668 = the night's 0.67). Published Qwen2-VL-7B (1 fps, 224 px): 69.1 / 12.4 / 20.0 = 33.8. At 64 frames the frozen 4-bit Qwen2.5-VL-7B beats that row by 22 points overall and on every family; the circular rule costs 11 points at N=64 and 13 at N=16 relative to try 0 (the option order moves about one answer in eight).
**Cell 6 (code):** `uniform_labels` is inside-interval for ranges (a <= t < b) and +-delta for points; tests. Only `verify`'s coverage column and the uniform in-view split read these labels.
**What week 1 adds to the story:** (i) the frozen model's own vote ranks needles almost perfectly but has no usable threshold (cells 1 and 3 agree); (ii) a linear gate on the slot state, fitted on half the haystacks at N=16, holds its per-frame accuracy at N=64 and on 7 of 9 unseen tasks: the gate is the cheap, transferable half of the method; (iii) the knockout now stands on that fitted gate, not on labels: 0.48 flat in N vs plain 0.79 -> 0.69, and the fenced read behind either gate cannot order or count (by k 0.93 / 0.43 / 0.25 / 0.01 at every N), so the trained read (cell 8) is the missing piece and the gate is ready for it; (iv) Gemma-3-12B repeats the two Qwen numbers (plain 0.71 -> 0.66 -> 0.58 at N=8/16/64, gated 0.51 -> 0.47 -> 0.47 at N=8/16/64: the fenced oracle read is flat and lower on a second backbone); InternVL is blocked by its readout, not by the mechanism.
Next: the InternVL readout decision; cells 7-9 when Tal releases the HERBench / MINERVA footage for synthetic needles; cells 10-12.

## [2026-10-12 09:30] VNBench week 2 COMPLETE (items 1-3 of "do 1-3 for now"; 8 jobs 177259-177544 + 4 smokes, 33.9 GPU-h on a packed-cluster night): the frozen two-pass read behind the fitted gate holds E's level at every N (0.87-0.88 vs plain 0.84 -> 0.69); the attention photograph shows the needle's share of frame mass falling as 1/N while the gated fence holds it fixed; a forced-choice letter readout unblocks InternVL and reproduces the Qwen shape
Ledger `outputs/scaleup/jobs.tsv` (177259-177544); smokes `outputs/_scratch/week2_smoke/` (176721, 177258, 177349, 177464). Code (uncommitted on `theory-2`): `evaluate.py --two-pass` (fenced scan with the gate -> `kept_sample()` -> `read_unfenced()`: the kept frames, in order, read in the official frames-first layout with full attention; one extra forward over ~3 frames) and `--readout letters` (`LetterReadout`: argmax over the offered letters of the best single-token spelling at the first answer position, `letter_margin` = best - second; `tests/test_readout.py`); `probe_attention.py --dataset vnbench` (M-RoPE q/k recapture at the answer row; MMReD path unchanged: anchor rows 0003834 / 0003841 reproduce the 2026-09-22 masses on 280/280 (qid, layer, head) within 0.022, seq_tokens 2756); `scaleup_cells.sbatch` STAGE=s4 + READOUT, `probe_attention.sbatch` video branch, `sbatch/smoke/smoke_week2.sbatch` (STEPS knob), `common.sh` TORCHINDUCTOR_COMPILE_THREADS=1. RESULTS.md untouched.
**Item 1, two-pass frozen method (cell 8 without training; `s4_method/native_mp151200/planted_N{N}/twopass_{foldA,foldB,oracle}`, pooled table `s4_method/TWOPASS.md`):** fold A rows read behind `gate_L20_fitB.npz`, fold B behind fit A (the week-1 gates, fitted on half the haystacks at N=16); 623 / 675 / 675 / 673 / 604 + 623 / 675 / 674 / 671 / 594 rows at N=8/16/32/64/128. Pooled **0.881 / 0.873 / 0.871 / 0.866 / 0.880** (fold A 0.883 / 0.886 / 0.884 / 0.878 / 0.881, fold B 0.880 / 0.859 / 0.858 / 0.854 / 0.879) vs the two-pass ORACLE (labels as the gate, same read) **0.885 / 0.877 / 0.878 / 0.877 / 0.891**, vs E frame@lo 0.880, vs the plain read 0.837 / 0.795 / 0.755 / 0.717 / 0.688, vs the fenced gated read of week 1 (same gate) 0.48-0.51. By k (fitted / oracle / plain at N=128): k=1 0.96 / 0.97 / 0.93, k=2 0.98 / 0.98 / 0.73, k=3-4 0.82 / 0.83 / 0.53, k=5-6 0.79 / 0.81 / 0.27, flat across N on every stratum. Per task at N=128 (fitted | oracle | plain | E): ret 1.00 / 0.98 / 0.91 | 1.00 / 0.99 / 0.95 | 1.00 / 0.99 / 0.85 | 1.00 / 0.99 / 0.95; ord 0.99 / 0.99 / 0.96 | 0.99 / 0.99 / 0.99 | 0.84 / 0.68 / 0.48 | 0.99 / 0.99 / 0.99; cnt_edit1 0.98 | 1.00 | 0.58 | 1.00; cnt_insert1 0.83 | 0.84 | 0.39 | 0.81; cnt_edit2 0.18 | 0.18 | 0.24 | 0.21. Gate stats (fold A / B): exact set 0.963 / 0.958 (N=8) -> 0.938 / 0.941 -> 0.923 / 0.923 -> 0.911 / 0.906 -> 0.896 / 0.911 (N=128); false frames per question 0.05 / 0.04 -> 1.18 / 0.79 (linear in N, ~1% of the fillers), missed 0.01-0.02 at every N; kept 2.7 -> 3.9 / 3.5 blocks for 2.9 true. Rows split by gate outcome (pooled): exact set **0.886-0.897 at every N** (1,197 -> 1,082 rows), false frames only 0.69-0.75 (32 -> 101 rows = 2.6% -> 8.7%), a missed needle 0.41-0.47 (15-19 rows). The whole gap to the oracle is the false frames, and the read tolerates most of them (a question with a wrong frame among the kept still answers at 0.7).
-> **The read step does not need training when the kept frames are read unfenced in the official layout.** Week 1's "the READ must be trained" was true of the fenced read (per-block position reset, question first): behind the same fitted gate it sits at 0.48 and cannot order or count; the two-pass read of the same kept blocks sits at E. The frozen model aggregates 1-6 frames perfectly (E) and fails only when the N-k fillers share the softmax; on real footage the gate's job is exactly to remove them, and once they are removed nothing has to be learned. Cost: one fenced forward over N frames (the batched gate path, 182 tokens per frame) + one unfenced forward over ~3 frames. Next cells: variant 2 of the read (kept frames at their original positions; `core/`), N=256 and the uniform protocol + circular score (the leaderboard-comparable method row).
**Item 2, attention photograph on real footage (`s5_photo/native_mp151200/planted_N{N}/{plain,gated}/`, 20 rows per task per N = 180 rows per cell, layers 12/16/20/24/27, mass at the answer row split into needle frames / filler frames / prompt + sink; `mass.csv` per (qid, layer, head), `samples.csv` first-token margins):** plain, L20 head-mean pooled over tasks: needle mass **0.030 / 0.019 / 0.011 / 0.008 / 0.005** at N=8/16/32/64/128, filler mass 0.032 / 0.050 / 0.059 / 0.073 / 0.091, prompt + sink 0.90-0.94 throughout. Per frame: a needle frame draws 0.015 -> 0.003, a filler frame 0.0058 -> 0.0007; the per-frame ratio needle / filler at L20 is **2.0 / 1.7 / 2.0 / 2.5 / 2.5** (row medians), so the needle's share of the frame mass is k*r / (k*r + N - k) with r ~ 2: it falls as ~1/N = the share law with e^s ~ 2 at the gate layer. Deeper layers are more selective and grow more so with N (L24 ratio 2.0 -> 5.8, L27 2.3 -> 7.1: at L27 a needle frame keeps 0.023 -> 0.012 while a filler frame falls 0.0083 -> 0.0010); the early layer is not (L12 1.0 -> 2.0). Gated (oracle fence, N=16 and 128): filler mass **exactly 0** at every layer, needle mass L20 **0.371 / 0.369** (L16 0.25, L27 0.19, L12 0.15, L24 0.10) = the same photograph at both N to three decimals (after the per-block reset the kept blocks are identical whatever N was) and 19x / 72x the plain read's needle mass at N=16 / 128; per task it rises with k (ret k=1 0.27-0.32, ord k=4 0.40-0.42, cnt k=6 0.46-0.51). First-token accuracy on the 20-row subsets: plain ord 0.95 / 1.00 / 0.70 (N=8) -> 0.90 / 0.70 / 0.50 (N=128), cnt_edit1 0.65-0.75 flat, ret 1.0; gated ret 1.0 / 1.0 / 0.9, ord and cnt 0.10-0.35 (cnt_edit2 0.50 at N=16) at both N = week 1's knockout on 180 rows. This is the share law's input on real footage (MMReD S10: s=0.30, C=8); fitting (s, C) per layer from `mass.csv` is a proposed CPU step, not done.
**Item 3, letter readout (`--readout letters`):** Qwen control on the dense plain path, N=8/16 (`s1_eval/native_mp151200/planted_N{N}/plain_letters`, 177360; `s1_eval/LETTERS_AGREEMENT.md`): **0.838 / 0.796** vs generated 0.837 / 0.795 on the same rows, row-level agreement **0.996 / 0.993**; letter margin median 4.2 on correct rows vs 0.6 on wrong ones. InternVL3.5-8B `--res lo`, dense path, parse_fail 0 everywhere (`internvl3.5-8b/{s1_eval/native/planted_N{N}/plain_letters,s3_oracle/native/planted_N{N}/gated_letters}`, `CURVES_LETTERS.md`): plain **0.803 / 0.781 / 0.734** at N=8/16/64 (by k 0.90 / 0.80 / 0.73 -> 0.87 / 0.82 / 0.71 / 0.19; per task ret 1.00 / 0.95 / 0.75 -> 1.00 / 0.94 / 0.63, ord 0.98 / 0.97 / 0.85 -> 0.94 / 0.95 / 0.77, cnt_edit1 0.65 flat, cnt_insert1 0.74 -> 0.45, cnt_edit2 0.27), oracle-gated **0.496 / 0.460 / 0.463** (ret 1.00 / 0.95 / 0.66 flat, ord 0.18-0.32 = chance, cnt 0.23-0.37; by k 0.86 / 0.33 / 0.25 / 0.04 at every N). The generated readout (week 1: 0.41 / 0.32 / 0.44 with parse_fail 0.34-0.58) is retired for InternVL. Three backbones now show the same two curves: plain falls with N (Qwen 0.84 -> 0.69, Gemma 0.71 -> 0.58, InternVL 0.80 -> 0.73) and the fenced oracle read is flat, lower (0.49-0.53 / 0.47-0.51 / 0.46-0.50) and blind to order and count.
**Operational:** a packed cluster (60 running + 60 pending on a100-public) saturated `/home` NFS: every python start took ~30 min (imports in state D, `rpc_wait_bit_killable`, GPU at 4 MiB), row loops ran at normal speed; the first smoke timed out in step 3 and the rest ran as STEPS subsets in three more smokes; n306 excluded. `TORCHINDUCTOR_COMPILE_THREADS=1` in `common.sh` removes the 300 s compile-worker shutdown at exit. Of the 33.9 GPU-h roughly 15 were import time.
Next (Tal decides): (a) share-law fit (s, C) per layer from the photo `mass.csv` (CPU); (b) two-pass at N=256 and under the uniform protocol + the circular 4-try score; (c) variant 2 of the read; (d) cells 9-12.

## [2026-10-13 12:30] VNBench week 3 (Tal 2026-10-12 "yeah do all three"; 6 jobs 178004-178014 + 3 smokes, 22.8 GPU-h): the share law holds C-free at the gate layer (e^s = 1.9) but the sink is a fixed-fraction absorber, not a constant weight; the two-pass read under the benchmark's own sampling rises to 0.84 at N=256 where the plain read turns down (0.65) and scores 0.677 circular at N=64 vs 0.556 plain / 33.8 published; variant 2 (kept frames at their original positions) loses what packing gained, most of all on counting (k=5-6 0.79 -> 0.17 at N=128)
Ledger 178004-178014 (177993 / 177994 cancelled: the first variant-2 build leaked GPU memory, decode outside inference_mode; fixed, memory flat 8.07 GB at rows 100 / 400). Code (uncommitted on `theory-2`): `experiments/share_law_fit.py`; `evaluate.py --two-pass-positions {packed,original}` (`Reader` dataclass, `original_index` / `full_prompt_inputs` / `positions_with_tail`, the read through core.fence.greedy_decode with a causal mask; tests in `tests/test_readout.py`); `scaleup_cells.sbatch` POSITIONS knob; `sbatch/smoke/smoke_week3.sbatch`; `sbatch/lib/splits/qids_vnbench4_fold{A,B}.txt`; sbatch skill decoder row. RESULTS.md untouched.
**Cell 1, share-law fit (`s5_photo/SHARE_LAW.md`, CPU on the week-2 cells):** the three-group law evid = k e^s / Z, nonevid = (N - k) / Z, other = C / Z, Z = k e^s + (N - k) + C, fitted per layer on 900 (row, N) observations, FAILS: C lands at 660-2,540 filler-equivalents and the per-N predictions miss (L20 nonevid 0.005 predicted vs 0.032 measured at N=8, 0.114 vs 0.091 at N=128); the gated cells contradict it too (k=1 needle share 0.289 measured vs 0.004 predicted). Reason: prompt + sink keep 0.90-0.94 of the mass at every N (frame mass 0.061 -> 0.096 from N=8 to 128, x1.6 for x16 frames): the sink is a fixed-fraction absorber, not a competitor at a constant weight. The C-free form, the needles' share of the FRAME mass = k e^s / (k e^s + N - k), fits L20 with **e^s = 1.94** (s = 0.66; measured / law 0.501 / 0.482, 0.299 / 0.293, 0.183 / 0.162, 0.109 / 0.084, 0.058 / 0.042 at N=8..128), L16 1.50, L12 1.02 (no selectivity). The deep layers do not follow one s: L24 e^s 2.85, L27 measured 0.146 vs 0.066 law at N=128 (the needle / filler per-frame ratio grows 2.3 -> 7.1 with N; the needle share decays ~N^-0.5 there). On VNBench the gate layer gives a needle frame twice a filler frame's weight; MMReD's S10 (s = 0.30, C = 8) used the three-group accounting that fails here.
**Cell 2a, two-pass under the uniform protocol (the official grid, try 0, fitted gate, folds pooled, 1,350 rows; `s4_method/TWOPASS_UNIFORM.md`; 178004 / 178005):** **0.508 / 0.737 / 0.823 / 0.836** at N=16 / 64 / 128 / 256 vs plain 0.49 / 0.67 / 0.68 / 0.65. Coverage (>= 1 needle frame sampled) 0.67 / 0.95 / 0.99 / 1.00 bounds the low end: at N=16 the gate keeps 1.8 blocks because most needles are out of view and the read sits at plain's level. From N=64 the curve keeps rising while the plain read flattens and turns down; at N=256, every needle in view, 0.836 vs 0.646 (+19), by k 0.97 / 0.96 / 0.76 / 0.63 vs 0.94 / 0.71 / 0.50 / 0.24; per task ret 1.00 / 1.00 / 0.92, ord 0.98 / 0.94 / 0.88 (plain 0.80 / 0.51 / 0.43), cnt_edit1 0.84 (0.60), cnt_insert1 0.72 (0.32), cnt_edit2 0.25 (0.30). The grid's side effect: at 6.4 fps (N=256 on a 40 s video) a 1 s needle covers 6-7 sampled frames and the per-frame gate keeps the run: kept 7.3 / 14.5 / 28.9 blocks for 2.9 needles at N=64 / 128 / 256, exact set 0.82 / 0.66 / 0.51 (false blocks 0.6-3.0 and missed labelled frames 0.1-0.7 per question; a missed frame is harmless when the needle's other frames are kept). Retrieval and ordering tolerate the runs; counting pays: cnt_edit1 0.84 and k=5-6 0.63 vs 0.98 / 0.79 on the planted ladder with one frame per needle. A run-merge rule (one frame per second of kept run; a naive adjacent merge would swallow back-to-back needles 1 s apart) is Tal's call.
**Cell 2b, the circular four-try score (all option orders, uniform N=64, folds pooled, 5,400 rows; `outputs/scaleup/vnbench4/qwen2.5-vl-7b/CIRCULAR_TWOPASS_N64.md`, pooled csv next to the fold dirs; 178006 / 178007):** **0.677** = retrieval 0.849 / ordering 0.744 / counting 0.438 (try 0 alone 0.737) vs the plain read 0.556 = 0.844 / 0.518 / 0.304 (week 1) vs published Qwen2-VL-7B 33.8 = 69.1 / 12.4 / 20.0. Per task circular: ret 0.867 / 0.880 / 0.800, ord 0.787 / 0.760 / 0.687 (plain 0.727 / 0.560 / 0.267), cnt 0.667 / 0.547 / 0.100 (plain 0.453 / 0.333 / 0.127). Gate on the four-try rows: exact 0.82 / 0.80, kept 7.3. The frozen model + the fitted gate + the plain read of the kept frames is +12 points over the same model's dense read and twice the published row; ordering gains most (+23). N=256 circular (~26 GPU-h) not run, pending the merge-rule decision.
**Cell 3, variant 2 of the read (`--two-pass-positions original`: the same kept frames under the same causal attention, every token at the position it had in the full N-frame prompt; `s4_method/TWOPASS_POSITIONS.md`; 178013 / 178014; planted N=8..128, folds pooled):** **0.880 / 0.862 / 0.842 / 0.820 / 0.819** vs packed (variant 1) 0.881 / 0.873 / 0.871 / 0.866 / 0.880 vs plain 0.837 / 0.795 / 0.755 / 0.717 / 0.688. k=1 unchanged (0.97 at every N), k=3-4 0.81 -> 0.76 (packed 0.82), **k=5-6 0.74 / 0.48 / 0.27 / 0.17** at N=16 / 32 / 64 / 128 (packed 0.78 / 0.78 / 0.76 / 0.79; plain 0.46 / 0.35 / 0.29 / 0.27). Per task at N=128: cnt_edit1 0.79 (packed 0.98), cnt_insert1 0.56 (0.83), ord 0.97 / 0.95 / 0.90 (0.99 / 0.99 / 0.96), retrieval unchanged. With ZERO filler tokens in the context, spreading the kept frames to their original positions (and the question to its original distance) reproduces the high-k counting collapse of the plain read; at k=5-6 it is as bad as the dense read of all N frames. Packing is what the two-pass read gains: the collapse of counting in N rides on the positional spread of the needle frames (M-RoPE distance between the frames and to the question), not only on the fillers' attention mass. Caveat: positions with empty gaps are out of distribution in their own way (no training prompt has them), so variant 2 bounds the positional effect from above; the clean split (packed frames with the question at its far position; spread frames with the question adjacent) is a cheap follow-up (~3 GPU-h per arm).
**Operational:** the first variant-2 build ran core.fence.greedy_decode outside torch.inference_mode(): GPU memory grew ~30 MB per row (OOM after ~200 rows at N=16 in the smoke; 23-25 GB after 575 rows at N=8 in the real jobs, cancelled), fixed and probed (8,068 -> 8,072 MiB between rows 100 and 400); decoder row in the sbatch skill. `--qids-file` pins the rows whatever --per-qtype says, so smokes with a fold file read 9-row files under outputs/_scratch/week3_smoke/.
Next (Tal): N=256 circular (as is, or behind a run-merge rule); the positions follow-up; cells 9-12.

## [2026-10-14 17:30] Gate design after the peer meeting: three CPU transfer fits (181599-181601, 4h_0g, minutes each) + the design brief `docs/GATE_DESIGN_2026-10-14.md`
Tal (peer meeting 2026-10-14): the gate is supervised and per frame; how do Sparse RAG and others train theirs; does the gate need the 7B;
how to handle relevance that depends on another frame; fine-tune the reader or not; the professor's one-pass soft significance vector.
Eight web-verified literature lenses + a six-proposal design panel with a hostile critique each (`docs/gate_design_2026-10-14/`); verdict and
ladder in the brief. RESULTS.md untouched. The one new measurement, the cross-dataset transfer of the existing L20 probe (no refit, `gate_fit.py
--train-runs/--test-runs`, group = video):
- **VNBench (planted N=16, 21,600 frames, construction labels) -> HERBench isolated N=8..128 (human intervals, real footage, 81,512 frames)**
  `outputs/scaleup/herbench/qwen2.5-vl-7b/s2_fit/vnbench_to_herbench.json`: L20 pooled AUC 0.712; **TSO 0.873 / SVA 0.879, flat at every N
  (0.871-0.888)**, specificity 0.77-0.79 / recall 0.80-0.82 at the trained threshold; AGAR/AGBI/AGLT 0.55-0.57, MEGL/MPDR 0.50 (as in-domain).
  In-domain 5-fold reference 0.91 (TSO/SVA); question-blind 0.53-0.57; chance 0.50. A splice/overlay detector would sit at chance here.
- **HERBench planted N=8+16 (12,624 frames; includes an 80-frame 09-25 smoke capture) -> VNBench planted N=16/64 (107,616 frames)**
  `outputs/scaleup/vnbench/qwen2.5-vl-7b/s2_gate/fits/herbench_to_vnbench.json`: L20 pooled AUC 0.821; subtitle needles cnt_edit1 0.989,
  ord_edit1 1.000, ret_edit1 1.000; inserted pictures ret_insert1 0.664, ret_insert2 0.684, ord_insert1 0.834, ord_insert2 0.778,
  cnt_insert1 0.774; cnt_edit2 0.593; specificity 0.63-0.98 / recall 0.20-1.00 by task (the threshold does not transfer across needle kinds).
  L24: ret_insert1 0.998, ret_insert2 0.901, cnt_insert1 0.938 (pictures read better one layer deeper).
- **MMReD diag N=8+16 train (28,800 frames, renders) -> VNBench N=16/64**: `.../fits/mmred_to_vnbench.json`: L20 pooled 0.587 (edit needles
  0.91 / 0.98 / 0.62; inserted pictures 0.23-0.51); L12 everything "yes" (spec 0.00).
Reading: the probe's direction is question-conditioned relevance (it transfers to human evidence on footage with no splices and from real
evidence to synthetic edits at AUC 1.0), ranking transfers across footage and needle kinds, the threshold does not (LOTO within VNBench already
showed spec 0.66-0.90 at AUC ~1). Same pattern as the label-free rankers: ranking is cheap, calibration is what the labels buy. Caveat:
VNBench captures at max_pixels 151,200, HERBench at native resolution.
Also re-run today (CPU, `outputs/_scratch/gate_design_1014/nav_sim.py` on the stored D-regime verdict scores, 9,034 units): the panel's
null-anchored midpoint rule on the model's own yes/no gives pooled recall 0.89 / rejection 0.95 / exact set 0.77 / false k=0 on 8% of questions
(ord_insert1/2 recall 0.69, 26-28% false k=0; 2.5-6 leaked haystack frames per question at N=128): the same ROC as the max-2 rule of cell 1.
**Design verdict (brief section 3):** keep the pointwise L20 probe + hard threshold + merge rule + frozen packed re-read; add the label-source
table, kept-frame accounting (false frames per question, exact-set rate, leak curve), the per-task relational table, and four cheap cells
(VNBench fenced_qlast control 0.5 GPU-h; judge recapture + reader-derived labels on the same probe 0.5 GPU-h; 3B fenced-slot probe 2 GPU-h;
time-boxed anchored relational cell 3 GPU-h); reader stays frozen, LoRA reader = labelled row, enumeration-format counting cell for the
ceiling. The professor's one-pass soft vector is refuted ingredient by ingredient by D4 (0.51 / 0.84 / 0.975), SOFTGATE and the week-3
positions ablation. Nothing submitted beyond the three CPU fits; GPU cells await Tal's OK.

## [2026-10-14 19:00] Week-3 cell 4 (read-geometry ablations, 181245-181254) and the Qwen2.5-VL-32B counting ceiling (181256) landed
**Read geometry (same fitted gate, same kept frames, folds pooled, planted; `s4_method/native_mp151200/planted_N{16,64,128}/twopass_*`):**
packed two-pass (reference) 0.873 / 0.866 / 0.880 at N=16/64/128; **fenced read of the packed kept frames (no cross-frame attention) 0.473 / 0.470 / 0.507**;
block-0 positions + fenced 0.488 / 0.486 / 0.523; + timestamp captions 0.491 / 0.491 / 0.525; packed frames with the question at its far
original distance 0.882 / 0.859 / 0.858; frames spread to their original positions with the question adjacent 0.858 / 0.827 / 0.819
(= variant 2's 0.862 / 0.820 / 0.819). Per task at N=128 (ord x3 | cnt_edit1, cnt_insert1): fenced 0.26 / 0.17 / 0.25 | 0.28 / 0.28;
packed-far-question 0.98 / 0.95 / 0.92 | 0.91 / 0.73; spread-adjacent 0.99 / 0.97 / 0.89 | 0.79 / 0.53; packed 0.99 / 0.99 / 0.96 | 0.98 / 0.83;
retrieval 0.89-1.00 under every geometry. Reading: ordering and counting need the kept frames to ATTEND TO EACH OTHER (the fence alone,
with packed distinct positions, puts ordering at chance and counting at 0.28; timestamps do not rescue it) AND to sit at packed
frame-to-frame distances (spreading costs counting 0.83 -> 0.53 on inserted images); the question's distance costs little. The week-1
attribution of the fenced read's blindness to the position reset alone is corrected: the fence (no cross-frame attention) is the first
cause, the positional spread the second. Any one-forward variant of the method must give the kept frames both; the two-pass read is the
minimal form (N question-conditioned block forwards + one joint O(k) read).
**32B counting ceiling (E regime, needle frames only, lo res, 450 rows; `vnbench/qwen2.5-vl-32b/s1u_unit/E_frame_lo/`, 14 min):**
overall 0.762: cnt_edit1 0.993, **cnt_insert1 0.953 (7B 0.81)**, cnt_edit2 0.340 (7B 0.18-0.21); by k: k=1 0.955, **k>=5 0.971 (7B k=5-6
0.58-0.69)**. The inserted-image counting wall moves with reader size; counting objects inside images barely does. A 32B two-pass would
lift the counting family to roughly 0.76 try-0 on these three tasks, not to the 0.9s, unless cnt_edit2 is addressed. RESULTS.md untouched.
