# MMReD faithful-benchmark campaign — STATE

> Handoff file. Updated after every phase/gate. Brief: [CAMPAIGN_BRIEF.md](CAMPAIGN_BRIEF.md)
> Started: 2026-08-01. Agent session in tmux `mmred`.

## Status: Phase 3 — formats + trainer (IN PROGRESS)

| Phase | State | Gate |
|-------|-------|------|
| 0 acquisition/fidelity/triage | **DONE — GO** | anchor well above chance, ordering plausible (paper-table check deferred: OpenReview walled) |
| 1 diagnostic ladder | **DONE — GO** | steps d′ 7.98 (2.80×) PASS; first_at_room d′ 6.21 (2.52×) PASS; tally 0.992 PASS |
| 2 carrier | **SKIPPED (reuse)** | carrier = 81% of domain-matched teacher (> ~70% bar) |
| 3 formats + trainer | in progress | — |
| 4 eval grid | not started | — |
| 5 safety cells | not started | — |

## Phase 0 record

### Step 1 — upstream repo
- Cloned to `data/mmred_hf/upstream_repo` (untracked), commit **56c6ee7041d539c7273d42d5a6c4c5e922015e40** (2026-06-12).
- **License: NONE in the repo** (no LICENSE file; pyproject has no license field). Flag for Tal before any redistribution; internal research use only for now.
- Authors: Kurkin & Shirokikh. Repo = generator + renderer + their own train/eval stack (SFT/RMT/GRPO, vLLM, MERA integration).

### Step 2 — venv (DEVIATION, mechanical)
- Upstream declares `requires-python >= 3.10` and uses `str | Path` annotations that crash python 3.9 at import. Brief said `python3.9 -m venv`; impossible.
- `.venv_mmred/` built instead from the cluster module `python/3.11.15-x86-7sp4zgy` (system module — no shared-env touch; same isolation intent). Installed: `-e upstream_repo[viz]`, `datasets 5.0.1`, pillow. Imports verified.
- Shared `.venv` untouched.

### Step 3 — HF dataset inventory (`ef1e43ce/mmred`, Arrow cache at `data/mmred_hf/hf/`, 44 MB)
- Configs: `default` + `seq_len_{1,2,4,8,16,32,64,128}`.
- **seq_len 1–16: train 4800 / val 1200 / test 1200. seq_len 32/64/128: test 1200 ONLY** (no train/val — consistent with the train-short/eval-long protocol; "val" tuning only possible at ≤16).
- 24 qtypes everywhere, exactly balanced: 200/qtype in train, 50/qtype in val/test.
- Row schema: `{question, seq_len, answer(str), qid, qtype, atype}` — the `question` column packs the full step context (one python-dict line per step: `{'step_id': k, 'rooms': {room: [chars]}}`, all 6 rooms explicit) + the actual question as the last line. **No relevance metadata in HF** — evidence must be re-derived from states per qtype (their generator's relevant_map logic is in `mmred/qgen/questions.py`; deterministic).
- Rooms: Kitchen, Bathroom, Garden, Office, Bedroom, Hallway. Chars: Sandra, Mary, John, Daniel, Michael. atype ∈ {room, person, number}.
- 24 qtypes = **15 NIAH + 9 DC** (brief guessed 14+10; actuals): NIAH = first_app, final_app, char_on_char_{first,final}_app, char_at_frame, first_at_room, last_at_room, room_on_char_{first,final}_app, room_at_frame, char_on_char_at_frame, n_room_on_char_{first,final}_app, n_char_at_frame, n_empty. DC = room_empty, where_spend, crowded_room, who_spend, spend_alone, spend_together, steps_in_room, rooms_visited, crowd_count.

### Step 4 — render probe (login node, 20 samples seq8 ≈ seconds)
- Wrote `scripts/mmred_hf_prep.py` (runs in `.venv_mmred`): HF row → upstream-renderer JSON (`data/mmred_hf/json/<config>_<split>.json`); asserts step_id contiguity and seq_len match.
- Renderer = upstream `scripts/render_images.py` (matplotlib, 512×512, `frame_%04d.png` per qid dir).
- **~19.8 KB/frame**; 160 frames in 3.5 s wall (login, process pool).
- **Determinism: PASS** — two renders of the same JSON, md5-of-md5s identical.
- Disk projection (render ALL needed: train 2/4/8/16 full, val 8/16, test 8/16/32/64/128): ≈ 470k frames ≈ **~10 GB** → well under the 60 GB stop line. GO.

### Step 5 — full renders (INCIDENT + resolution)
- **Incident:** first submission (jobs 127752–127763, upstream `render_images.py`) hung 10 min
  with zero output — upstream uses `ProcessPoolExecutor()` with default workers = the node's
  FULL core count, ignoring the 8-CPU cgroup; 10 jobs × ~dozens of workers stuck in NFS waits
  (`rpc_wait_bit_killable`/`d_alloc_parallel`, shared matplotlib font-cache). **All cancelled.**
- Fix: `scripts/mmred_hf/render.py` — workers capped at `SLURM_CPUS_PER_TASK`, node-local
  `MPLCONFIGDIR`, resume-safe (skips qids with complete frame sets). Wrapper
  `slurm/mmred_hf_render.sbatch` now calls it.
- Wave 1 (eval splits) jobs 127765–127770; **8_test / 8_val / 16_test complete (1200 qids each)
  within ~3 min each**. Wave 2 (train 2/4/8/16 + 16_val) jobs 127771–127775.
- Layout: `data/mmred_hf/images/<config>_<split>/<qid>/frame_%04d.png`.

### Step 6 — task triage (all 24 qtypes)
Schema = per-frame fact as a short token phrase + textual reduction (caption-scan scratchpad).

| tier | qtypes | scan shape | verdict |
|------|--------|-----------|---------|
| NIAH single-frame readout (15) | first_app, final_app, char_at_frame, char_on_char_{first,final}_app, first_at_room, last_at_room, room_on_char_{first,final}_app, room_at_frame, char_on_char_at_frame, n_room_on_char_{first,final}_app, n_char_at_frame, n_empty | per-frame fact slot + first/last/positional select | **IN (as-is)** |
| DC single running tally (3) | steps_in_room, crowd_count, rooms_visited | one running counter (rooms_visited: seen-set as per-room flags) | **IN (core)** — steps_in_room is the direct analog of our steps task |
| DC multi-counter + arg-max/min (6) | where_spend, who_spend, spend_alone, spend_together, room_empty, crowded_room | 5–6 parallel running counters + final argmax/argmin line | **IN (schema-compliant)** — honesty note: heavier scan; trainability risk flagged, not schema failure |

- **None of the 24 are out-of-schema** (brief anticipated declaring some out; the published data
  simplifies things: NO range-restricted variants — `fraction=1`, full range always).
  `least/fewer` variants are ~27% of DC rows (argmin instead of argmax; ties excluded by generator).
- Published wordings differ from repo HEAD for 3 qtypes (last_at_room "last to appear",
  who_spend "time alone in the <R>" [computation = occupancy count], crowd_count "How many
  times did a crowd appear" [computation = steps count]) — adapter accepts both; parity proves semantics.

### Step 7 — adapter + parity test: DONE, PERFECT PARITY
- `gnnformer/mmred_hf.py` (new module; core untouched): `load_index` / `row_states` /
  `load_mmred_hf_sample` → (sample_id, frames[PIL], question, states, answer); states in our
  `{"rooms": {room:[chars]}}` convention. `recompute_answer` re-derives gold for ALL 24 qtypes;
  `probe_evidence_mmred` gives (evidence_set, locus) for steps_in_room + 7 NIAH types.
  Dispatch by `qtype` column (several templates are prefix-ambiguous).
- `tests/test_mmred_hf_adapter.py`: **2400/2400 answers reproduced exactly (seq8+seq16 test,
  100/qtype, 0 mismatches)**; probe-evidence consistency (982 samples); frame loading OK.
- Full CPU suite `tests/test_*.py`: **ALL PASS** (fencing parity untouched).

### Step 8 — fidelity anchor: BLOCKED on paper numbers, proceeding with eval
- OpenReview (forum, PDF, api2) is Cloudflare-challenge-walled from the cluster; ICLR page has
  no PDF link; no arXiv version found. **Paper-table comparison deferred — Tal: please grab the
  PDF of H6fM44DOHP manually when convenient.**
- Qualitative claims recovered (ICLR abstract): consistent drop with N; reasoning models hit 0%
  on some tasks at N=128; their SFT used seq 1–16 (matches our protocol reading). MERA issue #22
  says benchmark data license = Apache-2.0 (repo itself still has NO license file).
- Gate will be judged vs chance levels (room 1/6, person ~1/6 incl. Nobody, number task-dep.)
  + qualitative plausibility; paper-band check retrofitted when the PDF lands.
- `scripts/mmred_hf/eval_frozen.py` written (arm A): mirrors the paper's image protocol —
  their SYSTEM_PROMPT verbatim (JSON {"answer": X} format), frames + question, greedy decode,
  JSON/Answer:/bare-value parse ladder, per-qtype + per-atype acc + parse-fail rate.

### Step 8 — fidelity anchor RESULT (job 127776, a100-public, seq8 test, 34/qtype):

| qtype | frozen acc | chance | verdict |
|-------|-----------|--------|---------|
| final_app (NIAH) | **0.765** | 0.167 | far above chance |
| steps_in_room (DC tally) | **0.559** | ~0.2–0.3 (majority count) | above chance |
| where_spend (DC argmax) | **0.353** | 0.167 | above chance |
| overall | 0.559 | — | — |

- Run dir: `outputs/mmred_hf/frozen/seq_len_8_test_127776/<ts>/` (report.txt + per_sample.csv).
- Difficulty ordering NIAH > DC-tally > DC-argmax matches the paper's qualitative story.
  Nothing near-zero → adapter not suspect. **GATE: GO.**
- Instrument note: `parse_fail=1.000` = the 7B never emits the instructed JSON format;
  all answers recovered by the closed-vocab fallback (metric = format-fallback rate, not
  extraction failure — extraction clearly works given the accuracies). Relabel later.
- Paper-band comparison DEFERRED (OpenReview Cloudflare-walled from cluster; forum+PDF+api2
  all blocked). **Tal: please fetch the PDF of H6fM44DOHP manually.**

## Phase 1 record — diagnostic ladder (seq 8, their renders)

Plan: tasks = steps_in_room (steps analog; question text matches our _STEPS_RE verbatim)
+ first_at_room (NIAH-style, single-evidence-frame, room-word locus). Probe data from the
seq_len_8 TRAIN split (200/qtype; val/test have only 50/qtype and stay clean for reporting).
- L1 supply probe (A3 arms, limit 150, L16) — GO bar: fenced d′ ≥ 4 AND ≥ 2× joint anchor.
- Transfer cell: carrier_token_room_k1_best.pt eval-only on their renders (% of teacher).
- L2 gate→tally on L1 caches — GO bar: ≥ 0.9.

IN FLIGHT (2026-08-01, a100-public):
- 127778 probe_supply steps_in_room (task=steps native; A3 arms, n=150, L16) → `probe/steps_in_room_seq8/`
- 127779 probe_supply first_at_room (NEW --task mmred_niah path; 174 usable of 200,
  26 Nobody-no-evidence correctly skipped) → `probe/first_at_room_seq8/`
- 127780 transfer cell: train_carrier_token --eval-only, carrier_token_room_k1_best.pt,
  steps_in_room dirs, n=150 (in-run scale-matched teacher anchor gives %-of-teacher) → `transfer/token_seq8/`
- Data: `data/mmred_hf/dirs/seq_len_8_train_{steps_in_room,first_at_room}/` (200 each,
  hardlinked frames; materialize_dirs.py round-trip-verified 50/50 through gnnformer.data).
- probe_supply gained `--task mmred_niah` (additive; steps/cooc defaults untouched; commit 0f3680d).
- L2 gate_tally queues on 127778's messages_cache.pt when it lands.

RESULTS SO FAR:
- Renders: ALL 11 config×splits complete (28.2k qid dirs, 9.1 GB — on projection).
- **Transfer cell (127780, DONE)**: carrier d′ on their renders **6.47±0.52** vs 11.45
  in-domain = **57% → below the ~70% bar → Phase 2 re-distill indicated** (final call after
  probe teacher-on-their-domain lands). Fresh logistic on the same states: per-frame err
  0.0037, tally exact **0.981±0.025** — info survives; amplitude/head is what's domain-bound
  (the predicted domain-bound-detection effect). ckpt-head zero-shot tally 0.573.
  Run: `transfer/token_seq8/20260801_171341_distill_room_k1/`. n=150, skip=18.
- NOTE their gold prior at seq8 steps_in_room is skewed (g0:86/150, then tail) — NOT our
  all_uniform prior; keep in mind for every eval readout + trainer mixture.
- **L1 probes (127778/127779, DONE)**, A3 arms, n=150, L16:

| task | replica d′ | joint anchor | ratio | GO bar (≥4 & ≥2×) |
|------|-----------|--------------|-------|--------------------|
| steps_in_room | **7.98±0.54** | 2.85 | **2.80×** | **PASS** (per-copy uniform 5.4–6.3) |
| first_at_room (single-answer-frame labels) | 0.98 | 0.67 | 1.46× | fail — instrument error, see below |

- **Diagnosis (first_at_room):** labels marked only the FIRST-occupied frame as evidence,
  but first-ness is a GLOBAL property — locally identical occupied frames at later t were
  labeled negative. Smoking gun: per-copy d′ index0 = 4.33 (where occupied⇔first-occupied)
  vs ~0.2–1.5 elsewhere. Supply must be probed on the LOCAL fact (room occupancy); the
  first/last selection is the readout's job. `probe_evidence_mmred` fixed to local-fact
  labels for all room-NIAH types (room_at_frame now returns None — positional, not a
  content-supply task); parity test updated; suite ALL PASS.
- **RERUN result (127784, DONE): first_at_room local-occupancy labels — replica d′
  6.21±0.26, joint anchor 2.46±0.24, ratio 2.52× → PASS (bar ≥4 & ≥2×).** Per-copy d′
  now uniform 4.18–5.55 (was 4.33 vs ~0.2–1.5) — confirms the earlier fail was label
  error, not supply. Run: `probe/first_at_room_seq8_localfact/20260801_172433/` (n=150,
  skip=23). L2 gate_tally on steps_in_room cache: 127785.
- **L2 gate→tally (127785, DONE): tally exact 0.992±0.016 (bar ≥0.9) → PASS.**
  Per-frame err 0.0010; majority baseline 0.573; per-count near-perfect incl. tail
  (g8: 5/5). Run: `gate_tally/steps_in_room_seq8/`.
- **Phase 2 decision — REUSE the existing carrier (re-distill SKIPPED):** carrier 6.47 =
  **81% of the domain-matched teacher** (7.98 on their renders, from probe 127778) — above
  the ~70% bar. The 57% figure vs the PARK teacher (11.45/13.5 scale) conflates the
  teacher's own visual-domain drop (13.5→7.98) with carrier binding; the carrier tracks its
  teacher proportionally, and the carrier-state tally (0.981 fresh-logistic) already clears
  the L2 bar. If Tal prefers the conservative reading (re-distill on their renders anyway,
  ~2–4 h), say so — data+trainer are ready (steps_in_room dirs materialized).
- **PHASE 1 VERDICT: GO on all gates** (L1 both tasks PASS, L2 tally 0.992 PASS,
  transfer 81%-of-teacher → carrier reused). Note: the L2 tally readout is defined for
  numeric golds only, so the ≥0.9 bar is judged on steps_in_room; first_at_room's L1 d′
  is its gate. Proceeding to Phase 3.

## Phase 3 record — formats + trainer

### 3.1 formats.md — DONE
- `outputs/mmred_hf/formats.md`: five families (A tally / B distinct-set / C positional-
  boundary / D conditional-retrieval-with-`*` / E multi-counter argmax-argmin), shared
  grammar `scan: f1:.. fN:.. | <total:|answer:|max:|min:> .. END`, decode budgets
  (6 tok×N; 16 tok×N family E), Tal's refinements folded in (`+`-joins, counter
  abbreviations k/b/g/o/be/h + da/jo/ma/mi/sa, LOCAL argmax read off the last slot's
  counter block, crowded_room always max).

### 3.2 mixture — DONE
- All 96 per-qtype train dirs materialized (`slurm/mmred_hf_materialize.sbatch`,
  4 configs × 24 qtypes × 200) under `data/mmred_hf/dirs/`.
- `slurm/lib/roots_mmred_hf.txt`: 96 roots with `=60` per-root caps → 5760 samples
  (comma-safe via new ROOTS_FILE wrapper knob; per-qtype roots make the caps exact).
- Scan builder `gnnformer/mmred_hf.py::build_scan_mmred` + `parse_answer_mmred`:
  **round-trip test: 2400/2400 published answers reproduced from generated gold scans,
  all 24 qtypes** (tests/test_mmred_hf_adapter.py, suite ALL PASS).
- `train_carrier_layer.py --task mmred_hf`: dir-name qtype dispatch, word golds allowed,
  scan targets from build_scan_mmred; park path unchanged. Commit d7f9837.

### 3.3 trainer — smoke first, then full
- SMOKE 1 (127798) FAILED at prefix-cache build: SDPA "bf16 attn_mask + fp32 query".
  **Park CONTROL smoke (127799) failed IDENTICALLY → latent refactor-port bug in
  `engine.py::build_training_cache`, NOT an mmred_hf issue** (the refactored trainer's
  prefix-cache path had never run on GPU; canonical caption ckpts predate the port,
  living in outputs_legacy). Root cause: runtime loads without torch_dtype, so
  unquantized norms are fp32 → queries upcast to fp32; the lo-mask was cast to
  emb.dtype (bf16) — the ONE deviation from the fp32-mask idiom used everywhere else
  in engine.py (decode, EFFICIENT, top_hidden all already fp32 with a comment).
  Fixed to fp32 (commit after d7f9837); full CPU suite ALL PASS. One root cause,
  one retry — not counting toward the trainer-fails-twice stop (that's for the full run).
- SMOKE 2 (127800): **PASS mechanically** — 40/40 prepped 0 skips (scan targets built for
  all families incl. word golds), 1 epoch trained (loss 2.37, 18 s/ep), ckpt saved.
  acc 0.000 expected at this scale (validation smoke, not a training run).
- FULL RUN attempt 1: job 127802 (a100-public, 24h_1g) — prep 5760/5760 0 skips, cache
  66.2 GB, ep0 0.000 baseline, **[ep 1] TF-acc 0.865 MAE 0.11** (crowd_count 1.00,
  rooms_visited 1.00, final_app 0.99; laggards room_empty 0.60, n_room_on_char_* 0.66 —
  family E as predicted). **KILLED BY WALLTIME at 2:00:18 mid-epoch-2** — infra failure #1.
- **ROOT CAUSE (new cluster gotcha, memorized): ALL partitions have DefaultTime=02:00:00;
  QOS only caps walltime, never raises the request. Long jobs MUST pass `sbatch --time`.**
  Same kill hit arm-A seq128 (127803, 5/24 qtypes lost).
- **FULL RUN attempt 2 in flight: job 127821** (l40s-shared, 24h_1g, --time=12:00:00),
  identical knobs → `outputs/mmred_hf/train/mix5760/`. Arm-A seq128 resubmitted as
  **127822** (rtx6k, 24h_1g, --time=16:00:00). One more trainer infra failure → stop
  and ask Tal per the brief.

## Phase 4 record — eval grid

- Tal 2026-08-01: full autonomy granted through the whole-dataset eval.
- **Arm A (frozen) FULL GRID launched** (no ckpt dependency): jobs 127803–127809 =
  test seq {128,64,32} on a100/24h_1g + test {8,16} on rtx6k/2h_2g + val {8,16} on
  rtx6k/12h_4g. All 24 qtypes × 50/qtype (= entire published splits), DEC=32,
  paper-protocol prompt → `outputs/mmred_hf/frozen/grid_seq<N>_<split>/`.
- Eval dirs for arms B/C materializing (all qtypes, test 8–128 + val 8/16) →
  `data/mmred_hf/dirs/seq_len_<N>_<split>/`.
- Arm C (deployed) + arm B (state-dump → gate_tally, steps_in_room) queue on the
  trained ckpt. eval_carrier gained --task mmred_hf (commit 6baaff1).
- All 7 eval configs materialized (1200 dirs each, 0 missing) + 14 NIAH/DC dirs-files
  under `data/mmred_hf/dirsfiles/` — arm B/C fully staged.

### Arm A results (as they land)

| config | overall | number | person | room | notes |
|--------|---------|--------|--------|------|-------|
| seq8 test | **0.533** | 0.457 | 0.493 | 0.645 | best: first_app 0.90, char_at_frame 0.82; worst: spend_alone 0.20 (=chance), crowd_count/spend_together 0.28 — NIAH≫DC exactly as the paper reports. parse-fallback 0.976 (7B ignores JSON format; closed-vocab extraction works) |
| seq8 val | 0.525 | — | — | — | test/val agree → stable |
| seq16 test | **0.422** | — | — | — | decay 0.533→0.422 with N, as the paper reports |
| seq16 val | 0.440 | — | — | — | |
| seq32 test | **0.361** | — | — | — | positional lookups collapse first (room_at_frame 0.78@8 → 0.30@32) |
| seq64 test | **0.304** | — | — | — | boundary NIAH still ~0.6 (first_app 0.64); aggregation tasks near chance |
| seq128 test | **0.247** | — | — | — | job 127822 (post-walltime-fix rerun) |

**Arm A COMPLETE.** Frozen curve: 0.533 → 0.422 → 0.361 → 0.304 → 0.247 (N=8→128).

## Phase 3 trainer RESULT (job 127821, DONE 2026-08-02)

- **BEST TF-answer acc 0.936 (tf-exact 0.477) @ ep 3 of 5** (ep1 0.863 → ep2 0.901 →
  ep3 0.936 → ep4 0.934 plateau). Ckpt:
  `outputs/mmred_hf/train/mix5760/20260801_205105_L12_r8/carrier_layer_best.pt`.
- Per-task at best: 10 qtypes ≥0.97 (char_at_frame, char_on_char_at_frame, crowd_count,
  final_app, first_app, n_char_at_frame, n_empty, room_at_frame, rooms_visited ~1.00);
  laggards = final-appearance conditionals (~0.81–0.87) + room_empty ~0.84.
- tf-exact 0.477 (vs park ~0.98): whole-scan verbatim fidelity low — family-E counter
  blocks suspected; answer-token acc is the selection metric; free decode (arm C) decides.

## Phase 4 wave 1 (in flight)

- Arm C test seq8/16 × {niah,dc}: jobs 127858–127861 (a100, DEC 112/192/160/320,
  --time explicit) → `exam/seq_len_{8,16}_test_{niah,dc}/`.
- Arm B: steps_in_room state dumps (L16) seq8/16: jobs 127862–127863 (rtx6k) →
  `armB/seq_len_{8,16}_test_steps/`; gate_tally follows on the dumps.
- Long-N arm C (32/64/128) sized after wave-1 decode-rate calibration.

### Arm C results (as they land; free decode, EM)

| config | group | acc | parse_fail | note |
|--------|-------|-----|-----------|------|
| seq8 test | NIAH (15 qtypes, n=750) | **0.833** | 0.000 | frozen overall @8 = 0.533 |
| seq8 test | DC (9 qtypes, n=450) | **0.702** | 0.007 | frozen DC @8 ≈ 0.3 — nearly doubled |
| seq8 test | **combined (1200)** | **0.784** | — | vs frozen 0.533 |
| seq8 test | steps_in_room (n=50) | 0.820 | 0.000 | from arm-B dump job |
| seq16 test | steps_in_room (n=50) | 0.580 | 0.000 | steeper in-length drop than park — watch full seq16 |

- **seq16 NIAH/DC (127860/61) TIMED OUT at 8 h** (time honored; the BASE decode path
  re-forwards the full fenced sequence per token: ~45 s/sample NIAH, ~115 s/sample DC at
  seq16 — and reports only write at the end, so partials lost). Long-N on the base path
  is infeasible (projected weeks at N=128).
- **BAD-NODE incident:** val8 arm-C pair (127898/99) ran on athena-post (draining) and
  produced DEGENERATE decodes (uniform hallucinated scans, f10 slots on 8-frame samples,
  acc 0.24/0.10 vs test 0.83/0.70) — jobs "COMPLETED" normally. Rerun as 127919/920 on
  a100. Corrupt run dirs left in place (superseded by newer timestamp in aggregator);
  athena-post added to the gotchas memory. All 10 other wave jobs verified on healthy nodes.
- Fix: the TRUNC instruments (`--fast-decode --truncate-at 12 --chunked-prefill`).
  **Exactness check first** (job 127886, 12 samples, base-vs-fast answer equality on
  mmred data) — then the whole remaining arm-C grid reruns on the fast path with
  timing-calibrated --time.
- **FAST-PATH REGRESSION (2026-08-02): every STANDALONE fast run produces the same
  degenerate scans** (uniform content, wrong slot counts) — seq16 test AND val agree
  (0.176/0.177) across different healthy nodes, so NOT hardware; the athena-post
  attribution for val8 is likely wrong (those were fast-path too). The exactness check
  passed because it runs base FIRST on the same sample record — suspicion: in-process
  state that base computes and standalone decode_fast lacks. ALL running fast jobs
  CANCELLED (10 jobs) to stop garbage burn.
- Diagnostic trio in flight (same 25 seq16-NIAH samples): 127928 fast-standalone /
  127929 truncate-only (mask arm) / 127930 base. Localizes the bug before any fix.
- **VERDICT (trio + re-read of the check's per-sample lines): fast 0.320 = trunc-only
  0.320 → the bug is TRUNCATION/dropkv, not decode_fast.** And the exactness check was
  **VACUOUS for word answers**: its internal comparisons used the int-only parser, so
  base(None)==mask(None) counted as "answer-equal 12/12" while the mask arm was already
  emitting degenerate scans (base 78t vs mask 64t, token-identical 0/12 was the visible
  tell I misread as benign truncation noise). acc_raw 1.000 came from the word-aware
  re-parse of the BASE arm only. Instrument fixed (parse_answer_mmred in all exactness
  arms, commit a2f89e8).
- **Root cause of the trunc failure: OUR ckpt was trained trunc=None** (full 66 GB cache;
  scan rows attended frames at train time) → dropping frame KV at eval removes trained-on
  inputs → uniform hallucinated scans. Park's ckpt passed trunc-eval genuinely (digit
  parses were real comparisons there). ALSO clears athena-post: val8 garbage was the
  fast path, not the node (memory note to be softened).
- **Decision: RETRAIN deploy-matched** (`--truncate-at 12`, the recipe's documented
  production mode; makes the fast path legitimate). Base decode at N=128 is infeasible
  (~15–30 min/sample). Job **127932** (l40s, --time=12h) → `train/mix5760_trunc/`.
  Not a "trainer failure" toward the stop-count: training succeeded; the eval-mode
  mismatch was a recipe omission (wrapper default lacks truncation).
- Current arm-C seq8 test numbers (0.833/0.702) were BASE-decode and remain valid for
  the old ckpt; the final grid will use the deploy-matched ckpt uniformly.
- **TRIO FINAL: base 0.960 / trunc-only 0.320 / fast 0.320** (same 25 seq16-NIAH
  samples) — diagnosis confirmed on all three counts.

### Deploy-matched ckpt v2 (127932) + the exposure-bias finding (2026-08-02)
- v2 trained: **BEST TF-answer 0.941 @ ep5 (beats frames-visible 0.936)** —
  `train/mix5760_trunc/20260802_110339_L12_r8/`. Truncation tax at TF level: ZERO.
- BUT free decode on v2 degenerates (word-aware exactness checks 127939/40: identical
  outputs across samples). Micro-probe `scripts/mmred_hf/probe_trunc_parity.py`
  (127946/48) PROVED: (a) prefill exact & sample-dependent (lg0 == training logits);
  (b) **forced-gold stepping through the eval loop == training-TF exactly (37-38/40)
  → the eval decode path is mathematically CORRECT**; (c) the failure is pure
  **exposure bias**: ~5–7% per-token error on payload words (boundary slips: 'itchen',
  'karden') → first free-decode slip locks repetition. Carriers-only supply feeds the
  linear gate (0.93) and TF answer read (0.941) fine, but token-level payload
  generation is fragile. Frames-visible ckpt1 free-decoded fine because the reader
  could re-check payloads against frames.
- Plan: (1) **v3 retrain launched (127952)**: family-E ×2 oversample
  (roots_mmred_hf_e2x.txt, 7200 samples), 10 epochs, truncate-at 12 →
  `train/mix_e2x_trunc/`; acceptance metric = forced-stepping per-token error toward
  park's ~1%. (2) **Grammar-constrained decoding** approved by Tal; implemented as
  `gnnformer/scan_grammar.py` (per-sample anchored regex, SYNTAX only — semantic
  values never computed; partial-match logit filter, fail-open; 2400/2400 gold scans
  legal, test-enforced) + `decode_fast(selector=)` + `eval_carrier --grammar`.
- **Grammar A/B on v2 (127957/58, seq16 25-sample slices): NIAH 0.320→0.440, DC
  garbage→0.280; parse_fail 0, perfectly formed scans — but content still
  input-invariant.** Re-reading the probe: the forced-stepping errors were all
  PAYLOAD-CHOICE positions → per-slot payload acc only ~60–70% under gold prefixes.
  TF-answer 0.941 was flattered (selection answers = copies of gold slots);
  tf-exact 0.094 was the honest signal. **Diagnosis: carrier CONTENT richness —
  binary match/no-match is strong (arm B 99%+/frame) but full room-identity per
  frame is weak on their harder visual domain (teacher d′ 7.9 vs park 13.5, where
  the same format free-decoded at 0.98).**
- Decision tree: v3 lands (~1–2 h) → A/B ±grammar on the same slices → grid on the
  winner; if payload reading doesn't improve, THAT is the result — grid reports
  A/B/C honestly with the carrier-content limitation as a central finding.

### STOP-AND-ASK (2026-08-03): deployed carrier-only free decode is degenerate at ALL lengths
- v3 trained fine (TF 0.959 @ep10). But seq8 sanity slices: NIAH **0.080**
  (char_at_frame 2/25), DC 0.440 — and decode samples show WHOLE-SCAN CONSTANT
  payloads (office×8 / garden×8; sample-varying but slot-invariant, mostly wrong).
- **All earlier slice A/B "improvements" (0.32→0.44 etc.) were COMPOSITION ARTIFACTS:**
  dirs-files are sorted by gold answer, so a constant emitter scores the majority-gold
  fraction of the slice. Lesson recorded: SHUFFLE eval slices.
- Net diagnosis across every instrument: carrier states on this domain support
  (a) binary per-frame relevance via a supervised linear gate (arm B: 99%+/frame,
  tally 0.93–0.82 flat to N=128), (b) teacher-forced ANSWER reading (TF 0.94–0.96 —
  partly selection-copy), but NOT free-running per-slot payload generation
  (room-identity content). Frames-visible reading works (ckpt1 base: 0.833/0.702 @8,
  0.96 slice @16). Format is not the issue (grammar: parse_fail 0, no rescue).
  Training budget is not the issue (v3 ×1.25 data, 2× E, 10 ep: no change).
- Awaiting Tal's call on how to finish the grid (options in chat).

### v4 RESCOPE (Tal-approved 2026-08-03): match the readout to the CLAIM
- Thesis claim = carriers carry the question-conditioned MESSAGE (aggregation repair),
  NOT whole-frame identity (VoCo-LLaMA's compression claim). The caption format
  silently demanded the latter; arm B proves the former holds (99%/frame, flat to 128).
- **v4 targets** (`build_target_v4`, commit above; 2400/2400 parse-back + grammar
  legality test-enforced):
  - verdict scans (`f1:- f2:x(1) … | total: k END`) for the 6 numeric-aggregation
    qtypes: steps_in_room, crowd_count, n_char_at_frame, n_empty, n_room_on_char_{f,f}
    — relevance bits + counts only, THE thesis pipeline, truncated + fast + grammar.
  - DIRECT answers (` answer: v END`) for the other 18 — single-fact read, no
    generative cascade; content supply at the read position becomes the measured
    quantity. rooms_visited + family E flagged: implicit aggregation-over-content,
    theory predicts weakness (frozen is their bar).
- **v4 trainer launched: job 128262** (9600 samples =100/root, 8 ep, truncate-at 12,
  --mmred-target v4) → `train/v4_mixed/`.
- Slice-artifact fix: `data/mmred_hf/dirsfiles/*_shuf.txt` (seed-0 shuffles, 19 files);
  ALL future slice evals use shuffled prefixes.
- Grid plan post-v4: shuffled seq8/16 A/B smoke → full 14-cell grid (v4, truncated,
  fast, grammar; direct-answer cells decode ~8 tokens → N=128 becomes cheap).

### CONTENT PROBE (2026-08-03, job 128278) — the fork resolves: READER, not carrier
- **Room-of-C is linearly present in the FROZEN carrier states: 6-way logistic acc
  0.814±0.022 at L12 (majority 0.185; 1600 labeled frames, char_at_frame train).**
- Content ladder: carrier states 0.81 (linear) → caption TF payloads ~0.6–0.7 →
  v4 trained direct read 0.10–0.23. The carriers transport content; the trained
  readout does not extract it. **e_c exonerated — the re-distill/locus/pooling track
  is unnecessary.** New instrument: `scripts/mmred_hf/content_probe.py`.
- Suspected causes: supervision density (probe: 1600 labeled frames; v4 direct cells:
  1 answer token/sample) + selection+extraction learned jointly + possible LoRA
  rotation at the read depth (cf. post-LoRA tally-gate 0.46 vs 0.93).
- In flight: L16 probe pair (128287 post-LoRA-v2 vs 128288 frozen reference) —
  distinguishes "reader erases content" from "reader never looks."
- Next fix candidate: **v5 dense-read supervision** — augment training with per-frame
  content QA generated from gold states (every step, not just the asked one) ≈ 8×
  labeled reads/sample, matching the probe's supervision density. In-recipe.
- **L16 PAIR VERDICT (128287/128289): frozen-L16 0.714 vs post-LoRA-L16 0.706 —
  LoRA does NOT erase content; the L12→L16 decline (0.814→0.71) is ordinary depth
  attrition.** Reader failure = pure supervision starvation, corroborated by v4's
  still-rising direct-read curve (char_at_frame 0.20→0.31→0.47→0.53 over eps 5–8,
  never plateaued; v4 BEST 0.539 @ ep8, verdict cells ~0.99 throughout).
- v4 ckpt: `train/v4_mixed/20260803_154941_L12_r8/`. Next: v4 shuffled smoke
  (verdict-cell free decode) + v5 dense-QA build.

### v4 smoke + steps diagnostic (2026-08-03, shuffled slices)
- v4 free decode: seq8 NIAH 0.417 / DC 0.146; seq16 0.354 / 0.312 (48-sample cells).
- **Steps diagnostic (128294): v4 free-decodes the ALL-`-` empty scan on every sample
  (total: 0; acc 0.600 = gold-zero fraction).** The verdict TF 0.99 was ALSO a
  copy artifact (total = copy of last gold tally). Root cause: **slot class
  imbalance** — their skewed prior makes ~90% of gold slots `-`; the constant
  predictor is near-CE-optimal so the carrier-consulting circuit never forms.
  Park had p(x)≈0.5 (uniform prior) — the controlled contrast that explains why
  the same recipe worked there.

### Content-ceiling probes (COMPLETE)
| carrier / res | linear ceiling |
|---|---|
| park e_c @392 | 0.814 | | park e_c @512 native | **0.846** | | re-distilled @392 | 0.827 |
- Ceiling robust ~0.81–0.85 across carrier variants; resolution the biggest lever
  (+3.2); distill choice ~noise. Encoding interventions deprioritized.

### OVERNIGHT PLAN 2026-08-03→04 (Tal: full permissions; ping in the morning)
- **Arm B for ALL 24 tasks** (commit b026a9b): (N1) decodability suites 128334/128335
  → 24-row table; (N2) dump fleet 128336–128342 (fit: seq8/16 train 100/qtype; eval:
  all 5 test splits); (N3) `armB_grid.py` — per-fact linear+MLP(GIN-ψ) heads fitted on
  seq8/16, per-frame predictions reduced symbolically (existing recompute logic),
  EM grid task × length; heads are per-frame ⇒ length-transfer by construction.
- Pilot verdict (128317, BEST 0.668@ep1 then overfit): verdict scans ✓ (steps TF 0.99+,
  slots formed), density does NOT unlock person→room direct reads (char_at_frame
  pinned at chance across all epochs). v5 (full mixture) still finishing for the
  definitive version + smoke.

### v5 (job 128299, l40s, IN FLIGHT)
- v4 recipe + 6000-cap augmentation pool (10,200 dirs): (a) dense per-step content QA
  (char_at_frame + room_at_frame from gold states, ~2N reads/sequence ≈ 10× density on
  the ρ content relation); (b) 1,200 high-k steps_in_room questions (k ≥ max(2, N/3)
  pairs from the count matrix) lifting the slot-positive rate 0.12→~0.30 (kills the
  all-`-` degenerate optimum). `scripts/mmred_hf/gen_dense_qa.py` + inline; both
  round-trip validated. 15.6k samples, 8 ep, truncate-12, targets v4.
- Tal 2026-08-02: launch the Phase-2 e_c re-distill after all → job **127936**
  (train_carrier_token --objective distill, steps_in_room seq8 renders, n=200/150,
  room init) → `carrier_distill/steps_seq8/`. Branch plan: current grid runs on the
  park-e_c deploy-matched LoRA (127932); the re-distilled e_c is the ready fallback /
  ablation — a LoRA retrain on top of it only if the grid shows carrier-limited
  behavior or Tal calls it.
- **RE-DISTILL RESULT (127936, DONE): d′ 6.79±0.59 @ ep15** (94% of eval-split teacher
  7.26) vs transferred park e_c 6.47±0.52 — **statistically indistinguishable → the
  Phase-2 reuse decision is now directly evidenced.** Ckpt kept as fallback:
  `carrier_distill/steps_seq8/20260802_114434_distill_room_k1/`.

## Phase 5 record — safety cells (DONE 2026-08-02, jobs 127900–02)

- **MME: zero drift** (all shown subsets delta +0.0 with adapter hooks always-on).
- **POPE: −2.2 pts acc (0.862→0.840; f1 −3.2)** — MARGINALLY OUTSIDE the ≤2 pt band,
  driven by POPE/popular (−3.5); adversarial −1.3, random −1.7. Flagged honestly:
  band is 2.0, we measure 2.2 at n=500.
- Plain-prompt drift (digit-argmax numeric subset of seq8 test, n=130): hooks-on 0.654
  vs hooks-off 0.708 (−5.4 pts; small-n caveat: ±~4 pts binomial, mostly g0 samples).
- Run dirs: `noharm/mix5760/`, `noharm/drift_seq8{,_baseline}/`.

### Arm-B instrument finding (2026-08-02)
- First wave-1 result: **arm C steps_in_room seq8 free-decode 0.820** (frozen 0.559),
  parse_fail 0. But gate_tally on the SAME run's L16 dump: **0.456 exact, per-frame err
  0.215** — far below the Phase-1 ceiling (0.981/0.992).
- Interpretation: the wave-1 dump is POST-LoRA (layers 12–16 carry the trained adapter);
  LoRA training rotates carrier features at L16 away from linear-gate readability while
  the info flows into generation (0.820 free-decode on the same samples). A finding, not
  a bug — kept as the "post-LoRA states" row.
- Post-LoRA dump seq16 gate_tally: 0.264 (vs no-LoRA ceiling 0.928) — rotation effect
  grows with length; secondary rows only.
- The brief's GIN ceiling = FROZEN fenced states → arm B re-instrumented as
  `train_carrier_token --eval-only` (no LoRA, Phase-1 methodology) per test length:
  jobs 127865–127869 → `armB/noLora_seq{8,16,32,64,128}_steps/`.
- Arm-B (GIN ceiling, no-LoRA, steps_in_room n=50): seq8 **0.928±0.069**, seq16
  **0.928±0.030**, seq32 **0.952±0.016**, seq64 **0.832±0.030** (per-frame err ≤0.7%
  everywhere; the 64 dip = error accumulation over more frames, still above the
  independence bound). seq128 **0.816±0.054**.
- **ARM B COMPLETE: 0.928 / 0.928 / 0.952 / 0.832 / 0.816 (N=8→128) — the GIN ceiling
  is essentially LENGTH-FLAT on the original benchmark** (per-frame err ≤0.7% at all
  lengths; decay = frame-count accumulation only). vs frozen 0.533→0.247.
- FULL (pending smoke): ROOTS_FILE=slurm/lib/roots_mmred_hf.txt (5760), caption recipe
  verbatim (L*=12, r8, frozen e_c carrier_token_room_k1_best.pt, lexicographic save),
  QOS 24h_1g → `outputs/mmred_hf/train/mix5760/`.

## Log

- 2026-08-01: Campaign started. Upstream cloned (56c6ee70); no LICENSE in repo (flagged).
- 2026-08-01: `.venv_mmred` on module python 3.11.15 (3.9 can't import upstream — deviation recorded above).
- 2026-08-01: HF dataset cached + inventoried; probe render deterministic, ~20 KB/frame, ~10 GB projected → disk GO.

### Decodability suite part 2 (128335, n=400 frames/row — 50-sample cells)
- room-of-C decodability is QUESTION-DEPENDENT: 0.81 under char_at_frame's direct
  question vs 0.63–0.68 under first/final_app/rooms_visited phrasings (maj 0.23).
- Conditional triggers/payloads weak: trig 0.63–0.66 (maj ~0.56), occ7 payloads
  ~0.51–0.57 (maj ~0.5), char_on_char_at_frame ≈ majority.
- spend_alone gate 0.78 (maj 0.54); spend_together 0.82 (maj 0.79 — marginal);
  crowded_room 0.78 (maj 0.71).
- Caveat: 200-frame probe fits; armB_grid's train-accs (2.4k–14k frames) are the
  better pooled estimates. Consistent theme: char→room strong, room→occupant weak,
  conditioning phrasing matters.

### Decodability suite part 1 (128359, n=400 frames/row)
- room_empty gate 0.987; crowd_count 0.870; n_empty 0.673; char_at_frame 0.659;
  n_char_at_frame 0.584; where_spend 0.587; occ7 rows ~0.51 (maj ~0.4-0.45);
  steps gate 0.824 ≈ maj 0.830 (test-prior skew at n=400 — armB fit on 2.4k train
  frames reads 0.94/frame; suite rows are small-n lower bounds, armB train-accs
  are the better estimates).

### OVERNIGHT RESULTS 2026-08-04 (arm-B-for-all-tasks + v5)
- **Arm-B GIN-floor grid COMPLETE** (heads fit on seq8/16 train dumps, applied to all
  test lengths; `armB_grid/armB_grid_{linear,mlp}.csv`):
  - LINEAR overall: **0.593 / 0.557 / 0.352 / 0.273 / 0.245** (N=8→128) — beats frozen
    at 8 (0.533) and 16 (0.422), ties beyond. MLP (sklearn) overfit → linear is canonical.
  - Star floors @8/16: char_at_frame 0.98/0.92, where_spend 0.96/0.98, room_empty
    0.98/1.00, final_app 0.98/0.88 — family E's argmax tasks have a STRONG external
    floor exactly where every in-model readout fails.
  - Weak floors: room→occupant (occofr train-acc 0.63) + conditionals (trig 0.65) —
    the same char→room ≫ room→occupant asymmetry as all other instruments.
  - **NEW FINDING — carrier-state length drift**: char_at_frame (single-frame readout,
    no compounding) decays 0.98→0.60 @N=16→32 → the states themselves shift with
    sequence length; separable from counting tasks' error compounding (0.94^N).
    Fix candidates: length-diverse head fitting, position-invariant features.
- **v5 (dense-read supervision) DONE**: BEST 0.574 @ep8 (v4 0.539; tf-exact 0.393 vs
  0.314). char_at_frame TF climbed 0.16→0.54 (plateau ~0.5 vs linear ceiling 0.71–0.81).
  Shuffled free-decode smoke: 8-niah **0.521** (v4 0.417), 8-dc 0.188 (0.146),
  16-niah 0.417 (0.354), 16-dc 0.167 (0.312 — only regression). Ckpt:
  `train/v5_dense/20260803_203521_L12_r8/`.

### STAGED LADDER (Tal's protocol, 2026-08-04) — stages 1+2 COMPLETE
Stage 1 alone-pass ceilings (single frame + per-frame question, logit-restricted):
roomofc 0.993 / occofr 0.986 / trig 0.997 / empty 0.986 → frozen encoder is NOT the
limit anywhere. Stage 2 PCW (fenced windows, per-frame-question replicas, message
probes): L16 span-mean hits CEILING on all four facts (0.995/0.969/0.997/0.991);
L16>L12, span-mean>last-token uniformly → locus-agnostic pooled read validated;
parallelization cost ≈ 0. Old carrier-STATE probes (0.81/0.63/0.65/0.99) are now
bracketed: loss is in carrier-or-state-probe, not encoder/PCW. Scripts:
stage1_alone.py / stage2_pcw.py; results outputs/mmred_hf/stage{1,2}/.
NEXT: stage 2b carrier-MESSAGE probe (deployed layout) decides whether stage-3
distillation is needed at all vs a feature-space switch in arm B.

### Stage 2b + control (2026-08-04): frozen open attention does NOT read carriers
- Carrier→tail messages under the DEPLOYED open mask: ALL facts ≈ majority (incl. the
  distilled steps read, control 0.838 vs maj 0.841). Park's 0.995 "message" numbers
  used ISOLATED reads (locus attends the carrier alone → message ≈ W_O·W_V · state,
  a fixed linear transform of the state). Two instruments, two questions:
  open-attention traffic ≈ 0 (why untrained decode fails); isolated reads track state
  decodability (usable as a cheap arm-B feature transform on existing dumps).
- Consequence: stage-3 design = STATE-matching distillation (carrier h@L12 ← stage-2's
  span-mean L16 teacher read), avoiding any frozen-attention hop. Plan: wave-0 scripts,
  wave-1 parallel target capture (8 shards), wave-2 parallel distill variants
  (k=1 mix / k=2 / single-fact control), wave-3 probes + head-transfer + re-dumps.

### STAGE 3 (2026-08-04) — state-matching distillation SOLVED via depth + aux-CE
- Objective ablation (roomofc control, final probes): raw-MSE@L12 ~0.79 (≤init) →
  +affine-W@L12 0.793 (geometry w/o content; gradient starvation diagnosed: loss/cos
  converge on bulk variance, probe flat) → +depth-align@L16 (Tal's hypothesis) 0.935 →
  **+aux-CE (per-fact heads, direct decodability gradient) 1.000**.
- **CE-mix (ONE e_c, 4-fact mixture, seq8+16 targets): per-fact probes at teacher grade
  by ep4-6: roomofc 1.000 / trig 1.000 / empty 0.988 / occofr 0.974-0.977** (teachers
  0.98-1.00; init states 0.63-0.81). Zero interference vs single-fact control.
- Instruments/design: teacher = wave-1 span-mean L16 PCW messages (validated 0.977-0.999);
  student = carrier h@L16 (open phase 12-15 included), affine bridge W + per-fact CE
  heads (discarded at eval; probes refit on held-out). scripts/mmred_hf/stage3_distill.py.
- Ckpts: stage3/k1_mix_L16_ce/e_c_best.pt (the candidate carrier), k1_roomofc_L16_ce
  (1.000 control), MSE-only arms as ablation rows.
- WAVE 3 next: length-generalization probes (new e_c @seq32/128 — trained on 8/16 only),
  re-dumps + per-condition heads + arm-B grid rows.

### Wave-3 probes of the CE-mix carrier (truncated L16, benchmark phrasing)
- room-of-C: seq8 0.667 (400-frame fit — probe-starved) / seq32 0.822 (1600) /
  **seq128 0.915 (6400 frames)** — apparent decay was probe-data artifact; the
  distilled carrier's states are length-robust to N=128 under deployment conditions.
- Gap to its 0.997 full-forward ceiling ≈ truncation+phrasing (~8 pts) → truncated-
  student CE re-distill in flight (128527, k1_mix_L16_ce_trunc). MSE-only fleet closed
  (ablation rows: L12 0.66-0.79, L16 0.79-0.935).

### TRUNC@16 (Tal's info-flow fix, 2026-08-04) — CONFIRMED; carrier FINAL
- Frame->text transfer lives in layers 12-18 (prior info-flow result) → truncating at 12
  amputated it. Probes of the CE-mix carrier at trunc16: **seq8 0.985, seq32 0.997**
  (vs 0.667/0.822 at trunc12). The trained CE-mix e_c is deployment-valid AS-IS with
  truncation moved to 16 (~+33% prefill, decode economics unchanged; fence still lifts
  at 12). Stalled truncated-student re-distill retired (ablation row: ~0.66 flat).
- **Arm-B v2 dump fleet launched** (128545-51): new e_c, L16 features, trunc16 —
  fit sets (8/16 × 24 qtypes) + all 5 test splits → armB_dumps_v2/. Heads + reductions
  + grid follow on completion.
- trunc16 probe series COMPLETE: **0.985 / 0.997 / 0.999 @ N=8/32/128** — the CE-mix
  carrier is length-perfect under deployment conditions. Recipe closed.

### ARM-B v2 GRID — PASS 1 (2026-08-04, job 128556; new carrier, L16/trunc16)
- Head fits: ALL ~37 fact heads at 0.996–1.000 train-acc (v1's occofr 0.63 / trig 0.65
  → 1.000/0.996; counts + per-entity banks outside the CE set also 1.000 — MSE anchor held).
- **Overall: seq8 0.889 / seq16 0.876 / seq32 0.718** (v1: 0.593/0.557/0.352; frozen:
  0.533/0.422/0.361). Per-task seq8: 5 tasks at 1.000; all 24 ≥ 0.70; conditionals
  0.74–0.94 (v1 ~0). Strict protocol (heads fit @8/16); seq32 dip = head extrapolation
  (states measured 0.997-decodable at 32 → relaxed protocol will recover).
- Pass 2 (64/128) auto-fires on remaining dumps. CSV: armB_grid_v2/armB_grid_linear.csv.
- L12+CE arm DONE: BEST 0.977 (vs L16+CE 0.995) — trunc@12 economics cost ~2 pts;
  full distill-arm ledger complete (see chat table). Ckpt: stage3/k1_mix_L12_ce/.

### ARM-B v2 STRICT GRID COMPLETE (pass 2, job 128595)
- Overall: **0.877 / 0.865 / 0.705 / 0.443 / 0.367 @ N=8-128** — beats frozen at every
  length (frozen 0.533→0.247). CSV: armB_grid_v2/armB_grid_linear.csv (with MAE/±1).
- **Position-extrapolation smoking gun @128**: first_app 1.000 vs final_app 0.380;
  first_at_room 0.960 vs last_at_room 0.140 — same head, same fact, early vs late slot
  positions. Counting = its compounded form (steps MAE 69). rooms_visited ±1-acc 0.98,
  n_empty ±1 0.86 (near-miss structure).
- Per-frame diagnostic (128565, 72 rows): heads read 0.85-1.00/frame even at 32 strict;
  reductions' arithmetic (compounding/margins/composition) explains the task-level decay.
- Per-length (headfit) protocol predicted ~0.85/0.79/0.73 @32/64/128 — dumps in flight.

### v6 (arm C on the L12CE carrier, job 128569) — DONE
- TF BEST **0.722 (tf-exact 0.548)** vs v5 0.574/0.393 — carrier swap alone.
- char_at_frame TF 0.10 (v4-era) → 0.95. Shuffled free-decode smoke:
  8-niah 0.583 / 8-dc 0.292 / 16-niah 0.500 / 16-dc 0.167 (v5: 0.521/0.188/0.417/0.167).
- Three-arm ordering established: frozen < arm C (generative) < arm B (heads); the
  C-B gap isolates the generative-readout cost on identical representations.
  Ckpt: train/v6_newcarrier/20260804_220050_L12_r8/.

### PER-LENGTH (PRIMARY) ARM-B GRID COMPLETE (jobs 128738-42, headfit=fresh generator seqs)
- **0.811 / 0.834 / 0.699 / 0.704 / 0.678 @ N=8-128 — FLAT 32→128, 2.7× frozen @128.**
- vs strict 0.877/0.865/0.705/0.443/0.367 (extrapolation tax recovered: +26 @64, +31 @128).
- 8/16 rows lower than strict = per-length fit-data volume (half/600 fresh samples), not
  representation. Best-protocol envelope: ~0.87 in-length, ~0.70 all long lengths.
- v6 arm C: TF 0.722, free-decode smoke 0.583/0.292/0.500/0.167 — three-arm ordering
  frozen < C < B with the C-B gap = generative-readout cost.
- CSV: armB_grid_v2_perlen/armB_grid_linear.csv. RESULTS PROGRAM COMPLETE — assembly
  (master table, figure, INDEX) next.

## 2026-08-05 — v7 program: full-fence L16 arm C (Tal-approved)
Diagnosis of the arm-C gap from the v6 smoke decodes: verdict scans COLLAPSE in free
decode (all-x or all-`-`; tally arithmetic itself perfect), i.e. TF 0.722 vs free 0.29
= exposure bias + pattern-copy, NOT bad carriers; and the argmax family had no explicit
format at all (v4 direct answers = implicit aggregation). Fix program:
1. **Full-fence L16 re-distill** — stage3_distill.py `--full-fence` (lo mask through all
   student layers; matches the PCW teacher's isolated windows AND an l_open=16 deploy).
   Job 128747 (l40s-public) → stage3/k1_mix_L16_ce_ff/. Gate: probe ≥ ~0.99.
2. **Verdict-token CE upweighting** — train_carrier_layer `--verdict-weight W`: w_hi on
   informative tokens (fN: slots, counts line, final value), 1.0 on copyable boilerplate
   (~80% of scan tokens). Token-alignment smoke-tested against the Qwen tokenizer.
3. **v5 counter-scan targets** — build_target_v5 (gnnformer/mmred_hf.py): per-entity
   running-counter scans for the 7 argmax/set qtypes (where/who_spend, spend_alone/
   together, crowded_room, room_empty; rooms_visited = first-visit stars) + complete
   counts line (zeros included → `least` works) + local max:/min: read-off; falls back
   to v4 otherwise. Validated: 840/840 gold parity, 0 parse mismatches, grammar
   (build_scan_regex_v5) fullmatch 210/210. tests/ all PASS.
4. **v7 train (pending distill gate)**: train_carrier_layer.sbatch with TASK=mmred_hf,
   ROOTS_FILE=slurm/lib/roots_mmred_hf_v5.txt, CARRIER_CKPT=stage3/k1_mix_L16_ce_ff/
   e_c_best.pt, L_OPEN=16, EXTRA="--mmred-target v5 --truncate-at 16 --verdict-weight 4",
   OUTPUT=outputs/mmred_hf/train/v7_L16_v5. ~14-18h; l_open=16 is native in the trainer
   (fence end = LoRA start = trunc; no code change needed).
Also: per-length grid CSV overwrite bug found (5 jobs, one filename — only seq128
survived); rebuilt all 118 rows from logs → armB_grid_v2_perlen/armB_grid_linear_ALL.csv.
DC-only per-length: 0.771/0.818/0.682/0.649/0.631 @8-128 (4.8x frozen @128).

### v7 (L16 full-fence carrier + v5 counter scans + verdict-weight 4) — DONE 2026-08-05
- Train 128757: BEST TF 0.873 / tf-exact 0.553 @ep5 (v6: 0.722/0.548). Counter-scan
  family at TF 0.95-1.00 (where_spend 203/206, crowd_count 218/218, steps 670/670,
  rooms_visited 216/216). Ckpt: train/v7_L16_v5/20260805_121814_L16_r8/.
- Free-decode smokes (128807-10, grammar v5, trunc16, n=48 each):
  8-niah 0.667 (v6 0.583) | 8-dc 0.333 (0.292) | 16-niah 0.646 (0.500) | 16-dc 0.188 (0.167).
  NIAH now ~length-flat free-running. parse_fail 0.000 everywhere.
- Verdict-collapse DIAGNOSIS UPDATE: binary/content reads fixed (frame-differentiated
  scans observed); residual failure isolated to SET-VALUED verdict banks (empty-rooms,
  chars-in-room) which decode as per-sample templates + two mechanical bugs (running-
  counter +1 slips; min: read-off ignores zero-count entities). who_spend regressed
  vs v6's direct answer at N=8 (1/8 vs 6/8) — scan only pays where verdicts grounded.
- v8 recipe (NOT launched; direction shifted to outputs/superquery/): dense per-frame
  QA for set banks, scheduled sampling on scan lines, symbolic read-off from counts line.
