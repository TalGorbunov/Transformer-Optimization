# TREEFOLD campaign — STATE (append-only; newest last)

- [2026-08-24] CAMPAIGN DRAFTED (Claude). Brief: outputs/treefold/CAMPAIGN_BRIEF.md — bands
  H1–H5 pre-registered there. Status: awaiting Tal's OK; no scripts written, no jobs submitted.
  Origin: peer thread 2026-08-24 (Chaim: over-squashing + GNN framing; Gabriele: no tool use;
  Tal: N-invariant) + artifact "Where Over-Squashing Lives". Baselines reused from recagg on
  identical samples: armC v3 (oracle+Python) ALL 0.89/0.92/0.85/0.90; Arm B (VLM full-state +
  Python) ALL 0.79/0.81/0.72/0.72; chunk-16 0.78/0.58/0.42; Arm A ~majority by N=32–64.
  Decisions recorded before any run: ONE model for every stage = Qwen2.5-VL-7B frozen nf4
  (Tal, 2026-08-24; 14B dropped, fallback-only with OK); primary Ask = zero-shot (format + conventions, no exemplars);
  leaf mode of the method = question-conditioned VLM emission (targeted perception; record:
  conditioned verdict recall 0.995–1.000 vs full-state 0.913/char, 0.637/frame); fan-2 tree,
  fan-in ablation k ∈ {2,4,8,16,N}; all parse failures counted WRONG and reported separately.

- [2026-08-24 20:11] CAMPAIGN AUTHORIZED (Tal) — start logged. Scope: GPU pre-authorized,
  no per-job asks, discipline rules §6 binding; ONE model everywhere (Qwen2.5-VL-7B frozen
  nf4 via gnnformer.runtime, text stages through the VL processor); no training, no probes,
  no code generation, no executor in the method; gnnformer/ untouched; no installs.
  MANDATORY CHECKPOINT added to the protocol (Tal, authorization message): after P0 lands
  (Ask zero-shot + few-shot on all 768 questions), write the P0 verdict in STATE (parse
  rate, combinable rate, per-type schema audit, primary-Ask-mode band decision) and STOP
  for Tal's schema review before ANY tree run T1–T5. T3 captions are exempt (submitted
  early as the long pole; they don't depend on P0's decision).
  Plan of record: read-in done (brief, recagg brief/STATE/INDEX, ask_compile_execute.py +
  caption_frames.py = the imported anchors, armor STATE); next = build scripts/treefold/,
  smoke at LIMIT≤3 N=16 in outputs/_scratch/treefold_smoke/, submit T3 captions, run P0.

- [2026-08-24 20:45] SCRIPTS BUILT (scripts/treefold/: tf_common, ask, leaf_text, leaf_vlm,
  tree_reduce, answer_score, run_cell, fig_treefold; slurm/treefold_{ask,leaf_vlm,cell,
  smoke}.sbatch — env-var driven via lib/common.sh, explicit --time on every wrapper, NS
  knob uses dashes not commas). Anchors imported READ-ONLY: norm/EM + sandbox run_program +
  type_of + SEEN_TYPES (ask_compile_execute), rebuild_task_dirs (armB_execute),
  sample_dirs + NAMES (caption_frames), parse_qa/ROOMS_HF (load_hf_sample). gnnformer
  untouched (runtime import only). CPU sanity PASSED on login: build_tasks(8) order
  byte-identical to armC v3 rebuild_task_dirs (768/768 dirs+Ns match), per-type-4 = strict
  subset (the Arm B rule), v3 program map loads (768), fidelity dry-run (choose-field →
  'count', agreement 1.0), prompt renders eyeballed, dir names collision-free across pools.
  Design decisions recorded BEFORE the runs they affect:
   * Prompts: orchestrator passes ONLY structural info (step index t-of-N to leaves,
     segment spans to merges); question text included in VLM leaf prompts (brief §2 MAP)
     but NOT in text-leaf prompts (leaf rule only — oracle record is complete; spec-literal);
     conventions block = armC v3 parity (1-based steps, appear/FIRST/FINAL, together/alone,
     "Nobody") in ask/leaf/answer prompts.
   * Generation: greedy everywhere (deterministic ⇒ stage resume is exact); LEFT padding
     through the VL processor (caption_frames' right-pad was safe only for identical
     prompts; ours vary). max_new: ask 512, leaf/merge notes 96, answer 32. DEVIATION from
     brief's "≤48 new tokens": 48 is too tight for dict-valued notes (e.g. where_spend
     per-room counts over 6 rooms) — 96 chosen before any run; token-budget table will
     report actuals.
   * Strict JSON: first balanced {...} → json.loads, no repair/no retry (greedy retry is
     a no-op); failure taxonomy ask_parse / not_combinable / caption_missing (source
     record absent) / leaf_parse / merge_parse / answer_empty — each WRONG + counted.
     One leaf parse failure kills the whole sample (no placeholder notes).
   * Tree: balanced left-to-right chunking into groups of k, singleton groups pass through
     uncalled; all merges of a level batched across samples; fan=0 ⇒ k=N single call
     (Arm A shape). T2 reuses T1's saved leaf notes (leaves are k-independent).
   * Fidelity instrument (count-like only, never touches EM): gold partial = v3 program
     (sandbox, read-only) on the subtree's ORACLE frames; the note field is chosen per
     sample by max leaf agreement (list-valued fields match int golds by length), threshold
     0.5 else counted out-of-coverage (schema-not-restriction-shaped, reported); subtrees
     where the program errors (e.g. step-index lookups outside the subtree) excluded from
     denominators. Prediction check EM ≈ p_leaf^N · p_merge^(N−1) in every T1 report.
   * Answer scoring: armC norm/EM verbatim; answer cleanup = first line, strip
     quotes/period only (no extraction heuristics). Majority baseline = mean over
     type-cells of modal-gold frequency (comparable to mean EM at equal per-type n).
  SMOKE submitted: job 136824 (l40s-shared n314 idle at submit, 2h_2g, --time=1:45) —
  ask zs+fs → T1 → T2 k4/kN (reused leaves) → T4 (Arm B captions) → leaf_vlm → T3, all
  LIMIT 3 N=16 → outputs/_scratch/treefold_smoke/.

- [2026-08-24 21:05] SMOKE FIRST READ + ONE ask-prompt iteration (BEFORE P0; armC v1→v3
  precedent: prompt-side clarity only). Ask smoke: parse 3/3, combinable 3/3 in BOTH modes,
  ~12s/3 questions. QUALITY SPLIT: few-shot asks are real programs (English rules, correct
  null-propagation merges; the room_on_char_final_app leaf rule shows the genuine
  method-level difficulty — a leaf cannot locally know FINAL-ness, and its left-priority
  merge is wrong for FINAL: exactly what T1/H1 is built to measure, NOT fixed by us).
  Zero-shot asks parsed but the 7B echoed JSON-template placeholders as its leaf/merge/
  answer "rules" ('$left.room' pseudo-refs) — passes the parse band, semantically empty.
  ITERATION (recorded before P0): ASK_FORMAT gains one content-free sentence — the three
  rules must be short instructions WRITTEN IN WORDS, never a JSON template; may refer to
  left.<field>/right.<field>. No exemplars added, no task content — zero-shot stays
  zero-shot. The P0 audit gains a template-echo check on top of the parse/combinable band.
  Teacher-forcing check (Tal's question, answered): NOTHING in the scored pipeline is
  teacher-forced — all notes are free-running greedy model output; oracle appears only as
  leaf INPUT in T1/T2/T5 (cell definition) and in the fidelity instrument's golds (never
  entering any prompt).

- [2026-08-24 21:20] SMOKE COMPLETE (136824, 4:50 wall, all 8 stages) + fidelity
  instrument CPU-verified (synthetic level-2 corruption detected and propagated exactly:
  L2 0.75 → L4 0.00; EM_pred formula renders). Smoke EMs 0.00 as expected (pre-fix
  template-echo asks) — mechanics all green: 4-level fan-2 tree, k=4/k=N reduce with
  reused leaves, T4 caption-leaf path, VLM leaves 48 frames/29s (~1.65 f/s @batch16 ⇒
  23k-frame T3 caption job ≈ 4h), failure counters surface correctly (nothing silent).

- [2026-08-24 21:30] ═══ P0 VERDICT (band + audit) — CHECKPOINT, STOPPING FOR TAL ═══
  Runs: jobs 136828 (zero-shot) / 136829 (few-shot), 768 questions each, ~14 min each →
  outputs/treefold/p0_ask/20260824_203137_{zs,fs}/ (asks.json, report.txt, audit.md).
  * Parse rate:      zs 0.990   fs 0.997   (band ≥0.95: MET)
  * Combinable rate: zs 0.990   fs 0.997   (band ≥0.95: MET)
  * Template-echo (post-fix audit): zs 0/760, fs 0/766 — the written-in-words iteration
    eliminated it. All failures are "missing:combinable" key omissions (8 zs / 2 fs),
    concentrated in char_on_char_* / room_on_char_first_app; counted ask_parse → WRONG.
  * Ask input verified question-only (no type name, no frames, no N) — the stop-condition
    check "schemas depend on type name / seeing frames" CANNOT fire by construction;
    prompt.txt saved in each run dir.
  * BAND DECISION (pre-registered rule): zero-shot ≥0.95 on both rates ⇒ ZERO-SHOT IS
    PRIMARY for T1–T4; few-shot = T5 control. Applied and recorded here before any tree run.
  * QUALITATIVE AUDIT (the reason Tal wants eyes on this): per-type examples in
    audit.md. Few-shot asks are crisp programs (accumulator fields, correct
    null-propagation merges). Zero-shot asks are English-worded and parse, but MANY are
    not restriction-shaped: e.g. steps_in_room designs {character, room, status} with NO
    count field and an answer rule reading "count ... across all steps" as if the root
    note were the full sequence; n_empty (a step-8 lookup) has no step condition in its
    leaf rule. Honest prediction logged BEFORE T1: zero-shot T1 EM may sit well below
    armC v3 with p_merge possibly clean — that lands in the pre-registered falsification
    branch "H1 fails ⇒ Ask (schema) is the problem → per-type audit", and T5/H5 measures
    the zs-vs-fs gap directly. NOT patched further (prompt-side clarity iteration is
    spent; anything more = task logic or exemplar leakage).
  T3 CAPTIONS SUBMITTED (checkpoint-exempt long pole): job 136841, l40s-shared n314
  (idle), 24h_1g --time=8h, zero-shot asks (the primary), per_type=4, all Ns (~23k
  frames, est ~4h) → outputs/treefold/t3_leaves/20260824_zs/. Resumable if killed.
  ⏸ STOPPED per mandatory checkpoint — NO tree runs (T1–T5) until Tal reviews the
  schemas (audit.md files) and replies. Queued next on OK: T1 (fan-2 oracle, 768) →
  T2 (k∈{4,8,16,N} reusing T1 leaves, N∈{64,128}) → T4 (Arm B captions) → T3 tree → T5.

- [2026-08-24 21:50] ▶ CHECKPOINT CLEARED — TAL'S DECISION: FEW-SHOT PRIMARY for T1–T4
  (overrides the pre-registered zero-shot-primary band rule; rationale: the parse-rate
  band measured JSON validity, not schema semantics — the qualitative audit showed
  zero-shot schemas often lack accumulator fields entirely; few-shot is also the matched
  convention to the armC v3 baseline, which carried the same 4 SEEN exemplars; the 20
  UNSEEN types remain the generalization claim). Zero-shot becomes the T5 control
  (T1 protocol under zero-shot at N∈{16,128}; H5 comparison direction unchanged).
  Actions: 136841 (zs captions) CANCELLED ~40 min in — partial kept resumable at
  outputs/treefold/t3_leaves/20260824_zs/ (no queued consumer; post-hoc zs-perception
  comparison possible later). SUBMITTED (n314 idle, spread across QOS pools):
    136847 T3 captions FEW-SHOT  l40s 24h_1g --time=8h  → t3_leaves/20260824_fs/
    136848 T1 fan-2 oracle fs    l40s 24h_1g --time=12h → t1_oracle/20260824_fs/  (768)
    136849 T4 fan-2 ArmB-caption fs l40s 12h_4g --time=10h → t4_fullstate/20260824_fs/ (384)
  T2 (k∈{4,8,16,N}, reuses T1 leaf notes) + T5 (zs control) submit when T1 lands,
  per brief order.

- [2026-08-24 22:20] T4 LANDED (136849, 1:22:43; outputs/treefold/t4_fullstate/
  20260824_fs/ = the cell of record). ═ BAND H4 VERDICT: DECISIVELY NOT MET ═
    T4 (model-merge on Arm B caption records): ALL 0.25 / 0.24 / 0.18 / 0.23 (N=16→128)
    Arm B (Python executor, SAME records, same samples): 0.79 / 0.81 / 0.72 / 0.72
    |T4 − Arm B| ≈ 0.5 everywhere (band ≤ 0.05); T4 sits BELOW majority (0.36–0.41).
  The executor swap costs ~0.5 EM: the frozen 7B is a catastrophically worse fan-2
  executor than exact execution on identical inputs. Failure decomposition (fidelity
  hooks, count-like): p_leaf 0.55–0.86 — the model cannot even reliably FILL a note from
  a complete text record (leaf-side semantic noise, before any merging); p_merge 0.14–0.70
  and degrading UP the tree (upper levels compound corrupted notes); both far below the
  ≥0.98 rung H1 needs. Parse failures are a minor term (merge_parse 1–13/96 per N) —
  the misses are semantic, not format. Fidelity coverage itself is partial (0–0.75 by
  type): many few-shot schemas are not restriction-shaped for count-like types, so the
  instrument's field-matching can't track them (reported per design, not silently).
  CAVEATS BEFORE ANY CONCLUSION: T4's leaves read CAPTION records (0.637 frame-exact
  perception noise) — p_leaf here conflates caption noise with leaf-fill noise; T1
  (oracle leaves, still running) is the clean H1/executor read and the pre-registered
  falsification branch (p_merge < 0.98 ⇒ 7B inadequate executor ⇒ 14B merge control
  needs Tal's OK) is judged there, not here. H4 itself is settled: NOT MET.

- [2026-08-24 22:35] T3 CAPTIONS LANDED (136847, 1:34:47; outputs/treefold/t3_leaves/
  20260824_fs/ — 382 samples, 23,008 frames @512, JSON parse-ok 0.990).
  ═ H3 PERCEPTION BAND: NOT MET ═ count-like per-frame leaf accuracy 0.921 (55 covered
  samples) < 0.98 band, also < the 0.95 fallback line ⇒ pre-registered consequence
  fires: question-conditioning does NOT buy note-level perception (0.921 ≈ Arm B's
  full-state 0.913/char), and T4 becomes the primary end-to-end arm — moot in practice,
  since T4 itself failed H4: the model-executor, not perception, is the binding error.
  Note the contrast with the ninv record (conditioned VERDICT-BIT recall 0.995–1.000):
  emitting a structured JSON note per frame is a harder emission than a 1-bit verdict —
  the gap 0.995→0.92 is note-emission cost, not seeing worse. T3 TREE submitted anyway
  (F-B needs the curve; captions sunk): job 136883, l40s 12h_4g --time=6h →
  outputs/treefold/t3_tree/20260824_fs/. T1 (136848) mid-tree: merge L1 10k/22.4k.

- [2026-08-24 23:20] T3 TREE LANDED (136883, 47:56; outputs/treefold/t3_tree/
  20260824_fs/). ═ BAND H3 VERDICT: NOT MET (both rungs) ═
    T3 end-to-end: ALL 0.31 / 0.23 / 0.18 / 0.16 (N=16→128) — DECLINING in N, below
    majority (0.36–0.41) from N=32 on; COUNT-LIKE 0.06–0.14. Bands asked: ALL ≥
    Arm B − 0.03 (Arm B: 0.79/0.81/0.72/0.72) — missed by ~0.5; leaf acc ≥ 0.98 —
    measured 0.921. (steps_in_room @128 = 0.50 incidentally touches its sub-band —
    n=4/cell quantization, not signal.)
  New structural finding: merge_parse deaths GROW with depth — 1/9/22/36 per 96 at
  N=16/32/64/128 — deeper trees accumulate JSON-format mortality on top of semantic
  merge noise (p_merge ~0.7 where instrumented). Single-frame lookup types survive best
  (first_at_room 0.75–1.00, room_at_frame 0.50–1.00, first_app 0.50–1.00) — same
  factorization as Arm B: touch-one-frame ≈ flat-ish, whole-sequence types die in the
  reduce. CAMPAIGN PATTERN AFTER TWO MODEL-EXECUTED CELLS: T4 ~0.23, T3 ~0.22 — the
  frozen 7B cannot faithfully execute its own note programs at fan-2; the executor, not
  perception and not the tree TOPOLOGY, is the dominating error term so far. T1 (oracle
  leaves, cleanest read) still running — merge L2 ~40%.

- [2026-08-24 23:45] T1 LANDED (136848, 2:36:45; outputs/treefold/t1_oracle/20260824_fs/).
  ═ BAND H1 VERDICT: NOT MET — AND THE PRE-REGISTERED FALSIFICATION BRANCH FIRES ═
    ALL 0.23 / 0.20 / 0.18 / 0.21 (band ≥0.85 at every N; majority 0.36–0.39) —
    roughly length-FLAT, but flat BELOW majority: flatness without competence.
    UNSEEN 0.24/0.20/0.18/0.20; COUNT-LIKE 0.06–0.10. armC v3 on identical samples:
    0.89/0.92/0.85/0.90 — the Python→7B executor swap on ORACLE records costs ~0.65 EM.
  FIDELITY (the campaign's central measurement): on complete GT text records,
  p_leaf 0.56–0.91 (the 7B mis-fills its own note in 10–40% of frames) and p_merge
  0.20–0.70 (band ≥0.98), with MONOTONE per-level decay everywhere — e.g. crowd_count
  @128: L1 0.84 → L2 0.72 → L3 0.55 → L4 0.42 → L5 0.19 → L6/L7 0.00. The
  p_leaf^N·p_merge^(N−1) prediction (≈0) matches measured EM (≈0) on instrumented
  count-like types — error compounding over N−1 noisy ops, exactly the p^depth law.
  (rooms_visited EM 0.25–0.67 > pred: list-set answers can survive noisy intermediates;
  instrument coverage 0.25–0.5 there, reported.) The ONE reliable op: null-propagation —
  single-frame lookups stay flat-high (first_at_room 0.88–1.00, first_app 0.75–0.88,
  room_at_frame 0.38–0.50). merge_parse again grows with depth (0/14/21/25 per 192).
  READING: the tree TOPOLOGY is fine; the MESSAGE FUNCTION is broken at 7B — a frozen
  7B cannot execute its own compiled note-program at per-op fidelity anywhere near the
  ~0.99 that N−1 compositions require. Converges with recagg's scratchpad verdict
  (≤14B LMs lose long procedures in their own token stream), here measured PER-OP with
  the tree factorization. Per pre-registration: "p_merge < 0.98 ⇒ propose a 14B merge
  control (one extra cell) before any conclusion" — PROPOSED to Tal, NOT run (needs OK;
  note 14B would run text-only merge/answer stages via venv_arch qwen14b as in armC).
  SUBMITTED next per brief order (T1 leaves reused, k-independent):
    136909 T2 k=4 (12h_4g) · 136910 k=8 (12h_4g) · 136911 k=16 (24h_1g) ·
    136912 k=N (24h_1g) — N∈{64,128}, → t2_fan/20260824_fs_k{4,8,16,N}/
    136913 T5 zero-shot control (24h_1g, T1 protocol @N∈{16,128}) → t5_zeroshot/20260824_zs/
  H2 note recorded BEFORE T2 lands: with EM(k=2)=0.21@128 the pre-registered band
  (EM(k=2)−EM(k=N) ≥ 0.40) CANNOT fire — but the shape is still informative: if k=N
  BEATS k=2 (fewer noisy ops > more shallow ops), that inverts the over-squashing
  prediction and localizes the regime where trees help to high per-op-fidelity
  executors. Logged as the honest hypothesis for the fan-in figure.

- [2026-08-25 00:10] T2 PARTIAL: k=8 (136910) and k=16 (136911) LANDED (~30–45 min each,
  → t2_fan/20260824_fs_k{8,16}/). ALL @N=64/128: k=8 0.15/0.21 · k=16 0.15/0.16, vs
  k=2 (T1) 0.18/0.21, majority 0.36/0.39. First shape read: EM ≈ FLAT in k at this
  executor fidelity — neither the over-squashing drop (k↑ ⇒ worse) nor the noisy-op
  inversion (k↑ ⇒ fewer ops ⇒ better) dominates; per-op noise saturates everything at
  the ~0.2 floor. k=N (136912) FAILED: CUDA OOM — batch 48 × k=N merge prompts
  (64–128 notes ≈ 4–8k tokens each) on 44GB L40S; RESUBMITTED as 136937 with BATCH=4
  (~384 calls total, cost negligible; resumes the same run dir). k=4 (136909) + T5
  (136913) still running.

- [2026-08-25 00:40] T2 COMPLETE (k=4 136909; k=N 136937 with BATCH=4, 16:42 —
  → t2_fan/20260824_fs_k{4,8,16,N}/). ═ BAND H2 VERDICT: NOT MET / NOT SUPPORTED ═
  The fan-in law at N=64/128 (ALL; majority 0.36/0.39):
    k=2 0.18/0.21 · k=4 0.18/0.19 · k=8 0.15/0.21 · k=16 0.15/0.16 · k=N 0.18/0.18
  EM is FLAT in k — no monotone decrease (band asked EM(k2)−EM(kN) ≥ 0.40; measured
  ≈ 0.03), and no noisy-op inversion either. The most telling contrast: k=N has near-zero
  merge-parse deaths (1–2 vs 22–32 at k∈{2,4}) yet IDENTICAL EM — depth-mortality and
  breadth-squashing net out; the binding constraint is per-op semantic fidelity, which
  no fan-in choice can route around. Pre-registered fallback applies verbatim: "H2
  non-monotone ⇒ the fan-in/over-squashing link is not supported at this scale — the
  paper's mechanism claim reverts to the in-forward measurements only" (i.e. the armor
  Hahn instrument + capacity law remain the mechanism evidence; the across-forwards tree
  at 7B adds the executor-fidelity boundary condition, not a fan-in law). T5 (136913)
  in leaves 16k/27k.

- [2026-08-25 02:10] T5 LANDED (136913, 2:36:14; outputs/treefold/t5_zeroshot/
  20260824_zs/). ═ BAND H5 VERDICT: NOT MET at N=128 ═ Zero-shot vs few-shot on
  identical samples (T1 protocol, oracle leaves, fan-2):
    N=16:  zs 0.208 vs fs 0.229 (gap +0.021 ≤ 0.05 ✓)
    N=128: zs 0.073 vs fs 0.208 (gap +0.135 > 0.05 ✗)
  Zero-shot's weakly-structured schemas collapse harder with depth: merge_parse 70/192
  @N=128 (fs: 25/192), leaf_parse 21 (fs: 8). Pre-registered fallback applies: "program
  quality depends on demonstrations" — reported as such; few-shot rows kept as primary.

════════════════════════════════════════════════════════════════════════════
CAMPAIGN CLOSE — [2026-08-25] all cells landed; H1–H5 verdicts in one table
════════════════════════════════════════════════════════════════════════════

  band | claim | verdict | the number
  -----|-------|---------|-----------
  P0   | Ask parses zero-shot            | MET      | zs 0.990/0.990 (echo 0) — but Tal set FEW-SHOT primary at the checkpoint (schema-quality audit; parse-rate band was the wrong filter)
  H1   | oracle-leaf reduce length-flat ≥0.85 | NOT MET | ALL 0.23/0.20/0.18/0.21 (≈flat, but BELOW majority 0.36–0.39; armC v3 same samples 0.89–0.92). p_merge 0.20–0.70 ≪ 0.98 ⇒ falsification branch: the frozen 7B is NOT an adequate fan-2 executor. 14B merge control PROPOSED — awaiting Tal's OK, the one cell left unrun.
  H2   | EM decreasing in k, k2−kN ≥ 0.40 | NOT MET | EM flat in k: 0.21/0.19/0.21/0.16/0.18 @N=128 (k=2/4/8/16/N) — all at the executor-noise floor; mechanism claim reverts to the in-forward (armor) measurements.
  H3   | targeted VLM leaf acc ≥0.98; ALL ≥ ArmB−0.03 | NOT MET | leaf acc 0.921 (= ArmB's 0.913/char full-state; note-emission cost, not perception); end-to-end 0.31→0.16, < majority from N=32.
  H4   | model-executor ≈ Python on same records (±0.05) | NOT MET | T4 0.25/0.24/0.18/0.23 vs Arm B 0.79/0.81/0.72/0.72 — the executor swap costs ~0.5 EM.
  H5   | zero-shot ≥ few-shot − 0.05     | NOT MET @128 | zs 0.073 vs fs 0.208 @N=128; zs merge-parse mortality 70/192.

  THE CAMPAIGN'S POSITIVE FINDING (from the fidelity instrument): per-level p_merge
  decays with TREE DEPTH and is ~N-invariant at fixed level (F-C: level-1 ≈ 0.59–0.69
  at every N, level-4 ≈ 0.05–0.29) — corruption tracks the number of composed model-ops,
  not sequence length; measured EM matches the p_leaf^N·p_merge^(N−1) composition law
  (both ≈ 0 on instrumented types). Null-propagation is the ONE reliably-executed merge
  (single-frame lookups flat at 0.75–1.00 to N=128). Converges with the recagg
  scratchpad verdict — ≤14B frozen LMs cannot run long procedures in their own token
  stream — now measured PER-OP: the message function, not the tree topology, is the wall.
  TREEFOLD does NOT replace Ask-Compile-Execute; it measures WHY the executor must be
  external at this scale (armC 0.89 flat vs T1 0.21: exact execution is worth 0.68 EM).

  LEFT OUT AND WHY: (a) 14B merge control — pre-registered falsification cell, needs
  Tal's OK per authorization scope (no 14B without asking); (b) zs T3 captions — 40-min
  partial kept resumable at t3_leaves/20260824_zs/, no consumer after fs became primary;
  (c) fenced one-forward leaf emission — brief-declared efficiency arm, moot until an
  adequate executor exists. Assets: 6 canonical cells + P0×2 in INDEX.md, figures
  F-A..F-D + CSVs in figs_20260825/, every ask/note/merge saved in run dirs. Footprint:
  ~14 GPU jobs, ~13 GPU-hours, zero installs, gnnformer/ and recagg anchors untouched.

════════════════════════════════════════════════════════════════════════════
DRAFT RESULTS.md ENTRIES (for Tal — NOT appended to RESULTS.md)
════════════════════════════════════════════════════════════════════════════

- [2026-08-25] POST-CLOSE DIAGNOSIS (Tal's morning question: "leaf or merge? the model
  should be able to do it") — CPU re-analysis of the saved T1 artifacts, and Tal's
  intuition is RIGHT about the merge: ═ CONDITIONAL merge fidelity (both children
  correct) = 0.995 (2005/2016; 1.00/0.99/0.99/0.98 by level) ═ — the 7B executes
  left+right essentially perfectly; the instrumented p_merge decay (F-C) is pure error
  PROPAGATION, not merge-op failure. The corruption enters at the two INTERFACES:
  (a) MAP: per-frame predicate evaluation fails at 10–40%/frame on clean oracle text —
  worked example steps_in_room_K1_0005856: record reads "Office: Daniel; Bedroom:
  Sandra", rule "count=1 if Daniel is in the Bedroom", model emits count=1 (7 such
  false frames of 16; every merge above them exact: 1+1=2, 2+2=4, 2+6=8);
  (b) ANSWER read-off: of 8 samples whose ROOT NOTE was exactly correct, 5 still
  answered wrong (root {"count": 8} → answered "Daniel"); 15/73 numeric-gold answers
  were non-numeric. Revised mechanism sentence for the write-up: the tree and the
  merge arithmetic are SOUND at 7B; the failure is per-frame predicate evaluation
  (0.6–0.9) and final read-off — p_leaf^N alone (0.8^16 ≈ 0.03) suffices to explain
  the collapse. Implication for the proposed 14B control: it should target the LEAF
  and ANSWER stages, not the merge. DRAFT 1 amended accordingly (drafts are living
  text until "log this"; the H1/H2 band verdicts are unchanged — bands were defined
  on cumulative measures).

- [2026-08-25] ▶ 14B CONTROL AUTHORIZED (Tal: "run the 14b control on leaf+answer").
  Design (recorded before any run): a STAGE-SWAP GRID on the T1 protocol (oracle
  records, few-shot asks, fan-2, all 768) — Qwen2.5-14B-Instruct (venv_arch, bf16,
  HF_HOME=/rg, armC pattern) swaps in ONLY at the diagnosed interfaces; the 7B keeps
  the merge (measured sound, 0.995 conditional):
    ctrl-C  7B leaf + 7B merge + 14B ANSWER   — answer-swap alone; reuses T1's saved
            merges.json verbatim (768 answer gens, minutes) — the cheapest attribution.
    ctrl-A  14B LEAF + 7B merge + 7B answer   — leaf-swap; 14B writes leaf_notes.json
            (46k gens), 7B tree via run_cell --leaf notes (unchanged code path).
    ctrl-B  14B LEAF + 7B merge + 14B ANSWER  — both interfaces; answers regenerated
            from ctrl-A's merges.
  New instrument scripts/treefold/stages_14b.py (modes: leaves/answers; tf_common
  prompts byte-identical — ONLY the generating model changes; chat-template greedy,
  left-pad, strict JSON, partial saves) + slurm/treefold_14b.sbatch (ARCH_PY, offline).
  Prediction on the record: if the interface diagnosis is right, ctrl-B recovers a
  large share of the armC gap (leaf predicate + read-off were the leaks); ctrl-C alone
  fixes read-off only (~small gain, concentrated in count types with correct roots —
  8 samples had exactly-correct roots, so ceiling on ctrl-C is low); ctrl-A carries
  most of the lift. Smoke first at LIMIT=3 in _scratch per discipline.

- [2026-08-25 15:05] 14B smokes PASSED (137099/137100, ~3 min each; leaves 48/48
  parsed). ctrl-C LANDED (137101, 22 min; outputs/treefold/ctrl14b/20260825_ctrlC_
  ans14b/): ALL 0.24/0.22/0.20/0.18 vs T1 0.23/0.20/0.18/0.21 — answer-swap alone
  changes NOTHING (±0.02, within cell noise), as predicted: the roots are already
  leaf-corrupted, a better reader has nothing to read. ctrl-A 14B leaves running
  (137102, ~10 gens/s, ETA ~1.3h) → 7B tree then ctrl-B chain on landing.
  (Entry relocated 18:35 to restore chronological order — it was misplaced by an
  edit-anchor slip; content unchanged.)

- [2026-08-25 ~15:50] LEAF FAILURE MODE DIAGNOSED (Tal's "why is p_leaf so low?") —
  ═ IT IS A BINDING (CONJUNCTION) FAILURE, NOT A READING FAILURE ═ steps_in_room leaf
  confusion over all T1 samples: TP 321 / FN **0** / FP 873 / TN 726 — recall on
  true frames is 1.000 (never misses char-in-room when true); ALL error is one-sided
  false positives (0.546 of gold-0 frames), and the FP rate splits on whether the
  queried ROOM is occupied by someone else: **0.762 occupied vs 0.293 empty**. The 7B
  detects "char present" and "room occupied" but fails the conjunction char∧room —
  shown "Bedroom: Sandra", asked about Daniel, it answers yes 3/4 of the time. Explains
  the systematic OVER-counts (root 8 vs gold 1). Resonances: MMReD is by construction a
  char×room binding task — the 7B cannot do ONE binding check on clean text; this is
  the same binding problem the fencing/carrier line attacks structurally. Sharpens the
  14B control: on ctrl-A landing, rerun this exact confusion analysis on the 14B leaf
  notes — the question is whether 14B computes the conjunction (FP-split table is the
  verdict instrument).

- [2026-08-25 16:15] ═ LEAF-PROMPT ABLATION (Tal: "can't we adjust the prompt?") —
  THE BINDING FAILURE IS PROMPT-SHAPED ═ New instrument scripts/treefold/leaf_ablate.py
  (job 137121 after a no-GPU resubmit 137119 — heredoc sbatch lacked --gres; 384
  steps_in_room frames × 4 variants, N∈{16,32}, ~7 min):
    V0 canonical (T1's leaf prompt): acc 0.500 · FP|room-occupied 0.86  (replicates T1)
    V1 MINIMAL (record+rule+schema): acc 0.984 · FP|room-occupied 0.024
    V2 think-first: 0.732 · V3 check-nudge: 0.844 (partial repairs)
  Dropping the conventions block + plan framing takes the SAME model on the SAME
  predicate from coin-flip to 0.984 — the 7B CAN compute the conjunction; the canonical
  framing induced the yes-bias. Previous "7B cannot bind" reading AMENDED to "7B cannot
  bind under contextual clutter". Composition math still binding: 0.984^128 ≈ 0.13 —
  minimal prompts should transform N≤32, not N=128. ACTIONS (Tal's question = the
  steer): (a) T1-MINIMAL arm — run_cell gains --prompt-style minimal (leaf AND answer
  prompts stripped; answer read-off failure plausibly same cause), full T1 protocol
  rerun; (b) 14B ablation twin (V0/V1 on the same 384 frames) to complete the
  model×prompt 2×2; (c) the running 14B-canonical chain stays = scale axis at fixed
  prompt. Canonical T1/T2/T3/T4/T5 cells stand as the pre-registered record; minimal
  arms are labeled prompt-repair extensions.

- [2026-08-25 16:25] 14B ABLATION TWIN LANDED (137123, ~8 min; outputs/treefold/
  leaf_ablate/20260825_14b/) — THE MODEL×PROMPT 2×2 CLOSES:
    7B canonical 0.500 · 7B minimal 0.984 · 14B canonical 1.000 · 14B minimal 1.000
  (384/384 frames, zero FP zero FN at 14B in BOTH framings). Reading: single-frame
  char∧room binding is at the 7B's reliability EDGE — context clutter tips it into
  yes-bias; the 14B has margin and is immune to framing. "Not smart enough" (Tal) is
  right for the canonical prompt at 7B, with the sharper statement: the failure is
  margin×clutter, and EITHER more scale OR less clutter restores the leaf.
  PREDICTIONS ON THE RECORD for the pending cells (composition law, conditional
  p_merge 0.995 at 7B): ctrl-B (14B leaf ≈1.0 + 7B merge + 14B answer) count-types
  ceiling ≈ 0.995^(N−1): ~0.93 @16, ~0.53 @128 — merge noise becomes the binding
  constraint at long N; ctrl-A same leaves but 7B read-off cap; T1-minimal (7B leaves
  0.984): ~0.7 @16 decaying to ~0.07 @128 — the p^N law should now be VISIBLE above
  the majority floor instead of saturated at it.

- [2026-08-25 18:00] T1-MINIMAL LANDED (137122, 1:39:53; t1_minimal/20260825_fs/) —
  AND EXPOSED A HARNESS BUG IN MY MINIMAL ANSWER PROMPT (caught by the fidelity
  instrument): steps_in_room @N=16 shows p_leaf 1.000, p_merge 1.000, roots exactly
  correct — yet EM 0.25, because the minimal answer prompt's isolated 'If no one or no
  room matches, answer "Nobody"' clause made the 7B answer "Nobody" on NUMERIC roots
  ({"count": 0..12} → "Nobody"; root-contains-gold 0.66 vs final EM 0.31 on the type).
  V1 artifacts preserved as *_v1_nobodybug.*; minimal_answer_prompt fixed (clause
  dropped — the ask's own answer rule covers Nobody); answers-only rerun 137190
  (resumes saved leaves+merges). Genuine v1 findings that stand regardless:
  rooms_visited 0.88/0.75/0.75/0.75 (canonical: 0.12–0.50) with p_merge 0.94–0.98 —
  the prompt repair transforms binding-limited types; crowd_count leaves stay ~0.85
  (its per-frame job is count-people-and-threshold, not a binding predicate — minimal
  framing doesn't fix arithmetic-in-one-glance); first_at_room 1.00 flat everywhere.
  ALL 0.30/0.27/0.27/0.23 (v1, read-off-suppressed) — v2 rescore pending.

════════════════════════════════════════════════════════════════════════════
EXTENSION CLOSE — [2026-08-25 18:30] stage-swap grid + prompt repair complete
════════════════════════════════════════════════════════════════════════════

  The full grid (ALL EM @N=16/32/64/128; majority 0.38/0.36/0.36/0.39):
    T1 canonical    7B leaf + 7B merge + 7B ans, canon prompts  0.23/0.20/0.18/0.21
    ctrl-C          7B + 7B + 14B ans, canon                    0.24/0.22/0.20/0.18
    ctrl-A          14B leaf + 7B + 7B, canon                   0.29/0.26/0.16/0.18
    ctrl-B          14B leaf + 7B + 14B ans, canon              0.33/0.33/0.22/0.19
    T1-MINIMAL v2   7B + 7B + 7B, MINIMAL prompts               0.36/0.29/0.28/0.22
  steps_in_room (the fully-instrumented type):
    canonical 0/0/0/0 → ctrl-B (14B interfaces) 1.00/1.00/0.88/0.50
                      → T1-minimal (7B-only!)   1.00/0.75/0.38/0.50
  ═ PREDICTION CHECK (logged 16:25 before landing): ctrl-B count-ceiling 0.995^(N−1)
  predicted ~0.93 @16 / ~0.53 @128 — MEASURED 1.00 / 0.50. The composition law is now
  PREDICTIVE, each factor measured independently. ═
  Findings hierarchy (the write-up's spine):
   1. Merge is sound at 7B (0.995 conditional); the leaks are the two interfaces.
   2. Leaf binding failure = margin × clutter (2×2: 7B 0.500→0.984 by prompt;
      14B 1.000 both) — EITHER scale OR framing repairs the leaf.
   3. Read-off is its own fragile interface (my Nobody-clause bug: even a CORRECT
      root note dies on one salient irrelevant instruction — instrument caught it).
   4. Repaired, the 7B-ONLY tree is real at short N (steps_in_room 1.00 @16) and
      decays lawfully (p^N above the floor now) — no fixed per-op fidelity survives
      long N; scale moves the crossover (ctrl-B 0.88 @64 vs minimal-7B 0.38).
   5. ALL stays below majority in every arm because 20/24 UNSEEN types have weaker
      model-designed asks — ask quality is the next wall after interface fidelity.
  Anomaly logged: first_at_room collapses in ctrl-A/B (0.12–0.50 vs 1.00 in every
  7B-leaf arm) — the 14B fills lookup-type notes with present-characters instead of
  null-propagation under canonical framing; per-type texture, not pursued.
  Assets: ctrl14b/{20260825_ctrlC_ans14b,20260825_leaf14b,20260825_ctrlA_tree,
  20260825_ctrlB_ans14b}/, t1_minimal/20260825_fs/ (v1 bug artifacts preserved),
  leaf_ablate/{20260825,20260825_14b}/. OPEN (needs Tal): 7B-compile armC cell
  (the "7B-only + exact executor" demonstration — proposed, one short job).

--- DRAFT 4 (extension: the interface law + repairs) ---

## [2026-08-25] ✅📊 TREEFOLD EXTENSION — the failure LOCALIZES to the two interfaces and the composition law becomes PREDICTIVE: merge is sound at 7B (conditional 0.995); leaf binding = margin×clutter (7B 0.500→0.984 by prompt-stripping; 14B 1.000 regardless); with repaired interfaces the tree is real at short N (steps_in_room 1.00 @N=16, 7B-ONLY) and decays as p^N — ctrl-B's pre-registered ceiling 0.995^(N−1) predicted 0.93/0.53 @16/128, measured 1.00/0.50

> Stage-swap grid `ctrl14b/` (14B = Qwen2.5-14B venv_arch at LEAF/ANSWER only; merge
> always 7B) + prompt ablation `leaf_ablate/` (V0 canon 0.500 / V1 minimal 0.984 /
> 14B both 1.000; FP|room-occupied 0.86→0.02) + repaired arm `t1_minimal/20260825_fs/`
> (v1 read-off bug — a stray Nobody clause turned correct numeric roots into "Nobody"
> — caught by the fidelity instrument, artifacts preserved, v2 rescored).

| steps_in_room EM | N=16 | 32 | 64 | 128 |
|---|---|---|---|---|
| canonical 7B tree | 0.00 | 0.00 | 0.00 | 0.00 |
| minimal prompts, 7B-only | 1.00 | 0.75 | 0.38 | 0.50 |
| 14B interfaces (ctrl-B) | 1.00 | 1.00 | 0.88 | 0.50 |
| predicted ceiling 0.995^(N−1) | 0.93 | 0.86 | 0.73 | 0.53 |

**Readings.** (1) The 7B *can* divide-and-merge — at short N, with clutter-free
interfaces; the negative headline becomes a boundary condition, not an impossibility.
(2) No fixed per-op fidelity survives long N (both repaired arms → 0.50 @128, on the
predicted merge-noise ceiling); scale only moves the crossover. (3) Answer-stage
fragility is real and instrument-detectable (Nobody-clause incident: correct roots,
wrong answers). (4) ALL EM stays ≈majority in every arm — residual is Ask quality on
the 20 UNSEEN types, not interface fidelity. **Caveats:** 2×2 ablation is
steps_in_room-only (384 frames); ctrl grids share the 7B merge; first_at_room
anomaly at 14B leaves (null-propagation schema filled with present-characters) noted.

- [2026-08-26] LOGGED: Tal said "log this" — DRAFTS 1–4 appended to RESULTS.md
  verbatim (order 1,2,3,4; drafts below kept for the record). Same message
  authorized the 7B-compile armC cell: scripts/treefold/armc_7b.py (THE campaign
  VL-7B as compiler, armC v3 SCHEMA/sandbox/stride/scoring imported verbatim;
  chat-templated generation, code from first fence), job 137358 (smoke LIMIT=6
  chained into full 768) → outputs/treefold/armc7b_compile/20260826/. Its result
  gets DRAFT 5, awaiting its own "log this".

- [2026-08-26 ~17:00] ═ armC-7B LANDED — THE 7B-ONLY + EXACT-EXECUTOR DEMONSTRATION ═
  (137461, 259s compute after a 137358 import-path false start — sys.path fix was
  already in the file at resubmit; outputs/treefold/armc7b_compile/20260826/.)
    ALL    0.64 / 0.60 / 0.61 / 0.63   (N=16→128) — LENGTH-FLAT
    SEEN   0.94 / 0.84 / 0.88 / 0.97   UNSEEN 0.58 / 0.56 / 0.56 / 0.56
    (14B compiler, same protocol: ALL ~0.89; the same 7B as tree EXECUTOR: 0.23→0.21
    declining, count types 0.) exec-fail 0.100, concentrated in 6 types (final_app,
    n_empty, first_at_room, last_at_room — IndexError/StopIteration/TypeError =
    genuinely wrong programs, not sandbox artifacts; 10 types compile 1.00 flat incl.
    steps_in_room/rooms_visited/crowd_count/n_room_on_*).
  READING: the arc closes. The SAME frozen 7B scores 3× higher as a COMPILER (0.63
  @N=128, one call per question) than as an EXECUTOR (0.21, ~2N calls) — at 1/180th
  the generation count (768 vs ~92k gens). Flatness comes from the architecture
  (exact N-fold execution), not the model; model scale buys only the CONSTANT
  compile-quality factor (0.62 vs 0.89, N-independent). This is the measured form of
  the campaign's final sentence: put the model where it runs O(1) times, keep the
  O(N) loop symbolic. DRAFT 5 below; awaiting its own "log this".

--- DRAFT 5 (armC with the 7B compiler — the 7B-only closing result) ---

## [2026-08-26] ✅📊 armC WITH THE CAMPAIGN 7B AS COMPILER — LENGTH-FLAT ALL 0.64/0.60/0.61/0.63 (N=16→128, oracle records, v3 prompt/sandbox verbatim): the same frozen VL-7B that scores 0.21-and-declining as an N-fold tree executor scores 3× higher as a one-call compiler; flatness is the architecture's property, scale only buys the constant compile-quality factor (14B: ~0.89)

> `scripts/treefold/armc_7b.py` (armC v3 SCHEMA/sandbox/stride/scoring imported
> verbatim; chat-templated generation, code from first fence), job 137461 →
> `outputs/treefold/armc7b_compile/20260826/` (768 programs, 259s).

| ALL EM | N=16 | 32 | 64 | 128 |
|---|---|---|---|---|
| 7B compiler + exact executor | 0.64 | 0.60 | 0.61 | 0.63 |
| 14B compiler + exact executor (armC v3) | 0.89 | 0.92 | 0.85 | 0.90 |
| same 7B as tree executor (best arm) | 0.36 | 0.29 | 0.28 | 0.22 |

**Readings.** (1) The "7B-only" system exists and is length-invariant: 10/24 types
compile 1.00 flat (incl. steps_in_room, rooms_visited, crowd_count, n_room_on_*);
UNSEEN holds 0.56 flat. (2) The 7B→14B gap is a constant (compile quality: exec-fail
0.100 vs 0.003, wrong-program mass in final_app/n_empty/first_at_room/last_at_room) —
NOT length-coupled; the p^N degradation that killed the tree is absent by
construction. (3) Cost inversion: 768 generations vs ~92k for the tree on the same
768 questions. **Caveats:** oracle records (perception factored out, as in armC v3);
chat-template extraction differs from the 14B's completion-style (fence parse);
exec-failures counted WRONG per convention.

--- DRAFT 1 (the central negative: model-as-executor) ---

## [2026-08-24→25] ❌📊 TREEFOLD — the frozen 7B as its own tree-fold executor FAILS the executor swap on every rung: oracle-leaf fan-2 tree ALL ~0.20 flat-below-majority vs armC v3's 0.89 on identical samples (exact execution is worth ~0.68 EM), and the per-level fidelity instrument localizes it: p_merge 0.20–0.70 per op, decaying with TREE DEPTH ~independent of N

> Campaign `outputs/treefold/` (brief + STATE + INDEX). One frozen Qwen2.5-VL-7B nf4 for
> every stage (Ask/leaf/merge/answer); few-shot Ask primary (Tal's checkpoint decision,
> armC-matched convention); samples/scoring byte-identical to armC v3 / Arm B.
> T1 `t1_oracle/20260824_fs/` · T4 `t4_fullstate/20260824_fs/` · figures `figs_20260825/`.

| ALL EM (N=16/32/64/128) | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| T1 tree, oracle records, 7B executor | 0.23 | 0.20 | 0.18 | 0.21 |
| armC v3: same records, Python executor | 0.89 | 0.92 | 0.85 | 0.90 |
| T4 tree, Arm B caption records | 0.25 | 0.24 | 0.18 | 0.23 |
| Arm B: same captions, Python executor | 0.79 | 0.81 | 0.72 | 0.72 |
| majority | 0.38 | 0.36 | 0.36 | 0.39 |

**Readings.** (1) H1/H4 decisively not met: swapping exact execution for model
execution costs ~0.5–0.68 EM on identical inputs. (2) The failure is LOCALIZED at the
interfaces, not the fold: conditional merge fidelity (both children correct) is 0.995
(2005/2016, ≥0.98 at every level) — the 7B executes left+right essentially perfectly,
and the per-level fidelity decay (F-C) is pure error propagation. What breaks is
(a) MAP — per-frame predicate evaluation on clean GT text fails 10–40%/frame (p_leaf
0.56–0.91; e.g. record "Office: Daniel; Bedroom: Sandra" + rule "count=1 if Daniel is
in the Bedroom" → count=1), and p_leaf^N alone (0.8^16 ≈ 0.03) explains the collapse;
(b) ANSWER read-off — 5 of the 8 samples with an exactly-correct root note still
answered wrong (root {"count": 8} → "Daniel"). (3) The composition check closes:
p_leaf^N·(cumulative p_merge)^(N−1) ≈ 0 ≈ measured EM on instrumented types.
(4) Null-propagation programs survive (single-frame lookups 0.75–1.00 flat to N=128) —
they are single-interface: one predicate evaluation, no accumulation. (5) T1 EM is
~length-flat — but flat below majority: flatness without competence. **Caveats:**
few-shot Ask (4 SEEN exemplars, armC parity); 8/cell per type; conditional-fidelity
n=2016 pooled over count-like types with an identified field; the pre-registered 14B
control (now aimed at leaf+answer stages, not merge) is proposed, not run.

--- DRAFT 2 (fan-in law + depth mechanism) ---

## [2026-08-25] ❌📊 TREEFOLD fan-in law: EM FLAT in k∈{2,4,8,16,N} at N=128 (0.16–0.21, all at the executor-noise floor) — no over-squashing signature AND no noisy-op inversion; per-level p_merge decays with composition depth ~N-invariantly (L1 0.59–0.69 → L4+ ≤0.29 at every N) — the wall is per-op semantic fidelity, which fan-in cannot route around

> T2 `t2_fan/20260824_fs_k{4,8,16,N}/` (k=2 = T1; leaves reused, k-independent);
> headline figure F-A + mechanism figure F-C in `figs_20260825/`. k=N ran at BATCH=4
> after an OOM at 48 (prompts carry 64–128 notes).

- k=N does ONE merge call and suffers ~zero parse deaths (1–2 vs 22–32 at k≤4) yet
  lands at the same EM — depth mortality and breadth squashing exactly net out.
- Pre-registered fallback stands: the fan-in/over-squashing link is NOT supported at
  this scale; the thesis' mechanism evidence remains the in-forward measurements
  (Hahn α=0.72 joint vs 0.00 fenced; capacity law c(fan)).
- H5 addendum: zero-shot Ask collapses harder with depth (0.073 vs few-shot 0.208
  @N=128; merge-parse 70/192 vs 25/192) — program quality depends on demonstrations.

--- DRAFT 3 (targeted perception) ---

## [2026-08-25] ⚠️📊 TREEFOLD T3 — question-conditioned per-frame JSON notes cost perception: count-like leaf accuracy 0.921 @512 (vs 0.995–1.000 for the 1-bit conditioned verdict, ≈0.913 for Arm B's full-state caption) — structured note EMISSION, not seeing, is the lossy step; end-to-end tree 0.31→0.16 with merge-parse mortality growing with depth (1/9/22/36 per 96)

> Captions `t3_leaves/20260824_fs/` (382 samples, 23,008 frames @512, parse 0.990);
> tree `t3_tree/20260824_fs/`. H3 not met on both rungs (0.921 < 0.98; ALL ≪ ArmB−0.03).
> The 0.995→0.92 drop vs the verdict-bit record isolates note-emission cost — relevant
> to any design that asks the VLM for structured per-frame state instead of a bit.
