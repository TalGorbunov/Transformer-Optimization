# PHASE 3 — the full-thesis 16-hour program: solve the READOUT, prove extrapolation, port the model, run the ablations

> **Agent instructions.** You are the long-running tmux session that ran Phases 1-2. Obey
> CLAUDE.md. Log EVERY result to `plans/carrier_stage2_DRAFT_RESULTS.md` + INDEX rows
> IMMEDIATELY as it lands (never batch); do NOT touch RESULTS.md/docs/. After EVERY phase
> transition, overwrite `plans/carrier_stage3_STATE.md` with: what landed (numbers + run dirs),
> what's running (job ids), what's next — so any successor session resumes instantly.
> **Pacing / session budget:** you must run ~16 h unattended. Prefer few large sbatch jobs over
> many small ones; while jobs run, wait with CHAINED synchronous bounded Bash waits (`until …;
> do sleep 600; done` capped ~10 min per call, then re-issue) — never busy-poll faster, never
> stop with only a passive watcher armed. Keep context lean: don't re-read large files; trust
> STATE.md. If you die on a session limit, the state file is the handoff.
> Budget ≤ ~35 GPU-h. Blocked → `plans/carrier_stage3_BLOCKED.md`, move on.

## Current state (2026-07-19 ~01:00)

Stage-2 GO: P1 (job 123741, 3-task 6k, trainable e_c) 0.963@ep5 and climbing — collect its
final. Cached arm (job 123937, 2-task 5.1k, frozen e_c) BEST 0.980@ep10 — collect final.
Scaffold 0.998 / frozen 0.219 / chance 0.111. Exams of the cached ckpt submitted by the main
session (collect FIRST): 124275 length N=32, 124276 rooms zero-shot, 124277 LoRA-drift on plain
prompt. Old h200 eval 123744 targets an OBSOLETE ckpt — scancel it, resubmit against the
pooled ckpts. Known negatives driving this phase: the silent readout does not extrapolate
(450-ckpt: N=32 0.138, cooc 0.179); supply/carrier DO transfer (d′ 11.4 @N=32 zero-shot; 88%
cross-task).

## The program (strict order within tracks; tracks A/B/C can interleave via the queue)

**TRACK A — the READOUT SOLUTION (the centerpiece): verdict-scratchpad + position jitter.**
A1. Extend `carrier_layer_lora.py` AND `carrier_layer_cached.py` training targets from a single
    digit to a teacher-forced token sequence: `"frames 2, 5, 7 -> 3"` (evidence indices then
    count; `"none -> 0"` for empty; rooms: `"rooms Garden, Park -> 2"`). CE over the full
    target; eval = greedy decode (≤24 tokens) + parse the integer after `->` (report parse-fail
    rate). Per-frame labels exist in prep (parse_task_labels evid sets) — use them for targets
    ONLY (they are legitimately available at training time; never at eval).
A2. Carrier POSITION JITTER: during training, add one random offset per sample to all carrier
    sequential positions (e.g. uniform 0..96); eval without jitter. Kills position-range
    binding (the readout has then seen carrier positions spanning the N=128 range).
A3. Train the scratchpad+jitter recipe on the cached fast trainer (steps 2-8 + cooc + rooms,
    LIMIT=900/root, 15 ep, frozen e_c) — smoke n=16 FIRST (verify targets render, decode
    parses, loss falls). Then THE EXAMS, all zero-shot on the resulting ckpt:
    N=32 and N=128 length (h200 for 128), held-out-task if a task was held out, count-OOD
    (the scratchpad reads out multi-digit counts natively — report per-count).
    Pre-registered bands: N=32 ≥0.80 (scaffold 0.917) = READOUT SOLVED for length; 0.5-0.8
    partial; ≤0.5 = scratchpad insufficient → log honestly, try A4.
A4 (only if A3 partial): add N=16/32 data (LIMIT 200 each) to training and re-exam.

**TRACK B — MODEL-AGNOSTIC: InternVL2.5-8B port (timebox 4-5 h, then BLOCKED if stuck).**
B1. Port the REPLICA PROBE (supply level only — no training): InternVL uses standard 1D RoPE
    (posreset = simple offsets) and its own image tokenization. Deliverable: joint vs
    blockfence+qfirst d′ at its carrier layer (~L20; existing InternVL d′ machinery in the
    repo — find it via `grep -ril internvl evaluations/ experiments/`). GO = replica/blockfence
    d′ ≥ 2× joint on 200-300 samples. This alone licenses "the mechanism ports"; the trained
    stack on InternVL is NOT required in this phase.

**TRACK C — ABLATIONS + BREADTH (cheap cached-trainer runs + CPU; fill GPU gaps).**
C1. L_OPEN ∈ {12, 22} and rank ∈ {4, 16} — one change each, cached trainer, steps+cooc only,
    8 ep (enough to rank; full convergence not needed).
C2. No-Qfirst ablation (question only at the end) and no-posreset ablation — same recipe;
    quantifies each architecture piece at the BEHAVIOR level (they're priced at supply level
    already: Qfirst +3 d′, posreset +0.6).
C3. NIAH/existence task: generate yes/no + which-frame questions from existing states (small
    script, CPU) for seq_len_8; eval the P1 ckpt zero-shot AND add to a scratchpad mixture if
    time permits. Pre-register: the theory says NIAH is the easy case — expect ≥0.9 once in
    the mixture.
C4. Collect + log EVERYTHING: P1 final, cached final, exams 124275-77, all Track A/B/C runs.

## Endgame deliverable (write BEFORE you run out of time)

At the end of `plans/carrier_stage2_DRAFT_RESULTS.md`: the FULL-THESIS TABLE — one row per
claim {in-model accuracy per task, length extrapolation, task transfer, model port, drift,
each ablation}, each with number + run dir + band verdict; plus a 10-line "what remains"
list. Update `plans/carrier_stage3_STATE.md` one last time.

## Pitfalls (Phase-1/2 list applies verbatim; new ones)

- mem-efficient SDPA rejects fp32-mask+bf16-query under no_grad → force MATH backend in any
  new no_grad forward path (see carrier_layer_cached.py MATH_SDPA).
- The distill/carrier ckpt head_w is untrained; alpha=16 hardcoded in drift-test hook loading.
- Scratchpad targets: keep them SHORT and fixed-format; verify tokenization roundtrip in the
  smoke; report parse-fail rate separately from accuracy.
- Jitter: positions must stay < the model's context (trivially true); jitter TRAIN only.
- scancel 123744 (obsolete-ckpt eval) before queueing new h200 work.

## Discretionary follow-ups (added 2026-07-19)

You are AUTHORIZED to design and run follow-up experiments beyond the tracks above whenever
they strengthen the thesis and the primary tracks are not starved of GPU or budget. Rules:
pre-register the question + bands in the draft BEFORE running; one change at a time; smoke
first; log like everything else; prefer cheap/decisive over broad. Pre-approved ideas, in
rough priority order:
- **Composition eval** (stress-test of the "programmable reduction" claim): generate questions
  requiring comparing two tallies ("did C spend more frames in the Park than the Garden?")
  from existing states (CPU script); eval the scratchpad ckpt zero-shot. Never-trained
  reduction → the sharpest task-agnostic demonstration.
- **Running-tally scratchpad format** ("frame 2 (1), frame 5 (2), ...") as an A3 variant if
  plain scratchpad shows count errors at long N.
- Attribute-bearing verdicts ("frame 3: John, Park") if rooms/cooc scratchpad underperforms.
- Anything else you can justify in one paragraph in the draft — the standard is "does this
  cell strengthen the final thesis table", not "was it listed here".
