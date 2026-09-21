# TREEFOLD campaign — message passing across forward passes (the no-tool, no-training, length-invariant aggregator)

**Status: AUTHORIZED (Tal, 2026-08-24) — GPU submissions pre-authorized, discipline rules §6 binding. Mandatory checkpoint: STOP after P0 for Tal's schema review before any tree run (T3 captions exempt).**
Scripts: `scripts/treefold/` (to be written after OK). Runs: `outputs/treefold/`. Log: `STATE.md`
(append-only, newest last). RESULTS.md is Tal's — DRAFT entries go at the bottom of STATE.md.

Context read before drafting: RESULTS.md tail ([2026-07-31]…[2026-08-22→23]), `outputs/recagg/`
INDEX + STATE + RELATED_WORK, `outputs/armor/` brief + STATE, `outputs/ninv` five-arm/cascade
entries, artifact "Where Over-Squashing Lives" (2026-08-24), peer thread 2026-08-24 (Chaim:
frame as over-squashing + GNN; Gabriele: lightweight, no code-generation tool use; Tal:
N-invariant architecture).

## 0. One paragraph

The joint forward computes over a degree-N star with a normalized aggregator (softmax) —
measured: answer-position sensitivity ∝ N^−0.72, at the bf16 floor by N=128. Rewiring that
graph *inside* the forward hit two walls (learned tree levels don't extrapolate in depth;
analog messages squash). TREEFOLD rewires it *across* forwards: the frozen model is the
message function on a **fan-2 tree** whose messages are **text notes** — weight tying and
re-quantization by construction. Ask (question → note schema + leaf rule + merge rule, in
words) → Map (one note per frame, question-conditioned) → Reduce (pairwise merges, log₂N
levels) → Answer. No training, no probe, no code generation, no executor. The campaign
measures (i) whether the reduce is length-flat, (ii) the **fan-in law** (2/4/8/16/N) that ties
it to the over-squashing degree term, (iii) targeted vs full-state perception, (iv) the
executor swap vs Arm C/B on identical samples, (v) zero-shot vs few-shot Ask.

## 1. Data, samples, baselines (all fixed, all shared with recagg)

- **Pools:** `data/mmred_hf/dirs/seq_len_{16,32,64,128}_test` (1200 dirs each), **512px**
  (canonical HF resolution, recagg policy 2026-08-17). 24 question types; `type_of(dirname)`
  and `parse_qa` from `scripts/ninv/load_hf_sample.py` / armC.
- **Sampling = armC's, byte-identical:** per-type strided (`dirs[::len//per_type][:per_type]`).
  `per_type=8` → 768 samples (armC v3 set); `per_type=4` → the 384-sample Arm B subset
  (`caption_frames.sample_dirs` subsetting rule). Every cell reports class distribution per N
  ([[mmred-hf-k0-sorted-trap]]) and the majority baseline next to every EM.
- **Baselines on the SAME samples (already on disk, no rerun):**
  armC v3 oracle+Python `armC_compile/20260817_193946_qwen14b_v3` — ALL 0.89/0.92/0.85/0.90,
  UNSEEN 0.88/0.93/0.85/0.88 · Arm B VLM-full-state+Python `armB_execute/20260817_213301_v3progs`
  — ALL 0.79/0.81/0.72/0.72 (steps_in_room 1.00→0.25, crowd_count 0.50→0.00 @128; caption
  fidelity 0.913/char, 0.637/frame) · Arm A all-notes-one-context ~majority by N=32–64 ·
  chunk-16 0.78/0.58/0.42 · best in-model zero-shot 0.087 @128.
- **Scoring:** armC's `norm`/EM (ints as ints, names/rooms case-insensitive, comma-lists as
  sets). SEEN = {steps_in_room, rooms_visited, where_spend, char_at_frame}; UNSEEN = other 20.
- **Failures are never silent:** Ask-parse-fail, leaf-parse-fail, merge-parse-fail,
  answer-parse-fail and "not combinable" are separate counters; each counts as WRONG.

## 2. The method as run (one frozen model does everything)

**One model for everything — Ask, leaf (image and text), merge, answer: Qwen2.5-VL-7B-Instruct,
frozen, nf4 (house runtime), text-only stages through the VL processor (Tal, 2026-08-24).**
Single frozen backbone = the cleanest statement of the method. The 14B is NOT in the campaign;
it exists only as a named fallback (§3, falsification of H1) and would need Tal's OK.

1. **ASK** (1 call / sample; input = question only, never the frames):
   ```
   {"note":   {<field>: "<type> — <meaning for a SEGMENT of frames>", ...},
    "leaf":   "<how to fill the note from ONE frame>",
    "merge":  "<how to fill the note for LEFT+RIGHT from left.note and right.note ONLY>",
    "answer": "<how to read the final answer from the root note>",
    "combinable": true|false}
   ```
   Conventions block (task definition, not a hint): 1-based steps; "appears" = in the room at
   that step; FIRST/FINAL appearance = earliest/latest such step; "together" = same room same
   step; "alone" = only person in that room at that step; rooms/characters listed.
   **Zero-shot** = format + conventions, NO example questions (primary).
   **Few-shot** = + the same 4 SEEN exemplars armC used, rewritten as notes (control).
   `combinable=false` ⇒ the question is declared outside the class; counted as WRONG, reported.
2. **MAP** — two leaf modes:
   - **VLM-conditioned** (the method): per-frame call, prompt = frame + question + leaf rule
     + note schema → JSON note. N independent single-image calls at 512 (multipass = measured
     fenced-forward equivalent, A3 parity; the one-forward fenced emission is an efficiency
     arm, not run here).
   - **text-leaf** (isolation arms): the LM applies the leaf rule to a text record — either
     the ORACLE state (`parse_qa` states) or Arm B's full-state caption.
3. **REDUCE** — balanced binary tree over frame order; each merge = 1 call: schema + merge
   rule + `left` (frames i–j) + `right` (frames j+1–k) → JSON note. All merges of a level
   batched (N/2, N/4, …). Fan-in `k` generalizes: k notes per call, ⌈log_k N⌉ levels; k=N is
   one call over all notes (= Arm A).
4. **ANSWER** — root note + question + answer rule → string.

Token budget per sample at N=128, k=2: 1 + 128 + 127 + 1 short generations (≤48 new tokens
each), batched by level → ~10 batches. Perception: 128 image calls (Arm B measured 23k
frames @512 in 56 min on one rtx6k).

## 3. Experiments and pre-registered bands

### P0 — harness + Ask quality (GPU-light)
Smoke every script at LIMIT≤3, N=16 → `outputs/_scratch/treefold_smoke/`. Then Ask on all
768 questions × {zero-shot, few-shot} with the 7B. Report parse rate, `combinable` rate,
per-type schema audit (asks.json saved for audit, like programs.json).
**Band P0:** zero-shot 7B parse rate ≥ 0.95 and combinable ≥ 0.95. Below ⇒ few-shot becomes
primary for T1–T4 (stated in STATE before any tree runs).

### T1 — ORACLE-LEAF TREE: is the reduce length-flat? (isolates Ask + Reduce)
Oracle states → text-leaf (7B) → fan-2 tree → answer. per_type=8, N∈{16,32,64,128} (768).
Per-level fidelity for **answer-equals-note types** (count-like: steps_in_room, crowd_count,
n_empty, spend_alone, spend_together, rooms_visited, n_room_on_*, n_char_at_frame): gold
partial note per subtree = armC's v3 program run on that subtree's frames (programs.json,
sandbox, read-only) → p_leaf, p_merge per level; prediction check EM ≈ p_leaf^N·p_merge^(N−1)
vs measured (log both).
**Band H1 (flat reduce):** ALL ≥ 0.85 at every N **and** max−min over N ≤ 0.05; UNSEEN ≥ 0.80;
p_merge ≥ 0.98 at every level. Compare to armC v3 on identical samples (executor swap, oracle).

### T2 — THE FAN-IN LAW (the mechanism figure; peers' figure)
Same oracle leaves, same reducer, **k ∈ {2, 4, 8, 16, N}**, N∈{64,128}, all 24 types
(count-like subset reported separately). k=N = Arm A protocol with our notes.
**Band H2:** EM non-increasing in k on the count-like subset at N=128; EM(k=2) − EM(k=N) ≥ 0.40;
EM(k=16) ≤ EM(k=2) − 0.20. Prediction from the capacity law ([2026-08-10b]): k=4 ≈ k=2,
k=8 mild drop, k=16 ≈ chunk-16 (0.42), k=N ≈ majority.

### T3 — QUESTION-CONDITIONED VLM LEAVES: the end-to-end method (no probe, no oracle)
Leaf rule + schema in the per-frame VLM prompt @512, per_type=4 (384 samples = Arm B subset),
N∈{16,32,64,128} → fan-2 tree → answer. Also log **per-frame leaf accuracy vs GT** (the p that
Arm B lacked: it had 0.913/char, 0.637/frame for full state).
**Band H3 (targeted perception):** per-frame leaf accuracy ≥ 0.98 on count-like types;
steps_in_room @128 ≥ 0.50 and crowd_count @128 ≥ 0.25 (Arm B: 0.25 / 0.00); ALL ≥ Arm B − 0.03
at every N. Predicted ALL @128 ≈ 0.70–0.80 (perception-bounded, p_leaf^N on count types).
**Long pole — submit its captions first** (~1 h/rtx6k for 23k frames).

### T4 — FULL-STATE LEAVES (control; zero perception GPU)
Arm B captions on disk → text-leaf → fan-2 tree → answer (384 samples). Two contrasts:
T4 vs Arm B = **model-merge vs Python on identical records** (executor swap, VLM records);
T3 vs T4 = **targeted vs untargeted perception**, same reducer.
**Band H4:** |T4 − Arm B| ≤ 0.05 at every N (the model is an adequate executor at fan-in 2).

### T5 — ASK ZERO-SHOT vs FEW-SHOT (the reasoning-graph claim)
T1 protocol under few-shot Ask at N∈{16,128} (half cost).
**Band H5:** zero-shot ALL ≥ few-shot ALL − 0.05 at both N. Larger gap ⇒ "program quality
depends on demonstrations" — reported as such, few-shot rows kept.

### Falsification, written down
- H1 fails with p_merge ≥ 0.98 ⇒ Ask (schema) is the problem, not the tree → per-type audit.
- H1 fails with p_merge < 0.98 ⇒ the frozen 7B is not an adequate fan-2 executor → propose a
  14B merge control (one extra cell, needs Tal's OK) before any conclusion.
- H2 non-monotone ⇒ the fan-in/over-squashing link is not supported at this scale — the
  paper's mechanism claim reverts to the in-forward measurements only.
- H3 leaf accuracy < 0.95 ⇒ question-conditioning does NOT buy perception; T4 becomes the
  primary end-to-end arm.

## 4. Deliverables (figures the paper/peers get)
- **F-A fan-in law:** EM vs k (log-x: 2,4,8,16,N) at N=128, count-like + ALL, majority line,
  capacity-law prediction overlaid. *The headline figure of the campaign.*
- **F-B length curves on identical samples:** T1 · armC v3 · T3 · Arm B · chunk-16 · best
  in-model · majority, N=16→128.
- **F-C per-level fidelity:** p_leaf, p_merge per level; predicted vs measured EM.
- **F-D perception:** per-frame accuracy targeted (T3) vs full-state (Arm B) and p^N
  prediction vs measured on steps_in_room/crowd_count.
- Tables: per-type × N for every cell (armC report format); failure-counter table; token/
  call budget per sample vs N (cost curve, O(N) small-constant vs joint O(N²)).

## 5. Scripts to write (`scripts/treefold/`; armC/caption_frames code imported, never edited)
`ask.py` · `leaf_vlm.py` (caption_frames pattern + leaf rule) · `leaf_text.py` ·
`tree_reduce.py` (fan-k, batched per level, strict JSON, per-level fidelity hooks) ·
`answer_score.py` (armC norm/EM, per-type report) · `run_cell.py` (one cell end-to-end,
resumable from saved asks/notes) · `fig_treefold.py` · `slurm/treefold_{ask,leaf_vlm,tree}.sbatch`
(env-var driven, `slurm/lib/common.sh`, explicit `--time`, no comma `--export`).

## 6. Discipline rules (inherited from ARMOR, binding)
Smoke first in `_scratch`; check `sinfo` across ALL partitions; right-size QOS with explicit
`--time`; no comma-lists in `--export`; no pip (missing dep ⇒ STOP, note in STATE); nothing
in `gnnformer/` touched (pure orchestration) — if it is, run `tests/`; leakage suspicion ⇒
stop the arm and log it; majority baseline + class distribution beside every EM; every
generated ask/note/merge saved for audit.

## 7. Order & pooling
P0 smoke → **T3 captions submitted early (long pole)** → P0 Ask → T1 → T2 → T4 → T3 tree →
T5. T1/T2/T4 are text-only cells (a100-public / l40s-public; 12h_4g or 24h_1g); T3 captions
on rtx6k/l40s as Arm B. End state: STATE.md verdicts vs bands H1–H5, figures + CSVs in run
dirs, INDEX.md rows, DRAFT RESULTS.md entries at the bottom of STATE.md.

## 8. Related work the write-up must cite (verified 2026-08-24)
LLM×MapReduce 2410.09342 (ACL 2025) · ToM tree-MapReduce 2511.00489 (EMNLP 2025) · Chain of
Agents 2406.02818 (linear fold) · **When Does Divide-and-Conquer Work 2506.16411 (ICLR 2026;
closest framing — "model noise grows with length" is our α, un-localized)** · DaC prompting
2402.05359 · Recursion of Thought 2306.06891 (learned decomposition does NOT length-generalize
— mirrors the register tree) · VideoTree 2405.19209 (tree for selection, fan-N at the top =
Arm A) · Video ReCap 2402.13250 · tool-use SSMs 2510.14826 · ViperGPT 2303.08128.
Delta sentence: *they treat length degradation as noise and the tree as a heuristic; we locate
the degradation in softmax normalization (α measured in the weights), show fan-in and message
discreteness are the two knobs that remove it, and derive the merge from the question.*
