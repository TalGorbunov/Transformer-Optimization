# PHASE 4 — extrapolation verdicts, layout freedom, no-harm, seeds (autonomous until Tal returns)

> Same discipline as Phase 3 (read its brief header if unsure): CLAUDE.md rules; log every
> result to plans/carrier_stage2_DRAFT_RESULTS.md + INDEX **immediately**; overwrite
> plans/carrier_stage4_STATE.md at every phase transition; chained ~10-min synchronous waits,
> never stop with only passive watchers; smoke everything; pre-register bands; honest
> negatives are findings. Budget ≤ ~30 GPU-h. Blocked → plans/carrier_stage4_BLOCKED.md.

## Collect FIRST (jobs launched by the main session)

- **124482** headline: 5-task ≤16 running-tally+jitter (cached trainer) — THE spec arm.
- **124483** upper: same + longN_32/64.
- **124484** SFT control: plain-LoRA steps ≤8 (lora_sft_baseline.py, r8) — summary only at END
  (walltime kill loses everything — do not cancel it early!).
- **124317 / 124377** N=128 baseline exams — STILL pending on h200 (queue held by h200-dds).
Log each with a RESULTS-style entry as it lands.

## E-A — the EXTRAPOLATION VERDICTS (top priority, right after 124482/83 land)

Zero-shot exams of BOTH ckpts (`carrier_tally_le16/`, `carrier_tally_le64/` best.pt) via the
eval-only path: N=32 and N=64 (a100, LIMIT=300/200, held-out dirs — use eval_dirs conventions;
for longN_32/64 roots the le16 ckpt never saw ANY of it = clean; for le64 use the
train_dirs.txt complement), N=128 (h200, LIMIT=65). Report acc / in-range acc / parse-fail /
MAE / per-count. **Pre-registered: le16@N=128 ≥0.80 = THE thesis headline (train ≤16 →
extrapolate 128); 0.5–0.8 partial → E-F; ≤0.5 = tally insufficient → diagnose verdict-
detection vs counting from the transcripts (log 20 sample decodes).** Fill the doubt-proof
table (draft end) as cells land.

## E-B — SFT control long-N eval

124484 saves adapters + writes summary.csv (train-length test acc). Then eval its adapter at
N=16/32/64: its script has no long-N eval mode — write a small generate-and-parse pass
(reuse its own eval helpers; plain prompt, adapter loaded via peft). Bands: theory predicts
collapse at N≥32 (joint supply d′~2). If its plumbing resists >2h, BLOCKED-note it with the
train-length numbers only.

## E-C — LAYOUT FREEDOM: carriers-at-the-end (Tal's design question, spec below)

Goal: kill the "question must be pasted first" objection. Add `--carriers-at-end` to
`carrier_token_distill.py`:
- content = [frame_1..N][question][c × N] (each carrier a separate single-token text item);
  NO leading question.
- Mask: frame blocks fenced as usual (block-diagonal incl. markers); carrier_i attends
  {prefix, question span, frame block i, itself}; carriers hidden from all non-carrier rows.
- Positions: frames share block-0 range (posreset as usual); question rebased right after
  block-0 (existing tail logic); carriers sequential after the question.
- Teacher for distillation: the FRAMES-FIRST blockfence replica (A3 layout — the in-run
  teacher WITHOUT --question-first), anchor ≈ 6.3.
- Train distill (~1-1.5h), report eval d′ + gate→tally @N=8.
**Bands: eval d′ ≥5 (~80% of the 6.34 teacher) AND tally within a few points of the Q-first
carrier ⇒ layout-freedom GO — the method statement becomes prompt-order-free.** Smoke n=8
first (mask-debug: carrier rows' allowed-keys = prefix + question + one frame + self).

## E-D — NO-HARM benchmarks (no installs needed — verified)

`datasets` 4.5.0 reaches lmms-lab/MME (~860MB) and lmms-lab/POPE (~255MB). Write
`experiments/glstm/noharm_bench.py`: load via datasets, run our runtime (gri.configure_runtime),
score MME yes/no per-subtask accuracy + POPE accuracy/F1. Two arms on identical samples:
(a) base model; (b) LoRA hooks active on plain prompts (loading pattern:
`frozen_baseline_eval.py --lora-ckpt`, alpha=16 hardcoded) with the le16 ckpt. Subsample to
~500 items/benchmark for budget. Report per-subtask deltas. Band: |Δ| ≤ 2 pts ⇒ no-harm GO.

## E-E — SEEDS: rerun the headline arm with seeds 1 and 2 (cached trainer, same everything,
`--seed`/`--shuffle-dirs` 1/2). Report mean±std for the headline cells. (Error bars are the
cheapest credibility upgrade in the whole program.)

## E-F (conditional) — if le16@N=128 is partial: generate more long-N data (find the longN
generator scripts via `grep -rl longN evaluations/ experiments/ | head`; make ~300 more N=32
and ~200 N=48 samples, CPU 4h_0g) and retrain the UPPER arm only; re-exam. Do NOT touch the
le16 headline arm (its claim is fixed).

## Endgame

Update the FULL-THESIS TABLE (draft end) with every new cell; one-screen Phase-4 summary;
final STATE.md. Known pitfalls: all Phase-1/2/3 lists apply (dtype/backends: prep mask bf16 +
hi mask fp32 + [EFFICIENT, MATH] fixed list; e-sorted dirs; AUC caps; h200 = only card fitting
N=128; per-QOS caps; --export env passthrough).

## DO NOT STOP (added at Tal's instruction)

You do not stop working until Tal returns — completing the program above is NOT a stopping
condition. If E-A…E-F are done (or blocked) and no jobs are running, CONTINUE with
thesis-strengthening follow-ups, in this order of value, forever until interrupted:
1. Any "What remains" item from the Phase-3 list at the end of the draft (rooms decode gap;
   composition/comparative questions generator + eval; L12+pooled-data run; natural-images
   carrier layer; HERBench expected-null; InternVL trained stack).
2. More seeds on any headline cell still single-seed.
3. More long-N data generation + upper-arm retraining iterations (each iteration = a better
   extrapolation curve point).
4. Deeper diagnostics of whichever table cell is weakest (transcript analyses are free).
Rules stay: pre-register in the draft before each run; one change at a time; log immediately;
keep STATE.md current; right-size QOS; never idle while a GPU could be answering a thesis
question. The only valid waiting state is a chained synchronous wait on a running job.

## BUDGET OVERRIDE (2026-07-20, Tal, verbatim authority)

GPU budget is NOT a constraint — run as many experiments as the science needs. All GPU-hour
caps in this brief and prior briefs are void. What remains binding: cluster etiquette
(CLAUDE.md — right-size QOS, spread across QOS to beat per-QOS caps, check idle GPUs across
ALL partitions, never block others needlessly, smokes to _scratch), pre-registration, and
logging. Never freeze work for budget reasons again; if a queue is saturated, work on
something else in parallel instead of waiting idle.

## E-G — POSITION-COUPLED TALLY (added 2026-07-20, Tal-approved; run alongside/after E-F)

Mechanism (Position Coupling, arXiv:2405.20671 + 2410.15787): give every generated verdict
token the POSITION of the carrier it describes, so "attend my carrier" is a ~zero-distance
lookup at every N — length stops being a distribution shift for the readout. We already own
arbitrary position assignment; this extends it to the target/generated rows.

Implementation (carrier_layer_cached.py train + carrier_layer_lora.py eval-only):
1. TRAIN (teacher forcing): build_target_tally already knows which frame each verdict segment
   describes. Emit alongside tgt_ids a per-token position-anchor list: all tokens of the
   segment for frame i (index, "(k)", comma) → carrier_i's position id; the leading " frames"/
   " rooms" token(s) → carrier_1's position; the tail " -> <count><EOS>" → last-carrier pos
   +1, +2, …; " none -> 0" → carrier_1 pos onward. Splice these into the pos tensor for the
   appended target rows (replacing the current sequential tail positions). Two sub-arms in the
   SMOKE only: (a) whole segment shares the carrier's exact position; (b) segment anchored at
   carrier pos with +0,+1,+2 intra-segment increments. Pick by TF loss/decode sanity, then one
   full arm only.
2. EVAL (greedy): positions must be assigned ONLINE with the SAME deterministic rule — parse
   as you decode: when a frame-index number is emitted, set anchor = that carrier's position;
   tokens couple to the current anchor (+increments if arm b); on "->" switch to tail rule;
   invalid/out-of-range index → keep previous anchor. This rule must match training exactly —
   verify with a round-trip check in the smoke (teacher-forced positions == online positions
   on the same string).
3. Jitter composes: anchors follow the (jittered) carrier positions automatically at train.

Runs: same data as E-F (upper roots + longN_park2) so E-F vs E-G is a clean coupled-vs-not
pair at matched data. Smoke n=16 (print one sample's segment→anchor map + round-trip check +
loss falls + decode parses). Then full train (~like le64 arm), exams: held-out N=32/64 +
N=128 (h200 ops lesson). Optional cheap ablation after: coupled-train × UNcoupled-decode.
Pre-registered bands (vs the matched E-F cells): GO = beats E-F at every OOD length and
parse-fail stays ~0; the N=128 cell is the prize. Log per-count + 20 transcripts per exam.

### E-G claim notice (2026-07-20): E-G is being implemented and run by the MAIN session — do
NOT implement or launch it yourself; treat the E-G section above as reference only. Your queue
remains: E-F retrain, SFT completion, N=128 collection, DO-NOT-STOP list.

### E-G handoff update (2026-07-20, later): implementation is DONE by the main session
(`couple_offsets` in carrier_layer_lora.py — shared by teacher-forcing and decode; `--pos-couple`
in carrier_layer_cached.py; eval auto-detects via the ckpt's pos_couple flag; rooms uncoupled by
design). The AGENT now owns shepherding E-G to completion: collect smoke 124700; if the main
session has not already launched the full train (check squeue + outputs/ladder/image_longN/
carrier_tally_pcouple*), launch it: CKPT=<distilled carrier_best.pt>, EXTRA_FLAGS="--pos-couple
--jitter-gap 16 --grad-ckpt --shuffle-dirs 0", DATA_ROOT = the E-F recipe roots (upper roots +
longN_park2 32/48), LIMIT=900, EPOCHS=4, mem 200G, OUTPUT=outputs/ladder/image_longN/
carrier_tally_pcouple. Never duplicate a run that exists — check first. Then its exams
(held-out N=32/64; N=128 with the h200 ops lesson), logged against the matched E-F cells
(pre-registered: GO = beats E-F at every OOD length, parse-fail ~0). Smoke acceptance: the
[couple-debug] map shows verdict tokens anchored to their carriers (c<m>+k pattern), TF loss
falls, eval decode parses.

### E-D(b) addendum (2026-07-20, Tal request): SFT no-harm arm — when the SFT rerun (124696)
finishes and its peft adapter is saved, extend noharm_bench.py with a --peft-adapter mode
(load via peft PeftModel on the same base) and run the SAME 500 MME + 500 POPE items with the
SFT adapter active on plain prompts. Fills the third column of the no-harm table (base / ours /
SFT-LoRA). Pre-register: no band — this is descriptive; SFT touched all layers' behavior via a
task-narrow objective, so a larger drift than ours (−0.2/−1.4) is plausible and would be a
pro-method finding; ~0 is also fine (both adapters are small).
