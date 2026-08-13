# Mission brief — Stage-2 carrier layer: the CLEANEST method (no gate/tally, task-agnostic, length-extrapolating, cross-frame)

> **Agent instructions.** Long-running autonomous session. Read `CLAUDE.md` FIRST and obey it
> (partitions/QOS rules, never pip install, smokes → `outputs/_scratch/`, plan cheap→decisive).
> Do NOT edit `RESULTS.md` or `docs/`. Write draft entries to
> `plans/carrier_stage2_DRAFT_RESULTS.md` + append rows to `outputs/ladder/INDEX.md`.
> If blocked, write `plans/carrier_stage2_BLOCKED.md` and move on. Budget ≤ ~30 GPU-h total.

## Context (read once; details in RESULTS.md last 3 entries + plans/oneforward_DRAFT_RESULTS.md)

The 2-day record (2026-07-17/18): question replicas + FULL block-diagonal fence + per-block
M-RoPE reset = multipass supply in ONE forward (d′ 6.34 = solo 6.01); Q-FIRST amplifies (9.24 @N=8,
12.67 @N=32, 11.57 @N=128 — question-conditioned frame ENCODING beats even isolated forwards);
supply flat to N=128; a learned CARRIER TOKEN (1 embedding, 3584 params, `<|box_start|>`) distills
the replica at 92-93% (eval d′ 8.35), is length-general (N=8-trained → 9.7 @N=128) and
task-general (steps→cooc zero-shot 5.58 = 88% of that task's ceiling). Gate→tally behavior (the
SCAFFOLD, to be replaced): steps 0.960-0.997 @N=8 … 0.876-0.984 @N=128; cooc 0.973/0.932/0.680.
STAGE-2 ("carrier layer", `experiments/glstm/carrier_layer_lora.py`): fence ≤L16, cross-carrier
attention ≥L17 (sequential carrier positions; tail attends carriers), zero-init LoRA r8 on
q/k/v/o of L17-27 (~2M params) + e_c, plain LM loss on the answer digit, model answers via its
OWN head. First run: 0.353→0.853 @12ep UNDERTRAINED (still climbing); no-LoRA ablation plateaus
~0.50 (cross-carrier attention alone cannot aggregate — trained integration required; keep this
ablation in every table). Scaffold ceiling on the same data: 0.997.

**Target = the thesis method:** ONE architecture, trained once on a task mixture, that (a) needs
NO task-specific gate/tally — the model emits the answer; (b) extrapolates in length (train N=8,
eval N=32/128); (c) does cross-frame reasoning (rooms-visited = set-union across carriers — a
reduction a per-frame tally provably cannot do); (d) all numbers on the FULL count prior.

## In-flight jobs — collect these FIRST

- 123205 `frozen_base` — frozen-model baseline on the same N=8 sample subset
  (`experiments/glstm/frozen_baseline_eval.py`), for the same-prior comparison.
- 123206 `carrier_layer` — stage-2 LoRA rerun with EPOCHS=30 (was climbing at ep12).
- 123208 `carrier_tok` — steps-trained carrier eval-only on `mmred_natural/dist_far` (n=50).
Logs `logs/<name>-<job>.out`; log results to the draft + INDEX.

## Experiments (in order; one change at a time; pre-registered bands in [])

**E1 — full-prior fix (cheap, do immediately).** `data/mmred_images_park/seq_len_8/all_uniform`
has 900 dirs = 100 per gold e0..e8, name-sorted so LIMIT=300 gives gold∈{0,1,2} ONLY (majority
0.333 — the audit caveat in RESULTS). Fix: either LIMIT=900 (preferred; runtime ~3× but N=8 is
cheap) or add a `--shuffle-dirs SEED` flag (stratified) to the three scripts
(`replica_carrier_probe.py`, `carrier_token_distill.py`, `carrier_layer_lora.py`). Then rerun on
the full prior: (a) Q-first blockfence probe N=8 (+gate→tally CPU), (b) frozen baseline
(LIMIT=900), (c) carrier-token distill (train 450/eval 450), (d) stage-2. These become the
HEADLINE N=8 numbers. [expect: probe d′ ≈ unchanged (prior-free); gate→tally vs majority 1/9;
frozen baseline ≈ historical ~0.2]

**E2 — stage-2 convergence + minimal sweep.** After 123206: if best acc still at final epoch,
extend to 60 ep (one job). Then ONE-change sweeps, 2-3 jobs total: L_OPEN ∈ {12, 22} (vs 17),
lr_lora 3e-4. Report eval acc + MAE + per-count row. [GO ≥ 0.95 on the full prior at N=8 —
i.e. match the scaffold in-model; 0.85-0.95 = partial; keep the no-LoRA row]

**E3 — stage-2 LENGTH extrapolation.** Add `--eval-only --ckpt` STREAMING mode to
`carrier_layer_lora.py` (mirror `carrier_token_distill.py --eval-only`: no RAM cache, no
training; load e_c + LoRA A/B from `carrier_layer_best.pt` and register the hooks). Eval emitted
accuracy at N=32 (`mmred_longN_park/seq_len_32`, a100) and N=128 (seq_len_128, h200 — seq≈28k,
math-attention needs the 140GB card; LIMIT=100). NOTE the longN priors include counts 12/16/24/
32/128 — the model must EMIT multi-digit numbers there; extend the answer readout from
digit-argmax to a short greedy decode (≤3 tokens) + integer parse, and train... e_c/LoRA were
trained only on 1-digit answers at N=8 — report both raw and 0-9-restricted accuracy; if
multi-digit fails, report per-count and say so — that boundary is itself a finding.
[bands: ≥0.85 @N=32 = length-extrapolating in-model GO; the scaffold zero-shot stack read
0.917/0.860]

**E4 — task-agnostic training (the core deliverable).** Extend `carrier_layer_lora.py` with a
task-mixture loader: steps (`mmred_images_park/seq_len_8`, full 900) + cooc
(`mmred_cooc_balanced/seq_len_8`, reuse the `--task cooc` evidence/label logic from
`carrier_token_distill.py`). Two arms: (a) train steps-only → eval cooc emitted ZERO-SHOT;
(b) train the 50/50 MIXTURE → eval both + held-out generalization. Same e_c + one LoRA for all
tasks; the question in-context is the only task signal. [GO: mixture ≥0.90 on both; zero-shot
arm quantifies the transfer gap. Scaffold refs: steps 0.997, cooc 0.973]

**E5 — CROSS-FRAME reasoning: rooms-visited.** Data: `data/mmred_rooms_balanced` or
`mmred_rooms_1char` at N=8 (inspect format first; per-frame room labels from states) +
`mmred_longN_rooms_visited/seq_len_32` for length. The task needs set-union across carriers —
a per-frame tally CANNOT express it (the K-channel record: linear 0.40 < threshold 0.65 ≪
decode-then-count 0.99). Add rooms to the mixture (answer = #distinct rooms, still ≤9 → single
digit); eval emitted. [GO ≥0.90 = the carrier layer performs a provably-nonlinear cross-carrier
reduction in-model — the cross-frame chapter's headline. Compare vs the 0.993 multiclass-gate
pipeline number.]

**E6 (stretch, only if E2-E5 land) —** InternVL2.5-8B probe port (model-agnostic leg; 1D RoPE
makes posreset simpler; existing d′ machinery in the repo targets L20) · HERBench expected-null
· natural-images carrier-layer eval.

## Pitfalls (every one of these cost us time — do not rediscover them)

- **Dirs are K/e-sorted everywhere**: smallN seq_len_1 = 600×K0 first (single class → NaN d′);
  images_park seq_len_8 = e-sorted 900. Never trust LIMIT<full without checking class/prior mix;
  print the gold histogram at prep end (add it if absent).
- **dtype**: cast rotary cos/sin to hidden dtype (`pe = (cos.to(emb.dtype), ...)`) or queries go
  fp32; pass 4D masks as fp32 (universally accepted); cast lm_head input
  `.to(model.lm_head.weight.dtype)`. SDPA with fp32 mask silently upcasts the stream — harmless
  for math, fatal at fp16 lm_head.
- **Never cache tensors under `torch.inference_mode()` that later enter autograd** — use
  `torch.no_grad()` in prep paths.
- **The distill ckpt's head_w is UNTRAINED (zeros)** — never use it for zero-shot head tests;
  fit a logistic on the N=8 messages instead (`messages_best.npz`).
- **Masks**: fence whole BLOCK SPANS (vision_start/end markers included) — per-token-class
  fencing leaks cross-frame content through marker residuals (+1.7 d′ when sealed). Verify with
  the mask-debug line: all blocks must show IDENTICAL allowed-key counts.
- **Positions**: per-block reset is legal only under the full fence; carriers get sequential
  positions (stage-2) — keep the `[pos-debug]` check `blocks_identical=True`.
- **off−9 anchor is INVALID under interleaved templates** — compare against external anchors.
- `image_token_groups(ids, expected_num_frames=…, processor=…)` — keyword `processor` required.
- **sbatch**: `--export=ALL,VAR=a b` comma/space-splits — set env vars in the shell then
  `sbatch --export=ALL` (see runners/of_*.sbatch EXTRA_FLAGS pattern). `--wrap` rejected.
  a100-public = 40GB (fine for N≤64 fenced; N=128 → h200-shared + 24h_1g). Spread QOS
  (2h_2g ×3 jobs/2 GPUs, 24h_1g ×4, 4d_1g ×8). Check `sinfo` for idle GPUs before submitting.
- **Estimator scales**: held-out d′ deflates hard at small n (6.01@n1200 → 4.76@n300); compare
  pooled-to-pooled at matched n only. AUC estimator caps at 5.26 — quote measured accuracies in
  saturated regimes (E4 battery verdict).
- **gold > 9 must be skipped OR multi-digit decoding added** in LM-loss paths (digit CE assumes
  single token).
- Teacher anchor in the distill trainer must reproduce 9.24±0.33 — if not, the data path broke.

## Deliverables

1. `plans/carrier_stage2_DRAFT_RESULTS.md` — RESULTS-style entries per experiment, every number
   traceable to a run dir; update `outputs/ladder/INDEX.md`.
2. Final one-screen summary: E1-E5 headline table (full-prior N=8, length, mixture, rooms), the
   no-LoRA ablation row, what's blocked, suggested next.
3. Keep smokes in `outputs/_scratch/`; timestamped run dirs under
   `outputs/ladder/image_longN/<name>/`.

---

# PHASE 2 (2026-07-18 evening) — break the data-starvation ceiling

Status: E1 done (full-prior scaffold 0.998, frozen 0.219, probe d′ 13.54); E2/E4b/E5 first
rounds CONVERGED at 0.678 / 0.693 / 0.509 — all data-starved (see the E2/E4b/E5 entry in
carrier_stage2_DRAFT_RESULTS.md). Nothing is running. Continue from here:

**P1 — pooled-data stage-2 (THE decisive run).** Extend `carrier_layer_lora.py`'s loader to pool
MULTIPLE data roots with per-root task labels and variable N in one training set:
steps `data/mmred_images_park/seq_len_{2..8}/all_uniform` (variable N — check per-root dir counts
and gold histograms first; seq_len_8 has 900) + cooc `data/mmred_cooc_balanced/seq_len_8` (+
rooms root from E5 if format-verified). Target ≥3–5k train samples, eval held out per root.
Variable N also trains length-robustness directly (helps E3). One a100 job, 4d_1g, expect
several hours (prep RAM: cache ~12MB/sample at N=8, less below — cap RAM by streaming prep or
capping per-root cache if needed). Bands: GO ≥0.90 overall with per-task ≥0.85; 0.75–0.90 =
data curve still rising → double data again before touching architecture.
**P2 — cheap evals on EXISTING ckpts (run in parallel with P1, eval-only):**
(a) E3: full-prior steps ckpt (`carrier_layer/…123235 run/carrier_layer_best.pt`) at N=32
(a100) and N=128 (h200, 24h_1g) via the streaming eval-only mode; report digit-restricted AND
multi-digit-decode accuracy. (b) E4a: steps-only ckpt evaluated zero-shot on cooc.
**P3 — after P1 lands:** rerun E3/E4a with the pooled ckpt (the real length/task generality
numbers); if P1 GO, run the L_OPEN {12,22} one-change arms on the pooled recipe.
**P4 (stretch):** mixed-visual-diet distill (MMRED+natural) to close the domain-transfer gap
(carrier→natural currently 0.43 vs replica 0.92); InternVL probe port; HERBench null.

All Phase-1 pitfalls in this brief still apply verbatim. Log to the same draft + INDEX.
