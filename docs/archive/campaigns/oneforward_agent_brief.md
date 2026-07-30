# Mission brief — the single-forward supply fixes (3 experiments: fence+replicas, composition, CoGNN-gate)

> **Agent instructions.** You are a long-running Claude Code session in tmux. Read `CLAUDE.md` first
> and obey it: never `pip install` into the shared `.venv` (nothing here needs installs), never run
> heavy compute on the login node, right-size QOS, check idle GPUs across ALL partitions
> (`a100-public` is usually the one; its GPUs are 40GB), smokes → `outputs/_scratch/`.
> **Do NOT edit `RESULTS.md` or anything in `docs/`.** Write draft entries instead (Deliverables).
> If blocked, write `plans/oneforward_BLOCKED.md` and move to the next experiment.

## Context (read once)

Per-frame evidence messages at the carrier (room token, off −9, L16) are linearly separable with
held-out d′ (shrinkage-LDA, `dprime_pair` in `experiments/glstm/dprime_vs_n.py`). The joint-pass
supply is d′ ≈ 2.0; isolated (multipass) ≈ 7.2; the 2×2 factorized the loss into a contaminated
carrier QUERY × damaged frame VALUES. Measured/priced cells (L16): joint×joint 2.09–2.33 ·
clean-q×joint-v 3.97–4.47 · clean-q×unmixed-v 6.17 · clean×clean 6.29–7.33. Exact counting at N
needs per-frame error p = 1−Φ(d′/2) ≲ 1/2N ⇒ d′ ≈ 6 for N=128. REPLICA CARRIERS (interleave a
copy of the question after every frame; read each copy's room token) already measured: unmasked
3.56, masked 2.52 (RESULTS [2026-07-14]; script `experiments/glstm/replica_carrier_probe.py`,
canonical runs under `outputs/ladder/image_longN/replica_carrier*/`). These three experiments try
to reach d′ ≈ 6 in ONE forward. All use steps task, `data/mmred_images_park/seq_len_8/all_uniform`,
392px, n=300, layers 14,16 — identical protocol to the replica runs so numbers are comparable.

**Known pitfalls (learned the hard way):** `image_token_groups(input_ids_1d, expected_num_frames=…,
processor=…)` — keyword `processor` required. The off−9 "in-run anchor" is INVALID under the
interleaved template (reads a wrong token; d′ ~0.2) — either locate the FINAL question's room token
by word-match (same `room_pos` helper, applied to `fin_span`) or compare only against the external
joint anchor 1.97 (B1 N=8). d′ needs both classes: never drop gold-0 samples. Smoke n=8 prints
nan d′ (too few for held-out folds) — check it runs end-to-end, not the numbers.

---

## Exp A — replicas + FULL frame fencing = multipass-in-one-forward (run FIRST: zero training, biggest prize)

**Idea.** The masked-replica arm cleaned queries but frames still attended each other (values stayed
joint — its per-copy ladder 3.70→2.0 proved it). Add full fencing: FRAME tokens attend only
{prefix, own frame}. Then every (frame + its replica) block is bit-for-bit an isolated forward —
multipass restructured as one sequence with block-diagonal attention.

**Implement:** add `--fence-frames` to `replica_carrier_probe.py`. In the 4D mask construction
(masked branch), additionally: for each frame i's visual-token rows, forbid keys in all OTHER
frames' visual tokens (replica spans are already hidden from them). Replica rows unchanged
({prefix, own frame, own span}). Leave the final-question rows as they are. Everything else
(capture, recompute, per-copy report) is already in the script.

**Run:** one a100 job, `2h_2g`, 64G, smoke n=8 (`outputs/_scratch/`) then full n=300
(`--output outputs/ladder/image_longN/replica_fence`). ~90 min total.

**Pre-registered bands:** replica-read d′ — GO ≥ 5.5 ("multipass in one forward" confirmed; expected
6.3–7.3 = the pad/mp anchor band; the shortfall vs true mp 7.2 is the long-context/position term).
3–5.5 = partial (position/RoPE effects eat some isolation). ≤3 = something's wrong with the mask —
verify with a debug print of one sample's allowed-key counts per row class before believing it.
Also report per-copy d′ (should now be FLAT — frame 0 is no longer special).

## Exp B — the composition: replica queries × UN-MIXED values (priced 6.17)

**Idea.** Keep frames jointly encoded (cross-frame context preserved) but repair the values with the
encoding un-mixer (a per-token MLP mapping joint k/v → mp k/v; recipe in
`experiments/glstm/encoding_unmixer.py`, 93% recovery measured at L16 with clean queries).

**Phase B1 (offline composition — the measurement that matters):**
1. Retrain g_k, g_v exactly per `encoding_unmixer.py` (MLP arm, L16; the weights were never saved) on
   the n=500 capture `outputs/ladder/image_longN/qkv_2x2/20260712_n500/` (same split seed 0,
   train 300). Save the nets this time. High-mem job (`--mem=160G`, a100, 2h_2g — the 120864
   footprint; the capture blob is 13.4GB).
2. Modify a copy of the replica probe (or add `--unmix-dir`) so the CPU message recompute applies
   g_k/g_v to the FRAME tokens' pre-rotary k/v before rotary+softmax — replica queries stay as
   captured from the UNMASKED replica forward (rerun the unmasked capture with the same n=300 or
   reuse saved `messages_cache.pt`? the cache stores only messages, not k/v — so RERUN the unmasked
   forward with an extra flag saving per-frame k/v at L16, or fold un-mixing into the same run).
   Simplest correct plan: one job = unmasked replica forward with g_k/g_v applied to frame-token
   k_proj/v_proj OUTPUTS at L16 via forward hooks DURING the pass (pre-rotary, matching the
   un-mixer's training space), then the normal message recompute. That is also automatically the
   deployed form (phase B2 for free). Note the recompute path must use the SAME modified k/v it
   reads from the qkv capture dict — hooks already capture post-modification if you modify inside
   the k_proj/v_proj hook by returning the edited tensor.
3. **Transfer risk (this IS the experiment):** the un-mixer was trained on standard-layout joint
   encodings; replica-prompt encodings differ (question-conditioning). Report d′ with and without
   the un-mixer on identical samples.

**Pre-registered bands:** GO ≥ 5.5 (composition confirmed; offline anchor 6.17); 4–5.5 = partial
transfer; ≤3.56 (no gain over replicas alone) = un-mixer does not transfer to this layout — report
as a real negative.

## Exp C — the CoGNN-style broadcast gate (the meeting's question, quantified)

**Idea.** Repair the routing from the CONTENT side: a per-token learned logit offset added to the
within-frame attention under the (contaminated) JOINT query. Evades the trained-query NO-GO
because the gate is computed from each token's own features (no frame/sample identity needed).

**Implement** `experiments/glstm/broadcast_gate_probe.py` on the n=500 capture (pattern:
`trained_query_ceiling.py` — reuse its differentiable message path, split seed 0, logistic proxy
objective, `dprime_pair` eval, anchor gate). Train b_j = MLP(features) → scalar logit offset for
token j, message = o_proj(Σ softmax(q_joint·k̃_j/√d + b_j)·ṽ_j). Two arms:
  (1) content-only: features = [k_j, v_j] (pre-rotary, 1024-dim in);
  (2) question-conditioned: features = [k_j, v_j, q_pad(frame)] (the pad arm's per-frame clean
      query from the same capture — deployable, since the question is always available).
Use ADDITIVE logit offsets (not [0,1] gates — suppression-only is provably weaker; note both in
the report if cheap). Anchors to reproduce in-run before trusting anything: joint-q 2.09 (eval),
mp-q 3.82 (eval). High-mem job like Exp B step 1 (can share it).

**Pre-registered bands:** floor 2.09 · ceiling 3.82 (eval scale; = clean-routing on joint values) ·
GO ≥ 3.0 (a deployable trained routing-repair module exists) · ≈2.1 = routing is NOT repairable
from content (completes the addressing story from the third direction — also a good result).
Trajectory diagnostic like the trained-q run (eval-d′ per checkpoint) to kill early-stopping doubt.

---

## Deliverables

1. report.txt + results.csv per run; every number traceable.
2. `plans/oneforward_DRAFT_RESULTS.md` — one RESULTS-style draft entry per experiment
   (runs/jobs, tables incl. per-copy rows, readings vs the pre-registered bands, caveats).
   Do NOT touch RESULTS.md.
3. Append rows to `outputs/ladder/INDEX.md`.
4. Final one-screen summary: the three d′ headline numbers vs bands, the single-forward supply
   ladder updated (joint 1.97 → replicas 3.56 → {A, B, C results}), what's blocked, suggested next.

## Ordering & budget

A first (cheapest, biggest prize, no training) → C's training job can queue in parallel on a second
QOS (`24h_1g`) → B after A lands (B reuses A's script fixes). Spread QOS; total target ≤ 8 GPU-hours.
If A hits ≥5.5, say so loudly in the summary — that's the headline.
