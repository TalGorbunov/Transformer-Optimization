# QGATE — STATE (append-only, newest last)

## [2026-09-15 ~05:40] Campaign start (Tal: "all done in the morning")

Brief written with fixed bands (H-QA1/QA2/QB1/QB2/QC2). Builder
`qgate_common.build_task_messages_layout` — replica path BYTE-IDENTICAL to the
sparse builder (CPU-verified for all 3 tasks), qlast/qfirst extended to
exists/majority, parse_layout verified for all 6 combos.
- **A1 P1b arm = 150577** (headscan --layout qlast-once, {count,majority} ×
  N{8,32,128}) — PD behind the 12h_4g 3-job cap; starts as the twins land.
- `capture_qb.py` WRITTEN (question-blind block states + text-only cacheable
  q_vecs + ALT-(C,R) relabelings for A2/B1) — submits on 150433's adapter landing.
- In-flight feeders: gated twins 150433/34 (long-N exams), gated-replica-5ep
  control 150572, G1b λ arms 150418/19 (both still bit=1.000 through ep4).

## [2026-09-16 ~01:20] Twins landed → QGATE critical path unblocked; N-free task in parallel

- Gated twins final (jobs 150433/34): qlast-gated 0.967/0.900/0.687/0.527 @8-64
  (mse +3.32@32, +13.76@64 — the prompt-N pull, gated form); qfirst-gated
  1.000/0.913/0.700/0.600 (mse +3.31/+13.81). N=128 cells skipped=150 in-job
  (known OOM-as-skip) → eval-only reruns 150717/18 queued on n318 (100/cell for
  the 2h wall — noted deviation).
- **A1 qlast-GATED headscan = 150719 RUNNING** (adapter from 150433's run dir,
  nfree, {count,majority} × N{8,32,128}); A1 P1b arm 150577 COMPLETED (verdict
  to be read at assembly). **B-capture = 150720 RUNNING** (150 dirs/root,
  ALT-(C,R) relabelings, L12+L20, text-only q_vecs).
- Tal's N-free layout task: 150580 qlast×nfree (2h16 in) + 150581 replica×nfree
  (2h) training; 150583 replica-N control RUNNING after the n317 stall kill.
- G1b λ arms at 6h05 (slower than estimate; two-forward cost) — still bit=1.000.

## [2026-09-16 ~09:00] A1 VERDICT: H-QA1 MISSED decisively — the "internal selector" is a VERDICT-READER; diagnostics launched

- A1 complete (150577 p1b arms + 150719 qlast-gated arms): under the qlast layout
  NO head separates evidence — best AUC 0.55–0.68 (vs 1.000 with replicas), in
  BOTH the replica-trained P1b and the qlast-trained gated adapter, every task,
  every N. Reading: L24h20 reads the REPLICA-BORNE VERDICT (computed in-block by
  the question copy), not frame content; qlast removes the verdict computation, so
  nothing exposes evidence in the answer-row QK geometry. The LORAMECH conditioning
  theorem at the gate level: selection requires conditioning somewhere.
  Consequences: B2 (internal-head threshold) is DEAD on qlast; the cacheable-
  encoding story rides entirely on B1 (external (question × block) scorer).
- B-capture 150720 was UNUSABLE (text-only qvec crash → 0 qvecs; masked di=0
  indexing bug). Both fixed (chat-templated qvec path; atomic per-dir commits) →
  **bcap2 = 150826 RUNNING**.
- **D1 (verdict-reader validation) = 150827 RUNNING:** replica-NEUTRAL layout
  (replica positions kept, question content replaced by a fixed filler) — if
  L24h20's AUC collapses to qlast levels, the verdict-reader mechanism is proven.
- D3 (are the facts in the blind encodings at all?) runs on bcap2's npz (CPU):
  symbolic-(C,R) probe on block states — decides extraction-vs-exposure, i.e.
  whether the gate ARCHITECTURALLY needs per-block conditioning or just a smarter
  external reader.

## [2026-09-16 ~11:30] A2/D3/B1 fits (bcap2, span-mean L20) — the extraction wall

- **A2 (H-QA2): PASSES as the negative control** — state-only = base rate exactly
  (0.802, AUC 0.519). Evidence is not a per-block property. ✓
- **D3: occupancy facts only WEAKLY linear in blind encodings** — full bilinear
  probe with a PERFECT symbolic (C,R) selector reaches AUC 0.747. Not gate-grade.
- **B1 (H-QB1): MISSED catastrophically as designed** — model-q_vec interaction
  = majority (AUC 0.52 heldout AND LOO). No model question-vector can beat the
  perfect symbolic selector that D3 already used → the bottleneck is the STATE
  side, not the question side.
- Synthesis (pending the pooling fair-shake): the frame facts ARE in the blocks
  (oracle-gated qlast reads 0.967 — full attention extracts them) but are NOT
  linearly exposed to any external reader. Extraction requires the model's own
  conditioned compute — which is what per-block replicas provide. QGATE's
  cacheable-blind-encoding premise fails at the EXTRACTION level, not selection.
- Fair-shake in flight: **bcap3 = 150848** (3-pooling sweep: mean/last/max ×
  L12/L20 — span-mean may wash out token structure); refit on landing.
  **D1b = 150841** (neutral-replica verdict-reader test; v1 died on a tokenizer
  boundary merge — ".You" — neutral filler now ends in "?", parse CPU-verified).
- D1 v1 instrument note: silent-continue on parse-fail hid the zero-sample run —
  headscan now a candidate for a loud skip counter (not patched mid-campaign).

## [2026-09-16 ~13:00] OPTION 1 GO (Tal) — the generic-replica (cacheable summary slot) stack

Design: blocks = [frame + FIXED generic filler "What is shown in this frame?"]
(cacheable — identical for every question); gate = external scorer over
(question, slot state); decode = gated fenced read. Pipeline launched:
- **E1 capture 150851 (running):** neutral-replica slot states (+mean/max chans)
  + ALT-(C,R) labels → the slot-decodability probe (Option 1's viability test).
- **O1 trainer = 150856 RUNNING** (`train_neutral.py` — thin launcher rebinding
  the sparse trainer's builders to layout=replica-neutral; sparse originals
  untouched; rebind CPU-verified): 5-ep anchor recipe, oracle gate, layout-family
  exclusions + exams N8-64 in-job. → `outputs/qgate/o1_train/`
- On E1 landing: slot-gate fit (symbolic (C,R) parse + model-qvec variants) →
  slot_gate.npz; then the deployed ladder (pass 1 neutral encode + slot gate,
  pass 2 gated decode with the O1 adapter). D1b + bcap3 verdicts fold in.

## [2026-09-16 ~14:10] D1b: VERDICT-READER CONFIRMED CAUSALLY · bcap3: extraction wall is pooling-independent

- **D1b (150841):** neutral fillers in the replica positions collapse P1b's
  selector to 0.62–0.78 AUC (vs 1.000 with question-copies; L24h20 off the board).
  The head reads question-conditioned VERDICTS — proven by intervention. P1b does
  NOT internally match tail-questions to generic slot summaries.
- **bcap3 refits (mean/last/max × L20):** D3 = 0.70–0.75 AUC and B1 = chance on
  EVERY pooling channel. The fair-shake is closed: raw question-blind block states
  do not linearly expose occupancy, full stop.
- All roads now lead to E1 (150851, running): does the GENERIC-FILLER SLOT expose
  occupancy where raw states don't? ≥~0.99 → Option 1 lives (cacheable slot gate);
  ~0.75 → Option 1 dead at the linear level → Option 2 (micro cross-attn reader)
  or Option 3 (replica replay over cached KV, the proven fallback).
