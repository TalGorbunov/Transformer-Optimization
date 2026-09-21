# HERBench retrieve-opt — STATE

Handoff log. Newest phase last. See CAMPAIGN_BRIEF.md for the plan/gates.

---

## Phase 0 — inventory & feasibility  — **DONE (analysis)**, extraction PENDING gate

Run: `scripts/herbench_retrieve_opt/phase0_inventory.py`
Report: `outputs/herbench_retrieve_opt/phase0/inventory.json`

### Confirmed route to ±δ neighbor frames ✅ (the pivotal Phase-0 question)
- Source videos ARE on disk: `/scratch/tmp/28010/claude-28010/-home-tal-gorbunov-projects-Transformer-Optimization/28233274-9c57-46bb-b3a9-c56f0731ca5a/scratchpad/herbench_videos/HD_EPIC/Pxx/<video_id>.mp4`
- 28 mp4s, **22 GB total**, all 28 reachable. Native **1408×1408 @ 30 fps** (so BOTH
  448 and 672 arms are downscales — clean resolution axis, no upsampling).
- Extraction tool: **PyAV** — not in shared `.venv`, but present at `~/.local/pyav-py39`
  (`PYTHONPATH=~/.local/pyav-py39`, av 12.3.0 verified opening + seeking a video). No pip
  needed. No HF-dataset reload needed: our `meta.json` already carries
  `occurrence_timestamps` (seconds), `pair`, `video_id`, `true_count`.
- ⚠️ RISK: videos live on volatile `/scratch/tmp`. Extraction writes JPEG frames to a
  stable `data/` dir, so once extracted the campaign is safe even if scratch is purged.
  → extract promptly.
- No STOP-AND-ASK trigger fired: 0 GB to fetch (on disk), no credentials, disk est ≪ 60 GB.

### Data inventory (armA_evidence_only, 144 questions)
- 144 questions, 28 videos, **820 occurrences** total, 129 multi-occurrence questions.
- true_count histogram: mostly 1–8 (n=15,27,13,15,16,15,7,10 for count 1..8), long tail to 25.

### Verb distribution (n *questions* per verb) — HEAVILY concentrated
| verb | n Q | | verb | n Q |
|---|---|---|---|---|
| pick | 45 | | pour | 5 |
| open | 27 | | put | 4 |
| close | 19 | | transfer | 4 |
- Only **pick / open / close** clear n≥20 questions. 35 verbs total, 32 are near-singletons.
- open+close are container-transition verbs → the "transition-verb subset" the brief flags
  as a usable PARTIAL-scope result. At the *unit* (occurrence) level these three have
  hundreds of units → per-verb d′ (n≥20 units) is easily met for them; the long tail will
  be reported pooled as "other".

### Intra-pair inter-occurrence gaps (constraint on δ) — 676 gaps
- min 0.07 s, p05 1.76 s, p10 4.09 s, p25 13.9 s, **median 42.2 s**, p75 118 s.
- Safety = fraction of occurrences whose nearest SAME-PAIR neighbor is within 2δ (i.e. a
  ±δ clip window would touch it):

| δ (s) | unsafe occ | frac |
|---|---|---|
| 0.5 | 26 / 805 | 3.2% |
| 1.0 | 72 / 805 | 8.9% |
| 2.0 | 113 / 805 | 14.0% |

- **Chosen SAFE δ grid: {0.5, 1.0} primary; 2.0 optional stretch.** Per-δ, DROP positives
  whose nearest same-pair neighbor ≤ 2δ so each positive clip is a clean single occurrence.

### Disk estimate (frames to extract, 5 frames/clip, ~1640 clips/arm)
- 448: 304 MB per δ; 672: 641 MB per δ. Full grid (both res, δ={0.5,1,2}) ≈ **2.8 GB**. ≪ 60 GB cap.

### Phase-1 feasibility note (probe machinery)
- Existing `scripts/probe_supply.py` = one replica PER FRAME. Clip arms need one replica
  PER CLIP-UNIT (a unit = 3–5 frames). Clean solution: a NEW standalone probe in
  `scripts/herbench_retrieve_opt/` that reuses gnnformer primitives
  (`build_replica_probe_mask`, `recompute_messages`, `dprime_pair`, `reset_positions`) with
  UNIT-level spans (vis-per-unit = union of the unit's frame image tokens; one block+replica
  per unit). No gnnformer-core edit. A0 (single-frame units) reduces to the existing probe →
  this is what makes the d′≈1 anchor reproducible.

### Decisions (Tal, 2026-08-02)
- Sequencing: **anchor-first** — Step 1 = A0 anchor + B-δ0.5@448, gate on A0, then full sweep.
- δ grid: **{0.5, 1.0, 2.0}**.
- Extraction route: **on the login node** (nice'd PyAV) → frames to shared `data/`. Forced by
  `/scratch` being login-node-local (`rl_athena--login-scratch`, 99% full) + volatile; `/home`
  is shared NFS (compute nodes read the extracted frames). No 22 GB stage.

---

## Phase 1 — probe sweep — **BUILD DONE; Step-1 extraction RUNNING; GPU pending**

### Scripts (all under `scripts/herbench_retrieve_opt/`, reuse gnnformer as a lib)
- `extract_clips.py` — PyAV ±δ clip extractor (cluster-seek decoder). Per (δ,res) arm:
  positives (per occurrence, drop if same-pair neighbour ≤2δ or edge), matched negatives
  (½ random-away ≥5 s, ½ hard = just-outside-window, validated to contain no occurrence).
  Smoke: hard-neg windows 0/28 contain an occurrence ✓; balanced pos/neg ✓.
- `probe_clips.py` — clip-UNIT supply probe. Per-question Q-first + block-fence + posreset over
  that question's units (one replica per UNIT; big Qs chunked ≤`--max-units`), recompute per-unit
  message at read layer(s), carrier-offset sweep (report best). Metrics: whitened LDA d′
  (QUESTION-grouped splits) + AUROC + logistic gate; pooled + per-verb (n≥20). A0 via
  `--input-mode armB` (armB frames = single-frame units). CPU-validated: both parse; grouped-d′
  unit test separable→4.2 / shuffled→0.36 ✓.
- `slurm/probe_clips.sbatch` — 2h_2g wrapper; STEP=1 (A0+B0.5@448), STEP=2 (full 672/δ sweep).
  DRY_RUN ✓.
- **CPU `tests/` suite: ALL PASS** after adding the scripts (fencing parity 8/8, +carrier/data/
  scratchpad/mmred_hf) — gnnformer core untouched.

### Anchor probe reference (docs/archive/RESULTS_pre_fencing.md 2026-07-07d/e)
- Legacy `probe_frame_to_carrier_message.py --task herbench_ac` on armB@448, per-frame message
  into question-token carrier, layers 12–20: carrier AUROC≈0.80, whitened d′≈0.98–1.10.
- **Prior null to remember:** resolution **448→896 was already a NULL** (carrier d′ flat) → the
  novel axis here is TEMPORAL ±δ (single-frame-only before). 672 arms measured for completeness.
- Fence gain was NULL on real video → the fenced-replica layout should reproduce the joint d′≈1.

### RUNNING
- Step-1 extraction `d0.5_r448` (login node, PID bg): log
  `outputs/herbench_retrieve_opt/phase0/extract_d0.5_r448.log`. ~6/28 videos at last check.

### Extraction DONE
- `d0.5_r448`: **1584 units (792 pos / 792 neg)**, balanced; per-verb pos pick230/open173/close130
  (≥20 ✓); drops pos_edge2 pos_unsafe_gap26; 350 MB; 0 videos missing.

### Probe pipeline — VALIDATED on GPU (smoke 127942, 6 Q/arm)
- Bug found+fixed: `find_question_spans` returned None (Q-first prefix vs post-image replicas
  have different BPE leading contexts → no single needle matches all copies). Replaced with
  **structural span location from vision tokens** (vstart/vend + q0 length from first inter-unit
  gap); CPU-validated single+multi-frame (replicas decode, carriers in-bounds, blocks contiguous).
- Smoke d′ (tiny n, noisy) rises with layer as expected:
  A0 armB L16=1.58 L18=1.61 L20=1.79 (gate .78–.85); B_d0.5 L18=1.90.
  → pipeline healthy. NOTE A0 sits ABOVE the 0.98–1.10 band — likely small-n LDA bias (19 pos)
    + best-of-4-offset selection; full run (~160 pos) will regress. Interpret gate loosely
    (order-of-magnitude sane, layer-monotone, not broken/leaking) rather than exact-band.
- ⚠️ **SPEED**: smoke took 27 min for 12 Q → full STEP-1 (278 Q) won't fit 2h QOS. Optimizing:
  (a) batched all carrier-offsets into ONE recompute per layer (done), (b) diagnosing forward
  vs recompute (job 127956), (c) will chunk by FRAME budget to bound seq for high-count Qs.

### Speed fix
- Root cause of 27-min smokes: `load_units_*` opened ALL clip JPEGs upfront (ignored --limit).
  Fixed → lazy path-based opening. Forwards are ~0.6 s; load_runtime ~60 s. Full STEP-1 = 62 min.

### ✅ STEP-1 RESULTS (job 127959, a100, whitened LDA d′, best carrier-offset, question-grouped)

**A0 GATE — PASSES.** armB single-frame @448, n=2144 (629 pos):

| layer | L12 | L14 | L16 | L18 | L20 |
|---|---|---|---|---|---|
| A0 pooled d′ | 0.60 | 0.96 | **1.05** | **1.08** | 1.05 |

→ squarely in archived band 0.98–1.10. Pipeline reproduces the anchor. (Smoke's 1.5–1.8 was
small-n LDA bias, as predicted.)

**B-δ0.5@448** (±0.5 s clips, n=1584, 792 pos) vs A0 — pooled d′ ~FLAT, but a per-verb split:

| verb (L18) | A0 d′ | B-δ0.5 d′ | Δ |
|---|---|---|---|
| open  | 1.18 | **1.67** | +0.49 |
| close | 1.33 | **1.52** | +0.19 |
| pick  | 0.96 | 0.96 | 0.00 |
| pooled| 1.08 | 1.06 | ~0 |

→ **Signal:** ±0.5 s temporal context lifts the TRANSITION verbs (open/close) but NOT the
manipulation verb (pick, which dominates n and flattens the pooled number). Consistent with the
hypothesis. **No arm clears the d′≥2.5 bar yet** (best per-verb open=1.67). The question for
Step 2: does larger δ (1, 2 s) push open/close over the bar? Runs: `probe/A0_r448/20260802_134415`,
`probe/B_d0.5_r448/20260802_141853`.

### STEP-2 δ-SWEEP (per-verb d′ @ L16 — first pass; see caveat) — jobs 127968/127969
Extractions all done (δ{0.5,1,2}×res{448,672}, balanced, 2.7 GB; armB_r672 134/134).
⚠️ CAVEAT: Step-2 jobs hit the **`sbatch --export` comma-split bug** — `LAYERS=16,18,20` in the
--export list truncated to `LAYERS=16`, so this pass is **L16-only** (MAXUNITS survived, no OOM).
Re-running all 3 groups with LAYERS via ENV (jobs 127976 B448 / 127977 C672 / 127978 A1) for the
full 16,18,20 peak. Numbers below are L16 (valid, just not peak-layer).

**per-verb d′ @ L16 (well-powered verbs; * = clears 2.5 bar):**

| verb (n_pos) | δ0/A0 | δ0.5·448 | δ1·448 | δ2·448 | δ0.5·672 | δ1·672 | δ2·672 |
|---|---|---|---|---|---|---|---|
| open (~165) | 1.40 | 1.50 | 1.94 | **2.10** | 1.45 | 1.89 | 1.98 |
| close (~125)| 1.30 | 1.35 | 1.62 | 1.65 | 1.47 | **2.15** | 1.58 |
| pick (~210) | 0.95 | 1.05 | 1.23 | 0.99 | 1.30 | 1.24 | 1.25 |
| put (~25,noisy)| — | 1.48 | 1.63 | 2.23 | 1.28 | 3.82* | 2.88* |
| POOLED | 1.05 | 0.92 | 1.08 | 1.13 | 0.95 | 1.17 | 1.20 |

**Reading (L16):**
- **Temporal context IS the lever, and it is transition-verb-specific.** open rises monotonically
  1.40→1.94→2.10 with δ@448; close peaks 2.15 @δ1·672. pick (the manipulation verb, largest n)
  stays ~1.0–1.3 at every δ/res → never lifts, and it drags the pooled number to ~1.1.
- **Resolution 448→672 ≈ null** (open δ1: 1.94 vs 1.89), consistent with the prior 448→896 null.
  The 672 gains that appear (close δ1, put) are within noise / small-n.
- **Nothing well-powered clearly clears d′≥2.5** at L16: open tops at 2.10, close at 2.15. put
  crosses (3.82/2.88) but n≈25 → noisy. Peak-layer (L18/20) re-run pending to confirm.
- Likely verdict: **PARTIAL-leaning-NULL** — temporal context gives transition verbs a real,
  sizable lift (d′ ~1.3→~2.1) but falls just short of the exact-match-enabling bar; the honest
  null largely stands, now with "perception levers tried: δ≤2 s (temporal helps open/close but
  <2.5), res≤672 (null)". Confirm with peak-layer numbers, then figures.

---

## Phase 2 — VERDICT (peak-over-layer L16/18/20; jobs 127976/7/8 full sweep)

Figures: `phase2/fig1_dprime_vs_delta.png`, `phase2/fig2_per_verb_best_arm.png`;
table: `phase2/verdict.md`. Peak-over-layer whitened d′, question-grouped, bar d′≥2.5.

**Per-verb peak d′ (well-powered n_pos≥40; put small-n in parens):**

| arm | open | close | pick | put |
|---|---|---|---|---|
| A0 δ0·448 | 1.40 | 1.33 | 0.96 | — |
| B δ0.5·448 | 1.67 | 1.52 | 1.05 | 1.85 (25) |
| B δ1·448 | 2.18 | 1.95 | 1.23 | 4.33* (27) |
| B δ2·448 | 2.18 | 1.91 | 1.33 | 2.56* (24) |
| A1 δ0·672 | 1.47 | 1.50 | 0.98 | — |
| C δ0.5·672 | 1.49 | 1.48 | 1.30 | 2.74* (25) |
| C δ1·672 | 2.19 | 2.15 | 1.69 | 5.22* (25) |
| C δ2·672 | **2.36** | 1.90 | 1.35 | 3.54* (22) |

pooled peak d′ stays 1.08→1.43 (dragged by pick, the largest class).

### VERDICT: **NULL CONFIRMED (well-powered) — with a strong PARTIAL signal at the margin.**
1. **Temporal context ±δ is the real lever, transition-verb-specific.** open rises 1.40→~2.2–2.4
   and close 1.33→~2.15 with δ (both saturate by δ≈1 s); the manipulation verb **pick never lifts**
   (≤1.7) and, as the biggest class, holds the pooled d′ at ~1.1–1.4.
2. **No well-powered verb CLEARS the d′≥2.5 bar.** Best is open=2.36 (±2 s @672) — striking distance,
   but under. So exact-match after p^N compounding stays out of reach → the aggregation method's
   honest null on HERBench **stands**.
3. **`put` clears everywhere with temporal context (2.56–5.22) but n_pos≈22–27 (~4 questions)** →
   suggestive, not conclusive; flag as small-n.
4. **Resolution 448→672 is a SMALL positive at peak layer (~+0.1–0.4), mostly IN COMBINATION with
   temporal** (best cell = C = both axes). This REFINES the earlier "resolution null": pure-res
   A0→A1 is ~flat (1.40→1.47 open), but res compounds with δ. Not the dominant axis; δ is.
5. Honest-null chapter gains: *"perception levers tried — temporal δ≤2 s lifts transition verbs
   (open/close) from d′~1.3 to ~2.2–2.4 (approaching, not clearing, 2.5); resolution ≤672 adds a
   modest further ~0.1–0.4 only in combination; the manipulation verb pick never responds."*
   Transition-verb share of the AC split ≈ open(27)+close(19) = 46/144 ≈ **32% of questions**.

---

## Extension (Tal, 2026-08-03) — PLAIN model yes/no accuracy sweep (eval-only)
Tal asked for the frozen model's OWN per-unit yes/no accuracy (not the probe) vs δ and
resolution, and to push δ higher (3,4,5). Decisions: **staged** (existing δ{0,0.5,1,2}×
{448,672} first, then decide on δ→5 / res→896); **fixed 5 frames/clip**.
- New: `scripts/herbench_retrieve_opt/eval_plain_acc.py` (look-again yes/no logit margin →
  AUROC + acc@0 + bal_acc + yes_rate, per verb, balanced→chance .50), `slurm/eval_plain_acc.sbatch`
  (GROUP=448|672|hi|res). Validated (metrics sep→auroc .997; yes/no ids distinct; DRY_RUN).
- RUNNING: 128249 (448 group), 128250 (672 group). Reuses on-disk clips (no new extraction).
- Then decide: extract δ{3,4,5}×{448,672} (GROUP=hi) and/or res sweep {336,448,672,896}@δ1
  (GROUP=res) based on whether plain-acc rises with δ or plateaus like the probe.

### PLAIN-ACC RESULTS (jobs 128249 @448, 128261 @672) — model's OWN yes/no AUROC (chance .50)
Fig `plain_acc/fig_plain_auroc_vs_delta.png`. (Model is NO-biased, yes_rate .06–.45 → use AUROC.)

| verb | δ0·448 | δ.5 | δ1 | δ2 | δ0·672 | δ.5 | δ1 | δ2 |
|---|---|---|---|---|---|---|---|---|
| open | .916 | .900 | **.938** | .937 | .928 | .902 | **.938** | .932 |
| close| .844 | .858 | **.915** | .886 | .841 | .888 | **.920** | .873 |
| pick | .827 | .826 | .837 | .852 | .844 | .833 | **.867** | .861 |
| pooled| .835 | .823 | .849 | .857 | .848 | .831 | .854 | .856 |

**Findings (decisive for the extension):**
1. **δ plateaus/peaks by δ=1 in BOTH readouts** (probe d′ AND plain AUROC); close peaks δ1 then
   declines at δ2. → **higher δ (3,4,5) is NOT worth running** — the plateau is confirmed twice.
2. **Resolution 448→672 ≈ flat** for the model's own judgment (open identical; pick/close +.02–.03).
   → res→896 unpromising; low value.
3. **KEY reconciliation:** the model's OWN isolated per-clip detection is GOOD — open AUROC ~.94,
   close ~.92 at δ1 (⇒ d′≈2.1–2.2), well ABOVE the fenced-supply probe (d′≈1) AND far above the
   joint-counting EM (4.9%). So **per-unit PERCEPTION is not the wall; AGGREGATION is** — the model
   can tell whether an occurrence is in an isolated clip, but cannot detect-and-tally jointly across
   many frames in one forward. This REINFORCES the thesis (aggregation/over-squashing bottleneck)
   and reconciles the "perception ceiling": the ceiling is on fenced single-carrier *supply*, not on
   the full model's per-unit perception.

### RECOMMENDATION
Stop the δ/resolution perception sweep — two independent readouts agree it plateaus by δ≈1 and
resolution is ~null; the levers can't clear the bar because the true bottleneck is aggregation, not
per-unit perception. Natural next step (if any) is the AGGREGATION side: **Phase 3 gate→tally** at
the best perception config (C·δ1, or just isolated per-clip gate → count) to quantify the tally gap
directly. Awaiting Tal's call.

### Phase 3 GATE→TALLY (offline from saved probe messages, CPU) — `scripts/.../gate_tally.py`
Logistic gate on per-unit carrier messages → sum per question → count-EM. Group(Q)-split, 5 seeds.
Construction caveat: pos:neg ≈1:1 (N≈2·count) → OPTIMISTIC vs sparse armB (K/16); law_EM_N16 is the
sparse reference. Frozen native HERBench EM = 0.049.

| arm | feat | d′/unit | count-EM | MAE | bias | law_EM(N16) |
|---|---|---|---|---|---|---|
| C·δ1·672 | L20 | 1.82 | 0.277 | 1.24 | +.07 | 0.180 |
| B·δ1·448 | L20 | 1.58 | 0.281 | 1.30 | +.33 | 0.157 |
| C·δ2·672 | L20 | 1.78 | **0.316** | 1.17 | +.08 | 0.176 |
per-verb (C·δ2·672): open .374 close .396 pick .335.

**EM collapses with N (C·δ1·672, the "high-N" answer):**
count 1–2 → .373 · 3–4 → .275 · 5–8 → .206 · **9+ → .174**.

**VERDICT (answers "is .95 enough / would learned carrier+GIN succeed"):**
- Carrier gate→tally gives a REAL **~4–6× uplift** (0.049 → ~0.28 dense / ~0.18 sparse-law) — the
  method DOES help on real video. But it **tops out ~0.2–0.3 and decays monotonically with N**
  (→.17 at count≥9), never approaching "solved."
- This is per-unit-perception-limited: pooled gate d′≈1.8 (open≈2.2), and the √N exact-match law
  caps it. **GIN ≠ better** — counting is a sum; GIN's expressiveness is irrelevant, sum is optimal.
- Learned carriers/LoRA could push the fenced supply d′ from ~1 toward the model's own per-clip
  ceiling (~2.2 open), but CANNOT exceed the frozen backbone's perception (AUROC≈.94) → still
  short of the d′≥2.5–3 (AUROC≈.955–.98) needed for high-N EM.
- **HERBench is perception-co-limited → a PARTIAL venue, not a success demo.** The method's clean
  success case remains MMRED (per-unit d′≈6.34, law EM .57@N16; measured gate→tally .96@N8).
Best config **C δ1·672** (open 2.19 / close 2.15 / pick 1.69 / put 5.22) or **C δ2·672** (open 2.36).
Proposal to present: (a) L2 gate→tally on mixed evidence+filler clip sequences at that config
(does the ~2.2 transition-verb d′ + a learned gate yield any count EM?); (b) costed in-domain
carrier-distillation plan (NOT launched). Awaiting decision.
