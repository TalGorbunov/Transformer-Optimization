# Plan 2026-07-08 — Ladder transfer, N-scaling to 128, readout injection

> Charter for an unattended Claude session (run inside `tmux new -s thesis`).
> Ground rules (from CLAUDE.md — binding): read `RESULTS.md` first; show plan + partition/QOS + cost
> before ANY job submission and wait for OK **unless the step is marked [preapproved-smoke]** (≤2h_2g,
> n≤150, one job at a time); right-size QOS; check free GPUs across ALL partitions incl. `*-public`;
> outputs follow `outputs/<group>/<category>/<name>/<ts>/` + update `outputs/<group>/INDEX.md`;
> smokes → `outputs/_scratch/`; "log this" to RESULTS.md is an explicit user-approved step.
> Stop-and-ask list at the bottom. Registered predictions are written BEFORE runs — do not edit them.

---

## Workstream A — the aggregation ladder (theory transfer beyond MMRED)

Goal: place the d′ framework on a monotone extraction-difficulty axis; show the E4 adequacy battery
correctly sorts regimes. The peer critique to answer: "the Gaussian framing only works on commonly
structured frames."

Rungs (extraction difficulty ↑):
1. **A1 — text-MMRED** (existing text-frames pipeline; see `mamba_lm_text` runs + text-frames baseline
   evals). Extraction ≈ perfect by construction. Run the full instrument battery with frozen Qwen-7B
   (text): carrier locmap → message cache → `probe_dprime_parity.py` → native axis. n≈400 cache,
   n≈250 behavior.
2. **A2 — RULER-CWE/FWE-style word counting** (generate locally, tiny script; or Counting-Stars format).
   d′ ≈ ∞ rung; the pure readout/aggregation isolate. Behavior vs N ∈ {8,16,32,64}; law prediction from
   native-axis d′.
3. **A3 — MLVU Action Count transfer** (arXiv 2406.04264; 266 Qs, natural inserted Kinetics probes,
   exact insertion GT). Port exactly like the HERBench port (`experiments/herbench/` is the template).
   Natural + diverse + per-item GT = the direct answer to the peer critique.
4. **A4 — "mmred_natural" (our own NIAH-style dataset)**: k needle frames + distractor frames, all
   NATURAL images, per-frame GT by construction. Two controllable knobs:
   - needle diversity: identical copies → distinct instances of one concept (different dog photos);
   - distractor similarity: far (cars) → near (cats).
   This gives a *d′ dial on natural images* — the thesis artifact that generalizes MMRED.
   Source images: COCO or similar (ask before downloading datasets). Start N=8, counts 0–8, ~1500
   samples. Verify per-frame extraction ≥0.95 with the look-again judge (`lookagain_frames.py` pattern)
   BEFORE building the full set.

**Registered predictions (A):** text rungs pass E4 adequacy with d′ high → law predicts probe ceiling
near 1 at N=8 and √N decay beyond; frozen model still fails (readout wall reappears with extraction
removed entirely). mmred_natural: d′ decreases monotonically with needle diversity + distractor
similarity; adequacy passes while evidence stays binary-groundable, regardless of diversity.

**Deliverables (A):**
- **Fig A1 "the ladder"** — x = measured per-frame d′ (whitened, held-out), y = accuracy; three series
  per rung: model own-answer, law-predicted linear ceiling, decode-then-count; rungs: text-CWE,
  text-MMRED, MMRED, mmred_natural (×3 diversity levels), MLVU-AC, HERBench. Adequacy verdict (✓/✗)
  as marker shape.
- **Table A1** — rung × {modality, per-item d′, E4 verdict (skew/kurt/std-ratio), pred, measured probe,
  measured model, dtc}.
- **Fig A2** — mmred_natural d′ vs the two knobs (heatmap), showing the dial works.

---

## Workstream B — N-scaling to 128

Hard constraints found 2026-07-08: frames are 512×512 → **324 visual tokens/frame**; N=128 → ~41.5k
visual tokens > 32k default context. Must shrink frames for long N.

- **Prior art (do not redo):** two resolution sweeps exist. (1) 2026-04-14 `mmred_image_size_sweep`
  (66–504px, behavioral): resolution nearly irrelevant at seq8. (2) 2026-06-20 per-frame is-evidence
  AUC by frame size (crowded 5-char): 224px 0.873 → 336 0.945 → 448 0.971 → native-512 0.969.
  So the knee is ~448; 336 costs ~2.5pts AUC; 224 costs ~10pts. Token math: tokens/frame = (px/28)²
  → 448=256tok (N=128 → 32.8k, DOESN'T fit 32k), 392=196tok (25k ✓ tight), 336=144tok (18.4k ✓
  comfortable), 224=64tok (8.2k ✓).
- **B0a [first, cheap] resolution check at N=8**: what's missing from the prior sweeps is the
  **deployed carrier d′** (frames-first room-token messages) at reduced res. Measure carrier d′ at
  {512, 392, 336} on crowded 5-char, N=8. Decision rule: pick 392 if its d′ ≈ 512-baseline and memory
  allows the 25k context at N=128, else 336; if 336's d′ drop is large (>15%), stop-and-ask (fallback:
  fewer chars/frame at low res — 1-char extraction was perfect and robust). [preapproved-smoke]
- **B0b data gen**: `seq_len_{16,32,64,128}` steps_in_room at the chosen res, counts spanning 0–N
  (uniform-ish, plus a low-count band shared across N for calibration). CPU QOS `4h_0g`.
  Budget: token count = N × tok/frame + text; keep ≤ ~20k at N=128.
- **B1 d′ vs N** — cache carrier messages at N ∈ {8,16,32,64,128}, arms: **joint / fenced
  (isolation mask) / multipass**. (Fence = the known joint→multipass lever: 3.1→5.2 measured vs
  multipass 6.6; B1 measures how much of that survives at long N.)
- **B2 dilution + gate calibration** — the Σσ/tally gate under three inputs, trained at N=8, evaluated
  at all N: (a) raw message msg_f; (b) **mass-normalized** msg_f / Σ_{j∈f} A[c,j] (attention-weighted
  mean — magnitude N-invariant); (c) fenced per-frame reps. Report per-frame FN/FP vs N and tally bias.
- **B3 model behavior vs N** (n≈150/N, digit-argmax; counts >9 need multi-token answer handling —
  extend the answer reader first, small code task).

**Registered predictions (B):** raw-msg gate FN inflates with N (threshold drift from 1/N dilution);
mass-normalized and fenced gates stay ~flat. Joint d′ stays ≪ 6.3 (the 128-crush line: per-frame error
~8e-4 needed for ~90% exact at N=128); fenced/multipass approach it. Model emitted accuracy collapses
toward prior as `2Φ(d′_native/2√N)−1` dictates. Tally bias follows bias(g) ≈ N·FP − g·(FN+FP).

**Deliverables (B):**
- **Fig B1** — d′ vs N (log-x), lines joint/fenced/multipass, horizontal crush line at 6.3, secondary
  axis or panel: law-predicted exact-match at each (d′, N).
- **Fig B2** — gate FN and FP vs N for the three gate inputs (train-at-8 frozen thresholds); overlay
  predicted tally bias line.
- **Table B1** — N × {model acc, tally acc (per gate input), predicted ceiling, wall-clock/sample}.
- Runtime note per N in the table (N=128 forwards are slow; budget n before submitting — see QOS rules).

---

## Workstream C — readout injection (can a perfect external count be verbalized OOD?)

Established walls (do not re-run): single linear direction at answer site = non-causal; per-count
codebook = in-range only; number axis saturates ~5; token space is the only dictatable interface.

**Range-design constraint (important):** counts 0–8 are single tokens/digits, so per-digit and
per-count codebooks coincide — no extrapolation dissociation inside 0–8. Compositional-extrapolation
tests need counts ≥10 → use **text-MMRED with N up to ~40** (cheap, from A1) and/or B0b long visual
seqs. Multi-token answer reading must be handled (same code task as B3).

- **C1 token interface (baseline to beat)** — write the oracle tally into the prompt as digits
  ("Counted occurrences so far: 17."), zero training; then the honest version: tally-adapter output
  → text. Test train-range and unseen counts. Expect ≈1.0 everywhere (digits compose). [preapproved-smoke]
- **C2 digit-compositional injection** — learned soft-token codebook **per digit** injected at the
  answer region (sequence of digit-vectors for multi-digit counts), small trained readout head allowed
  (LoRA-rank-4 on late layers or trained unembedding — keep it minimal and reported). Train counts
  covering all digits in mixed positions (e.g., 0–9, 12, 25, 30); test held-out two-digit counts
  (11, 13–19, 21–24, 26–29...). Success = exact-match ≥0.8 on held-out counts.
- **C3 native-geometry injection** — fit Qwen's own number geometry (helix/Fourier: Kantamneni &
  Tegmark 2502.00873 method; Fourier features 2406.03445) on digit-token embeddings/activations;
  inject count as a point in that basis; readout with and without the small trained head; same
  train/test split as C2.
- **C-control** — per-count codebook (the known method) on the same split → should fail OOD; this is
  the contrast column.

**Registered predictions (C):** C1 ≈ 1.0 (trivial, but defines the bar and the interface cost);
C-control fails held-out counts (~0); C2 is the real bet — if digit-compositional injection
extrapolates, that is an unpublished positive result (nothing in the literature shows extrapolative
verbalization of injected activation-level quantities in a frozen LM); if it fails while C1 succeeds,
that is a clean negative: **the token interface is not just sufficient but necessary** — also a thesis
claim. C3 secondary.

**Deliverables (C):**
- **Fig C1** — accuracy vs gold count, one line per route (C1 / C2 / C3 / C-control), train counts
  shaded; the single money-plot for the readout chapter.
- **Table C1** — route × {params trained, in-range acc, held-out-count acc, unseen-N acc}.

---

## Ordering & budget

1. A1 text-MMRED battery (mostly CPU + 1–2 short GPU cache jobs) and B0a resolution smoke — first,
   cheap, unblock everything else.
2. B0b data gen (CPU) ‖ A2 text-CWE ‖ C1 smoke.
3. B1/B2 caches (few 12h_4g or 24h_1g jobs), C2 training (2h–12h), A3 MLVU port.
4. A4 mmred_natural build (ask before downloading source images), C3, B3.

## Reporting (per-experiment, amended ground rule — Tal pre-approved 2026-07-08)

1. **RESULTS.md logging is pre-approved for this plan**: after each experiment LANDS (run complete,
   numbers verified against the run dir), append an entry in RESULTS.md's format — date-tagged header,
   run paths + job IDs, headline numbers, registered-prediction verdict (✓/✗/partial), caveats. Every
   number must trace to a file in a run dir; never log from memory. One entry per experiment, appended
   at the end — no edits to existing entries.
2. **Living Artifact report**: maintain ONE artifact ("Aggregation ladder + N-scaling + readout report",
   keep the same file path across redeploys) with: the registered-prediction scorecard table at top
   (prediction / verdict / evidence link), then per-workstream sections embedding the pre-registered
   figures (A1/A2, B1/B2, C1) as base64 data-URI PNGs and the tables (A1, B1, C1) as HTML tables.
   Update it after each experiment lands. If artifact publishing is unavailable in the session,
   fall back to `outputs/ladder_report/report.md` + PNGs and say so.
3. Update the relevant `outputs/<group>/INDEX.md` whenever a run becomes canonical.

## Stop-and-ask (do not proceed without Tal)
- Any dataset download (MLVU videos, COCO) — size + disk location first.
- Any `pip install`.
- B0b generation parameters before generating 128-length data (disk cost).
- Before logging anything to RESULTS.md.
- If B0a shows 5-char extraction collapses at ≤336px (changes the whole B design → discuss 1-char or
  larger canvas option).
- If any C2/C3 result looks too good: check the readout head isn't memorizing count→token mappings
  (report head param count and an ablation without it).
