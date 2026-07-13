# QUESTIONS for Tal — 2026-07-08 session (batched at stop-and-ask gates)

> ## ✅ ANSWERS — Tal approved 2026-07-09 (via Claude session; blanket go for autonomous run)
> - **Q1 APPROVED** as proposed (long-N gen, render 512 downscale-at-load, ~1.6 GB, `data/mmred_longN_park/`).
> - **Q2 APPROVED** (MLVU-AC subset: AC-referenced videos only, ≤25 GB transient to login scratch,
>   keep only frames in `data/mlvu_ac/`).
> - **Q3 APPROVED — take BOTH** (COCO val2017 + Oxford-IIIT Pets, <2 GB total).
> - **Q4 APPROVED** incl. the proposed QOS split (2×24h_1g for N=64/128; 2h_2g/12h_4g rest).
> - **Q5 APPROVED** incl. design amendments (a) fact-slot injection site and (b) adapter renders facts.
> - **Additional items approved 2026-07-09** (from the block-read/theory sessions):
>   A1-fu1 text multipass cache (definitive write-cap test) · A1-fu2 easy-text minimal pair
>   ("Frame i: Michael@Park, Sara@Kitchen." — binding in surface form; expect d′ → CWE-like if the
>   binding account is right) · A1-fu3 cooc block-read on the existing n=1500 6-offset cache (CPU,
>   `block_read_completeness.py`) · C1b fact-phrasing robustness grid (paraphrases, position,
>   digits-vs-words, distractor fact, source-attributed; n≈120) · misc: frozen-model evidence-only
>   behavioral number at N=8 (cheap, closes a flagged gap).
> - **Optional stretch (approved if primary work lands): VNBench counting rung** (VideoNIAH,
>   arXiv 2406.09367) — synthetic-needle NIAH with exact GT; ≤20 GB transient, frames-only durable.
> - Standing rails unchanged: no pip installs, never delete outputs, right-size QOS, RESULTS.md
>   logging per the plan's Reporting section.

> Format: question → my recommendation → what's blocked on it. I keep working on non-blocked
> items; answer inline / in chat and I'll pick up from here.
>
> **Status when you read this** (details: RESULTS [2026-07-08]–[2026-07-08d] + `outputs/ladder_report/report.md`):
> A1 ✅ (text wall replicates 0.196; carrier d′ 1.8 ≈ image — legibility ≠ d′; fence NULL on text;
> law closes on native axis 0.163 vs 0.165) · A2 ✅ (CWE = high-d′ rung, model 0.79→0.47 with N,
> escapes the wall via literal-match addressing; E4 rejects law in a 3rd way) · B0a ✅ (**392px
> decided**, −12.5% d′) · C1 ✅ (**token interface 0.95–0.99 incl. OOD multi-digit — but ONLY as
> a semantic fact statement; "The correct answer is k." = 0.00!**) · In flight: text d′-vs-N
> (N=16/24/40) + C-range N=40 fact test. Blocked on Q1–Q5 below.

## Q1 — B0b long-N data generation parameters (gate: disk cost + params)

**Proposal.** Use `datasets/mmred/generate_mmred_balanced.py --task steps_in_room` (already
verifies gold == K by assertion) with a small `--counts` override I'd add (choose count values
explicitly instead of 0..N):

- seq_len ∈ {16, 32, 64, 128}, crowded 5-char, all 6 park rooms (deployed regime).
- counts per N: shared low band {0..8} ∪ spread {12,16,24,32,48,64,96,128}∩[0,N] → 13–17 values/N.
- per-count 30 → ~450–510 samples per N (enough for n=400 caches + n=150 behavior with reuse).
- **Render at native 512px** and downscale at load time via the probe's new `--resize` flag
  (added today) — keeps one dataset serving any resolution B0a picks. Token budget at load-res
  392px: 128×196+text ≈ 25.4k (fits 32k, tight); at 336px: 128×144 ≈ 18.6k (comfortable).
- Disk: ~13 KB/frame × Σ N×samples ≈ **~1.6 GB** total under `data/mmred_longN_park/`
  (single frame copy, no legacy duplicates). Generation on CPU QOS `4h_0g` (~1–2 h).

**OK to generate?** (yes/no/param changes)

## Q2 — A3 MLVU Action-Count download (gate: any dataset download)

- Need: MLVU "Action Count" subset only (266 Qs). Videos on HF (`MLVU/MVLU`); full benchmark is
  ~78 GB but AC-task videos are a subset — I'd fetch only AC-referenced files via HTTP-range /
  per-file download like the HERBench port, to **login scratch** (not repo), then extract frames
  to `data/mlvu_ac/` (durable artifact, est. a few hundred MB at 448px).
- Est. video download: ~10–20 GB transient scratch (AC videos only; exact number known after
  reading the AC json).
**OK to fetch AC-subset videos to scratch + keep only frames?**

## Q3 — A4 mmred_natural source images (gate: dataset download)

- Need natural images with per-frame GT by construction: k needle frames + distractors,
  needle-diversity and distractor-similarity dials.
- **Recommendation: COCO val2017 (~1 GB, 5k images, has per-image category annotations —
  dogs/cats/cars etc. for the similarity dial), to `data/coco_val2017/`.** Alternative:
  Oxford-IIIT Pets (~800 MB, fine-grained cat/dog breeds — better for the near-distractor dial)
  — could take both (<2 GB total).
**Which source(s) may I download?**

## Q4 — B1/B2 long-N cache jobs (approval: >2h GPU jobs)

Blocked on B0b data (Q1) + B0a decision (landing today). Planned once unblocked:
- **B1**: carrier message caches at N ∈ {16,32,64,128} × arms {joint, fenced, multipass-style}
  — joint+fenced are one probe run each (fence flag), multipass = isolated per-frame forwards
  (existing pattern). Est: N≤32 runs ≤2h each (2h_2g); N=64/128 runs need `12h_4g` or `24h_1g`
  (single GPU, est 3–8 h each: 128-frame forwards at 392px ≈ 25k tokens).
- **B2**: CPU on the B1 caches (gate calibration, three inputs) — no approval needed.
- **B3**: behavior vs N, n≈150/N, generation-based reader (multi-digit) — N=128 run est 2–4 h.
**OK to submit B1/B3 when data lands? Preferred QOS split?** (my plan: 2× `24h_1g` for N=64/128,
`2h_2g`/`12h_4g` for the rest, spread to respect the 3-per-QOS cap)

## Q5 — C2/C3 training runs (approval: training jobs, 2h–12h)

C2 = per-digit soft-token codebook injected at the answer region + minimal trained readout
(LoRA-r4 late layers OR trained unembedding — I'll report param counts and run the no-head
ablation per the plan's "too-good" gate). C3 = helix/Fourier native-geometry injection, same
splits. Train counts {0–9,12,25,30}, test held-out two-digit. Est 2–6 h each on one GPU.
**OK to run C2 (and C3 after) once C1 lands?**

*Update after C1 landed (RESULTS [2026-07-08b]):* the token bar is 0.95–0.99 incl. OOD, but ONLY
under semantic-fact phrasing ("Note: C spent exactly k steps in the R."); answer-directives score
0.00. Two design consequences I propose for C2: (a) the injection site comparison should include
injecting the digit soft-tokens at the FACT-slot position (replacing the fact sentence's count
token) — not just the answer region; (b) the honest tally-adapter→text route must render facts,
not hints. Confirm you're happy with (a)+(b) in the C2 design.

## Q6 — text-CWE / text-MMRED N>8 extensions (cheap, flagging for transparency)

A2 (text-CWE) runs N ∈ {8,16,32,64} entirely with synthetic text prompts — no data gen, no
download, ~1h single GPU total; I'm treating it as charter-covered (ordering item 2) and running
it. Same for a text-MMRED N-sweep up to ~40 for the C-range design (states-only, no rendering
needed — text pipeline reads states.json/qa.txt; I'd generate states with `--no-render` at
negligible disk). Flag here in case you want it done differently.

## NEW QUESTIONS (2026-07-10 autonomous session)

### Q7 — home-quota incident (RESOLVED operationally; one decision for you)
At 21:29 the home quota hit its HARD limit (330G) mid-campaign: every running job (the whole
first B1 wave, a1_easy, c1b, one block-read) FAILED with EDQUOT — they could not write caches
or even logs. Root cause: home was already at ~327G before the campaign (the HF model cache
alone is ~109G, incl. 64G for Qwen2.5-VL-32B) and my ~3.5G of new artifacts tipped it over.
**Action taken:** freed 30.7G by deleting two RE-DOWNLOADABLE model caches via the official
HF cache API — `OpenGVLab/InternVL2_5-8B` (16.2G, Track B complete) and
`tiiuae/falcon-mamba-7b-instruct` (14.5G, Mamba-LM runs complete). Nothing under outputs*/
or data/ was touched. Corrupt truncated caches from the crashed wave are left in place with
CORRUPT.marker files. All jobs resubmitted; a1_easy's 957MB cache had landed intact before
the crash and is reused. MLVU transfers re-batched to ≤7 GB with prompt deletion.
**Decision for you:** the 64G Qwen2.5-VL-32B cache is the big remaining consumer — OK to
delete it too if no 32B runs are planned soon? (Re-download is one command but 64G.)
> **ANSWERED (Tal, 2026-07-11): APPROVED.** Executed 2026-07-11 (continuation session):
> 68.3 GB freed via the HF cache API; quota now 248G/330G. All other model caches untouched.

### Q8 — VNBench stretch rung: SKIPPED (disk), revisit after cleanup
The optional VNBench counting rung (≤20 GB transient) was approved contingent on primary work
landing — which it did — but after the quota incident the safe transient headroom is ~8 GB
(usage ~310G vs 330G hard limit, my downloader guard trips at 318G). Downloading 20 GB risks a
repeat of the 21:29 cascade. Skipped; everything else on the charter landed. If you clear the
64G Qwen-32B cache (Q7) or raise quota, the port is straightforward: the MLVU pipeline
(experiments/mlvu/{prep_ac_frames,eval_ac_behavior,lookagain_ac}.py) transfers nearly unchanged
to VideoNIAH's counting subset.
> **ANSWERED (Tal, 2026-07-11): APPROVED** — run the VNBench counting stretch after Q7 frees
> the space. Q7 executed; VNBench port proceeding in the continuation session (P3).
