# outputs/herbench — HERBench Action-Counting transfer test of the d′ theory

> Goal: test the MMRED d′/parity framework on a real benchmark counting task
> (HERBench lite, Action Counting, HD-EPIC egocentric video; arXiv 2512.14870).
> Data prep: `experiments/herbench/prep_ac_frames.py` → `data/herbench_ac/`
> (armA_evidence_only: N=true_count, all evidence; armB_ev_fill16: k evidence
> + fillers to N=16, binary per-frame labels from `required_timestamps`).
> All runs 2026-07-07, 7B nf4, 448px frames, single GPU (l40s/a100), jobs noted.

| experiment | canonical run | headline |
|---|---|---|
| own-answer armA (evidence-only, n=144) | `own_answer/armA_20260707_213726/` (job 119014) | **EM 0.049, bias −3.5**; gold1 0.47, gold≥2 → 0.00; mean_pred saturates ~2–3 (MMRED undercount signature on real video); MCQ-mapped 0.27 |
| own-answer armB (ev+fill16, n=134) | `own_answer/armB_20260707_213726/` (job 119014) | EM 0.17 (gold-2 collapse), bias −2.5 |
| carrier localization (25-offset × 5-layer map) | `probes/ac_locmap/20260707_213723/` (job 119013) | carrier = quoted action-pair tokens (off −9/−10), L12–16, naive SNR 0.86; per-frame AUROC ≈ 0.80; model digit-argmax 0.082 |
| message cache joint | `probes/ac_msgcache_joint/20260707_215544/` (job 119017) | d′_AUC 0.98–1.10 @carrier; linear-on-sum 0.17–0.22 (maj 0.19); dtc 0.18–0.23 (maj 0.22) |
| message cache fenced (isolation mask) | `probes/ac_msgcache_fenced/20260707_215544/` (job 119018) | **d′ 0.86–1.09 ≈ joint — isolation NULL on real video** (fence verified, Δv@L12=5.75); extraction weak at the source, unlike MMRED (3.1→5.2) |
| adequacy (E4) + law prediction | CPU on joint cache (console, this session) | **Gaussianity REJECTED**: skew→+3.5, exkurt→+35, std-ratio~2, d′_gap≠d′_AUC (graded-evidence mixture); pooled sum partially re-Gaussianizes (exkurt +4.7); law (d′ 0.98, N=16, prior-mixed) predicts 0.158 vs measured 0.196±0.041; corr(sum-proj, gold)=0.73 |

**Registered-prediction scorecard:** (a) ladder ordering model 0.08 < linear ~0.19 ≤ dtc ~0.20–0.23 ✓ (compressed, as low d′ dictates); (b) parity within error ✓ but weakly discriminating (prediction ≈ majority regime) and adequacy fails → quote as ordering + magnitude only; (c) fence raises d′ ✗ — NULL, the informative surprise; (d) evidence-only undercount ✓✓.

**Two-regime synthesis:** MMRED = binary evidence, extraction-strong, aggregation/readout-limited (law exact). Real video = graded evidence (mixture, not Gaussian), extraction-weak at the source (d′≈1, isolation null), everything downstream compressed to the prior. Same instruments diagnose both regimes; the E4 self-diagnosis correctly rejects the closed form where it shouldn't be quoted.

| look-again judge 448 (curation) | `data/herbench_ac/armB_ev_fill16/*/lookagain.json` (job 119020) | judge AUROC 0.832; median P(yes) on true occurrences 0.40 — ~3/4 of evidence unverifiable per-frame |
| forced-binary curated d′ (CPU, 448 cache × judge 0.7/0.3) | console (this session) | keeps 26% of evidence; **d′ 0.98→2.10**, kurtosis +25→+3.3 — the crisp quarter behaves like MMRED |
| hi-res 896 caches joint/fenced | `probes/ac_msgcache_{joint,fenced}_hi/20260707_22*/` (jobs 119027/119028) | **resolution NULL**: joint d′ 0.90–1.07, fenced ≈ joint, model 0.052, dtc 0.200 |
| look-again judge 896 | `data/herbench_ac_hi/.../lookagain.json` (job 119029) | AUROC 0.860 (+0.03) — perception ceiling intrinsic, not pixels |

Smokes: `outputs/_scratch/herbench_smoke_*` (n=5, jobs 118973/118974).
Videos: 28 mp4s (22.7 GB) on login-node /scratch (session scratchpad) — NOT visible to compute nodes; re-fetch via `scratchpad/range_fetch_ac_videos.py` if needed. Frames (130 MB) are the durable artifact in `data/herbench_ac/`.
