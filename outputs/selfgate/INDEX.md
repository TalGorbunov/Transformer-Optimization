# outputs/selfgate — INDEX (SELFGATE campaign, 2026-09-14→16; brief: CAMPAIGN_BRIEF.md, log: STATE.md)

From the oracle gate toward a label-free gate: G0 = gate off the model's own selector head (no training), G1 = CoGNN-style
straight-through gate trained by the answer loss only. Scripts now under `legacy/v1/scripts/selfgate/`
(`probe_headscan.py`, `train_sft_selfgate.py`, `diag_grad_selfgate.py`, `diag_maskgrad_micro.py`); sbatch `legacy/v1/slurm/selfgate_*.sbatch`.
Substrates: `checkpoints/sft_fenced_gated_vn_nfree_adapter` (S9b, N-free) and `sft_fenced_le16_ep10_adapter` (P1b). Regime: park data, count, N-free prompt.
Verdict: G1 closes as measured (gate saturates open at every λ); G0b and G1-multi were not run.

| experiment | canonical run | headline |
|---|---|---|
| G0a smoke | `outputs/_scratch/selfgate_g0a_smoke/` (148359) | machinery + both adapters + nfree layout pass; first atrophy signal on S9b |
| G0a headscan, P1b | `g0a/p1b_N{8,32,128}/` (148381, 90 samples/cell) | S10 anchor reproduced (L24h20 0.9997/0.9998/1.0000); a POPULATION of selectors (L20 h1/h3/h15, L24 h4/h5/h6/h17/h20 ≥0.998 at N≥32) |
| G0a headscan, deployed S9b adapter | `g0a/s9b_N{8,32,128}/` (148381) | **H-G0 clause 1 MISSED**: no head ≥0.98 at any N (best 0.52–0.75; 0.586 @128) — oracle-gated training atrophied the internal selector |
| G0c cross-question AUC | `g0c/p1b_{exists,majority}_N{8,32,128}/` (148593) | population 1.000 at every N under both questions; L24h20 0.9998–1.000 in 5/6 cells, 0.789 at majority@N=8 (`auc.json`) |
| H-SAFETY diag chain | `outputs/_scratch/selfgate_{diag,diag2,diag3,diag4,diag5,diag6,micro,micro2}/` (148493/148501/148522/148538/148542/148563/148505) + `logs/sg_diag*.out` | EFFICIENT sdpa has no mask grad → MATH; **diag4: hard-closed gates are gradient black holes** (gmask.grad 2.3, bits.grad 0) → soft-log train mask; diag5 PASS (gate cos 1.000000, both directions live); diag6: GC × grad-mask crashes in both modes → LoRA ≥L12, GC off |
| G1a smokes | `outputs/_scratch/selfgate_g1a_smoke{,2}/` (148554/148574) | GC-off all-layer OOMs @N=16 (~100 GB); ≥L12 scope (11.53M p) trains @N=16 |
| G1a attempt 1 | `g1a/20260914_143253_selfgate/` (148543) | crashed 2m43s: reentrant GC re-consumes the grad-carrying mask graph |
| **G1a answer-loss-only gate (λ=0)** | `g1a/20260914_145650_selfgate/` (148575, 3h29) | **mean_bit 1.000 from ep1** (fp 609/1547/3459, fn 0); val 1.000 @ep7, TEST_IID 1.000; exams 1.000/1.000/0.360 @8/16/32 = ungated snap (oracle-gated same substrate 0.847 @32) — H-G1a missed, G1b trigger MET |
| G1b λ=1e-3 | `g1b_l1e-3/20260915_201150_selfgate/` (150418, ~6h) | mean_bit 1.000 every epoch; exams 0.927/0.273/0.160 |
| G1b λ=1e-2 | `g1b_l1e-2/20260915_201205_selfgate/` (150419, ~6h) | mean_bit 1.000 every epoch; exams 1.000/0.273/0.160 — L1-dampened train mask ⇒ train/eval mismatch when the gate never commits |
| G0b threshold self-gate · G1-multi | — (not run) | superseded: share law predicts a fixed θ fails at N≥32; deployable-agnostic-gate line moved to QGATE, incentive fix (virtual-N) deferred to QGATE C3 (also unrun) |

No adapter promoted. Figures FG1–FG3 from the brief were not produced. RESULTS entry logged 2026-09-21.
