# outputs/recagg/ — INDEX (hand-maintained; experiment → canonical run → headline)

Campaign brief: [CAMPAIGN_BRIEF.md](CAMPAIGN_BRIEF.md) · log: [STATE.md](STATE.md)

**Resolution policy (Tal, 2026-08-17): HF cells run at 512 (the benchmark's native
resolution). 392 is DEPRECATED — kept only as the resolution-comparison arm.**
**Canonical head recipe: epochs=20000** (the 3k-epoch budget undertrains the SSM's
CE readout — Tal's undertraining hypothesis, confirmed 2026-08-17).

| experiment | canonical run | headline |
|---|---|---|
| P0 heads, park N=8 (in-length, balanced; maj 0.124, bound 0.988) | `p0_heads/20260817_152313_ssm20k_park8/` | R1 0.988 · **R2 GRU 0.992** · **R3 SSM 0.984** — all three at the perception bound; **R4 attn-pool 0.302** (trainEM 1.0 — failure is structural, not budget) |
| P0 heads, MMReD-HF N=8 @512 (PRIMARY HF cell; maj 0.632) | `p0_heads/20260817_152712_ssm20k_hf8_512/` | **R1 0.998 (bal 1.000)** · R2 GRU 0.970 (bal 0.932) · R3 SSM 0.910 (bal 0.818) · R4 0.728 = prior-riding (bal 0.262) |
| P0 heads, park N=16 (in-length; maj 0.108) | `p0_heads/20260817_153943_20k_park16/` | R1 0.986 · **R2 GRU 0.998** · **R3 SSM 0.952** · R4 0.242 — all three integrators at the bound @N=16, attn-pool out |
| SSM undertraining ablation (3k vs 20k epochs) | `p0_heads/20260817_150845_v2_{park8,hf8_392}/`, `20260817_152313_v2_hf8_512/` (3k) vs the 20k rows above | R3 cls park8: 0.298 @3k → 0.984 @20k (optimization budget, not architecture); R1/R2/R4 move ≤0.02 |
| P0 heads, MMReD-HF N=8 @392 (DEPRECATED — resolution comparison) | `p0_heads/20260817_150845_v2_hf8_392/` | R1 0.988 (bal 0.969); R2 0.914 (bal 0.765); 392→512 gains mirror ninv leaf-recall 0.995→1.000 |
| P0 v1 (ablation: count-only supervision, raw 3584-dim) | `p0_heads/20260817_144419_{park8,hf8_392}/` | all trained heads memorize (trainEM 1.0, eval ~0.2/~majority) — motivates aux-bit supervision + subset-count augmentation |
| P1 leaf captures N=16–64 (park + HF@512, L16+L20) | `p1_captures/20260817_154347_*/` | 5 captures, 59 GPU-min total; all sanity-checked (Y-sum==G, healthy dists) |
| **P2 extrapolation, HF@512 (H2)** — fit @{8,16} → zero-shot @{32,64} | `p2_extrap/20260817_160326_hf512/` | perception bound = 1.000 everywhere; EM_reg @N32_zs/N64_zs: **R1 sum-probe 0.996/1.000 (exactly invariant)** · R3 SSM 0.800/0.220 · R2 GRU 0.640/0.256 · R4 attn 0.236/0.048. Ordering as pre-registered; absolute invariance only in the symbolic probe→sum |
| **Arm A frozen readers** (oracle captions, 7 models, N=8–128) | `armA_readers/20260817_162559_*/` | NO frozen LM counts reliably (best Qwen14B 0.695@8, 0.515@16; all ~majority by 32–64); mamba/pythia pair floors at majority (task-prior absence, not architecture); zero-shot frozen readers cannot carry the task-agnostic endpoint → adapter route |
| Arm A tally-scratchpad variant (xlstm/mistral/qwen14b) | `armA_readers/*_tally/` | NO rescue at ≤14B — even Qwen14B loses its own running count by N=64 (0.060, at 20× the compute); externalized state needs external *execution* too |
| Adapter route: mamba-2.8b LoRA count fine-tune (train ≤16) | `armB_adapter/*_mamba_count_ft/` | circuit installed (0.96@8, 0.88@16 on real pools from synthetic training) but drifts like from-scratch R3: 0.58@32, 0.14@64, 0.00@128; **chunked inference repairs it: 0.78/0.58/0.42 @32/64/128** (`*_mamba_ft_chunk16/`) |
| H2 mechanism: R3b-noleak + chunk-16 D&C | `p2_extrap/*_hf512_mech/` | @N=64 zero-shot: R3b noleak 0.864 (drift = length-coupled learned params); chunk-16 rescues all integrators (R3 0.900, R2 0.824, R3b 0.928); R4 unrescued (0.064) |
| **Arm C Ask–Compile–Execute** (task-agnostic endpoint; oracle records, 24 types, N=16–128) | `armC_compile/*_qwen14b_v3/` (v1 = sandbox audit, v2 = pre-definitions level) | **LENGTH-FLAT: ALL ~0.89, UNSEEN ~0.88 across N=16→128** (v3: +definitions block, +reversed-scan exemplar); 16/24 types ≥0.88; residual = nested two-stage lookups; exec-fail 0.3% |
| **Arm B composed frozen system** (VLM captions → v3 programs, controlled perception swap) | `armB_captions/20260817_203538_full/` + `armB_execute/20260817_213301_v3progs/` | caption fidelity 0.913/char, 0.637/frame-exact; **ALL 0.79/0.81/0.72/0.72 vs oracle 0.92/0.89/0.85/0.90 (N=16→128)**; single-frame types ~1.00 flat; exact-count types decay as p^N — residual length-coupling lives ONLY in perception, as predicted |
| Related-work sweep (delegation/length-gen literature) | `RELATED_WORK.md` | Apple 2510.14826 = necessity precedent (text/SSM); PCW/APE = fencing's structural twin (text); combination (matched-perception drift law + fenced VLM perception + spectrum map) unclaimed |
