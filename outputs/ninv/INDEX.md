# ninv — canonical runs index

Experiment -> canonical run dir -> headline number. Update when a run becomes
canonical (see CLAUDE.md run conventions).

| Experiment | Canonical run | Headline |
|---|---|---|
| Baseline cross-N leak (pre-fix) | superquery runs 128773 + 128790 (features) | N=8->64 transfer 0.320 (in-N 0.979) |
| **Phase 0 PARK: node posreset — HEADLINE** | `20260809_231854_cap16` + `20260809_232635_cap64_n200` | **N=16 -> N=64 transfer 0.998** (in-N 0.999; CI [0.997, 0.999], n=200) |
| Phase 0 park: N=8 capture (images_park) | `20260809_231331_cap8` | in-N heldout 0.996 |
| Phase 0 park: N=32 capture (longN) | `20260809_231854_cap32` | in-N heldout 0.989 |
| Phase 0 park: cross-domain cell (literal original gate) | `20260809_231331_cap8` -> `20260809_232635_cap64_n200` | 0.940 — FAILS 0.95; cross-DOMAIN not cross-length |
| Phase 0 park: fixed-N cross-root control | `20260809_232635_cap32_park2` | park<->park2 @N=32: 0.976 / 0.998 |
| **Phase 0 HF: GATE PASSED — HEADLINE** | `20260809_234057_hf16` + `20260809_234057_hf64` | **N=16 -> N=64 transfer 0.990** (CI [0.984, 0.996]; majority 0.782; bal 0.981, c2 0.947) |
| Phase 0 HF: N=8 capture | `20260809_234057_hf8` | in-N heldout raw 0.961 / bal 0.908 / c2 0.811 |
| Phase 0 HF: N=16 capture | `20260809_234057_hf16` | in-N heldout raw 0.973 / bal 0.949 / c2 0.880 |
| Phase 0 HF: N=64 capture | `20260809_234057_hf64` | in-N heldout raw 0.956 / bal 0.944 / c2 0.883 |
| **Leaf quality on HF (Phase 1.1 gate, already met)** | `20260809_235142_hf8_leaf392` / `..._leaf512` | **leaf evidence-recall 0.995 @392, 1.000 @512** (gate >=0.985 met with zero training) |
| ~~HF merge anomaly~~ **RETRACTED — readout artefact** | same two runs | ridge+round shrinks under HF's 0.8 zero prior; c2 0.822 -> 0.950 from the decision rule alone (park unchanged) |
| **Merge encodes the count (OR-vs-SUM rejected)** | `..._leaf392` / `..._leaf512` / `20260810_000358_park{8,16}_leaf` | binary "exactly two" balanced 0.982 @392, 0.998 @512, 0.996-0.999 park |
| **Real residual: verdict MARGIN** | same four runs | park margin +10.0/+10.4 with ZERO c2 failures; HF +7.96 success vs +3.98 failure |
| Control: park vs HF c2 recall | all park + HF captures | park 0.995-1.000 at every N; HF 0.950-0.990 under a non-shrinking readout |
| **FROZEN two-pass headline on HF N=8** | `20260810_151238_twopass_hf512` / `_hf392` | **EMIT-EM 0.960 @512 / 0.920 @392** (park anchor 0.980@392; cond-EM 1.000 both; benchmark test pool n=50) |
| L14 leaf probe (for code writes) | `20260810_151634_hf8_rawL14/probe_L14.npz` | held acc 0.990, evidence-recall 0.952 (verdict still forming at L14) |
| **Capacity law over CODES (fan-8)** | `20260810_151952_hf8_quantcap` vs `_hf8_rawL14` | **fan-8 merge: raw 0.527 -> codes 0.887**; fan-4: 0.857 -> 0.917 |
| Quantize-at-L14 margin-tail verdict | same pair | codes perfect HIGH-margin c2 (67/67) but freeze the L14 probe's own tail errors (LOW 0.855) |
| **P1 arm B (quantized) — training** | `20260810_150229_p1_armB/20260810_150504_quantized_r8` | val EM 0.730 (maj 0.510), cut mid-climb at +0.08/ep; registers_best.pt = ep10 |
| P1 arm A (raw) — training | `20260810_150229_p1_armA/20260810_150505_raw_r8` | val EM 0.510 = majority; never counted |
| **P1 step-4 eval, arm B** | `20260810_170231_evalB` | per-level 0.989/0.981/0.968/0.926; N=64 transfer 0.976/0.960/0.932/0.880; lv5-6 (never trained) dead |
| P1 step-4 eval, arm A | `20260810_165814_evalA` | 1.000/0.890/dead/dead; N=64: lv1 1.000, lv2 0.921, deeper dead |
| Canary (corrected): no leak | `20260810_174323_possym` + `20260810_173729_canaryB_T0` | 3-channel pos symmetric, T0 bit-exact, 0/10 answer flips on content-preserving swaps; bit-identity canary retired |
| **P1 arm D (token-anchor) — THE ARM** | `20260810_201213_p1_armD/20260810_202412_raw_r8` + eval `20260810_231226_evalD` | **per-level 1.000/1.000/1.000/1.000 in-length; N=64 transfer 1.000/1.000/0.998/0.975; EM 0.870** (maj 0.510) |
| P1 arm C (frozen leaves) | `20260810_201213_p1_armC/...` + eval `20260810_230152_evalC` | tree alive (0.998/0.994/0.961/0.778) but EM never left 0.510 — stability fixes merge, not readout coupling |
| Depth extrapolation: CLOSED NEGATIVE | both evals | lv5/6 dead in C AND D; shared alphabet does not buy zero-shot depth; Phase 1.5 or capped-tree REQUIRED |
| **v2 (D + mixture + ans-balance)** | `20260810_215334_p1_v2/20260810_215544_raw_r8` + eval `20260811_013156_evalV2` | **EM 0.920 (gate ≥0.90 PASSED); probes 1.000 flat; N=64 transfer 0.999/0.999/0.998/0.990** |
| P1.5 (v2 + synthetic depth) | `20260811_105241_p15/20260811_105339_raw_r8` + eval `20260811_132203_evalP15` | in-length EM 0.925; lv1-4 @N=64 1.000 flat; lv5/6 real-tree transfer DEAD (interface located) |
| **THE FLAT-IN-N EMITTED CURVE (cascade)** | `20260811_145257_cascade_p15c` | **0.925 (N≤16) / 0.875 (N=32) / 0.811 (N=64)** vs majority 0.42/0.16/0.04; lv4 fidelity 0.99+; heldout head 1.000 |

Instruments: `scripts/ninv/transfer_test.py` (single gate cell),
`scripts/ninv/transfer_matrix.py` (full matrix + error anatomy),
`scripts/ninv/load_hf_sample.py` (MMReD-HF adapter + self-check).
