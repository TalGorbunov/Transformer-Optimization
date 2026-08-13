# frame_axis — result index

> Canonical run per experiment + headline number. Read this to find "the" result; full archive of all
> runs (incl. smokes/superseded) is in `../../outputs_long/`. Tasks: steps_in_room / rooms_visited / co_occupancy.

## Message-sum decodability (`probes/message_sum/`) — RESULTS.md [2026-06-28]
- **headline:** mixing evidence+non-evidence in one additive channel, and normalizing it, are *each* sufficient to wreck linear count decodability. Script: `experiments/glstm/probe_message_sum_decodability.py`.
| read | canonical | result |
|------|-----------|--------|
| Exp2a interference (steps, L19, seq8) | `probes/message_sum/20260628_193258/` | S_evid→g **1.000**, S_nonev→(8−g) **1.000**, **S_all→g 0.45** (crowd) / 0.46 (decrowd) |
| Exp2b normalization | same | SUM_evid **1.000** vs MEAN_evid **0.42** → count is in magnitude |
| Exp1 prefix curve | same | round-acc 0.99→0.45 (j 1→8) but **R² flat ~0.88**; crowded≈decrowded |
| U-shape | same | S_all per-count **flat** (no U) → model U-shape is downstream readout, not sum interference |
| ref: last-token L19 → gold | same | acc 0.30 / **R² 0.78** (count graded but poorly discretized) |
| caches | `probes/message_sum/cache_{crowded,decrowded}/minimal_L19_steps_in_room.pt` | 800 ex each, gold ≈ uniform 0–8 |

### Corrected mechanism + decomposition (RESULTS.md [2026-06-28b])
- **mechanism:** count read off a ~1%-magnitude direction δ; `corr(‖S_evid‖,g)=+1.0` vs `corr(‖S_all‖,g)=−0.14` (fixed set-size kills the magnitude readout); per-frame SNR 0.33; fix = per-frame nonlinearity **before** the sum.
| read | script | result |
|------|--------|--------|
| decomposition | `probe_aggregation_decomposition.py` | ‖μ_all‖87 / ‖δ‖1.0 / σ6.0 / SNR 0.33 / pred S_all SNR/count 0.12 |
| nonlinearity ceiling | `probe_nonlinearity_ceiling.py` | sigmoid-then-sum **0.73** > linear 0.45 > MLP-after-sum 0.34 > last-tok 0.30 |
| layer sweep (L27 R²) | `probe_message_sum_layersweep.py` | S_all R² L19 0.89 → L27 0.86; last-tok R² always below sum |
| Garcia align (image) | `probe_count_readout_alignment.py` → `count_readout_alignment/image_crowded_n150/` | reg_r2 0.85, align 0.008 (orthogonal), realign ceiling 0.37 |
| GAIN causal null | `denom_gain_vs_temp.py` → `denom_gain/` | gain×N → 0.00 at sl6/8; temp hurts; no in-place attention fix |
| flow staircase (front-Q) | `group_restoration_importance.py` → `flow_diagnostic/` | frames@L13 0.75; last-tok L19→27 0.29→1.0; ⚠️ switch to frames-first |

## OOD count-extrapolation benchmark (`readout_benchmark/`, `cache/minimal_L19_*.pt`) — RESULTS.md [2026-06-23]
- **headline:** per-frame-supervised **fixed-extensive readout** (sum=occurrence, soft-OR=distinct) **extrapolates** to unseen counts; base/CoT/LoRA/classifier and all **learned** readouts collapse.
| method | steps OOD | rooms OOD | co-occ OOD |
|---|---|---|---|
| **sum / soft-OR (per-frame-sup)** | **0.996** | **1.000** (soft-OR) / 0.974 (sum) | **0.974** |
| LoRA / base / classifier | 0.45 / 0.11 / 0.00 | 0.00 / 0.00 / 0.00 | 0.09 / 0.00 / 0.00 |
| canonical DeepSets (learned ρ, even +sup) | 0.13–0.59 ✗ | 0.20–0.54 ✗ | 0.08–0.38 ✗ |
- **principle:** extrapolation needs a *parameter-free* fixed extensive readout on the supervised per-frame quantity; any learned ρ breaks it. CSVs: `readout_benchmark/{benchmark,stability,deepsets_proper,deepsets_framesup,deepsets_universal,auxloss}.csv`. Scripts: `experiments/glstm/benchmark_*.py`, `cache_minimal_frame_reps.py`.

## Minimal-crowding aggregator experiment (`agg_min/`) — RESULTS.md [2026-06-23]
- **headline:** decrowding (clean extraction) lifts every task to ~0.9–1.0 with plain DeepSets; **soft-OR perfects rooms (1.000)**. Spec: `SPEC_minimal_crowding_aggregator.md`.
| cell | path | test_iid lm |
|------|------|------|
| rooms deepsets / logic | `agg_min/rooms_visited_{deepsets,logic}/` | **0.973 / 1.000** (perfect all counts) |
| steps deepsets / logic | `agg_min/steps_in_room_{deepsets,logic}/` | **0.950 / 0.961** (beat 0.86 ceiling) |
| co-occ deepsets / logic | `agg_min/co_occupancy_{deepsets,logic}/` | **0.899 / 0.938** (soft-sum +0.039) |
| LoRA baseline (softmax) | `agg_min/lora_{rooms,steps,cooc}/` | running |
| extraction vs #chars (probe) | `probes/{phimsg_min,crowding_min}/` | rooms 0.835(5ch)→0.996(1ch); φ preserves |

## Headline
- **Adapter relieves the aggregation bottleneck task-agnostically; deepsets wins; residual is frozen
  perception (steps/co-occ) + one-pass set-aggregation (rooms_visited, ~12pt gap).**

## Canonical runs
| what | path | headline |
|------|------|----------|
| **shared rep cache (L19)** | `cache/L19.pt` | 10500 (dir,task) reps, seq 1–8 |
| **best deployable (deepsets live +10ep)** | `adapter_live/live_h2h_cont/` | **IID 0.802** (steps .79/rooms .75/co-occ .87), unbiased |
| live 4-way head-to-head (LM-injection) | `adapter_live/live_h2h/` | deepsets best OOD 0.558; PNA hurts OOD; codebook dead |
| live IID/OOD diagnostic plots (deepsets) | `adapter_live/live_deepsets_eval/` | mean_pred tracks y=x |
| attn-pool (negative) | `adapter_live/live_attnpool/` | 0.727 < mean-pool 0.754 — does NOT help |
| cached aggregator sweep (6 configs) | `adapter_cached/sweep/` | deepsets 0.720 best; no config beats it; codebook falsified |
| sum ablation (sum/summax/deepsets) | `adapter_cached/sum_ablation/` | summax≈deepsets; sum-only worst |
| logic aggregator (sum+OR+AND) | `adapter_cached/logic_test/` | helps co-occ (+0.06); **fails rooms** (OR saturates) — has ceiling plots |
| deepsets train+test 1–8 | `adapter_cached/aggregator_cached_1to8/` | bias→0 on full length range |
| **acc-per-count + ceiling plot** (h2h_cont best) | `adapter_live/h2h_cont_evalplot/` | steps 0.79 (on ceiling) / rooms 0.748 (below) / co-occ 0.867 (above hard bound); canonical figure |
| **rooms-only deepsets 30ep** (single-task) | `adapter_live/rooms30/20260620_182313_deepsets/` | test_iid **0.691** ≪ 0.865 bound; plateaus → **rooms aggregation-limited** |

## Probes (read side exhausted → ~0.94/frame frozen ceiling)
| probe | path | result |
|-------|------|--------|
| evidence layer sweep (steps) | `probes/evidence_selection_image/`, `_linear/` | peak L19–21, AUC 0.98–0.997 |
| per-task per-layer extraction | `probes/pertask_extraction/` | rooms room-decode L21 0.925 (L19 .915); co-occ L19 0.996 → L19 not limiting |
| φ preserves evidence | `probes/adapter_messages/` | adapter φ 0.931 ≈ raw 0.939 → pooling not lossy |
| multi-layer read | `probes/multilayer_evidence/` | concat gains ≈0 → redundant |
| token-level extractor | `probes/token_extraction/` | mean+linear best; attn/max don't beat it |
| perception vs binding | `probes/perception_binding/` | ⚠️ FLAWED (mean-pool + arbitrary pairs) — discarded |
| **32B steps is-evidence** | `probes/evidence_selection_image_32b/` | L43 0.892/0.951 vs 7B 0.939/0.984 → **no perception lift** |
| **32B co-occ + rooms** | `probes/pertask_extraction_32b/` | co-occ AUC 0.999 (7B 0.996, saturated); rooms 0.858 → no lift |
| **resolution sweep + crowding** | `probes/extraction_resolution_crowding/` | AUC plateaus at native512 (.969→.982@672); crowd 5<4 (weak) → not resolution-bound |
| **per-frame verification (steps)** | `probes/per_frame_verify_steps/` | per-frame AUC **1.000**; count **0.79→0.928** → 0.94 is single-pass superposition |
| **per-frame verification (co-occ)** | `probes/per_frame_verify_cooc/` | count 0.722 < single-pass 0.867 → co-occ soft-agg-optimal, not extraction-limited |
| **balanced image vs text deepsets** | `adapter_live`... `balanced/{rooms,cooc}_{image,text}/` | rooms img 0.50/txt 0.40; co-occ img 0.586/txt 0.44; image≥text; balanced→honest numbers |
| **text controls (steps / target@L1)** | `balanced/{steps_text,rooms_text_targetL1,cooc_text_long}/` | steps·txt 0.70; **rooms target@L1=perfect-extraction → still 0.41** (aggregation-bound) |
| **L19 extraction vs crowding** | `probes/crowding_min/` | rooms 1ch 0.996→2ch 0.915→5ch 0.835; steps 0.984→0.970→0.939; co-occ ~0.99 always; **superposition** |
| **frozen base acc (minimal crowding)** | `probes/base_acc/` | steps·1ch 0.583 (U-shape), rooms·1ch 0.289, co-occ·2ch 0.204 — frozen aggregation fails even at 1 entity |
| **text pooling × layer sweep** | `probes/text_pooling_sweep/` | rooms only via target@L1=1.0; co-occ mean-best 0.964 |

## d′-parity validation (`probes/dprime_parity/`) — RESULTS.md [2026-07-03c]
- **headline:** the zero-fitted-parameter law `acc = prior-mixed 2Φ(d′/2√N)−1` predicts the measured linear decode of the summed reps/messages in every well-powered regime, question-first AND at the deployed room-carrier; ρ(joint)≈0.10 vs ρ(multipass)≈0.01 (superposition = noise correlation). Script: `experiments/glstm/probe_dprime_parity.py`.
| read | canonical | result |
|------|-----------|--------|
| Q-first 16-regime parity (E1–E4) | `probes/dprime_parity/20260703_162729/` | N-sweep pred .849/.662/.575/.504 vs meas .853/.650/.593/.471; MLP−linear < 0 everywhere; parity.png |
| deployed-locus parity | `probes/dprime_parity/20260703_170355/` | room@L16 pred **.375** vs meas **.363**; ladder model .236 < lin-sum .36–.40 < dtc .47 |
| deployed message cache | `probes/carrier_message/count_msgcache/count/` | messages_cache.pt, n=600, L14/16/18/20, off 9=room/13=char (job 116866) |
| E5 causal dose-response | `probes/dprime_dose/20260703_182429/` | λ=4 dose: last-token decode .367→**.533** (random ctrl .400) yet emitted answer flat .23→.20 → readout wall causal; λ=0 ablation leaky (job 116926) |
| E5b multi-layer dose + scrub | `probes/dprime_dose/20260703_185555/` | scrub → undercount collapse (MAE 1.51→2.47, g≥3 acc≈0) = δ channel load-bearing; single λ=16 decode **.700** vs emitted .193 → readout wall widens with dose (job 116938) |
| E5c scrub-control + repaired readout | `probes/dprime_dose/20260703_192542/` | random-axis scrub inert (.247/1.50) vs δ̂-scrub collapse (.180/2.47); decode@FINAL ≈ decode@L24 (λ8: **.683**) vs emitted .187 → gap lives entirely in the unembedding (job 116943; high-λ ladder job 116945 ▶) |
| E5c high-λ ladder | `probes/dprime_dose/20260703_194553/` | decode saturates .70–.78 (λ16–64; pipe limit = room→last hop), declines at λ128; emitted degrades to .133; λ64: repaired **.783** vs emitted .133 = 5.9× readout gap (job 116945) |

## d′-theory rollout: co-occ + rooms (RESULTS.md [2026-07-04])
| read | canonical | result |
|------|-----------|--------|
| cooc carrier map | `probes/carrier_message/cooc_locmap/co_occupancy/` | distributed: char1 1.49@L14, char2 1.29@L14; model .155, dtc .601 (job 116997) |
| rooms carrier map | `probes/carrier_message/rooms_locmap/rooms_visited/` | char token 1.22@L16; model .087, dtc **.803 = 10× model** (job 116998) |
| message caches | `probes/carrier_message/{cooc,rooms}_msgcache/` | cooc off 13/15/10, rooms off 9/10, L14-20, n=600 (jobs 117005/117014) |
| cooc deployed parity + block | `probes/dprime_parity/20260704_011808/` | single-token under-predicts; **2-name block d′ 3.19: pred .456 vs meas .484** |
| cooc 6-locus block read | `probes/carrier_message/cooc_block_read/20260710_210215/` | **block d′ 3.43@L14** vs single 3.05 (gain +0.38/+0.42 = borderline); 3-task ordering image +0.14 < cooc +0.4 < text +0.47; E4 PASS; ladder model .138 < law .476 ≈ ridge .534 < dtc .648 (job 119985) |
| rooms K-channel parity | `probes/dprime_parity/20260704_015706/` | d′_r≈5.0@L14; **linear .40 < Σ-threshold .65 (pred .51) ≪ dtc .99** — structural split measured |
| E5-cooc + H3 causal map | `probes/dprime_dose_cooc/` | **double dissociation**: scrub char2 → .100 collapse, scrub char1 → null, random inert; dose ×16 → decode .767, emitted .133 (job 117015) |
| E5-rooms subspace scrub + tally dose | `probes/dprime_dose_rooms/` | scrub span{δ_r}@char → **.053** collapse (random 6-dim subspace .120 ≈ base); tally dose ×16 emitted .133→.233 but MAE flat = magnitude-coupled, structure-blind readout (job 117100) |
| big caches + verdicts | `probes/carrier_message/{rooms,cooc}_msgcache_big/` | rooms n=720 (dataset max): all K-channel numbers replicate; cooc n=1080: ternary hypothesis REFUTED, 1-D same-axis ≈ full-dim (sufficiency at cooc carrier) |
| carrier anatomy (cooc) | `probes/dprime_dose_cooc_anatomy/` | relay-then-bind: char1 matters EARLY (scramble L2-12 → .113) & inert late; char2 inert early & load-bearing late; dose at causal carrier moves behavior (.187→.240); last-token δ̂ null (job 117213) |
| distinct_* K-channel (Q-first) | console log w/ `dprime_parity/20260704_015706/` | d′_c 1.1-1.5 (K=9) → hard threshold COLLAPSES (.13) vs linear .51 — the gate's low-d′ regime; mirror of rooms |
| anatomy n=400 (steps+cooc) | `probes/dprime_dose_{steps_anatomy,cooc_anatomy400}/` | transfer completes ~L17 (late scrub NULL, mid scrub collapses); relay-then-bind replicates both tasks; clean controls (jobs 117351/117352) |
| first_occurrence localization | `probes/carrier_message/firstocc_locmap/` | model .432 < majority .513; carrier d′ 1.27@L16; per-frame AUROC .948 (job 117354) |
| **BATCH-0 tally-register solution** | `probes/tally_solution/20260704_150209/` | **rooms 0.99-1.00 IID / 0.96-0.98 count-OOD (model .087); cooc block-gate 0.74-0.77 (model .155); steps 0.52 = its carrier-d′ ceiling as predicted** |
| E6 native reading axis | `probes/native_axis/20260705_153147/` | cos(native,w*)=**.005**, d′_native **0.51** → law on the model's OWN axis predicts its accuracy (.17 pred vs .21 meas); L20 native axis ρ=+.40 sink-like (job 117809) |
| dilution/tilt N-sweep (N=2/4/8) | `probes/carrier_message/count_msgcache_N{2,4}/` + console | amplitude dilutes ∝1/N exactly (‖msg‖ 11.9→6.5→3.4) but d′ FLAT (~2.2–2.5): dilution is info-neutral ≤N=8 — √N law is the whole story in range; volnorm reader +0.2–0.3 d′ at every N (jobs 117950/117951) |
| length-sweep dilution test | `probes/carrier_message/count_msgcache_N{2,4}/` | ‖μ_msg‖ ∝ 1/N confirmed but d′-NEUTRAL (raw 2.08→2.47); content length-invariant; unit-norm reading +0.23-0.34 d′ = "Tier 1.5" (jobs 117950/117951) |
| native-axis stability across N (E6b) | `probes/native_axis_N{2,4}/` vs `native_axis/20260705_153147` | cos(N2,N8)=0.82, cos(N4,N8)=0.92, coherence 0.63–0.66 at all N: the model's readout direction is length-invariant (frozen), so the readout wall is constant across seq lens (jobs 118543/118544) |
| fenced attention (single-pass multipass emulation) | `probes/carrier_message/count_msgcache_fenced/` | REFUTED: blocking cross-frame visual attention L0-13 HURTS — d′ 2.47→1.93 @L16, model acc 0.215→0.120 (n=400, job 118557, self-check |Δv|=5.4). Multipass gain = in-distribution isolated processing, not absence of cross-frame edges; Tier-2 remains a real pass |
| native-port doses (E6c, user-proposed) | `probes/dprime_dose/native_port/20260706_225256/` | FIRST dose to move behavior: natdose×2 emitted 0.240→0.300, MAE 1.48→1.06, unlocks g5-g7 (base emits 0.00 there); overshoots at λ≥4. natscrub: emitted −4.4pp with decode@L24 intact (0.22≈base) = double dissociation vs δ̂-arms (content w/o behavior vs behavior w/o content). natset: following 0.176 < coincidence 0.240 → port NOT graded (confidence knob, not number line); write-count-into-port fix refuted; symbolic re-entry stands (n=250, job 118594) |
| overnight causal battery (6 experiments) | `probes/dprime_dose/causal_night/`, `probes/carrier_transplant/`, `probes/attn_edge_patch/` | TRANSPLANT: real-state carrier swap, emitted follows DONOR 0.04→0.18 monotone in α, repaired-read 0.05→0.20, same-gold control harmless (0.28, MAE 1.16) — count transplants with NO synthetic direction. EDGE-PATCH: cutevid 0.353→0.207 vs cutrand 0.380 (route exclusivity); cutlate NULL → carrier→last transfer causally ≤L17. RESCUE@L21 null (= scrub exactly) → post-window repair impossible = third window confirmation; window-timed rescue resubmitted. NOISEW: null at L14-20/±caveat (decode instrument weak; noise post-window). SETG along δ̂: anti-follows (0.068 < 0.24 coincidence) = Liu's decodable≠correctable at our locus (jobs 118588/89/90) |
| window-timed rescue + pixel pairs (follow-ups) | `probes/dprime_dose/rescue_early/`, `probes/pixel_minimal_pair/` | RESCUE: early injection also fails (0.204 ≈ scrub 0.196) — continuous scrub re-erases the injection at the next layer; with single-site scrub known leaky (re-delivery), rescue is INFEASIBLE-BY-MECHANISM: attention re-delivers messages every layer, so removal must be continuous, leaving no restoration slot. The early/late nulls bracket the ≤L17 window. PIXPAIRS: behavioral down-edits register (56% of predictions move, 53% move down vs 23% control churn; MAE→new gold improves 1.37→1.24); up-edits blocked by the undershoot bias wall (19%); carrier-state w*-projection instrument CONFOUNDED (control shifts −0.034 ≈ down −0.038) — clean version needs message-level capture (jobs 118607/118608) |
| InternVL2.5-8B baseline (Track B phase 1) | `internvl/baseline/` (job 118959) | CROSS-FAMILY FINGERPRINT REPLICATES, exaggerated: steps 0.124 (≈majority 0.132), slope 0.08, answers "2" for everything (g2:0.90, rest ~0); rooms 0.088 (majority 0.392!), slope 0.10, collapsed onto 2-3. Worse than Qwen (0.21/0.09) — consistent with MMReD's own ranking. Protocol identical to Qwen (single-forward digit argmax). n=250/task |
| InternVL2.5-8B carrier map (Track B phase 2) | `internvl/carrier_map/` (job 118962) | CROSS-FAMILY ANATOMY REPLICATES: peak carrier = the ROOM token (offset 13, "Bedroom"), d′=1.90 @L20/32 (~62% depth; Qwen: room token, 2.47 @L16/28 ≈57%). Reconstruction self-check cos=1.0000 (fused wqkv unpack exact). Law at d′=1.90, N=8 → probe ceiling ≈0.345; model emits 0.137 → emits ~40% of its carrier ceiling (Qwen: 0.21/0.47 ≈45%) — BOTH walls replicate; supply difference (1.9 vs 2.5) explains the family ranking. Caveat: sweep right-censored at offset 13 (peak at edge); MAXOFF=20 rerun pending. n=300 |
| InternVL2.5-8B parity + extended map (Track B phase 3) | `internvl/carrier_map_ext/`, `internvl/parity/20260707_202109` | LAW PARITY CONFIRMED CROSS-FAMILY: pred_iid 0.27–0.30 vs measured ridge 0.23–0.37 (mean 0.29≈0.29, 3 seeds, 2 layers); d′_w≈d′_auc (adequacy ✓); ρ +0.04–0.13, iid fits (Qwen pattern); naive 0.5–1.1 ≪ whitened 1.7–2.0. Anatomy uncensored: room token peak 1.90@L20, char token secondary 1.0@L12 — Qwen's two-site structure. Model 0.137 = 47% of probe ceiling (Qwen 45%). Quick-ridge 0.183 was estimator artifact — parity verdicts only from the engine (jobs 118968 + CPU) |
| InternVL E5 diagnosis + closures (Track B complete) | `internvl/{e5_diagnose,multipass_bench,native_axis_N4}/` | E5-null explained: scrub=wrong axis (state count ⊥ message-δ̂, state-decode unchanged 0.21-0.23), dose DID deliver (logit-lens 0.23→0.33 @L14+; earlier repaired-head was off-distribution — CORRECTION to "pipe closed" claim), pipe window L8-19 peak L13-16 (Qwen mirror). Multipass d′ 6.4-6.6 vs joint 1.9 → low d′ ≈ all interference (3.5×); perception acc 0.675 = readout wall at N=1. NATIVE CLOSURE: cos(native,w*)=0.02, d′_native=0.19 → law 0.135 vs observed 0.117-0.137 ✓ (jobs 118996/119000/119001) |
| InternVL routing thread (λ-ladder + 2× edge-cut) | `internvl/{dose_ladder,edge_cut,edge_cut_late}/` | dose delivers at λ≥32 (0.163→0.375 final decode, emissions flat = C6 on family #2; earlier null = under-dose + off-dist head); NO single edge necessary in either window (all cuts within 1 SE) → count content redundantly routed, unlike Qwen's concentrated evidence edges (jobs 119009-11) |
