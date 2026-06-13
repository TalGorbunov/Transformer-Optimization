# RESULTS.md — Research progress log

> **Purpose:** the single overall view of the thesis that no individual output dir gives me.
> Research memory that carries continuity across Claude Code sessions.
>
> **How to update (Claude, read this):**
> - Only update when I explicitly ask ("log this run", "update results").
> - Append to the Experiment Log table — never rewrite past rows.
> - Every row's metric must come from a real output dir, named in the row.
> - Put *why it matters* in the Synthesis section, not the table.
> - If a result is uncertain or suspicious, say so in the Notes — don't launder it into a clean number.

> **⚠️ Backfill provenance (2026-06-12):** The table below was reconstructed by scanning the existing
> `outputs_*/` / `output_*/` trees (summary.md / eval_metrics.csv / accuracy_by_*.csv / README.md per
> run). Numbers trace to the named dir but were **not all hand-verified** — spot-check any number
> before it goes in the thesis. Several headline accuracies are **trained-on-clean** or **oracle-masked
> upper bounds**, not deployable results; these are flagged ⚠️ / 📊 and explained in Synthesis.

---

## At a glance

- **Task / headline metric:** MMRED — *"how many frames was character C in room R?"*, answer ∈ {0..8}.
  Metric = **exact-match accuracy** of the predicted count (argmax over candidate count tokens),
  reported **overall** and **stratified by `evidence_count`** (the gold count). Secondary signals:
  `gold_margin` (gold-token logit/logprob minus best competitor), predicted-count **MAE**, and
  `fix_rate`/`break_rate` of an intervention vs the frozen base.
- **The bottleneck (well-established):** baseline Qwen2.5-VL accuracy collapses as the number of
  frames/evidence grows — ~85–100% at seq_len 1–2 down to ~19–30% at seq_len 8. The collapse is an
  **aggregation/over-squashing** failure: per-frame evidence is present and decodable, but the model
  cannot combine many frames in a single forward pass. Last-token representations and the gold margin
  collapse sharply once **2–4 frames** must be aggregated.
- **Current leading approach (updated Phase 5):** a frozen-Qwen **DeepSets readout** — `ρ(Σ φ(message_i))`:
  project each frame's attention "message", **unnormalized sum** over frames, inject at carrier/last-token
  residuals (~L14–17). gLSTM's associative memory-addressing is **dispensable** (sum ties/beats it on
  matched controls); the load-bearing ingredients are **unnormalized aggregation + width ≥ max-count**.
- **Status:**
  - ✅ **Evidence-only counting is solved** — sum / layer-local / raw-matrix / PNA adapters hit **100%**
    at seq_len 1–8 (vs ~39% base), incl. **100% count/length OOD** (train 1–4 → eval 5–8). Minimal form:
    a single layer, one shared φ/ρ, one inject at last token.
  - ✅ **Two causal knobs isolated (Phase 5):** unnormalized sum vs mean/softmax = +24pp OOD
    (normalization is the failure); d_mem sweep saturates at **width = max-count N** (capacity bound).
  - ❌ **Distractors are the open frontier** — learned adapters plateau ~40–64%; only oracle-masked
    upper bounds (96%) recover. The gap is **selection** (per-frame evidence detection), not aggregation;
    the per-frame gating interface is falsified, pointing to count-level stream supervision.

---

## Experiment Log

> Append-only. One row per experiment family (tight group of runs). Dirs are relative to repo root.
> Status: ✅ done & trusted · ⚠️ done but suspect/partial · ❌ failed/no-gain · 📊 characterization/probe only · ▶ running

### Phase 0 — Locating the bottleneck (Feb–Apr 2026, 32B/7B)

| Date | Output dir | Method / change | Key config | Metric | Status | Notes |
|------|-----------|-----------------|-----------|--------|--------|-------|
| 2026-03 | `output_old/seq_len_*/LD_*`, `find_af1_transition` | Layer-decomposition (CAMA/AF1-style) of per-layer info contribution | 32B/7B, seq 2/4/8/16 | — | 📊 | Info concentrates through mid layers (~10–27) into last token; ~0 contribution past L32. Bottleneck is seq-length-agnostic. |
| 2026-03 | `output_less_old/transfer_bottleneck_scaling/*` | Frame-token rescue/restoration curves | 32B, seq 2/4/8 | max_rescue ~0.99 (seq2 success) | ✅/⚠️ | >98% of signal is recoverable from the last token via late layers; failures show *misrouting*, not loss. |
| 2026-04-08 | `output_less_less_less_old/mmred_new_accuracy_all_uniform` | Baseline accuracy by seq_len | 7B, all_uniform, 900 samp | seq2 62.7% / seq4 42.3% / seq8 19.3% | 📊 | Canonical baseline degradation curve. |
| 2026-04-09 | `output_less_less_less_old/seq_len_*/unified_bottleneck_analysis` | Layer-wise clean-ablation + restoration | 7B, seq 2/4/8 | norm. damage peaks L10–27 | ✅ | Evidence frames carry more causal damage (~0.36) than question tokens (~0.23); recoverable → routing, not capacity. |
| 2026-04-11→16 | `outputs_least_oldest/mmred_accuracy_all_uniform`, `mmred_basic_acc_mosaic` | Baseline accuracy mosaics | 7B/3B, seq 2/4/8/9 | seq2 85% → seq8 ~29% | 📊 | Confirms steep seq-length sensitivity. |
| 2026-04-14 | `outputs_least_oldest/mmred_image_size_sweep` | Image-resolution sweep | seq 2/4/8, 66–504 px | seq8 11%→18% (66→504px) | ✅ | Resolution helps a little at low seq, negligibly at seq8 → vision detail is *not* the bottleneck. |
| 2026-04-18 | `outputs_least_oldest/mmred_frame_causal_separation_heatmaps` | Per-frame causal ablation | 32B, seq 2/4/8 | evidence drop 6.5(seq2)→0.64(seq8) @count1 | ✅ | Individual frame influence vanishes at seq8 → aggregation, not per-frame signal. |
| 2026-04-19 | `outputs_least_oldest/mmred_non_evidence_empty_context` | Remove non-evidence frames | 32B, seq 2/4/8 | ~neutral | ⚠️ | Removing distractors alone doesn't fix it — bottleneck is in aggregating evidence. |
| 2026-04-25 | `outputs_least_oldest/stage34_count_probe[_all]` | Linear/MLP probe of count by layer | 32B, seq 2/4/8, L0–63 | probe seq2 95.6% / seq8 40.7% (vs model 30.4%) | ✅ | **Key:** count info is decodable (probe ≫ model); failure is in the model *using* it. Info stabilizes after L40. |
| 2026-04-25 | `outputs_least_oldest/last_token_rep_collapse_evidence1` | Last-token representation collapse metrics | 32B, seq 1–8 | cosine-dist 0.060→0.019 (seq1→8) | ✅ | Last-token reps homogenize as seq grows — mechanistic signature of over-squashing. |
| 2026-04-25 | `outputs_least_oldest/mmred_evidence_index_attention`, `_sweep`, `evidence_count_seq_len_heatmap` | Attention to evidence vs non-evidence by layer | 32B | question-tokens > last-token evidence focus mid-layers | 📊 | Last token under-attends evidence; carrier (question) tokens hold more. Motivates "carrier" framing. |
| 2026-04-26 | `outputs_least_oldest/wrong_routing_decoys` | Semantic decoys (swapped char–room bindings) | 32B, seq8, L24/48 | decoy +29 pp swing | ✅/❌ | Model follows spurious char–room associations rather than robustly counting — a real failure mode. |
| 2026-04-29 | `outputs_least_oldest/stage34_nonlinear_probe_control` | Linear vs MLP/KNN probes | 32B | nonlinear +3–5pp early layers only | ⚠️ | Bottleneck isn't probe complexity; nonlinearity only matters in early layers. |
| 2026-04-29 | `outputs_least_oldest/carrier_capacity_probe` | Can carrier tokens encode count? | 32B, seq 1–8, slot prompts | probe≈model+~10pp; gap grows with seq | ✅/📊 | Carriers retain count info but the model's *use* degrades with length. |
| 2026-04-29→30 | `outputs_least_oldest/cognn_oracle_routing[_new*]` | Oracle: question attends ONLY to evidence keys | 32B, seq 2/4/8, T=30–42 | ~−2% to +2% vs base | ⚠️/❌ | **Perfect routing does NOT help** → bottleneck is upstream of routing (encoding/recognizing evidence). |
| 2026-04-29 | `outputs_least_oldest/selective_routing_oracle` | Keep only evidence frames (compact context) | 32B, seq 2/4/8 | seq8 28.7%→59.7% (compact) | ✅ | Shrinking the context to evidence frames recovers a lot → dilution/over-squashing from many frames. |
| 2026-05-08 | `outputs_least_oldest/evidence_count_ablation` | Frame influence + entropy by count | 32B, seq8 | total influence 0.48(c1)→1.38(c8) | ✅ | Each frame's signal is weak; aggregate grows but model can't sum many weak signals. |

### Phase 1 — Probing the frame→carrier→last pathway (May 2026, 7B)

| Date | Output dir | Method / change | Key config | Metric | Status | Notes |
|------|-----------|-----------------|-----------|--------|--------|-------|
| 2026-05-02→03 | `outputs_kitkat/last_question_text_mixed_counts_T30`, `mmred_frame_carrier_prompt`, `mmred_semantic_carrier_expansion*` | Prompt-level carrier interventions (frame labels, semantic expansion, late question text) | 32B, seq 2–8, T=30 | all ≤ base (e.g. seq8 28.1%, no gain) | ❌ | Prompt/text manipulation of the carrier doesn't help; late question-text injection biases toward predict-1. Bottleneck isn't carrier text capacity. |
| 2026-05-03 | `outputs_kitkat/entity_room_soft_bias_T30` | Soft visual attention bias to target room | 32B, seq 2–8 | marginal/negative | ⚠️ | Hard attention bias alone insufficient. |
| 2026-05-03 | `outputs_kitkat/mmred_additivity_saturation` | Are token-group reps additive vs saturating? | 32B, seq 4/8, L36–63 | additivity ratio ≈1.0, residual ≈0 | 📊 | Reps combine **additively, no saturation** → motivates additive (sum) aggregation design. |
| 2026-05-05 | `outputs_kitkat/mmred_oracle_frame_hints`, `outputs_best/oracle*` | Oracle/binary frame hints in prompt | 32B, seq 1–8 | seq8 27.8% → 79.2% (binary yes/no hint) | ✅(oracle) | If you tell the model per-frame yes/no, counting is near-solved → the bottleneck is **extracting per-frame evidence**, not final summation. |
| 2026-05-06 | `outputs_kitkat/mmred_bottleneck_by_evidence_count*` | Token-group causal importance by count | 32B, seq 1/2/4/8, L30–62 | — | 📊 | Evidence frames + last token most critical at high counts. |
| 2026-05-08 | `outputs_kitkat/outputs_best/mmred_nested_evidence_growth` | Add evidence frames k=1→16, track margin | 32B, L40/63 | acc 100%(k1)→57%(k2)→12%(k8); margin +6.7→−2.2 | 📊 | **Margin collapses by k=2–4** even for *genuine* evidence — the core failure curve. |
| 2026-05-08 | `outputs_kitkat/outputs_best/mmred_nested_distractor_drift` | Add distractor frames k=1→16 | 32B, L40/63 | acc 100%→25%(k8); margin +6.4→−1.0 | 📊 | Distractors collapse margin similarly fast; mid-layer representation drift (r≈0.5–0.7 @L40). |
| 2026-05-10 | `outputs_kitkat/mmred_hierarchical_slicing_seq8_park_room` | Decompose 8 frames into chunks, count, sum | 32B, seq8 | flat 28.1% → slice_2 75.6% → slice_1 97.8% → gold-text 100% | ✅ | **Strongest signal:** per-frame decomposition ≈ solves it. Bottleneck = simultaneous aggregation, not the count itself. |
| 2026-05-10→11 | `outputs_kitkat/mmred_frame_summary_routing_seq8_park_room` | Mean-pool each frame into first token | 32B, seq8, L32/40/48 | ≤ base; suppress variant 14.8% | ❌ | Hand-crafted frame summaries don't help (too lossy / slot saturated). |
| 2026-05-12 | `outputs_kitkat/attention_rollout_group_heatmaps_seq8*` | Attention-rollout maps by token group | 32B, seq8 | — | 📊 | Baseline attention-flow maps for interpreting interventions. |
| 2026-05-15→19 | `outputs_oh_man/chefer_*` (frame patch / question-token / second-hop / step-marker) | Chefer relevance attribution of frames→tokens | 7B, seq 2–8, L12–24 | evidence relevance peaks L14–17; question tokens 2–5% of total relevance | 📊/⚠️ | Evidence relevance is real but **suppressed** between vision and language; step markers help ~5–7pp. |
| 2026-05-15 | `outputs_oh_man/qwen7b_accuracy_heatmap` | 7B baseline accuracy grid | 7B, seq 2/4/8 × count 0–8 | degrades with seq & count | 📊 | The 7B reference surface the interventions target. |
| 2026-05-17 | `outputs_oh_man/perm_bias_seq8` | Frame-order permutation sensitivity | 7B, seq8 | count2 −3pp, count3 −11pp, count4 −7pp | ✅ | Order matters more at higher counts → recency/order-dependent (not order-invariant) aggregation. |
| 2026-05-18→19 | `outputs_oh_man/evidence_to_carrier_routing*`, `two_stage_question_answer_routing*`, `additive_evidence_aggregation*` | Attention-logit boosting of evidence→carrier→answer | 7B, seq8, L14–17/20–21, γ=1–4 | best ~7.3% (stage A+B); additive interventions 0% | ⚠️/❌ | Direct attention boosting yields only small gains; raw additive boosts fail → problem is post-routing, at answer generation. |
| 2026-05-21 | `outputs_oh_man/frame_to_carrier_evidence_sum_probe_seq8_7b_20260521_164621` | Learn per-frame binary evidence, sum to count | 7B, seq8, L14–17, d256, 40 ep | target char×room MLP **88.1%** (MAE 0.12) | ✅ | **Reused downstream as a "source run".** Evidence is linearly present and additively integrable at carriers. |
| 2026-05-21 | `outputs_oh_man/frame_to_carrier_message_memory_probe_*` (multilayer/linear/mlp) | Detect frame→carrier "message memory" by layer | 7B, seq8, L14–17 | actual-message MLP up to 57.8% count acc | ✅ | Message-memory > content-only > raw; MLP needed. Multilayer run reused as a source run. |
| 2026-05-21 | `outputs_oh_man/message_probe_sum_vs_joint_mlp_*`, `outputs_oreo/message_probe_base_vs_per_frame_*` | Per-frame **sum** vs joint MLP readout | 7B, seq8, counts 0–8 | per-frame sum **63.7%** vs joint MLP ~22%; binary per-frame **93%** | ✅ | **Aggregation is additive, not joint** — sum of per-frame signals wins decisively. Per-frame evidence ≈93% recoverable vs 16% base. |

### Phase 2 — Memory-adapter / count-direction line (late May 2026, 7B)

| Date | Output dir | Method / change | Key config | Metric (overall acc) | Status | Notes |
|------|-----------|-----------------|-----------|--------|--------|-------|
| 2026-05-26 | `outputs_kitkat/glstm_memory_adapter_7b_seq8/20260526_123006` | Additive **gLSTM memory adapter** on frozen Qwen (memory_q vs memory_rc) | 7B, seq8, L14–17, d_mem 64, lr 3e-4, 3 ep | base 25.2% → **memory_q 31.1%** (memory_rc 9.6%, breaks) | ⚠️ | First trainable adapter; memory_q helps at extreme counts (0→86.7%, 8→93.3%) but breaks mid-counts (break_rate 56%). |
| 2026-05-26 | `outputs_oreo/message_memory_adapter_stage1_stage3_seq8_7b_*` | Stage1 count head on messages → Stage3 inject to carriers | 7B, seq8, L14–21 | base 24.4% → Stage3 **45.9%** | ✅ | Per-frame message sum probe 68%; Stage3 injection roughly doubles base. |
| 2026-05-27 | `outputs_oreo/shared_count_direction_memory_seq8_7b_20260527_203756` | Learn a shared **count-direction** vector + small residual | 7B, seq8, L14–21 | base 45% → **54.8%** (shared+residual) | ✅ | **Stage3 checkpoint reused downstream.** Pure direction alone fails (11%); residual essential. Count corr 0.996. |
| 2026-05-27 | `outputs_oreo/shared_count_direction_calibrated_seq8_7b_*` | Add calibration (λ_count, λ_res) | 7B, seq8 | **48.9%** overall, mid-count 43.3% | ✅ | Calibration controls residual norm (48→9.6) and lifts mid-counts +10pp; corr preserved 0.948. |
| 2026-05-27 | `outputs_oreo/message_memory_count_channel_ablation_seq8_7b_*` | memory-only vs gate-sum-only vs combined readout | 7B, seq8 | gate-sum corr 0.70 (stage1); memory-only best stage3 mid | ⚠️ | Different channels win at different stages; combining doesn't help stage1. |
| 2026-05-28 | `outputs_oreo/memory_injection_site_sweep_seq8_7b_20260528_000508` | Sweep injection site × layer window (on the 54.8% ckpt) | 7B, seq8, L14–24 | best **56.3%** (all_question, L18–21 multi) | ✅ | Marginal but consistent: multi-layer L18–21 > single; all_question ≳ room_char; last-token-only caps at 45.9%. |
| 2026-05-28 | `outputs_oreo/memory_carrier_site_layer_norm_sweep_seq8_7b_*` | Raw α vs √|S| token-count normalization | 7B, seq8, L12–21 | room_char L16–19 **28.2%** | ⚠️ | After proper normalization the all_question advantage disappears → earlier gain was answer-steering, not cleaner aggregation. **Honesty flag.** |
| 2026-05-28 | `outputs_oreo/evidence_gate_to_codebook_seq8_7b_20260528_204057` | Sigmoid evidence gate → gated count codebook | 7B, seq8, L14–21 | gated best 26.7% vs **oracle codebook 48.9%**, base 24.4% | ❌ | Evidence detection AUC only 0.687 → learned gating can't beat oracle; detection is the weak link here. |
| 2026-05-28 | `outputs_oreo/answer_aligned_count_codebook_memory_seq8_7b_20260528_000551` (also in `outputs_no_train`) | Count-aware codebook aligned to answer tokens | 7B, seq8, L14–17, inject L18, 3 ep | base 24.4% → **60.7%** (Qwen-init, τ=2) | ⚠️ | Best learned distractor-task adapter so far; per-layer count score corr 0.94. Still ~35pp below oracle posneg. |
| 2026-05-28→29 | `outputs_oreo/oracle_count_multilayer_injection`, `oracle_count_injection_site_sweep_seq8_7b` | Inject **oracle gold count** into residuals | 7B, seq8, L14–24 | (ceiling probe) | 📊 | Designed to measure the upper bound of count injection. |
| 2026-05-29 | `outputs_oreo/translator_*` (codebook / layer-suffix / gold-count ablation) | Inject oracle count via codebook vs learned translator | 7B, seq8, L14–17…17–17 | static/energy-norm L14–17 **100%**; state-conditioned 51.9% | ✅(oracle) | Oracle count injection → 100% (model *can* use a clean count). Static/low-rank translators work; learned state-conditioned translators are fragile. Single late layer insufficient. |

### Phase 3 — Evidence-only solved, distractor frontier (late May–Jun 2026, 7B)

| Date | Output dir | Method / change | Key config | Metric | Status | Notes |
|------|-----------|-----------------|-----------|--------|--------|-------|
| 2026-05-29 | `outputs_no_train/evidence_only_layer_local_seq1_8_7b` | Layer-local frame aggregation, **all frames are evidence** | 7B, seq 1–8, L14–17, 1 ep | base 39.2% → **100%** (high counts 20%→100%) | ✅ | Pure-aggregation task is **solved**; MAE 1.26→0.00. |
| 2026-05-29 | `outputs_no_train/evidence_only_all_question_to_last_seq1_8_7b` | Raw-matrix (QK) readout, last token reads all-question messages | 7B, seq 1–8, L14–17 | base 39.2% → **100%** | ✅ | Content-addressed mixing also fully solves evidence-only. |
| 2026-05-30 | `outputs_no_train/evidence_only_sum_evidence_adapter_seq1_8_7b` (`experiments/...py`) | Pure additive sum of frame messages → last token | 7B, seq 1–8, L14–17, d256 | base 39.2% → **100%** | ✅ | No gates/queries needed when every frame is evidence — confirms additive hypothesis. |
| 2026-05-30 | `outputs_no_train/distractor_oracle_mask_sum_adapter_seq8_7b` | Oracle mask selects evidence, then sum (with distractors) | 7B, seq8, L14–17 | base 24.4% → **57.0%** | ⚠️📊 | **Surprise:** perfect selection + sum only reaches 57% → injection/representation compat is *also* limiting, not just selection. |
| 2026-05-30 | `outputs_no_train/distractor_oracle_posneg_write_read_adapter_seq8_7b` | Oracle pos/neg streams, write L14–17, **read L20–27** | 7B, seq8, 5 ep | base 24.4% → **96.3%** | ✅📊 | Upper bound: distractors carry essential *negative* signal; separate streams + delayed read nearly recover evidence-only. The 57→96 gap = stream separation + late read. |
| 2026-05-30 | `outputs_no_train/distractor_oracle_posneg_{sum,concat}_adapter_seq8_7b` | Oracle pos/neg sum vs concat variants | 7B, seq8 | (variants of above) | 📊 | Ablations of the pos/neg upper-bound mechanism. |
| 2026-05-30 | `outputs_no_train/distractor_supervised_gated_sum_adapter_seq8_7b` | Supervised (non-oracle) gated sum | 7B, seq8 | — | ⚠️ | Learned version of the oracle gate; the gap to 96% is the open problem. |
| 2026-05-29 | `outputs_no_train/message_memory_carrier_update_seq8_7b` | Layer-local vs cumulative-sum memory update (distractor task) | 7B, seq8, L14–17, d256 | base 24.4% → layer-local **46.7%**, cumulative 45.2% | ⚠️ | Helps, but gate-sum barely correlates with evidence count (0.04–0.16) → mixing is noisy. |
| 2026-06-02 | `outputs_no_train/gated_token_mixer_adapter` (`experiments/gated_token_mixer_adapter.py`) | MLP vs LoRA-attn vs gated token mixer (distractor) | 7B, seq8, L14–17, 3 ep | base 24.4% → LoRA-attn **52.6%**, GTM 51.1%, MLP 38.5% | ⚠️ | Learned mixers plateau ~50% on distractors (high-count ~48–53%); far from oracle 96%. |
| 2026-06-03 | `outputs_no_train/pna_gated_token_mixer_adapter`, `evidence_only_pna_gated_token_mixer_adapter` | PNA-style (multi-aggregator) gated mixers | 7B, seq8 | softmax 40%, sigmoid-gated-sum 42.2% | ⚠️ | PNA variants underperform LoRA-attn; aggregator choice matters but doesn't close the gap. |
| 2026-06-05 | `outputs/pna_carrier_mixing_lora[_diagnostics]`, `pnamix_layer_sweep_lora` | LoRA + carrier mixing (maxmix/pnamix), layer sweep | 7B, 12 LoRA layers | base→ maxmix high_count +5.5pp, long +6pp; best layers L14–17 | ✅/⚠️ | maxmix ≳ pnamix; **early-layer (14–17) injection is most effective**; alpha·v gating best on high counts. |
| 2026-06-06 | `outputs/pnamix_clean_aggregation_lora` | maxmix vs pnamix on **clean** data (no distractors) | 7B, LoRA 12 layers, 2400 train | maxmix high_count **97.3%**, long **95.5%** vs LoRA base 64.8%/77.8% | ✅ | On clean data maxmix nearly solves high-count & length OOD. (Clean-trained — not distractor-robust.) |
| 2026-06-06 | `outputs/visual_fixed8_iid_carrier_slots_lora` | Discrete carrier **slots** (k=1/4/8) | 7B, L14–17, fixed-8 | base 69.4% → **k8 82.8%** (high-count 52.5%→81.3%) | ✅ | Explicit slot memory beats continuous mixing; more slots → better high-count. |
| 2026-06-06 | `outputs/visual_fixed8_count_sweep_lora[_label_control]`, `visual_fixed8_iid_all_counts_lora` | LoRA/maxmix/pnamix under true count-OOD vs all-counts-IID | 7B, fixed-8 | OOD high-count ~0%; all-counts-IID plateau ~72% | ⚠️ | **Honesty flag:** true count-extrapolation (train 0–5, test 7) fails ~0% for all methods; even all-counts-IID ceilings ~72%. Count generalization is a separate hard problem from routing. |
| 2026-06-07 | `outputs/frame_sigmoid_sum_attention_patch` | Frame sigmoid-sum attention patch + LoRA | 7B, L14–17 | iid 37.8%→70.0% (patch+LoRA) | ✅/⚠️ | Sigmoid-sum aggregation helps IID; doesn't fix high-count OOD. |
| 2026-06-07→09 | `outputs/layerwise_frame_message_glstm`, `layerwise_glstm_mechanism_ablation` | gLSTM mechanism ablation: persistent vs fresh, layerwise vs final-only | 7B, L14–17, d_mem 64, 3.3M params | layerwise-persistent **98.1% IID / 88.1% length-OOD** vs direct-sum 94.5/73.9 | ✅ | Cross-layer **persistent state +18.7pp length-OOD**; layerwise injection +~1pp IID. ⚠️ verify what's frozen — "memory-disabled" drops to ~58%, so a lot rides on the adapter. |
| 2026-06-08 | `outputs/layerwise_fresh_query_aggregation_ablation`, `native_frame_factorized_attention` | Fresh-query / native-attention factorization | 7B, smoke | ~0% | ❌ | Generating fresh queries / factorizing native attention fails; explicit messaging or memory is needed. |
| 2026-06-09→10 | `outputs/final_glstm_aggregation_comparison` | **Head-to-head:** gLSTM (final-only) vs sum vs LoRA baseline | 7B, L14–17, lr 1e-4, 3 ep | gLSTM **70.5%** high-aggregation-extrap vs sum 46.0% / LoRA 48.5%; IID 98.9% | ✅ | **Current leading comparison:** gLSTM memory > sum > LoRA on the hard extrapolation split. |
| 2026-06-12 | `outputs/evidence_only_sum_adapter_train14_eval58_7b/{20260612_160218,20260612_161616}` | Sum adapter **OOD extrapolation**: train seq 1–4 only, eval seq 5–8 (new `--train-seq-lens` split; OOD test n=100/seq) | 7B, evidence-only, L14–17, d256, lr 1e-4, 1 vs 3 ep | 1 ep (val 96.7%): OOD 60–96%, undercounts; **3 ep (val 100%): OOD 100%** (460/460, mean pred = gold exactly) | ✅ | **Perfect length+count extrapolation** once converged — counts 5–8 never seen as labels. 1-ep degradation was undertraining, not an extrapolation limit. Diagnostics clean (frozen Qwen, exact messages, disjoint splits). Caveat: in evidence-only, count ≡ seq_len (axes confounded); jobs 93447/93452. |

### Phase 4 — Aggregator-ingredient decomposition + distractor-gap mechanism (2026-06-12→13, 7B)

> All single-seed. Three first-submission gLSTM jobs (93513–15) ran ~10 min with mangled `TRAIN_LENGTHS`
> (sbatch `--export` comma-splitting); cancelled, dirs marked `ABORTED.md`, resubmitted correctly. The
> new posneg script (`experiments/distractor/distractor_posneg_write_read_adapter_seq8_7b.py`) was
> recovered from an orphaned git blob (`2e75c2c`) and extended (gate mode/source/hardness, stream
> ablation, frozen/oracle-init readout, mask noise, gate-AUC diagnostics).

| Date | Output dir | Method / change | Key config | Metric | Status | Notes |
|------|-----------|-----------------|-----------|--------|--------|-------|
| 2026-06-12 | `outputs/layerwise_glstm_train14_ood58_7b/20260612_155614_carrier_glstm_layerwise`; `outputs/layerwise_frame_message_glstm/{20260612_185319_sum_ctrl_*,20260612_175221_softmax_ctrl_*}` | **Read-mechanism control** (matched harness/data): unnormalized associative (gLSTM) vs plain sum vs softmax-normalized read. Train seq 1–4 → length-OOD 5–8, neutral fillers | 7B, L14–17, d_mem 64, 3 ep | IID: gLSTM 1.000 / sum 0.982 / softmax 0.993. **len-OOD: gLSTM 0.778 / sum 0.762 / softmax 0.680** (len8 .583/.633/.517) | ✅ | **Normalization is the causal variable.** Two unnormalized reads tie (sum even wins at len8); softmax −9.8pp len-OOD, IID unchanged → predicted over-squashing signature. Associative q·k addressing adds nothing over sum. Caveat: degradation not collapse at short train lengths; memory-disabled 88.6/94.3/94.6%. |
| 2026-06-12 | `outputs/layerwise_frame_message_glstm/{20260612_175227_distractor_sum_carrier_direct_sum,20260612_175225_distractor_glstm_carrier_glstm_layerwise}` | **gLSTM vs sum read on distractor fillers** (new `--filler-kind distractor`). Train seq 4,6,8 → OOD 5,7,10 | 7B, L14–17, d_mem 64, 3 ep | sum: IID **63.8%** / len-OOD 60.0% / comp-OOD 59.5%. gLSTM: IID 61.7% / 58.6% / 57.6% | ✅ | Sum ties/beats associative read **with distractors too** → q·k addressing dispensable even under selection pressure. Sum's 63.8% IID > all prior learned distractor methods (codebook 60.7%, LoRA-attn 52.6%). U-shaped by-count (extremes easy, mid ~50%). Still ~32pp below oracle 96.3% → gap is signed-stream separation, not aggregation capacity. |
| 2026-06-12 | `outputs/distractor_posneg_write_read_adapter_seq8_7b/{oracle_pos_only_w14_17_r20_27_5ep,oracle_posneg_w14_17_r18_19_5ep,oracle_posneg_noise015_w14_17_r20_27_5ep}` | **Oracle posneg decomposition**: pos-only late-read; posneg early-read (L18–19); posneg readout trained with ε=0.15 mask-flip noise | 7B, seq8, base 24.4%, 5 ep | pos-only late **74.8%**; posneg early-read **91.1%**; noise-robust readout **61.5%** (eval w/ clean mask) | ✅📊 | Decomposes the 96.3% bound: **negative stream ≈ +20pp (dominant)**, read depth ≈ +5pp. Neg stream fixes low counts (cnt0 60→100, cnt1 7→93) = explicit absence signal. Noise training craters 96→62 even at clean eval → 15%/frame ≈ count-label noise (corrupts ~73% of 8-frame samples). |
| 2026-06-12→13 | `outputs/distractor_posneg_write_read_adapter_seq8_7b/{learned_posneg_w14_17_r20_27_5ep,...10ep,learned_posneg_w18_21_r22_27_5ep,learned_posneg_lategate_w14_17_r20_27_5ep,learned_posneg_hardgate_w14_17_r20_27_5ep}` | **Learned-gate plateau** (sigmoid gate + aux mask BCE/count loss): epochs, gate layer (14–17 vs 18–21), late-gate-on-early-messages, straight-through hard gate | 7B, seq8, base 24.4%, 5–10 ep | **39–47% regardless of gate AUC (0.61↔0.91)**: 46.7/47.4/39.3/43.0/46.7%; gate AUC 0.81/0.84/0.87/0.87/0.61 | ⚠️ | **Accuracy invariant to gate quality** → learned-vs-oracle gap is *not* detection. Per-sample: |gate-count-err| uncorrelated with |pred-err| (r=−0.18). 46.7% = old gateless adapter score → joint training collapses to ungated-aggregate optimum; gate is decorative. Detection/hardness/epochs/layer all falsified. |
| 2026-06-13 | `outputs/distractor_posneg_write_read_adapter_seq8_7b/{learned_posneg_frozenreadout_w14_17_r20_27_5ep,learned_posneg_frozenro_hardgate_noce_5ep}` | **Two-stage** (frozen oracle-trained 96.3% readout, train gate only): soft gate; hard gate + λ_ce=0 (pure detector → hard mask at interface) | 7B, seq8, base 24.4%, 5 ep, `--init-streams-from`/`--freeze-streams` | soft **12.6%**, hard-detector **16.3%** (both **below base**) | ❌ | Frozen exact-mask readout is brittle off the binary manifold: soft α≈0.5 halves streams → decodes "≈4"; hard mask compounds errors ≈ p⁸ (0.8⁸≈0.17). **Per-frame gating interface is the wrong abstraction** for closing 47→96 — needs ≥0.99/frame detection (unreached) or count-level stream supervision. |

### Phase 5 — DeepSets baseline: isolating the two causal knobs (2026-06-13, 7B)

> The aggregation fix reframed as **DeepSets** (`ρ(Σ φ(message_i))`). New flags on
> `experiments/evidence_only/evidence_only_sum_evidence_adapter_seq1_8_7b.py`: `--pool {sum,mean,softmax,pna}`,
> `--share-weights`; `CARRIER_PNA` read variant added to the gLSTM harness. Runner
> `runners/evidence_only_sum_ablation.sbatch`. All evidence-only, train seq 1–4 → eval OOD 5–8, **single-seed**.

| Date | Output dir | Method / change | Key config | Metric | Status | Notes |
|------|-----------|-----------------|-----------|--------|--------|-------|
| 2026-06-13 | `outputs/evidence_only_sum_evidence_adapter_seq1_8_7b/20260613_141248_{sum,mean,softmax,pna}_L14_17` | **Pooling ablation** (matched harness/data, only the aggregator changes): sum vs mean vs softmax vs PNA readout | 7B, L14–17, d256, 3 ep | **IID all 1.00**; OOD: sum **1.00** / pna **1.00** / mean **0.76** / softmax **0.76** (s5–8 mean 1.00/0.85/0.61/0.57) | ✅ | **Normalization is the causal variable.** IID hides it (all tie); count/length OOD exposes it. mean≈softmax exactly → it's the Σ=1 constraint, not the weighting. PNA=sum (its degree-scaler×mean reproduces sum). = GIN sum>mean inside a frozen VLM, as a causal ablation. The direct softmax-vs-sum baseline the project previously lacked. |
| 2026-06-13 | `outputs/evidence_only_sum_evidence_adapter_seq1_8_7b/20260613_141248_sum_{L14,L15,L16,L17,L14_15,L15_16,L16_17,L15_17,L14_17,L14_17_shared,L14_17_carriers}` | **Cleanest-baseline ablation**: layer window (singles/pairs/triples), shared vs per-layer φ/ρ, inject at last-token vs carriers | 7B, sum, d256, 3 ep | single L14 **1.00** / L16 **1.00** (L15 0.91, L17 0.86); any 2+ window 1.00; **shared-weights 1.00**; carriers = last-token 1.00 (all OOD) | ✅ | **One DeepSets block suffices**: a single mid-layer (L14 or L16), one shared φ/ρ, inject once at last token → 100% incl. OOD. Inject site irrelevant; weights shareable. Single-layer is mildly layer-dependent (L14/L16 perfect, L15/L17 weaker) — single-seed, don't over-read *which* layer. |
| 2026-06-13 | `outputs/evidence_only_sum_evidence_adapter_seq1_8_7b/2026*_dmem{1,2,4,8,16,64}_sum_L14_17_iid` | **Capacity (width) sweep**: vary d_mem on the sum readout, evidence-only **IID counts 0–8** (isolate width from extrapolation) | 7B, sum, L14–17, 3 ep | overall acc d1 **0.53** / d2 0.60 / d4 0.78 / **d8 0.99** / d16 1.00 / d64 1.00; high-count(6,7,8) 0.29/0.40/0.64/**1.00**/1.00/1.00 | ✅ | **Saturates exactly at d_mem=8 = max count N**; monotonic, failures concentrate at high counts. = DeepSets **width ≥ N** bound measured inside a VLM. Prescription: d_mem ≥ max expected count. (Refutes "scalar count needs no width" — width is a genuine second constraint.) |
| 2026-06-13 | `outputs/layerwise_frame_message_glstm/20260613_142804_distractor_pna_carrier_pna` | **PNA on distractors** (falsification: does aggregator richness close the distractor gap?) vs sum 63.8% / gLSTM 61.7% | 7B, distractor fillers, train 4,6,8 → OOD 5,7,10, 3 ep | (job 93819) | ▶ | Running. Expected ~63% (≈sum) → would confirm distractor gap is selection, not aggregation richness; clean PNA (no gated mixer) removes the confound from the earlier 40–42% PNA mixers. |

---

## Synthesis — what's working, what isn't

### What's working
- **The diagnosis is solid and convergent.** Probes (count decodable at ~95% seq2 / 40% seq8 while the
  model gets 30%), causal ablations (evidence-frame influence vanishes by seq8), last-token cosine
  collapse, and the nested-growth margin curves all say the same thing: **per-frame evidence is present
  and recoverable; the model cannot aggregate many frames in one forward pass.** This is an
  over-squashing / bottleneck story ([Alon & Yahav 2021](https://arxiv.org/abs/2006.05205)), which is
  exactly the GNN-message-passing framing the thesis wants.
  - **Direct prior for the last-token signature:** [Barbero et al. 2024, *Transformers need glasses!*](https://arxiv.org/abs/2406.04267)
    prove decoder-only transformers suffer **last-token representational collapse** — distinct input
    sequences map to arbitrarily close final-token representations — explicitly connect it to GNN
    over-squashing, and show it produces errors *specifically in counting and copying*, **worsened by
    low-precision floats**. Our Phase 0 last-token cosine-collapse metric (0.060→0.019, seq1→8) is the
    empirical signature of exactly this; their bf16 caveat is also a flag for our 4-bit/bf16 runs. This
    is the single closest transformer-side precedent — cite it as the mechanism behind the diagnosis.
- **Evidence-only counting is fully solved** by a small frozen-Qwen adapter that **sums per-frame
  attention messages** and injects them at carrier/question tokens (L14–17): 100% at seq 1–8 vs ~39%
  base, MAE→0. Sum, layer-local, and raw-matrix readouts all reach 100% — confirming the
  **additive-aggregation hypothesis** (sum of per-frame signals ≫ joint MLP, 63.7% vs ~22% on the probe).
- **And it extrapolates perfectly (2026-06-12):** trained on seq/count 1–4 only, the converged (3-ep)
  sum adapter scores **100% on seq/count 5–8** (n=100/seq) with exactly calibrated mean predictions —
  lengths, counts, and answer labels all unseen in training. The additive mechanism *computes* the
  count rather than memorizing the mapping (contrast: LoRA/maxmix get 0% count-OOD). This is the
  "simple solution that generalizes" for the evidence-only task; the open problems remain distractors
  and count-extrapolation on the distractor task.
- **gLSTM beats everything simpler on clean data.** Layerwise *persistent* gLSTM ≈ 98–99% IID and
  88–89% length-OOD, and in the head-to-head `final_glstm_aggregation_comparison` it leads on the hard
  high-aggregation-extrapolation split (70.5% vs sum 46% vs LoRA 48.5%). Cross-layer state persistence
  is the ingredient that buys length extrapolation (+18.7pp). This is the leading approach.
- **Injection geometry is understood:** write early (L14–17), read later (L18–27); multi-layer >
  single-layer; carrier/question tokens > last-token-only (which caps ~46%); discrete slots (k=8) beat
  continuous mixing. Oracle count injection → 100%, so **the frozen model can use a clean count signal**
  — the problem is producing one, not consuming it.

### Phase 4 (2026-06-13): which aggregator ingredients actually matter, and why distractors are hard
- **The aggregator decomposes into three ingredients; only two are load-bearing.** Matched-harness
  controls (same data/LoRA/training, only the read changes) isolate each: **(1) unnormalized read is
  necessary** — softmax-normalizing the memory read costs −9.8pp length-OOD (77.8→68.0) with IID
  unchanged, the predicted over-squashing signature; **(2) persistent cross-layer slot memory is
  necessary** — prior +18.7pp ablation; **(3) associative q·k addressing is *not* necessary** — a plain
  sum read ties or beats the gLSTM's associative read on clean (76.2 vs 77.8 len-OOD) *and* distractor
  (63.8 vs 61.7 IID) data. **Net: the bottleneck is relieved by a persistent, unnormalized, additive
  virtual-node memory — a sum aggregator. The gLSTM's ([arXiv:2510.08450](https://arxiv.org/abs/2510.08450))
  distinctive memory-addressing machinery is dispensable on MMRED.** (Earlier gLSTM > sum results came
  from clean/extrapolation splits with a *fresh*-query or final-only sum, not this matched persistent-sum control.)
- **A plain sum memory is the new best learned distractor method** (63.8% IID, distractor fillers) —
  above codebook (60.7%), LoRA-attn (52.6%), gated mixer (51%) — using no gating at all.
- **The 96.3% oracle bound is decomposed:** the **negative/absence stream is the dominant ingredient
  (~+20pp)**, late read only ~+5pp. The neg stream works by calibrating *low* counts (cnt0 60→100,
  cnt1 7→93): counting needs evidence for the count *and* for its complement (signed aggregation), not
  a filter-then-sum.
- **The learned-vs-oracle gap is an optimization problem, not a detection problem.** Across 5 learned-gate
  variants, task accuracy is pinned at 39–47% **independent of gate AUC (0.61↔0.91)**, training length,
  gate layer, and hardness; per-sample gate-count error is uncorrelated with prediction error (r=−0.18).
  46.7% is exactly the old gateless-adapter score → joint training reaches the ungated-aggregate optimum
  and the gate stays decorative. Detection, hardness, epochs, and gate placement are each falsified as
  the cause.

### Phase 5 (2026-06-13): the counting fix is DeepSets, with two measured causal knobs
- **The whole story collapses to one sentence:** the model cannot count because attention is a
  **normalized, bounded-width mean**, and the minimal fix is an **unnormalized sum (DeepSets,
  `ρ(Σ φ(message_i))`) with width ≥ max-count** — injectable as a *single-layer* adapter, not gLSTM.
  (Grounding: DeepSets [1703.06114], GIN sum>mean [1810.00826], width≥N [1901.09006]; see References.)
- **Knob 1 — operation (sum vs normalized), now a clean causal ablation.** Matched harness, only the
  pool changes: on IID everything ties at 100%, but on count/length-OOD the normalized pools collapse
  (mean 0.76, softmax 0.76, identical curves → it is the Σ=1 constraint) while sum and PNA hold 1.00.
  This is the direct softmax-vs-sum baseline the project previously lacked, and it is the
  [GIN](https://arxiv.org/abs/1810.00826) sum>mean theorem ([DeepSets](https://arxiv.org/abs/1703.06114),
  `ρ(Σ φ)`) instantiated inside a frozen VLM.
- **Knob 2 — capacity (width), measured.** Sweeping d_mem on the sum readout (IID, counts 0–8) gives a
  monotonic curve that **saturates exactly at d_mem = 8 = max count N** (0.53→0.60→0.78→**0.99**→1.00→1.00),
  with failures concentrated at high counts. This is the [Wagstaff et al. 2019](https://arxiv.org/abs/1901.09006)
  set-representation **width ≥ N** bound measured directly (and aligns with [Di Giovanni et al. 2023](https://arxiv.org/abs/2302.02941),
  where width mitigates over-squashing); it refutes the tempting "a scalar count needs no width" intuition.
  Prescription: d_mem ≥ max expected count.
- **The deployable baseline is tiny.** A single mid-layer, one shared φ/ρ, injected once at the last token
  → 100% incl. count/length OOD. Inject site is irrelevant (carriers = last-token); weights are shareable.
- **PNA = sum (safe task-agnostic default).** On evidence-only [PNA](https://arxiv.org/abs/2004.05718)
  exactly matches sum (it contains sum via its degree-scaler×mean), so it is a no-regression generalization
  for tasks whose right aggregator is unknown.

### What isn't / dead ends
- **Distractors are the unsolved frontier.** With non-evidence frames present, learned adapters plateau:
  message-memory 46.7%, gated token mixer 51%, LoRA-attn 52.6%, answer-aligned codebook 60.7%. The
  **oracle pos/neg write-read upper bound is 96.3%**, and even oracle *selection*-then-sum is only 57% —
  so the gap is partly suppression *and* partly that distractors carry **negative evidence** that must be
  routed through a separate stream and read at later layers. Closing 60→96 without oracle masks is the
  next milestone.
- **Prompt / attention-bias / hand-crafted-summary interventions all failed** (frame labels, semantic
  carrier expansion, soft room bias, mean-pool frame summaries, fresh-query aggregation, native-attention
  factorization) — ≤ base. The carrier bottleneck is **not** about question-text capacity.
- **Oracle attention routing did not help** (cognn_oracle_routing, ~±2%): perfect routing to evidence
  keys isn't enough, confirming the bottleneck is upstream of routing.
- **Count extrapolation is a distinct, unsolved problem.** Train on counts 0–5, test on 7 → ~0% for every
  method; even all-counts-IID ceilings ~72%. Don't conflate "aggregation" wins with "count generalization".
- **The per-frame gating interface is the wrong abstraction for closing the distractor gap** (Phase 4).
  Every route through it fails for an understood reason: soft gates get bypassed by joint training (47%);
  a learned gate feeding a frozen exact-mask readout compounds errors ≈ pⁿ (12–16%, below base); and
  training the readout to tolerate mask noise destroys the count signal (96→62 even at clean eval, since
  15%/frame ≈ count-label noise over 8 frames). Closing 47→96 needs either ≥0.99/frame detection
  (unreached; AUC plateaus ~0.87–0.91) or **count-level / distributed stream supervision** rather than
  per-frame masks.

### Honesty flags (verify before thesis)
- Several **near-100% numbers are trained-on-clean or oracle-masked** (`pnamix_clean` 97%,
  `layerwise_glstm` 98–99%, all `evidence_only` 100%, all `*oracle*` rows). They are upper bounds / clean
  conditions, **not** distractor-robust deployable results.
- For the gLSTM "full-model" variants (`layerwise_frame_message_glstm`), confirm exactly which params are
  trainable — "memory-disabled" collapses to ~58%, so the adapter is doing heavy lifting; check for any
  train/eval leakage on the synthetic splits.
- `memory_carrier_site_layer_norm_sweep` showed the earlier all-question injection edge **disappears
  after proper √|S| normalization** → some prior "gains" may have been answer-steering, not cleaner
  aggregation. Re-audit injection-site claims with normalization held fixed.
- **All Phase 4 numbers are single-seed.** The matched read-mechanism control (sum 76.2 vs gLSTM 77.8
  len-OOD) is a tie *within seed noise* — the "associative read is dispensable" claim should be confirmed
  with ≥3 seeds before it goes in the thesis. The softmax −9.8pp effect is larger and more likely robust,
  but also single-seed and at short train lengths (degradation, not collapse). Phase 4 gLSTM runs still
  carry the unresolved "memory-disabled ~88–95%" flag (LoRA does much of the work at short lengths).
- **All Phase 5 numbers are single-seed too.** The two headline curves are clean (pooling: sum/pna 1.00 vs
  mean/softmax 0.76, a 24pp gap; width: monotonic, knee exactly at d_mem=8=N) so seed-robustness risk is
  lower than a tie — but re-run ≥3 seeds before the thesis, especially the single-layer L14/L16=1.00 vs
  L15/L17 0.86–0.91 split (could be seed noise, don't claim a *specific* layer). Phase 5 uses
  `--load-in-4bit` default per the runner; confirm consistent with other rows.

### Open questions / next experiments
1. ~~**gLSTM on the distractor task**~~ — **done (Phase 4):** gLSTM ties/loses to plain sum on distractors
   (61.7 vs 63.8 IID); associative read is dispensable.
2. ~~**Why oracle-selection-then-sum is only 57%**~~ — **answered (Phase 4):** it's the missing negative
   stream (≈+20pp) plus read depth (≈+5pp), not an injection/representation incompatibility.
3. **Count-level / distributed stream supervision** for distractors — the per-frame gating interface is
   falsified (Phase 4); supervise the *summed* pos/neg streams or the count directly instead of per-frame
   masks. This is now the headline open problem for closing 47→96.
4. **Re-run the matched read-mechanism control with ≥3 seeds** to firm up sum-vs-associative-read (and
   ideally extend to longer train-length ranges where the softmax/normalization gap should widen).
5. **Decouple count generalization** from aggregation (the 0% count-OOD result) — possibly a count-direction
   /codebook that extrapolates beyond trained counts.

### Interesting insights & surprises
- **Hierarchical slicing → ~98–100%**: just processing frames in small chunks and summing nearly solves
  the task, the single most striking "the bottleneck is simultaneity" demonstration.
- **Binary per-frame oracle hint → 79% at seq8** (from 28%): the model can count if told per-frame yes/no.
- **Additive ≫ joint**: the model's evidence integration is genuinely a *sum*, which is why
  sum/PNA-mean style readouts and the gLSTM additive memory work and joint-MLP readouts don't.
- **Margin collapses at k=2–4 frames regardless of frame type** — even adding *true* evidence past ~2
  frames hurts the margin, which is the clean over-squashing signature.

---

## Method comparison

| Method / family | Best metric | Setting | Verdict | Evidence (dirs) |
|-----------------|-------------|---------|---------|-----------------|
| Frozen baseline | seq8 ~19–30% | as-is | reference | `output_less_less_less_old/mmred_new_accuracy_all_uniform`, `outputs_oh_man/qwen7b_accuracy_heatmap` |
| Prompt / semantic carrier / soft-bias | ≤ base | 32B | ❌ dead end | `outputs_kitkat/mmred_{frame_carrier_prompt,semantic_carrier_expansion,frame_summary_routing}`, `entity_room_soft_bias` |
| Oracle attention routing | ±2% | 32B | ❌ (routing not the bottleneck) | `outputs_least_oldest/cognn_oracle_routing*` |
| Compact context / hierarchical slicing | seq8 → 98–100% | oracle-ish | ✅ (proves bottleneck) | `outputs_kitkat/mmred_hierarchical_slicing*`, `outputs_least_oldest/selective_routing_oracle` |
| Oracle count / per-frame hints | 79–100% | oracle | ✅ upper bound | `outputs_kitkat/mmred_oracle_frame_hints`, `outputs_oreo/translator_*`, `oracle_count_*` |
| LoRA + carrier mixing (maxmix/slots) | clean high-count 82–97% | trained-clean | ✅ (clean only) | `outputs/pnamix_clean_aggregation_lora`, `visual_fixed8_iid_carrier_slots_lora` |
| Memory adapter / count-direction | 45–61% | distractor task | ⚠️ partial | `outputs_oreo/{shared_count_direction*,answer_aligned*}`, `outputs_no_train/message_memory_carrier_update` |
| Learned token mixers (GTM/PNA/LoRA-attn) | 50–53% | distractor task | ⚠️ partial | `outputs_no_train/{gated_token_mixer_adapter,pna_gated_token_mixer_adapter}` |
| **Evidence-only sum/layer-local/raw-matrix** | **100%** (incl. **100% OOD**: train 1–4 → eval 5–8) | evidence-only | ✅ solved + extrapolates | `outputs_no_train/evidence_only_*`, `outputs/evidence_only_sum_adapter_train14_eval58_7b` |
| **gLSTM memory adapter** | IID 98–99% / length-OOD 88% / hi-extrap 70.5% | clean splits | ✅ **leading** | `outputs/{final_glstm_aggregation_comparison,layerwise_*glstm*}`, `outputs_kitkat/glstm_memory_adapter_7b_seq8` |
| Oracle pos/neg write-read (distractor) | **96.3%** | oracle mask | 📊 target upper bound | `outputs_no_train/distractor_oracle_posneg_write_read_adapter_seq8_7b` |
| Read-mechanism control: sum vs associative vs softmax (matched) | len-OOD: sum 76.2 / assoc 77.8 / **softmax 68.0** | neutral, train 1–4→5–8 | ✅ unnormalized necessary; assoc dispensable (single-seed) | `outputs/layerwise_glstm_train14_ood58_7b/20260612_155614_*`, `outputs/layerwise_frame_message_glstm/{20260612_185319_sum_ctrl_*,20260612_175221_softmax_ctrl_*}` |
| Sum vs gLSTM memory on **distractors** | sum **63.8%** IID ≥ gLSTM 61.7% | distractor task | ✅ sum = new best learned distractor method | `outputs/layerwise_frame_message_glstm/{20260612_175227_distractor_sum_*,20260612_175225_distractor_glstm_*}` |
| Learned posneg gate (all variants) | 39–47% (invariant to gate AUC) | distractor task | ⚠️ gap is optimization, not detection | `outputs/distractor_posneg_write_read_adapter_seq8_7b/learned_posneg_*` |
| Two-stage (frozen oracle readout + learned gate) | 12–16% (below base) | distractor task | ❌ per-frame interface compounds errors ≈pⁿ | `outputs/distractor_posneg_write_read_adapter_seq8_7b/learned_posneg_frozen*` |
| **Pooling ablation: sum/pna vs mean/softmax** | OOD: sum/pna **1.00** vs mean/softmax 0.76 | evidence-only, matched | ✅ normalization is the causal failure | `outputs/evidence_only_sum_evidence_adapter_seq1_8_7b/20260613_141248_{sum,mean,softmax,pna}_L14_17` |
| **DeepSets width (d_mem) sweep** | saturates at **d_mem=8=N**; d2 0.60 / d4 0.78 / d8 0.99 | evidence-only IID 0–8 | ✅ width≥max-count bound measured | `outputs/evidence_only_sum_evidence_adapter_seq1_8_7b/2026*_dmem*_sum_L14_17_iid` |
| **Minimal DeepSets baseline** (1 layer / shared φ,ρ / last-token) | **100%** IID + OOD | evidence-only | ✅ cleanest deployable fix | `outputs/evidence_only_sum_evidence_adapter_seq1_8_7b/20260613_141248_sum_{L14,L16,L14_17_shared}` |

---

## References (GNN / set-aggregation grounding)

> External literature this work builds on. Relevance noted per entry; arXiv IDs verified 2026-06-13.

- **DeepSets** — Zaheer et al., *Deep Sets*, NeurIPS 2017. [arXiv:1703.06114](https://arxiv.org/abs/1703.06114).
  The `ρ(Σ φ(x_i))` form our sum adapter instantiates; sum is the universal permutation-invariant pool.
- **Width ≥ set-size bound** — Wagstaff et al., *On the Limitations of Representing Functions on Sets*,
  ICML 2019. [arXiv:1901.09006](https://arxiv.org/abs/1901.09006). Proves continuous set-function
  representation needs latent dim ≥ max #elements — **exactly our d_mem-saturates-at-N=8 result**.
- **GIN (sum > mean > max)** — Xu et al., *How Powerful are Graph Neural Networks?*, ICLR 2019.
  [arXiv:1810.00826](https://arxiv.org/abs/1810.00826). Counting/multiplicity is the canonical task
  separating sum from mean — our pooling ablation (sum/pna 1.00 vs mean/softmax 0.76) inside a VLM.
- **PNA** — Corso et al., *Principal Neighbourhood Aggregation for Graph Nets*, NeurIPS 2020.
  [arXiv:2004.05718](https://arxiv.org/abs/2004.05718). Multiple aggregators + degree scalers; its
  degree-scaler×mean reproduces sum, so PNA = sum on our task (safe task-agnostic readout).
- **Over-squashing: width/depth/topology** — Di Giovanni et al., ICML 2023.
  [arXiv:2302.02941](https://arxiv.org/abs/2302.02941). Width mitigates over-squashing (matches our
  capacity knob); depth does not. Frames the two-knob (operation + width) story.
- **Over-squashing (origin)** — Alon & Yahav, *On the Bottleneck of GNNs…*, ICLR 2021.
  [arXiv:2006.05205](https://arxiv.org/abs/2006.05205). Defines the bottleneck the thesis ports to VLM attention.
- **Transformer last-token collapse** — Barbero et al., *Transformers need glasses! Information
  over-squashing in language tasks*, NeurIPS 2024. [arXiv:2406.04267](https://arxiv.org/abs/2406.04267).
  Proves decoder last-token **representational collapse**, ties it to GNN over-squashing, and shows it
  causes failures *specifically in counting/copying* (exacerbated by low-precision FP) — the theory behind
  our Phase 0 last-token cosine-collapse diagnostic and the bf16 caveat.
- **gLSTM** — *Mitigating Over-Squashing by Increasing Storage Capacity*, 2025.
  [arXiv:2510.08450](https://arxiv.org/abs/2510.08450). The capacity/associative-memory approach our
  matched controls show is over-engineered for MMRED (its addressing is dispensable; sum suffices).

## Backfill checklist

- [x] Infer common run structure (config.json / summary.* / eval_metrics.csv / accuracy_by_*.csv / README.md).
- [x] Define THE metric (exact-match count accuracy, by evidence_count; + gold_margin, MAE, fix/break).
- [x] Backfill the Experiment Log from existing dirs (oldest → newest).
- [x] Write first-pass Synthesis.
- [ ] **Spot-check the flagged numbers** (trained-on-clean / oracle / normalization) before any go in the thesis.
