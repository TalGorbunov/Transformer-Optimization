# ARCHIVE_MAP.md — where cited output paths actually live

> **2026-07-30 consolidation (serious-refactor):** ALL pre-refactor output trees now live
> under `outputs_legacy/`. Translation rules, applied on top of everything below:
> - `outputs/<anything>` (pre-refactor citations) → `outputs_legacy/outputs/<anything>`
> - `outputs_<codename>/…` and `output_*old/…` → `outputs_legacy/<same tree>/…`
> - The repo-root `outputs/` is fresh (post-refactor runs only; `outputs/carrier/…`).
> - Canonical checkpoints/caches keep their stable names under `checkpoints/` (symlinks
>   re-pointed to `outputs_legacy/…`).
> - Space cleanup (Tal-approved): the refuted qkv_2x2 raw captures (17G) and superseded
>   InternVL carrier_map caches (4.8G) were DELETED — each dir has a `CACHE_DELETED.txt`
>   breadcrumb; all reports/CSVs intact. The empty `outputs_cache/` was removed.
> - `outputs_olds/`, `outputs_long/` (the pre-refactor archive hops documented below)
>   are inside `outputs_legacy/` too: e.g. a Table-1 row "`outputs/X` → `outputs_olds/X`"
>   now resolves to `outputs_legacy/outputs_olds/X`.

**Purpose.** `RESULTS.md` is an append-only research log: once a row is written, its cited
`outputs/<name>` path is never rewritten. Meanwhile whole output trees were archived wholesale
(mostly into `outputs_long/` and `outputs_olds/`) without updating citations, so ~45 cited paths no
longer resolve. **This file is the translation layer** — look a dead citation up here instead of
editing `RESULTS.md`. (`STORY.md` cites no output paths; every citation below is from `RESULTS.md`.)

Method (2026-07-29): every `outputs*` path in `RESULTS.md`/`STORY.md` was extracted
(brace/glob citations expanded), existence-checked, and each dead path traced to its current
location by basename search across all archive trees; "Verified" says what was actually confirmed
present at the new location (cited run subdirs / files, not just the top dir).

---

## Table 1 — dead citation → real location

### Moved `outputs/<name>` → `outputs_olds/<name>` (23 dirs)

| Cited in RESULTS.md | Real location now | Verified |
|---|---|---|
| `outputs/agg_moredata` | `outputs_olds/agg_moredata` | yes — cited run `20260617_040354_md_rv_sum_ml_carrier_direct_sum` present |
| `outputs/agg_sweep` | `outputs_olds/agg_sweep` | yes — 13 `*rv*` run dirs present |
| `outputs/distractor_posneg_write_read_adapter_seq8_7b` | `outputs_olds/distractor_posneg_write_read_adapter_seq8_7b` | yes — all 14 cited variant subdirs present (`learned_posneg_*`, `detceiling_*`, `b2_countsup_*`, `oracle_pos*`) |
| `outputs/eval_mmred_cooccupancy_baseline` | `outputs_olds/eval_mmred_cooccupancy_baseline` | yes — cited `full/` present |
| `outputs/eval_mmred_rooms_visited_baseline` | `outputs_olds/eval_mmred_rooms_visited_baseline` | yes — cited `full/` present |
| `outputs/evidence_only_sum_adapter_train14_eval58_7b` | `outputs_olds/evidence_only_sum_adapter_train14_eval58_7b` | yes — cited runs `20260612_160218`, `20260612_161616` present |
| `outputs/evidence_only_sum_evidence_adapter_seq1_8_7b` | `outputs_olds/evidence_only_sum_evidence_adapter_seq1_8_7b` | yes — `20260613_141248_*` layer/op sweep, `20260613_153743_*`, seed and `dmem` runs all present (62 run dirs matched) |
| `outputs/final_glstm_aggregation_comparison` | `outputs_olds/final_glstm_aggregation_comparison` | yes — dir present |
| `outputs/frame_sigmoid_sum_attention_patch` | `outputs_olds/frame_sigmoid_sum_attention_patch` | yes — dir present |
| `outputs/layerwise_frame_message_glstm` | `outputs_olds/layerwise_frame_message_glstm` | yes — all 7 cited run families present (`20260612_175227_distractor_sum_*`, `..._175225_distractor_glstm_*`, `..._185319_sum_ctrl_*`, `..._175221_softmax_ctrl_*`, `20260613_142804_distractor_pna_carrier_pna`, slot/memonly runs) |
| `outputs/layerwise_fresh_query_aggregation_ablation` | `outputs_olds/layerwise_fresh_query_aggregation_ablation` | yes — dir present |
| `outputs/layerwise_glstm_train14_ood58_7b` | `outputs_olds/layerwise_glstm_train14_ood58_7b` | yes — cited `20260612_155614_*` runs present |
| `outputs/oracle_text_distinct_count` | `outputs_olds/oracle_text_distinct_count` | yes — both cited `*_result.txt` files present |
| `outputs/pna_carrier_mixing_lora` | `outputs_olds/pna_carrier_mixing_lora` | yes — dir present (also `_diagnostics` sibling) |
| `outputs/pnamix_clean_aggregation_lora` | `outputs_olds/pnamix_clean_aggregation_lora` | yes — dir present |
| `outputs/probe_aggregation_stages` | `outputs_olds/probe_aggregation_stages` | yes — cited `rooms_visited/`, `co_occupancy/` present |
| `outputs/probe_frame_to_carrier_message` | `outputs_olds/probe_frame_to_carrier_message` | yes — cited `rooms_visited/`, `co_occupancy/`, `count/` present |
| `outputs/probe_frame_token_states` | `outputs_olds/probe_frame_token_states` | yes — cited `rooms_visited/`, `co_occupancy/` present |
| `outputs/rooms_visited_adapter` | `outputs_olds/rooms_visited_adapter` | yes — cited `*rv7b_ev_vitlora_late16*` / `*rv7b_ev_vitlmlora*` runs present |
| `outputs/stages_7b_plots` | `outputs_olds/stages_7b_plots` | yes — cited `stages_restoration_by_group_7b_n40.png` present |
| `outputs/token_group_corruption_new_tasks` | `outputs_olds/token_group_corruption_new_tasks` | yes — cited `{count,rooms_visited,co_occupancy}_7b_n40` present |
| `outputs/visual_fixed8_count_sweep_lora` | `outputs_olds/visual_fixed8_count_sweep_lora` | yes — dir present (also `_label_control` sibling) |
| `outputs/visual_fixed8_iid_carrier_slots_lora` | `outputs_olds/visual_fixed8_iid_carrier_slots_lora` | yes — dir present |

### Moved `outputs/<name>` → `outputs_long/<name>` (17 dirs)

| Cited in RESULTS.md | Real location now | Verified |
|---|---|---|
| `outputs/eval_mmred_text_frames_acc` | `outputs_long/eval_mmred_text_frames_acc` | yes — dir present |
| `outputs/eval_mmred_text_frames_acc_cot` | `outputs_long/eval_mmred_text_frames_acc_cot` | yes — dir present |
| `outputs/eval_mmred_text_frames_acc_prec_nf4` | `outputs_long/eval_mmred_text_frames_acc_prec_nf4` | yes — dir present (also `_prec_bf16` sibling) |
| `outputs/frame_axis_aggregator_cached` | `outputs_long/frame_axis_aggregator_cached` | yes — dir present |
| `outputs/frame_axis_aggregator_cached_1to8` | `outputs_long/frame_axis_aggregator_cached_1to8` | yes — dir present |
| `outputs/frame_axis_cache/L19.pt` | `outputs_long/frame_axis_cache/L19.pt` | yes — cited file present |
| `outputs/frame_axis_live_attnpool` | `outputs_long/frame_axis_live_attnpool` | yes — dir present |
| `outputs/frame_axis_live_deepsets_eval` | `outputs_long/frame_axis_live_deepsets_eval` | yes — dir present |
| `outputs/frame_axis_live_h2h` | `outputs_long/frame_axis_live_h2h` | yes — all 4 cited variants present (`deepsets`, `deepsets_balanced`, `pna_balanced`, `pna_cb_balanced`) |
| `outputs/frame_axis_sweep` | `outputs_long/frame_axis_sweep` | yes — dir present (the cited "{6 configs}" sweep) |
| `outputs/probe_adapter_messages` | `outputs_long/probe_adapter_messages` | yes — dir present |
| `outputs/probe_evidence_selection_image` | `outputs_long/probe_evidence_selection_image` | yes — dir present |
| `outputs/probe_evidence_selection_linear` | `outputs_long/probe_evidence_selection_linear` | yes — dir present |
| `outputs/probe_multilayer_evidence` | `outputs_long/probe_multilayer_evidence` | yes — dir present |
| `outputs/probe_perception_binding` | `outputs_long/probe_perception_binding` | yes — dir present |
| `outputs/probe_pertask_extraction` | `outputs_long/probe_pertask_extraction` | yes — dir present |
| `outputs/probe_token_extraction` | `outputs_long/probe_token_extraction` | yes — dir present |

### Other dead citations (4)

| Cited in RESULTS.md | Real location now | Verified |
|---|---|---|
| `outputs_kitkat/outputs_best/mmred_nested_distractor_drift` | `outputs_kitkat/mmred_nested_distractor_drift` | yes — kitkat copy is the superset (runs `20260505_160524`…`20260508_001041` + cache); `outputs_best/mmred_nested_distractor_drift/20260508_001041` is a byte-identical copy of that run |
| `outputs_kitkat/outputs_best/mmred_nested_evidence_growth` | `outputs_kitkat/mmred_nested_evidence_growth` | yes — same situation: kitkat copy is the superset; `outputs_best/` holds an identical subset |
| `outputs_oreo/answer_aligned_count_codebook_memory_seq8_7b_20260528_000551` | `outputs_no_train/answer_aligned_count_codebook_memory_seq8_7b_20260528_000551` | yes — exact name+timestamp match, codebook diagnostics/plots inside, file mtimes 2026-05-28 (only dir of the oreo campaign that landed in `outputs_no_train/`) |
| `outputs_no_train/message_memory_carrier_update` | `outputs_no_train/message_memory_carrier_update_seq8_7b` | yes — shorthand for the full name (cited in full elsewhere in RESULTS.md); `20260529_*` runs inside match the row date |

### Citations that look dead but are only shorthand (nothing moved)

| Cited in RESULTS.md | Actual dirs (same tree) |
|---|---|
| `outputs/frame_axis/agg_min/lora_{rooms,steps,cooc}` | `outputs/frame_axis/agg_min/lora_{rooms_visited,steps_in_room,co_occupancy}` |
| `outputs/frame_axis/ood_holdout/{steps,rooms}_additive` | `outputs/frame_axis/ood_holdout/{steps_in_room,rooms_visited}_additive` |
| `outputs/frame_axis/ood_holdout/{steps,rooms,co_occupancy}_{deepsets,logic,lora}` | `outputs/frame_axis/ood_holdout/{steps_in_room,rooms_visited,co_occupancy}_{deepsets,logic,lora}` (all 9 present) |
| `outputs/readout/c2_digit_codebook/{20260710_220702,225300,233141}` | full timestamps: `20260710_{220702,225300,233141}` (all present) |
| `outputs_kitkat/mmred_{semantic_carrier_expansion,frame_summary_routing}` | present with suffix: `outputs_kitkat/mmred_{semantic_carrier_expansion,frame_summary_routing}_seq8_park_room` |
| `outputs/ladder/image_longN/{joint,fenced,multipass}/N{8..128}` | `N{8..128}` is prose for N ∈ {8,16,32,64,128} — those N-dirs exist (joint also has N1–N6) |

### Split trees — same name in `outputs/` AND `outputs_long/`, different runs

The archive move was partial for three dirs; a citation resolves in `outputs/` but some run
timestamps live only in the `outputs_long/` copy:

| Name | `outputs/` copy holds | `outputs_long/` copy holds |
|---|---|---|
| `diffmamba2_coocc` | runs `20260620_175958_*`, `20260620_181409_*` (the cited 2026-06-21 numbers) | runs `20260620_143728_*` |
| `eval_mmred_text_frames_acc_oracle` | runs `20260620_*` per task | runs `20260619_*` per task |
| `frame_axis_aggregator_adapter` | `cat1_*` runs | `20260619_193049_*` runs |

Note also: `evidence_only_sum_evidence_adapter_seq1_8_7b` exists in **both** `outputs_olds/`
(the `20260613_*` sweeps cited as `outputs/...`) and `outputs_no_train/` (different, earlier runs
`sum_evidence_{1,3}epoch` — cited as `outputs_no_train/...`). Map each citation to its own tree.

---

## Table 2 — archive-tree inventory

Era = min..max **file** mtimes inside the tree (file mtimes survive `mv`). Citations = occurrences
of `<tree>/` in RESULTS.md (path citations; 0 for trees only ever cited via their old `outputs/`
names).

| Tree | Era (file mtimes) | Content / campaign | Size | RESULTS.md citations |
|---|---|---|---|---|
| `output_old` | 2026-02-21..03-31 | Earliest era: `seq_len_{2..16}` (+`_all_frames`/`_evidence_only`), `LD_*` layer-depth patching, `find_af1_transition` — AF1-transition search | 37M | 1 |
| `output_less_old` | 2026-03-09..03-23 | `seq_len_{2..16}` sweeps + `transfer_bottleneck_scaling` | 53M | 1 |
| `outputs_less_less_old` | 2026-03-29..04-07 | AF1 campaign: `af1_runs`, `af1_frame_cama_cache`, `find_non_frame_prompt_*` rescues | 215M | 0 |
| `output_less_less_less_old` | 2026-04-08..04-09 | `seq_len_{2,4,8}/unified_bottleneck_analysis` + `mmred_new_accuracy_all_uniform` | 23M | 3 |
| `outputs_least_oldest` | 2026-04-10..04-30 | Accuracy mosaics/heatmaps, `cognn_oracle_routing*`, evidence-count ablation, image-size sweep, stage3/4 count probes, rep-collapse, wrong-routing decoys | 79M | 15 |
| `outputs_best` | 2026-02-25..05-08 | Curated "best of" early work: attention+probe, importance, frames_entropy, `oracle*`, `mmred_oracle_frame_hints`, `mmred_nested_{distractor_drift,evidence_growth}` (the two nested_* dirs are subsets of the fuller `outputs_kitkat/` copies) | 87M | 3 |
| `outputs_kitkat` | 2026-05-01..05-26 | 32B mechanistic campaign: attention-rollout heatmaps, additivity saturation, bottleneck-by-evidence-count, hierarchical slicing (`*_seq8_park_room`), oracle frame hints, soft-bias, first 7B `glstm_memory_adapter_7b_seq8` | 237M | 15 |
| `outputs_oh_man` | 2026-05-15..05-21 | Chefer relevance probes, frame→carrier message/evidence-sum probes, evidence→carrier routing, additive aggregation, `perm_bias_seq8`, `qwen7b_accuracy_heatmap` | 553M | 8 |
| `outputs_oreo` | 2026-05-21..05-29 | Count-direction / codebook / translator campaign: injection-site sweeps, `shared_count_direction*`, `translator_*`, `oracle_count_multilayer_injection` (its `answer_aligned_*` run sits in `outputs_no_train/`, see Table 1) | 52M | 13 |
| `outputs_no_train` | 2026-05-28..06-03 | Distractor oracle/supervised adapters, evidence-only adapters, gated token mixers, `message_memory_carrier_update_seq8_7b` | 67M | 14 |
| `outputs_olds` | 2026-06-05..06-17 | Biggest archive — early/mid-June adapter campaign: evidence_only + distractor_posneg write/read + layerwise gLSTM/frame-message, `agg_sweep`/`agg_moredata`, visual_fixed8 LoRA, pna/pnamix, probes (aggregation stages, frame→carrier), rooms_visited_adapter, unified_count, `stages_7b_plots`, `SPRINT_FINDINGS_DRAFT.md`. **Hosts 23 dirs still cited as `outputs/<name>`** (Table 1) | 3.7G | 0 (only via old names) |
| `outputs_long` | 2026-06-17..06-20 | Pre-reorg archive of mid-June frame-axis campaign: cached/live aggregators, h2h, sweep, `frame_axis_cache`, text-frames evals, diffmamba v1/v2, `probe_{adapter_messages,evidence_selection_*,multilayer,perception,pertask,token}`. **Hosts 17 dirs still cited as `outputs/<name>`** (Table 1) | 1.7G | 0 (only via old names) |
| `outputs_cache` | — (empty) | Empty directory, 0 files (created 2026-05-30) | 4.0K | 0 |

---

## Unresolved

None. Every dead cited path was located and content-verified (Table 1). Residual ambiguities are
documented above rather than unresolved: the three split trees (citation resolves but sibling runs
live in `outputs_long/`), the dual-location `evidence_only_sum_evidence_adapter_seq1_8_7b`, and the
one oreo-campaign run stored under `outputs_no_train/`.
