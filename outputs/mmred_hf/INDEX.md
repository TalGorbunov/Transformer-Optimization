# outputs/mmred_hf — original-MMReD (HF ef1e43ce/mmred) campaign INDEX

> Experiment → canonical run → headline number. Campaign brief + running log:
> CAMPAIGN_BRIEF.md / STATE.md (untracked, same dir). Data: `data/mmred_hf/`
> (HF Arrow cache, prepped JSON, upstream repo @56c6ee70, rendered frames).

| experiment | canonical run | headline |
|-----------|----------------|----------|
| render (all 11 config×splits) | `render/` (jobs 127765–127775, 2026-08-01) | ~20 KB/frame, deterministic, capped-worker driver |
| frozen fidelity anchor (arm A, seq8 test, 34/qtype) | `frozen/seq_len_8_test_127776/` | final_app 0.765 / steps_in_room 0.559 / where_spend 0.353 → Phase-0 GO |
| arm A frozen grid (24 qtypes × full splits) | `frozen/grid_seq*/` (127803–09, 127822) | 0.533 / 0.422 / 0.361 / 0.304 / 0.247 @ N=8/16/32/64/128 (test) |
| Phase-3 trainer (5760 mix, caption scans) | `train/mix5760/20260801_205105_L12_r8/` (127821) | TF-answer 0.936 @ep3 (tf-exact 0.477) |
| arm B GIN ceiling (no-LoRA, steps_in_room, n=50/len) | `armB/noLora_seq*/` (127865–69) | 0.928/0.928/0.952/0.832/0.816 @ N=8–128 — length-flat (per-frame err ≤0.7%) |
| arm B ALL-TASKS GIN floor (linear heads, fit seq8/16) | `armB_grid/armB_grid_linear.csv` (128336–42, 128409) | 0.593/0.557/0.352/0.273/0.245 @N=8–128; stars: char_at_frame 0.98, where_spend 0.96, room_empty 0.98 @8 |
| decodability suites (24 qtypes) | `probe_suite/` (128335, 128359) | char→room strong, room→occupant weak; question-conditioning dependent |
| v5 dense-read trainer + smoke | `train/v5_dense/20260803_203521_L12_r8/` (128299) | TF 0.574; free-decode 8-niah 0.521 (v4 0.417) |
