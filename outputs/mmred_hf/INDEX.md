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
