# outputs/port — the rewrite's acceptance gates (2026-09-21 plan, docs/REWRITE_EXECUTION_2026-09-21.md Part C)

Port checks run under a labelled legacy preset (`--port-check`) and are recorded here only; they are never cited.
Faithful runs are the new baselines.

| gate | experiment | canonical run | headline | status |
|---|---|---|---|---|
| B1 | prepare_data verify (seq_len_8_val, 50 qids) | job 156677 (4h_0g, 29 s) | JSON bytes EQUAL; gold parity 1200/1200; **400/400 frames pixel-identical** (PNG md5 0/400: only the "Software" chunk, matplotlib 3.11.1 then / 3.11.2 now) | **PASS** 2026-09-22 |
| C1 port | evaluate --port-check arm-a, seq_len_8_test, 24 qtypes | `outputs/port/evaluate/seq_len_8_test/20260922_151901_156675/20260922_151917_pc-arm-a` (job 156675) | **637/1200 = 0.531 [0.502, 0.560]** vs 640/1200 (3 samples); per-qtype = legacy grid (char_at_frame 0.82, steps_in_room 0.56, final_app 0.74, where_spend 0.36, crowd_count 0.28) | **PASS** 2026-09-22 |
| C1 port | evaluate --port-check arm-a, anchor3 × 34 | — | target 26/34, 19/34, 12/34 | not run |
| C1 faithful | evaluate --frozen, paper layout, seq_len_8_test | `outputs/port/evaluate/seq_len_8_test/20260922_151900_156676/20260922_151917_faithful` (job 156676) | **618/1200 = 0.515 [0.487, 0.543]**, parse_fail 0 (strict upstream parser, native 512 px); per qtype within ±0.12 of the 392-px preset on 50-row cells (steps_in_room 0.56, char_at_frame 0.86, where_spend 0.36, crowd_count 0.24) | **DONE** 2026-09-22 — the frozen N=8 baseline row |
| C2 port | evaluate --port-check p2 + sft_fenced_hf_adapter, seq 8 (16/32 not run) | `outputs/scaleup/mmred/qwen2.5-vl-7b/s0_proof/c2_p2_N8/20260923_201335_pc-p2` (job 159297, through the 2026-09-23 data/backbone seams) | **44/50 = 0.880 [0.780, 0.960]** vs 45/50; one row moved (qid 0004694, gold 5: legacy 5 → 6), 49/50 identical (`port_check_diff.txt`) | **PASS** 2026-09-23 (1-row tolerance) |
| B3 | train (P2 recipe, faithful prompt, steps_in_room 8+16) | — | NEW fenced baseline row | not run |
| C3 | probe_hahn N = 8, 16 (band check) | — | plain α ∈ [0.7, 1.3], fenced \|α\| < 0.2, replay 0 | not run |
