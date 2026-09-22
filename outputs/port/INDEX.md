# outputs/port — the rewrite's acceptance gates (2026-09-21 plan, docs/REWRITE_EXECUTION_2026-09-21.md Part C)

Port checks run under a labelled legacy preset (`--port-check`) and are recorded here only; they are never cited.
Faithful runs are the new baselines.

| gate | experiment | canonical run | headline | status |
|---|---|---|---|---|
| B1 | prepare_data verify (seq_len_8_val, 50 qids) | — | JSON bytes / frame pixels vs 2026-08-01 tree | login-node smoke 2026-09-21 (2 qids): JSON bytes EQUAL, 16/16 frames pixel-identical, 0/16 PNG md5 (only the "Software" chunk: matplotlib 3.11.1 then, 3.11.2 now); 50-qid run pending |
| C1 port | evaluate --port-check arm-a, seq_len_8_test, 24 qtypes | — | target 640/1200 = 0.533 | not run |
| C1 port | evaluate --port-check arm-a, anchor3 × 34 | — | target 26/34, 19/34, 12/34 | not run |
| C1 faithful | evaluate --frozen, paper layout, seq_len_8_test | — | NEW frozen baseline | not run |
| C2 port | evaluate --port-check p2 + sft_fenced_hf_adapter, seq 8/16/32 | — | target 45/50, 33/50, 15/50 | not run |
| B3 | train (P2 recipe, faithful prompt, steps_in_room 8+16) | — | NEW fenced baseline row | not run |
| C3 | probe_hahn N = 8, 16 (band check) | — | plain α ∈ [0.7, 1.3], fenced \|α\| < 0.2, replay 0 | not run |
