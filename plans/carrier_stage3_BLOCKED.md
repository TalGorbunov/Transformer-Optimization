# Stage-3 BLOCKED items (2026-07-19 session end)

## N=128 exam cells (thesis-table rows 9-N128 / 19)

- Jobs **124317** (A3 scratchpad ckpt, N=128 zero-shot, LIMIT=65, decode 220) and **124377**
  (A4 long-N ckpt, N=128 = 4× max trained length, same protocol) are QUEUED on `h200-shared`
  (24h_1g) and have been pending ~14 h: node n315's 8 H200s are held by `h200-dds`-partition
  jobs, which don't drain into the shared queue. My 2 jobs are the only ones in the
  h200-shared queue and will start unattended the moment GPUs free.
- N=128 cannot run elsewhere: attention scores at seq≈23k (28 heads, math backend, bf16)
  ≈ 30 GB + weights > a100-public's 40 GB. (a100-public is 40 GB, not 80 — known gotcha.)
- **Successor action:** when they finish, read `logs/cl_eval-124317.out` /
  `logs/cl_eval-124377.out` (reports also in
  `outputs/ladder/image_longN/scratchpad_eval_N128/` and `…/scratchpadLN_eval_N128/`),
  log RESULTS-style entries in `plans/carrier_stage2_DRAFT_RESULTS.md`, append INDEX rows,
  and fill FULL-THESIS TABLE rows 9 (N=128 half) and 19.
- Everything else in the Phase-3 program is complete (see the one-screen summary at the end
  of the draft and `plans/carrier_stage3_STATE.md`).
