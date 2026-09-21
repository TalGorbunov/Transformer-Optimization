# outputs/redux/ — INDEX (experiment → canonical run → headline)

Campaign: REDUX (brief `CAMPAIGN_BRIEF.md` + §7 addendum; log `STATE.md`).
Status at last update: C0+C1+C4b complete; CHECKPOINT — awaiting Tal before C2.

| experiment | canonical run | headline |
|---|---|---|
| C0 pools | `data/mmred_redux/seq_len_{16..128}/` (jobs 138933–36) | 550–1000 exam + 80 probe-base dirs/N, engineered K |
| C0 exam cells | `examdirs/` + REPORT.txt | 15 cells, yes/no cells balanced (maj 0.500), 0 bad; rows shuffled seed 4 |
| C1 frozen plain ladder | `c1_plain_{short,long}/` (138954/55) | exists 0.833→0.490 (chance @128, k=1 recall 0/16); majority ≈chance ∀N; count 0.200@8 ✓anchor |
| C1 fenced-frozen yes/no | `c1_p1fence_frozen_{short,long}/` (138956/57) | DEGENERATE always-no ∀N (prediction tables) |
| C4b (i) α under logN | `c4b_n/p1fence_ep10_logn_N*/` (138968) | α 0.605≈0.603(off): **H-SHARP(a) REFUTED**; margins −2.89→−0.10 @64 |
| C4b (ii) decay-in-k | `c4b_k/N64_k*_logn*/` (138969) | γ=0.390 [0.375,0.402], C=0; logN-invariant (descriptive) |
| C2 P1m trainer | — | GATED on review |
| v3 count-only N-in-text pair (mis-split submit) | `c2b_gated/20260920_232350_gated/`, `c2a_ungated/20260921_001236_gated/` (153961/62) | count N8/16/32: gated 0.993/0.887/0.780, ungated 1.000/0.893/0.567; zero-shot majority: ungated 0.98/0.97/0.93 (boundary-only errors), gated chance |
| v3 count-only N-free controls (480 samples) | `c2b_countctrl/`, `c2a_countctrl/` (153963/64) | gated 0.980/0.913/0.767 with k≤8 = 1.000 at N ≤ 32; ungated 0.973/0.713/0.267 with k≤8 0.65@16, 0.38@32 |
| v3 multitask readers (count/exists/majority, 243 each, N-free) | `c2b_gated_mt/20260921_024331_gated/`, `c2a_ungated_mt/20260921_024331_gated/` (154067/68) | exists 1.000 both; majority ungated 0.973/0.868/0.780 (matched-ratio 0.96/0.94/0.88), gated 0.960/0.861/0.373 (H-R6 MET @32); count UNDER-TRAINED (gated k+1 for k=4..7 at N=8) — count bands scored on the controls |
| v3 multitask N=64/128 eval-only | `c2b_gated_mt_eval64_128/`, `c2a_ungated_mt_eval64_128/` (154212/13, running) | decides H-R2 (exists decay), H-R3 (majority flat at matched ratio), H-R5 |
