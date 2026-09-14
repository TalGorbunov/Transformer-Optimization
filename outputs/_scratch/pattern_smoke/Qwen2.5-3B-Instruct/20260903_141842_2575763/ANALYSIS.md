# Bundle D — Qwen/Qwen2.5-3B-Instruct L27 band 18–24 FL=[13, 18, 19, 20, 21, 22, 23, 24, 27, 31] (outputs/_scratch/pattern_smoke/Qwen2.5-3B-Instruct/20260903_141842_2575763)

## HC  (n=9, K=[2, 8, 24], sentences F=[64]…)
feature set                                         exact  ex≤16  within1 | fit≤16→test>16:  exact  rel.err med K̂@max
pattern: band layers, tail+answer, all stats         0.11   0.17     0.22 |                    0.00     0.82        5.0
pattern: band layers, k_eff only                     0.22   0.33     0.33 |                    0.00     0.78        5.0
pattern: band layers, peak counts only               0.00   0.00     0.11 |                    0.00     0.81        5.0
pattern: all FL layers, all stats                    0.00   0.00     0.22 |                    0.00     0.79        6.0
pattern: probe layer L only                          0.11   0.17     0.56 |                    0.00     0.39       14.0
ORACLE needle share, band layers                     0.22   0.33     0.33 |                    0.00     8.91      317.0
state: answer row at L                               0.22   0.33     0.22 |                    0.00     0.44       15.0
state: answer row final                              0.22   0.33     0.33 |                    0.00     0.64        8.0
D3 (layer,head) with k_eff slope∈[.6,1.1] & R²≥.95: 0/160; best L22H5 slope 0.40 R² 0.971; k_eff by K: 2:39.3 8:40.2 24:47.8
D3 single-head estimator (fit K≤16 → K>16): rel.err 0.75, within-1 0.00, median K̂ at K=24: 6.1
