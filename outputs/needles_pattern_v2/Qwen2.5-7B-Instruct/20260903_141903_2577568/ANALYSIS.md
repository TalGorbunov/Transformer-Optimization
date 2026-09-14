# Bundle D — Qwen/Qwen2.5-7B-Instruct L22 band 15–18 FL=[11, 15, 16, 17, 18, 22, 26] (outputs/needles_pattern_v2/Qwen2.5-7B-Instruct/20260903_141903_2577568)

## HC  (n=440, K=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48], sentences F=[64]…)
feature set                                         exact  ex≤16  within1 | fit≤16→test>16:  exact  rel.err med K̂@max
pattern: band layers, tail+answer, all stats         0.18   0.22     0.42 |                    0.05     0.29       41.5
pattern: band layers, k_eff only                     0.20   0.26     0.47 |                    0.00     0.92        1.0
pattern: band layers, peak counts only               0.13   0.17     0.32 |                    0.01     0.52       14.0
pattern: all FL layers, all stats                    0.18   0.23     0.49 |                    0.01     0.37       22.0
pattern: probe layer L only                          0.15   0.19     0.36 |                    0.03     0.62        5.0
ORACLE needle share, band layers                     0.25   0.33     0.56 |                    0.00     6.85      750.0
state: answer row at L                               0.21   0.26     0.56 |                    0.03     0.31       24.0
state: answer row final                              0.22   0.25     0.54 |                    0.01     0.42       17.0
D3 (layer,head) with k_eff slope∈[.6,1.1] & R²≥.95: 0/196; best L11H0 slope -0.02 R² 0.386; k_eff by K: 1:22.0 2:22.1 3:21.6 4:21.4 6:21.0 8:22.3 12:20.5 16:21.4 24:21.4 32:20.8 48:20.7
D3 single-head estimator (fit K≤16 → K>16): rel.err 0.80, within-1 0.00, median K̂ at K=48: 6.6

## HF  (n=440, K=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48], sentences F=[65, 66, 67]…)
feature set                                         exact  ex≤16  within1 | fit≤16→test>16:  exact  rel.err med K̂@max
pattern: band layers, tail+answer, all stats         0.29   0.37     0.59 |                    0.00     6.14      769.5
pattern: band layers, k_eff only                     0.25   0.33     0.51 |                    0.00     0.93        3.0
pattern: band layers, peak counts only               0.17   0.22     0.45 |                    0.02     4.37      439.0
pattern: all FL layers, all stats                    0.35   0.47     0.65 |                    0.00     6.44      842.0
pattern: probe layer L only                          0.18   0.25     0.47 |                    0.00     8.29      877.0
ORACLE needle share, band layers                     0.29   0.37     0.53 |                    0.01     3.34      233.0
state: answer row at L                               0.29   0.38     0.62 |                    0.00     0.43       26.5
state: answer row final                              0.28   0.36     0.60 |                    0.00     0.51       18.0
D3 (layer,head) with k_eff slope∈[.6,1.1] & R²≥.95: 2/196; best L18H7 slope 1.00 R² 1.000; k_eff by K: 1:65.0 2:66.0 3:67.0 4:68.0 6:70.0 8:72.0 12:76.0 16:80.0 24:88.0 32:96.0 48:112.0
D3 single-head estimator (fit K≤16 → K>16): rel.err 0.00, within-1 1.00, median K̂ at K=48: 48.0
