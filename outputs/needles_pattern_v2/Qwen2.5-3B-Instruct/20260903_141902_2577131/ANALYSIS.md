# Bundle D — Qwen/Qwen2.5-3B-Instruct L27 band 18–24 FL=[13, 18, 19, 20, 21, 22, 23, 24, 27, 31] (outputs/needles_pattern_v2/Qwen2.5-3B-Instruct/20260903_141902_2577131)

## HC  (n=440, K=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48], sentences F=[64]…)
feature set                                         exact  ex≤16  within1 | fit≤16→test>16:  exact  rel.err med K̂@max
pattern: band layers, tail+answer, all stats         0.12   0.15     0.38 |                    0.05     0.43       55.5
pattern: band layers, k_eff only                     0.23   0.31     0.51 |                    0.01     0.60        5.0
pattern: band layers, peak counts only               0.07   0.09     0.25 |                    0.01     0.56       19.0
pattern: all FL layers, all stats                    0.18   0.23     0.45 |                    0.03     0.29       31.0
pattern: probe layer L only                          0.17   0.23     0.40 |                    0.03     0.46       13.5
ORACLE needle share, band layers                     0.25   0.33     0.54 |                    0.00    10.03      952.0
state: answer row at L                               0.24   0.30     0.55 |                    0.07     0.30       25.0
state: answer row final                              0.27   0.34     0.59 |                    0.02     0.44       14.0
D3 (layer,head) with k_eff slope∈[.6,1.1] & R²≥.95: 0/160; best L13H0 slope 0.03 R² 0.598; k_eff by K: 1:32.3 2:32.3 3:32.5 4:32.7 6:32.5 8:32.6 12:32.1 16:32.1 24:33.9 32:33.6 48:33.7
D3 single-head estimator (fit K≤16 → K>16): rel.err 0.80, within-1 0.00, median K̂ at K=48: 6.3

## HF  (n=440, K=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48], sentences F=[65, 66, 67]…)
feature set                                         exact  ex≤16  within1 | fit≤16→test>16:  exact  rel.err med K̂@max
pattern: band layers, tail+answer, all stats         0.32   0.43     0.64 |                    0.00    11.17     1433.5
pattern: band layers, k_eff only                     0.21   0.28     0.45 |                    0.01     0.79        3.0
pattern: band layers, peak counts only               0.19   0.26     0.44 |                    0.02     5.40      653.5
pattern: all FL layers, all stats                    0.40   0.52     0.69 |                    0.00     7.98     1016.5
pattern: probe layer L only                          0.22   0.30     0.52 |                    0.00    20.86     2556.5
ORACLE needle share, band layers                     0.29   0.39     0.52 |                    0.00     1.71      154.0
state: answer row at L                               0.27   0.34     0.65 |                    0.00     0.65       11.0
state: answer row final                              0.26   0.32     0.61 |                    0.00     0.54       14.0
D3 (layer,head) with k_eff slope∈[.6,1.1] & R²≥.95: 9/160; best L27H8 slope 1.00 R² 1.000; k_eff by K: 1:65.0 2:66.0 3:67.0 4:68.0 6:70.0 8:72.0 12:76.0 16:80.0 24:88.0 32:96.0 48:112.0
D3 single-head estimator (fit K≤16 → K>16): rel.err 0.00, within-1 1.00, median K̂ at K=48: 48.0
