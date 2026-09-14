# Bundle D — meta-llama/Llama-3.1-8B-Instruct L20 band 11–15 FL=[10, 11, 12, 13, 14, 15, 20, 24] (outputs/needles_pattern/Llama-3.1-8B-Instruct/20260903_141628_2574349)

## HC  (n=440, K=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48], sentences F=[64]…)
feature set                                         exact  ex≤16  within1 | fit≤16→test>16:  exact  rel.err med K̂@max
pattern: band layers, tail+answer, all stats         0.23   0.32     0.53 |                    0.02     0.35       27.0
pattern: band layers, k_eff only                     0.25   0.33     0.50 |                    0.01     0.64        5.0
pattern: band layers, peak counts only               0.15   0.20     0.41 |                    0.01     0.69        7.0
pattern: all FL layers, all stats                    0.28   0.35     0.60 |                    0.03     0.35       25.0
pattern: probe layer L only                          0.14   0.18     0.35 |                    0.01     0.53       41.0
ORACLE needle share, band layers                     0.29   0.37     0.58 |                    0.03     1.99      242.0
state: answer row at L                               0.26   0.33     0.63 |                    0.01     0.31       25.5
state: answer row final                              0.28   0.35     0.62 |                    0.01     0.44       15.0
D3 (layer,head) with k_eff slope∈[.6,1.1] & R²≥.95: 0/256; best L10H0 slope 0.04 R² 0.365; k_eff by K: 1:24.4 2:23.0 3:23.1 4:22.6 6:22.0 8:22.8 12:22.5 16:22.9 24:24.3 32:23.7 48:24.8
D3 single-head estimator (fit K≤16 → K>16): rel.err 0.80, within-1 0.00, median K̂ at K=48: 6.1

## HF  (n=440, K=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48], sentences F=[65, 66, 67]…)
feature set                                         exact  ex≤16  within1 | fit≤16→test>16:  exact  rel.err med K̂@max
pattern: band layers, tail+answer, all stats         0.37   0.48     0.67 |                    0.00     7.76     1027.0
pattern: band layers, k_eff only                     0.28   0.36     0.55 |                    0.04     0.89        0.0
pattern: band layers, peak counts only               0.28   0.36     0.53 |                    0.01     5.77      744.5
pattern: all FL layers, all stats                    0.40   0.52     0.70 |                    0.00     7.94     1033.0
pattern: probe layer L only                          0.22   0.30     0.47 |                    0.00    19.93     2575.5
ORACLE needle share, band layers                     0.31   0.40     0.55 |                    0.02     1.26      137.0
state: answer row at L                               0.35   0.43     0.71 |                    0.00     0.56       14.0
state: answer row final                              0.35   0.44     0.68 |                    0.00     0.59       12.0
D3 (layer,head) with k_eff slope∈[.6,1.1] & R²≥.95: 1/256; best L24H14 slope 0.62 R² 0.989; k_eff by K: 1:45.2 2:45.7 3:46.6 4:47.5 6:50.0 8:51.5 12:54.6 16:56.4 24:61.3 32:65.6 48:74.3
D3 single-head estimator (fit K≤16 → K>16): rel.err 0.33, within-1 0.01, median K̂ at K=48: 29.8
