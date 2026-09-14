# Bundle D — meta-llama/Llama-3.1-8B-Instruct L20 band 11–15 FL=[10, 11, 12, 13, 14, 15, 20, 24] (outputs/needles_pattern_v2/Llama-3.1-8B-Instruct/20260903_141904_2577565)

## HC  (n=440, K=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48], sentences F=[64]…)
feature set                                         exact  ex≤16  within1 | fit≤16→test>16:  exact  rel.err med K̂@max
pattern: band layers, tail+answer, all stats         0.23   0.30     0.53 |                    0.00     0.35       27.0
pattern: band layers, k_eff only                     0.25   0.33     0.52 |                    0.01     0.73        3.5
pattern: band layers, peak counts only               0.15   0.20     0.41 |                    0.01     0.69        7.0
pattern: all FL layers, all stats                    0.27   0.34     0.61 |                    0.05     0.34       25.0
pattern: probe layer L only                          0.15   0.20     0.36 |                    0.03     0.73       64.0
ORACLE needle share, band layers                     0.29   0.37     0.58 |                    0.03     1.99      242.0
state: answer row at L                               0.26   0.33     0.63 |                    0.01     0.31       25.5
state: answer row final                              0.28   0.35     0.62 |                    0.01     0.44       15.0
D3 (layer,head) with k_eff slope∈[.6,1.1] & R²≥.95: 0/256; best L10H0 slope 0.04 R² 0.365; k_eff by K: 1:24.4 2:23.0 3:23.1 4:22.6 6:22.0 8:22.8 12:22.5 16:22.9 24:24.3 32:23.7 48:24.8
D3 single-head estimator (fit K≤16 → K>16): rel.err 0.80, within-1 0.00, median K̂ at K=48: 6.1

## HF  (n=440, K=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48], sentences F=[65, 66, 67]…)
feature set                                         exact  ex≤16  within1 | fit≤16→test>16:  exact  rel.err med K̂@max
pattern: band layers, tail+answer, all stats         0.35   0.46     0.66 |                    0.00     7.89     1049.0
pattern: band layers, k_eff only                     0.30   0.41     0.58 |                    0.03     0.37       22.0
pattern: band layers, peak counts only               0.28   0.36     0.53 |                    0.01     5.77      744.5
pattern: all FL layers, all stats                    0.38   0.51     0.71 |                    0.00     7.90     1023.0
pattern: probe layer L only                          0.25   0.32     0.47 |                    0.00    22.24     2527.0
ORACLE needle share, band layers                     0.31   0.40     0.55 |                    0.02     1.26      137.0
state: answer row at L                               0.35   0.43     0.71 |                    0.00     0.56       14.0
state: answer row final                              0.35   0.44     0.68 |                    0.00     0.59       12.0
D3 (layer,head) with k_eff slope∈[.6,1.1] & R²≥.95: 1/256; best L15H2 slope 0.44 R² 0.992; k_eff by K: 1:26.2 2:26.1 3:25.9 4:26.8 6:28.6 8:29.0 12:30.7 16:33.6 24:35.4 32:39.9 48:45.9
D3 single-head estimator (fit K≤16 → K>16): rel.err 0.55, within-1 0.00, median K̂ at K=48: 20.2
