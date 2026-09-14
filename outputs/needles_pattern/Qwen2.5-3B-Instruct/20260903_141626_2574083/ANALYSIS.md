# Bundle D — Qwen/Qwen2.5-3B-Instruct L27 band 18–24 FL=[13, 18, 19, 20, 21, 22, 23, 24, 27, 31] (outputs/needles_pattern/Qwen2.5-3B-Instruct/20260903_141626_2574083)

## HC  (n=440, K=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48], sentences F=[64]…)
feature set                                         exact  ex≤16  within1 | fit≤16→test>16:  exact  rel.err med K̂@max
pattern: band layers, tail+answer, all stats         0.11   0.15     0.37 |                    0.04     0.46       66.5
pattern: band layers, k_eff only                     0.25   0.33     0.48 |                    0.03     0.68        3.0
pattern: band layers, peak counts only               0.07   0.09     0.25 |                    0.01     0.56       19.0
pattern: all FL layers, all stats                    0.18   0.24     0.43 |                    0.03     0.31       29.5
pattern: probe layer L only                          0.18   0.24     0.40 |                    0.02     0.52       10.0
ORACLE needle share, band layers                     0.25   0.33     0.54 |                    0.00    10.03      952.0
state: answer row at L                               0.24   0.30     0.55 |                    0.07     0.30       25.0
state: answer row final                              0.27   0.34     0.59 |                    0.02     0.44       14.0
D3 (layer,head) with k_eff slope∈[.6,1.1] & R²≥.95: 0/160; best L13H0 slope 0.03 R² 0.598; k_eff by K: 1:32.3 2:32.3 3:32.5 4:32.7 6:32.5 8:32.6 12:32.1 16:32.1 24:33.9 32:33.6 48:33.7
D3 single-head estimator (fit K≤16 → K>16): rel.err 0.80, within-1 0.00, median K̂ at K=48: 6.3

## HF  (n=440, K=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48], sentences F=[65, 66, 67]…)
feature set                                         exact  ex≤16  within1 | fit≤16→test>16:  exact  rel.err med K̂@max
pattern: band layers, tail+answer, all stats         0.34   0.43     0.67 |                    0.00     5.09      628.5
pattern: band layers, k_eff only                     0.25   0.34     0.47 |                    0.01     0.63        3.0
pattern: band layers, peak counts only               0.19   0.26     0.44 |                    0.02     5.40      653.5
pattern: all FL layers, all stats                    0.41   0.53     0.70 |                    0.00     4.63      568.0
pattern: probe layer L only                          0.23   0.31     0.50 |                    0.01     9.37      977.0
ORACLE needle share, band layers                     0.29   0.39     0.52 |                    0.00     1.71      154.0
state: answer row at L                               0.27   0.34     0.65 |                    0.00     0.65       11.0
state: answer row final                              0.26   0.32     0.61 |                    0.00     0.54       14.0
D3 (layer,head) with k_eff slope∈[.6,1.1] & R²≥.95: 1/160; best L13H14 slope 0.79 R² 0.992; k_eff by K: 1:40.5 2:41.2 3:41.1 4:41.6 6:44.3 8:46.8 12:47.5 16:53.2 24:60.8 32:64.7 48:76.7
D3 single-head estimator (fit K≤16 → K>16): rel.err 0.30, within-1 0.02, median K̂ at K=48: 30.5
