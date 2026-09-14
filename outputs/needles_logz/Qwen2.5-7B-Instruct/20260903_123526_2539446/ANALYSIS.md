# log Z probe — Qwen/Qwen2.5-7B-Instruct L22 n=50 (outputs/needles_logz/Qwen2.5-7B-Instruct/20260903_123526_2539446)

## H0  (n=600, K=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64])
feature set                         exact  exact≤16  within1     R2 |  fit K≤16 → test K>16:  exact  within1  rel.err med K̂@64
logZ_ctx L                           0.31      0.44     0.58   0.97 |                          0.00     0.00     0.61      13.0
logZ_all L                           0.26      0.38     0.52   0.96 |                          0.00     0.00     0.69       9.0
logZ_ctx all layers                  0.84      0.98     0.98   1.00 |                          0.00     0.00     0.37      32.0
dlogZ ctrl L                         0.26      0.38     0.50   0.95 |                          0.00     0.00     0.68      10.0
dlogZ ctrl all layers                0.49      0.68     0.72   0.99 |                          0.00     0.00     0.63      11.0
logZ_ctx L, selective heads          0.31      0.44     0.58   0.97 |                          0.00     0.00     0.61      13.0
state L                              0.70      0.92     0.88   1.00 |                          0.00     0.00     0.45      21.0
state L+6                            0.69      0.91     0.86   1.00 |                          0.00     0.00     0.60      12.0
state final                          0.66      0.86     0.85   1.00 |                          0.00     0.00     0.61      12.0
N3 slope d(mean logZ_ctx)/d(log K), 3 most selective heads: 0.22   K1:0.48 K2:0.76 K3:1.43 K4:1.34 K6:1.30 K8:1.40 K12:1.44 K16:1.64 K24:1.41 K32:1.44 K48:1.64 K64:1.58
N3 slope d(mean logZ_ctx)/d(log K), head-mean: 0.48   K1:-1.35 K2:-1.05 K3:-0.54 K4:-0.20 K6:0.16 K8:0.33 K12:0.56 K16:0.58 K24:0.63 K32:0.63 K48:0.53 K64:0.56
selective heads at L (share>.8 at K>=12): 28/28; best-head share by K: K1:1.00 K2:1.00 K3:1.00 K4:1.00 K6:1.00 K8:1.00 K12:1.00 K16:1.00 K24:1.00 K32:1.00 K48:1.00 K64:1.00

## HC  (n=600, K=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64])
feature set                         exact  exact≤16  within1     R2 |  fit K≤16 → test K>16:  exact  within1  rel.err med K̂@64
logZ_ctx L                           0.16      0.20     0.43   0.89 |                          0.01     0.03     0.54      14.0
logZ_all L                           0.14      0.18     0.40   0.88 |                          0.01     0.04     0.54      10.0
logZ_ctx all layers                  0.12      0.18     0.35   0.85 |                          0.03     0.07     0.57      69.0
dlogZ ctrl L                         0.13      0.18     0.36   0.86 |                          0.01     0.03     0.59      11.0
dlogZ ctrl all layers                0.14      0.20     0.33   0.82 |                          0.01     0.06     0.50      14.5
logZ_ctx L, selective heads          0.12      0.17     0.33   0.83 |                          0.00     0.00     0.68      15.5
state L                              0.22      0.30     0.50   0.94 |                          0.02     0.09     0.40      18.0
state L+6                            0.20      0.27     0.53   0.94 |                          0.01     0.02     0.56      10.0
state final                          0.22      0.28     0.53   0.94 |                          0.01     0.03     0.61       8.0
N3 slope d(mean logZ_ctx)/d(log K), 3 most selective heads: 0.61   K1:1.39 K2:1.70 K3:1.85 K4:2.08 K6:2.19 K8:2.64 K12:2.74 K16:2.92 K24:3.08 K32:3.27 K48:3.52 K64:4.08
N3 slope d(mean logZ_ctx)/d(log K), head-mean: 0.16   K1:-0.32 K2:-0.30 K3:-0.31 K4:-0.27 K6:-0.28 K8:-0.16 K12:-0.17 K16:-0.16 K24:-0.12 K32:-0.02 K48:0.19 K64:0.56
selective heads at L (share>.8 at K>=12): 12/28; best-head share by K: K1:0.26 K2:0.47 K3:0.50 K4:0.58 K6:0.73 K8:0.81 K12:0.89 K16:0.93 K24:0.97 K32:0.98 K48:1.00 K64:1.00

## HF  (n=600, K=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64])
feature set                         exact  exact≤16  within1     R2 |  fit K≤16 → test K>16:  exact  within1  rel.err med K̂@64
logZ_ctx L                           0.17      0.23     0.44   0.91 |                          0.01     0.03     0.44      24.0
logZ_all L                           0.17      0.23     0.42   0.90 |                          0.01     0.02     0.48      24.0
logZ_ctx all layers                  0.32      0.46     0.62   0.97 |                          0.01     0.06     0.24      49.0
dlogZ ctrl L                         0.14      0.18     0.36   0.84 |                          0.03     0.04     0.53      20.5
dlogZ ctrl all layers                0.14      0.20     0.37   0.83 |                          0.01     0.04     0.48      23.0
dlogZ counterfactual L               0.14      0.20     0.36   0.87 |                          0.01     0.02     0.44      26.0
logZ_ctx L, selective heads          0.10      0.15     0.25   0.60 |                          0.00     0.00     0.73       8.0
state L                              0.29      0.40     0.61   0.97 |                          0.00     0.00     0.52      25.0
state L+6                            0.27      0.38     0.60   0.97 |                          0.00     0.00     0.52      15.0
state final                          0.27      0.36     0.57   0.97 |                          0.00     0.00     0.54      14.0
N3 slope d(mean logZ_ctx)/d(log K), 3 most selective heads: 0.35   K1:1.14 K2:1.60 K3:1.74 K4:1.94 K6:2.22 K8:2.50 K12:2.64 K16:2.83 K24:2.68 K32:2.62 K48:2.80 K64:2.26
N3 slope d(mean logZ_ctx)/d(log K), head-mean: 0.11   K1:-0.38 K2:-0.33 K3:-0.35 K4:-0.35 K6:-0.22 K8:-0.19 K12:-0.13 K16:-0.03 K24:-0.06 K32:-0.07 K48:0.11 K64:-0.08
selective heads at L (share>.8 at K>=12): 6/28; best-head share by K: K1:0.36 K2:0.44 K3:0.56 K4:0.67 K6:0.72 K8:0.81 K12:0.88 K16:0.89 K24:0.92 K32:0.94 K48:0.96 K64:0.96
