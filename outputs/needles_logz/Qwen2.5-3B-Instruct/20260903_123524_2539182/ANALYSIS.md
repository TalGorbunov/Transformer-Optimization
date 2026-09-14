# log Z probe — Qwen/Qwen2.5-3B-Instruct L27 n=50 (outputs/needles_logz/Qwen2.5-3B-Instruct/20260903_123524_2539182)

## H0  (n=600, K=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64])
feature set                         exact  exact≤16  within1     R2 |  fit K≤16 → test K>16:  exact  within1  rel.err med K̂@64
logZ_ctx L                           0.30      0.43     0.56   0.97 |                          0.00     0.00     0.45      24.0
logZ_all L                           0.30      0.43     0.52   0.96 |                          0.00     0.00     0.50      20.0
logZ_ctx all layers                  0.83      0.99     0.96   1.00 |                          0.00     0.00     0.23      41.0
dlogZ ctrl L                         0.23      0.32     0.44   0.92 |                          0.01     0.03     0.50      15.0
dlogZ ctrl all layers                0.30      0.43     0.55   0.97 |                          0.00     0.00     0.61      12.0
logZ_ctx L, selective heads          0.30      0.43     0.56   0.97 |                          0.00     0.00     0.45      24.0
state L                              0.69      0.89     0.90   1.00 |                          0.00     0.00     0.46      20.0
state L+6                            0.61      0.82     0.85   1.00 |                          0.00     0.00     0.51      17.0
state final                          0.61      0.82     0.83   1.00 |                          0.00     0.00     0.63      12.0
N3 slope d(mean logZ_ctx)/d(log K), 3 most selective heads: -0.24   K1:13.85 K2:14.46 K3:14.10 K4:13.71 K6:13.61 K8:13.44 K12:13.11 K16:13.31 K24:13.08 K32:13.10 K48:13.49 K64:13.38
N3 slope d(mean logZ_ctx)/d(log K), head-mean: -0.21   K1:-2.00 K2:-3.01 K3:-3.64 K4:-3.92 K6:-4.35 K8:-4.12 K12:-4.07 K16:-4.01 K24:-3.82 K32:-4.00 K48:-3.22 K64:-3.49
selective heads at L (share>.8 at K>=12): 16/16; best-head share by K: K1:1.00 K2:1.00 K3:1.00 K4:1.00 K6:1.00 K8:1.00 K12:1.00 K16:1.00 K24:1.00 K32:1.00 K48:1.00 K64:1.00

## HC  (n=600, K=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64])
feature set                         exact  exact≤16  within1     R2 |  fit K≤16 → test K>16:  exact  within1  rel.err med K̂@64
logZ_ctx L                           0.15      0.19     0.42   0.90 |                          0.06     0.10     0.24      66.0
logZ_all L                           0.15      0.20     0.41   0.89 |                          0.03     0.06     0.38      90.5
logZ_ctx all layers                  0.15      0.21     0.35   0.86 |                          0.02     0.06     1.11     190.0
dlogZ ctrl L                         0.11      0.15     0.33   0.80 |                          0.03     0.04     0.50      16.0
dlogZ ctrl all layers                0.10      0.15     0.29   0.68 |                          0.03     0.04     0.59       9.5
logZ_ctx L, selective heads          0.13      0.17     0.35   0.84 |                          0.01     0.03     0.38      49.5
state L                              0.22      0.30     0.50   0.94 |                          0.01     0.04     0.42      15.0
state L+6                            0.21      0.28     0.53   0.94 |                          0.02     0.06     0.44      16.0
state final                          0.24      0.32     0.54   0.94 |                          0.01     0.04     0.53      10.0
N3 slope d(mean logZ_ctx)/d(log K), 3 most selective heads: 0.77   K1:13.17 K2:13.74 K3:14.28 K4:14.28 K6:14.43 K8:15.13 K12:15.24 K16:15.73 K24:15.65 K32:15.93 K48:16.21 K64:16.34
N3 slope d(mean logZ_ctx)/d(log K), head-mean: -0.71   K1:-1.29 K2:-1.68 K3:-1.83 K4:-2.06 K6:-1.68 K8:-1.95 K12:-2.60 K16:-2.96 K24:-3.64 K32:-3.99 K48:-4.17 K64:-3.49
selective heads at L (share>.8 at K>=12): 7/16; best-head share by K: K1:0.41 K2:0.60 K3:0.66 K4:0.73 K6:0.82 K8:0.84 K12:0.89 K16:0.93 K24:0.97 K32:0.99 K48:1.00 K64:1.00

## HF  (n=600, K=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64])
feature set                         exact  exact≤16  within1     R2 |  fit K≤16 → test K>16:  exact  within1  rel.err med K̂@64
logZ_ctx L                           0.13      0.18     0.35   0.87 |                          0.03     0.08     0.30      47.0
logZ_all L                           0.12      0.17     0.38   0.87 |                          0.04     0.10     0.33      47.0
logZ_ctx all layers                  0.32      0.44     0.61   0.97 |                          0.01     0.01     1.35     194.0
dlogZ ctrl L                         0.12      0.17     0.31   0.78 |                          0.01     0.07     0.41      32.5
dlogZ ctrl all layers                0.12      0.17     0.30   0.73 |                          0.03     0.04     0.71      72.5
dlogZ counterfactual L               0.14      0.20     0.42   0.85 |                          0.03     0.07     0.43      47.5
logZ_ctx L, selective heads          0.11      0.16     0.29   0.76 |                          0.00     0.00     0.84       5.0
state L                              0.28      0.39     0.61   0.97 |                          0.00     0.00     0.70       9.0
state L+6                            0.24      0.34     0.54   0.96 |                          0.00     0.00     0.66      11.0
state final                          0.25      0.34     0.56   0.96 |                          0.00     0.00     0.67       9.0
N3 slope d(mean logZ_ctx)/d(log K), 3 most selective heads: 0.55   K1:12.87 K2:13.78 K3:13.82 K4:14.14 K6:14.55 K8:14.67 K12:14.89 K16:14.96 K24:15.57 K32:15.20 K48:15.25 K64:15.03
N3 slope d(mean logZ_ctx)/d(log K), head-mean: -0.92   K1:-1.29 K2:-1.65 K3:-1.82 K4:-1.98 K6:-1.73 K8:-2.11 K12:-2.45 K16:-3.08 K24:-3.48 K32:-4.51 K48:-4.46 K64:-4.96
selective heads at L (share>.8 at K>=12): 5/16; best-head share by K: K1:0.40 K2:0.58 K3:0.66 K4:0.76 K6:0.77 K8:0.82 K12:0.87 K16:0.90 K24:0.92 K32:0.95 K48:0.97 K64:0.98
