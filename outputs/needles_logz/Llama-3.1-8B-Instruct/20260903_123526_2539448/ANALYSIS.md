# log Z probe — meta-llama/Llama-3.1-8B-Instruct L20 n=50 (outputs/needles_logz/Llama-3.1-8B-Instruct/20260903_123526_2539448)

## H0  (n=600, K=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64])
feature set                         exact  exact≤16  within1     R2 |  fit K≤16 → test K>16:  exact  within1  rel.err med K̂@64
logZ_ctx L                           0.45      0.62     0.70   0.99 |                          0.00     0.00     0.44      24.0
logZ_all L                           0.39      0.56     0.63   0.99 |                          0.01     0.02     0.41      23.0
logZ_ctx all layers                  0.85      1.00     0.98   1.00 |                          0.00     0.00     0.33      31.0
dlogZ ctrl L                         0.29      0.41     0.53   0.96 |                          0.00     0.00     0.59      14.0
dlogZ ctrl all layers                0.53      0.72     0.75   1.00 |                          0.00     0.00     0.59      14.0
logZ_ctx L, selective heads          0.45      0.62     0.70   0.99 |                          0.00     0.00     0.44      24.0
state L                              0.67      0.85     0.88   1.00 |                          0.00     0.00     0.53      17.0
state L+6                            0.68      0.90     0.87   1.00 |                          0.00     0.00     0.57      14.0
state final                          0.62      0.82     0.84   1.00 |                          0.00     0.00     0.63      11.0
N3 slope d(mean logZ_ctx)/d(log K), 3 most selective heads: 0.80   K1:-1.94 K2:-1.00 K3:-0.28 K4:0.20 K6:0.75 K8:0.97 K12:1.21 K16:1.32 K24:1.51 K32:1.48 K48:1.51 K64:1.48
N3 slope d(mean logZ_ctx)/d(log K), head-mean: 0.51   K1:-2.43 K2:-2.06 K3:-1.72 K4:-1.33 K6:-1.04 K8:-0.92 K12:-0.83 K16:-0.72 K24:-0.54 K32:-0.45 K48:-0.40 K64:-0.32
selective heads at L (share>.8 at K>=12): 32/32; best-head share by K: K1:1.00 K2:1.00 K3:1.00 K4:1.00 K6:1.00 K8:1.00 K12:1.00 K16:1.00 K24:1.00 K32:1.00 K48:1.00 K64:1.00

## HC  (n=600, K=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64])
feature set                         exact  exact≤16  within1     R2 |  fit K≤16 → test K>16:  exact  within1  rel.err med K̂@64
logZ_ctx L                           0.23      0.31     0.51   0.94 |                          0.02     0.06     0.25      47.0
logZ_all L                           0.23      0.32     0.51   0.94 |                          0.01     0.04     0.37      31.5
logZ_ctx all layers                  0.21      0.30     0.46   0.93 |                          0.01     0.04     0.42      23.0
dlogZ ctrl L                         0.17      0.24     0.43   0.91 |                          0.02     0.07     0.43      18.5
dlogZ ctrl all layers                0.19      0.27     0.45   0.92 |                          0.03     0.07     0.39      44.0
logZ_ctx L, selective heads          0.09      0.12     0.24   0.62 |                          0.00     0.01     0.59      20.0
state L                              0.28      0.37     0.62   0.97 |                          0.02     0.07     0.31      42.5
state L+6                            0.28      0.39     0.59   0.96 |                          0.03     0.05     0.39      30.0
state final                          0.26      0.36     0.56   0.96 |                          0.00     0.01     0.53      12.0
N3 slope d(mean logZ_ctx)/d(log K), 3 most selective heads: 0.39   K1:-0.81 K2:-0.93 K3:-1.00 K4:-0.90 K6:-0.82 K8:-0.78 K12:-0.59 K16:-0.41 K24:-0.16 K32:0.17 K48:0.44 K64:0.63
N3 slope d(mean logZ_ctx)/d(log K), head-mean: 0.15   K1:-0.94 K2:-0.98 K3:-1.05 K4:-1.01 K6:-1.00 K8:-0.98 K12:-0.91 K16:-0.85 K24:-0.73 K32:-0.61 K48:-0.46 K64:-0.32
selective heads at L (share>.8 at K>=12): 1/32; best-head share by K: K1:0.18 K2:0.26 K3:0.32 K4:0.39 K6:0.47 K8:0.57 K12:0.67 K16:0.76 K24:0.84 K32:0.89 K48:0.96 K64:1.00

## HF  (n=600, K=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64])
feature set                         exact  exact≤16  within1     R2 |  fit K≤16 → test K>16:  exact  within1  rel.err med K̂@64
logZ_ctx L                           0.23      0.31     0.53   0.95 |                          0.01     0.04     0.53      18.0
logZ_all L                           0.22      0.32     0.52   0.94 |                          0.04     0.07     0.30      30.5
logZ_ctx all layers                  0.40      0.55     0.66   0.98 |                          0.02     0.07     2.12     380.0
dlogZ ctrl L                         0.14      0.18     0.40   0.87 |                          0.00     0.00     0.63      11.0
dlogZ ctrl all layers                0.23      0.33     0.45   0.92 |                          0.00     0.01     0.55      11.0
dlogZ counterfactual L               0.20      0.28     0.48   0.93 |                          0.04     0.06     0.47      18.5
logZ_ctx L, selective heads          0.09      0.13     0.27   0.74 |                          0.00     0.01     0.58      17.0
state L                              0.35      0.47     0.68   0.98 |                          0.00     0.00     0.62      12.0
state L+6                            0.32      0.44     0.63   0.98 |                          0.00     0.00     0.62      12.0
state final                          0.34      0.47     0.65   0.98 |                          0.00     0.00     0.66      11.0
N3 slope d(mean logZ_ctx)/d(log K), 3 most selective heads: 0.38   K1:-1.03 K2:-1.00 K3:-1.15 K4:-1.08 K6:-1.01 K8:-0.96 K12:-0.67 K16:-0.42 K24:-0.23 K32:0.01 K48:0.28 K64:0.28
N3 slope d(mean logZ_ctx)/d(log K), head-mean: 0.17   K1:-1.01 K2:-1.03 K3:-1.10 K4:-1.10 K6:-1.07 K8:-1.03 K12:-0.91 K16:-0.83 K24:-0.70 K32:-0.59 K48:-0.45 K64:-0.39
selective heads at L (share>.8 at K>=12): 0/32; best-head share by K: K1:0.17 K2:0.25 K3:0.32 K4:0.43 K6:0.46 K8:0.56 K12:0.63 K16:0.70 K24:0.75 K32:0.79 K48:0.86 K64:0.88
