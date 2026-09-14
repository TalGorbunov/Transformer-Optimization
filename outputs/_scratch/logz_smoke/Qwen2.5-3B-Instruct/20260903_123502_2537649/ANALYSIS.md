# log Z probe — Qwen/Qwen2.5-3B-Instruct L27 n=3 (outputs/_scratch/logz_smoke/Qwen2.5-3B-Instruct/20260903_123502_2537649)

## H0  (n=9, K=[1, 4, 16])
feature set                         exact  exact≤16  within1     R2 |  fit K≤16 → test K>16:  exact  within1  rel.err med K̂@64
logZ_ctx L                           0.56      0.56     0.67   0.95 |                           nan      nan      nan       nan
logZ_all L                           0.67      0.67     0.67   0.97 |                           nan      nan      nan       nan
logZ_ctx all layers                  0.56      0.56     0.89   0.97 |                           nan      nan      nan       nan
dlogZ ctrl L                         0.33      0.33     0.78   0.93 |                           nan      nan      nan       nan
dlogZ ctrl all layers                0.22      0.22     0.78   0.89 |                           nan      nan      nan       nan
logZ_ctx L, selective heads          0.56      0.56     0.67   0.95 |                           nan      nan      nan       nan
state L                              0.56      0.56     0.78   0.97 |                           nan      nan      nan       nan
state L+6                            0.67      0.67     0.89   0.99 |                           nan      nan      nan       nan
state final                          0.78      0.78     0.89   0.99 |                           nan      nan      nan       nan
N3 slope d(mean logZ_ctx)/d(log K), 3 most selective heads: 0.49   K1:11.16 K4:11.09 K16:12.53
N3 slope d(mean logZ_ctx)/d(log K), head-mean: -0.85   K1:-1.59 K4:-4.10 K16:-3.94
selective heads at L (share>.8 at K>=12): 16/16; best-head share by K: K1:1.00 K4:1.00 K16:1.00

## HF  (n=9, K=[1, 4, 16])
feature set                         exact  exact≤16  within1     R2 |  fit K≤16 → test K>16:  exact  within1  rel.err med K̂@64
logZ_ctx L                           0.22      0.22     0.44   0.25 |                           nan      nan      nan       nan
logZ_all L                           0.22      0.22     0.56   0.13 |                           nan      nan      nan       nan
logZ_ctx all layers                  0.33      0.33     0.67   0.89 |                           nan      nan      nan       nan
dlogZ ctrl L                         0.11      0.11     0.33   0.10 |                           nan      nan      nan       nan
dlogZ ctrl all layers                0.22      0.22     0.33   0.11 |                           nan      nan      nan       nan
dlogZ counterfactual L               0.22      0.22     0.44   0.71 |                           nan      nan      nan       nan
logZ_ctx L, selective heads          0.22      0.22     0.44   0.40 |                           nan      nan      nan       nan
state L                              0.11      0.11     0.67   0.88 |                           nan      nan      nan       nan
state L+6                            0.22      0.22     0.67   0.93 |                           nan      nan      nan       nan
state final                          0.22      0.22     0.78   0.92 |                           nan      nan      nan       nan
N3 slope d(mean logZ_ctx)/d(log K), 3 most selective heads: 1.01   K1:13.49 K4:14.46 K16:16.27
N3 slope d(mean logZ_ctx)/d(log K), head-mean: -0.60   K1:-1.20 K4:-2.29 K16:-2.86
selective heads at L (share>.8 at K>=12): 4/16; best-head share by K: K1:0.30 K4:0.44 K16:0.93
