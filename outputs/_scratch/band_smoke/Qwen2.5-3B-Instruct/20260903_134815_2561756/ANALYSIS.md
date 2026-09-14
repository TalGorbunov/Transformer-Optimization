# Bundle C — Qwen/Qwen2.5-3B-Instruct L27 band 18–24 (outputs/_scratch/band_smoke/Qwen2.5-3B-Instruct/20260903_134815_2561756)

C1 head-mean needle share of the ANSWER row by layer (rows: K): layers     0     1     2     3     4     5     6     7     8     9    10    11    12    13    14    15    16    17    18    19    20    21    22    23    24    25    26    27    28    29    30    31    32    33    34    35
K 2  0.01  0.01  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.01  0.01  0.01  0.01  0.11  0.00  0.01  0.00  0.00  0.00  0.00  0.00  0.00
K 8  0.04  0.03  0.01  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.01  0.02  0.23  0.01  0.01  0.00  0.01  0.00  0.00  0.00  0.01
C1 head-mean needle share of the TAIL rows (mean over last 12 rows) at band layers, by K: K2:0.019 K8:0.034
C1 same at the probe layer L: K2:0.049 K8:0.106
C1 k_eff over needles, answer row, mean over band layers: K2:1.8 K8:5.3

C2 clamp — median emitted count (exact) by K:
none     : K2:4(0.00) K8:6(0.00)
band     : K2:4(0.00) K8:6(0.00)
all      : K2:4(0.00) K8:5(0.00)
amp_band : K2:4(0.00) K8:6(0.00)

C3 head patch effect at the band (mean over pairs), top |heads|: L24H5:-2.00 L24H0:-1.05 L22H3:-1.05 L20H13:-1.05 L22H4:+1.00 L24H8:-1.00 L23H5:+1.00 L24H1:+1.00 L23H13:-1.00 L23H9:-1.00
C3 per-layer sum of head effects: L18:+3.73 L19:+3.77 L20:+0.77 L21:-1.18 L22:+1.77 L23:+0.00 L24:-4.05
C3 joint patch of each pair's top-5 heads (per layer, mean effect): L18:+0.00(n=1) L19:+2.00(n=1) L22:+0.45(n=1) L23:+0.27(n=1) L24:-4.00(n=1)
