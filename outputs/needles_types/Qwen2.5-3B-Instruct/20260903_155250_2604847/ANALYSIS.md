# Bundle E — Qwen/Qwen2.5-3B-Instruct band 18+24 n=30 (outputs/needles_types/Qwen2.5-3B-Instruct/20260903_155250_2604847)

E1 exact by competitor type × K:
kind  K  1 K  2 K  4 K  6 K  8 K 12 K 16
HC    0.00 0.33 0.57 0.20 0.07 0.00 0.00
HN    0.43 0.33 0.03 0.03 0.00 0.00 0.00
HP    0.27 0.30 0.20 0.03 0.00 0.00 0.00
HR    0.07 0.50 0.50 0.27 0.03 0.03 0.00
HU    0.80 0.87 0.10 0.00 0.00 0.00 0.00
E1 median emitted by type × K:
HC       2    3    4    4    6   10   11
HN       0    1    2    3    4    6    8
HP       0    1    2    4    4    6   10
HR       0    2    4    4    6   11   14
HU       1    2    2    4    4    6    8

E2 score structure at the band (head-mean logits; fraction = (class − no)/(full − no)):
HC: full:3.54(frac 1.00) half_p:3.52(frac 0.85) half_r:3.41(frac 0.26) no:3.36(frac 0.00)
HN: full:3.77(frac 1.00) no:3.23(frac 0.00)
HP: full:3.29(frac nan) half_p:3.11(frac nan)
HR: full:3.85(frac nan) half_r:3.58(frac nan)
HU: full:6.07(frac 1.00) filler:2.80(frac 0.00)

E3 sharpening (exact by β × K), HC and HN:
HC none           0.00 0.33 0.57 0.20 0.07 0.00 0.00  | median   2   3   4   4   6  10  11
HC sharp1.5_band  0.03 0.23 0.57 0.13 0.07 0.00 0.00  | median   3   4   4   4   6   7  11
HC sharp2.0_band  0.07 0.20 0.53 0.17 0.03 0.00 0.00  | median   3   3   4   4   4   6   8
HC sharp3.0_band  0.07 0.20 0.33 0.03 0.00 0.00 0.00  | median   3   3   3   4   4   4   4
HC sharp2.0_all   0.07 0.33 0.33 0.03 0.03 0.00 0.00  | median   2   3   3   3   4   4   5
HN none           0.43 0.33 0.03 0.03 0.00 0.00 0.00  | median   0   1   2   3   4   6   8
HN sharp1.5_band  0.40 0.43 0.10 0.00 0.00 0.00 0.00  | median   0   1   2   3   4   4   8
HN sharp2.0_band  0.33 0.37 0.00 0.00 0.00 0.00 0.00  | median   0   1   2   3   4   4   6
HN sharp3.0_band  0.30 0.43 0.07 0.00 0.00 0.00 0.00  | median   1   2   2   3   4   4   4
HN sharp2.0_all   0.50 0.33 0.00 0.00 0.00 0.00 0.00  | median   1   1   2   2   3   4   4

E4 transfer function (HC, needle share × m at all layers): median emitted (exact)
K=4: m=0.5:3(0.33) m=1:4(0.57) m=2:5(0.33) m=3:6(0.23) m=4:6(0.17)
K=8: m=0.5:4(0.00) m=1:6(0.07) m=2:10(0.17) m=3:10(0.10) m=4:10(0.17)
