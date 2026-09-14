# Bundle E — Qwen/Qwen2.5-7B-Instruct band 15+18 n=30 (outputs/needles_types/Qwen2.5-7B-Instruct/20260903_155251_2605280)

E1 exact by competitor type × K:
kind  K  1 K  2 K  4 K  6 K  8 K 12 K 16
HC    0.07 0.30 0.60 0.00 0.00 0.00 0.00
HN    0.80 0.63 0.03 0.00 0.00 0.00 0.00
HP    0.57 0.63 0.63 0.00 0.00 0.00 0.00
HR    0.67 0.40 0.03 0.00 0.00 0.00 0.00
HU    0.93 1.00 0.67 0.00 0.00 0.00 0.00
E1 median emitted by type × K:
HC       2    3    4    4    5    5    7
HN       1    2    2    4    4    5    5
HP       1    2    4    5    5    7    7
HR       1    2    3    4    4    5    5
HU       1    2    4    4    5    7    7

E2 score structure at the band (head-mean logits; fraction = (class − no)/(full − no)):
HC: full:2.40(frac 1.00) half_p:2.33(frac 0.77) half_r:2.18(frac 0.28) no:2.09(frac 0.00)
HN: full:2.44(frac 1.00) no:1.96(frac 0.00)
HP: full:2.07(frac nan) half_p:1.99(frac nan)
HR: full:2.44(frac nan) half_r:2.24(frac nan)
HU: full:4.24(frac 1.00) filler:-0.13(frac 0.00)

E3 sharpening (exact by β × K), HC and HN:
HC none           0.07 0.30 0.60 0.00 0.00 0.00 0.00  | median   2   3   4   4   5   5   7
HC sharp1.5_band  0.07 0.30 0.57 0.00 0.00 0.00 0.00  | median   2   3   4   4   4   5   5
HC sharp2.0_band  0.03 0.33 0.47 0.00 0.00 0.00 0.00  | median   2   3   4   4   4   4   5
HC sharp3.0_band  0.00 0.27 0.37 0.00 0.00 0.00 0.00  | median   2   3   3   4   4   4   4
HC sharp2.0_all   0.03 0.23 0.50 0.00 0.00 0.00 0.00  | median   3   3   4   4   4   4   4
HN none           0.80 0.63 0.03 0.00 0.00 0.00 0.00  | median   1   2   2   4   4   5   5
HN sharp1.5_band  0.67 0.60 0.17 0.00 0.00 0.00 0.00  | median   1   2   3   4   4   4   5
HN sharp2.0_band  0.50 0.50 0.23 0.00 0.00 0.00 0.00  | median   1   2   3   4   4   4   4
HN sharp3.0_band  0.27 0.37 0.23 0.00 0.00 0.00 0.00  | median   1   2   2   4   4   4   4
HN sharp2.0_all   0.33 0.73 0.07 0.00 0.00 0.00 0.00  | median   2   2   2   3   3   4   4

E4 transfer function (HC, needle share × m at all layers): median emitted (exact)
K=4: m=0.5:3(0.23) m=1:4(0.60) m=2:4(0.70) m=3:4(0.53) m=4:4(0.57)
K=8: m=0.5:4(0.00) m=1:5(0.00) m=2:5(0.10) m=3:7(0.17) m=4:7(0.23)
