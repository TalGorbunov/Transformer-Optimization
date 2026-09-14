# Bundle E — meta-llama/Llama-3.1-8B-Instruct band 11+15 n=30 (outputs/needles_types/Llama-3.1-8B-Instruct/20260903_155252_2605281)

E1 exact by competitor type × K:
kind  K  1 K  2 K  4 K  6 K  8 K 12 K 16
HC    0.27 0.17 0.10 0.00 0.07 0.50 0.00
HN    0.90 0.13 0.17 0.07 0.07 0.73 0.00
HP    0.40 0.03 0.00 0.00 0.17 0.27 0.00
HR    0.57 0.27 0.07 0.03 0.07 0.20 0.03
HU    0.97 1.00 0.40 0.10 0.47 0.73 0.00
E1 median emitted by type × K:
HC       2    4    6    9   12   12   17
HN       1    1    2    6    9   12   16
HP       0    1    2    7   12   16   20
HR       1    1    5    9   12   17   17
HU       1    2    4    6    8   12   12

E2 score structure at the band (head-mean logits; fraction = (class − no)/(full − no)):
HC: full:-6.17(frac 1.00) half_p:-6.29(frac 0.77) half_r:-6.57(frac 0.21) no:-6.68(frac 0.00)
HN: full:-5.89(frac 1.00) no:-6.71(frac 0.00)
HP: full:-6.32(frac nan) half_p:-6.59(frac nan)
HR: full:-6.08(frac nan) half_r:-6.54(frac nan)
HU: full:-4.32(frac 1.00) filler:-8.42(frac 0.00)

E3 sharpening (exact by β × K), HC and HN:
HC none           0.27 0.17 0.10 0.00 0.07 0.50 0.00  | median   2   4   6   9  12  12  17
HC sharp1.5_band  0.23 0.13 0.03 0.03 0.40 0.47 0.00  | median   2   5   5   8   9  10  12
HC sharp2.0_band  0.33 0.13 0.00 0.03 0.50 0.20 0.00  | median   2   5   5   7   8   8  12
HC sharp3.0_band  0.50 0.03 0.00 0.00 0.13 0.10 0.00  | median   1   3   3   4   5   5   5
HC sharp2.0_all   0.17 0.23 0.00 0.00 0.17 0.13 0.00  | median   2   2   3   3   3   5   8
HN none           0.90 0.13 0.17 0.07 0.07 0.73 0.00  | median   1   1   2   6   9  12  16
HN sharp1.5_band  0.90 0.10 0.07 0.03 0.40 0.27 0.00  | median   1   1   3   5   8   9  12
HN sharp2.0_band  0.90 0.03 0.03 0.00 0.30 0.10 0.00  | median   1   1   4   5   7   8   8
HN sharp3.0_band  0.93 0.00 0.00 0.00 0.17 0.03 0.00  | median   1   1   5   3   5   5   5
HN sharp2.0_all   0.50 0.10 0.00 0.00 0.03 0.07 0.00  | median   0   1   2   3   3   3   3

E4 transfer function (HC, needle share × m at all layers): median emitted (exact)
K=4: m=0.5:4(0.07) m=1:6(0.10) m=2:8(0.00) m=3:9(0.00) m=4:11(0.00)
K=8: m=0.5:8(0.23) m=1:12(0.07) m=2:17(0.00) m=3:17(0.00) m=4:17(0.00)
