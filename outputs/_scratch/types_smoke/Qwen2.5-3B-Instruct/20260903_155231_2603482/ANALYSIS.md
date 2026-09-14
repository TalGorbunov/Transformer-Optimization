# Bundle E — Qwen/Qwen2.5-3B-Instruct band 18+24 n=2 (outputs/_scratch/types_smoke/Qwen2.5-3B-Instruct/20260903_155231_2603482)

E1 exact by competitor type × K:
kind  K  4 K  8
HC    1.00 0.50
HU    0.00 0.00
E1 median emitted by type × K:
HC       4    6
HU       2    4

E2 score structure at the band (head-mean logits; fraction = (class − no)/(full − no)):
HC: full:3.43(frac 1.00) half_p:3.53(frac -0.46) half_r:3.41(frac 1.30) no:3.50(frac -0.00)
HU: full:6.01(frac 1.00) filler:2.78(frac 0.00)

E3 sharpening (exact by β × K), HC and HN:
HC none           1.00 0.50  | median   4   6
HC sharp1.5_band  1.00 0.00  | median   4   5
HC sharp2.0_band  1.00 0.00  | median   4   4
HC sharp3.0_band  0.50 0.00  | median   2   4
HC sharp2.0_all   0.50 0.00  | median   7   2

E4 transfer function (HC, needle share × m at all layers): median emitted (exact)
K=4: m=0.5:4(0.50) m=1:4(1.00) m=2:4(0.50) m=3:5(0.00) m=4:4(0.50)
K=8: m=0.5:3(0.00) m=1:6(0.50) m=2:8(0.00) m=3:9(0.00) m=4:9(0.00)
