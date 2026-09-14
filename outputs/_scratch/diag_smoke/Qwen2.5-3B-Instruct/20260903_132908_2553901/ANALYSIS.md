# Bundle A — Qwen/Qwen2.5-3B-Instruct L27 n=2 (outputs/_scratch/diag_smoke/Qwen2.5-3B-Instruct/20260903_132908_2553901)

## H0
emission exact by K: base: 2:1.00 8:0.50 | time: 2:1.00 8:0.50 | number: 2:1.00 8:0.00 | fp32: 2:1.00 8:0.50
A1 depth 13: needle→rank: exact 0.80 (≤16: 0.80) within1 1.00 n=20
A1 depth 27: needle→rank: exact 0.75 (≤16: 0.75) within1 0.95 n=20
A1 depth 33: needle→rank: exact 0.65 (≤16: 0.65) within1 0.90 n=20
A1 depth fin: needle→rank: exact 0.60 (≤16: 0.60) within1 0.80 n=20
A4 |q|        by K: 2:33.04 8:31.84
A4 s_needle   by K: 2:-7.63 8:-11.13
A4 s_junk     by K: 2:nan 8:nan
A4 cos_needle by K: 2:0.18 8:0.15
A4 |k_needle| by K: 2:44.29 8:45.12
A4 cos_junk   by K: 2:nan 8:nan

## HC
emission exact by K: base: 2:0.00 8:0.00 | time: 2:0.50 8:0.00 | number: 2:1.00 8:0.00 | fp32: 2:0.00 8:0.00
A1 depth 13: needle→rank: exact 0.15 (≤16: 0.15) within1 0.45 n=20 | distractor→needles before: exact 0.16 (≤16: 0.16) within1 0.50 n=32
A1 depth 27: needle→rank: exact 0.40 (≤16: 0.40) within1 0.50 n=20 | distractor→needles before: exact 0.34 (≤16: 0.34) within1 0.56 n=32
A1 depth 33: needle→rank: exact 0.25 (≤16: 0.25) within1 0.45 n=20 | distractor→needles before: exact 0.19 (≤16: 0.19) within1 0.41 n=32
A1 depth fin: needle→rank: exact 0.15 (≤16: 0.15) within1 0.45 n=20 | distractor→needles before: exact 0.16 (≤16: 0.16) within1 0.38 n=32
A4 |q|        by K: 2:31.27 8:31.58
A4 s_needle   by K: 2:-10.91 8:-11.85
A4 s_junk     by K: 2:-14.58 8:-14.99
A4 cos_needle by K: 2:0.14 8:0.13
A4 |k_needle| by K: 2:44.84 8:45.41
A4 cos_junk   by K: 2:0.07 8:0.07
