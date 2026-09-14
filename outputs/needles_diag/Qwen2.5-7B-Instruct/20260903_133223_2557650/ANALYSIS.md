# Bundle A — Qwen/Qwen2.5-7B-Instruct L22 n=40 (outputs/needles_diag/Qwen2.5-7B-Instruct/20260903_133223_2557650)

## H0
emission exact by K: base: 1:1.00 2:1.00 3:0.97 4:0.80 6:0.03 8:0.07 12:0.00 16:0.00 24:0.00 32:0.00 | time: 1:1.00 2:1.00 3:1.00 4:0.53 6:0.03 8:0.05 12:0.00 16:0.00 24:0.00 32:0.00 | number: 1:1.00 2:1.00 3:1.00 4:0.40 6:0.00 8:0.00 12:0.00 16:0.00 24:0.00 32:0.00 | fp32: 1:1.00 2:1.00 3:1.00 4:0.80 6:0.03 8:0.05 12:0.00 16:0.00 24:0.00 32:0.00
A1 depth 11: needle→rank: exact 0.95 (≤16: 1.00) within1 1.00 n=4320 | last needle→K: exact 0.80 (≤16: 0.90) within1 0.96 n=400 | tail→K: exact 0.86 (≤16: 0.94) within1 0.99 n=400 | answer→K: exact 0.91 (≤16: 0.98) within1 1.00 n=400
A1 depth 22: needle→rank: exact 0.85 (≤16: 0.94) within1 0.98 n=4320 | last needle→K: exact 0.68 (≤16: 0.79) within1 0.91 n=400 | tail→K: exact 0.65 (≤16: 0.75) within1 0.86 n=400 | answer→K: exact 0.80 (≤16: 0.91) within1 0.96 n=400
A1 depth 27: needle→rank: exact 0.84 (≤16: 0.94) within1 0.99 n=4320 | last needle→K: exact 0.68 (≤16: 0.78) within1 0.91 n=400 | tail→K: exact 0.62 (≤16: 0.74) within1 0.89 n=400 | answer→K: exact 0.78 (≤16: 0.90) within1 0.95 n=400
A1 depth fin: needle→rank: exact 0.81 (≤16: 0.92) within1 0.97 n=4320 | last needle→K: exact 0.65 (≤16: 0.76) within1 0.88 n=400 | tail→K: exact 0.58 (≤16: 0.69) within1 0.82 n=400 | answer→K: exact 0.79 (≤16: 0.89) within1 0.94 n=400
A4 |q|        by K: 1:17.53 2:18.12 3:18.13 4:17.95 6:18.05 8:18.04 12:17.88 16:17.94 24:17.79 32:17.75
A4 s_needle   by K: 1:-4.19 2:-4.79 3:-4.74 4:-4.81 6:-4.87 8:-4.97 12:-5.15 16:-5.36 24:-5.78 32:-6.10
A4 s_junk     by K: 1:nan 2:nan 3:nan 4:nan 6:nan 8:nan 12:nan 16:nan 24:nan 32:nan
A4 cos_needle by K: 1:-0.14 2:-0.15 3:-0.15 4:-0.15 6:-0.15 8:-0.15 12:-0.16 16:-0.16 24:-0.17 32:-0.18
A4 |k_needle| by K: 1:19.58 2:20.10 3:20.42 4:20.56 6:20.95 8:21.31 12:21.63 16:21.99 24:22.37 32:22.69
A4 cos_junk   by K: 1:nan 2:nan 3:nan 4:nan 6:nan 8:nan 12:nan 16:nan 24:nan 32:nan

## HC
emission exact by K: base: 1:0.03 2:0.25 3:0.33 4:0.72 6:0.00 8:0.00 12:0.00 16:0.00 24:0.00 32:0.00 | time: 1:0.00 2:0.57 3:0.40 4:0.28 6:0.00 8:0.00 12:0.00 16:0.00 24:0.00 32:0.00 | number: 1:0.03 2:0.38 3:0.40 4:0.45 6:0.00 8:0.00 12:0.00 16:0.00 24:0.00 32:0.00 | fp32: 1:0.03 2:0.25 3:0.35 4:0.72 6:0.00 8:0.00 12:0.00 16:0.00 24:0.00 32:0.00
A1 depth 11: needle→rank: exact 0.25 (≤16: 0.29) within1 0.59 n=4320 | distractor→needles before: exact 0.18 (≤16: 0.19) within1 0.45 n=3200 | last needle→K: exact 0.14 (≤16: 0.16) within1 0.39 n=400 | tail→K: exact 0.11 (≤16: 0.13) within1 0.34 n=400 | answer→K: exact 0.18 (≤16: 0.21) within1 0.42 n=400
A1 depth 22: needle→rank: exact 0.29 (≤16: 0.34) within1 0.63 n=4320 | distractor→needles before: exact 0.18 (≤16: 0.19) within1 0.45 n=3200 | last needle→K: exact 0.24 (≤16: 0.29) within1 0.54 n=400 | tail→K: exact 0.16 (≤16: 0.18) within1 0.41 n=400 | answer→K: exact 0.23 (≤16: 0.27) within1 0.63 n=400
A1 depth 27: needle→rank: exact 0.28 (≤16: 0.33) within1 0.61 n=4320 | distractor→needles before: exact 0.20 (≤16: 0.21) within1 0.48 n=3200 | last needle→K: exact 0.18 (≤16: 0.22) within1 0.48 n=400 | tail→K: exact 0.17 (≤16: 0.21) within1 0.45 n=400 | answer→K: exact 0.26 (≤16: 0.30) within1 0.61 n=400
A1 depth fin: needle→rank: exact 0.25 (≤16: 0.29) within1 0.59 n=4320 | distractor→needles before: exact 0.18 (≤16: 0.19) within1 0.48 n=3200 | last needle→K: exact 0.19 (≤16: 0.22) within1 0.48 n=400 | tail→K: exact 0.17 (≤16: 0.18) within1 0.47 n=400 | answer→K: exact 0.23 (≤16: 0.25) within1 0.62 n=400
A4 |q|        by K: 1:17.89 2:17.84 3:17.86 4:17.77 6:17.68 8:17.87 12:17.64 16:17.55 24:17.50 32:17.50
A4 s_needle   by K: 1:-6.67 2:-6.67 3:-6.70 4:-6.73 6:-6.83 8:-6.75 12:-6.94 16:-7.06 24:-7.15 32:-7.15
A4 s_junk     by K: 1:-8.30 2:-8.33 3:-8.37 4:-8.45 6:-8.54 8:-8.50 12:-8.61 16:-8.60 24:-8.69 32:-8.69
A4 cos_needle by K: 1:-0.19 2:-0.19 3:-0.20 4:-0.20 6:-0.20 8:-0.19 12:-0.20 16:-0.21 24:-0.21 32:-0.21
A4 |k_needle| by K: 1:22.39 2:22.51 3:22.48 4:22.49 6:22.53 8:22.72 12:22.60 16:22.78 24:22.88 32:22.96
A4 cos_junk   by K: 1:-0.24 2:-0.24 3:-0.24 4:-0.25 6:-0.25 8:-0.25 12:-0.25 16:-0.25 24:-0.26 32:-0.26
