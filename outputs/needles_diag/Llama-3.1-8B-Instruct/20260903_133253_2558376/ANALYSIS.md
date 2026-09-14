# Bundle A — meta-llama/Llama-3.1-8B-Instruct L20 n=40 (outputs/needles_diag/Llama-3.1-8B-Instruct/20260903_133253_2558376)

## H0
emission exact by K: base: 1:1.00 2:1.00 3:1.00 4:0.80 6:0.42 8:0.65 12:0.53 16:0.00 24:0.00 32:0.00 | time: 1:1.00 2:1.00 3:0.95 4:0.47 6:0.23 8:0.42 12:0.38 16:0.00 24:0.00 32:0.03 | number: 1:1.00 2:1.00 3:0.95 4:0.42 6:0.38 8:0.55 12:0.38 16:0.15 24:0.68 32:0.42 | fp32: 1:1.00 2:1.00 3:1.00 4:0.78 6:0.42 8:0.62 12:0.53 16:0.00 24:0.00 32:0.00
A1 depth 10: needle→rank: exact 0.92 (≤16: 0.99) within1 1.00 n=4320 | last needle→K: exact 0.77 (≤16: 0.89) within1 0.93 n=400 | tail→K: exact 0.82 (≤16: 0.90) within1 0.98 n=400 | answer→K: exact 0.95 (≤16: 0.99) within1 1.00 n=400
A1 depth 20: needle→rank: exact 0.87 (≤16: 0.96) within1 0.99 n=4320 | last needle→K: exact 0.74 (≤16: 0.85) within1 0.91 n=400 | tail→K: exact 0.65 (≤16: 0.75) within1 0.86 n=400 | answer→K: exact 0.77 (≤16: 0.87) within1 0.95 n=400
A1 depth 26: needle→rank: exact 0.87 (≤16: 0.97) within1 0.99 n=4320 | last needle→K: exact 0.73 (≤16: 0.86) within1 0.91 n=400 | tail→K: exact 0.64 (≤16: 0.72) within1 0.88 n=400 | answer→K: exact 0.74 (≤16: 0.87) within1 0.93 n=400
A1 depth fin: needle→rank: exact 0.84 (≤16: 0.94) within1 0.98 n=4320 | last needle→K: exact 0.69 (≤16: 0.81) within1 0.90 n=400 | tail→K: exact 0.56 (≤16: 0.64) within1 0.84 n=400 | answer→K: exact 0.74 (≤16: 0.88) within1 0.94 n=400
A4 |q|        by K: 1:11.96 2:12.00 3:12.01 4:11.95 6:12.06 8:12.06 12:12.09 16:12.15 24:12.15 32:12.35
A4 s_needle   by K: 1:-4.78 2:-5.21 3:-5.29 4:-5.26 6:-5.48 8:-5.69 12:-6.11 16:-6.35 24:-6.63 32:-6.92
A4 s_junk     by K: 1:nan 2:nan 3:nan 4:nan 6:nan 8:nan 12:nan 16:nan 24:nan 32:nan
A4 cos_needle by K: 1:-0.20 2:-0.21 3:-0.21 4:-0.21 6:-0.22 8:-0.22 12:-0.24 16:-0.24 24:-0.25 32:-0.25
A4 |k_needle| by K: 1:22.56 2:23.29 3:23.56 4:23.61 6:23.91 8:24.09 12:24.46 16:24.68 24:25.06 32:25.38
A4 cos_junk   by K: 1:nan 2:nan 3:nan 4:nan 6:nan 8:nan 12:nan 16:nan 24:nan 32:nan

## HC
emission exact by K: base: 1:0.25 2:0.28 3:0.05 4:0.00 6:0.00 8:0.07 12:0.55 16:0.00 24:0.30 32:0.05 | time: 1:0.12 2:0.33 3:0.10 4:0.07 6:0.03 8:0.42 12:0.40 16:0.00 24:0.17 32:0.00 | number: 1:0.10 2:0.15 3:0.05 4:0.00 6:0.03 8:0.15 12:0.20 16:0.15 24:0.17 32:0.05 | fp32: 1:0.28 2:0.25 3:0.05 4:0.00 6:0.00 8:0.05 12:0.50 16:0.00 24:0.30 32:0.05
A1 depth 10: needle→rank: exact 0.29 (≤16: 0.33) within1 0.63 n=4320 | distractor→needles before: exact 0.17 (≤16: 0.19) within1 0.43 n=3200 | last needle→K: exact 0.18 (≤16: 0.21) within1 0.53 n=400 | tail→K: exact 0.12 (≤16: 0.14) within1 0.35 n=400 | answer→K: exact 0.17 (≤16: 0.20) within1 0.46 n=400
A1 depth 20: needle→rank: exact 0.35 (≤16: 0.41) within1 0.70 n=4320 | distractor→needles before: exact 0.20 (≤16: 0.22) within1 0.49 n=3200 | last needle→K: exact 0.25 (≤16: 0.30) within1 0.60 n=400 | tail→K: exact 0.14 (≤16: 0.16) within1 0.42 n=400 | answer→K: exact 0.33 (≤16: 0.38) within1 0.70 n=400
A1 depth 26: needle→rank: exact 0.31 (≤16: 0.37) within1 0.64 n=4320 | distractor→needles before: exact 0.19 (≤16: 0.21) within1 0.49 n=3200 | last needle→K: exact 0.22 (≤16: 0.25) within1 0.55 n=400 | tail→K: exact 0.15 (≤16: 0.18) within1 0.49 n=400 | answer→K: exact 0.30 (≤16: 0.34) within1 0.68 n=400
A1 depth fin: needle→rank: exact 0.29 (≤16: 0.35) within1 0.64 n=4320 | distractor→needles before: exact 0.21 (≤16: 0.22) within1 0.49 n=3200 | last needle→K: exact 0.21 (≤16: 0.25) within1 0.51 n=400 | tail→K: exact 0.16 (≤16: 0.18) within1 0.40 n=400 | answer→K: exact 0.30 (≤16: 0.36) within1 0.68 n=400
A4 |q|        by K: 1:12.17 2:12.09 3:12.06 4:12.07 6:12.04 8:12.02 12:12.02 16:12.04 24:12.03 32:12.12
A4 s_needle   by K: 1:-7.07 2:-7.26 3:-7.43 4:-7.34 6:-7.41 8:-7.44 12:-7.49 16:-7.51 24:-7.54 32:-7.60
A4 s_junk     by K: 1:-8.02 2:-8.12 3:-8.21 4:-8.18 6:-8.18 8:-8.19 12:-8.21 16:-8.17 24:-8.09 32:-8.08
A4 cos_needle by K: 1:-0.27 2:-0.27 3:-0.28 4:-0.28 6:-0.28 8:-0.28 12:-0.28 16:-0.28 24:-0.28 32:-0.28
A4 |k_needle| by K: 1:25.03 2:25.09 3:25.05 4:25.05 6:25.31 8:25.33 12:25.35 16:25.55 24:25.71 32:25.94
A4 cos_junk   by K: 1:-0.30 2:-0.31 3:-0.31 4:-0.31 6:-0.31 8:-0.31 12:-0.31 16:-0.31 24:-0.31 32:-0.31
