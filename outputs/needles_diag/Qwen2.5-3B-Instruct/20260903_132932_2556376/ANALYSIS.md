# Bundle A — Qwen/Qwen2.5-3B-Instruct L27 n=40 (outputs/needles_diag/Qwen2.5-3B-Instruct/20260903_132932_2556376)

## H0
emission exact by K: base: 1:0.90 2:1.00 3:1.00 4:0.80 6:0.38 8:0.55 12:0.40 16:0.12 24:0.07 32:0.00 | time: 1:1.00 2:1.00 3:0.93 4:0.53 6:0.65 8:0.62 12:0.40 16:0.25 24:0.03 32:0.00 | number: 1:0.95 2:1.00 3:0.88 4:0.47 6:0.45 8:0.25 12:0.78 16:0.25 24:0.88 32:0.38 | fp32: 1:0.90 2:1.00 3:1.00 4:0.80 6:0.35 8:0.50 12:0.45 16:0.12 24:0.03 32:0.00
A1 depth 13: needle→rank: exact 0.97 (≤16: 1.00) within1 1.00 n=4320 | last needle→K: exact 0.85 (≤16: 0.96) within1 0.97 n=400 | tail→K: exact 0.87 (≤16: 0.96) within1 0.99 n=400 | answer→K: exact 0.94 (≤16: 0.99) within1 1.00 n=400
A1 depth 27: needle→rank: exact 0.82 (≤16: 0.93) within1 0.97 n=4320 | last needle→K: exact 0.69 (≤16: 0.81) within1 0.88 n=400 | tail→K: exact 0.67 (≤16: 0.78) within1 0.90 n=400 | answer→K: exact 0.81 (≤16: 0.91) within1 0.95 n=400
A1 depth 33: needle→rank: exact 0.83 (≤16: 0.92) within1 0.98 n=4320 | last needle→K: exact 0.70 (≤16: 0.81) within1 0.91 n=400 | tail→K: exact 0.69 (≤16: 0.80) within1 0.92 n=400 | answer→K: exact 0.74 (≤16: 0.84) within1 0.94 n=400
A1 depth fin: needle→rank: exact 0.77 (≤16: 0.89) within1 0.96 n=4320 | last needle→K: exact 0.66 (≤16: 0.77) within1 0.89 n=400 | tail→K: exact 0.57 (≤16: 0.67) within1 0.83 n=400 | answer→K: exact 0.71 (≤16: 0.80) within1 0.92 n=400
A4 |q|        by K: 1:31.76 2:33.38 3:33.01 4:32.35 6:32.11 8:31.92 12:31.35 16:31.57 24:31.30 32:31.32
A4 s_needle   by K: 1:-5.78 2:-7.92 3:-8.84 4:-9.47 6:-10.42 8:-10.88 12:-11.44 16:-11.80 24:-12.12 32:-13.06
A4 s_junk     by K: 1:nan 2:nan 3:nan 4:nan 6:nan 8:nan 12:nan 16:nan 24:nan 32:nan
A4 cos_needle by K: 1:0.20 2:0.18 3:0.17 4:0.16 6:0.16 8:0.15 12:0.14 16:0.14 24:0.13 32:0.13
A4 |k_needle| by K: 1:44.20 2:44.60 3:44.76 4:44.75 6:44.88 8:44.94 12:44.95 16:44.98 24:45.00 32:45.11
A4 cos_junk   by K: 1:nan 2:nan 3:nan 4:nan 6:nan 8:nan 12:nan 16:nan 24:nan 32:nan

## HC
emission exact by K: base: 1:0.05 2:0.35 3:0.17 4:0.62 6:0.57 8:0.12 12:0.03 16:0.00 24:0.00 32:0.00 | time: 1:0.00 2:0.42 3:0.17 4:0.62 6:0.25 8:0.05 12:0.00 16:0.00 24:0.00 32:0.00 | number: 1:0.25 2:0.75 3:0.23 4:0.35 6:0.05 8:0.00 12:0.07 16:0.00 24:0.03 32:0.00 | fp32: 1:0.05 2:0.33 3:0.12 4:0.65 6:0.57 8:0.23 12:0.05 16:0.00 24:0.00 32:0.00
A1 depth 13: needle→rank: exact 0.26 (≤16: 0.30) within1 0.59 n=4320 | distractor→needles before: exact 0.19 (≤16: 0.20) within1 0.46 n=3200 | last needle→K: exact 0.10 (≤16: 0.12) within1 0.36 n=400 | tail→K: exact 0.12 (≤16: 0.14) within1 0.33 n=400 | answer→K: exact 0.15 (≤16: 0.18) within1 0.37 n=400
A1 depth 27: needle→rank: exact 0.27 (≤16: 0.33) within1 0.62 n=4320 | distractor→needles before: exact 0.18 (≤16: 0.19) within1 0.46 n=3200 | last needle→K: exact 0.22 (≤16: 0.25) within1 0.47 n=400 | tail→K: exact 0.17 (≤16: 0.21) within1 0.41 n=400 | answer→K: exact 0.24 (≤16: 0.28) within1 0.58 n=400
A1 depth 33: needle→rank: exact 0.25 (≤16: 0.30) within1 0.59 n=4320 | distractor→needles before: exact 0.19 (≤16: 0.20) within1 0.49 n=3200 | last needle→K: exact 0.16 (≤16: 0.20) within1 0.44 n=400 | tail→K: exact 0.18 (≤16: 0.21) within1 0.46 n=400 | answer→K: exact 0.28 (≤16: 0.33) within1 0.65 n=400
A1 depth fin: needle→rank: exact 0.23 (≤16: 0.27) within1 0.56 n=4320 | distractor→needles before: exact 0.19 (≤16: 0.20) within1 0.48 n=3200 | last needle→K: exact 0.17 (≤16: 0.18) within1 0.47 n=400 | tail→K: exact 0.14 (≤16: 0.17) within1 0.42 n=400 | answer→K: exact 0.29 (≤16: 0.36) within1 0.60 n=400
A4 |q|        by K: 1:31.44 2:31.61 3:31.71 4:31.67 6:31.35 8:31.44 12:31.11 16:31.17 24:30.80 32:30.69
A4 s_needle   by K: 1:-10.33 2:-10.32 3:-10.88 4:-10.60 6:-10.88 8:-10.87 12:-12.14 16:-12.38 24:-13.49 32:-13.95
A4 s_junk     by K: 1:-12.81 2:-13.49 3:-13.84 4:-13.89 6:-13.80 8:-13.73 12:-14.58 16:-14.76 24:-15.38 32:-15.60
A4 cos_needle by K: 1:0.14 2:0.14 3:0.14 4:0.14 6:0.14 8:0.14 12:0.13 16:0.13 24:0.12 32:0.12
A4 |k_needle| by K: 1:45.31 2:45.20 3:45.08 4:45.14 6:45.12 8:45.28 12:45.28 16:45.28 24:45.24 32:45.31
A4 cos_junk   by K: 1:0.08 2:0.07 3:0.07 4:0.07 6:0.08 8:0.08 12:0.08 16:0.08 24:0.08 32:0.08
