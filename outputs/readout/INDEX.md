# outputs/readout/ — readout-injection experiment group (plan 2026-07-08, workstream C)

| experiment | canonical run | headline |
|---|---|---|
| C1b phrasing-robustness grid (job 120012) | `c1b_phrasing/20260710_215647/` | **interface gated on predicate-match + pre-question position**; within the gate: words 1.000 OOD, source-attr 0.992, max-distance 0.992, distractor 0.967; paraphrase/post-question **0.000** (RESULTS [2026-07-10d]) |
| C2/C3 codebook injection table (120027/120040/120060) | `c2_digit_codebook/{20260710_220702,20260710_225300,20260710_233141}/` | **token interface NECESSARY**: token 1.000/1.000 @0 params; digit 0.808/0.098; count (C-control) 0.808/0.000; fourier 1.000/0.000; fourierE (native-anchored) 1.000/0.000 (RESULTS [2026-07-11b]) |

C1 (original token-interface smoke) lives in `outputs/_scratch/c1_token_interface/` (RESULTS [2026-07-08b]).
| C2 residual-level injection (120233) | `c2_residual/20260711_181926/` | **all learned routes fail OOD at the residual level too** (digit .750/.080, count 1.000/.018, Fourier .923/.000, fourierE 1.000/.000) — token-necessity at both injection levels (RESULTS [2026-07-11v]) |
