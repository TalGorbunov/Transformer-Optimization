# Scratchpad-FORMAT sweep STATE (overwrite at every transition)

Updated: 2026-07-24 (collection ADOPTED by the p0p2 campaign agent — original agent stale
since 13:29 2026-07-23; its jobs were never touched. **ALL PRIMARY BANDS DECIDED; WINNER
NAMED = C (caption).**)

## Verdict (bands from `plans/scratchpad_format_PREREG.md`, decided 2026-07-24)

| band | verdict |
|---|---|
| 1 in-dist sanity ≥0.90 ×4 | MET (A/B/C 1.000, D 0.987) |
| 2 B vs A @N=64 cap-adj ≥0.761 | **GO — B 0.956** (43/45; raw 0.942, pf 0.019) |
| 3 C vs B parity | **GO** — in-dist/rooms exact parity; N=32 −0.013; N=64 +0.039 (C better) |
| 4 D chunking @N=64 | **NO** — 0.615 < max(A 0.711, B 0.956) |
| 5 rooms ordering | parity only (control A also 1.000; the 0.84 gap was the L17 ckpt) |

**WINNER = C (caption)** — hardest cell 0.981 (pf 0) vs B 0.942 (pf 0.019), parity
elsewhere, and the caption text is the agnosticism bet. Ckpt:
`outputs/ladder/image_longN/carrier_fmt_caption/20260722_222032_L12_r8/carrier_layer_best.pt`.

## Final table (greedy exams, identical dirs-files across arms)

| cell | A poslist | B scan | C caption | D chunked |
|---|---|---|---|---|
| in-dist-150 | 1.000 | 1.000 | 1.000 | 0.987 |
| rooms-100 | 1.000 | 1.000 | 1.000 | 0.920 |
| N=32 (150) | 0.953 | **1.000** (125194) | 0.987 (125195) | 0.907 (125185) |
| N=48 (109) | 0.789 / 0.878 cap-adj | **0.982** (125196) | 0.972 (125197) | 0.679 (125186) |
| N=64 (52) | 0.615 / 0.711 cap-adj | 0.942 / 0.956 cap-adj (125198) | **0.981** (125199) | 0.615 (125187) |
| pf worst | 0.135 (N=64) | 0.019 (N=64) | **0.000 everywhere** | 0.000 everywhere |

Run dirs: `fmt{B,C,D}_eval_*` + `tallyL12v2_eval_*` siblings under
`outputs/ladder/image_longN/`. NOTE: the previous STATE's job→cell mapping (125185-87 =
D, 125194/95 = B/C N32, 125196-99 = B/C N48/N64) was approximate — actual cells were
identified from each report header (125198 = B N=64, 125199 = C N=64; 125196/97 = the
N=48 pair, still running on 4d_1g, secondary/descriptive, cannot change the bands).

## Remaining

- NOTHING — sweep CLOSED 2026-07-24 (N=48 pair collected: B 0.982 / C 0.972, pf 0 both;
  final migration entry written).
- Downstream (p0p2): seed retrains ×2 of C + LOTO + MLVU launched 2026-07-24
  (125347/48/49/50) — tracked in `plans/p0p2_STATE.md`, not here.
