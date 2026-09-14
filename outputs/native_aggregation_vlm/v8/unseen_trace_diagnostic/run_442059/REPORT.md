# V8 unseen-count natural traces

All 512 original unseen-count outputs retained; no new model calls.

| Condition | Seed | N | K | n | First token correct | Whole + EOS | Correct first, immediate EOS | Wrong first | Malformed | Truncated |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| ce | 10 | 32 | K9 | 8 | 0 | 0 | 0 | 8 | 0 | 0 |
| ce | 10 | 32 | K10_16 | 56 | 0 | 0 | 0 | 56 | 0 | 0 |
| ce | 10 | 64 | K9 | 8 | 0 | 0 | 0 | 8 | 8 | 0 |
| ce | 10 | 64 | K10_16 | 56 | 0 | 0 | 0 | 56 | 12 | 0 |
| ce | 11 | 32 | K9 | 8 | 0 | 0 | 0 | 8 | 0 | 0 |
| ce | 11 | 32 | K10_16 | 56 | 0 | 0 | 0 | 56 | 0 | 0 |
| ce | 11 | 64 | K9 | 8 | 0 | 0 | 0 | 8 | 1 | 0 |
| ce | 11 | 64 | K10_16 | 56 | 0 | 0 | 0 | 56 | 1 | 0 |
| consistency | 10 | 32 | K9 | 8 | 0 | 0 | 0 | 8 | 0 | 0 |
| consistency | 10 | 32 | K10_16 | 56 | 0 | 0 | 0 | 56 | 0 | 0 |
| consistency | 10 | 64 | K9 | 8 | 0 | 0 | 0 | 8 | 0 | 0 |
| consistency | 10 | 64 | K10_16 | 56 | 0 | 0 | 0 | 56 | 0 | 0 |
| consistency | 11 | 32 | K9 | 8 | 0 | 0 | 0 | 8 | 0 | 0 |
| consistency | 11 | 32 | K10_16 | 56 | 0 | 0 | 0 | 56 | 0 | 0 |
| consistency | 11 | 64 | K9 | 8 | 0 | 0 | 0 | 8 | 0 | 0 |
| consistency | 11 | 64 | K10_16 | 56 | 0 | 0 | 0 | 56 | 0 | 0 |

Each count uses the stated complete group denominator. Immediate EOS overlaps correct answers at K9.

[All traces](traces.json) · [Histograms and conditional denominators](groups.json) · [Provenance](plan.json)

- First generated token is compared directly; no search for a later numeral.
- K9 immediate EOS can be correct; K10--16 immediate EOS after the first token is incomplete.
- Metrics overlap; malformed-or-truncated is a union and all rates use the full group denominator.
- Numeral histograms include parseable truncated outputs; completed histograms are also retained.
- Already verified natural outputs only; this does not prove presence or absence of latent count information.
- No new model call, prediction, fitted decoder, oracle prefix, or reasoning-composition claim.
