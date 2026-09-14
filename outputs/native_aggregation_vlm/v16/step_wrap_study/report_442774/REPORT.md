# V16 Step-label input intervention

Independent rescoring and all16 original-image sentinel gates passed. Every wrapped trajectory is retained.

| Model | N | Mean | Original whole answer | Wrapped whole answer | Change |
|---|---:|---|---:|---:|---:|
| centered_s18 | 32 | learned | 16/17 | 13/17 | -3 |
| centered_s18 | 32 | bank | 16/17 | 17/17 | +1 |
| centered_s18 | 64 | learned | 9/17 | 9/17 | +0 |
| centered_s18 | 64 | bank | 3/17 | 14/17 | +11 |
| centered_s19 | 32 | learned | 13/17 | 12/17 | -1 |
| centered_s19 | 32 | bank | 16/17 | 17/17 | +1 |
| centered_s19 | 64 | learned | 7/17 | 10/17 | +3 |
| centered_s19 | 64 | bank | 5/17 | 14/17 | +9 |
| offset_s18 | 32 | learned | 4/17 | 3/17 | -1 |
| offset_s18 | 32 | bank | 4/17 | 3/17 | -1 |
| offset_s18 | 64 | learned | 2/17 | 3/17 | +1 |
| offset_s18 | 64 | bank | 2/17 | 3/17 | +1 |
| offset_s19 | 32 | learned | 6/17 | 6/17 | +0 |
| offset_s19 | 32 | bank | 6/17 | 6/17 | +0 |
| offset_s19 | 64 | learned | 2/17 | 3/17 | +1 |
| offset_s19 | 64 | bank | 2/17 | 3/17 | +1 |

The counting labels and nonfooter image content are held fixed. Cyclic IDs introduce repetitions and therefore change ordinal grammar; no range-specific mechanism or restored general aggregation is identified.

[All K cells, transitions, raw captures, sentinels and allocated cost](analysis.json).
