# V9 fixed synthetic response comparison

432 fixed synthetic first-token points; zero fitting and zero VLM calls. This is a mechanism diagnostic, not held-out accuracy.

| Condition | Seed | Question | N16 first-token correct | N64 first-token correct |
|---|---:|---|---:|---:|
| path | 12 | How many frames show Daniel in the Bathroom? | 27/27 | 11/27 |
| path | 12 | How many frames show Daniel in the Bedroom? | 27/27 | 21/27 |
| path | 13 | How many frames show Daniel in the Bathroom? | 27/27 | 11/27 |
| path | 13 | How many frames show Daniel in the Bedroom? | 27/27 | 13/27 |
| residual | 12 | How many frames show Daniel in the Bathroom? | 27/27 | 16/27 |
| residual | 12 | How many frames show Daniel in the Bedroom? | 27/27 | 12/27 |
| residual | 13 | How many frames show Daniel in the Bathroom? | 27/27 | 12/27 |
| residual | 13 | How many frames show Daniel in the Bedroom? | 27/27 | 17/27 |

The same training-only local prototypes, native head and response computation are used for every model. Successful synthetic N64 decoding does not establish actual N64 generalization or remove the later-Step distribution shift.

[All raw predictions, geometry, floating-point checks and input identities](analysis.json).
