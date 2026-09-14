# V8 fixed synthetic response comparison

432 fixed synthetic first-token points; zero fitting and zero VLM calls. This is a mechanism diagnostic, not held-out accuracy.

| Condition | Seed | Question | N16 first-token correct | N64 first-token correct |
|---|---:|---|---:|---:|
| ce | 10 | How many frames show Daniel in the Bathroom? | 27/27 | 3/27 |
| ce | 10 | How many frames show Daniel in the Bedroom? | 27/27 | 3/27 |
| ce | 11 | How many frames show Daniel in the Bathroom? | 27/27 | 5/27 |
| ce | 11 | How many frames show Daniel in the Bedroom? | 27/27 | 6/27 |
| consistency | 10 | How many frames show Daniel in the Bathroom? | 27/27 | 15/27 |
| consistency | 10 | How many frames show Daniel in the Bedroom? | 27/27 | 17/27 |
| consistency | 11 | How many frames show Daniel in the Bathroom? | 27/27 | 15/27 |
| consistency | 11 | How many frames show Daniel in the Bedroom? | 27/27 | 15/27 |

The same training-only local prototypes, native head and response computation are used for every model. Successful synthetic N64 decoding does not establish actual N64 generalization or remove the later-Step distribution shift.

[All raw predictions, geometry, floating-point checks and input identities](analysis.json).
