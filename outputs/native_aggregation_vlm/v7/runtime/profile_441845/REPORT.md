# V7 native runtime software profile

Computational integrity and zero-U identity passed. Global numerical failures: 0/16; all-row failures: 0/336.

| Arm | N | Condition | Tokens | Generation seconds | Prefix replay seconds |
|---|---:|---|---:|---:|---:|
| parallel | 16 | native | 2 | 1.101 | 0.000 |
| parallel | 16 | zero | 2 | 0.605 | 1.368 |
| parallel | 16 | active | 2 | 0.589 | 1.368 |
| parallel | 64 | native | 2 | 1.978 | 0.000 |
| parallel | 64 | zero | 2 | 1.955 | 4.941 |
| parallel | 64 | active | 2 | 1.945 | 5.065 |
| joint | 16 | native | 2 | 0.506 | 0.000 |
| joint | 16 | zero | 2 | 0.491 | 1.179 |
| joint | 16 | active | 2 | 0.493 | 1.212 |
| joint | 64 | native | 2 | 1.701 | 0.000 |
| joint | 64 | zero | 2 | 1.687 | 4.317 |
| joint | 64 | active | 2 | 1.693 | 4.456 |

No aggregation accuracy was measured. All numerical failures and the original mixed/cache failure remain recorded.
