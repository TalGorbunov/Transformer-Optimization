# Frozen V2 native-read linear recoverability

Alpha selected by development MAE only. Predictions are continuous; h+r concatenates the two channels.

| Feature | alpha | Dev MAE | Familiar N32/64 MAE | K9 MAE | K10–16 MAE |
|---|---:|---:|---:|---:|---:|
| h | 1000.0 | 1.0417 | 2.1886 | 3.0403 | 6.5616 |
| r | 1000.0 | 1.0383 | 2.1535 | 4.3825 | 7.5832 |
| h+r | 1000.0 | 1.0064 | 2.0652 | 3.9929 | 7.2701 |
| h+N | 1000.0 | 1.0417 | 2.1885 | 3.0405 | 6.5618 |
| r+N | 1000.0 | 1.0383 | 2.1534 | 4.3818 | 7.5825 |
| h+r+N | 1000.0 | 1.0064 | 2.0651 | 3.9935 | 7.2706 |
| N_only | 1000.0 | 2.2222 | 2.2222 | 5.0000 | 9.0000 |
| train_mean | None | 2.2222 | 2.2222 | 5.0000 | 9.0000 |

R-squared and rounded-integer exact are in results.json, including every original split/N and per-K cell.
No fitted weights are saved. Low probe accuracy does not establish that information is absent.
