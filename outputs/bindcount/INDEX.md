# outputs/bindcount — bind, then count (text MMRED + needle stories), Qwen2.5-7B frozen

Script `outputs/_scratch/dbg/train_bindcount.py`; all arms trained on N ≤ 32 (1,200 samples), evaluated n = 100 per cell. Numbers = exact count.

| arm | run | N=8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | needles 64/128/256 |
|---|---|---|---|---|---|---|---|---|---|---|
| sum (fence all layers + pos reset + verdict + sum) | `sum/20260903_160058_2610047` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 / 1.00 |
| sum_nofence (carriers, full attention) | `sum_nofence/20260903_160058_2610312` | .96 | .97 | .91 | .81 | .62 | .56 | .44 | .30 | .37 / .10 / .05 |
| mean_lora (fence < L12, LoRA r8 ≥ 12, digit read) | `mean_lora/20260903_161756_2615783` | 1.00 | .98 | .94 | .60 | .26 | .10 | .09 | .09 | .00 / .02 / .02 |
| digit_lora (plain prompt, LoRA r8 ≥ 12) | `digit_lora/20260903_161755_2615784` | 1.00 | .99 | .92 | .63 | .32 | .37 | .28 | .25 | .11 / .05 / .02 |
| sum_noreset (fence, NATIVE positions) | `sum_noreset/20260903_170524_2630656` | 1.00 | 1.00 | 1.00 | 1.00 | .98 | .63 | .32 | .23 | .95 / .90 / .70 |
| sum on Qwen2.5-3B (`outputs/bindcount_q3/`) | `sum/20260903_170522_2631186` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 / .99 / 1.00 |
| sum on Llama-3.1-8B (`outputs/bindcount_ll/`) | `sum/20260903_171237_2633566` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 / 1.00 |
| sum_multi (+7-way room head; 6 set functions) | `sum_multi/20260903_162835_2621052` | count 1.00 at every N; exists/first/last 1.00; distinct rooms 1.0 → .71 (N=1024); most room 1.0 → .82 |

Invalid: `mean_lora/20260903_160059_2610319` (lr 2e-3, did not train). Frozen Qwen2.5-7B reference: outputs/mmred_frozen/INDEX.md (full .59/.39/.345/.31/.24/.145/.16/.15).
