# outputs/bindcount_vlm — bind, then count on MMRED VISION, Qwen2.5-VL-7B (4-bit, frozen)

Script `outputs/_scratch/dbg/train_bindcount_vlm.py`; trained on vfiltered N=8/32 train splits (450 samples, 2 epochs); test splits n = 100. Numbers = exact count (per-frame verdict accuracy).

| arm | run | N=8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| sum (fence all layers + M-RoPE reset + verdict + sum) | `sum/20260903_162936_2622612` | .99 (.9988) | .99 (.9994) | 1.00 (1.00) | .99 (.9998) |
| sum_nofence (carriers, full attention, native pos) | `sum_nofence/20260903_162936_2622613` | .60 (.906) | .33 (.918) | .33 (.947) | .27 (.962) |
| frozen VLM full / oracle / judge (outputs/mmred_vfrozen) | — | .40 / .80 / .45 | .28 / .725 / .41 | .20 / .65 / .385 | .215 / .765 / .36 |
