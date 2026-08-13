# Stage-4 BLOCKED items

## E-B — SFT-control long-N eval (2026-07-20)

- 124484 COMPLETED: **test_iid (train length N≤8) acc 0.9984** (630 items, best_epoch=1,
  bias +0.002) → `outputs/ladder/image_longN/sft_control_le8/20260719_185022_lora/`
  (summary.csv/predictions.csv). At train length the 23.8M-param plain LoRA matches our
  2M-param carrier layer (0.999/1.000) — the control is only informative at LONG N.
- **BLOCKED: `lora_sft_baseline.py` does not save the adapter** (writes CSVs only — verified
  by grep and by the run dir contents). The trained weights are gone; the N=16/32/64 eval
  cannot run without a ~4-5h retrain + save-code change. Per the brief's 2h timebox → logged
  with train-length numbers only. If Tal wants the cell: add `model.save_pretrained()` after
  the train loop, EPOCHS=2 (best was ep1), then a generate-and-parse pass at N=16/32/64.
- The theory-side prediction for this cell (collapse at N≥32, joint supply d′≈2) is already
  independently supported (frozen 0.219→0.097 at N=32; joint-context tax memory).
