# scripts/learnmask/

Code for the learned-discrete-mask campaign (mirrors outputs/learnmask/, see
outputs/learnmask/CAMPAIGN_BRIEF.md). Planned entrypoints:

- `train_mask_gates.py` — trains gate logits (relation × layer) with task CE,
  frozen backbone + frozen carriers; estimator flag: ste | soft | st-gumbel | hc.
- `eval_mask_transfer.py` — assembles a trained gate set at arbitrary N (zero-shot
  length transfer) and runs the exam + audits.

Nothing implemented yet (P0 pending).
