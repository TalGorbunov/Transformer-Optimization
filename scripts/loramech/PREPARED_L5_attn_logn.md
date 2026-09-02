# PREPARED (not applied) — L5 delta: `--attn-logn-scale` on the SFT eval path

**Why not applied yet:** `scripts/train_sft_baseline.py` is on the code path of the
PENDING P0 jobs (SLURM reads the script at job START, not submit) — applying now would
change what the trainers run. Apply AFTER both P0 trainers have started/finished, and
only after Tal's H0 review green-lights L5.

**Spec (CAMPAIGN_BRIEF L5, Chiang–Cholak remedy):** at eval, set on every LM decoder
attention module `module.scaling = head_dim**-0.5 * log(S)/log(S_ref)` where S = the
sample's text sequence length and S_ref = the N=32 sequence length; vision tower
untouched. **VERIFIED 2026-08-26 (this venv, transformers 4.57.6):**
`Qwen2_5_VLAttention.__init__` sets `self.scaling = self.head_dim**-0.5` and `forward`
passes `scaling=self.scaling` into the attention-interface call — setting the attribute
per-sample works exactly as the brief assumes.

## Patch to `scripts/train_sft_baseline.py`

1. argparse (next to `--frames-first`):

```python
    ap.add_argument("--attn-logn-sref", type=int, default=0,
                    help="L5: reference seq len S_ref; >0 enables log-N attention "
                         "scaling (head_dim^-0.5 * ln(S)/ln(S_ref)) on LM decoder "
                         "attention at eval; vision tower untouched")
```

2. after model load (any branch), collect the LM decoder attention modules once:

```python
    if args.attn_logn_sref > 0:
        from gnnformer.runtime import attention_dims, get_layers
        _attn_mods = [ly.self_attn for ly in get_layers(model)]
        _hd = attention_dims(model)["head_dim"]
        _base_scaling = _attn_mods[0].scaling  # sanity: should equal _hd ** -0.5

        def set_logn_scaling(seq_len):
            import math
            s = (_hd ** -0.5) * math.log(max(seq_len, 2)) / math.log(args.attn_logn_sref)
            for m_ in _attn_mods:
                m_.scaling = s
```

3. in `predict()`, right before `model.generate`:

```python
        if args.attn_logn_sref > 0:
            set_logn_scaling(inp["input_ids"].shape[1])
```

(`train_loss` intentionally NOT touched — L5 is an eval-time intervention on the saved
ff_le32 adapter via `--eval-only-adapter`.)

4. run-dir provenance: `config.json` already dumps `attn_logn_sref` via `vars(args)`.

## Invocation (after OK)

S_ref = the measured N=32 in-length seq len (read it from one exam sample's tokenized
length in the ff_le32 run log, or compute with the processor; brief: "S_ref = N=32 seq
len"). Cells N∈{32,64,128}:

```bash
sbatch -p a100-public --qos=12h_4g --time=04:00:00 --job-name=lm_L5 \
  --export="ALL,ADAPTER=checkpoints/sft_ff_le32_adapter,FRAMES_FIRST=1,DIRS_FILES=outputs/loramech/examdirs/exam_ff_N32.txt outputs/loramech/examdirs/exam_ff_N64.txt outputs/loramech/examdirs/exam_ff_N128.txt,LONGN_LIMIT=150,EXTRA_SREF=<S_ref>,OUTPUT=outputs/loramech/l5_logn" \
  slurm/train_sft_baseline.sbatch
```
(add an `ATTN_LOGN_SREF` knob to the wrapper together with the patch;
H3 bands: N=64 ≥ +0.10 abs AND N=32 within −0.02 → GAIN confirmed + new rung;
N=32 drop > 0.05 → harmful, report.)
