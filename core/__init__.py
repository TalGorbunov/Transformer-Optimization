"""core — the method, on the official MMReD benchmark only.

A frozen Qwen2.5-VL-7B reads a sequence of N frames. Three pieces:

  fence   every frame is encoded in its own block (block-diagonal attention + per-block
          position reset) with the question visible in the shared prefix — no frame sees
          another frame.
  gate    a per-frame classifier on each block's <|vision_end|> state decides which frames
          are evidence for the question; non-evidence blocks are hidden from the readout.
  read    the answer is decoded from the prefix + the kept blocks only (a small LoRA is
          trained for this read; the backbone stays frozen).

Modules (each has a legacy twin under legacy/v1/gnnformer/ to diff against, and a test
under tests/ that pins its contract):
  constants   benchmark vocabulary + mask/model constants
  model       loading the frozen 4-bit backbone, layer/token accessors
  fence       block masks, position reset, slot positions, hook plumbing
  mmred       the official benchmark: rows, frames, gold recomputation, evidence labels
  prompt      the paper's prompt (verbatim), message layouts, answer parsing
  metrics     exact match per question type, d′, confidence intervals
"""
__version__ = "2.0.0"
