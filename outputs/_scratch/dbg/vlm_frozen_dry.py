"""CPU dry run of vlm_frozen.py on a tiny random Qwen2.5-VL with the REAL 7B processor: exercises
prep/sub/frame_mass/run, the startup parity assertions (manual stack == model(**inp), positions),
every arm, the summary, and --scan."""
import sys

sys.path.insert(0, "scripts/condmask")
import torch  # noqa: E402
from transformers import AutoProcessor, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration  # noqa: E402

import vlm_frozen as vf  # noqa: E402
from gnnformer.runtime import ModelRuntime  # noqa: E402

torch.manual_seed(0)
proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=False)
tok = proc.tokenizer
vc = dict(depth=2, hidden_size=64, intermediate_size=128, num_heads=4, patch_size=14, spatial_merge_size=2,
          temporal_patch_size=2, out_hidden_size=64, fullatt_block_indexes=[1], window_size=112, in_channels=3)
tc = dict(vocab_size=len(tok) + 64, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32768,
          rope_scaling={"type": "mrope", "mrope_section": [2, 2, 4]}, attn_implementation="sdpa")
cfg = Qwen2_5_VLConfig(vision_config=vc, text_config=tc,
                       image_token_id=tok.convert_tokens_to_ids("<|image_pad|>"),
                       vision_start_token_id=tok.convert_tokens_to_ids("<|vision_start|>"),
                       vision_end_token_id=tok.convert_tokens_to_ids("<|vision_end|>"))
model = Qwen2_5_VLForConditionalGeneration(cfg).eval()
vf.load_runtime = lambda *a, **k: ModelRuntime(model_name="Tiny", processor=proc, model=model)

base = ["x", "--root", "data/mmred_vfiltered/seq_len_8/test", "--limit", "2", "--k", "4", "--sel-layer", "1",
        "--resize", "112", "--output", "outputs/_scratch/dbg/vlm_dry"]
sys.argv = base
assert vf.main() == 0
sys.argv = base + ["--scan"]
assert vf.main() == 0
print("DRY OK")
