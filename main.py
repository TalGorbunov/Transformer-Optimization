from nnsight import LanguageModel
import torch
from utils import describe, load_mmred_sample, print_top_k
from model import hf_model, processor, get_layers
from pathlib import Path

DATA_SAMPLE_PATH = Path("data/mmred_images/seq_len_8/train/0023251")

def main():
    lm = LanguageModel(hf_model, tokenizer=processor.tokenizer)
    layers = get_layers(lm.model)

    sample_id, frames, question, states, answer = load_mmred_sample(DATA_SAMPLE_PATH)
    print(f"Question sample_id={sample_id}")
    print(f"states:")
    for s in states:
        print(f"step_id={s['step_id']}: {s['rooms']}")
    print(f"Question: {question}")
    print(f"Answer: {answer}")

    img_tok = getattr(processor, "image_token", None) or getattr(processor.tokenizer, "image_token", None) or "<image>"
    img_prefix = " ".join([img_tok] * len(frames))

    prompt = f"{img_prefix}\nYou will be shown 8 frames describing steps in a house.\nRespond with a single integer from 0 to 8 (0 is allowed). Output only the integer.\nQuestion: {question}\nAnswer: "
    inputs = processor(images=frames, text=prompt, return_tensors="pt")
    inputs = dict(inputs)

    cache = {}

    with torch.inference_mode():
        with lm.trace(inputs):
            layer_states = [layers[i].output.save() for i in range(len(layers))]
            logits = lm.output.logits.save()
            cache["layer_states"] = layer_states
            cache["logits"] = logits

    print("Cached:")
    print(" - layer_states:", len(cache["layer_states"]), cache["layer_states"][0].shape)
    print(" - logits:", cache["logits"].shape)

    last_logits = cache["logits"][0, -1]      
    print_top_k(last_logits, processor.tokenizer, k=10)


if __name__ == "__main__":
    main()
