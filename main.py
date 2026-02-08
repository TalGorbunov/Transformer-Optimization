from linecache import cache
from nnsight import LanguageModel
from utils import describe
from model import hf_model, processor, find_blocks

def main():
    # Wrap with nnsight
    lm = LanguageModel(hf_model, tokenizer=processor.tokenizer)

    # Find transformer blocks
    blocks, blocks_path = find_blocks(lm.model)
    n_layers = len(blocks)
    print(f"Found blocks at: {blocks_path} | num_layers={n_layers}")

    # Prepare inputs
    prompt = "User: How many steps did Alice spend in kitchen?\nAssistant:"
    inputs = processor(text=prompt, return_tensors="pt")

    # Move inputs to model device
    first_param = next(lm.model.parameters())
    device = first_param.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Cache activations
    cached = {}

    with lm.trace(inputs):
        layer_states = [blocks[i].input[0].save() for i in range(n_layers)]
        logits = lm.output.logits.save()
        cached["layer_states"] = layer_states
        cached["logits"] = logits

    print("Cached:")
    print(" - layer_states:", len(cached["layer_states"]), cached["layer_states"][0].shape)
    print(" - logits:", cached["logits"].shape)

    last_tok_layer0 = cached["layer_states"][0][-1]
    print("Last token hidden (layer0) shape:", last_tok_layer0.shape)


if __name__ == "__main__":
    main()
