
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
from nnsight import LanguageModel

from utils import describe, load_mmred_sample
from model import hf_model, processor, get_layers


def first_token_id_of_answer(answer_text: str) -> int:
    """
    a* = first token of the correct answer string (no special tokens).
    """
    ids = processor.tokenizer.encode(str(answer_text).strip(), add_special_tokens=False)
    if not ids:
        raise ValueError(f"Answer text tokenized to empty: {answer_text!r}")
    return int(ids[0])


def build_prompt(question: str, num_frames: int) -> str:
    img_tok = getattr(processor, "image_token", None) or getattr(processor.tokenizer, "image_token", None) or "<image>"
    img_prefix = " ".join([img_tok] * num_frames)

    return (
        f"{img_prefix}\n"
        f"You will be shown {num_frames} frames describing steps in a house.\n"
        f"Respond with a single integer from 0 to {num_frames} (0 is allowed). Output only the integer.\n"
        f"Question: {question}\n"
        f"Answer: "
    )


def clean_run_per_sample(
    lm: LanguageModel,
    layers,
    sample_dir: Path,
) -> Dict[str, Any]:
    """
    Runs ONE clean forward pass for a sample directory.
    """
    sample_id, frames, question, states, answer = load_mmred_sample(sample_dir)

    prompt = build_prompt(question, num_frames=len(frames))
    inputs = processor(images=frames, text=prompt, return_tensors="pt")
    inputs = dict(inputs)

    cache = {}

    with torch.inference_mode():
        with lm.trace(inputs):
            layer_states = [layers[i].output.save() for i in range(len(layers))]
            last_logits = lm.output.logits[:, -1, :].save()
            cache["layer_states"] = layer_states
            cache["last_logits"] = last_logits

    last_logits = cache["last_logits"][0]
    
    a_star_id = first_token_id_of_answer(answer)
    greedy_id = int(torch.argmax(last_logits).item())

    ld = float((last_logits[a_star_id] - last_logits[greedy_id]).item())

    return {
        "layer_states": cache["layer_states"],
        "a_star_id": a_star_id,
        "ld": ld,
    }


def iter_sample_dirs(data_root: Path) -> List[Path]:
    """
    Finds sample directories under data_root (directories that contain qa.txt).
    """
    out: List[Path] = []
    for p in sorted(data_root.iterdir()):
        if p.is_dir() and (p / "qa.txt").exists():
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="e.g. data/mmred_images/seq_len_8/train")
    ap.add_argument("--limit", type=int, default=1, help="number of samples to run")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"--data_root not found: {data_root}")

    lm = LanguageModel(hf_model, tokenizer=processor.tokenizer)
    layers = get_layers(lm.model)

    sample_dirs = iter_sample_dirs(data_root)
    if not sample_dirs:
        raise RuntimeError(f"No sample dirs (with qa.txt) found under: {data_root}")

    sample_dirs = sample_dirs[: max(args.limit, 0)]

    for idx, sample_dir in enumerate(sample_dirs, start=1):
        r = clean_run_per_sample(lm, layers, sample_dir)

        print(f"[{idx}/{len(sample_dirs)}] sample_id={r['sample_id']} LD(clean)={r['ld']:.4f} "
              f"a*={r['a_star_id']} a^-={r['greedy_id']} answer={r['answer']!r}")


if __name__ == "__main__":
    main()
