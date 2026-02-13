
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import torch
from nnsight import LanguageModel

from utils import describe, iter_sample_dirs, load_mmred_sample
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


def compute_ld(last_logits_1d: torch.Tensor, a_star_id: int) -> Tuple[float, int]:
    """
    last_logits_1d: [vocab]
    LD = logit(a*) - logit(a^-), where a^- is greedy argmax token.
    """
    greedy_id = int(torch.argmax(last_logits_1d).item())
    ld = float((last_logits_1d[a_star_id] - last_logits_1d[greedy_id]).item())
    return ld, greedy_id


def forward_with_cache(
    lm,
    layers,
    inputs: Dict[str, torch.Tensor],
    save_layer_states: bool = True,
    patch_layer_idx: int = None,
    patch_value: torch.Tensor = None,
) -> Tuple[Optional[List[torch.Tensor]], torch.Tensor]:
    """
    Runs one forward pass.
    Optionally:
      - save all layers' outputs (layer_states)
      - patch a single layer output during the run
    Returns: (layer_states or None, last_logits_1d[vocab])
    """
    def _materialize_saved(x):
        # nnsight may return either a save-handle (.value) or a raw Tensor.
        return x.value if hasattr(x, "value") else x

    if patch_layer_idx is not None and patch_value is not None:
        # Avoid in-place writes from an inference tensor captured in a previous pass.
        patch_value = patch_value.detach().clone()

    cache = {}

    with torch.inference_mode():
        with lm.trace(inputs):
            saved = None
            if save_layer_states:
                cache["layer_states"] = [layers[i].output.save() for i in range(len(layers))]

            if patch_layer_idx is not None:
                layers[patch_layer_idx].output = patch_value

            cache["last_logits"] = lm.output.logits[:, -1, :].save()

    last_logits = _materialize_saved(cache["last_logits"])[0]
    if save_layer_states:
        layer_states = [_materialize_saved(t) for t in cache["layer_states"]]
        return layer_states, last_logits

    return None, last_logits


def clean_run(lm, layers, sample_dir: Path) -> Dict[str, Any]:
    sample_id, frames, question, states, answer = load_mmred_sample(sample_dir)

    prompt = build_prompt(question, num_frames=len(frames))
    inputs = processor(images=frames, text=prompt, return_tensors="pt")
    inputs = dict(inputs)

    clean_layer_states, last_logits = forward_with_cache(
        lm, layers, inputs, save_layer_states=True
    )

    a_star_id = first_token_id_of_answer(answer)
    ld, greedy_id = compute_ld(last_logits, a_star_id)

    return {
        "sample_id": sample_id,
        "answer": answer,
        "a_star_id": a_star_id,
        "greedy_id": greedy_id,
        "ld": ld,
        "layer_states": clean_layer_states,  # List[tensor], one per layer
    }


def corrupted_runs(lm, layers, corrupted_dir: Path, a_star_id: int) -> Dict[str, Any]:
    """
    For each evidence-frame corrupted sample dir:
      run forward and compute LD.
    """
    # checks if there are any corrupted samples for this clean sample, if not returns empty evidence list
    if not corrupted_dir.is_dir():
        return {
            "corrupted_dir": str(corrupted_dir),
            "evidence": [],
        }

    evidence_dirs = iter_sample_dirs(corrupted_dir)
    evidence_dirs = sorted(evidence_dirs)

    out = []
    for ev_dir in evidence_dirs:
        ev_id, frames, question, states, answer = load_mmred_sample(ev_dir)
        prompt = build_prompt(question, num_frames=len(frames))
        inputs = processor(images=frames, text=prompt, return_tensors="pt")
        inputs = dict(inputs)

        _, last_logits = forward_with_cache(
            lm, layers, inputs, save_layer_states=False
        )
        ld, greedy_id = compute_ld(last_logits, a_star_id)

        out.append({
            "evidence_dir": str(ev_dir),
            "sample_id": ev_id,
            "ld": ld,
            "greedy_id": greedy_id,
        })

    return {
        "corrupted_dir": str(corrupted_dir),
        "evidence": out,
    }


def patched_runs(
    lm,
    layers,
    corrupted_dir: Path,
    clean_layer_states: List[torch.Tensor],
    a_star_id: int,
) -> Dict[str, Any]:
    """
    For each evidence corrupted sample:
      For each layer L:
        run corrupted forward but overwrite layer L output with clean_layer_states[L].
      Return patched LDs: evidence x layer.
    """
    if not corrupted_dir.is_dir():
        return {
            "corrupted_dir": str(corrupted_dir),
            "evidence": [],
        }

    evidence_dirs = iter_sample_dirs(corrupted_dir)
    evidence_dirs = sorted(evidence_dirs)

    all_results = []
    for ev_dir in evidence_dirs:
        ev_id, frames, question, states, answer = load_mmred_sample(ev_dir)
        prompt = build_prompt(question, num_frames=len(frames))
        inputs = processor(images=frames, text=prompt, return_tensors="pt")
        inputs = dict(inputs)

        per_layer = []
        for layer_idx in range(len(layers)):
            _, last_logits = forward_with_cache(
                lm,
                layers,
                inputs,
                save_layer_states=False,
                patch_layer_idx=layer_idx,
                patch_value=clean_layer_states[layer_idx],
            )
            ld, greedy_id = compute_ld(last_logits, a_star_id)
            per_layer.append({
                "layer": layer_idx,
                "ld": ld,
                "greedy_id": greedy_id,
            })

        all_results.append({
            "evidence_dir": str(ev_dir),
            "sample_id": ev_id,
            "patched": per_layer,
        })

    return {
        "corrupted_dir": str(corrupted_dir),
        "evidence": all_results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--corrupted_data_root", type=str, required=True)
    ap.add_argument("--limit", type=int, default=1)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    corrupted_root = Path(args.corrupted_data_root)

    lm = LanguageModel(hf_model, tokenizer=processor.tokenizer)
    layers = get_layers(lm.model)

    sample_dirs = iter_sample_dirs(data_root)
    sample_dirs = sample_dirs[: max(args.limit, 0)]

    for idx, sample_dir in enumerate(sample_dirs, start=1):
        clean = clean_run(lm, layers, sample_dir)

        corrupted_sample_dir = corrupted_root / str(clean["sample_id"])
        corrupted = corrupted_runs(lm, layers, corrupted_sample_dir, clean["a_star_id"])

        patched = patched_runs(lm, layers, corrupted_sample_dir, clean["layer_states"], clean["a_star_id"])

        print(
            f"[{idx}/{len(sample_dirs)}] sample_id={clean['sample_id']} "
            f"LD(clean)={clean['ld']:.4f} a*={clean['a_star_id']} "
            f"a^-={clean['greedy_id']} answer={clean['answer']!r}"
        )
        print(f"  corrupted evidence frames: {len(corrupted['evidence'])}")
        if corrupted["evidence"]:
            print(f"  first corrupted LD: {corrupted['evidence'][0]['ld']:.4f}")
        if patched["evidence"]:
            for p in patched["evidence"][0]["patched"]:
                print(f"  first patched: evidence0 layer{p['layer']} LD = {p['ld']:.4f}")



if __name__ == "__main__":
    main()
