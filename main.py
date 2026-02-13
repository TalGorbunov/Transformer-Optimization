
import argparse
import math
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

def compute_layer_importance_entropy(
    clean_ld: float,
    corrupted: Dict[str, Any],
    patched: Dict[str, Any],
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """
    Computes per-layer:
      r_i^l = (LD_patched(i,l) - LD_corrupted(i)) / (LD_clean - LD_corrupted(i) + eps)
      p_i^l = r_i^l / sum_j r_j^l
      H(l)  = -sum_j p_j^l * log(p_j^l)
    where i/j index evidence frames.
    """
    if not corrupted["evidence"] or not patched["evidence"]:
        return {"layers": []}

    corrupted_ld_by_dir = {e["evidence_dir"]: e["ld"] for e in corrupted["evidence"]}
    num_layers = len(patched["evidence"][0]["patched"])
    num_evidence = len(patched["evidence"])

    r_by_layer: List[List[float]] = [[0.0 for _ in range(num_evidence)] for _ in range(num_layers)]

    for i, ev in enumerate(patched["evidence"]):
        corr_ld = corrupted_ld_by_dir.get(ev["evidence_dir"])
        if corr_ld is None:
            continue
        denom = float(clean_ld - corr_ld + eps)
        for pl in ev["patched"]:
            l = int(pl["layer"])
            num = float(pl["ld"] - corr_ld)
            r_by_layer[l][i] = num / denom

    layers_out = []
    for l in range(num_layers):
        r_vals = r_by_layer[l]
        denom = sum(r_vals)
        if denom > 0.0:
            p_vals = [r / denom for r in r_vals]
        else:
            p_vals = [0.0 for _ in r_vals]

        entropy = -sum(p * math.log(p) for p in p_vals if p > 0.0)
        layers_out.append({
            "layer": l,
            "r": r_vals,
            "p": p_vals,
            "entropy": entropy,
        })

    return {"layers": layers_out}

def write_sample_metrics(sample_metrics: List[Dict[str, Any]], output_dir: Path) -> Path:
    def _fmt_float_list(vals: List[float]) -> str:
        return "[" + ", ".join(f"{v:.6f}" for v in vals) + "]"

    lines: List[str] = []
    for sm in sample_metrics:
        lines.append(f"sample_id={sm['sample_id']}")
        for lmtr in sm["layer_metrics"]["layers"]:
            lines.append(
                f"layer={lmtr['layer']} "
                f"r={_fmt_float_list(lmtr['r'])} "
                f"p={_fmt_float_list(lmtr['p'])} "
                f"H={lmtr['entropy']:.6f}"
            )
        lines.append("")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sample_metrics.txt"
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--corrupted_data_root", type=str, required=True)
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--output", type=str, default="outputs")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    corrupted_root = Path(args.corrupted_data_root)

    lm = LanguageModel(hf_model, tokenizer=processor.tokenizer)
    layers = get_layers(lm.model)
    sample_metrics = []

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

        # per layer metrics
        layer_metrics = compute_layer_importance_entropy(clean["ld"], corrupted, patched)
        
        sample_metrics.append({
            "sample_id": clean["sample_id"],
            "layer_metrics": layer_metrics,
        })

    output_path = write_sample_metrics(sample_metrics, Path(args.output))
    print(f"Wrote sample metrics to: {output_path}")


if __name__ == "__main__":
    main()
