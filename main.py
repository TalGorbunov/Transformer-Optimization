
import argparse
import math
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import torch
from nnsight import LanguageModel

from utils import (
    describe,
    format_centered_indices,
    format_centered_values,
    iter_sample_dirs,
    load_mmred_sample,
    plot_entropy_summary,
    write_sample_metrics,
)
from model import model as base_model, processor, get_layers


def first_token_id_of_answer(answer_text: str) -> int:
    """
    a* = first token of the correct answer string (no special tokens).
    """
    ids = processor.tokenizer.encode(str(answer_text).strip(), add_special_tokens=False)
    if not ids:
        raise ValueError(f"Answer text tokenized to empty: {answer_text!r}")
    return int(ids[0])


def build_prompt(question: str, num_frames: int) -> str:
    return (
        f"You will be shown {num_frames} frames describing steps in a house.\n"
        f"Respond with a single integer from 0 to {num_frames} (0 is allowed). Output only the integer.\n"
        f"Question: {question}\n"
        f"Answer: "
    )


def build_inputs(frames, question: str) -> Dict[str, torch.Tensor]:
    """
    Build model inputs using the chat template expected by the current VLM.
    """
    prompt = build_prompt(question, num_frames=len(frames))
    messages = [{
        "role": "user",
        "content": (
            [{"type": "image", "image": im} for im in frames] +
            [{"type": "text", "text": prompt}]
        ),
    }]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return dict(inputs)


def move_inputs_to_model_device(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    device = next(base_model.parameters()).device
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}


def pick_a_minus_from_clean(answer_text: str, a_star_id: int) -> int:
    """
    Select a^- from the clean run ground-truth answer:
      if the clean answer is integer k (a*), set a^- to token(k - 1).
    """
    try:
        k = int(str(answer_text).strip())
    except Exception as e:
        raise ValueError(f"Answer is not an integer, cannot set a^- = k-1: {answer_text!r}") from e

    a_minus_text = str(k - 1)
    a_minus_ids = processor.tokenizer.encode(a_minus_text, add_special_tokens=False)
    if not a_minus_ids:
        raise ValueError(f"a^- text tokenized to empty: {a_minus_text!r}")

    a_minus_id = int(a_minus_ids[0])
    if a_minus_id == a_star_id:
        raise ValueError(
            f"a^- token id equals a* token id ({a_star_id}) for answer={answer_text!r}"
        )
    return a_minus_id


def compute_ld(last_logits_1d: torch.Tensor, a_star_id: int, a_minus_id: int) -> float:
    """
    last_logits_1d: [vocab]
    LD = logit(a*) - logit(a^-), where a^- is fixed from the clean run.
    """
    return float((last_logits_1d[a_star_id] - last_logits_1d[a_minus_id]).item())


def parse_corrupted_frame_index(sample_id: str) -> Optional[int]:
    m = re.fullmatch(r"corrupted_frame_(\d+)", str(sample_id).strip())
    if not m:
        return None
    return int(m.group(1))


def image_token_positions_for_frame(
    input_ids_1d: torch.Tensor,
    frame_idx: int,
    expected_num_frames: int,
) -> Optional[List[int]]:
    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(processor.tokenizer, "image_token_id", None)
    if image_token_id is None:
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if image_token_id is None or frame_idx < 0:
        return None

    pos = (input_ids_1d == int(image_token_id)).nonzero(as_tuple=True)[0]
    if pos.numel() == 0:
        return None

    pos_list = [int(x) for x in pos.tolist()]
    groups: List[List[int]] = []
    cur: List[int] = [pos_list[0]]
    for p in pos_list[1:]:
        if p == cur[-1] + 1:
            cur.append(p)
        else:
            groups.append(cur)
            cur = [p]
    groups.append(cur)

    # For this setup, we expect one contiguous image-token block per frame.
    if len(groups) < expected_num_frames or frame_idx >= len(groups):
        return None
    return groups[frame_idx]


def forward_with_cache(
    lm,
    layers,
    inputs: Dict[str, torch.Tensor],
    save_layer_states: bool = True,
    patch_layer_idx: int = None,
    patch_value: torch.Tensor = None,
    patch_token_positions: Optional[List[int]] = None,
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

    def _to_hidden_tensor(x):
        # Some architectures expose layer outputs as tuples; use the hidden-state tensor.
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, (tuple, list)) and len(x) > 0:
            return _to_hidden_tensor(x[0])
        raise TypeError(f"Unsupported layer output type for patching: {type(x)}")

    if patch_layer_idx is not None and patch_value is not None:
        # Avoid in-place writes from an inference tensor captured in a previous pass.
        patch_value = _to_hidden_tensor(patch_value).detach().clone()

    cache = {}

    context = torch.no_grad() if patch_layer_idx is not None else torch.inference_mode()
    with context:
        with lm.trace(inputs):
            saved = None
            if save_layer_states:
                cache["layer_states"] = [layers[i].output.save() for i in range(len(layers))]

            if patch_layer_idx is not None:
                patch_pos = patch_token_positions if patch_token_positions else [-1]
                try:
                    layers[patch_layer_idx].output[:, patch_pos, :] = patch_value[:, patch_pos, :]
                except Exception:
                    # Qwen-like blocks may expose output as a tuple-like (hidden, ...).
                    layers[patch_layer_idx].output[0][:, patch_pos, :] = patch_value[:, patch_pos, :]

            cache["last_logits"] = lm.output.logits[:, -1, :].save()

    last_logits = _materialize_saved(cache["last_logits"])[0]
    if save_layer_states:
        layer_states = [_to_hidden_tensor(_materialize_saved(t)) for t in cache["layer_states"]]
        return layer_states, last_logits

    return None, last_logits


def clean_run(lm, layers, sample_dir: Path) -> Dict[str, Any]:
    sample_id, frames, question, states, answer = load_mmred_sample(sample_dir)

    inputs = build_inputs(frames, question)
    inputs = move_inputs_to_model_device(inputs)

    clean_layer_states, last_logits = forward_with_cache(
        lm, layers, inputs, save_layer_states=True
    )

    a_star_id = first_token_id_of_answer(answer)
    a_minus_id = pick_a_minus_from_clean(answer, a_star_id)
    ld = compute_ld(last_logits, a_star_id, a_minus_id)
    pred_token_id = int(torch.argmax(last_logits).item())
    model_answer = processor.tokenizer.decode([pred_token_id], skip_special_tokens=True).strip()

    return {
        "sample_id": sample_id,
        "answer": answer,
        "a_star_id": a_star_id,
        "a_minus_id": a_minus_id,
        "pred_token_id": pred_token_id,
        "model_answer": model_answer,
        "ld": ld,
        "layer_states": clean_layer_states,  # List[tensor], one per layer
    }


def corrupted_runs(lm, layers, corrupted_dir: Path, a_star_id: int, a_minus_id: int) -> Dict[str, Any]:
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
        inputs = build_inputs(frames, question)
        inputs = move_inputs_to_model_device(inputs)

        _, last_logits = forward_with_cache(
            lm, layers, inputs, save_layer_states=False
        )
        ld = compute_ld(last_logits, a_star_id, a_minus_id)

        out.append({
            "evidence_dir": str(ev_dir),
            "sample_id": ev_id,
            "ld": ld,
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
    a_minus_id: int,
    clean_ld: float,
    corrupted_ld_by_dir: Dict[str, float],
    min_corrupted_diff: float,
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
    skipped_by_min_corrupted_diff = 0
    for ev_dir in evidence_dirs:
        corr_ld = corrupted_ld_by_dir.get(str(ev_dir))
        if corr_ld is None:
            continue
        if (clean_ld - float(corr_ld)) < float(min_corrupted_diff):
            skipped_by_min_corrupted_diff += 1
            continue

        ev_id, frames, question, states, answer = load_mmred_sample(ev_dir)
        inputs = build_inputs(frames, question)
        inputs = move_inputs_to_model_device(inputs)
        ev_frame_idx = parse_corrupted_frame_index(ev_id)
        patch_token_positions = (
            image_token_positions_for_frame(inputs["input_ids"][0], ev_frame_idx, len(frames))
            if ev_frame_idx is not None
            else None
        )

        per_layer = []
        for layer_idx in range(len(layers)):
            _, last_logits = forward_with_cache(
                lm,
                layers,
                inputs,
                save_layer_states=False,
                patch_layer_idx=layer_idx,
                patch_value=clean_layer_states[layer_idx],
                patch_token_positions=patch_token_positions,
            )
            ld = compute_ld(last_logits, a_star_id, a_minus_id)
            per_layer.append({
                "layer": layer_idx,
                "ld": ld,
            })

        all_results.append({
            "evidence_dir": str(ev_dir),
            "sample_id": ev_id,
            "patched": per_layer,
        })

    return {
        "corrupted_dir": str(corrupted_dir),
        "evidence": all_results,
        "skipped_by_min_corrupted_diff": skipped_by_min_corrupted_diff,
    }

def compute_layer_importance_entropy(
    clean_ld: float,
    corrupted: Dict[str, Any],
    patched: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Computes per-layer:
      r_i^l = (LD_patched(i,l) - LD_corrupted(i)) / (LD_clean - LD_corrupted(i))
      p_i^l = r_i^l / sum_j r_j^l
      H(l)  = -sum_j p_j^l * log(p_j^l)
      H_norm(l) = H(l) / N_evidence
    where i/j index evidence frames.
    """
    if not corrupted["evidence"] or not patched["evidence"]:
        return {"layers": []}

    patched_dirs = {e["evidence_dir"] for e in patched["evidence"]}
    corrupted_ld_by_dir = {
        e["evidence_dir"]: e["ld"]
        for e in corrupted["evidence"]
        if e["evidence_dir"] in patched_dirs
    }
    num_layers = len(patched["evidence"][0]["patched"])
    num_evidence = len(patched["evidence"])

    r_by_layer: List[List[float]] = [[0.0 for _ in range(num_evidence)] for _ in range(num_layers)]

    for i, ev in enumerate(patched["evidence"]):
        corr_ld = corrupted_ld_by_dir.get(ev["evidence_dir"])
        if corr_ld is None:
            continue
        denom = float(clean_ld - corr_ld)
        if denom <= 0.0:
            continue
        for pl in ev["patched"]:
            l = int(pl["layer"])
            num = float(pl["ld"] - corr_ld)
            r_by_layer[l][i] = max(num / denom, 0.0)

    layers_out = []
    for l in range(num_layers):
        r_vals = r_by_layer[l]
        denom = sum(r_vals)
        if denom > 0.0:
            p_vals = [r / denom for r in r_vals]
        else:
            p_vals = [0.0 for _ in r_vals]

        entropy_raw = -sum(p * math.log(p) for p in p_vals if p > 0.0)
        entropy_norm = entropy_raw / num_evidence if num_evidence > 0 else 0.0
        layers_out.append({
            "layer": l,
            "r": r_vals,
            "p": p_vals,
            "entropy": entropy_norm,
            "entropy_raw": entropy_raw,
            "num_evidence_frames": num_evidence,
        })

    return {"layers": layers_out}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--corrupted_data_root", type=str, required=True)
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--min_clean_ld", type=float, default=1.0)
    ap.add_argument("--min_corrupted_diff", type=float, default=0.001)
    ap.add_argument("--output", type=str, default="outputs")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    corrupted_root = Path(args.corrupted_data_root)
    seq_len_match = re.search(r"(seq_len_\d+)", str(data_root))
    seq_len_label = seq_len_match.group(1) if seq_len_match else None

    lm = LanguageModel(base_model, tokenizer=processor.tokenizer)
    layers = get_layers(lm.model)
    sample_metrics = []
    processed_samples = 0
    target_processed_samples = max(args.limit, 0)

    sample_dirs = iter_sample_dirs(data_root)

    for idx, sample_dir in enumerate(sample_dirs, start=1):
        if processed_samples >= target_processed_samples:
            break

        corrupted_sample_dir = corrupted_root / str(sample_dir.name)
        num_evidence_frames = len(iter_sample_dirs(corrupted_sample_dir)) if corrupted_sample_dir.is_dir() else 0
        if num_evidence_frames < 2:
            print(
                f"[{idx}/{len(sample_dirs)}] sample_id={sample_dir.name} "
                f"skipped: evidence frames={num_evidence_frames} < 2"
            )
            continue

        # clean run
        try:
            clean = clean_run(lm, layers, sample_dir)
        except Exception as e:
            print(
                f"[{idx}/{len(sample_dirs)}] sample_id={sample_dir.name} "
                f"skipped: failed to load/run clean sample ({e})"
            )
            continue
        if clean["ld"] < args.min_clean_ld:
            print(
                f"[{idx}/{len(sample_dirs)}] sample_id={clean['sample_id']} "
                f"skipped: LD(clean)={clean['ld']:.4f} < {args.min_clean_ld:.4f} "
                f"(evidence frames={num_evidence_frames})"
            )
            continue

        # corrupted runs
        corrupted = corrupted_runs(
            lm,
            layers,
            corrupted_sample_dir,
            clean["a_star_id"],
            clean["a_minus_id"],
        )

        # patched runs
        corrupted_ld_by_dir = {
            e["evidence_dir"]: float(e["ld"]) for e in corrupted["evidence"]
        }
        patched = patched_runs(
            lm,
            layers,
            corrupted_sample_dir,
            clean["layer_states"],
            clean["a_star_id"],
            clean["a_minus_id"],
            clean["ld"],
            corrupted_ld_by_dir,
            args.min_corrupted_diff,
        )
        if len(patched["evidence"]) < 2:
            print(
                f"[{idx}/{len(sample_dirs)}] sample_id={clean['sample_id']} "
                f"skipped: patched evidence frames={len(patched['evidence'])} < 2 "
                f"(after min_corrupted_diff={args.min_corrupted_diff:.4f})"
            )
            continue

        print(
            f"[{idx}/{len(sample_dirs)}] sample_id={clean['sample_id']} "
            f"LD(clean)={clean['ld']:.4f} a*={clean['a_star_id']} "
            f"a^-={clean['a_minus_id']} answer={clean['answer']!r} "
            f"model_answer={clean['model_answer']!r} (id={clean['pred_token_id']})"
        )
        print(f"  corrupted evidence frames: {len(corrupted['evidence'])}")
        if patched.get("skipped_by_min_corrupted_diff", 0) > 0:
            print(
                f"  skipped patched frames by min_corrupted_diff: "
                f"{patched['skipped_by_min_corrupted_diff']} "
                f"(threshold={args.min_corrupted_diff:.4f})"
            )
        if corrupted["evidence"]:
            corrupted_lds = [float(ev["ld"]) for ev in corrupted["evidence"]]
            print(f"  frame idx: {format_centered_indices(len(corrupted_lds))}")
            print(f"  corr LD : {format_centered_values(corrupted_lds)}")
        if patched["evidence"]:
            num_layers = len(patched["evidence"][0]["patched"])
            print(f"  frame idx: {format_centered_indices(len(patched['evidence']))}")
            for layer_idx in range(num_layers):
                layer_lds = [
                    float(ev["patched"][layer_idx]["ld"])
                    for ev in patched["evidence"]
                ]
                print(f"  patched layer{layer_idx:>2}: {format_centered_values(layer_lds)}")

        # per layer metrics
        layer_metrics = compute_layer_importance_entropy(clean["ld"], corrupted, patched)
        
        sample_metrics.append({
            "sample_id": clean["sample_id"],
            "layer_metrics": layer_metrics,
        })
        processed_samples += 1

    output_path = write_sample_metrics(sample_metrics, Path(args.output))
    print(f"Wrote sample metrics to: {output_path}")
    print(
        f"Model actually ran on {processed_samples} samples "
        f"(target limit={target_processed_samples})."
    )
    plot_path = plot_entropy_summary(
        sample_metrics,
        Path(args.output),
        seq_len_label=seq_len_label,
    )
    if plot_path is not None:
        print(f"Wrote entropy plot to: {plot_path}")
    else:
        print("Skipped entropy plot: no layer metrics available.")


if __name__ == "__main__":
    main()
