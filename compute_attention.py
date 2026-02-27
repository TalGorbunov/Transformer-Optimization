import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import model, processor
from utils import iter_sample_dirs, load_mmred_sample

_STEPS_IN_ROOM_RE = re.compile(
    r"How many steps did\s+([A-Za-z]+)\s+spend in\s+the\s+([A-Za-z]+)",
    flags=re.IGNORECASE,
)


def build_prompt(question: str, num_frames: int) -> str:
    return (
        f"You will be shown {num_frames} frames describing steps in a house.\n"
        f"Respond with a single integer from 0 to {num_frames} (0 is allowed). Output only the integer.\n"
        f"Question: {question}\n"
        "Answer: "
    )


def build_inputs(frames, question: str) -> Dict[str, torch.Tensor]:
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
    device = next(model.parameters()).device
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}


def parse_target_character_room(question_text: str) -> Optional[Tuple[str, str]]:
    m = _STEPS_IN_ROOM_RE.search(question_text)
    if not m:
        return None
    character = m.group(1).strip()
    room = m.group(2).strip()
    room_norm = room[:1].upper() + room[1:].lower() if room else room
    return character, room_norm


def rooms_to_room2chars(rooms: Dict[str, Any]) -> Dict[str, List[str]]:
    if not isinstance(rooms, dict):
        return {}

    if any(isinstance(v, list) for v in rooms.values()):
        out: Dict[str, List[str]] = {}
        for k, v in rooms.items():
            if not isinstance(k, str):
                continue
            rk = k.strip()
            rk_norm = rk[:1].upper() + rk[1:].lower() if rk else rk
            out.setdefault(rk_norm, [])
            if isinstance(v, list):
                out[rk_norm].extend([str(x) for x in v])
        for r in out:
            out[r] = sorted(set(out[r]))
        return out

    out: Dict[str, List[str]] = {}
    for ch, rm in rooms.items():
        if not isinstance(rm, str):
            continue
        rm_norm = rm[:1].upper() + rm[1:].lower()
        out.setdefault(rm_norm, []).append(str(ch))
    for r in out:
        out[r] = sorted(set(out[r]))
    return out


def char_in_room(step_rooms: Dict[str, Any], character: str, room: str) -> bool:
    r2c = rooms_to_room2chars(step_rooms)
    return character in r2c.get(room, [])


def get_evidence_frame_indices(question: str, states: List[Dict[str, Any]]) -> Optional[List[int]]:
    parsed = parse_target_character_room(question)
    if parsed is None:
        return None
    character, room = parsed
    indices: List[int] = []
    for i, state in enumerate(states):
        step_rooms = state.get("rooms", {}) if isinstance(state, dict) else {}
        if char_in_room(step_rooms, character, room):
            indices.append(i)
    return indices


def image_token_groups(input_ids_1d: torch.Tensor, expected_num_frames: int) -> List[List[int]]:
    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(processor.tokenizer, "image_token_id", None)
    if image_token_id is None:
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if image_token_id is None:
        return []

    pos = (input_ids_1d == int(image_token_id)).nonzero(as_tuple=True)[0]
    if pos.numel() == 0:
        return []

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

    if expected_num_frames <= 0:
        return groups
    return groups[:expected_num_frames]


def compute_scores_from_attentions(
    attentions: Tuple[torch.Tensor, ...],
    frame_to_tokens: Dict[int, List[int]],
) -> List[Dict[str, Any]]:
    per_layer: List[Dict[str, Any]] = []
    ordered_frames = sorted(frame_to_tokens.keys())

    for layer_idx, attn in enumerate(attentions):
        if attn is None:
            continue
        # attn shape: [B, H, T, T]
        a = attn[0].detach()
        recv_per_head = a.sum(dim=-2)  # [H, T] column-sum over query tokens

        frame_scores: Dict[int, float] = {}
        for frame_idx in ordered_frames:
            tok_pos = frame_to_tokens[frame_idx]
            if not tok_pos:
                frame_scores[frame_idx] = 0.0
                continue
            token_received = recv_per_head[:, tok_pos].sum(dim=-1)  # [H]
            frame_scores[frame_idx] = float(token_received.mean().item())  # avg over heads

        per_layer.append({
            "layer": layer_idx,
            "frame_scores": frame_scores,
        })

    if not per_layer:
        raise RuntimeError(
            "No usable attention tensors found (all layers are None)."
        )

    return per_layer


def _set_attr_if_exists(obj: Any, attr: str, value: Any) -> bool:
    if obj is None or not hasattr(obj, attr):
        return False
    try:
        setattr(obj, attr, value)
        return True
    except Exception:
        return False


def force_eager_attention_backend() -> None:
    """
    Qwen2.5-VL with SDPA can return attentions as all-None even when output_attentions=True.
    Force eager attention implementation on known config objects.
    """
    targets = [
        ("model.config", getattr(model, "config", None)),
        ("model.model.config", getattr(getattr(model, "model", None), "config", None)),
        (
            "model.model.language_model.config",
            getattr(getattr(getattr(model, "model", None), "language_model", None), "config", None),
        ),
    ]
    for _, cfg in targets:
        if cfg is None:
            continue
        _set_attr_if_exists(cfg, "_attn_implementation", "eager")
        _set_attr_if_exists(cfg, "attn_implementation", "eager")
        _set_attr_if_exists(cfg, "output_attentions", True)


def print_layer_frame_scores_table(result: Dict[str, Any], precision: int = 6) -> None:
    per_layer = result.get("per_layer", [])
    evidence_frames = [int(x) for x in result.get("evidence_frames", [])]
    if not per_layer or not evidence_frames:
        print("[scores] no per-layer/frame scores to print")
        return

    frame_cols = [f"f{f}" for f in evidence_frames]
    layer_label = "layer"
    layer_width = max(len(layer_label), 5)
    col_width = max(14, precision + 8)

    header = layer_label.rjust(layer_width) + " | " + " ".join(c.rjust(col_width) for c in frame_cols)
    print("[scores] received attention per layer x evidence frame")
    print(header)
    print("-" * len(header))

    for layer_entry in per_layer:
        layer_idx = int(layer_entry.get("layer", -1))
        frame_scores = layer_entry.get("frame_scores", {})
        vals: List[str] = []
        for f in evidence_frames:
            score = float(frame_scores.get(f, 0.0))
            vals.append(f"{score:.{precision}f}".rjust(col_width))
        row = str(layer_idx).rjust(layer_width) + " | " + " ".join(vals)
        print(row)


@torch.no_grad()
def process_sample(sample_dir: Path) -> Dict[str, Any]:
    sample_id, frames, question, states, answer = load_mmred_sample(sample_dir)
    inputs = build_inputs(frames, question)
    groups = image_token_groups(inputs["input_ids"][0], expected_num_frames=len(frames))

    evidence_frames = get_evidence_frame_indices(question, states)
    parse_warning = None
    if evidence_frames is None:
        evidence_frames = list(range(min(len(frames), len(groups))))
        parse_warning = "failed_to_parse_evidence_from_question_using_all_frames_with_token_groups"

    frame_to_tokens: Dict[int, List[int]] = {}
    for frame_idx in evidence_frames:
        if frame_idx < len(groups):
            frame_to_tokens[frame_idx] = groups[frame_idx]
        else:
            frame_to_tokens[frame_idx] = []

    model_inputs = move_inputs_to_model_device(inputs)
    outputs = model(**model_inputs, output_attentions=True, use_cache=False, return_dict=True)

    attentions = outputs.attentions
    if attentions is None:
        raise RuntimeError("Model did not return outputs.attentions.")

    per_layer = compute_scores_from_attentions(attentions, frame_to_tokens)

    return {
        "sample_id": sample_id,
        "sample_dir": str(sample_dir),
        "question": question,
        "answer": answer,
        "num_frames": len(frames),
        "num_layers": len(per_layer),
        "evidence_frames": evidence_frames,
        "frame_token_lengths": {int(k): len(v) for k, v in frame_to_tokens.items()},
        "parse_warning": parse_warning,
        "per_layer": per_layer,
    }


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Iterate MMRed samples and compute per-layer received-attention per evidence frame.\n"
            "Score definition per layer: sum columns over frame-linked tokens, then average across heads."
        )
    )
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--limit", type=int, default=-1, help="Max number of samples; -1 means all.")
    ap.add_argument(
        "--out_jsonl",
        type=str,
        default=None,
        help="Optional JSONL output path; if omitted, only prints progress.",
    )
    args = ap.parse_args()

    sample_dirs = iter_sample_dirs(Path(args.data_root))
    if args.limit >= 0:
        sample_dirs = sample_dirs[: args.limit]

    if not sample_dirs:
        print(f"No samples found under {args.data_root}")
        return

    force_eager_attention_backend()

    out_fp = None
    if args.out_jsonl:
        out_path = Path(args.out_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_fp = out_path.open("w", encoding="utf-8")

    total = len(sample_dirs)
    for i, sample_dir in enumerate(sample_dirs, start=1):
        result = process_sample(sample_dir)
        print(
            f"[{i}/{total}] sample={result['sample_id']} "
            f"evidence_frames={result['evidence_frames']} "
            f"layers={result['num_layers']}"
        )
        print_layer_frame_scores_table(result)
        if out_fp is not None:
            out_fp.write(json.dumps(result, ensure_ascii=True) + "\n")

    if out_fp is not None:
        out_fp.close()
        print(f"Wrote JSONL to {args.out_jsonl}")


if __name__ == "__main__":
    main()
