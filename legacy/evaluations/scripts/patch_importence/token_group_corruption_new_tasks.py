#!/usr/bin/env python3
"""Token-group restoration/corruption ablation for the NEW MMReD tasks (rooms_visited, co_occupancy).

The existing group_restoration_importance.py corrupts the COUNTING-evidence frames by swapping in
pre-rendered corrupted images -- but those assets only exist for the counting task's evidence frames.
The new tasks corrupt different frames, so we use an ASSET-FREE corruption: blank (mid-gray) the
signal-bearing frames on the fly. This is task-agnostic and a strictly stronger corruption.

Pipeline per sample (reusing the proven patching_core + group_restoration_importance machinery):
  1. recompute (question, gold answer, evidence frames) for the chosen task from the sample states
  2. clean run  -> logprob of gold
  3. corrupted run (signal frames blanked) -> logprob of gold
  4. restore clean activations at each layer for 3 token groups: frames / last(carrier) token / question
  5. normalized_rescue = (patched - corrupted) / (clean - corrupted), bootstrapped CIs

Output mirrors group_restoration_importance.py (per_sample + aggregate CSV + plot).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from nnsight import LanguageModel

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluations.helpers import patching_core as tgi
from evaluations.helpers import utils as eval_utils
from evaluations.helpers.utils import iter_sample_dirs, load_mmred_sample
from evaluations.scripts.patch_importence import group_restoration_importance as gri
from models.model import DEFAULT_MODEL_ID, get_layers


def _present_frames(states: List[Dict[str, Any]], char: str) -> List[int]:
    out = []
    for fi, st in enumerate(states):
        r2c = eval_utils.rooms_to_room2chars(st.get("rooms", {}))
        if any(char in occ for occ in r2c.values()):
            out.append(fi)
    return out


def _all_chars(states: List[Dict[str, Any]]) -> List[str]:
    return sorted(eval_utils.extract_characters_from_states(states))


def _rooms_visited(states: List[Dict[str, Any]], char: str) -> int:
    rooms = set()
    for st in states:
        for room, occ in eval_utils.rooms_to_room2chars(st.get("rooms", {})).items():
            if char in occ:
                rooms.add(room)
    return len(rooms)


def _shared_frames(states: List[Dict[str, Any]], c1: str, c2: str) -> List[int]:
    out = []
    for fi, st in enumerate(states):
        r2c = eval_utils.rooms_to_room2chars(st.get("rooms", {}))
        if any(c1 in occ and c2 in occ for occ in r2c.values()):
            out.append(fi)
    return out


def task_spec(task: str, states: List[Dict[str, Any]], num_frames: int,
              qa_question: Optional[str] = None, qa_answer: Optional[str] = None) -> Optional[Tuple[str, str, List[int]]]:
    """Return (question, gold_answer_text, evidence_frame_indices) for the task, or None to skip."""
    if task == "count":
        # the original counting task: question/answer from qa.txt, evidence = frames matching the query
        if qa_question is None or qa_answer is None:
            return None
        evidence = eval_utils.collect_evidence_frame_indices(qa_question, states)
        if not evidence:
            return None
        return qa_question, str(qa_answer).strip(), [int(i) for i in evidence]
    chars = _all_chars(states)
    if not chars:
        return None
    if task == "rooms_visited":
        # pick the character present in the most frames (strongest signal), tie-break alphabetical
        char = max(chars, key=lambda c: (len(_present_frames(states, c)), c))
        present = _present_frames(states, char)
        if not present:
            return None
        gold = _rooms_visited(states, char)
        q = f"How many distinct rooms did {char} visit across the {num_frames} frames?"
        return q, str(int(gold)), present
    if task == "co_occupancy":
        if len(chars) < 2:
            return None
        best = None
        for i in range(len(chars)):
            for j in range(i + 1, len(chars)):
                sh = _shared_frames(states, chars[i], chars[j])
                if best is None or len(sh) > len(best[2]):
                    best = (chars[i], chars[j], sh)
        if best is None:
            return None
        c1, c2, shared = best
        gold = len(shared)
        # evidence for co-occupancy = frames where they share a room (the signal). If none, skip
        # (gold=0 has no signal frame to corrupt).
        if not shared:
            return None
        q = f"In how many of the {num_frames} frames were {c1} and {c2} in the same room?"
        return q, str(int(gold)), shared
    raise ValueError(f"unknown task {task}")


def blank_frames(frames: List[Any], indices: List[int], mode: str) -> List[Any]:
    out = list(frames)
    targets = set(int(i) for i in indices) if mode == "evidence" else set(range(len(frames)))
    for i in targets:
        im = frames[i]
        out[i] = Image.new("RGB", im.size, (128, 128, 128))
    return out


def build_blank_corrupted_inputs(frames: List[Any], question: str, evidence: List[int], prompt_len: int, mode: str) -> Optional[Dict[str, Any]]:
    corrupted = blank_frames(frames, evidence, mode)
    inputs = tgi.move_inputs_to_model_device(tgi.build_inputs(corrupted, question))
    if int(inputs["input_ids"].shape[1]) != prompt_len:
        return None
    return inputs


def build_surgical_corrupted_inputs(sample_id: str, frames: List[Any], question: str, evidence: List[int],
                                    prompt_len: int, corrupted_root: Path) -> Optional[Dict[str, Any]]:
    """Surgical corruption (char removed from room), exactly like the original counting experiments:
    compose the per-frame char-removed renders for every evidence frame via build_composite_corrupted_frames."""
    corrupted_frames, issues = eval_utils.build_composite_corrupted_frames(
        sample_id=sample_id, clean_frames=frames, evidence_frame_indices=evidence, corrupted_data_root=corrupted_root)
    if corrupted_frames is None:
        return None
    inputs = tgi.move_inputs_to_model_device(tgi.build_inputs(corrupted_frames, question))
    if int(inputs["input_ids"].shape[1]) != prompt_len:
        return None
    return inputs


_EPS = 1e-8


def _patch_score(lm, layers, target_inputs, source_inputs, layer_idx, positions, prompt_len, a_ids, fallback):
    """Patch source activations into target at one layer & positions; return gold-answer logprob (or fallback)."""
    if not list(positions):
        return float(fallback)
    try:
        return float(tgi.run_layer_corrupted_sequence_logprob(
            lm=lm, layers=layers, target_scoring_inputs=target_inputs, source_scoring_inputs=source_inputs,
            layer_idx=int(layer_idx), target_token_positions=list(positions), source_token_positions=list(positions),
            prompt_len=prompt_len, answer_token_ids=list(a_ids)))
    except Exception as exc:
        print(f"  layer {layer_idx} patch failed ({exc}); fallback")
        return float(fallback)


def compute_dual_direction(lm, layers, selected_layers, clean_inputs, corrupted_inputs,
                           clean_score, corrupted_score, prompt_len, a_ids, group_positions):
    """Both families like the original unified script:
       restoration  = patch CLEAN into CORRUPTED run  -> (restored - corrupted)/denom
       clean_ablation = patch CORRUPTED into CLEAN run -> (clean - ablated)/denom
    Returns long-format rows: {layer, patch_target, metric_type, raw_value, normalized_value}."""
    denom = float(clean_score - corrupted_score)
    rows = []
    for L in selected_layers:
        for gname, pos in group_positions.items():
            restored = _patch_score(lm, layers, corrupted_inputs, clean_inputs, L, pos, prompt_len, a_ids, corrupted_score)
            ablated = _patch_score(lm, layers, clean_inputs, corrupted_inputs, L, pos, prompt_len, a_ids, clean_score)
            nr = (restored - corrupted_score) / denom if abs(denom) > _EPS else float("nan")
            nd = (clean_score - ablated) / denom if abs(denom) > _EPS else float("nan")
            rows.append({"layer": int(L), "patch_target": gname, "metric_type": "restoration",
                         "raw_value": float(restored - corrupted_score), "normalized_value": float(nr)})
            rows.append({"layer": int(L), "patch_target": gname, "metric_type": "clean_ablation_damage",
                         "raw_value": float(clean_score - ablated), "normalized_value": float(nd)})
    return rows


def process_sample(sample_dir: Path, idx: int, total: int, task: str, lm: LanguageModel, layers: Any,
                   selected_layers: List[int], frame_patch_mode: str, corruption_mode: str,
                   clean_top1_must_match_gold: bool, corrupted_root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    try:
        sample_id, frames, _q0, states, _a0 = load_mmred_sample(sample_dir)
    except Exception as exc:
        print(f"[{idx}/{total}] {sample_dir.name} skipped: load failure ({exc})")
        return None
    spec = task_spec(task, states, len(frames), qa_question=_q0, qa_answer=_a0)
    if spec is None:
        return None
    question, answer_text, evidence = spec

    clean_res = gri.build_clean_inputs(sample_id, frames, question, idx, total)
    if clean_res is None:
        return None
    clean_inputs, prompt_len = clean_res
    carrier_index = int(prompt_len - 1)

    a_ids = gri.build_answer_token_ids(sample_id, answer_text, idx, total)
    if a_ids is None:
        return None

    clean_metrics = gri.score_clean_sample(sample_id, frames, answer_text, clean_inputs, prompt_len,
                                           clean_top1_must_match_gold, lm, idx, total)
    if clean_metrics is None:
        return None
    clean_answer_score = float(clean_metrics["clean_answer_score"])

    patch_meta = gri.extract_patch_positions(sample_id, clean_inputs, question, len(frames), idx, total)
    if patch_meta is None:
        return None
    frame_groups = patch_meta["frame_groups"]
    question_positions = patch_meta["question_positions"]
    selected_frame_indices, frame_patch_positions = gri._select_frame_patch_positions(
        frame_groups=frame_groups, evidence_frame_indices=evidence, frame_patch_mode=frame_patch_mode)

    if corruption_mode == "surgical":
        corrupted_inputs = build_surgical_corrupted_inputs(sample_id, frames, question, evidence, prompt_len, corrupted_root)
        if corrupted_inputs is None:
            print(f"[{idx}/{total}] {sample_id} skipped: missing surgical corrupted assets")
            return None
    else:
        corrupted_inputs = build_blank_corrupted_inputs(frames, question, evidence, prompt_len, corruption_mode)
        if corrupted_inputs is None:
            print(f"[{idx}/{total}] {sample_id} skipped: corrupted seq_len mismatch")
            return None

    corr_res = gri.score_corrupted_sample(sample_id, clean_inputs, corrupted_inputs, a_ids, prompt_len, lm, idx, total)
    if corr_res is None:
        return None
    clean_answer_inputs = corr_res["clean_answer_inputs"]
    corrupted_answer_inputs = corr_res["corrupted_answer_inputs"]
    corrupted_answer_score = float(corr_res["corrupted_answer_score"])
    denominator = clean_answer_score - corrupted_answer_score

    print(f"[{idx}/{total}] {sample_id} task={task} gold={answer_text} clean={clean_answer_score:.3f} "
          f"corrupt={corrupted_answer_score:.3f} denom={denominator:.3f} evidence={evidence} "
          f"frame_patch={selected_frame_indices}")

    group_positions = {
        "frames": [int(p) for p in frame_patch_positions],
        "last_token": [-1],
        "question": [int(p) for p in question_positions],
    }
    dual_rows = compute_dual_direction(
        lm, layers, selected_layers, clean_answer_inputs, corrupted_answer_inputs,
        clean_answer_score, corrupted_answer_score, prompt_len, a_ids, group_positions)
    # attach sample-level fields to each row
    for r in dual_rows:
        r["sample_id"] = sample_id
        r["denominator"] = float(denominator)
        r["clean_score"] = float(clean_answer_score)
        r["corrupted_score"] = float(corrupted_answer_score)

    return {
        "sample_id": sample_id, "seq_len": int(len(frames)), "question": question, "answer": answer_text,
        "clean_answer_score": clean_answer_score, "corrupted_answer_score": corrupted_answer_score,
        "clean_top1_correct": bool(clean_metrics["clean_top1_correct"]), "denominator": denominator,
        "evidence_frames": [int(i) for i in evidence],
        "frame_patch_frames": [int(i) for i in selected_frame_indices],
        "carrier_index": carrier_index, "selected_layers": [int(l) for l in selected_layers],
        "rows": dual_rows,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Asset-free token-group corruption/restoration for new MMReD tasks.")
    ap.add_argument("--task", choices=["rooms_visited", "co_occupancy", "count"], required=True)
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--output", type=str, default="outputs")
    ap.add_argument("--layers", type=str, default=None)
    ap.add_argument("--model_name", "--model", dest="model_name", type=str, default=DEFAULT_MODEL_ID)
    ap.add_argument("--frame_patch_mode", type=str, default="evidence_only",
                    choices=["evidence_only", "all_frames", "non_evidence_only"])
    ap.add_argument("--corruption_mode", type=str, default="evidence", choices=["evidence", "all_frames", "surgical"])
    ap.add_argument("--corrupted_root", type=str, default=None,
                    help="for --corruption_mode surgical: root with <sample_id>/corrupted_frame_{t}/ renders")
    ap.add_argument("--clean_top1_must_match_gold",
                    type=lambda r: str(r).strip().lower() in {"1", "true", "yes", "y", "on"}, default=False)
    ap.add_argument("--disable_plots", action="store_true")
    return ap.parse_args()


def main() -> None:
    start = time.perf_counter()
    args = parse_args()
    gri.configure_runtime(args.model_name)
    data_root = Path(args.data_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    seq_len_label = eval_utils.resolve_seq_len_label(data_root)

    lm = LanguageModel(gri._model(), tokenizer=gri._processor().tokenizer)
    layers = get_layers(lm.model)
    selected_layers = gri.parse_layer_selection(args.layers, num_layers=len(layers))

    sample_dirs = iter_sample_dirs(data_root)
    if not sample_dirs:
        raise RuntimeError(f"No sample directories under: {data_root}")

    collected: List[Dict[str, Any]] = []
    for idx, sample_dir in enumerate(sample_dirs, start=1):
        if len(collected) >= int(args.limit):
            break
        try:
            row = process_sample(sample_dir, idx, len(sample_dirs), args.task, lm, layers, selected_layers,
                                 args.frame_patch_mode, args.corruption_mode, bool(args.clean_top1_must_match_gold),
                                 corrupted_root=Path(args.corrupted_root) if args.corrupted_root else None)
        except Exception as exc:
            print(f"[{idx}/{len(sample_dirs)}] {sample_dir.name} skipped: unexpected ({exc})")
            continue
        if row is None:
            continue
        row["selected_layers_spec"] = args.layers
        row["model_name"] = gri._runtime().model_name
        row["control_debug"] = {"control_type": (f"surgical_charremoved" if args.corruption_mode == "surgical"
                                                  else f"blank_{args.corruption_mode}_frames"), "task": args.task}
        collected.append(row)

    # long-format CSV: one row per (sample, layer, token-group, metric_type) — both restoration & clean_ablation
    import csv as _csv
    all_rows = [r for row in collected for r in row["rows"]]
    csv_path = output_dir / "patch_metrics_long.csv"
    fields = ["sample_id", "layer", "patch_target", "metric_type", "raw_value", "normalized_value",
              "denominator", "clean_score", "corrupted_score"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k) for k in fields})
    import json as _json
    (output_dir / "sample_metrics.json").write_text(_json.dumps(collected, indent=2) + "\n", encoding="utf-8")
    print(f"task={args.task} collected={len(collected)} rows={len(all_rows)} -> {csv_path}")
    print(eval_utils.format_runtime(time.perf_counter() - start))


if __name__ == "__main__":
    main()
