#!/usr/bin/env python3
"""
MMReD frame renderer (spec-compliant)

Spec:
- 512x512 image
- 2x3 grid of room rectangles, room name at bottom
- characters are colored circles with names above
- step number at bottom
Deterministic rules:
- Fixed room layout order:
    Top:    Kitchen, Bathroom, Garden
    Bottom: Office, Bedroom, Hallway
- Fixed 6 anchor positions per room
- Fixed per-character colors

Also:
- Only renders examples with qtype == "steps_in_room"
- Writes qa.txt next to images with the original question/answer metadata.
"""

import os
import ast
import argparse
import re
import shutil
from typing import Dict, List, Tuple

from datasets import load_dataset, load_from_disk
from PIL import Image, ImageDraw, ImageFont

# Fixed room order (spec)
ROOMS: List[str] = ["Kitchen", "Bathroom", "Garden", "Office", "Bedroom", "Hallway"]

# Fixed character colors (deterministic map; constant across all frames)
CHAR_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Sandra": (231, 76, 60),     # red-ish
    "Mary": (46, 204, 113),      # green-ish
    "Michael": (52, 152, 219),   # blue-ish
    "John": (155, 89, 182),      # purple-ish
    "Daniel": (241, 196, 15),    # yellow-ish
}

IMG_SIZE = 512

def load_font(size: int) -> ImageFont.ImageFont:
    # DejaVuSans is commonly available on Linux. Falls back to default bitmap font.
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()

def room_grid_boxes(img: int = IMG_SIZE, pad: int = 24, gap: int = 14) -> Dict[str, Tuple[int,int,int,int]]:
    """
    Create a 2x3 grid of rectangles with fixed room order.

    Top row: Kitchen, Bathroom, Garden
    Bottom row: Office, Bedroom, Hallway

    Returns dict: room -> (x0, y0, x1, y1)
    """
    # Leave space at bottom for step number text
    bottom_margin = 36
    grid_w = img - 2 * pad
    grid_h = img - 2 * pad - bottom_margin

    cols, rows = 3, 2
    cell_w = (grid_w - (cols - 1) * gap) // cols
    cell_h = (grid_h - (rows - 1) * gap) // rows

    boxes: Dict[str, Tuple[int,int,int,int]] = {}
    idx = 0
    for r in range(rows):
        for c in range(cols):
            x0 = pad + c * (cell_w + gap)
            y0 = pad + r * (cell_h + gap)
            x1 = x0 + cell_w
            y1 = y0 + cell_h
            boxes[ROOMS[idx]] = (x0, y0, x1, y1)
            idx += 1
    return boxes

def room_anchor_slots(room_box, n_slots=6):
    x0, y0, x1, y1 = room_box
    inner = 18

    # usable region inside room (leave room for names + room label)
    rx0, ry0 = x0 + inner, y0 + inner + 18
    rx1, ry1 = x1 - inner, y1 - inner - 26

    cols, rows = 2, 3

    # centers for rows, then spread them further apart deterministically
    # row_spacing > 1 increases vertical gaps between rows
    row_spacing = 1.20

    # compute base row centers in [0,1]
    base_row_centers = [(r + 0.5) / rows for r in range(rows)]
    # re-center around 0.5 then scale
    adj_row_centers = [0.5 + (c - 0.5) * row_spacing for c in base_row_centers]
    # clamp so we never go outside the room
    adj_row_centers = [min(0.95, max(0.05, c)) for c in adj_row_centers]

    slots = []
    for r in range(rows):
        for c in range(cols):
            cx = rx0 + (c + 0.5) * (rx1 - rx0) / cols
            cy = ry0 + adj_row_centers[r] * (ry1 - ry0)
            slots.append((int(cx), int(cy)))

    return slots[:n_slots]


def parse_question_states(question: str) -> List[Dict]:
    """
    Parse per-step states encoded as Python-dict-per-line at the top of `question`.

    We ONLY parse lines that look like dicts and include a rooms field.
    This avoids crashing on the natural-language question line(s).
    """
    states: List[Dict] = []
    for line in question.splitlines():
        line = line.strip()
        if not line:
            continue
        if not line.startswith("{"):
            continue
        if "'rooms'" not in line and '"rooms"' not in line:
            continue

        try:
            d = ast.literal_eval(line)
        except Exception:
            continue

        if isinstance(d, dict) and "rooms" in d:
            states.append(d)

    if not states:
        raise ValueError("No step states found in question text.")
    return states

def render_frame(rooms: Dict[str, List[str]], step_id: int, out_path: str) -> None:
    """
    Render a single 512x512 frame according to spec.
    """
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    font_room = load_font(16)
    font_name = load_font(12)  # slightly smaller to reduce collision risk
    font_step = load_font(16)

    boxes = room_grid_boxes()

    # Draw room rectangles + room names at bottom
    for room, box in boxes.items():
        x0, y0, x1, y1 = box
        draw.rectangle(box, outline=(0, 0, 0), width=2)

        tw, th = draw.textbbox((0, 0), room, font=font_room)[2:]
        draw.text((x0 + (x1 - x0 - tw) // 2, y1 - th - 4), room, fill=(0, 0, 0), font=font_room)

    # Draw characters as colored circles with names above (deterministic anchors)
    circle_r = 16
    for room in ROOMS:
        box = boxes[room]
        slots = room_anchor_slots(box, n_slots=6)

        # deterministic ordering within room:
        chars = sorted(rooms.get(room, []))
        for i, ch in enumerate(chars[:6]):
            cx, cy = slots[i]
            color = CHAR_COLORS.get(ch, (120, 120, 120))

            # circle
            draw.ellipse((cx - circle_r, cy - circle_r, cx + circle_r, cy + circle_r),
                         fill=color, outline=(0, 0, 0), width=2)

            # name above
            tw, th = draw.textbbox((0, 0), ch, font=font_name)[2:]
            draw.text((cx - tw // 2, cy - circle_r - th - 2), ch, fill=(0, 0, 0), font=font_name)

    # Step number at bottom (outside rooms)
    step_text = f"Step {step_id}"
    tw, th = draw.textbbox((0, 0), step_text, font=font_step)[2:]
    draw.text(((IMG_SIZE - tw) // 2, IMG_SIZE - th - 6), step_text, fill=(0, 0, 0), font=font_step)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)

def write_qa_txt(sample_dir: str, ex: Dict) -> None:
    path = os.path.join(sample_dir, "qa.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"qid: {ex.get('qid')}\n")
        f.write(f"qtype: {ex.get('qtype')}\n")
        f.write(f"atype: {ex.get('atype')}\n")
        f.write(f"seq_len: {ex.get('seq_len')}\n")
        f.write("question:\n")
        f.write(ex.get("question", ""))
        if not ex.get("question", "").endswith("\n"):
            f.write("\n")
        f.write("answer:\n")
        f.write(str(ex.get("answer")) + "\n")


# ---------------------------
# Corruption helpers
# ---------------------------

_STEPS_IN_ROOM_RE = re.compile(
    r"How many steps did\s+([A-Za-z]+)\s+spend in\s+the\s+([A-Za-z]+)",
    flags=re.IGNORECASE,
)

def parse_target_character_room(question_text: str) -> Tuple[str, str]:
    """
    Extract (C, R) from the natural-language question line:
      "How many steps did [C] spend in the [R] ..."

    Returns (character, room) with room capitalized to match ROOMS list.
    """
    # Search the full question text (usually the last line).
    m = _STEPS_IN_ROOM_RE.search(question_text)
    if not m:
        raise ValueError("Could not parse target character/room from question text.")
    c = m.group(1).strip()
    r = m.group(2).strip()
    # Normalize room capitalization to match renderer's ROOMS naming (Kitchen, Bathroom, ...)
    r_norm = r[:1].upper() + r[1:].lower()
    return c, r_norm

def split_question_into_states_and_tail(question_text: str) -> Tuple[List[Dict], List[str]]:
    """
    Returns (states, tail_lines) where tail_lines are the non-dict lines
    that come after the step-state dicts (e.g., the natural-language question).
    """
    lines = question_text.splitlines()
    states: List[Dict] = []
    last_state_line_idx = -1
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or not s.startswith("{"):
            continue
        if "'rooms'" not in s and '"rooms"' not in s:
            continue
        try:
            d = ast.literal_eval(s)
        except Exception:
            continue
        if isinstance(d, dict) and "rooms" in d:
            states.append(d)
            last_state_line_idx = i
    if not states:
        raise ValueError("No step states found in question text.")
    tail_lines = lines[last_state_line_idx + 1:] if last_state_line_idx >= 0 else []
    return states, tail_lines

def rooms_to_room2chars(rooms: Dict) -> Dict[str, List[str]]:
    """
    MMReD step 'rooms' sometimes may come as:
      - room -> [chars]
      - char -> room
    Renderer expects room -> [chars]. This normalizes to that.
    """
    if not isinstance(rooms, dict):
        raise ValueError("rooms field is not a dict.")

    # Heuristic: if any value is a list, assume room->list
    if any(isinstance(v, list) for v in rooms.values()):
        # rooms are already room->list, but keys might be lower/upper/mixed.
        out: Dict[str, List[str]] = {r: [] for r in ROOMS}
        for k, v in rooms.items():
            if not isinstance(k, str):
                continue
            rk = k.strip()
            rk_norm = rk[:1].upper() + rk[1:].lower() if rk else rk
            if rk_norm not in out:
                # keep unknown rooms too (but normalized)
                out.setdefault(rk_norm, [])
            if isinstance(v, list):
                out[rk_norm].extend([str(x) for x in v])
        # Sort + dedup for determinism
        for r in out:
            out[r] = sorted(set(out[r]))
        return out

    # Otherwise assume char->room
    out: Dict[str, List[str]] = {r: [] for r in ROOMS}
    for ch, rm in rooms.items():
        if not isinstance(rm, str):
            continue
        rm_norm = rm[:1].upper() + rm[1:].lower()
        out.setdefault(rm_norm, []).append(str(ch))
    # Sort for determinism
    for r in out:
        out[r] = sorted(out[r])
    return out

def char_in_room(step_rooms: Dict, character: str, room: str) -> bool:
    r2c = rooms_to_room2chars(step_rooms)
    return character in r2c.get(room, [])

def corrupt_step_rooms(step_rooms: Dict, character: str, room: str) -> Dict[str, List[str]]:
    """
    Remove `character` from `room` in this step's rooms representation.
    Returns a NEW normalized room->list dict.
    """
    r2c = rooms_to_room2chars(step_rooms)
    new = {k: list(v) for k, v in r2c.items()}
    if room in new and character in new[room]:
        new[room] = [c for c in new[room] if c != character]
    # Keep deterministic ordering
    for r in new:
        new[r] = sorted(new[r])
    return new

def write_corrupted_qa_txt(out_dir: str, orig_qa: Dict, new_question_text: str) -> None:
    """
    Write qa.txt in the same format as write_qa_txt(), but with a corrupted question.
    """
    path = os.path.join(out_dir, "qa.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"qid: {orig_qa.get('qid')}\n")
        f.write(f"qtype: {orig_qa.get('qtype')}\n")
        f.write(f"atype: {orig_qa.get('atype')}\n")
        f.write(f"seq_len: {orig_qa.get('seq_len')}\n")
        f.write("question:\n")
        f.write(new_question_text)
        if not new_question_text.endswith("\n"):
            f.write("\n")
        f.write("answer:\n")
        f.write(str(orig_qa.get("answer")) + "\n")

def read_sample_qa(sample_dir: str) -> Dict:
    """
    Parse qa.txt written by write_qa_txt() back into a dict with keys:
      qid, qtype, atype, seq_len, question, answer
    """
    qa_path = os.path.join(sample_dir, "qa.txt")
    if not os.path.exists(qa_path):
        raise FileNotFoundError(f"qa.txt not found in sample_dir: {sample_dir}")

    meta: Dict[str, str] = {}
    question_lines: List[str] = []
    answer_lines: List[str] = []

    mode = None
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("qid:"):
                meta["qid"] = line.split(":", 1)[1].strip()
            elif line.startswith("qtype:"):
                meta["qtype"] = line.split(":", 1)[1].strip()
            elif line.startswith("atype:"):
                meta["atype"] = line.split(":", 1)[1].strip()
            elif line.startswith("seq_len:"):
                try:
                    meta["seq_len"] = int(line.split(":", 1)[1].strip())
                except Exception:
                    meta["seq_len"] = line.split(":", 1)[1].strip()
            elif line.strip() == "question:":
                mode = "question"
            elif line.strip() == "answer:":
                mode = "answer"
            else:
                if mode == "question":
                    question_lines.append(line.rstrip("\n"))
                elif mode == "answer":
                    answer_lines.append(line.rstrip("\n"))

    meta["question"] = "\n".join(question_lines).rstrip("\n")
    meta["answer"] = "\n".join(answer_lines).strip()
    return meta

def generate_corrupted_sample_from_rendered(
    sample_dir: str,
    corrupt_frame_idx: int,
    character: str,
    room: str,
    out_root: str,
    split: str,
) -> str:
    """
    Given a rendered sample folder (with frames + qa.txt), generate a NEW corrupted sample folder:
      out_root/seq_len_{x}/{split}/{sample_id}/corrupted_frame_{y}/

    It re-renders ALL frames, but applies the corruption ONLY to frame y by removing `character`
    from `room` in that step's rooms state.

    Returns the output directory path.
    """
    qa = read_sample_qa(sample_dir)
    states, tail_lines = split_question_into_states_and_tail(qa["question"])
    seq_len = len(states)

    if corrupt_frame_idx < 0 or corrupt_frame_idx >= seq_len:
        raise ValueError(f"corrupt_frame_idx {corrupt_frame_idx} out of range for seq_len {seq_len}")

    # Apply corruption to the selected frame, keeping other frames identical.
    new_states: List[Dict] = []
    for t, s in enumerate(states):
        s_new = dict(s)
        if t == corrupt_frame_idx:
            s_new["rooms"] = corrupt_step_rooms(s_new["rooms"], character, room)
        else:
            # normalize for rendering consistency
            s_new["rooms"] = rooms_to_room2chars(s_new["rooms"])
        new_states.append(s_new)

    # Rebuild the question text: dict-per-line + tail (natural language line(s))
    # Use repr() to keep it python-literal parseable by ast.literal_eval later.
    dict_lines = [repr(s) for s in new_states]
    new_question_text = "\n".join(dict_lines + tail_lines).rstrip("\n") + "\n"

    sample_id = os.path.basename(os.path.normpath(sample_dir))
    out_dir = os.path.join(
        out_root,
        f"seq_len_{seq_len}",
        split,
        sample_id,
        f"corrupted_frame_{corrupt_frame_idx}",
    )

    os.makedirs(out_dir, exist_ok=True)

    # Render frames
    for t, s in enumerate(new_states):
        step_id = int(s.get("step_id", t + 1))
        rooms_norm = rooms_to_room2chars(s["rooms"])
        render_frame(rooms_norm, step_id, os.path.join(out_dir, f"{t:03d}.png"))

    # Write qa.txt (corrupted)
    qa_out = {
        "qid": qa.get("qid", sample_id),
        "qtype": qa.get("qtype"),
        "atype": qa.get("atype"),
        "seq_len": qa.get("seq_len", seq_len),
        "answer": qa.get("answer"),
    }
    write_corrupted_qa_txt(out_dir, qa_out, new_question_text)

    return out_dir

def create_all_corruptions_for_sample(sample_dir: str, out_root: str, split: str) -> int:
    """
    For a given rendered sample_dir, create corrupted samples for each evidence frame:
    a frame t is an evidence frame if character C is in room R at step t.

    Returns number of corruptions created.
    """
    qa = read_sample_qa(sample_dir)
    print(f"Processing sample_id={qa.get('qid')} with qtype={qa.get('qtype')}")
    states, _ = split_question_into_states_and_tail(qa["question"])
    c, r = parse_target_character_room(qa["question"])
    seq_len = len(states)
    print(f" - target character: {c}, target room: {r}")
    print(f" - total steps in room (evidence frames): {sum(1 for s in states if char_in_room(s['rooms'], c, r))} out of {seq_len}")

    evidence = [t for t, s in enumerate(states) if char_in_room(s["rooms"], c, r)]
    created = 0
    for t in evidence:
        generate_corrupted_sample_from_rendered(
            sample_dir=sample_dir,
            corrupt_frame_idx=t,
            character=c,
            room=r,
            out_root=out_root,
            split=split,
        )
        created += 1
    return created

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="seq_len_16")
    ap.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    ap.add_argument(
        "--dataset-root",
        default=None,
        help="Optional local HF dataset root containing seq_len_* folders saved via DatasetDict.save_to_disk.",
    )
    ap.add_argument("--out", default="data/mmred_images")
    ap.add_argument("--limit", type=int, default=10, help="max rendered samples AFTER qtype filter")
    ap.add_argument("--corrupt_out", default="data/mmred_corrupted", help="Output root for corrupted samples")
    args = ap.parse_args()

    split_names = ["train", "val", "test"] if args.split == "all" else [args.split]
    out_split_name = "all" if args.split == "all" else args.split
    rendered_root = os.path.join(args.out, args.config, out_split_name)
    if args.split == "all":
        if os.path.isdir(rendered_root):
            shutil.rmtree(rendered_root)
        corrupt_all_dir = os.path.join(args.corrupt_out, args.config, "all")
        if os.path.isdir(corrupt_all_dir):
            shutil.rmtree(corrupt_all_dir)

    rendered = 0
    rendered_sample_dirs: List[str] = []
    for split_name in split_names:
        if args.dataset_root:
            ds_dict = load_from_disk(os.path.join(args.dataset_root, args.config))
            ds = ds_dict[split_name]
        else:
            ds = load_dataset("ef1e43ce/mmred", args.config, split=split_name)
        for idx in range(len(ds)):
            ex = ds[idx]

            # Only qtype == steps_in_room (as requested)
            if ex.get("qtype") != "steps_in_room":
                continue

            states = parse_question_states(ex["question"])
            qid = str(ex.get("qid", idx))
            sample_dir = os.path.join(args.out, args.config, out_split_name, qid)

            for t, s in enumerate(states):
                step_id = int(s.get("step_id", t + 1))
                rooms = s["rooms"]
                render_frame(rooms, step_id, os.path.join(sample_dir, f"{t:03d}.png"))

            write_qa_txt(sample_dir, ex)
            rendered_sample_dirs.append(sample_dir)

            rendered += 1
            if rendered >= args.limit:
                break
        if rendered >= args.limit:
            break

    print(f"Rendered {rendered} samples (qtype=steps_in_room) into: {os.path.abspath(rendered_root)}")

    total_created = 0
    for sd in rendered_sample_dirs:
        try:
            total_created += create_all_corruptions_for_sample(sd, args.corrupt_out, out_split_name)
        except Exception as e:
            print(f"[WARN] skipping corruption for {sd}: {e}")

    print(f"Created {total_created} corrupted sample folders under: {os.path.abspath(args.corrupt_out)}")

if __name__ == "__main__":
    main()
