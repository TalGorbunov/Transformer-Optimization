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
from typing import Dict, List, Tuple

from datasets import load_dataset
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="seq_len_16")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--out", default="data/mmred_images")
    ap.add_argument("--limit", type=int, default=10, help="max rendered samples AFTER qtype filter")
    args = ap.parse_args()

    ds = load_dataset("ef1e43ce/mmred", args.config, split=args.split)

    rendered = 0
    for idx in range(len(ds)):
        ex = ds[idx]

        # Only qtype == steps_in_room (as requested)
        if ex.get("qtype") != "steps_in_room":
            continue

        states = parse_question_states(ex["question"])
        qid = str(ex.get("qid", idx))
        sample_dir = os.path.join(args.out, args.config, args.split, qid)

        for t, s in enumerate(states):
            step_id = int(s.get("step_id", t + 1))
            rooms = s["rooms"]
            render_frame(rooms, step_id, os.path.join(sample_dir, f"{t:03d}.png"))

        write_qa_txt(sample_dir, ex)

        rendered += 1
        if rendered >= args.limit:
            break

    print(f"Rendered {rendered} samples (qtype=steps_in_room) into: {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
