#!/usr/bin/env python3
"""
Standalone MMReD mosaic renderer.

This renderer keeps the original per-frame drawing logic intact in spirit:
each step is rendered as a normal frame first, then the already-rendered frames
are resized and pasted into one final mosaic image named 000.png.

Supported layouts: seq_len=2 -> 1x2, seq_len=3-4 -> 2x2,
seq_len=5-9 -> 3x3. Extra cells are left blank white.
"""

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw

from render_mmred import (
    CHAR_COLORS,
    IMG_SIZE,
    ROOMS,
    char_in_room,
    dataset_relative_path,
    load_font,
    load_plain_dataset,
    parse_question_states,
    parse_target_character_room,
    read_sample_qa,
    resolve_dataset_roots,
    room_anchor_slots,
    room_grid_boxes,
    rooms_to_room2chars,
    split_question_into_states_and_tail,
    corrupt_step_rooms,
    write_corrupted_qa_txt,
    write_qa_txt,
)


MOSAIC_GUTTER_PX = 8
MOSAIC_BACKGROUND = (255, 255, 255)
SUPPORTED_MOSAIC_LAYOUTS: Dict[int, Tuple[int, int]] = {
    2: (1, 2),
    3: (2, 2),
    4: (2, 2),
    5: (3, 3),
    6: (3, 3),
    7: (3, 3),
    8: (3, 3),
    9: (3, 3),
}
_MOSAIC_SIZE_RE = re.compile(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$")

# Frame readability tuning. These preserve the original room layout and
# deterministic placement while making resized mosaic tiles easier to read.
ROOM_FONT_SIZE = 18
CHARACTER_FONT_SIZE = 15
STEP_FONT_SIZE = 17
CHARACTER_CIRCLE_RADIUS = 17
ROOM_OUTLINE_WIDTH = 3
CIRCLE_OUTLINE_WIDTH = 3
ROOM_GRID_PADDING = 20
ROOM_GRID_GAP = 12
ROOM_LABEL_BOTTOM_PADDING = 5
CHARACTER_NAME_GAP = 4
STEP_BOTTOM_PADDING = 7

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_LANCZOS = Image.LANCZOS


class MosaicRenderError(ValueError):
    pass


def format_mosaic_size(mosaic_size: Tuple[int, int]) -> str:
    return f"{mosaic_size[0]}x{mosaic_size[1]}"


def format_rendering_constants_summary() -> str:
    return (
        "Frame readability constants: "
        f"room_font={ROOM_FONT_SIZE}, char_font={CHARACTER_FONT_SIZE}, step_font={STEP_FONT_SIZE}, "
        f"circle_radius={CHARACTER_CIRCLE_RADIUS}, room_outline={ROOM_OUTLINE_WIDTH}, "
        f"circle_outline={CIRCLE_OUTLINE_WIDTH}, room_padding={ROOM_GRID_PADDING}, room_gap={ROOM_GRID_GAP}"
    )


def parse_mosaic_size(value: str) -> Tuple[int, int]:
    match = _MOSAIC_SIZE_RE.match(value)
    if not match:
        raise argparse.ArgumentTypeError(
            f"invalid mosaic size {value!r}; expected WIDTHxHEIGHT, e.g. 1024x512"
        )

    width = int(match.group(1))
    height = int(match.group(2))
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError(
            f"invalid mosaic size {value!r}; WIDTH and HEIGHT must be positive integers"
        )
    return width, height


def validate_seq_len(seq_len: int) -> None:
    if seq_len not in SUPPORTED_MOSAIC_LAYOUTS:
        supported = f"{min(SUPPORTED_MOSAIC_LAYOUTS)} through {max(SUPPORTED_MOSAIC_LAYOUTS)}"
        raise MosaicRenderError(
            f"Unsupported seq_len={seq_len} for mosaic rendering; supported seq_len values are {supported}."
        )


def render_frame_image(rooms: Dict[str, List[str]], step_id: int) -> Image.Image:
    """
    Render a single 512x512 frame according to the original frame spec.
    """
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), MOSAIC_BACKGROUND)
    draw = ImageDraw.Draw(img)

    font_room = load_font(ROOM_FONT_SIZE)
    font_name = load_font(CHARACTER_FONT_SIZE)
    font_step = load_font(STEP_FONT_SIZE)

    boxes = room_grid_boxes(pad=ROOM_GRID_PADDING, gap=ROOM_GRID_GAP)

    # Draw room rectangles + room names at bottom.
    for room, box in boxes.items():
        x0, y0, x1, y1 = box
        draw.rectangle(box, outline=(0, 0, 0), width=ROOM_OUTLINE_WIDTH)

        tw, th = draw.textbbox((0, 0), room, font=font_room)[2:]
        draw.text(
            (x0 + (x1 - x0 - tw) // 2, y1 - th - ROOM_LABEL_BOTTOM_PADDING),
            room,
            fill=(0, 0, 0),
            font=font_room,
        )

    # Draw characters as colored circles with names above (deterministic anchors).
    for room in ROOMS:
        box = boxes[room]
        slots = room_anchor_slots(box, n_slots=6)

        chars = sorted(rooms.get(room, []))
        for i, ch in enumerate(chars[:6]):
            cx, cy = slots[i]
            color = CHAR_COLORS.get(ch, (120, 120, 120))

            draw.ellipse(
                (
                    cx - CHARACTER_CIRCLE_RADIUS,
                    cy - CHARACTER_CIRCLE_RADIUS,
                    cx + CHARACTER_CIRCLE_RADIUS,
                    cy + CHARACTER_CIRCLE_RADIUS,
                ),
                fill=color,
                outline=(0, 0, 0),
                width=CIRCLE_OUTLINE_WIDTH,
            )

            tw, th = draw.textbbox((0, 0), ch, font=font_name)[2:]
            draw.text(
                (cx - tw // 2, cy - CHARACTER_CIRCLE_RADIUS - th - CHARACTER_NAME_GAP),
                ch,
                fill=(0, 0, 0),
                font=font_name,
            )

    # Step number at bottom (outside rooms).
    step_text = f"Step {step_id}"
    tw, th = draw.textbbox((0, 0), step_text, font=font_step)[2:]
    draw.text(
        ((IMG_SIZE - tw) // 2, IMG_SIZE - th - STEP_BOTTOM_PADDING),
        step_text,
        fill=(0, 0, 0),
        font=font_step,
    )

    return img


def resize_frame_to_cell(frame: Image.Image, cell_size: Tuple[int, int]) -> Image.Image:
    cell_w, cell_h = cell_size
    scale = min(cell_w / frame.width, cell_h / frame.height)
    target_w = max(1, int(round(frame.width * scale)))
    target_h = max(1, int(round(frame.height * scale)))
    return frame.convert("RGB").resize((target_w, target_h), RESAMPLE_LANCZOS)


def compose_mosaic(
    frame_images: Sequence[Image.Image],
    mosaic_size: Tuple[int, int],
    gutter_px: int = MOSAIC_GUTTER_PX,
) -> Image.Image:
    seq_len = len(frame_images)
    validate_seq_len(seq_len)

    width, height = mosaic_size
    rows, cols = SUPPORTED_MOSAIC_LAYOUTS[seq_len]
    available_w = width - gutter_px * (cols - 1)
    available_h = height - gutter_px * (rows - 1)
    if available_w < cols or available_h < rows:
        raise MosaicRenderError(
            f"--mosaic-size {format_mosaic_size(mosaic_size)} is too small for seq_len={seq_len} "
            f"with gutter={gutter_px}px"
        )

    cell_w = available_w // cols
    cell_h = available_h // rows
    grid_w = cell_w * cols + gutter_px * (cols - 1)
    grid_h = cell_h * rows + gutter_px * (rows - 1)
    origin_x = (width - grid_w) // 2
    origin_y = (height - grid_h) // 2

    mosaic = Image.new("RGB", (width, height), MOSAIC_BACKGROUND)
    for idx, frame in enumerate(frame_images):
        row, col = divmod(idx, cols)
        resized = resize_frame_to_cell(frame, (cell_w, cell_h))
        cell_x = origin_x + col * (cell_w + gutter_px)
        cell_y = origin_y + row * (cell_h + gutter_px)
        paste_x = cell_x + (cell_w - resized.width) // 2
        paste_y = cell_y + (cell_h - resized.height) // 2
        mosaic.paste(resized, (paste_x, paste_y))

    return mosaic


def render_states_mosaic(states: Sequence[Dict], mosaic_size: Tuple[int, int]) -> Image.Image:
    seq_len = len(states)
    validate_seq_len(seq_len)

    frame_images: List[Image.Image] = []
    for t, state in enumerate(states):
        step_id = int(state.get("step_id", t + 1))
        rooms = rooms_to_room2chars(state["rooms"])
        frame_images.append(render_frame_image(rooms, step_id))

    return compose_mosaic(frame_images, mosaic_size)


def save_states_mosaic(states: Sequence[Dict], out_path: Path, mosaic_size: Tuple[int, int]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    render_states_mosaic(states, mosaic_size).save(out_path)


def generate_corrupted_sample_from_rendered(
    sample_dir: Path,
    corrupt_frame_idx: int,
    character: str,
    room: str,
    out_root: Path,
    dataset_rel_dir: Path,
    mosaic_size: Tuple[int, int],
) -> str:
    """
    Given a rendered sample folder (with 000.png + qa.txt), generate a NEW corrupted sample folder:
      out_root/<dataset_rel_dir>/<sample_id>/corrupted_frame_{y}/

    It re-renders ALL frames, but applies the corruption ONLY to frame y by removing `character`
    from `room` in that step's rooms state, then composes the frames into one 000.png mosaic.
    """
    qa = read_sample_qa(str(sample_dir))
    states, tail_lines = split_question_into_states_and_tail(qa["question"])
    seq_len = len(states)
    validate_seq_len(seq_len)

    if corrupt_frame_idx < 0 or corrupt_frame_idx >= seq_len:
        raise ValueError(f"corrupt_frame_idx {corrupt_frame_idx} out of range for seq_len {seq_len}")

    # Apply corruption to the selected frame, keeping other frames identical.
    new_states: List[Dict] = []
    for t, state in enumerate(states):
        state_new = dict(state)
        if t == corrupt_frame_idx:
            state_new["rooms"] = corrupt_step_rooms(state_new["rooms"], character, room)
        else:
            state_new["rooms"] = rooms_to_room2chars(state_new["rooms"])
        new_states.append(state_new)

    # Rebuild the question text: dict-per-line + tail (natural language line(s)).
    # Use repr() to keep it python-literal parseable by ast.literal_eval later.
    dict_lines = [repr(state) for state in new_states]
    new_question_text = "\n".join(dict_lines + tail_lines).rstrip("\n") + "\n"

    sample_id = sample_dir.name
    out_dir = out_root / dataset_rel_dir / sample_id / f"corrupted_frame_{corrupt_frame_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_states_mosaic(new_states, out_dir / "000.png", mosaic_size)

    qa_out = {
        "qid": qa.get("qid", sample_id),
        "qtype": qa.get("qtype"),
        "atype": qa.get("atype"),
        "seq_len": qa.get("seq_len", seq_len),
        "answer": qa.get("answer"),
    }
    write_corrupted_qa_txt(os.fspath(out_dir), qa_out, new_question_text)

    return os.fspath(out_dir)


def create_all_corruptions_for_sample(
    sample_dir: Path,
    out_root: Path,
    dataset_rel_dir: Path,
    mosaic_size: Tuple[int, int],
) -> int:
    """
    For a given rendered sample_dir, create corrupted samples for each evidence frame:
    a frame t is an evidence frame if character C is in room R at step t.

    Returns number of corruptions created.
    """
    qa = read_sample_qa(str(sample_dir))
    print(f"Processing sample_id={qa.get('qid')} with qtype={qa.get('qtype')}")
    states, _ = split_question_into_states_and_tail(qa["question"])
    validate_seq_len(len(states))
    character, room = parse_target_character_room(qa["question"])
    seq_len = len(states)
    evidence_count = sum(1 for state in states if char_in_room(state["rooms"], character, room))
    print(f" - target character: {character}, target room: {room}")
    print(f" - total steps in room (evidence frames): {evidence_count} out of {seq_len}")

    evidence = [t for t, state in enumerate(states) if char_in_room(state["rooms"], character, room)]
    created = 0
    for t in evidence:
        generate_corrupted_sample_from_rendered(
            sample_dir=sample_dir,
            corrupt_frame_idx=t,
            character=character,
            room=room,
            out_root=out_root,
            dataset_rel_dir=dataset_rel_dir,
            mosaic_size=mosaic_size,
        )
        created += 1
    return created


def render_dataset(
    dataset_root: Path,
    out_root: Path,
    corrupt_out_root: Path,
    limit: int,
    mosaic_size: Tuple[int, int],
) -> Tuple[int, int]:
    dataset_rel = dataset_relative_path(dataset_root)
    rendered_root = out_root / dataset_rel
    corrupt_root = corrupt_out_root / dataset_rel

    if rendered_root.is_dir():
        shutil.rmtree(rendered_root)
    if corrupt_root.is_dir():
        shutil.rmtree(corrupt_root)

    ds = load_plain_dataset(dataset_root)

    rendered = 0
    rendered_sample_dirs: List[Path] = []
    for idx in range(len(ds)):
        ex = ds[idx]

        # Only qtype == steps_in_room (as requested).
        if ex.get("qtype") != "steps_in_room":
            continue

        states = parse_question_states(ex["question"])
        validate_seq_len(len(states))
        qid = str(ex.get("qid") or ex.get("sample_id") or idx)
        sample_dir = rendered_root / qid

        save_states_mosaic(states, sample_dir / "000.png", mosaic_size)
        write_qa_txt(os.fspath(sample_dir), ex)
        rendered_sample_dirs.append(sample_dir)

        rendered += 1
        if rendered >= limit:
            break

    print(
        f"Rendered {rendered} mosaic samples (qtype=steps_in_room, "
        f"mosaic_size={format_mosaic_size(mosaic_size)}) into: {rendered_root.resolve()}"
    )

    total_created = 0
    for sample_dir in rendered_sample_dirs:
        try:
            total_created += create_all_corruptions_for_sample(
                sample_dir,
                corrupt_out_root,
                dataset_rel,
                mosaic_size,
            )
        except Exception as exc:
            print(f"[WARN] skipping corruption for {sample_dir}: {exc}")

    print(
        f"Created {total_created} corrupted mosaic sample folders "
        f"(mosaic_size={format_mosaic_size(mosaic_size)}) under: {corrupt_out_root.resolve()}"
    )
    return rendered, total_created


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render one local MMReD plain HF Dataset directory as single-image mosaics. "
            "Supported layouts: seq_len=2 -> 1x2, seq_len=3-4 -> 2x2, seq_len=5-9 -> 3x3. "
            "Unused cells remain blank white. "
            "With --recursive, a seq_len_* directory or generated root will render all "
            "all_uniform and exact_* dataset folders underneath."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to one plain HF Dataset directory to render, or a parent directory with --recursive.",
    )
    parser.add_argument("--out", type=Path, default=Path("data/mmred_mosaic_images"))
    parser.add_argument(
        "--corrupt-out",
        type=Path,
        default=Path("data/mmred_mosaic_corrupted"),
        help="Output root for corrupted mosaic samples.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max rendered samples after qtype filter. In --recursive mode this applies per dataset folder.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively render all all_uniform and by_evidence_count/exact_* dataset folders under --dataset-root.",
    )
    parser.add_argument(
        "--mosaic-size",
        type=parse_mosaic_size,
        required=True,
        metavar="WIDTHxHEIGHT",
        help=(
            "Final composed mosaic canvas size, e.g. 1024x512 for 1x2, "
            "1024x1024 for 2x2/3x3, or another WIDTHxHEIGHT."
        ),
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    dataset_roots = resolve_dataset_roots(dataset_root, args.recursive)
    mosaic_size = args.mosaic_size
    print(f"Using mosaic_size={format_mosaic_size(mosaic_size)}")
    print(format_rendering_constants_summary())

    total_rendered = 0
    total_corrupted = 0
    for root in dataset_roots:
        try:
            rendered, corrupted = render_dataset(root, args.out, args.corrupt_out, args.limit, mosaic_size)
        except MosaicRenderError as exc:
            parser.error(str(exc))
        total_rendered += rendered
        total_corrupted += corrupted

    if len(dataset_roots) > 1:
        print(
            f"Finished rendering {len(dataset_roots)} dataset folders with "
            f"mosaic_size={format_mosaic_size(mosaic_size)}: "
            f"rendered_samples={total_rendered} corrupted_samples={total_corrupted}"
        )


if __name__ == "__main__":
    main()
