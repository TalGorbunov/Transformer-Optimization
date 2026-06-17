#!/usr/bin/env python3
"""Generate SURGICAL corrupted frames (character removed from room) for the new MMReD tasks, so the
restoration/corruption ablation matches the ORIGINAL counting experiments (not the gray-blank fallback).

For each sample and each "evidence" frame of the task, re-render that frame with the queried character
removed from the room they occupy in that frame (via render_mmred.generate_corrupted_sample_from_rendered).
Output layout matches build_composite_corrupted_frames:
    <out_root>/<seq_len_X>/<split>/<sample_id>/corrupted_frame_{t}/{000..NNN}.png + qa.txt

Char selection MUST match token_group_corruption_new_tasks.task_spec so the ablation lines up:
  rooms_visited -> character present in the most frames (tie-break alphabetical); evidence = present frames
  co_occupancy  -> the pair sharing the most frames; evidence = shared frames; remove c1 from the shared room
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
# import render_mmred directly (avoid the HF `datasets` package name clash)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluations.helpers import utils as eval_utils
from evaluations.helpers.utils import iter_sample_dirs, load_mmred_sample
import render_mmred as R

# The park dataset's 6th room is "Park", not the renderer's default "Hallway". The renderer reads a
# module-global ROOMS list for the grid layout, room normalization, and which rooms to draw, so we MUST
# override it to the park layout or re-rendering silently drops Park (and anyone in it).
PARK_ROOMS = ["Kitchen", "Bathroom", "Garden", "Office", "Bedroom", "Park"]


def char_room_at(states, frame_idx, char):
    r2c = eval_utils.rooms_to_room2chars(states[frame_idx].get("rooms", {}))
    for room, occ in r2c.items():
        if char in occ:
            return room
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["rooms_visited", "co_occupancy"], required=True)
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--dataset_rel_dir", default="seq_len_8/all_uniform")
    ap.add_argument("--out_root", required=True, help="e.g. data/mmred_corrupted_park_rooms_visited")
    ap.add_argument("--limit", type=int, default=90)
    ap.add_argument("--rooms", default=",".join(PARK_ROOMS),
                    help="comma room list in grid order; MUST match the dataset's rooms (park uses ...,Park)")
    args = ap.parse_args()

    # override the renderer's room layout to match the dataset (park: 6th room is Park, not Hallway)
    R.ROOMS = [r.strip() for r in str(args.rooms).split(",") if r.strip()]
    print(f"renderer ROOMS overridden to: {R.ROOMS}")

    out_root = Path(args.out_root)
    rel = Path(args.dataset_rel_dir)
    n = made = 0
    for sd in iter_sample_dirs(Path(args.data_root)):
        if n >= int(args.limit):
            break
        try:
            sid, frames, q0, states, a0 = load_mmred_sample(sd)
        except Exception as e:
            continue
        chars = sorted(eval_utils.extract_characters_from_states(states))
        if not chars:
            continue
        if args.task == "rooms_visited":
            present = lambda c: [t for t in range(len(states)) if char_room_at(states, t, c)]
            char = max(chars, key=lambda c: (len(present(c)), c))
            evidence = present(char)
            removals = [(t, char, char_room_at(states, t, char)) for t in evidence]
        else:  # co_occupancy
            if len(chars) < 2:
                continue
            best = None
            for i in range(len(chars)):
                for j in range(i + 1, len(chars)):
                    sh = []
                    for t in range(len(states)):
                        r2c = eval_utils.rooms_to_room2chars(states[t].get("rooms", {}))
                        if any(chars[i] in o and chars[j] in o for o in r2c.values()):
                            sh.append(t)
                    if best is None or len(sh) > len(best[2]):
                        best = (chars[i], chars[j], sh)
            c1, c2, shared = best
            if not shared:
                continue
            removals = [(t, c1, char_room_at(states, t, c1)) for t in shared]  # remove c1 from shared room
        ok = True
        for t, c, room in removals:
            if room is None:
                ok = False
                break
            try:
                R.generate_corrupted_sample_from_rendered(
                    sample_dir=sd, corrupt_frame_idx=int(t), character=str(c), room=str(room),
                    out_root=out_root, dataset_rel_dir=rel,
                )
                made += 1
            except Exception as e:
                print(f"  {sid} frame {t} corrupt failed: {e}")
                ok = False
                break
        if ok:
            n += 1
            if n % 10 == 0:
                print(f"  {n} samples done ({made} corrupted frames)")
    print(f"task={args.task} samples={n} corrupted_frames={made} out_root={out_root}/{rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
