"""V4 schedule/provenance contracts; execute on a Slurm CPU allocation."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import random
import sys
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.native_aggregation_vlm_v4 import (
    initial_parameter_hashes, slot_order, validate_schedule,
)


def fixture(profile=False):
    root = "/mnt/data/gabriele/gnn_transformer/v4_diversity"
    slots = []
    if profile:
        specs = [(8, 0), (8, 1), (16, 0), (16, 1)]
    else:
        specs = [(n, k) for n in (8, 16) for k in range(9) for _ in range(10)]
    for slot, (n, k) in enumerate(specs):
        character, room = f"C{slot % 9}", f"R{slot % 6}"
        slots.append(dict(slot=slot, n_frames=n, gold=k,
                          question=f"How many frames show {character} in the {room}?",
                          target_character=character, target_room=room,
                          original_sid=f"original_{slot}", saturated=n == 8 and k == 8))
    rows, blocks = {}, []
    for block in range(2 if profile else 9):
        selected = []
        for slot in slots:
            sid = slot["original_sid"] if block == 0 or slot["saturated"] else f"fresh_{block}_{slot['slot']}"
            if sid not in rows:
                rows[sid] = dict(sid=sid, split="train", content_sha256=f"{len(rows):064x}",
                                 **{key: slot[key] for key in
                                    ("n_frames", "gold", "question", "target_character", "target_room")})
            selected.append(sid)
        blocks.append(selected)
    manifest = dict(schema_version=1, dataset_root=root,
                    splits={f"train_N{n}": dict(samples=[row for row in rows.values() if row["n_frames"] == n])
                            for n in (8, 16)})
    schedule = dict(schema_version=1, dataset_root=root, slot_metadata=slots,
                    conditions=dict(repeat=[list(blocks[0]) for _ in blocks], refresh=blocks))
    return schedule, manifest


class ScheduleTests(unittest.TestCase):
    def test_main_budget_and_saturation(self):
        schedule, manifest = fixture()
        result = validate_schedule(schedule, manifest)
        self.assertEqual(result["presentations"], 1620)
        self.assertEqual(result["image_frame_presentations"], 19440)
        self.assertEqual(result["distinct_by_condition"], {"repeat": 180, "refresh": 1540})
        self.assertEqual(len(result["saturated_slots"]), 10)

    def test_profile_uses_both_blocks_and_lengths(self):
        schedule, manifest = fixture(True)
        result = validate_schedule(schedule, manifest, profile=True)
        self.assertEqual(result["presentations"], 8)
        self.assertEqual(result["image_frame_presentations"], 96)
        self.assertEqual(result["distinct_by_condition"], {"repeat": 4, "refresh": 8})

    def test_refresh_cannot_reuse_a_nonsaturated_view(self):
        schedule, manifest = fixture()
        schedule["conditions"]["refresh"][1][0] = schedule["conditions"]["refresh"][0][0]
        with self.assertRaisesRegex(ValueError, "reused prior content"):
            validate_schedule(schedule, manifest)

    def test_repeat_cannot_change_after_shared_first_block(self):
        schedule, manifest = fixture()
        schedule["conditions"]["repeat"][1][0] = schedule["conditions"]["refresh"][1][0]
        with self.assertRaisesRegex(ValueError, "repeat cannot change"):
            validate_schedule(schedule, manifest)

    def test_question_changes_are_detected_independently_of_gold(self):
        schedule, manifest = fixture()
        manifest["splits"]["train_N16"]["samples"][-1]["question"] += " Extra"
        with self.assertRaisesRegex(ValueError, "fixed slot field question"):
            validate_schedule(schedule, manifest)

    def test_duplicate_semantic_content_with_new_sid_is_rejected(self):
        schedule, manifest = fixture()
        rows = manifest["splits"]["train_N8"]["samples"]
        rows[-1]["content_sha256"] = rows[0]["content_sha256"]
        with self.assertRaisesRegex(ValueError, "distinct canonical content"):
            validate_schedule(schedule, manifest)

    def test_partial_or_reordered_slot_metadata_is_rejected(self):
        schedule, manifest = fixture()
        schedule["slot_metadata"][0], schedule["slot_metadata"][1] = schedule["slot_metadata"][1], schedule["slot_metadata"][0]
        with self.assertRaisesRegex(ValueError, "complete ordered index"):
            validate_schedule(schedule, manifest)

    def test_shuffle_contract_uses_slot_indices_and_zero_based_blocks(self):
        expected = list(range(180))
        random.Random(2 + 1).shuffle(expected)
        self.assertEqual(slot_order(180, 2, 1), expected)
        self.assertEqual(sorted(slot_order(180, 3, 8)), list(range(180)))
        self.assertNotEqual(slot_order(180, 2, 0), slot_order(180, 2, 1))

    def test_initial_hash_detects_changed_lora_or_branch_tensors(self):
        import torch
        torch.manual_seed(2)
        branch = torch.nn.Linear(3, 2)
        class Lora:
            params = {(14, "q"): (torch.nn.Parameter(torch.ones(2, 3)),
                                  torch.nn.Parameter(torch.zeros(3, 2)))}
        lora = Lora()
        initial = initial_parameter_hashes(branch, lora)
        same = initial_parameter_hashes(copy.deepcopy(branch), copy.deepcopy(lora))
        self.assertEqual(initial["combined_sha256"], same["combined_sha256"])
        with torch.no_grad():
            lora.params[14, "q"][1][0, 0] = 1
        changed = initial_parameter_hashes(branch, lora)
        self.assertNotEqual(initial["combined_sha256"], changed["combined_sha256"])
        self.assertEqual(initial["branch_sha256"], changed["branch_sha256"])
        self.assertNotEqual(initial["lora_sha256"], changed["lora_sha256"])


if __name__ == "__main__":
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Run V4 CPU tests inside a Slurm allocation")
    unittest.main()
