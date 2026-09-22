"""CPU tests for experiments/_port_check.py: the 392 px processor knob (smart_resize floors
392*392 to 364 px), the legacy parser ladder, and the legacy count prompt byte-identical to
legacy/v1/gnnformer/data.py. Run: python tests/test_port_check.py"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments._port_check import (
    MAX_PIXELS_392, PORT_CHECKS, first_integer, legacy_count_prompt, legacy_norm, legacy_regex_ladder,
)


def test_max_pixels_boundary():
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    assert smart_resize(512, 512, factor=28, min_pixels=56 * 56, max_pixels=392 * 392) == (364, 364)
    assert smart_resize(512, 512, factor=28, min_pixels=56 * 56, max_pixels=MAX_PIXELS_392) == (392, 392)
    assert (392 // 14) ** 2 // 4 == 196, "P2 geometry: 14 px patches, 2x2 merge -> 196 image tokens per frame"


def test_legacy_ladder():
    assert legacy_regex_ladder('{"answer": "Kitchen"}') == ("Kitchen", True)
    assert legacy_regex_ladder("Answer: 3") == ("3", True)
    assert legacy_regex_ladder("It was the Garden I think") == ("Garden", False)
    assert legacy_regex_ladder("maybe 12 steps") == ("12", False)
    assert legacy_regex_ladder("no idea") == (None, False)
    assert legacy_norm("03") == "3" and legacy_norm('"Kitchen"') == "kitchen"
    assert first_integer("Answer: 7\n") == "7" and first_integer("+4") == "4" and first_integer("none") is None


def test_count_prompt_verbatim_legacy():
    sys.path.insert(0, str(_REPO / "legacy" / "v1"))
    try:
        from gnnformer.data import build_count_prompt  # type: ignore
    except Exception as exc:
        print(f"  [skip] legacy/v1 data import unavailable: {exc}"); return
    q = "How many steps did Daniel spend in the Kitchen?"
    assert legacy_count_prompt(q, 8) == build_count_prompt(q, 8)


def test_presets():
    assert set(PORT_CHECKS) == {"arm-a", "p2"}
    assert PORT_CHECKS["arm-a"].fence is False and PORT_CHECKS["p2"].fence is True
    assert all(pc.max_pixels == MAX_PIXELS_392 for pc in PORT_CHECKS.values())


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(name); fn()
    print("ALL OK")
