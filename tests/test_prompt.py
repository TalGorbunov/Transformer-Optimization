"""CPU tests for core.prompt: the paper's prompt is verbatim, the layouts put the same words in
the documented places, and the answer parser agrees with the upstream parser.

Run: python tests/test_prompt.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.prompt import LAYOUTS, SYSTEM_PROMPT, answer_target, build_messages, exact_match, parse_answer

UPSTREAM = _REPO / "data" / "mmred_hf" / "upstream_repo" / "scripts" / "openai_server_inference.py"
FRAMES = ["img0", "img1", "img2"]   # any objects; the processor is not involved here
Q = "How many steps did Daniel spend in the Kitchen?"


def _texts(msgs):
    return [c["text"] for m in msgs for c in (m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]) if c.get("type") == "text"]


def _user_content(msgs):
    (u,) = [m for m in msgs if m["role"] == "user"]
    return u["content"]


def test_system_prompt_is_verbatim_upstream():
    if not UPSTREAM.exists():
        print("  [skip] upstream repo not present"); return
    tree = ast.parse(UPSTREAM.read_text())
    up = [n.value.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
          and any(getattr(t, "id", None) == "SYSTEM_PROMPT" for t in n.targets)]
    assert up and up[0] == SYSTEM_PROMPT, "SYSTEM_PROMPT drifted from upstream"


def test_layouts():
    assert set(LAYOUTS) == {"paper", "question-first", "replica"}
    for layout in LAYOUTS:
        msgs = build_messages(FRAMES, Q, layout=layout)
        assert msgs[0]["role"] == "system" and _texts(msgs[:1]) == [SYSTEM_PROMPT]
        content = _user_content(msgs)
        imgs = [c for c in content if c["type"] == "image"]
        assert [c["image"] for c in imgs] == FRAMES, "all frames, in order, exactly once"
        assert all(t == Q for t in _texts(msgs)[1:]), "the only user text is the question, verbatim"
    paper = _user_content(build_messages(FRAMES, Q, layout="paper"))
    assert paper[-1]["type"] == "text" and all(c["type"] == "image" for c in paper[:-1])
    qf = _user_content(build_messages(FRAMES, Q, layout="question-first"))
    assert qf[0]["type"] == "text" and all(c["type"] == "image" for c in qf[1:])
    rep = _user_content(build_messages(FRAMES, Q, layout="replica"))
    assert [c["type"] for c in rep] == ["text"] + ["image", "text"] * len(FRAMES), "question, then (frame, question)×N"


def test_training_target():
    msgs = build_messages(FRAMES, Q, layout="question-first", answer="3")
    assert msgs[-1]["role"] == "assistant" and _texts(msgs[-1:]) == [answer_target("3")]
    assert parse_answer(answer_target("3")) == "3"
    assert parse_answer(answer_target("Kitchen")) == "Kitchen"
    assert parse_answer(answer_target("Nobody")) == "Nobody"


CASES = [
    ('{ "answer": "Kitchen" }', "Kitchen"),
    ('{"answer": 3}', "3"),
    ('{"answer": "3"}', "3"),
    ('```json\n{"answer": "Mary"}\n```', "Mary"),
    ('{"reasoning": "she was there twice", "answer": "Garden"}', "Garden"),
    ("Answer: Nobody", "Nobody"),
    ("I think it is the Garden.", "Garden"),
    ("<3>", "3"),
    ("The answer is 12.", "12"),
    ("no idea", None),
    ("", None),
]


def test_parse_answer_fixture():
    for raw, want in CASES:
        assert parse_answer(raw) == want, f"{raw!r}: got {parse_answer(raw)!r}, want {want!r}"


def test_parse_answer_agrees_with_upstream():
    """Cross-check against the upstream parser when its dependencies are installed."""
    sys.path.insert(0, str(UPSTREAM.parents[1]))
    try:
        from scripts.utils.parse_answers import parse_predicted_answer, strip_until_first_brace  # type: ignore
    except Exception as exc:
        print(f"  [skip] upstream parser not importable: {exc}"); return

    def up_parse(raw):
        v = parse_predicted_answer(strip_until_first_brace(raw))
        if isinstance(v, set):                       # upstream returns {"Nobody"} / {name}
            v = next(iter(v)) if len(v) == 1 else "None"
        return None if str(v) == "None" else str(v)

    for raw, _ in CASES:
        assert parse_answer(raw) == up_parse(raw), (raw, parse_answer(raw), up_parse(raw))


def test_exact_match():
    assert exact_match("3", "3") and exact_match("03", "3") and exact_match("kitchen", "Kitchen")
    assert not exact_match(None, "3") and not exact_match("4", "3") and not exact_match("Kitchen", "Garden")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(name); fn()
    print("ALL OK")
