import ast
import torch
from PIL import Image
from pathlib import Path

def describe(x, name="x", max_list=8):
    """Print structure + (if tensor) shape/dtype/device."""
    print(f"\n=== {name} ===")
    if x is None:
        print("None")
        return

    # torch tensor
    if isinstance(x, torch.Tensor):
        print("Tensor")
        print(" shape:", tuple(x.shape))
        print(" dtype:", x.dtype)
        print(" device:", x.device)
        return

    # tuple/list
    if isinstance(x, (tuple, list)):
        print(type(x).__name__, "len=", len(x))
        for i, xi in enumerate(x[:max_list]):
            describe(xi, name=f"{name}[{i}]", max_list=max_list)
        if len(x) > max_list:
            print(f"... ({len(x)-max_list} more)")
        return

    # dict
    if isinstance(x, dict):
        print("dict keys:", list(x.keys())[:max_list])
        for k in list(x.keys())[:max_list]:
            describe(x[k], name=f"{name}['{k}']", max_list=max_list)
        if len(x) > max_list:
            print(f"... ({len(x)-max_list} more keys)")
        return

    # fallback
    print("type:", type(x))
    s = str(x)
    print(s[:500] + ("..." if len(s) > 500 else ""))


num_of_frames = 8


def load_mmred_sample(data_root: Path):
    """
    Returns:
      (sample_id, frames_list[PIL.Image], question_text, states_list[dict], answer_text)

    Expected qa.txt format (like your example):
      qid: ...
      qtype: ...
      ...
      question:
      { ... }        <-- num_of_frames lines of python dicts (states)
      ...
      How many steps did John spend in the Garden?   <-- the NL question line
      answer:
      2
    """
    sample_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
    if not sample_dirs:
        raise FileNotFoundError(f"No sample directories under: {data_root}")

    sample_dir = sample_dirs[0]
    sample_id = sample_dir.name

    # frames
    frame_paths = [sample_dir / f"{i:03d}.png" for i in range(num_of_frames)]
    frames = [Image.open(p).convert("RGB") for p in frame_paths]

    qa_path = sample_dir / "qa.txt"
    lines = qa_path.read_text(encoding="utf-8").splitlines()

    # find block markers
    q_idx = next((i for i, ln in enumerate(lines) if ln.strip() == "question:"), -1)
    a_idx = next((i for i, ln in enumerate(lines) if ln.strip() == "answer:"), -1)
    if q_idx == -1 or a_idx == -1 or a_idx <= q_idx:
        raise RuntimeError(f"Bad qa.txt format: {qa_path}")

    states = []
    question_text = None

    for ln in lines[q_idx + 1 : a_idx]:
        s = ln.strip()
        if not s:
            continue

        # state lines
        if s.startswith("{") and s.endswith("}"):
            states.append(ast.literal_eval(s))
            continue

        # THIS is the NL question (first non-dict line)
        question_text = s
        break

    if question_text is None:
        raise RuntimeError(f"Could not find NL question line in {qa_path}")

    # answer is first non-empty line after answer:
    answer_text = next((ln.strip() for ln in lines[a_idx + 1 :] if ln.strip()), None)
    if answer_text is None:
        raise RuntimeError(f"Could not find answer in {qa_path}")

    return sample_id, frames, question_text, states, answer_text

def print_top_k(logits, tokenizer, k=5):
    topk = torch.topk(logits, k=k)

    top_ids = topk.indices.tolist()

    probs = torch.softmax(logits, dim=-1)
    print(f"\nTop-{k} probs:")
    for rank, tok_id in enumerate(top_ids, start=1):
        print(f"{rank:>2}. id={tok_id:<6} p={probs[tok_id].item():.4f} token={tokenizer.decode([tok_id])!r}")