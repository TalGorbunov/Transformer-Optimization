import ast
import math
import random
from typing import Any, Dict, List, Optional
import torch
import matplotlib.pyplot as plt
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


def load_mmred_sample(sample_dir: Path):
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
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")
    sample_id = sample_dir.name

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

    # frames: infer count from parsed states instead of using a global constant.
    frame_paths = [sample_dir / f"{i:03d}.png" for i in range(len(states))]
    missing = [p for p in frame_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frame(s) for sample {sample_id}: {missing[0]}")
    frames = [Image.open(p).convert("RGB") for p in frame_paths]

    return sample_id, frames, question_text, states, answer_text

def print_top_k(logits, tokenizer, k=5):
    topk = torch.topk(logits, k=k)

    top_ids = topk.indices.tolist()

    probs = torch.softmax(logits, dim=-1)
    print(f"\nTop-{k} probs:")
    for rank, tok_id in enumerate(top_ids, start=1):
        print(f"{rank:>2}. id={tok_id:<6} p={probs[tok_id].item():.4f} token={tokenizer.decode([tok_id])!r}")


def iter_sample_dirs(data_root: Path) -> List[Path]:
    """
    Finds sample directories under data_root (directories that contain qa.txt).
    """
    out: List[Path] = []
    for p in sorted(data_root.iterdir()):
        if p.is_dir() and (p / "qa.txt").exists():
            out.append(p)
    return out


def format_centered_indices(n: int, cell_width: int = 9) -> str:
    return " ".join(str(i).center(cell_width) for i in range(n))


def format_centered_values(vals: List[float], cell_width: int = 9, precision: int = 4) -> str:
    return " ".join(f"{v:.{precision}f}".center(cell_width) for v in vals)


def write_sample_metrics(sample_metrics: List[Dict[str, Any]], output_dir: Path) -> Path:
    def _fmt_float_list(vals: List[float]) -> str:
        return "[" + ", ".join(f"{v:.8f}" for v in vals) + "]"

    lines: List[str] = []
    for sm in sample_metrics:
        lines.append(f"sample_id={sm['sample_id']}")
        for lmtr in sm["layer_metrics"]["layers"]:
            lines.append(
                f"layer={lmtr['layer']} "
                f"r={_fmt_float_list(lmtr['r'])} "
                f"p={_fmt_float_list(lmtr['p'])} "
                f"H_norm={lmtr['entropy']:.8f}"
            )
        lines.append("")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sample_metrics.txt"
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


def plot_entropy_summary(
    sample_metrics: List[Dict[str, Any]],
    output_dir: Path,
    n_bootstrap: int = 1000,
    seed: int = 0,
    seq_len_label: Optional[str] = None,
) -> Optional[Path]:
    """
    Plot mean/median normalized entropy H(l)/N_evidence across layers
    with 95% bootstrap CIs.
    """
    entropy_by_layer: Dict[int, List[float]] = {}
    for sm in sample_metrics:
        for lmtr in sm["layer_metrics"]["layers"]:
            l = int(lmtr["layer"])
            entropy_by_layer.setdefault(l, []).append(float(lmtr["entropy"]))

    if not entropy_by_layer:
        return None

    rng = random.Random(seed)
    layers = sorted(entropy_by_layer.keys())

    means: List[float] = []
    medians: List[float] = []
    mean_lo: List[float] = []
    mean_hi: List[float] = []
    med_lo: List[float] = []
    med_hi: List[float] = []

    for l in layers:
        vals = entropy_by_layer[l]
        n = len(vals)
        sorted_vals = sorted(vals)

        mean = sum(vals) / n
        median = sorted_vals[n // 2] if n % 2 == 1 else 0.5 * (sorted_vals[n // 2 - 1] + sorted_vals[n // 2])

        boot_mean: List[float] = []
        boot_median: List[float] = []
        for _ in range(n_bootstrap):
            sample = [vals[rng.randrange(n)] for _ in range(n)]
            s_sorted = sorted(sample)
            b_mean = sum(sample) / n
            b_median = s_sorted[n // 2] if n % 2 == 1 else 0.5 * (s_sorted[n // 2 - 1] + s_sorted[n // 2])
            boot_mean.append(b_mean)
            boot_median.append(b_median)

        boot_mean.sort()
        boot_median.sort()
        lo_idx = int(0.025 * (n_bootstrap - 1))
        hi_idx = int(0.975 * (n_bootstrap - 1))

        means.append(mean)
        medians.append(median)
        mean_lo.append(boot_mean[lo_idx])
        mean_hi.append(boot_mean[hi_idx])
        med_lo.append(boot_median[lo_idx])
        med_hi.append(boot_median[hi_idx])

    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
    ax.plot(layers, means, color="#1f77b4", linewidth=2.2, label="Mean H(l)/N")
    ax.fill_between(layers, mean_lo, mean_hi, color="#1f77b4", alpha=0.2, label="Mean 95% CI")
    ax.plot(layers, medians, color="#d62728", linewidth=2.2, label="Median H(l)/N")
    ax.fill_between(layers, med_lo, med_hi, color="#d62728", alpha=0.2, label="Median 95% CI")

    title = "Normalized Entropy by Layer (H/N)"
    if seq_len_label:
        title = f"{title} ({seq_len_label})"
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Layer l", fontsize=11)
    ax.set_ylabel("H(l)/N", fontsize=11)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    ax.legend(frameon=True)
    # Avoid label collisions on wide/deep models by showing a spaced subset of layers.
    tick_step = max(1, math.ceil(len(layers) / 32))
    xticks = layers[::tick_step]
    if layers[-1] not in xticks:
        xticks.append(layers[-1])
    ax.set_xticks(xticks)
    ax.tick_params(axis="x", labelrotation=45, labelsize=9)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "entropy_summary.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def _bootstrap_center_and_ci(
    values: List[float],
    reducer,
    n_bootstrap: int,
    rng: random.Random,
) -> tuple[float, float, float]:
    n = len(values)
    center = reducer(values)
    if n <= 1:
        return center, center, center

    boot_values: List[float] = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boot_values.append(reducer(sample))

    boot_values.sort()
    lo_idx = int(0.025 * (n_bootstrap - 1))
    hi_idx = int(0.975 * (n_bootstrap - 1))
    return center, boot_values[lo_idx], boot_values[hi_idx]


def _plot_attention_importance_subplot(
    ax: Any,
    summary: Dict[str, List[float]],
    stat_name: str,
) -> None:
    layers = summary["layers"]
    if not layers:
        raise RuntimeError(f"No data available for {stat_name} subplot.")

    ax.plot(layers, summary["importance"], color="#1f77b4", linewidth=2.2, label=f"{stat_name} importance")
    ax.fill_between(
        layers,
        summary["importance_lo"],
        summary["importance_hi"],
        color="#1f77b4",
        alpha=0.18,
        label=f"{stat_name} importance 95% CI",
    )
    ax.plot(layers, summary["attention"], color="#d62728", linewidth=2.2, label=f"{stat_name} attention ratio")
    ax.fill_between(
        layers,
        summary["attention_lo"],
        summary["attention_hi"],
        color="#d62728",
        alpha=0.18,
        label=f"{stat_name} attention 95% CI",
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Value")
    ax.set_title(f"Per-layer {stat_name.lower()}: importance vs attention-share")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    ax.legend(frameon=True)
    tick_step = max(1, math.ceil(len(layers) / 32))
    xticks = layers[::tick_step]
    if layers[-1] not in xticks:
        xticks.append(layers[-1])
    ax.set_xticks(xticks)
    ax.tick_params(axis="x", labelrotation=45, labelsize=9)


def plot_attention_importance_summary(
    importance_by_layer: Dict[int, List[float]],
    attention_by_layer: Dict[int, List[float]],
    output_path: Path,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> Optional[Path]:
    candidate_layers = sorted(set(importance_by_layer) & set(attention_by_layer))
    if not candidate_layers:
        return None

    def _mean(vals: List[float]) -> float:
        return sum(vals) / len(vals)

    def _median(vals: List[float]) -> float:
        ordered = sorted(vals)
        n = len(ordered)
        mid = n // 2
        return ordered[mid] if n % 2 == 1 else 0.5 * (ordered[mid - 1] + ordered[mid])

    def _build_summary(reducer, rng_seed: int) -> Dict[str, List[float]]:
        rng = random.Random(rng_seed)
        summary: Dict[str, List[float]] = {
            "layers": [],
            "importance": [],
            "importance_lo": [],
            "importance_hi": [],
            "attention": [],
            "attention_lo": [],
            "attention_hi": [],
        }
        for layer_idx in candidate_layers:
            importance_values = importance_by_layer.get(layer_idx, [])
            attention_values = attention_by_layer.get(layer_idx, [])
            if not importance_values or not attention_values:
                continue

            imp_center, imp_lo, imp_hi = _bootstrap_center_and_ci(
                importance_values,
                reducer,
                n_bootstrap,
                rng,
            )
            att_center, att_lo, att_hi = _bootstrap_center_and_ci(
                attention_values,
                reducer,
                n_bootstrap,
                rng,
            )
            summary["layers"].append(layer_idx)
            summary["importance"].append(imp_center)
            summary["importance_lo"].append(imp_lo)
            summary["importance_hi"].append(imp_hi)
            summary["attention"].append(att_center)
            summary["attention_lo"].append(att_lo)
            summary["attention_hi"].append(att_hi)
        return summary

    mean_summary = _build_summary(_mean, seed)
    median_summary = _build_summary(_median, seed + 1)
    if not mean_summary["layers"] or not median_summary["layers"]:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=140, sharey=False)
    _plot_attention_importance_subplot(axes[0], mean_summary, "Mean")
    _plot_attention_importance_subplot(axes[1], median_summary, "Median")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
