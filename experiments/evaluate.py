#!/usr/bin/env python3
"""Evaluate the frozen model or a LoRA adapter on one split of one dataset: the dataset's
official prompt and parser, native-resolution frames, exact match / MCQ accuracy per task.

Dataset and backbone are registry names (core.data.DATASETS, core.backbones.BACKBONES);
the defaults reproduce the pre-seam behaviour: official MMReD on Qwen2.5-VL-7B.

Faithful protocol on MMReD (every run without --port-check): core.prompt.SYSTEM_PROMPT as the
system turn; the user turn = the frames and the bare question (layout paper = frames then
question, question-first = their --prefix_question order = THE method's layout, replica = the
S1-S11 layout); greedy decode; core.prompt.parse_answer / exact_match; per-qtype EM with
bootstrap CIs; numeric qtypes also split by answer <= 16 / > 16 (the spec's report_extras).

--fence puts every frame in its own attention block with a per-block position reset
(core.fence via core.backbones); --gate oracle additionally hides every non-evidence block from
the tail (the Sample's evidence labels; NEVER used for a reported model-gate row). --arm names a
(layout, fence, gate) triple from core.prompt.ARMS instead of the three flags.

UNIT baseline (plan 2026-10-06; unfenced, the model's own answer): --no-frames = regime T (the
question alone); --unit <frame|clip3_d1|clip5_d1|clip5_d2|clip9_d2> = regime E (ONLY the evidence
units, no fillers, the official layout; video datasets read units_native/, MMReD keeps the
evidence frames of the config split; accuracy also by k); --res lo|hi = the backbone's own named
resolution mode (spec.res_modes). Run dirs carry _T / _E-<unit> and _res<mode>.

--two-pass is THE METHOD's read (plan 2026-10-11, cell 8): the fenced scan (--fast, --arm gated, the oracle
or a --gate-npz gate) decides which frames to keep and stops; the kept frames, in time order, are then read
unfenced in the official layout (frames, then the question) — regime E on the gate's frames. eval.csv keeps
the kept indices (`kept`) so the read can be re-run without the scan. --two-pass-positions original (variant 2,
2026-10-12) keeps every token of that read at the position it had in the full N-frame prompt (dropped frames leave
gaps; the question keeps its distance from the frames) instead of packing the kept frames; run dirs carry
_twopass_original. --readout letters is a forced choice
among the option letters at the first answer token (their logits; no generation, no parsing; MCQ datasets
only): the readout for backbones that answer in prose (InternVL3.5 on VNBench, 2026-10-11). On the batched
path the same forward gives both readouts (`raw` = the generated text, `pred` = the letter). Run dirs carry
_twopass / _letters.

Anchors (ONE-TIME port checks, experiments/_port_check.py; MMReD + Qwen only; never a reported row):
  --port-check arm-a  --frozen  seq_len_8 test, all 24 qtypes  -> 0.533 (640/1200), grid_seq8_test
                      --qtypes final_app steps_in_room where_spend --limit-per-qtype 34
                                                                 -> 0.765 / 0.559 / 0.353 (127776)
  --port-check p2     --adapter checkpoints/sft_fenced_hf_adapter --qids-file sbatch/lib/splits/qids_p2_seq8_test.txt
                      (and seq16 / seq32)                        -> 45/50, 33/50, 15/50
Refactor proof (docs/SCALEUP_2026-09-23.md §8): faithful seq_len_8 test on the defaults = 0.515.

Outputs (run dir): config.json, eval.csv (one row per sample), summary.csv, report.txt.
Usage:
  python experiments/evaluate.py --config seq_len_8 --split test --frozen --limit 20 --output outputs/_scratch/eval
  python experiments/evaluate.py --dataset mmred --n-frames 8 --split test --frozen --arm fenced_qfirst ...
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.backbones import BACKBONES, DEFAULT_BACKBONE, get_backbone  # noqa: E402
from core.backbones import base as bb  # noqa: E402
from core.data import DATASETS, DEFAULT_DATASET, get_dataset, stratified_order  # noqa: E402
from core.data.base import Sample  # noqa: E402
from core.data.videoqa import UNIT_OFFSETS  # noqa: E402
from core.fastpath import fenced_forward  # noqa: E402
from core.fence import FENCED_SDPA, FenceHooks, Span, build_block_mask, greedy_decode  # noqa: E402
from core.metrics import bootstrap_ci  # noqa: E402
from core.model import get_layers  # noqa: E402
from core.prompt import ARMS, LAYOUTS, build_messages  # noqa: E402
from experiments._diag_common import cond_tag, set_logn, set_sharpen  # noqa: E402
from experiments._port_check import (  # noqa: E402
    PORT_CHECKS, first_integer, legacy_norm, legacy_p2_blocks, legacy_p2_messages, legacy_regex_ladder, set_max_pixels,
)


OPTION_LETTERS = "ABCDE"                     # core.data.mcq.OPTIONS: at most five lettered choices on every MCQ dataset
LETTER_SPELLINGS = ("{letter}", " {letter}")  # how a letter can start the answer; only single-token spellings count
TWO_PASS_LAYOUT = "paper"                    # the official layout (frames, then the question): regime E's read


def _stop_newline(text: str) -> bool:
    return "\n" in text


@dataclass(frozen=True)
class LetterReadout:
    """Forced choice at the first answer token: the logit of each option letter (its best single-token
    spelling), argmax over the letters the question offers. No generation, no parsing."""
    ids_by_letter: Dict[str, List[int]]

    def predict(self, first_logits: torch.Tensor, letters: str) -> Tuple[str, float]:
        """(letter, margin): the best letter and its logit lead over the runner-up."""
        scores = torch.stack([first_logits[self.ids_by_letter[letter]].max() for letter in letters])
        order = scores.argsort(descending=True)
        margin = float(scores[order[0]] - scores[order[1]]) if len(letters) > 1 else float("inf")
        return letters[int(order[0])], margin


def letter_readout(tokenizer: Any, letters: str = OPTION_LETTERS) -> LetterReadout:
    """The option letters' first-token ids for this tokenizer (LETTER_SPELLINGS; fails if a letter has none)."""
    ids_by_letter: Dict[str, List[int]] = {}
    for letter in letters:
        ids = []
        for spelling in LETTER_SPELLINGS:
            token_ids = tokenizer(spelling.format(letter=letter), add_special_tokens=False).input_ids
            if len(token_ids) == 1:
                ids.append(int(token_ids[0]))
        if not ids:
            raise SystemExit(f"option letter {letter!r} is not a single token for this tokenizer; "
                             "--readout letters cannot run")
        ids_by_letter[letter] = ids
    return LetterReadout(ids_by_letter)


def option_letters(sample: Sample) -> str:
    """The letters a multiple-choice sample can answer with (meta['choices'] holds the lettered options)."""
    choices = sample.meta.get("choices")
    if not choices:
        raise SystemExit(f"--readout letters needs multiple-choice rows with meta['choices']; {sample.qid} has none")
    return OPTION_LETTERS[: len(choices)]


def kept_sample(sample: Sample, keep: Sequence[bool]) -> Sample:
    """The sample reduced to the frames the gate kept, in time order: what the two-pass read sees. meta keeps the
    kept indices and the scan's frame count (`scan_frames`), which the original-positions read needs."""
    kept = [t for t, flag in enumerate(keep) if flag]
    return replace(sample, frame_paths=tuple(sample.frame_paths[t] for t in kept), evidence=None,
                   meta={**sample.meta, "protocol": "two-pass", "kept": kept, "scan_frames": len(keep)})


def original_index(blocks: Sequence[Span], fin_start: int, kept: Sequence[int], n_frames: int,
                   seq_len: int) -> List[int]:
    """Token index map from the kept-frame prompt into the full n_frames prompt it was cut from: the prefix stays,
    kept block j becomes block kept[j], the tail follows all n_frames blocks. The frames of one video share a grid,
    so the blocks must be equal and adjacent; anything else is refused rather than mapped wrongly."""
    if len(blocks) != len(kept):
        raise ValueError(f"{len(blocks)} frame blocks in the prompt for {len(kept)} kept frames")
    first = blocks[0][0]
    block_len = blocks[0][1] - first
    expected = [(first + j * block_len, first + (j + 1) * block_len) for j in range(len(kept))]
    if list(blocks) != expected or fin_start != expected[-1][1]:
        raise ValueError(f"frame blocks are not equal and adjacent: {list(blocks)}, tail at {fin_start}")
    prefix = list(range(first))
    frames = [first + t * block_len + offset for t in kept for offset in range(block_len)]
    full_tail_start = first + n_frames * block_len
    tail = [full_tail_start + offset for offset in range(seq_len - fin_start)]
    return prefix + frames + tail


def full_prompt_inputs(enc: Dict[str, Any], block: Span, fin_start: int, n_frames: int) -> Dict[str, Any]:
    """The kept-frame prompt rebuilt as the full n_frames prompt (prefix + n_frames copies of its first block + tail),
    as text and grid only: what the model's own position rule reads (no pixels)."""
    ids = enc["input_ids"][0]
    full_ids = torch.cat([ids[: block[0]], ids[block[0]: block[1]].repeat(n_frames), ids[fin_start:]]).unsqueeze(0)
    inputs = {"input_ids": full_ids, "attention_mask": torch.ones_like(full_ids)}
    if "image_grid_thw" in enc:
        inputs["image_grid_thw"] = enc["image_grid_thw"][:1].repeat(n_frames, 1)
    return inputs


def positions_with_tail(pos: torch.Tensor, seq_len: int) -> torch.Tensor:
    """pos (…, L) extended to seq_len tokens: generated tokens continue after the last position on every axis, the
    model's own rule for text after the prompt."""
    have = int(pos.shape[-1])
    if seq_len <= have:
        return pos[..., :seq_len]
    extra = torch.arange(1, seq_len - have + 1, device=pos.device, dtype=pos.dtype)
    return torch.cat([pos, pos[..., -1:] + extra], dim=-1)


@dataclass(frozen=True)
class Reader:
    """The model side of a read: handles and decoding knobs that every sample shares."""
    model: Any
    base_model: Any
    tokenizer: Any
    processor: Any
    bspec: Any
    spec: Any
    sid: Any
    hooks: Optional[FenceHooks]
    readout: Optional["LetterReadout"]
    max_new: int
    eos_id: int
    stop_fn: Any
    device: Any


def first_logits_dense(model: Any, enc: Dict[str, Any]) -> torch.Tensor:
    """Next-token logits at the prompt's last position: one unfenced forward, no decoding."""
    with torch.inference_mode():
        out = model(**enc, use_cache=False)
    return out.logits[0, -1].float().cpu()


SetupFn = Callable[[Dict[str, Any]], Tuple[torch.Tensor, torch.Tensor]]   # inputs -> (mask_2d, position_ids): greedy_decode's


def first_logits_fenced(model: Any, hooks: FenceHooks, enc: Dict[str, Any], setup_fn: SetupFn,
                        passthrough_keys: Sequence[str]) -> torch.Tensor:
    """Next-token logits at the prompt's last position under the fence: the first step of core.fence.greedy_decode
    (setup_fn builds mask + positions from the inputs, the hooks inject the mask) without the decoding loop."""
    ids = enc["input_ids"]
    cur: Dict[str, Any] = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    for key in passthrough_keys:
        if key in enc:
            cur[key] = enc[key]
    mask, pos = setup_fn(cur)
    cur.pop("attention_mask", None)
    hooks.set_mask(mask, ids.device)
    try:
        with torch.inference_mode(), sdpa_kernel(FENCED_SDPA):
            logits = model(**cur, position_ids=pos.to(ids.device), use_cache=False).logits
    finally:
        hooks.clear_mask()
    return logits[0, -1].float().cpu()


def kept_indices(keep: Sequence[bool]) -> str:
    """The gate's kept frame indices as the eval.csv `kept` column (space-separated; no commas in csv text)."""
    return " ".join(str(t) for t, flag in enumerate(keep) if flag)


def answer_of(readout: Optional[LetterReadout], spec: Any, sample: Sample, raw: str,
              first_logits: Optional[torch.Tensor]) -> Tuple[Optional[str], str]:
    """(prediction, margin): the dataset's parser on the generated text, or the letter readout on the
    first-token logits with its margin as csv text ('' under generation)."""
    if readout is None:
        return spec.parse(raw), ""
    letter, margin = readout.predict(first_logits, option_letters(sample))
    return letter, f"{margin:.3f}"


ReadResult = Tuple[str, Optional[torch.Tensor], int]   # (generated text, first-token logits, prompt tokens)


def encode_read(reader: Reader, sample: Sample) -> Dict[str, Any]:
    """The sample's frames and question in the official layout, encoded for the model."""
    messages = build_messages(reader.spec.frames(sample), reader.spec.question_text(sample), layout=TWO_PASS_LAYOUT,
                              system_prompt=reader.spec.system_prompt)
    return bb.encode(reader.bspec, reader.processor, messages, reader.device)


def read_unfenced(reader: Reader, sample: Sample) -> ReadResult:
    """The plain read of a sample's frames. Generation when the reader has no letter readout, else one forward for the
    letter logits (the text stays empty)."""
    enc = encode_read(reader, sample)
    n_tokens = int(enc["input_ids"].shape[1])
    if reader.readout is not None:
        return "", first_logits_dense(reader.model, enc), n_tokens
    with torch.inference_mode():
        gen = reader.model.generate(**enc, max_new_tokens=reader.max_new, do_sample=False, pad_token_id=reader.eos_id)
    return reader.tokenizer.decode(gen[0, n_tokens:], skip_special_tokens=True), None, n_tokens


def original_positions(reader: Reader, enc: Dict[str, Any], kept: Sequence[int], n_frames: int) -> torch.Tensor:
    """Position ids of the kept-frame prompt as they were in the full n_frames prompt: the model's own rule on the
    rebuilt full prompt, gathered at the kept tokens. Under M-RoPE a block spans 2 + max(H, W) positions, not its
    token count, which is why the rule is called and not copied."""
    ids = enc["input_ids"][0].cpu()
    blocks, fin_start = bb.blocks(reader.bspec, ids, reader.sid, TWO_PASS_LAYOUT)
    index = original_index(blocks, fin_start, kept, n_frames, int(ids.shape[0]))
    full_pos = bb.base_positions(reader.bspec, reader.base_model, full_prompt_inputs(enc, blocks[0], fin_start, n_frames))
    return full_pos[..., torch.tensor(index, device=full_pos.device)]


def read_at_original_positions(reader: Reader, sample: Sample) -> ReadResult:
    """Variant 2 of the two-pass read: the kept frames under plain causal attention, every token at the position it had
    in the full scan prompt. Cache-free decoding through core.fence.greedy_decode with a causal mask in place of the
    fence (the hooks must be installed)."""
    if reader.hooks is None:
        raise ValueError("the original-positions read needs the fence hooks installed")
    enc = encode_read(reader, sample)
    n_tokens = int(enc["input_ids"].shape[1])
    pos = original_positions(reader, enc, sample.meta["kept"], sample.meta["scan_frames"])

    def setup(cur: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = int(cur["input_ids"].shape[1])
        return build_block_mask(seq, [], []), positions_with_tail(pos, seq)

    if reader.readout is not None:
        return "", first_logits_fenced(reader.model, reader.hooks, enc, setup, reader.bspec.passthrough_keys), n_tokens
    with torch.inference_mode():                 # greedy_decode builds no context of its own; without this, autograd
        raw = greedy_decode(reader.model, reader.hooks, enc, setup, tokenizer=reader.tokenizer,   # state piles up per row
                            max_new=reader.max_new, eos_id=reader.eos_id, stop_fn=reader.stop_fn,
                            passthrough_keys=reader.bspec.passthrough_keys)
    return raw, None, n_tokens


def two_pass_read(reader: Reader, sample: Sample, positions: str) -> ReadResult:
    """The read of the gate's kept frames: packed (the official k-frame prompt) or at their original positions. An
    empty gate leaves a text-only prompt with no frame positions to restore, so it is read packed either way."""
    if positions == "original" and sample.frame_paths:
        return read_at_original_positions(reader, sample)
    return read_unfenced(reader, sample)


def load_gate(path: Path) -> Tuple[Callable[[Dict[int, torch.Tensor], int], List[bool]], List[int]]:
    """A fitted linear gate (experiments/gate_fit.py --save-gate) as the batched path's keep callable: block t is kept
    iff ((x_t - mu) / sd) . w + b > 0 on its slot state at the gate layer (legacy numbering L = decoder module L - 1).
    The rule of the SPARSE S8/S9 rows (legacy/v1/scripts/sparse/train_sft_gated.py::infer_gate), pinned by
    tests/test_gate_model.py. Returns (keep, capture layers)."""
    g = np.load(path)
    mod = int(g["layer"]) - 1
    mu, sd, w, b = (torch.as_tensor(np.asarray(g[k]), dtype=torch.float32) for k in ("mu", "sd", "w", "b"))

    def keep(states: Dict[int, torch.Tensor], first: int) -> List[bool]:
        x = states[mod].float().cpu()
        return ((((x - mu) / sd) @ w + b) > 0).tolist()

    return keep, [mod]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DEFAULT_DATASET, choices=sorted(DATASETS))
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE, choices=sorted(BACKBONES))
    ap.add_argument("--data-root", type=Path, default=None, help="default: the dataset's data/ symlink")
    ap.add_argument("--config", default=None, help="MMReD config (seq_len_<N>); or use --n-frames")
    ap.add_argument("--protocol", default=None, help="frame protocol for --n-frames (default: the dataset's first)")
    ap.add_argument("--n-frames", type=int, default=None, help="N; with --protocol names the split via the dataset spec")
    ap.add_argument("--split", default="test")
    ap.add_argument("--qtypes", nargs="+", default=None, help="default: all of the dataset's task types")
    ap.add_argument("--qids-file", type=Path, default=None, help="one qid per line; pins the rows and their order")
    ap.add_argument("--limit", type=int, default=0, help="rows after stratified_order (class-interleaved)")
    ap.add_argument("--limit-per-qtype", type=int, default=0, help="rows per qtype in JSON order (legacy semantics)")
    ap.add_argument("--per-qtype", type=int, default=0, help="rows per qtype after a per-qtype stratified_order (the S1-S3 row budget)")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--adapter", type=Path, default=None)
    grp.add_argument("--frozen", action="store_true")
    ap.add_argument("--arm", choices=sorted(ARMS), default=None, help="(layout, fence, gate) triple; replaces the three flags")
    ap.add_argument("--layout", choices=LAYOUTS, default=None, help="default: adapter's train_config.json, else paper")
    ap.add_argument("--fence", action="store_true", help="default: adapter's train_config.json")
    ap.add_argument("--gate", choices=["none", "oracle"], default=None)
    ap.add_argument("--gate-npz", type=Path, default=None, help="a fitted gate (gate_fit.py --save-gate: layer in legacy numbering, mu, sd, w, b) replaces the arm's oracle on the batched path (--fast): run dir column n_evid = blocks kept, gate_fp / gate_fn vs the labels")
    ap.add_argument("--two-pass", action="store_true",
                    help="THE METHOD's read: the fenced gated scan (--fast --arm gated [--gate-npz]) decides the kept "
                         "frames, then they are read unfenced in the official layout; eval.csv column kept = their indices")
    ap.add_argument("--two-pass-positions", choices=("packed", "original"), default="packed",
                    help="packed = the kept frames as a k-frame prompt; original = every token at the position it had in "
                         "the full N-frame prompt (variant 2 of the read)")
    ap.add_argument("--readout", choices=("generate", "letters"), default="generate",
                    help="letters = forced choice among the option letters at the first answer token "
                         "(their logits; no parsing; MCQ datasets only)")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--max-seq-tokens", type=int, default=24000, help="refuse the dense fenced path above this")
    ap.add_argument("--fast", action="store_true", help="fenced arms through the batched path (core.fastpath): no mask, "
                    "per-frame encoding, the read sees prefix + kept blocks only")
    ap.add_argument("--chunk-tokens", type=int, default=16000, help="--fast: tokens per batch of blocks")
    ap.add_argument("--frame-set", default=None, help="video datasets: stored frames to read (native | 512); default = the spec's")
    ap.add_argument("--max-pixels", type=int, default=0, help="processor's own bound on pixels per frame (0 = its default)")
    ap.add_argument("--res", choices=("lo", "hi"), default=None, help="UNIT baseline: the backbone's named resolution mode (replaces --max-pixels)")
    ap.add_argument("--unit", choices=sorted(UNIT_OFFSETS), default=None,
                    help="UNIT baseline, regime E: only the evidence units, read at this unit (video: names the split; MMReD: 'frame' on --config)")
    ap.add_argument("--no-frames", action="store_true", help="UNIT baseline, regime T: the question alone, no image")
    ap.add_argument("--attn-sharpen", type=float, default=0.0, help="DIAG: tau on decoder modules >= --sharpen-from-layer (0 = off)")
    ap.add_argument("--sharpen-from-layer", type=int, default=12)
    ap.add_argument("--attn-logn-sref", type=int, default=0, help="DIAG: log-N logit scaling reference length in tokens (0 = off)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None, help="HF id / path overriding the backbone spec's model_id")
    ap.add_argument("--port-check", choices=sorted(PORT_CHECKS), default=None)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    # ---- dataset + backbone
    spec = get_dataset(args.dataset)
    if args.frame_set is not None:
        if not hasattr(spec, "frame_set"):
            raise SystemExit(f"{spec.name} has one frame set; --frame-set does not apply")
        spec.frame_set = args.frame_set
    bspec = get_backbone(args.backbone)
    if args.model:
        bspec = replace(bspec, model_id=str(args.model))
    data_root = args.data_root or Path(spec.default_root)
    if args.qtypes:
        unknown = sorted(set(args.qtypes) - set(spec.qtypes))
        if unknown:
            raise SystemExit(f"unknown qtypes for {spec.name}: {unknown}; known: {list(spec.qtypes)}")
    if args.no_frames and (args.unit is not None or args.fast or args.fence or args.gate not in (None, "none")):
        raise SystemExit("--no-frames is the unfenced text-only regime; no --unit/--fast/--fence/--gate")
    if args.unit is not None and (args.fast or args.fence or args.gate not in (None, "none") or args.port_check):
        raise SystemExit("--unit is the unfenced evidence-only regime; no --fast/--fence/--gate/--port-check")
    if args.res and args.max_pixels:
        raise SystemExit("give --res or --max-pixels, not both")
    if args.unit is not None and "evidence" in spec.protocols:
        if args.config is not None or args.n_frames is not None or args.protocol is not None:
            raise SystemExit("--unit on a video dataset names the split itself; drop --config/--protocol/--n-frames")
        split_name = spec.split_name("evidence", args.unit, args.split)
    elif args.config is not None:
        if spec.name != "mmred":
            raise SystemExit("--config is MMReD's config name; other datasets take --protocol/--n-frames")
        if args.n_frames is not None:
            raise SystemExit("give --config or --n-frames, not both")
        split_name = f"{args.config}_{args.split}"
    elif args.n_frames is not None:
        split_name = spec.split_name(args.protocol or spec.protocols[0], args.n_frames, args.split)
    else:
        raise SystemExit("one of --config (MMReD) or --n-frames [--protocol] is required")

    # ---- resolve the eval contract
    adapter_cfg: Dict[str, Any] = {}
    if args.adapter is not None:
        p = args.adapter / "train_config.json"
        if p.exists():
            adapter_cfg = json.loads(p.read_text())
    if args.arm is not None:
        if args.layout is not None or args.fence or args.gate is not None:
            raise SystemExit("--arm replaces --layout/--fence/--gate; do not combine")
        arm = ARMS[args.arm]
        layout, fence, gate = arm.layout, arm.fence, arm.gate
    else:
        layout = args.layout or adapter_cfg.get("layout") or "paper"
        fence = bool(args.fence or adapter_cfg.get("fence", False))
        gate = args.gate or "none"
    if args.attn_logn_sref == 0 and adapter_cfg.get("attn_logn_sref"):
        args.attn_logn_sref = int(adapter_cfg["attn_logn_sref"])      # the log-N training prior is part of the adapter contract
    pc = PORT_CHECKS[args.port_check] if args.port_check else None
    if pc is not None:
        if spec.name != "mmred" or bspec.family != "qwen2.5-vl":
            raise SystemExit("--port-check reproduces legacy MMReD/Qwen2.5-VL numbers only")
        if gate != "none" or args.layout is not None or args.arm is not None:
            raise SystemExit("--port-check fixes layout/gate; do not combine with --layout/--gate/--arm")
        if pc.name == "p2" and args.adapter is None:
            raise SystemExit("--port-check p2 needs --adapter checkpoints/sft_fenced_hf_adapter")
        if pc.name == "arm-a" and args.adapter is not None:
            raise SystemExit("--port-check arm-a is the frozen model")
        layout, fence = pc.layout, pc.fence
    if gate == "oracle" and not fence:
        raise SystemExit("--gate oracle requires --fence (the gate hides blocks)")
    gate_fn, gate_layers = None, ()
    if args.gate_npz is not None:
        if not (fence and args.fast):
            raise SystemExit("--gate-npz needs a fenced arm on the batched path (--fast)")
        gate, (gate_fn, gate_layers) = "model", load_gate(args.gate_npz)
    if adapter_cfg and args.layout is None and args.arm is None and pc is None and adapter_cfg.get("layout") != layout:
        print(f"[warn] adapter trained with layout={adapter_cfg.get('layout')} fence={adapter_cfg.get('fence')}", flush=True)
    if args.two_pass and not (args.fast and gate != "none"):
        raise SystemExit("--two-pass needs the batched fenced scan with a gate: --fast --arm gated [--gate-npz]")
    if args.two_pass_positions == "original" and not args.two_pass:
        raise SystemExit("--two-pass-positions original is a two-pass read: add --two-pass")
    if args.readout == "letters" and pc is not None:
        raise SystemExit("--readout letters is not a port check (the anchors are generated answers)")

    cond = cond_tag(args.attn_sharpen, args.sharpen_from_layer, args.attn_logn_sref)
    if args.fast and (not fence or pc is not None):
        raise SystemExit("--fast is the batched FENCED path; it needs a fenced arm and no port check")
    regime = "T" if args.no_frames else (f"E-{args.unit}" if args.unit else "")
    read = f"two-pass/{args.two_pass_positions}" if args.two_pass else "one-pass"
    twopass_tag = "" if not args.two_pass else ("_twopass_original" if args.two_pass_positions == "original" else "_twopass")
    run_dir = args.output / (f"{time.strftime('%Y%m%d_%H%M%S')}_{'pc-' + pc.name if pc else 'faithful'}"
                             f"{'' if cond == 'base' else '_' + cond}{'_fast' if args.fast else ''}"
                             f"{'_mp' + str(args.max_pixels) if args.max_pixels else ''}"
                             f"{'_' + regime if regime else ''}{'_res' + args.res if args.res else ''}"
                             f"{twopass_tag}{'_letters' if args.readout == 'letters' else ''}")
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {**vars(args), "frame_set": getattr(spec, "frame_set", None), "dataset": spec.name, "backbone": bspec.name, "model_id": bspec.model_id, "split_name": split_name,
           "data_root": str(data_root), "layout": layout, "fence": fence, "gate": gate, "regime": regime or "full",
           "read": read, "port_check": pc.name if pc else None, "cond": cond, "adapter_train_config": adapter_cfg}

    # ---- rows
    samples = spec.load(data_root, split_name, args.qtypes)
    if args.qids_file is not None:
        want = [ln.strip() for ln in args.qids_file.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
        by = {s.qid: s for s in samples}
        missing = [q for q in want if q not in by]
        strict = pc is not None or (args.unit is None and not split_name.startswith(("planted_", "isolated_")))
        if missing and strict:
            raise SystemExit(f"{len(missing)} qids not in {split_name}: {missing[:5]}")
        if missing:   # evidence protocol: rows without units; planted / isolated: rows the protocol cannot build at this N
            print(f"[rows] {len(missing)} qids of {args.qids_file} are not in {split_name} (skipped)", flush=True)
            cfg["qids_missing"] = len(missing)
        samples = [by[q] for q in want if q in by]
    elif args.limit_per_qtype:
        seen: Dict[str, int] = {}
        keep_rows = []
        for s in samples:                                # file order: the legacy eval_frozen semantics
            if seen.get(s.qtype, 0) < args.limit_per_qtype:
                keep_rows.append(s)
                seen[s.qtype] = seen.get(s.qtype, 0) + 1
        samples = keep_rows
    elif args.per_qtype:
        keep_rows = []
        for qt in (args.qtypes or spec.qtypes):
            keep_rows += stratified_order([s for s in samples if s.qtype == qt], args.seed, key=spec.stratify_key)[: args.per_qtype]
        samples = keep_rows
    else:
        samples = stratified_order(samples, args.seed, key=spec.stratify_key)
        if args.limit:
            samples = samples[: args.limit]
    n_no_evidence = 0
    if args.unit is not None and "evidence" not in spec.protocols:     # MMReD: the config split reduced to its evidence frames
        if args.unit != "frame":
            raise SystemExit(f"{spec.name} has one unit (a frame); --unit frame")
        reduced = [spec.evidence_only(s) for s in samples]
        n_no_evidence = sum(1 for r in reduced if r is None)
        samples = [r for r in reduced if r is not None]
    if args.no_frames:
        samples = [replace(s, frame_paths=(), evidence=None, meta={**s.meta, "protocol": "text"}) for s in samples]
    cfg["rows_without_evidence"] = n_no_evidence
    if args.readout == "letters" and samples:
        option_letters(samples[0])                       # lettered choices are required; fail before the model loads
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
    print(f"[rows] {len(samples)} from {spec.name}/{split_name} backbone={bspec.name} layout={layout} fence={fence} "
          f"gate={gate} regime={regime or 'full'} read={read} readout={args.readout} "
          f"port_check={pc.name if pc else None} load={getattr(spec, 'last_load', {}) or 'n/a'}", flush=True)

    # ---- model
    rt = bb.load_runtime(bspec)
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    base_model = model                                    # positions (get_rope_index) come from the unwrapped model
    layers = get_layers(model)                            # structural refs BEFORE the PEFT wrap
    base_scaling = set_sharpen(layers, args.attn_sharpen, args.sharpen_from_layer)   # DIAG knobs (eval-only)
    sid = bb.special_ids(bspec, tok)
    eos_id = int(tok.eos_token_id)
    readout = letter_readout(tok) if args.readout == "letters" else None
    if pc is not None:
        set_max_pixels(processor, pc.max_pixels)
    bb.set_max_pixels(bspec, processor, args.max_pixels)
    res_kwargs = bb.set_resolution(bspec, processor, args.res)
    if res_kwargs:
        cfg["res_kwargs"] = res_kwargs
        (run_dir / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
        print(f"[model] resolution mode {args.res}: {res_kwargs}", flush=True)
    if args.adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
        model.eval()
        print(f"[model] adapter {args.adapter}", flush=True)
    needs_hooks = (fence and not args.fast) or (args.two_pass and args.two_pass_positions == "original")
    hooks = FenceHooks(layers).install() if needs_hooks else None
    max_new = pc.max_new if pc else args.max_new_tokens
    stop_fn = _stop_newline if (pc and pc.name == "p2") else spec.stop_decoding
    reader = Reader(model=model, base_model=base_model, tokenizer=tok, processor=processor, bspec=bspec, spec=spec,
                    sid=sid, hooks=hooks, readout=readout, max_new=max_new, eos_id=eos_id, stop_fn=stop_fn,
                    device=rt.device)

    # ---- loop
    records: List[Dict[str, Any]] = []
    preds: List[Optional[str]] = []
    used: List[Any] = []
    n_gold_mismatch = n_gate_skip = n_layout_skip = n_parse_fail = n_too_long = 0
    t0 = time.time()
    for i, s in enumerate(samples):
        if not spec.verify(s):
            n_gold_mismatch += 1
        if args.fast:
            n = s.n_frames
            keep = None
            if gate == "oracle":
                if s.evidence is None:
                    n_gate_skip += 1
                    continue
                keep = [t in s.evidence for t in range(n)]
            elif gate == "model":
                keep = gate_fn
            try:
                r = fenced_forward(bspec, model, base_model, processor, list(s.frame_paths), spec.question_text(s),
                                   layout=layout, system_prompt=spec.system_prompt, keep=keep,
                                   capture_layers=gate_layers, decode=not args.two_pass, max_new=max_new,
                                   stop_fn=stop_fn, chunk_tokens=args.chunk_tokens, max_read_tokens=args.max_seq_tokens)
            except ValueError as exc:
                if "read too long" in str(exc):
                    n_too_long += 1
                    continue
                n_layout_skip += 1
                print(f"  [skip] {s.qid}: {exc}", flush=True)
                continue
            raw, first_logits, n_tokens = r.text, r.first_logits, r.tokens_read
            if args.two_pass:                            # the scan only decided; the kept frames are read unfenced
                try:
                    raw, first_logits, n_tokens = two_pass_read(reader, kept_sample(s, r.keep), args.two_pass_positions)
                except ValueError as exc:                # original positions: frame blocks the index map refuses
                    n_layout_skip += 1
                    print(f"  [skip] {s.qid}: {exc}", flush=True)
                    continue
            pred, margin = answer_of(readout, spec, s, raw, first_logits)
            ok = pred is not None
            correct = spec.match(pred, s)
            n_parse_fail += not ok
            used.append(s)
            preds.append(pred)
            records.append({"qid": s.qid, "qtype": s.qtype, "seq_len": s.n_frames, "atype": s.meta.get("atype", ""),
                            "gold": s.answer, "raw": raw[:160].replace("\n", "\\n"), "pred": "" if pred is None else pred,
                            "correct": int(correct), "parse_ok": int(ok), "letter_margin": margin,
                            "n_evid": "" if keep is None else sum(r.keep),
                            "kept": "" if keep is None else kept_indices(r.keep),
                            "gate_fp": "" if not (gate == "model" and s.evidence is not None) else sum(k and t not in s.evidence for t, k in enumerate(r.keep)),
                            "gate_fn": "" if not (gate == "model" and s.evidence is not None) else sum((not k) and t in s.evidence for t, k in enumerate(r.keep)),
                            "tokens": n_tokens, "tokens_per_frame": r.tokens_block,
                            "k": s.meta.get("k", ""), "unit_eff": s.meta.get("unit_eff", "")})
            if (i + 1) % 25 == 0:
                acc = sum(x["correct"] for x in records) / max(1, len(records))
                print(f"  {i + 1}/{len(samples)} acc={acc:.3f} read {n_tokens} tok, {r.tokens_block}/frame ({time.time() - t0:.0f}s)", flush=True)
            continue
        fr = spec.frames(s)
        n = len(fr)
        if pc is not None and pc.name == "p2":
            msgs = legacy_p2_messages(fr, s.question)
        else:
            qtext = (pc.question_prefix if pc else "") + spec.question_text(s)
            msgs = build_messages(fr, qtext, layout=layout, system_prompt=spec.system_prompt)
        enc = bb.encode(bspec, processor, msgs, rt.device)
        ids = enc["input_ids"]
        if fence and int(ids.shape[1]) > args.max_seq_tokens:
            n_too_long += 1
            continue
        keep = None
        if gate == "oracle":
            if s.evidence is None:
                n_gate_skip += 1
                continue
            keep = [t in s.evidence for t in range(n)]
        raw = ""
        first_logits = None
        set_logn(layers, int(ids.shape[1]), args.attn_logn_sref, base_scaling)   # per prompt length
        with torch.inference_mode():
            if not fence:
                try:
                    if readout is not None:
                        first_logits = first_logits_dense(model, enc)
                    else:
                        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=eos_id)
                        raw = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
                except torch.cuda.OutOfMemoryError:          # the longest unfenced prompts (evidence-only clips, ~55 K tokens)
                    torch.cuda.empty_cache()
                    n_too_long += 1
                    print(f"  [skip] {s.qid}: CUDA OOM at {int(ids.shape[1])} tokens ({n} frames)", flush=True)
                    continue
            else:
                if pc is not None and pc.name == "p2":
                    parsed = legacy_p2_blocks(ids[0].tolist(), tok, n, sid.vision_start)
                    if parsed is None:
                        n_layout_skip += 1
                        continue
                    blocks, fin = parsed
                else:
                    try:
                        blocks, fin = bb.blocks(bspec, ids[0].cpu(), sid, layout)
                    except ValueError:
                        n_layout_skip += 1
                        continue
                    if len(blocks) != n:
                        n_layout_skip += 1
                        continue
                setup = lambda cur: bb.fenced_setup(bspec, base_model, cur, blocks, fin, keep)  # noqa: E731
                if readout is not None:
                    first_logits = first_logits_fenced(model, hooks, enc, setup, bspec.passthrough_keys)
                else:
                    raw = greedy_decode(model, hooks, enc, setup, tokenizer=tok, max_new=max_new, eos_id=eos_id,
                                        stop_fn=stop_fn, passthrough_keys=bspec.passthrough_keys)
        # ---- parse + score
        margin = ""
        if pc is None:
            pred, margin = answer_of(readout, spec, s, raw, first_logits)
            ok = pred is not None
            correct = spec.match(pred, s)
        elif pc.parser == "legacy-ladder":
            pred, ok = legacy_regex_ladder(raw)
            correct = pred is not None and legacy_norm(pred) == legacy_norm(s.answer)
        else:
            pred = first_integer(raw)
            ok = pred is not None
            correct = pred is not None and int(pred) == int(s.answer)
        n_parse_fail += not ok
        used.append(s)
        preds.append(pred)
        records.append({"qid": s.qid, "qtype": s.qtype, "seq_len": s.n_frames, "atype": s.meta.get("atype", ""),
                        "gold": s.answer, "raw": raw[:160].replace("\n", "\\n"), "pred": "" if pred is None else pred,
                        "correct": int(correct), "parse_ok": int(ok), "letter_margin": margin,
                        "n_evid": "" if keep is None else sum(keep), "kept": "" if keep is None else kept_indices(keep),
                        "tokens": int(ids.shape[1]), "tokens_per_frame": "",
                        "k": s.meta.get("k", ""), "unit_eff": s.meta.get("unit_eff", "")})
        if (i + 1) % 25 == 0:
            acc = sum(r["correct"] for r in records) / max(1, len(records))
            print(f"  {i + 1}/{len(samples)} acc={acc:.3f} ({time.time() - t0:.0f}s)", flush=True)
    if hooks is not None:
        hooks.remove()

    # ---- summaries (from the recorded flags: the dataset's own rule, port checks included)
    lines = []
    toks = [r["tokens"] for r in records if r.get("tokens") != ""]
    head = f"{'PORT CHECK (deviation: ' + pc.name + ') ' if pc else 'FAITHFUL '}dataset={spec.name} split={split_name} " \
           f"regime={regime or 'full'} res={args.res or 'default'} " \
           f"frames={getattr(spec, 'frame_set', 'native')} max_pixels={args.max_pixels or 'default'} path={'fast' if args.fast else 'dense'} " \
           f"tokens_mean={sum(toks) / max(1, len(toks)):.0f} " \
           f"backbone={bspec.name} layout={layout} fence={fence} gate={gate} read={read} readout={args.readout} " \
           f"adapter={args.adapter or 'none'} n={len(records)} " \
           f"parse_fail={n_parse_fail} ({n_parse_fail / max(1, len(records)):.3f}) gold_mismatch={n_gold_mismatch} " \
           f"gate_skip={n_gate_skip} layout_skip={n_layout_skip} too_long={n_too_long} no_evidence={n_no_evidence}"
    lines.append(head)
    if pc:
        lines.append(f"  anchor: {pc.anchor}")
    summary = [("group", "n", "correct", "acc", "ci_lo", "ci_hi")]
    if records:
        flags = [bool(r["correct"]) for r in records]
        lo, hi = bootstrap_ci(flags)
        lines.append(f"OVERALL acc {sum(flags)}/{len(flags)} = {sum(flags) / len(flags):.3f}  [{lo:.3f}, {hi:.3f}]")
        summary.append(("all", len(flags), sum(flags), f"{sum(flags) / len(flags):.4f}", f"{lo:.4f}", f"{hi:.4f}"))
        for q in sorted({r["qtype"] for r in records}):
            fl = [bool(r["correct"]) for r in records if r["qtype"] == q]
            lo, hi = bootstrap_ci(fl)
            lines.append(f"  {q}: {sum(fl)}/{len(fl)} = {sum(fl) / len(fl):.3f}")
            summary.append((q, len(fl), sum(fl), f"{sum(fl) / len(fl):.4f}", f"{lo:.4f}", f"{hi:.4f}"))
        gated = [r for r in records if r.get("gate_fp", "") != ""]
        if gated:
            exact = sum(r["gate_fp"] == 0 and r["gate_fn"] == 0 for r in gated)
            lines.append(f"  gate: exact set {exact}/{len(gated)} = {exact / len(gated):.3f}, mean fp {sum(r['gate_fp'] for r in gated) / len(gated):.2f}, "
                         f"mean fn {sum(r['gate_fn'] for r in gated) / len(gated):.2f}, kept {sum(r['n_evid'] for r in gated) / len(gated):.2f} blocks")
        for at in sorted({r["atype"] for r in records if r["atype"] != ""}):
            fl = [bool(r["correct"]) for r in records if r["atype"] == at]
            lines.append(f"  atype {at}: {sum(fl)}/{len(fl)} = {sum(fl) / len(fl):.3f}")
            summary.append((f"atype_{at}", len(fl), sum(fl), f"{sum(fl) / len(fl):.4f}", "", ""))
        if pc is None:
            for k, (acc, nk) in spec.report_extras(used, preds).items():
                lines.append(f"  {k}: n={nk} acc={acc:.3f}" if nk else f"  {k}: n=0")
                summary.append((k, nk, "", f"{acc:.4f}" if nk else "", "", ""))
    report = "\n".join(lines)
    (run_dir / "report.txt").write_text(report + "\n")
    with open(run_dir / "eval.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()) if records else ["qid"])
        w.writeheader()
        w.writerows(records)
    with open(run_dir / "summary.csv", "w", newline="") as f:
        csv.writer(f).writerows(summary)
    print(report)
    print("wrote", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
