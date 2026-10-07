"""The dataset seam: one Sample shape and one DatasetSpec contract for every benchmark the
method runs on (official MMReD today; HERBench lite_v2 and MINERVA next —
docs/SCALEUP_2026-09-23.md §1.1).

A Sample is what an experiment iterates over: its frames (paths resolved at load time, in
temporal order), question and gold, the per-frame evidence labels at THIS frame count
(None = the dataset does not know), and a `group` key that every held-out split must respect
(the qid on MMReD; the video id on real video, where the questions of one video share frames).

A DatasetSpec is the benchmark itself: split naming, rows -> Samples, the official system
prompt, the official parser and scorer, what an integrity check means there, and the extra
report rows the benchmark's protocol asks for. Experiments never import a dataset module
directly; they go through core.data.get_dataset(name).

Contract pinned by tests/test_data.py on the MMReD adapter (core/data/mmred.py).
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

from PIL import Image


@dataclass(frozen=True)
class Sample:
    qid: str
    question: str                      # the benchmark's own wording, bare
    answer: str                        # canonical gold: the MMReD value, or an MCQ letter
    qtype: str                         # task family (MMReD qtype / HERBench task / MINERVA skill)
    frame_paths: Tuple[Path, ...]      # temporal order; one image item per path
    evidence: Optional[FrozenSet[int]]  # frame indices that are evidence at this N; None = unknown
    group: str                         # leakage key for held-out splits (qid / video_id)
    meta: Dict[str, Any] = field(default_factory=dict, compare=False)   # the raw row + extras

    @property
    def n_frames(self) -> int:
        return len(self.frame_paths)


class DatasetSpec(ABC):
    """The benchmark contract. Class attributes are data; methods are one-liners in adapters."""

    name: str = ""
    default_root: str = ""                 # data/<...> symlink the loader reads by default
    qtypes: Sequence[str] = ()
    numeric_qtypes: Sequence[str] = ()
    protocols: Sequence[str] = ()          # how a frame count N is realised (MMReD: "mmred")
    has_train: bool = False
    supports_counterfactual: bool = False  # probe_hahn re-renders edited frames: MMReD only
    system_prompt: str = ""

    # ---- splits + loading
    @abstractmethod
    def split_name(self, protocol: str, n_frames: int, split: str) -> str:
        """The on-disk split for (protocol, N, split): json/<name>.json + its frame dirs."""

    @abstractmethod
    def load(self, root: Path, split_name: str, qtypes: Optional[Sequence[str]] = None) -> List[Sample]:
        """Rows of one split as Samples, in the file's order (callers stratify)."""

    def frames(self, sample: Sample) -> List[Image.Image]:
        """The frames as RGB images, native resolution (no resize code here)."""
        return [Image.open(p).convert("RGB") for p in sample.frame_paths]

    # ---- evidence units (the UNIT baseline, plan 2026-10-06; every dataset: one frame = one unit)
    def evidence_only(self, sample: Sample) -> Optional[Sample]:
        """The sample reduced to its evidence frames, in order (the E regime at unit "frame");
        None when the labels are unknown or empty. meta gains protocol="evidence", unit, k, m_eff,
        unit_of_frame, as the video loader's evidence protocol records them."""
        if not sample.evidence:
            return None
        keep = sorted(sample.evidence)
        return replace(sample, frame_paths=tuple(sample.frame_paths[i] for i in keep),
                       evidence=frozenset(range(len(keep))),
                       meta={**sample.meta, "protocol": "evidence", "unit": "frame", "unit_eff": "frame",
                             "k": len(keep), "m_eff": 1.0, "unit_of_frame": list(range(len(keep))),
                             "n_frames": len(keep)})

    def units(self, root: Path, sample: Sample, unit: str = "frame") -> List[Dict[str, Any]]:
        """Per-unit records for the per-unit detection regime (D): one record per frame of the
        sample, kind "pos" when it is evidence, else "neg" (the same sequence's other frames).
        Video datasets override this with their stored clip grids and hard / easy negatives."""
        if unit != "frame":
            raise ValueError(f"{self.name}: the only unit is one frame; got {unit!r}")
        if sample.evidence is None:
            return []
        return [{"uid": f"f{t:02d}", "unit": t, "kind": "pos" if t in sample.evidence else "neg", "centre": float(t),
                 "interval": None, "m": 1, "k": len(sample.evidence), "frame_paths": (p,), "times": [float(t)]}
                for t, p in enumerate(sample.frame_paths)]

    def bare_question(self, sample: Sample) -> str:
        """The question as a per-unit judge sees it: the benchmark's words without its answer-format
        instruction (MCQ datasets add the lettered options). Default: the bare question."""
        return sample.question

    # ---- prompt + scoring (the benchmark's official protocol)
    @abstractmethod
    def question_text(self, sample: Sample) -> str:
        """The user text sent with the frames (MMReD: the bare question; MCQ: question + options
        + the benchmark's instruction line)."""

    @abstractmethod
    def answer_target(self, sample: Sample) -> str:
        """The assistant text a read is trained to emit."""

    @abstractmethod
    def parse(self, raw: str) -> Optional[str]:
        """Model output -> canonical prediction, or None (parse failure)."""

    @abstractmethod
    def match(self, pred: Optional[str], sample: Sample) -> bool:
        """The benchmark's correctness rule."""

    def stop_decoding(self, text: str) -> bool:
        """Early-stop rule for cache-free fenced decoding (default: eos only)."""
        return False

    # ---- integrity + bookkeeping
    def verify(self, sample: Sample) -> bool:
        """Integrity check per row (MMReD: recomputed gold == published). Default: nothing to check."""
        return True

    def stratify_key(self, sample: Sample) -> str:
        """Class key for stratified_order (default: the answer)."""
        return sample.answer

    def report_extras(self, samples: Sequence[Sample], preds: Sequence[Optional[str]]
                      ) -> Dict[str, Tuple[float, int]]:
        """Extra report rows the protocol asks for ({label: (acc, n)}); default none."""
        return {}


K_STRATA: Tuple[Tuple[str, int, int], ...] = (("k=1", 1, 1), ("k=2-4", 2, 4), ("k>=5", 5, 10 ** 9))


def evidence_strata(samples: Sequence[Sample], preds: Sequence[Optional[str]],
                    match: Callable[[Optional[str], Sample], bool]) -> Dict[str, Tuple[float, int]]:
    """Accuracy by the number of evidence units k (meta["k"], set by the evidence protocol /
    evidence_only): k = 1 rows are pure perception, k > 1 rows add the joint read without
    distractors. Empty when no sample carries k. Also the mean effective frames per unit."""
    rows = [(s, p) for s, p in zip(samples, preds) if "k" in s.meta]
    if not rows:
        return {}
    out: Dict[str, Tuple[float, int]] = {}
    for label, lo, hi in K_STRATA:
        sel = [(s, p) for s, p in rows if lo <= int(s.meta["k"]) <= hi]
        if sel:
            out[label] = (sum(match(p, s) for s, p in sel) / len(sel), len(sel))
    m = [float(s.meta.get("m_eff", 1.0)) for s, _ in rows]
    out["mean_m_eff"] = (sum(m) / len(m), len(m))
    fell = [s for s, _ in rows if s.meta.get("unit_eff") != s.meta.get("unit")]
    out["unit_fallback_rows"] = (len(fell) / len(rows), len(rows))
    return out


def stratified_order(samples: Sequence[Sample], seed: int,
                     key: Optional[Callable[[Sample], str]] = None) -> List[Sample]:
    """Class-interleaved shuffle (round-robin over classes), the same algorithm as
    core.mmred.stratified_order so both give the same order for the same rows and seed
    (tests/test_data.py::test_stratified_order_parity). Every loader goes through this:
    HF rows arrive answer-sorted, so a plain head slice is single-class (the K0 trap)."""
    key = key or (lambda s: s.answer)
    rng = random.Random(seed)
    groups: Dict[str, List[Sample]] = {}
    for s in samples:
        groups.setdefault(str(key(s)), []).append(s)
    keys = sorted(groups)
    rng.shuffle(keys)
    for k in keys:
        rng.shuffle(groups[k])
    out: List[Sample] = []
    while any(groups[k] for k in keys):
        for k in keys:
            if groups[k]:
                out.append(groups[k].pop())
    return out
