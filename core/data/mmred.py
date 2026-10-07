"""The official MMReD benchmark as a DatasetSpec: a thin adapter over core.mmred and
core.prompt (nothing moves; those modules keep their tests). Rows become Samples whose
frame paths are the official renders (`images/<config>_<split>/<qid>/frame_%04d.png`),
whose evidence is core.mmred.evidence_frames, and whose group is the qid.

Pinned by tests/test_data.py: same qids and order as load_split, evidence == evidence_frames,
verify() == gold parity on the real seq_len_8 test split, parse/match == core.prompt.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..constants import SEQ_LENS
from ..metrics import split_by_answer
from ..mmred import NUMERIC_QTYPES, QTYPES, evidence_frames, load_split, recompute_answer, states
from ..prompt import SYSTEM_PROMPT, answer_target, exact_match, parse_answer
from .base import DatasetSpec, Sample, evidence_strata


class MMReD(DatasetSpec):
    name = "mmred"
    default_root = "data/mmred_hf"
    qtypes = tuple(QTYPES)
    numeric_qtypes = tuple(NUMERIC_QTYPES)
    protocols = ("mmred",)                 # N is a generator config, not a sampling protocol
    has_train = True
    supports_counterfactual = True
    system_prompt = SYSTEM_PROMPT

    def split_name(self, protocol: str, n_frames: int, split: str) -> str:
        if protocol != "mmred":
            raise ValueError(f"mmred has one protocol ('mmred'), got {protocol!r}")
        if int(n_frames) not in SEQ_LENS:
            raise ValueError(f"seq_len {n_frames} is not one of {SEQ_LENS}")
        return f"seq_len_{int(n_frames)}_{split}"

    def load(self, root: Path, split_name: str, qtypes: Optional[Sequence[str]] = None) -> List[Sample]:
        config, split = split_name.rsplit("_", 1)          # "seq_len_8_test" -> ("seq_len_8", "test")
        return [self.sample(r, Path(root)) for r in load_split(config, split, Path(root), qtypes)]

    @staticmethod
    def sample(row: Dict, root: Path) -> Sample:
        """One published row -> Sample. Frame paths mirror core.mmred.frames (native 512 px)."""
        n = len(row["sequence"])
        base = Path(root) / "images" / row["split_dir"] / str(row["qid"])
        ev = evidence_frames(row["qtype"], row["question"], states(row))
        return Sample(qid=str(row["qid"]), question=row["question"], answer=str(row["answer"]),
                      qtype=row["qtype"], frame_paths=tuple(base / f"frame_{i + 1:04d}.png" for i in range(n)),
                      evidence=None if ev is None else frozenset(ev), group=str(row["qid"]), meta=row)

    def question_text(self, sample: Sample) -> str:
        return sample.question                             # upstream process_row: the bare question

    def answer_target(self, sample: Sample) -> str:
        return answer_target(sample.answer)

    def parse(self, raw: str) -> Optional[str]:
        return parse_answer(raw)

    def match(self, pred: Optional[str], sample: Sample) -> bool:
        return exact_match(pred, sample.answer)

    def stop_decoding(self, text: str) -> bool:
        """Stop at the first '}' after "answer" — the JSON object is complete."""
        return "answer" in text and "}" in text.split("answer", 1)[1]

    def verify(self, sample: Sample) -> bool:
        return recompute_answer(sample.qtype, sample.question, states(sample.meta)) == sample.answer

    def report_extras(self, samples: Sequence[Sample], preds: Sequence[Optional[str]]
                      ) -> Dict[str, Tuple[float, int]]:
        """The paper protocol's coverage split for numeric qtypes: answer <= 16 vs > 16."""
        rows = [{"qtype": s.qtype, "answer": s.answer} for s in samples]
        out = {f"numeric{k}": v for k, v in split_by_answer(rows, preds, 16).items()}
        out.update(evidence_strata(samples, preds, self.match))       # evidence-only rows: accuracy by k
        return out
