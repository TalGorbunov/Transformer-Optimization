"""BABILong external evaluation of TRAINING-FREE select-and-repack.

Context = PG-19 filler with bAbI facts spliced in as whole sentences. We segment the
context into sentences (= "frames"), and every arm produces a kept sentence set that is
answered at SHORT geometry (kept sentences concatenated in original order, positions
renumbered contiguously), except where noted:

  full       the whole context (the published baseline; official BABILong prompt)
  oracle     keep exactly the bAbI fact sentences (ceiling; facts identified via the 0k
             split's sentence vocabulary — 0k contexts are facts only)
  attn       OURS: frozen model run to --sel-layer under native attention, the tail
             (question) rows' attention mass summed per sentence, top-k kept, repacked
  attn_norp  the SAME kept sentences at their ORIGINAL positions (select WITHOUT
             relabeling = the SnapKV-style control the paper's diagnosis predicts fails)
  lex        top-k by question-word overlap (keyword retrieval)
  emb        top-k by all-MiniLM-L6-v2 cosine similarity (embedding RAG)
  rand       random k sentences (floor)
  oracle_gap@P   oracle sentences, contiguous, but the TAIL's position ids start P tokens
                 later (edge length answer->evidence = P, degree fixed): the position
                 dose-response, --gap-tokens
  oracle_pgap@P  the gap sits between the PREFIX and the evidence instead (evidence->
                 answer edges short, prefix->answer edges long)
  attn@kK    the attn arm at another sentence budget K (--k-sweep; one scoring pass)

Selection variants (the selector is itself a read node, so it must be bounded-degree):
  --sel-chunk C          score the context in <= C-token windows, concatenate masses
  --sel-tournament KL    ...and re-score the per-window top-KL finalists in ONE window
                         (recursing while the union exceeds C): every comparison happens
                         inside one bounded softmax
  --sel-rounds 2         hop-2: copies of the round-1 kept sentences are appended to the
                         tail and the context is re-scored (rounds = problem radius)

--probe adds the over-squashing metrics of osq.py (answer-row fan-in / evidence mass /
margin / Jacobian influence at --probe-layer) to every probed arm's record.

Scoring = the official compare_answers (vendored, babilong_official/). --scan reports
per-layer fact recall@k of the tail-row attention (the retrieval-signal scan that
predicts transferability) instead of answering.

--mmred <data/mmred_filtered/seq_len_N/test> runs the SAME fully frozen protocol on the MMRED
filtered-counting text task: one frame line = one "sentence", facts = the frames where the
asked character is in the asked room (from the state dicts), counting prompt MMRED_PROMPT,
scoring = first integer in the output == gold. Nothing is trained anywhere.

  python scripts/condmask/eval_babilong.py --model Qwen/Qwen2.5-3B-Instruct \
      --task qa1 --length 4k --limit 100 --k 32 --sel-layer 20 --arms full,attn,attn_norp
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from modeling import load_text_model, model_parts  # noqa: E402
from osq import PROBED, osq_probe, rope as _rope  # noqa: E402
from babilong_official.prompts import DEFAULT_PROMPTS, get_formatted_input  # noqa: E402
from babilong_official.metrics import TASK_LABELS, compare_answers  # noqa: E402
from textdata import load_text_sample, render_frame_line  # noqa: E402

ARMS = ("full", "oracle", "attn", "attn_norp", "lex", "emb", "rand", "oracle_gap", "oracle_pgap")
_CTX = "@@CTX@@"
_STOP = {"where", "is", "was", "the", "what", "who", "did", "to", "of", "before", "a",
         "in", "give", "gave", "does", "do", "how", "many", "objects", "carrying",
         "frames", "show"}
# MMRED counting prompt in the BABILong prompt format (instruction / examples / post_prompt);
# fixed once, never tuned per model or arm.
MMRED_PROMPT = {
    "instruction": "I will give you a list of frames. Each frame says which person is in which "
                   "room. You need to answer the question by counting the frames that match it.",
    "examples": "<example>\nFrame 1: Mary is in the Kitchen. Frame 2: John is in the Garden. "
                "Frame 3: Mary is in the Kitchen. Frame 4: Mary is in the Garden.\n"
                "Question: How many frames show Mary in the Kitchen?\nAnswer: 2\n</example>\n"
                "<example>\nFrame 1: Peter is in the Bathroom. Frame 2: Peter is in the Bedroom.\n"
                "Question: How many frames show Sandra in the Bathroom?\nAnswer: 0\n</example>",
    "post_prompt": "Your answer should contain only the number of matching frames, written in "
                   "digits. Do not write anything else after that.",
}


def split_sents(text: str) -> List[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ").strip()) if s]


def load_mmred(root: Path, seed: int) -> pd.DataFrame:
    """data/mmred_filtered/seq_len_<N>/<split>: one frame line per 'sentence' (input = lines
    joined by '\\n'); facts = frames with the asked character in the asked room, parsed from
    the state dicts (their count IS the gold; asserted). Dirs are grouped by gold, so the
    rows are shuffled (seeded) before the caller's --offset/--limit slice."""
    rows = []
    for d in sorted(root.iterdir()):
        s = load_text_sample(d)
        if s is None:
            continue
        _, states, question, gold = s
        who, room = re.fullmatch(r"How many frames show (\w+) in the (\w+)\?", question).groups()
        facts = [j for j, st in enumerate(states)
                 if who in ast.literal_eval(st)["rooms"].get(room, [])]
        assert len(facts) == gold, d
        rows.append({"input": "\n".join(map(render_frame_line, states)), "question": question,
                     "target": str(gold), "facts": facts})
    random.Random(seed).shuffle(rows)
    return pd.DataFrame(rows)


def load_split(root: Path, task: str, length: str) -> pd.DataFrame:
    """root/<length>/<task>-*.parquet (babilong-1k-samples, 1000/split, <=32k) or
    root/<length>/<task>.json (the official 100-sample set; we use it for 64k/128k).
    `root` may hold several roots joined by '+': first hit wins."""
    for r in str(root).split("+"):
        j = Path(r) / length / f"{task}.json"
        if j.exists():
            return pd.DataFrame(json.loads(j.read_text()))
        pq = list((Path(r) / length).glob(f"{task}-*.parquet"))
        if pq:
            return pd.read_parquet(pq[0])
    raise FileNotFoundError(f"{task}/{length} under {root}")


def fact_vocab(root: Path, task: str) -> set:
    """0k contexts are bAbI facts only -> the set of all their sentences is the task's
    fact vocabulary (bAbI is template-generated; ~hundreds of distinct sentences)."""
    df = load_split(root, task, "0k")
    return {s for t in df["input"] for s in split_sents(t)}


def build(tok, task: str, question: str, sents: List[str], P: Optional[dict] = None) -> Dict[str, Any]:
    """Official prompt through the model's chat template, tokenized SEGMENT BY SEGMENT
    so sentence spans are exact by construction (textdata.build_segmented_prompt).
    `P` overrides the task's prompt dict (instruction/examples/post_prompt)."""
    P = P or DEFAULT_PROMPTS[task]
    user = get_formatted_input(_CTX, question, P["examples"], P["instruction"],
                               P["post_prompt"])
    txt = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                  add_generation_prompt=True)
    pre_txt, tail_txt = txt.split(_CTX)
    ids = list(tok(pre_txt, add_special_tokens=False).input_ids)
    prefix_end = len(ids)
    blocks: List[Tuple[int, int]] = []
    for i, s in enumerate(sents):
        seg = tok(("" if i == 0 else " ") + s, add_special_tokens=False).input_ids
        blocks.append((len(ids), len(ids) + len(seg)))
        ids.extend(seg)
    fin = len(ids)
    ids.extend(tok(tail_txt, add_special_tokens=False).input_ids)
    return {"ids": ids, "prefix_end": prefix_end, "blocks": blocks, "fin": fin,
            "seq": len(ids)}


def _row_mass(layer, h, pe, rec, rows: str) -> torch.Tensor:
    """Mean-over-(heads,rows) attention weight of the query rows at `layer`'s input h ->
    per-token weights [S]. rows: tail (question + generation prompt), last (answer row),
    copies (the round-2 hop queries, rec["rows2"])."""
    hn = layer.input_layernorm(h)
    at = layer.self_attn
    S = hn.shape[1]
    hd = pe[0].shape[-1]
    nh = at.q_proj.weight.shape[0] // hd
    q = _rope(at.q_proj(hn).view(1, S, nh, hd).transpose(1, 2), *pe)
    k = _rope(at.k_proj(hn).view(1, S, -1, hd).transpose(1, 2), *pe)
    k = k.repeat_interleave(nh // k.shape[1], dim=1)
    r0, r1 = {"tail": (rec["fin"], S), "last": (S - 1, S), "copies": rec.get("rows2")}[rows]
    sc = q[:, :, r0:r1] @ k.transpose(-1, -2) / hd ** 0.5         # [1,H,T,S]
    T = r1 - r0
    causal = torch.arange(S, device=h.device)[None, :] > (r0 + torch.arange(T, device=h.device))[:, None]
    sc = sc.masked_fill(causal, float("-inf"))
    return torch.softmax(sc.float(), -1).mean((0, 1, 2))        # [S]


def sent_mass(w: torch.Tensor, blocks) -> torch.Tensor:
    return torch.stack([w[a:b].sum() for a, b in blocks])


@torch.no_grad()
def attn_scores(model, rec, layer: int, rows: str, dev, scan: bool = False):
    """Frozen model, native causal attention (no mask -> flash), to `layer`; returns the
    per-sentence mass [F] (or, scan=True, a list of [F] for every layer)."""
    embed, layers, _, _, rotary = model_parts(model)
    x = torch.tensor([rec["ids"]], device=dev)
    h = embed(x)
    pos = torch.arange(rec["seq"], device=dev)[None]
    c, s = rotary(h, pos)
    pe = (c.to(h.dtype), s.to(h.dtype))
    out = []
    for li in range(len(layers) if scan else layer):
        if scan:
            out.append(sent_mass(_row_mass(layers[li], h, pe, rec, rows), rec["blocks"]))
        o = layers[li](h, attention_mask=None, position_ids=pos, position_embeddings=pe)
        h = o[0] if isinstance(o, tuple) else o
    return out if scan else sent_mass(_row_mass(layers[layer], h, pe, rec, rows), rec["blocks"])


def sub_rec(rec, kept: List[int]) -> Tuple[Dict[str, Any], List[int]]:
    """Prefix + the kept blocks (original order) + tail, as a new record with remapped
    block spans; also returns the original token indices (the noRP arm's positions)."""
    idx = list(range(rec["prefix_end"]))
    blocks = []
    for j in kept:
        a, b = rec["blocks"][j]
        blocks.append((len(idx), len(idx) + b - a))
        idx.extend(range(a, b))
    fin = len(idx)
    idx.extend(range(rec["fin"], rec["seq"]))
    ids = [rec["ids"][t] for t in idx]
    out = {"ids": ids, "prefix_end": rec["prefix_end"], "blocks": blocks, "fin": fin, "seq": len(ids)}
    if "rows2" in rec:                              # copy rows ride along in the tail
        out["rows2"] = tuple(fin + r - rec["fin"] for r in rec["rows2"])
    return out, idx


def _groups(rec, chunk: int) -> List[List[int]]:
    """Consecutive sentence groups whose sub-record (prefix + group + tail) fits in `chunk` tokens."""
    budget = chunk - rec["prefix_end"] - (rec["seq"] - rec["fin"])
    assert budget > 0, "chunk smaller than prefix+tail"
    groups, cur, used = [], [], 0
    for j, (a, b) in enumerate(rec["blocks"]):
        if cur and used + b - a > budget:
            groups.append(cur)
            cur, used = [], 0
        cur.append(j)
        used += b - a
    groups.append(cur)
    return groups


def chunked_scores(model, rec, layer: int, rows: str, dev, chunk: int) -> torch.Tensor:
    """Selection with every forward inside the trained window: consecutive sentence
    groups of <= `chunk` tokens (prefix + tail included), each scored on its own, masses
    concatenated. ponytail: raw masses compared across chunks — each is a softmax over
    a similar-sized window, so they are commensurate up to the last (shorter) chunk."""
    return torch.cat([attn_scores(model, sub_rec(rec, g)[0], layer, rows, dev)
                      for g in _groups(rec, chunk)])


def tournament_scores(model, rec, layer: int, rows: str, dev, chunk: int, k_local: int,
                      trace: Optional[list] = None) -> torch.Tensor:
    """Bounded-degree selection: per window keep the top-k_local sentences, then re-score
    the union of finalists in ONE window (recursing while the union still exceeds the
    window), so every comparison that decides the kept set happens inside one softmax of
    <= `chunk` keys. Non-finalists get -inf. Reduces to attn_scores when the record fits.
    `trace` (stage-loss diagnostic): one entry per pruning stage, in the caller's sentence
    indices -- {"windows", "finalists", "rank": {sentence: rank inside its window}}."""
    groups = _groups(rec, chunk)
    if len(groups) == 1:
        return attn_scores(model, rec, layer, rows, dev).float().cpu()
    local = [attn_scores(model, sub_rec(rec, g)[0], layer, rows, dev) for g in groups]
    finalists = sorted(g[j] for g, m in zip(groups, local) for j in topk_keep(m, k_local))
    if len(finalists) >= len(rec["blocks"]):     # windows too small to prune -> plain concat
        return torch.cat(local).float().cpu()
    inner = [] if trace is not None else None
    if trace is not None:
        rank = {g[j]: r for g, m in zip(groups, local) for r, j in enumerate(m.argsort(descending=True).tolist())}
        trace.append({"windows": len(groups), "finalists": finalists, "rank": rank})
    sub = tournament_scores(model, sub_rec(rec, finalists)[0], layer, rows, dev, chunk, k_local, inner)
    for e in inner or []:                        # lift inner-stage indices to this level's
        e["finalists"] = [finalists[j] for j in e["finalists"]]
        e["rank"] = {finalists[j]: r for j, r in e["rank"].items()}
        trace.append(e)
    out = torch.full((len(rec["blocks"]),), float("-inf"))
    out[finalists] = sub
    return out


def two_round_rec(rec, kept: List[int]) -> Dict[str, Any]:
    """Round-2 scoring record: copies of the round-1 kept sentences are inserted at the
    start of the TAIL (they see the whole context and act as hop-2 queries; rec["rows2"]
    marks their rows); the context blocks are unchanged, so scores stay indexed by sentence."""
    copies = [t for j in kept for t in range(*rec["blocks"][j])]
    ids = rec["ids"][: rec["fin"]] + [rec["ids"][t] for t in copies] + rec["ids"][rec["fin"]:]
    return {**rec, "ids": ids, "seq": len(ids), "rows2": (rec["fin"], rec["fin"] + len(copies))}


def select_scores(model, rec, args, dev, trace: Optional[list] = None) -> torch.Tensor:
    """Per-sentence selection scores under the configured selector (+ extra hop rounds).
    Round r>1: the copy rows' mass per sentence (their induction hits on their own
    originals zeroed) is added to the running score, each round a distribution over
    sentences -- hop-2 sentences displace weak hop-1 ones, strong hop-1 ones stay."""
    def one(r, rows):
        if args.sel_tournament:
            return tournament_scores(model, r, args.sel_layer, rows, dev, args.sel_chunk, args.sel_tournament,
                                     trace if rows == args.sel_rows else None)
        if args.sel_chunk:
            return chunked_scores(model, r, args.sel_layer, rows, dev, args.sel_chunk)
        return attn_scores(model, r, args.sel_layer, rows, dev)
    dist = lambda m: (lambda f: f / f.sum().clamp_min(1e-12))(torch.where(torch.isfinite(m), m, 0.).float().cpu())
    mass = one(rec, args.sel_rows)
    for _ in range(args.sel_rounds - 1):
        kept = topk_keep(mass, args.k)
        s2 = dist(one(two_round_rec(rec, kept), "copies"))
        s2[kept] = 0
        mass = dist(mass) + dist(s2)
    return mass


def arm_positions(label: str, sr, idx: List[int]) -> List[int]:
    """Position ids of an arm's repacked prompt: contiguous, except attn_norp (original
    token indices) and the gap arms (a P-token hole before the tail / after the prefix)."""
    base, _, par = label.partition("@")
    if base == "attn_norp":
        return idx
    if base == "oracle_gap":
        return list(range(sr["fin"])) + [t + int(par) for t in range(sr["fin"], sr["seq"])]
    if base == "oracle_pgap":
        return list(range(sr["prefix_end"])) + [t + int(par) for t in range(sr["prefix_end"], sr["seq"])]
    return list(range(sr["seq"]))


@torch.no_grad()
def decode(model, ids: List[int], pos: List[int], eos: set, dev, max_new: int) -> List[int]:
    """Greedy, KV-cached, explicit position_ids (the noRP arm needs gapped positions)."""
    x = torch.tensor([ids], device=dev)
    p = torch.tensor([pos], device=dev)
    out = model(input_ids=x, position_ids=p, use_cache=True, logits_to_keep=1)
    past, toks, cur = out.past_key_values, [], pos[-1]
    for _ in range(max_new):
        t = int(out.logits[0, -1].argmax())
        if t in eos:
            break
        toks.append(t)
        cur += 1
        out = model(input_ids=torch.tensor([[t]], device=dev),
                    position_ids=torch.tensor([[cur]], device=dev),
                    past_key_values=past, use_cache=True)
        past = out.past_key_values
    return toks


class Embedder:
    """all-MiniLM-L6-v2 via plain transformers (mean pooling) — no sentence-transformers dep."""

    def __init__(self, dev, name="sentence-transformers/all-MiniLM-L6-v2"):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.m = AutoModel.from_pretrained(name).to(dev).eval()
        self.dev = dev

    @torch.no_grad()
    def __call__(self, texts: List[str]) -> torch.Tensor:
        vs = []
        for i in range(0, len(texts), 256):
            b = self.tok(texts[i:i + 256], padding=True, truncation=True, max_length=256,
                         return_tensors="pt").to(self.dev)
            o = self.m(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).to(o.dtype)
            vs.append(torch.nn.functional.normalize((o * m).sum(1) / m.sum(1), dim=-1))
        return torch.cat(vs)


def topk_keep(score: torch.Tensor, k: int) -> List[int]:
    return sorted(score.topk(min(k, len(score))).indices.tolist())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--root", default="data/babilong/1k+data/babilong/100")
    ap.add_argument("--mmred", default="",
                    help="MMRED text root (data/mmred_filtered/seq_len_N/test) instead of BABILong")
    ap.add_argument("--task", default="qa1")
    ap.add_argument("--length", default="4k")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--k", type=int, default=32, help="sentence budget for all selective arms")
    ap.add_argument("--sel-layer", type=int, default=20)
    ap.add_argument("--sel-rows", choices=("tail", "last"), default="tail",
                    help="tail = question+generation-prompt rows (SnapKV-style window); "
                         "last = the single answer row (MMRED convention)")
    ap.add_argument("--sel-chunk", type=int, default=0,
                    help="tokens per selection window (e.g. 32768 = score the context in "
                         "in-range chunks, global top-k); 0 = one forward over everything")
    ap.add_argument("--sel-tournament", type=int, default=0,
                    help="finalists per window; re-score their union in one window "
                         "(needs --sel-chunk); 0 = plain chunk concatenation")
    ap.add_argument("--sel-rounds", type=int, default=1,
                    help="selection hops: round r>1 re-scores with copies of round r-1's "
                         "kept sentences appended to the tail")
    ap.add_argument("--k-sweep", default="",
                    help="'+'-list of extra sentence budgets for the attn arm (attn@kK labels)")
    ap.add_argument("--gap-tokens", default="",
                    help="'+'-list of position gaps P for the oracle_gap/oracle_pgap arms")
    ap.add_argument("--arms", default="full,oracle,attn,attn_norp,lex,emb,rand")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--scan", action="store_true",
                    help="per-layer fact recall@k of the selection signal; no answering")
    ap.add_argument("--probe", action="store_true",
                    help="add osq.py answer-row metrics to full/oracle/attn/attn_norp/gap arms")
    ap.add_argument("--probe-layer", type=int, default=-1, help="default = --sel-layer")
    ap.add_argument("--jac-limit", type=int, default=10 ** 9,
                    help="Jacobian influence only for the first this-many samples")
    ap.add_argument("--yarn", type=float, default=0.0,
                    help="YaRN rope-scaling factor (e.g. 4.0 = Qwen2.5's documented "
                         "128k recipe) for ALL forwards; 0 = native config")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="outputs/babilong")
    args = ap.parse_args()
    arms = [a for a in args.arms.replace("+", ",").split(",") if a]
    assert all(a.partition("@")[0] in ARMS for a in arms), arms
    assert not args.sel_tournament or args.sel_chunk, "--sel-tournament needs --sel-chunk"
    plist = lambda s: [int(v) for v in s.split("+") if v]
    if args.k_sweep and "attn" in arms:
        arms += [f"attn@k{K}" for K in plist(args.k_sweep)]
    for g in ("oracle_gap", "oracle_pgap"):          # bare gap arms expand over --gap-tokens
        if g in arms:
            arms = [x for x in arms if x != g] + [f"{g}@{P}" for P in plist(args.gap_tokens)]
    if args.probe_layer < 0:
        args.probe_layer = args.sel_layer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.root)

    kw = {}
    if args.yarn:
        from transformers import AutoConfig
        orig = AutoConfig.from_pretrained(args.model).max_position_embeddings
        kw["rope_scaling"] = {"type": "yarn", "factor": args.yarn,
                              "original_max_position_embeddings": orig}
        kw["max_position_embeddings"] = int(orig * args.yarn)
    tok, model = load_text_model(args.model, device=dev, **kw)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    if args.mmred:                                    # run dir = <output>/<model>/mmred_filtered_seq_len_N/
        mroot = Path(args.mmred)
        args.task, args.length = mroot.parent.parent.name, mroot.parent.name
        df, vocab = load_mmred(mroot, args.seed), set()
    else:
        df, vocab = load_split(root, args.task, args.length), fact_vocab(root, args.task)
    df = df.iloc[args.offset: args.offset + args.limit]
    emb = Embedder(dev) if "emb" in arms else None
    rng = random.Random(args.seed)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    odir = Path(args.output) / args.model.split("/")[-1] / f"{args.task}_{args.length}" / f"{stamp}_{os.getpid()}"
    odir.mkdir(parents=True, exist_ok=True)
    (odir / "config.json").write_text(json.dumps(vars(args), indent=1))
    fout = open(odir / "results.jsonl", "w")
    acc: Dict[str, List[int]] = {a: [] for a in arms}
    rec_at_k: Dict[str, List[float]] = {a: [] for a in arms}
    stage_rec: List[List[float]] = []                 # per example: recall after each pruning stage
    metr: Dict[str, Dict[str, List[float]]] = {}
    scan_rec: Optional[torch.Tensor] = None
    t0 = time.time()

    for i, r in enumerate(df.itertuples()):
        sents = r.input.split("\n") if args.mmred else split_sents(r.input)
        rec = build(tok, args.task, r.question, sents, MMRED_PROMPT if args.mmred else None)
        facts = list(r.facts) if args.mmred else [j for j, s in enumerate(sents) if s in vocab]
        F = len(sents)
        if args.scan:
            per_layer = attn_scores(model, rec, 0, args.sel_rows, dev, scan=True)
            rc = torch.tensor([len(set(topk_keep(m, args.k)) & set(facts)) / max(len(facts), 1)
                               for m in per_layer])
            scan_rec = rc if scan_rec is None else scan_rec + rc
            continue
        trace: List[dict] = []
        mass = select_scores(model, rec, args, dev, trace) if any(a.startswith("attn") for a in arms) else None
        gold = tok(str(r.target), add_special_tokens=False).input_ids[0] if args.probe else None
        # stage-loss diagnostic: fact recall after each tournament pruning stage (+ final top-k),
        # and every fact's rank inside its window at each stage (None once it is out)
        stages = {"stage_recall": [len(set(e["finalists"]) & set(facts)) / max(len(facts), 1) for e in trace],
                  "stage_windows": [e["windows"] for e in trace],
                  "fact_rank": [[e["rank"].get(j) for j in facts] for e in trace]} if trace else {}
        if trace:
            stage_rec.append(stages["stage_recall"])
        keeps: Dict[str, List[int]] = {}
        for a in arms:
            base, _, par = a.partition("@")
            if base == "full":
                keeps[a] = list(range(F))
            elif base in ("oracle", "oracle_gap", "oracle_pgap"):
                keeps[a] = facts
            elif base == "attn":
                keeps[a] = topk_keep(mass, int(par[1:]) if par else args.k)
            elif base == "attn_norp":
                keeps[a] = topk_keep(mass, args.k)
            elif base == "lex":
                qw = {w for w in re.findall(r"[a-z]+", r.question.lower())} - _STOP
                sc = torch.tensor([float(len(qw & set(re.findall(r"[a-z]+", s.lower()))))
                                   + j * 1e-6 for j, s in enumerate(sents)])  # tie -> recency
                keeps[a] = topk_keep(sc, args.k)
            elif base == "emb":
                v = emb([r.question] + sents)
                keeps[a] = topk_keep(v[1:] @ v[0], args.k)
            elif base == "rand":
                keeps[a] = sorted(rng.sample(range(F), min(args.k, F)))
        for a in arms:
            kept = keeps[a]
            sr, idx = sub_rec(rec, kept)
            ids = sr["ids"]
            pos = arm_positions(a, sr, idx)
            text = tok.decode(decode(model, ids, pos, eos, dev, args.max_new))
            if args.mmred:                            # first integer in the output == gold
                m = re.search(r"\d+", text)
                ok = bool(m) and int(m[0]) == int(r.target)
            else:
                ok = bool(compare_answers(r.target, text, r.question, TASK_LABELS[args.task]))
            rcl = len(set(kept) & set(facts)) / max(len(facts), 1)
            acc[a].append(int(ok))
            rec_at_k[a].append(rcl)
            row = {"i": args.offset + i, "arm": a, "ok": ok, "out": text,
                   "target": r.target, "n_sents": F, "n_facts": len(facts),
                   "kept": len(kept), "recall": rcl, "tokens": len(ids), **(stages if a == "attn" else {})}
            if args.probe and a.partition("@")[0] in PROBED and kept and len(facts):
                kf = [n for n, j in enumerate(kept) if j in set(facts)]   # evidence among kept
                row.update(osq_probe(model, ids, pos, sr["blocks"], kf, args.probe_layer, gold,
                                     dev, jac=i < args.jac_limit))
                for key in ("mass", "keff_sent", "gap", "m_eff", "lse_e", "lse_j", "smax_e", "infl"):
                    if row.get(key) is not None:
                        metr.setdefault(a, {}).setdefault(key, []).append(row[key])
            fout.write(json.dumps(row) + "\n")
        if i % 10 == 0:
            print(f"[{i+1}/{len(df)}] {time.time()-t0:.0f}s F={F} facts={len(facts)} seq={rec['seq']} "
                  + " ".join(f"{a}={sum(acc[a])/len(acc[a]):.2f}" for a in arms), flush=True)

    if args.scan:
        rc = (scan_rec / len(df)).tolist()
        print("per-layer fact recall@%d (%s rows):" % (args.k, args.sel_rows))
        for li, v in enumerate(rc):
            print(f"  L{li:02d} {v:.3f}")
        (odir / "scan.json").write_text(json.dumps({"recall_at_k": rc, "k": args.k}))
        return 0
    summ = {a: {"acc": sum(acc[a]) / len(acc[a]), "recall": sum(rec_at_k[a]) / len(rec_at_k[a]),
                "n": len(acc[a]), **{k: sum(v) / len(v) for k, v in metr.get(a, {}).items()}}
            for a in arms}
    summ["_meta"] = {"model": args.model, "task": args.task, "length": args.length,
                     "k": args.k, "sel_layer": args.sel_layer, "sel_rows": args.sel_rows,
                     "sel_chunk": args.sel_chunk, "sel_tournament": args.sel_tournament,
                     "sel_rounds": args.sel_rounds, "yarn": args.yarn, "probe": args.probe,
                     "probe_layer": args.probe_layer, "elapsed_s": time.time() - t0}
    if stage_rec:                                     # mean recall after stage 1, 2, ... (examples with that many stages)
        depth = max(map(len, stage_rec))
        summ["_stages"] = [sum(s[d] for s in stage_rec if len(s) > d) / sum(len(s) > d for s in stage_rec)
                           for d in range(depth)]
        print("tournament stage recall:", " -> ".join(f"{v:.3f}" for v in summ["_stages"]),
              f"-> final {sum(rec_at_k['attn']) / len(rec_at_k['attn']):.3f}" if "attn" in arms else "")
    (odir / "summary.json").write_text(json.dumps(summ, indent=1))
    print(f"\n{args.model} {args.task} {args.length} n={len(df)} k={args.k} L{args.sel_layer}")
    for a in arms:
        extra = "".join(f"  {k} {v:.3f}" for k, v in summ[a].items() if k in metr.get(a, {}))
        print(f"  {a:16s} acc {summ[a]['acc']:.3f}  fact-recall {summ[a]['recall']:.3f}{extra}")
    print("->", odir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
