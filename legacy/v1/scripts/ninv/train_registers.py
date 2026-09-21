#!/usr/bin/env python3
"""Phase 1 register trainer (ninv): learned merge registers on the fenced b=2 tree.

Layout/masks/positions are EXACTLY probe_tree_ninv.py's (block fence, per-block
posreset, node posreset), restricted to the b=2 arm. Node spans keep the probe's
tokenised question-replica layout, but their EMBEDDINGS are replaced by trained
register vectors (row j of a shared (R_max, hidden) table -> span position j;
level-agnostic so depth generalises to any N). Trained params:
  (a) register table            (init = mean question-replica token embedding)
  (b) LoRA on mid layers 12-19  (in-tree aggregation)
  (c) LoRA on late layers 20-27 (readout)
  (+) a shared linear aux head (hidden -> 17) — TRAINING MACHINERY ONLY: it exists
      so the per-level aux CE is computable; it is saved for probing but is not part
      of the deployed readout.
Backbone + vision tower frozen (4-bit). Answer = teacher-forced digit tokens + EOS
through the frozen LM head (support 0..16 by data filter).

ANSWER-PATH ISOLATION (the design's point): every row from the end of the last node
span onward — chat suffix, generation prompt, teacher-forced answer rows — attends
ONLY {prefix (system+question, i.e. cols < first vision_start), the ROOT node span,
itself causally}. Frames and non-root registers are invisible to the answer. The
first sample prints the answer row's open columns (in-vivo check, same style as the
node-posreset verification). Tail positions are also canonicalised (continue right
after the root span's canonical positions) so the answer's RoPE geometry is
N-invariant — without this the tail offset scales with node count, re-creating the
Phase 0 leak at the readout.

--leaf-input quantized: at layer --quant-layer (default 14) each leaf replica span
is REPLACED by a norm-matched binary verdict code (unanimous " yes"/" no" token
embedding direction, scaled to the span's mean state norm). Verdict source = a
FROZEN linear leaf probe (logistic on standardized PCA, collapsed to (w, b) in
hidden space) fit at prep on the train split with all trained params at init —
never gold labels, at train OR eval.

Usage (defaults are the campaign spec — HF steps_in_room N in {8,16}, 512px):
  python scripts/ninv/train_registers.py --leaf-input raw --output outputs/ninv/p1_armA
  python scripts/ninv/train_registers.py --leaf-input quantized --output outputs/ninv/p1_armB
Smoke: add --limit 16 --epochs 2.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts/ninv"))

from load_hf_sample import evidence_bits, iter_hf_sample_dirs, load_hf_sample  # noqa: E402

from gnnformer.carriers import Lora, attach_lora  # noqa: E402
import torch.nn as nn  # noqa: E402


def attach_masked_lora(layers, l_open, *, rank, alpha, device, holder, state=None,
                       key_offset=0):
    """attach_lora with a PER-SAMPLE ROW MASK: the LoRA delta is applied only at
    rows where holder["rows"] is 1 (node spans / tail / optionally leaf replicas),
    so every other position — frames, prefix, and in arm C the leaf spans — passes
    through the EXACT frozen backbone at every layer. holder["rows"] is a
    (seq+e,) float tensor set by forward() per sample; None = apply everywhere
    (bit-identical to plain attach_lora). Returns a carriers.Lora so state()/
    checkpoint format are unchanged. key_offset maps slice-local layer indices to
    global ones (layers[:20] with l_open=12 -> keys '12.q_proj'..)."""
    scale = alpha / rank
    params, handles = {}, []
    for li in range(l_open, len(layers)):
        for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
            mod = getattr(layers[li].self_attn, nm)
            gkey = li + key_offset
            if state is not None:
                A0, B0 = state[f"{gkey}.{nm}"]
                A = nn.Parameter(A0.float().to(device))
                B = nn.Parameter(B0.float().to(device))
            else:
                A = nn.Parameter(torch.randn(rank, mod.in_features, device=device) * 0.01)
                B = nn.Parameter(torch.zeros(mod.out_features, rank, device=device))
            params[(gkey, nm)] = (A, B)

            def mk(A=A, B=B):
                def hook(_m, inp, o):
                    x = inp[0]
                    d = (scale * (x.float() @ A.T) @ B.T).to(o.dtype)
                    rows = holder.get("rows")
                    if rows is not None:
                        d = d * rows.to(d.dtype).view(1, -1, 1)
                    return o + d
                return hook

            handles.append(mod.register_forward_hook(mk()))
    return Lora(params=params, handles=handles, l_open=l_open + key_offset,
                rank=rank, alpha=alpha)
from gnnformer.constants import MASK_MIN  # noqa: E402
from gnnformer.fencing import (  # noqa: E402
    build_replica_probe_mask,
    find_question_spans,
    frame_blocks,
    reset_positions,
)
from gnnformer.runtime import (  # noqa: E402
    get_layers,
    get_rope_index_fn,
    image_token_groups,
    load_runtime,
    move_to_device,
)

DEFAULT_ROOTS = ("data/mmred_hf/dirs/seq_len_8_train_steps_in_room=200,"
                 "data/mmred_hf/dirs/seq_len_16_train_steps_in_room=200")
# Named presets so multi-root mixtures never ride a comma through sbatch --export
# (which silently truncates at the first comma — the campaign's oldest trap).
ROOT_PRESETS = {
    "v2mix": ("data/mmred_hf/dirs/seq_len_2_train_steps_in_room=200,"
              "data/mmred_hf/dirs/seq_len_4_train_steps_in_room=200,"
              "data/mmred_hf/dirs/seq_len_8_train_steps_in_room=200,"
              "data/mmred_hf/dirs/seq_len_16_train_steps_in_room=200,"
              "data/mmred_hf/dirs/aug_dense_qa=400"),
}
R_MAX = 24          # register rows; question replica spans are ~10-16 tokens
N_CLASSES = 17      # answer/aux support 0..16


def tree_levels_b2(n_leaves: int):
    levels, cur = [], n_leaves
    while cur > 1:
        levels.append([list(range(i, min(i + 2, cur))) for i in range(0, cur, 2)])
        cur = len(levels[-1])
    return levels


def level_counts(bits, levels):
    """Per level: each node's subtree evidence count (from leaf bits)."""
    out, prev = [], list(bits)
    for groups in levels:
        cur = [sum(prev[c] for c in g) for g in groups]
        out.append(cur)
        prev = cur
    return out


def build_mask(seq, e, rep_spans, vis, blocks, sq_spans, arm_levels, root_span):
    """Probe mask + b2 node rows + the ISOLATED tail/answer rows, extended by e
    teacher-forced rows. Convention identical to probe_tree_ninv.py."""
    m_core = build_replica_probe_mask(seq, rep_spans, vis, fence_frames=True,
                                      fence_blocks=True, blocks=blocks)
    m = torch.full((seq + e, seq + e), MASK_MIN)
    m[:seq, :seq] = m_core
    prefix_cols = torch.arange(0, blocks[0][0])
    si = 0
    span_of = {}
    for li, groups in enumerate(arm_levels):
        for gi, g in enumerate(groups):
            a, b = sq_spans[si]
            span_of[(li, gi)] = (a, b)
            si += 1
            child_spans = ([rep_spans[c] for c in g] if li == 0
                           else [span_of[(li - 1, c)] for c in g])
            rows = torch.arange(a, b)
            m[rows] = MASK_MIN
            cols = torch.cat([prefix_cols]
                             + [torch.arange(ca, cb) for ca, cb in child_spans])
            m[rows.unsqueeze(1), cols.unsqueeze(0)] = 0.0
            blk = torch.zeros(b - a, b - a)
            blk.masked_fill_(torch.triu(torch.ones(b - a, b - a, dtype=torch.bool), 1),
                             MASK_MIN)
            m[a:b, a:b] = blk
    # tail + answer rows: {prefix, root span, self-causal} ONLY (no frames, no
    # non-root registers — otherwise information routes around the tree)
    t0 = sq_spans[-1][1]
    ra, rb = root_span
    tail_cols = torch.cat([prefix_cols, torch.arange(ra, rb)])
    for r in range(t0, seq + e):
        m[r] = MASK_MIN
        m[r, tail_cols] = 0.0
        m[r, t0 : r + 1] = 0.0
    return m


def canonical_positions(base_pos, blocks, sq_spans, root_span, seq, e):
    """reset_positions + node posreset + canonical tail (N-invariant answer RoPE)."""
    fin_start = sq_spans[0][0]
    pos = reset_positions(base_pos, blocks, fin_start).clone()
    s0, e0 = blocks[0]
    node_start = int(pos[:, :, s0:e0].max()) + 1
    for (a, b) in sq_spans:
        pos[:, :, a:b] = node_start + torch.arange(b - a)
    t0 = sq_spans[-1][1]
    tail_start = node_start + (root_span[1] - root_span[0])
    tail = torch.zeros(3, 1, seq + e, dtype=pos.dtype)
    tail[:, :, :seq] = pos
    tail[:, :, t0:] = tail_start + torch.arange(seq + e - t0)
    return tail, node_start, tail_start


def collapse_probe(Xtr, ytr):
    """Logistic-on-(standardize+PCA) collapsed to one (w, b) in hidden space."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    p = PCA(n_components=min(256, Xtr.shape[0] - 1, Xtr.shape[1]),
            random_state=0).fit((Xtr - mu) / sd)
    clf = LogisticRegression(max_iter=1000).fit(p.transform((Xtr - mu) / sd), ytr)
    w_eff = (clf.coef_[0] @ p.components_) / sd
    b_eff = float(clf.intercept_[0] - w_eff @ mu)
    # verify the collapse is exact (linear algebra, not approximation)
    ref = clf.decision_function(p.transform((Xtr[:8] - mu) / sd))
    got = Xtr[:8] @ w_eff + b_eff
    assert np.abs(ref - got).max() < 1e-3, "probe collapse mismatch"
    acc = float(((Xtr @ w_eff + b_eff > 0).astype(int) == ytr).mean())
    return w_eff.astype(np.float32), b_eff, acc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-roots", default=DEFAULT_ROOTS,
                    help="comma roots, each path=LIMIT (default: the campaign spec)")
    ap.add_argument("--limit", type=int, default=0,
                    help="override every per-root cap (smoke: 16)")
    ap.add_argument("--resize", type=int, default=512)
    ap.add_argument("--leaf-input", choices=("raw", "quantized"), required=True)
    ap.add_argument("--lora-rows", choices=("all", "nodes_tail", "leaves_nodes_tail"),
                    default="all",
                    help="rows the LoRA delta applies to. 'all' = arms A/B "
                         "(bit-identical to before). 'nodes_tail' = arm C: leaf/"
                         "frame/prefix rows pass through the exact frozen backbone "
                         "at every layer — the stationary-base fix. "
                         "'leaves_nodes_tail' = arm D: leaves trainable too "
                         "(their stability must come from --token-anchor)")
    ap.add_argument("--token-anchor", action="store_true",
                    help="arm D: CE targets through the frozen LM head pulling "
                         "each LEAF span's last token to its verdict token "
                         "(' yes'/' no') and each NODE span's tail tokens to the "
                         "digit token(s) of its subtree count — the model learns "
                         "its own quantizer in its own vocabulary; nothing is "
                         "injected at train or eval")
    ap.add_argument("--anchor-weight", type=float, default=1.0)
    ap.add_argument("--init-ckpt", default=None,
                    help="warm-start registers/LoRA/aux from a prior ckpt "
                         "(Phase 1.5: continue v2 rather than retrain)")
    ap.add_argument("--synthetic-n", type=int, default=0, metavar="K",
                    help="Phase 1.5 SYNTHETIC LEVEL TRAINING: add K image-free "
                         "samples whose leaves are literal ' 0'/' 1' digit text "
                         "and whose b=2 tree runs to depth log2(NF). Trains merge "
                         "levels 5-6 (dead in every arm: the 2026-08-11 negative) "
                         "with no videos; magnitudes sampled 0..16. Leaf-level "
                         "geometry differs from real samples (documented); node-"
                         "to-node geometry is identical (canonical positions)")
    ap.add_argument("--synthetic-ns", default="32 64",
                    help="virtual leaf counts for synthetic samples")
    ap.add_argument("--ans-balance", type=float, default=1.0, metavar="W",
                    help="answer-CE weight for samples with gold > 0 (gold-0 "
                         "samples keep weight 1). The 51%% zero prior makes "
                         "'always emit 0' a strong local minimum that parked "
                         "every arm's EM at the 0.510 baseline for 5-9 epochs; "
                         "W~2-3 counters the prior without distorting the task")
    ap.add_argument("--quant-layer", type=int, default=14,
                    help="write verdict codes after layers[Q] (probe reads same states)")
    ap.add_argument("--mid-open", type=int, default=12)
    ap.add_argument("--late-open", type=int, default=20)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--lr-reg", type=float, default=1e-3)
    ap.add_argument("--lr-aux", type=float, default=1e-3)
    ap.add_argument("--aux-weight", type=float, default=1.0)
    ap.add_argument("--train-frac", type=float, default=0.75)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="outputs/ninv/train_registers")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    rt = load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    layers = get_layers(model)
    text_model = model.model.language_model
    dev = model.device
    rope_fn = get_rope_index_fn(model)
    vs_id = int(model.config.vision_start_token_id)
    n_layers = len(layers)
    eos = tok.eos_token_id

    out = Path(args.output) / (time.strftime("%Y%m%d_%H%M%S")
                               + f"_{args.leaf_input}_r{args.rank}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    # ---------------- prep: layout + labels + cached image embeds (vision frozen)
    data, rep_emb_sum, rep_emb_n = [], None, 0
    n_skip = 0
    t0 = time.time()
    roots_spec = ROOT_PRESETS.get(args.data_roots, args.data_roots)
    for root in roots_spec.split(","):
        root = root.strip()
        lim = 10 ** 9
        if "=" in root:
            root, lim = root.rsplit("=", 1)
            lim = int(lim)
        if args.limit:
            lim = args.limit
        n_root = 0
        pool = iter_hf_sample_dirs(Path(root))
        # Stride, never head-slice: dirs are named ..._K<evidence>_<id>, so sorted
        # order puts every K0 (gold 0) first. A head-slice smoke gets ONLY gold-0
        # samples — the quantized arm's leaf probe then sees one class and dies,
        # and the raw arm "passes" on degenerate data (the load_hf_sample
        # self-check hit the identical trap on 2026-08-09).
        if 0 < lim < len(pool):
            pool = pool[:: max(1, len(pool) // lim)]
        for sd in pool:
            if n_root >= lim:
                break
            try:
                _sid, frames, q0, states, a0 = load_hf_sample(sd, resize=args.resize)
                gold = int(str(a0).strip())
                bits = evidence_bits(q0, states)
                if bits is None or sum(bits) != gold or gold > 16:
                    n_skip += 1
                    continue
            except Exception:
                n_skip += 1
                continue
            NF = len(frames)
            levels = tree_levels_b2(NF)
            n_nodes = sum(len(g) for g in levels)
            content = [{"type": "text", "text": q0}]
            for f in frames:
                content += [{"type": "image", "image": f},
                            {"type": "text", "text": q0}]
            content += [{"type": "text", "text": q0}] * n_nodes
            inputs = processor.apply_chat_template(
                [{"role": "user", "content": content}], add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt")
            ids = inputs["input_ids"][0].tolist()
            seq = len(ids)
            fg = image_token_groups(inputs["input_ids"][0], expected_num_frames=NF,
                                    processor=processor)
            spans = find_question_spans(ids, tok, q0, NF + 1 + n_nodes)
            vstarts = [p for p, t in enumerate(ids) if t == vs_id]
            if len(fg) != NF or spans is None or len(vstarts) != NF:
                n_skip += 1
                continue
            rep_spans = spans[1 : NF + 1]
            sq_spans = spans[NF + 1 :]
            blocks = frame_blocks(vstarts, sq_spans[0][0])
            with torch.no_grad():
                mv = move_to_device(inputs, dev)
                base_pos, _ = rope_fn(mv["input_ids"],
                                      image_grid_thw=mv.get("image_grid_thw"),
                                      attention_mask=mv.get("attention_mask"))
                img = model.model.get_image_features(mv["pixel_values"],
                                                     mv["image_grid_thw"])
                img = torch.cat(img, dim=0) if isinstance(img, (list, tuple)) else img
                emb0 = text_model.embed_tokens(mv["input_ids"])[0]
                for (a, b) in rep_spans:      # register init statistic
                    v = emb0[a:b].float().mean(0).cpu()
                    rep_emb_sum = v if rep_emb_sum is None else rep_emb_sum + v
                    rep_emb_n += 1
            tgt_ids = tok(str(gold), add_special_tokens=False).input_ids + [eos]
            data.append(dict(
                ids=inputs["input_ids"][0], img=img.detach().to(torch.float16).cpu(),
                base_pos=base_pos.cpu(), seq=seq, NF=NF, gold=gold, bits=bits,
                levels=levels, lc=level_counts(bits, levels),
                rep=rep_spans, sq=sq_spans, blocks=blocks,
                vis=[torch.tensor(sorted(int(p) for p in g)) for g in fg],
                tgt=tgt_ids, sd=str(sd)))
            n_root += 1
            if len(data) % 50 == 0:
                print(f"  prep {len(data)} (skip {n_skip}) {time.time()-t0:.0f}s "
                      f"seq={seq}", flush=True)
    # ---------------- Phase 1.5: synthetic deep-tree samples (no images)
    if args.synthetic_n > 0:
        SYN_Q = "How many steps did Mary spend in the Kitchen?"
        syn_ns = [int(x) for x in args.synthetic_ns.replace(",", " ").split()]
        rng_syn = np.random.default_rng(args.seed + 7)
        # BARE digits: Qwen tokenizes " 0" as [space, "0"] (two tokens) but "0"
        # as a single token, and always splits number strings per digit (the
        # --digit-multi precedent), so concatenated leaves stay one token each.
        tok0 = tok("0", add_special_tokens=False).input_ids
        tok1 = tok("1", add_special_tokens=False).input_ids
        assert len(tok0) == 1 and len(tok1) == 1, "digit leaves must be 1 token"
        n_syn = 0
        for k in range(args.synthetic_n):
            NF = syn_ns[k % len(syn_ns)]
            gold = int(rng_syn.integers(0, 17))
            bits = [0] * NF
            for p in rng_syn.choice(NF, size=gold, replace=False):
                bits[int(p)] = 1
            levels = tree_levels_b2(NF)
            n_nodes = sum(len(g) for g in levels)
            content = ([{"type": "text", "text": SYN_Q}]
                       + [{"type": "text", "text": "1" if b else "0"}
                          for b in bits]
                       + [{"type": "text", "text": SYN_Q}] * n_nodes)
            inputs = processor.apply_chat_template(
                [{"role": "user", "content": content}], add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt")
            ids = inputs["input_ids"][0].tolist()
            spans = find_question_spans(ids, tok, SYN_Q, 1 + n_nodes)
            if spans is None:
                continue
            pre_end, sq0 = spans[0][1], spans[1][0]
            digit_ids = {tok0[0], tok1[0]}
            leaf_pos = [p for p in range(pre_end, sq0) if ids[p] in digit_ids]
            if len(leaf_pos) != NF:
                continue
            leaf_spans = [(p, p + 1) for p in leaf_pos]
            # verify the leaf tokens match the sampled bits (order-faithful)
            if [1 if ids[p] == tok1[0] else 0 for p in leaf_pos] != bits:
                continue
            with torch.no_grad():
                mv = move_to_device(inputs, dev)
                base_pos, _ = rope_fn(mv["input_ids"], image_grid_thw=None,
                                      attention_mask=mv.get("attention_mask"))
            data.append(dict(
                ids=inputs["input_ids"][0], img=None, base_pos=base_pos.cpu(),
                seq=len(ids), NF=NF, gold=gold, bits=bits, levels=levels,
                lc=level_counts(bits, levels), rep=leaf_spans, sq=spans[1:],
                blocks=leaf_spans,
                vis=[torch.zeros(0, dtype=torch.long) for _ in range(NF)],
                tgt=tok(str(gold), add_special_tokens=False).input_ids + [eos],
                sd=f"SYNTH_N{NF}_{k}"))
            n_syn += 1
        print(f"[synthetic] added {n_syn}/{args.synthetic_n} deep-tree samples "
              f"(NF in {syn_ns}, depth up to "
              f"{max(len(tree_levels_b2(m)) for m in syn_ns)})", flush=True)
    n_done = len(data)
    print(f"prep done: n={n_done} skip={n_skip} {time.time()-t0:.0f}s "
          f"NF {Counter(d['NF'] for d in data)} "
          f"golds {Counter(d['gold'] for d in data)}", flush=True)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n_done)
    n_tr = int(n_done * args.train_frac)
    tr_idx, ev_idx = list(order[:n_tr]), list(order[n_tr:])
    (out / "train_dirs.txt").write_text("\n".join(data[i]["sd"] for i in tr_idx) + "\n")
    (out / "eval_dirs.txt").write_text("\n".join(data[i]["sd"] for i in ev_idx) + "\n")

    # ---------------- trained params
    hidden = text_model.embed_tokens.weight.shape[1]
    init_ck = torch.load(args.init_ckpt, map_location="cpu") if args.init_ckpt else None
    if init_ck is not None:
        registers = torch.nn.Parameter(init_ck["registers"].float().to(dev))
        print(f"[init-ckpt] warm start from {args.init_ckpt} "
              f"(epoch {init_ck.get('epoch')}, val_em {init_ck.get('val_em')})",
              flush=True)
    else:
        reg_init = (rep_emb_sum / rep_emb_n).to(dev)
        registers = torch.nn.Parameter(
            reg_init.unsqueeze(0).repeat(R_MAX, 1).contiguous())
    aux_head = torch.nn.Linear(hidden, N_CLASSES).to(dev).float()
    if init_ck is not None:
        aux_head.load_state_dict(init_ck["aux_head"])
    _st_mid = init_ck["lora_mid"] if init_ck is not None else None
    _st_late = init_ck["lora_late"] if init_ck is not None else None
    row_holder: dict = {"rows": None}
    if args.lora_rows == "all":
        lora_mid = attach_lora(layers[: args.late_open], args.mid_open,
                               rank=args.rank, alpha=args.alpha, device=dev,
                               state=_st_mid)
        lora_late = attach_lora(layers, args.late_open,
                                rank=args.rank, alpha=args.alpha, device=dev,
                                state=_st_late)
    else:
        lora_mid = attach_masked_lora(layers[: args.late_open], args.mid_open,
                                      rank=args.rank, alpha=args.alpha, device=dev,
                                      holder=row_holder, state=_st_mid)
        lora_late = attach_masked_lora(layers, args.late_open,
                                       rank=args.rank, alpha=args.alpha, device=dev,
                                       holder=row_holder, state=_st_late)
    if args.token_anchor and args.lora_rows == "nodes_tail":
        raise SystemExit("--token-anchor with frozen leaf rows is dead gradient: "
                         "nothing trainable produces the leaf states. Use "
                         "--lora-rows leaves_nodes_tail (arm D) or drop the anchor")
    yes_id = tok(" yes", add_special_tokens=False).input_ids[0]
    no_id = tok(" no", add_special_tokens=False).input_ids[0]
    opt = torch.optim.Adam([
        {"params": [registers], "lr": args.lr_reg},
        {"params": aux_head.parameters(), "lr": args.lr_aux},
        {"params": lora_mid.parameters(), "lr": args.lr_lora},
        {"params": lora_late.parameters(), "lr": args.lr_lora}])
    print(f"[params] registers {registers.numel()} aux {sum(p.numel() for p in aux_head.parameters())} "
          f"lora_mid({args.mid_open}-{args.late_open-1}) {lora_mid.num_parameters()} "
          f"lora_late({args.late_open}-{n_layers-1}) {lora_late.num_parameters()}", flush=True)

    # ---------------- forward
    def forward(d, train: bool, verbose: bool = False, quant=None):
        seq, e = d["seq"], len(d["tgt"])
        root_span = d["sq"][-1]
        m = build_mask(seq, e, d["rep"], d["vis"], d["blocks"], d["sq"],
                       d["levels"], root_span)
        pos, node_start, tail_start = canonical_positions(
            d["base_pos"], d["blocks"], d["sq"], root_span, seq, e)
        if args.lora_rows != "all":
            rows = torch.zeros(seq + e)
            for (a, b) in d["sq"]:
                rows[a:b] = 1.0
            rows[d["sq"][-1][1]:] = 1.0            # tail + teacher-forced rows
            if args.lora_rows == "leaves_nodes_tail":
                for (a, b) in d["rep"]:
                    rows[a:b] = 1.0
            row_holder["rows"] = rows.to(dev)
            if verbose:
                print(f"  [lora-rows] {args.lora_rows}: {int(rows.sum())}/{seq+e} "
                      f"rows carry the LoRA delta", flush=True)
        if verbose:
            open_cols = (m[seq + e - 1] == 0).nonzero(as_tuple=True)[0]
            runs, st = [], None
            for c in open_cols.tolist() + [-9]:
                if st is None:
                    st = pc = c
                elif c == pc + 1:
                    pc = c
                else:
                    runs.append((st, pc))
                    st = pc = c
            print(f"  [answer-mask] open col ranges for the LAST row: {runs[:-1] if runs and runs[-1][0]==-9 else runs} "
                  f"| prefix=[0,{d['blocks'][0][0]}) root={tuple(root_span)} "
                  f"tail=[{d['sq'][-1][1]},{seq+e})", flush=True)
            print(f"  [answer-pos] node_start={node_start} tail_start={tail_start} "
                  f"tail pos={pos[0,0,d['sq'][-1][1]:d['sq'][-1][1]+3].tolist()}... "
                  f"answer pos={int(pos[0,0,-1])}", flush=True)
        ids = d["ids"].to(dev)
        emb = text_model.embed_tokens(ids.unsqueeze(0)).clone()
        if d["img"] is not None:
            im_mask = ids == model.config.image_token_id
            emb[0, im_mask] = d["img"].to(dev).to(emb.dtype)
        for (a, b) in d["sq"]:
            emb[0, a:b] = registers[: b - a].to(emb.dtype)
        tgt_t = torch.tensor(d["tgt"], device=dev)
        emb = torch.cat([emb, text_model.embed_tokens(tgt_t.unsqueeze(0))], dim=1)
        pos = pos.to(dev)
        cos, sin = text_model.rotary_emb(emb, pos)
        pe = (cos.to(emb.dtype), sin.to(emb.dtype))
        m4 = m.to(dev).to(emb.dtype).view(1, 1, seq + e, seq + e)
        h = emb
        for li in range(n_layers):
            if train:
                h = torch.utils.checkpoint.checkpoint(
                    lambda x, ly=layers[li]: ly(x, attention_mask=m4,
                                                position_embeddings=pe)[0],
                    h, use_reentrant=False)
            else:
                h = layers[li](h, attention_mask=m4, position_embeddings=pe)[0]
            if quant is not None and li == args.quant_layer:
                # Hard quantization: verdict + code are computed WITHOUT grad and
                # written as detached constants. Two reasons: (1) a hard quantizer
                # has no gradient by definition (no straight-through requested);
                # (2) the code vectors are built once at setup from
                # embed_tokens.weight — if they stay graph-connected, every
                # sample's backward retraverses that shared setup graph and the
                # second one dies with "backward through the graph a second time"
                # (smoke iteration 2, arm B, 2026-08-10).
                w, b, cy, cn = quant
                with torch.no_grad():
                    codes = []
                    for (a, bnd) in d["rep"]:
                        span = h[0, a:bnd].float()
                        verdict = float(span.mean(0) @ w + b) > 0
                        codes.append(((cy if verdict else cn)
                                      * span.norm(dim=-1).mean()).to(h.dtype))
                hn = h.clone()
                for (a, bnd), code in zip(d["rep"], codes):
                    hn[0, a:bnd] = code
                h = hn
        hf = text_model.norm(h)
        lg_ans = model.lm_head(hf[0, seq - 1 : seq + e - 1].to(
            model.lm_head.weight.dtype)).float()
        loss_ans = F.cross_entropy(lg_ans, tgt_t)
        anchor_terms = {}
        if args.token_anchor:
            # leaf spans: last token pulled to the verdict token. SKIPPED for
            # synthetic samples — their leaves ARE literal digit tokens already;
            # anchoring them to yes/no would fight the identity.
            if d["img"] is not None:
                lp = torch.stack([hf[0, b - 1] for (a, b) in d["rep"]])
                lt = torch.tensor([yes_id if bit else no_id for bit in d["bits"]],
                                  device=dev)
                lgl = model.lm_head(lp.to(model.lm_head.weight.dtype)).float()
                anchor_terms["leaf_tok"] = F.cross_entropy(lgl, lt)
            # node spans: tail token(s) pulled to the digit token(s) of the count
            npos, ntgt = [], []
            si = 0
            for li_, groups in enumerate(d["levels"]):
                for gi, cnt in enumerate(d["lc"][li_]):
                    a, b = d["sq"][si + gi]
                    ids_ = tok(str(min(cnt, 16)), add_special_tokens=False).input_ids
                    for j, t_ in enumerate(ids_):
                        npos.append(b - len(ids_) + j)
                        ntgt.append(t_)
                si += len(groups)
            npv = torch.stack([hf[0, p] for p in npos])
            lgn = model.lm_head(npv.to(model.lm_head.weight.dtype)).float()
            anchor_terms["node_tok"] = F.cross_entropy(
                lgn, torch.tensor(ntgt, device=dev))
        aux_losses, aux_preds = [], []
        si = 0
        for li_, groups in enumerate(d["levels"]):
            states = torch.stack([h[0, d["sq"][si + gi][1] - 1]
                                  for gi in range(len(groups))]).float()
            si += len(groups)
            lab = torch.tensor(d["lc"][li_], device=dev).clamp(0, 16)
            lgx = aux_head(states)
            aux_losses.append(F.cross_entropy(lgx, lab))
            aux_preds.append(lgx.argmax(-1).tolist())
        em = bool((lg_ans.argmax(-1) == tgt_t).all())
        return loss_ans, aux_losses, em, aux_preds, anchor_terms

    # ---------------- quantized arm: fit the frozen leaf probe at init
    quant = None
    if args.leaf_input == "quantized":
        print("[quant] fitting the frozen leaf probe at init "
              f"(states = output of layers[{args.quant_layer}], train split only)...",
              flush=True)
        feats, labs = [], []
        with torch.no_grad():
            for i in tr_idx:
                d = data[i]
                ids = d["ids"].to(dev)
                emb = text_model.embed_tokens(ids.unsqueeze(0)).clone()
                emb[0, ids == model.config.image_token_id] = d["img"].to(dev).to(emb.dtype)
                for (a, b) in d["sq"]:
                    emb[0, a:b] = registers[: b - a].detach().to(emb.dtype)
                m = build_mask(d["seq"], 0, d["rep"], d["vis"], d["blocks"], d["sq"],
                               d["levels"], d["sq"][-1])
                pos, _, _ = canonical_positions(d["base_pos"], d["blocks"], d["sq"],
                                                d["sq"][-1], d["seq"], 0)
                pos = pos.to(dev)
                cos, sin = text_model.rotary_emb(emb, pos)
                pe = (cos.to(emb.dtype), sin.to(emb.dtype))
                m4 = m.to(dev).to(emb.dtype).view(1, 1, d["seq"], d["seq"])
                h = emb
                for li in range(args.quant_layer + 1):
                    h = layers[li](h, attention_mask=m4, position_embeddings=pe)[0]
                for fi, (a, b) in enumerate(d["rep"]):
                    feats.append(h[0, a:b].float().mean(0).cpu().numpy())
                    labs.append(d["bits"][fi])
        w, b, acc = collapse_probe(np.array(feats), np.array(labs))
        print(f"[quant] leaf probe train-fit acc {acc:.3f} on {len(labs)} frames "
              f"(prior {1 - np.mean(labs):.3f})", flush=True)
        wv = torch.tensor(w, device=dev)
        def _code(word):
            tid = tok(word, add_special_tokens=False).input_ids
            v = text_model.embed_tokens.weight[tid[0]].float()
            return (v / v.norm()).to(dev)
        quant = (wv, b, _code(" yes"), _code(" no"))

    # ---------------- eval
    def evaluate():
        em_n = 0
        loss_terms: dict = {}
        dist: dict = {}
        with torch.no_grad():
            for i in ev_idx:
                la, laux, em, preds, anch = forward(data[i], train=False, quant=quant)
                em_n += em
                loss_terms.setdefault("ans", []).append(float(la))
                for lv, l in enumerate(laux):
                    loss_terms.setdefault(f"lv{lv+1}", []).append(float(l))
                for k, v in anch.items():
                    loss_terms.setdefault(k, []).append(float(v))
                for lv, p in enumerate(preds):
                    dist.setdefault(lv + 1, Counter()).update(p)
        terms = {k: float(np.mean(v)) for k, v in loss_terms.items()}
        return em_n / max(len(ev_idx), 1), terms, dist

    em0, terms0, dist0 = evaluate()
    print(f"[ep 0] val EM {em0:.3f} losses "
          + " ".join(f"{k} {v:.3f}" for k, v in sorted(terms0.items()))
          + f" auxdist {dict((k, dict(v)) for k, v in dist0.items())}", flush=True)

    def save_ckpt(path, ep, em, terms):
        torch.save(dict(
            registers=registers.detach().cpu(), aux_head=aux_head.state_dict(),
            lora_mid=lora_mid.state(), lora_late=lora_late.state(),
            mid_open=args.mid_open, late_open=args.late_open, rank=args.rank,
            alpha=args.alpha, leaf_input=args.leaf_input,
            quant_layer=args.quant_layer,
            quant_probe=(None if quant is None else
                         dict(w=quant[0].cpu(), b=quant[1],
                              yes=quant[2].cpu(), no=quant[3].cpu())),
            lora_rows=args.lora_rows, token_anchor=args.token_anchor,
            epoch=ep, val_em=em, val_losses=terms, args=vars(args)), path)

    lines = [f"=== REGISTER TRAINER (arm={args.leaf_input}, n={n_done}, "
             f"train={len(tr_idx)}, mid={args.mid_open}-{args.late_open-1}, "
             f"late={args.late_open}+, r={args.rank}, resize={args.resize}, "
             f"quantL={args.quant_layer if quant else '-'}) ===",
             f"ep0 EM {em0:.3f} " + " ".join(f"{k} {v:.3f}" for k, v in sorted(terms0.items()))]
    best = (em0, -1)
    for ep in range(1, args.epochs + 1):
        rng.shuffle(tr_idx)
        te = time.time()
        tots: dict = {}
        for step, i in enumerate(tr_idx):
            la, laux, _, _, anch = forward(data[i], train=True,
                                           verbose=(ep == 1 and step == 0),
                                           quant=quant)
            w_ans = args.ans_balance if data[i]["gold"] > 0 else 1.0
            loss = w_ans * la + args.aux_weight * torch.stack(laux).mean()
            if anch:
                loss = loss + args.anchor_weight * sum(anch.values())
            (loss / args.grad_accum).backward()
            tots.setdefault("ans", []).append(float(la))
            for lv, l in enumerate(laux):
                tots.setdefault(f"lv{lv+1}", []).append(float(l))
            for k, v in anch.items():
                tots.setdefault(k, []).append(float(v))
            if (step + 1) % args.grad_accum == 0:
                opt.step()
                opt.zero_grad()
        opt.step()
        opt.zero_grad()
        em, terms, dist = evaluate()
        tr_s = " ".join(f"{k} {np.mean(v):.3f}" for k, v in sorted(tots.items()))
        ev_s = " ".join(f"{k} {v:.3f}" for k, v in sorted(terms.items()))
        print(f"[ep {ep}] train[{tr_s}] val EM {em:.3f} val[{ev_s}] "
              f"auxdist {dict((k, dict(v)) for k, v in dist.items())} "
              f"({time.time()-te:.0f}s/ep)", flush=True)
        lines.append(f"ep{ep} train[{tr_s}] val EM {em:.3f} val[{ev_s}]")
        save_ckpt(out / "registers_last.pt", ep, em, terms)
        if ep == 1:   # smoke gate: the ckpt must load back
            ck = torch.load(out / "registers_last.pt", map_location="cpu")
            assert torch.allclose(ck["registers"], registers.detach().cpu())
            assert set(ck["lora_mid"]) == set(lora_mid.state())
            print("  [ckpt] load-back OK "
                  f"({len(ck['lora_mid'])+len(ck['lora_late'])} lora tensors)", flush=True)
        if em >= best[0]:
            best = (em, ep)
            save_ckpt(out / "registers_best.pt", ep, em, terms)
        (out / "report.txt").write_text("\n".join(
            lines + [f"BEST val EM {best[0]:.3f} @ ep {best[1]}"]) + "\n")
    lines.append(f"BEST val EM {best[0]:.3f} @ ep {best[1]}")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("wrote", out, flush=True)
    lora_mid.remove()
    lora_late.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
