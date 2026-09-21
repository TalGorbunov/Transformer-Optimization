#!/usr/bin/env python3
"""q/kv swap 2x2: decompose the joint-context tax into its QUERY and VALUE halves.

Two forwards per sample on the SAME Q-first replica prompt with SHARED (non-reset)
positions — only the mask differs:
  CLEAN  block-fence mask (islands; mask-only, no posreset so q/k mixing is
         position-consistent)
  DIRTY  no mask (plain causal joint)

At layer --layer, the pre-attention hidden states from both forwards are combined
in a 2x2: the read-position query (replica room token, the validated anchor) from
arm A, the frame-token keys/values from arm B. The own-frame message is computed
with an IDENTICAL softmax key-set in all four cells (own frame's vision tokens
only), so the cells differ ONLY in whose hidden states feed q vs k/v:

    (q clean, kv clean)  = fenced supply         (in-experiment upper anchor)
    (q clean, kv dirty)  = VALUE POLLUTION isolated
    (q dirty, kv clean)  = QUERY SQUASHING isolated
    (q dirty, kv dirty)  = joint supply          (in-experiment lower anchor)

Each cell: held-out d' + gate->tally (gate_tally.py protocol). Direct, paired
measurement of the dissociation RESULTS.md [2026-07-14] inferred from mask arms
("the mask cleans queries everywhere but values only for frame 0"; clean queries
alone ~+28%; both clean -> A3 band).

Usage:
  python scripts/probe_qkv_swap.py --limit 150 --layer 16 \
      --output outputs/presentation/qkv_swap/<ts>
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)

from gnnformer.constants import ROOMS
from gnnformer.data import iter_sample_dirs_shuffled, load_mmred_sample, probe_evidence
from gnnformer.fencing import (
    build_replica_probe_mask,
    find_question_spans,
    frame_blocks,
    locate_word_token,
)
from gnnformer.metrics import dprime_pair
from gnnformer.runtime import (
    attention_dims,
    dequantize_linear_weight,
    get_layers,
    get_rope_index_fn,
    image_token_groups,
    load_runtime,
    move_to_device,
)

CELLS = ["qC_kvC", "qC_kvD", "qD_kvC", "qD_kvD"]
CELL_LABEL = {"qC_kvC": "q clean\nkv clean", "qC_kvD": "q clean\nkv dirty",
              "qD_kvC": "q dirty\nkv clean", "qD_kvD": "q dirty\nkv dirty"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--resize", type=int, default=392)
    ap.add_argument("--task", default="steps")
    ap.add_argument("--shuffle-dirs", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rt = load_runtime(args.model) if args.model else load_runtime()
    model, processor, tok = rt.model, rt.processor, rt.tokenizer
    layers = get_layers(model)
    L = args.layer
    dims = attention_dims(model)
    n_heads, n_kv, hd = dims["n_heads"], dims["n_kv"], dims["head_dim"]
    text_model = model.model.language_model
    dev = model.device
    rope_fn = get_rope_index_fn(model)
    vs_id = int(model.config.vision_start_token_id)
    W_O = dequantize_linear_weight(layers[L].self_attn.o_proj)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    def rope_qk(q_flat, k_flat, pe, seq):
        q = q_flat.view(1, seq, n_heads, hd).transpose(1, 2)
        k = k_flat.view(1, seq, n_kv, hd).transpose(1, 2)
        qr, kr = apply_multimodal_rotary_pos_emb(
            q.float(), k.float(), pe[0].float(), pe[1].float(), dims["mrope_section"])
        return qr[0], repeat_kv(kr, n_heads // n_kv)[0]

    feats = {c: [] for c in CELLS}
    ys, golds = [], []
    n_done = n_skip = 0
    t0 = time.time()
    for sd in iter_sample_dirs_shuffled(Path(args.data_root), args.shuffle_dirs):
        if n_done >= args.limit:
            break
        try:
            _sid, frames, q0, states, a0 = load_mmred_sample(sd)
            gold = int(str(a0).strip())
            pe_ = probe_evidence(args.task, q0, states, gold, ROOMS)
            if pe_ is None:
                n_skip += 1
                continue
            evid, room = pe_
        except Exception:
            n_skip += 1
            continue
        if not evid and gold != 0:
            n_skip += 1
            continue
        NF = len(frames)
        if args.resize > 0:
            frames = [f.resize((args.resize, args.resize)) for f in frames]
        content = [{"type": "text", "text": q0}]
        for f in frames:
            content.append({"type": "image", "image": f})
            content.append({"type": "text", "text": q0})
        content.append({"type": "text", "text": q0})
        inputs = processor.apply_chat_template([{"role": "user", "content": content}],
                                               add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt")
        inputs = move_to_device(inputs, dev)
        ids = inputs["input_ids"][0].tolist()
        seq = len(ids)
        fg = image_token_groups(inputs["input_ids"][0].cpu(), expected_num_frames=NF,
                                processor=processor)
        spans = find_question_spans(ids, tok, q0, NF + 2)
        vstarts = [p for p, t in enumerate(ids) if t == vs_id]
        if len(fg) != NF or spans is None or len(vstarts) != NF:
            n_skip += 1
            continue
        spans = spans[1:]
        rep_spans, fin_span = spans[:NF], spans[NF]
        rep_c = [locate_word_token(ids, tok, room, sp) for sp in rep_spans]
        if any(c is None for c in rep_c):
            n_skip += 1
            continue
        blocks = frame_blocks(vstarts, fin_span[0])
        vis = [torch.tensor(sorted(int(p) for p in g), dtype=torch.long) for g in fg]
        m_fence = build_replica_probe_mask(seq, rep_spans, vis, fence_frames=True,
                                           fence_blocks=True, blocks=blocks)
        m_causal = build_replica_probe_mask(seq, [], [torch.tensor([], dtype=torch.long)])

        with torch.no_grad():
            base_pos, _ = rope_fn(inputs["input_ids"], image_grid_thw=inputs.get("image_grid_thw"),
                                  attention_mask=inputs.get("attention_mask"))
            emb = text_model.embed_tokens(inputs["input_ids"])
            img = model.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
            img = torch.cat(img, dim=0) if isinstance(img, (list, tuple)) else img
            im_mask = inputs["input_ids"][0] == model.config.image_token_id
            emb = emb.clone()
            emb[0, im_mask] = img.to(emb.dtype)
            cos_, sin_ = text_model.rotary_emb(emb, base_pos.to(dev))
            pe = (cos_.to(emb.dtype), sin_.to(emb.dtype))

            qk = {}
            at = layers[L].self_attn
            for arm, mask2d in (("C", m_fence), ("D", m_causal)):
                mask4 = mask2d.to(dev).to(emb.dtype).view(1, 1, seq, seq)
                h = emb
                for ly in layers[:L]:
                    h = ly(h, attention_mask=mask4, position_embeddings=pe)[0]
                ln = layers[L].input_layernorm(h)
                qr, kr = rope_qk(at.q_proj(ln), at.k_proj(ln), pe, seq)
                vv = repeat_kv(at.v_proj(ln).view(1, seq, n_kv, hd).transpose(1, 2),
                               n_heads // n_kv)[0].float()
                qk[arm] = (qr, kr, vv)

            for cell in CELLS:
                qa, kb = cell[1], cell[5]  # "qC_kvD" -> q arm 'C', kv arm 'D'
                qr = qk[qa][0]
                kr, vv = qk[kb][1], qk[kb][2]
                msgs = []
                for i, c in enumerate(rep_c):
                    fidx = vis[i].to(dev)
                    lg = torch.einsum("hd,htd->ht", qr[:, c], kr[:, fidx]) / (hd ** 0.5)
                    wgt = torch.softmax(lg, -1)  # identical key-set in all 4 cells
                    ctx = torch.einsum("ht,htd->hd", wgt, vv[:, fidx]).reshape(-1)
                    msgs.append((W_O.to(dev) @ ctx).cpu().numpy())
                feats[cell].append(np.stack(msgs))
        ys.append([1 if t in evid else 0 for t in range(NF)])
        golds.append(gold)
        n_done += 1
        if n_done % 25 == 0:
            print(f"  {n_done}/{args.limit} (skip {n_skip}) {time.time()-t0:.0f}s", flush=True)

    from sklearn.linear_model import LogisticRegression

    Y = np.array(ys)
    G = np.array(golds)
    lines = [f"=== Q/KV SWAP 2x2 (n={n_done}, skip={n_skip}, L{L}, Q-first replica layout, "
             f"mask-only fence, shared base positions, own-frame softmax keys, "
             f"data={args.data_root}) ==="]
    res = {}
    for cell in CELLS:
        X = np.stack(feats[cell]).astype(np.float32)
        n, NF, H = X.shape
        dp, sdv, auc = dprime_pair(X, Y)
        accs = []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            idx = rng.permutation(n)
            tr, ev = idx[: n // 2], idx[n // 2:]
            clf = LogisticRegression(max_iter=2000).fit(X[tr].reshape(-1, H), Y[tr].reshape(-1))
            pr = clf.predict(X[ev].reshape(-1, H)).reshape(len(ev), NF)
            accs.append(float((pr.sum(1) == G[ev]).mean()))
        res[cell] = (dp, sdv, float(np.mean(accs)), float(np.std(accs)))
        lines.append(f"{cell}: d' {dp:.2f}±{sdv:.2f} (auc {auc:.2f})  "
                     f"gate->tally {np.mean(accs):.3f}±{np.std(accs):.3f}")
        print(lines[-1], flush=True)
    dfull = res["qC_kvC"][0] - res["qD_kvD"][0]
    lines += [f"[decomposition] total tax (clean-clean minus dirty-dirty) = {dfull:.2f} d'",
              f"  query half  (kv clean, q dirty->clean): {res['qC_kvC'][0]-res['qD_kvC'][0]:.2f}",
              f"  value half  (q clean, kv dirty->clean): {res['qC_kvC'][0]-res['qC_kvD'][0]:.2f}"]
    np.savez(out / "qkv_swap.npz", Y=Y, G=G,
             **{c: np.stack(feats[c]).astype(np.float16) for c in CELLS})

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2))
    for ax, k_, ylab in [(axes[0], 0, "held-out d′"), (axes[1], 2, "gate→tally exact")]:
        M = np.array([[res["qC_kvC"][k_], res["qC_kvD"][k_]],
                      [res["qD_kvC"][k_], res["qD_kvD"][k_]]])
        im = ax.imshow(M, cmap="Blues", vmin=0)
        for r in range(2):
            for c in range(2):
                ax.annotate(f"{M[r, c]:.2f}", (c, r), ha="center", va="center",
                            fontsize=13, color="#0b0b0b" if M[r, c] < M.max() * 0.7 else "white")
        ax.set_xticks([0, 1], ["kv clean", "kv dirty"])
        ax.set_yticks([0, 1], ["q clean", "q dirty"])
        ax.set_title(ylab, fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"q/kv swap 2×2 @L{L} (n={n_done}) — where does the joint-context tax live?",
                 fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"qkv_swap.{ext}", dpi=300)

    (out / "report.txt").write_text("\n".join(lines) + "\n")
    (out / "ABOUT.md").write_text(
        "# q/kv swap 2x2 — query squashing vs value pollution, measured directly\n\n"
        "Two forwards per sample on the SAME Q-first replica prompt with shared (non-reset)\n"
        "positions: fenced (clean) and unmasked joint (dirty). At layer " + str(L) + " the\n"
        "read-position query and the frame-token keys/values are mixed across arms in a 2x2;\n"
        "the softmax key-set is identical (own frame only) in all four cells, so cells differ\n"
        "only in whose hidden states feed q vs k/v. Off-diagonal cells isolate the two halves\n"
        "of the joint-context tax: a clean reader on polluted frames (value pollution) and a\n"
        "polluted reader on clean frames (query squashing). Direct paired measurement of the\n"
        "dissociation inferred in RESULTS.md [2026-07-14] from imperfect mask arms. Corners\n"
        "anchor the experiment internally (clean-clean ~ mask-only fence band; dirty-dirty ~\n"
        "unmasked band). NOTE: no posreset anywhere (position-consistent mixing), so the\n"
        f"clean corner sits below the full A3 band. Data: {args.data_root}, n={n_done}.\n"
        "Artifacts: qkv_swap.png/pdf, qkv_swap.npz, report.txt.\n")
    print("\n".join(lines))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
