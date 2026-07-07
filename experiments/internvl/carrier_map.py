#!/usr/bin/env python3
"""Track B phase 2: InternVL2.5-8B carrier-localization map (per-token x layer message d').

InternLM2 attention is fused: wqkv packs (b, s, kv_heads=8, 2+groups=6, 128) with q in slots :4,
k at -2, v at -1 (verified against the cached remote code). RoPE is llama-style, theta=1e6, plain
at our lengths. We hook wqkv, recompute the carrier-row softmax offline (fp32), and build per-frame
messages m_{f->c} = wo(concat_h sum_{j in f} A[c,j] v_j) for the last --max-offset question tokens.

Self-check (first sample): reconstruct the full carrier-row attention context and compare with the
model's own wo INPUT at that position — cos must exceed 0.98 or we abort (unpack/RoPE bug guard).

Outputs: report with per-layer x offset held-out LDA d' + messages_cache.pt (msgs[L][off][n,NF,H],
labels, gold, model_correct) compatible with the existing CPU parity/decomposition tooling.
"""
from __future__ import annotations
import argparse, random, sys, time
from pathlib import Path
from collections import Counter
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluations.helpers.utils import iter_sample_dirs, load_mmred_sample
from evaluations.helpers import utils as eval_utils
from experiments.internvl.baseline_eval import build_transform


def rope_cos_sin(S, dim, theta, device):
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    t = torch.arange(S, dtype=torch.float32, device=device)
    f = torch.outer(t, inv)
    emb = torch.cat((f, f), dim=-1)
    return emb.cos(), emb.sin()


def rot_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--task", choices=["count"], default="count")
    ap.add_argument("--layers", default="8,12,16,20,24")
    ap.add_argument("--max-offset", type=int, default=13)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--model_name", default="OpenGVLab/InternVL2_5-8B")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--output", default="outputs/frame_axis/internvl/carrier_map")
    args = ap.parse_args()
    out = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    Ls = [int(x) for x in args.layers.replace(",", " ").split()]
    MAXOFF = int(args.max_offset)

    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModel.from_pretrained(args.model_name, quantization_config=bnb, trust_remote_code=True,
                                      use_flash_attn=False, low_cpu_mem_usage=True, device_map={"": 0}).eval()
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=False)
    model.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
    IMG_CTX = model.img_context_token_id
    lm_layers = model.language_model.model.layers
    cfg = model.config.llm_config
    nH, nKV, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads
    gs = nH // nKV
    theta = float(cfg.rope_theta)
    scale = hd ** -0.5
    tfm = build_transform()
    vdt = next(p_.dtype for p_ in model.vision_model.parameters() if p_.is_floating_point())
    import copy as _copy
    cand = [(d, tok.encode(str(d), add_special_tokens=False)[0]) for d in range(9)]
    cand_vals = [d for d, _ in cand]
    cand_ids_t = torch.tensor([i for _, i in cand], dtype=torch.long)

    qkv_cap, wo_in_cap = {}, {}
    def mk_qkv(L):
        def h(_m, _i, o):
            qkv_cap[L] = o.detach()[0]
        return h
    handles = [lm_layers[L].attention.wqkv.register_forward_hook(mk_qkv(L)) for L in Ls]
    def wo_pre(_m, i):
        wo_in_cap["x"] = i[0].detach()[0]
    handles.append(lm_layers[Ls[0]].attention.wo.register_forward_pre_hook(wo_pre))

    def build(frames, question):
        pv = torch.cat([tfm(f).unsqueeze(0) for f in frames]).to(vdt).cuda()
        tpl = _copy.deepcopy(model.conv_template)
        prefix = "".join(f"Frame-{i+1}: <image>\n" for i in range(len(frames)))
        tpl.append_message(tpl.roles[0], prefix + question + "\nAnswer with a single number.")
        tpl.append_message(tpl.roles[1], None)
        prompt = tpl.get_prompt()
        for _ in frames:
            blk = "<img>" + "<IMG_CONTEXT>" * model.num_image_token + "</img>"
            prompt = prompt.replace("<image>", blk, 1)
        enc = tok(prompt, return_tensors="pt")
        return pv, enc["input_ids"].cuda(), enc["attention_mask"].cuda()

    def unpack(qkv, S):
        # (S, kvh*(gs+2)*hd) -> q (nH,S,hd), k,v (nKV,S,hd)
        x = qkv.view(S, nKV, gs + 2, hd)
        q = x[:, :, :gs, :].reshape(S, nH, hd).permute(1, 0, 2)
        k = x[:, :, -2, :].permute(1, 0, 2)
        v = x[:, :, -1, :].permute(1, 0, 2)
        return q.float(), k.float(), v.float()

    dec: dict = {L: {o: [] for o in range(MAXOFF + 1)} for L in Ls}
    labels, golds, model_ok = [], [], []
    tok_ctr = {o: Counter() for o in range(MAXOFF + 1)}
    all_dirs = list(iter_sample_dirs(Path(args.data_root)))
    random.Random(args.sample_seed).shuffle(all_dirs)
    n = 0; fails = 0; checked = False
    for sd in all_dirs:
        if n >= args.limit:
            break
        try:
            sid, frames, q0, states, a0 = load_mmred_sample(sd)
            gold = int(str(a0).strip())
            evid = set(int(i) for i in eval_utils.collect_evidence_frame_indices(q0, states))
            if not evid:
                continue
            pv, ids, am = build(frames, q0)
            S = int(ids.shape[1])
            fl = torch.ones(pv.shape[0], dtype=torch.long, device=pv.device)
            qkv_cap.clear()
            with torch.no_grad():
                outp = model(pixel_values=pv, input_ids=ids, attention_mask=am, image_flags=fl)
            lg = outp.logits[0, -1].float().cpu()
            pred = int(cand_vals[int(torch.argmax(lg[cand_ids_t]).item())])
            idrow = ids[0].detach().cpu()
            # frame groups = contiguous IMG_CONTEXT runs
            isimg = (idrow == IMG_CTX).numpy()
            groups, cur = [], []
            for p_, b_ in enumerate(isimg):
                if b_:
                    cur.append(p_)
                elif cur:
                    groups.append(cur); cur = []
            if cur:
                groups.append(cur)
            if len(groups) != len(frames):
                continue
            carrier = [S - 1 - o for o in range(MAXOFF + 1)]
            cos_t, sin_t = rope_cos_sin(S, hd, theta, "cpu")
            per_layer_msgs = {}
            for L in Ls:
                q, k, v = unpack(qkv_cap[L].float().cpu(), S)
                q = q * cos_t[None] + rot_half(q) * sin_t[None]
                k = k * cos_t[None] + rot_half(k) * sin_t[None]
                k = k.repeat_interleave(gs, 0); v = v.repeat_interleave(gs, 0)
                car_t = torch.tensor(carrier, dtype=torch.long)
                sc = torch.einsum("hcd,hkd->hck", q[:, car_t], k) * scale
                allow = torch.arange(S)[None, :] <= car_t[:, None]
                sc = sc.masked_fill(~allow.unsqueeze(0), float("-inf"))
                A = torch.softmax(sc, dim=-1)                       # [H,|C|,S]
                if not checked and L == Ls[0]:
                    ctx_full = torch.einsum("hck,hkd->chd", A, v).reshape(len(carrier), -1)
                    ref = wo_in_cap["x"][car_t].float().cpu()
                    cs = torch.nn.functional.cosine_similarity(ctx_full, ref, dim=1).min().item()
                    print(f"[self-check] min cos(reconstructed ctx, model wo-input) over carrier rows = {cs:.4f}", flush=True)
                    assert cs > 0.98, "attention reconstruction mismatch — unpack/RoPE bug"
                    checked = True
                ctxs = []
                for gi, grp in enumerate(groups):
                    pos = torch.tensor(grp, dtype=torch.long)
                    ctx = torch.einsum("hcj,hjd->chd", A[:, :, pos], v[:, pos, :]).reshape(len(carrier), -1)
                    ctxs.append(ctx)
                stack = torch.stack(ctxs, 1).reshape(len(carrier) * len(groups), -1)  # [(C*F), H*hd]
                wo = lm_layers[L].attention.wo
                dv = next(wo.parameters()).device
                with torch.no_grad():
                    m_out = wo(stack.to(device=dv, dtype=torch.bfloat16)).float().cpu()
                per_layer_msgs[L] = m_out.view(len(carrier), len(groups), -1)  # [C,F,H]
            for o in range(MAXOFF + 1):
                tok_ctr[o][tok.decode([int(idrow[S - 1 - o])]).strip()] += 1
                for L in Ls:
                    dec[L][o].append(per_layer_msgs[L][o].numpy().astype(np.float16))
            labels.append(np.array([1 if t in evid else 0 for t in range(len(frames))]))
            golds.append(gold); model_ok.append(int(pred == gold))
            n += 1
            if n % 25 == 0:
                print(f"  {n}/{args.limit} (model acc so far {np.mean(model_ok):.3f})", flush=True)
        except Exception as exc:
            fails += 1
            print(f"{sd} failed: {type(exc).__name__}: {exc}")
            if fails >= 25 and n == 0:
                raise RuntimeError("25 consecutive failures with 0 successes — aborting")
            continue
    for h in handles:
        h.remove()

    lab = np.stack(labels)
    print(f"\ncollected n={n}; model acc={np.mean(model_ok):.3f}; fitting per-cell LDA d' ...", flush=True)
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    rng = np.random.RandomState(0)
    idx = rng.permutation(n); ntr = int(0.6 * n)
    lines = [f"=== InternVL2.5-8B carrier map ({args.task}; n={n}; model acc {np.mean(model_ok):.3f}) ===",
             "held-out LDA d' [rows=layer, cols=offset-from-end]",
             "carrier tokens by offset: " + " ".join(f"{o}:{tok_ctr[o].most_common(1)[0][0]!r}" for o in range(MAXOFF + 1))]
    best = (0.0, None, None)
    dmat = {}
    for L in Ls:
        row = []
        for o in range(MAXOFF + 1):
            M = np.stack(dec[L][o]).astype(np.float32)            # [n,NF,H]
            Xtr = M[idx[:ntr]].reshape(-1, M.shape[-1]); ytr = lab[idx[:ntr]].reshape(-1)
            Xte = M[idx[ntr:]].reshape(-1, M.shape[-1]); yte = lab[idx[ntr:]].reshape(-1)
            sub = rng.permutation(len(Xtr))[:4000]
            try:
                lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xtr[sub], ytr[sub])
                w = lda.coef_[0] / (np.linalg.norm(lda.coef_[0]) + 1e-12)
                pE, pN = Xte[yte == 1] @ w, Xte[yte == 0] @ w
                d = float(abs(pE.mean() - pN.mean()) / (0.5 * (pE.std() + pN.std()) + 1e-12))
            except Exception:
                d = float("nan")
            row.append(d)
            if d == d and d > best[0]:
                best = (d, L, o)
        dmat[L] = row
        lines.append(f"  L{L:>2}: " + " ".join(f"{x:5.2f}" for x in row))
    lines.append(f"PEAK: d'={best[0]:.2f} at L{best[1]} offset {best[2]}")
    rep = "\n".join(lines) + "\n"
    print(rep)
    (out / "report.txt").write_text(rep)
    torch.save({"msgs": {L: {o: np.stack(dec[L][o]) for o in range(MAXOFF + 1)} for L in Ls},
                "labels": lab, "gold": np.array(golds), "model_correct": np.array(model_ok),
                "dmat": dmat, "config": vars(args)}, out / "messages_cache.pt")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
