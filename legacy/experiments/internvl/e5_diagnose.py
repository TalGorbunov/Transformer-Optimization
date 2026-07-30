#!/usr/bin/env python3
"""Why did E5 fail on InternVL2.5-8B? Three hypotheses, one run.

H-window : carrier->last transfer completes BEFORE L12 (doses at 12/16/20 were post-window).
            -> dose at L8 should deliver to the final state where L12+ didn't.
H-axis   : message-delta-hat is rotated away from the carrier STATE's count direction.
            -> per-layer carrier-state decode under base vs scrub: if scrub doesn't dent it, wrong axis.
H-nopipe : the last token never reads the carrier at all.
            -> per-layer logit-lens (count decode from last-token state at every layer) flat at chance
               + attention geography (last-token row's mass on carrier/question/images per layer) ~ 0.

Arms: base | dose8:16 (L8 only) | doseMid:16 (L12,16,20) | scruball (dhat out at L8..24 cached layers).
Captures per arm: last-token hidden state at EVERY layer; carrier(off13) hidden state at every layer.
Base arm also logs the last-token attention row aggregated to {carrier, question, images} per layer.
"""
from __future__ import annotations
import argparse, glob, random, sys, time
from pathlib import Path
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluations.helpers.utils import iter_sample_dirs, load_mmred_sample
from experiments.internvl.baseline_eval import build_transform
from experiments.internvl.carrier_map import rope_cos_sin, rot_half


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--cache", default="", help="carrier_map_ext messages_cache.pt (deltas)")
    ap.add_argument("--offset", type=int, default=13)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--model_name", default="OpenGVLab/InternVL2_5-8B")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--output", default="outputs/frame_axis/internvl/e5_diagnose")
    args = ap.parse_args()
    out = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    off = int(args.offset)

    cache = args.cache or sorted(glob.glob("outputs/frame_axis/internvl/carrier_map_ext/*/messages_cache.pt"))[-1]
    mc = torch.load(cache, map_location="cpu", weights_only=False)
    lab = mc["labels"].reshape(-1)
    delta, dhat = {}, {}
    for L in mc["msgs"]:
        M = mc["msgs"][L][off].astype(np.float32).reshape(-1, 4096)
        d = M[lab == 1].mean(0) - M[lab == 0].mean(0)
        delta[L] = d; dhat[L] = d / (np.linalg.norm(d) + 1e-12)
    dose_layers_mid = [12, 16, 20]; dose_layer_early = 8
    scrub_layers = sorted(delta.keys())
    print(f"deltas from {cache}: layers {scrub_layers}")

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
    gs = nH // nKV; theta = float(cfg.rope_theta); scale = hd ** -0.5
    nL = len(lm_layers)
    tfm = build_transform()
    vdt = next(p_.dtype for p_ in model.vision_model.parameters() if p_.is_floating_point())
    import copy as _copy
    cand = [(d_, tok.encode(str(d_), add_special_tokens=False)[0]) for d_ in range(9)]
    cand_vals = [d_ for d_, _ in cand]
    cand_ids_t = torch.tensor([i for _, i in cand], dtype=torch.long)
    dev = next(lm_layers[0].parameters()).device
    delta_t = {L: torch.tensor(delta[L], device=dev, dtype=torch.float32) for L in delta}
    dhat_t = {L: torch.tensor(dhat[L], device=dev, dtype=torch.float32) for L in dhat}

    st = {"mode": "base", "pos": None, "gold": 0, "last": {}, "car": {}, "attn": {}, "img_pos": None}

    def mk_pre(L):
        def pre(_m, hargs, hkwargs):
            hs = hkwargs.get("hidden_states", hargs[0] if hargs else None)
            if hs is None:
                return hargs, hkwargs
            # capture entering-layer states (last token + carrier)
            st["last"][L] = hs[0, -1, :].detach().float().cpu()
            if st["pos"] is not None and hs.shape[1] > st["pos"]:
                st["car"][L] = hs[0, st["pos"], :].detach().float().cpu()
            # edits (delta injected at entry of L, i.e. output of L-1; matches dose_scrub convention L+1)
            m = st["mode"]; ed = None
            if m == "dose8" and L == dose_layer_early + 1:
                ed = ("dose", dose_layer_early)
            elif m == "doseMid" and L - 1 in dose_layers_mid:
                ed = ("dose", L - 1)
            elif m == "scruball" and L - 1 in scrub_layers:
                ed = ("scrub", L - 1)
            if ed is not None:
                hs = hs.clone()
                h = hs[0, st["pos"], :].float()
                if ed[0] == "dose":
                    h = h + 15.0 * st["gold"] * delta_t[ed[1]]
                else:
                    u = dhat_t[ed[1]]; h = h - (h @ u) * u
                hs[0, st["pos"], :] = h.to(hs.dtype)
                if hargs:
                    return (hs,) + tuple(hargs[1:]), hkwargs
                hkwargs = dict(hkwargs); hkwargs["hidden_states"] = hs
            return hargs, hkwargs
        return pre
    for L in range(nL):
        lm_layers[L].register_forward_pre_hook(mk_pre(L), with_kwargs=True)

    def mk_qkv(L):
        def h(_m, _i, o):
            if st["mode"] != "base" or st["pos"] is None:
                return
            qkv = o.detach()[0]
            S = qkv.shape[0]
            x = qkv.view(S, nKV, gs + 2, hd)
            q = x[:, :, :gs, :].reshape(S, nH, hd).permute(1, 0, 2)
            k = x[:, :, -2, :].permute(1, 0, 2)
            cos_t, sin_t = rope_cos_sin(S, hd, theta, qkv.device)
            qq = q[:, -1:, :] * cos_t[None, -1:] + rot_half(q[:, -1:, :]) * sin_t[None, -1:]
            kk = k * cos_t[None] + rot_half(k) * sin_t[None]
            kk = kk.repeat_interleave(gs, 0)
            sc = torch.einsum("hcd,hkd->hk", qq.float(), kk.float()) * scale
            A = torch.softmax(sc, dim=-1).mean(0)          # [S] mean over heads, last-token row
            img = st["img_pos"]
            car = st["pos"]
            qspan = torch.arange(img.max().item() + 1, S, device=A.device)
            st["attn"][L] = (float(A[car]), float(A[qspan].sum() - A[car]), float(A[img].sum()))
        return h
    for L in range(nL):
        lm_layers[L].attention.wqkv.register_forward_hook(mk_qkv(L))

    def build(frames, question):
        pv = torch.cat([tfm(f).unsqueeze(0) for f in frames]).to(vdt).cuda()
        tpl = _copy.deepcopy(model.conv_template)
        prefix = "".join(f"Frame-{i+1}: <image>\n" for i in range(len(frames)))
        tpl.append_message(tpl.roles[0], prefix + question + "\nAnswer with a single number.")
        tpl.append_message(tpl.roles[1], None)
        prompt = tpl.get_prompt()
        for _ in frames:
            prompt = prompt.replace("<image>", "<img>" + "<IMG_CONTEXT>" * model.num_image_token + "</img>", 1)
        enc = tok(prompt, return_tensors="pt")
        return pv, enc["input_ids"].cuda(), enc["attention_mask"].cuda()

    arms = ["base", "dose8", "doseMid", "scruball"]
    store = {a: {"last": [], "car": [], "pred": []} for a in arms}
    golds = []; attn_geo = []
    all_dirs = list(iter_sample_dirs(Path(args.data_root)))
    random.Random(args.sample_seed).shuffle(all_dirs)
    n = 0; fails = 0
    for sd in all_dirs:
        if n >= args.limit:
            break
        try:
            sid, frames, q0, states, a0 = load_mmred_sample(sd)
            gold = int(str(a0).strip())
            pv, ids, am = build(frames, q0)
            S = int(ids.shape[1])
            st["pos"] = S - 1 - off; st["gold"] = gold
            st["img_pos"] = torch.nonzero(ids[0] == IMG_CTX).flatten()
            fl = torch.ones(pv.shape[0], dtype=torch.long, device=pv.device)
            for a in arms:
                st["mode"] = a
                st["last"], st["car"], st["attn"] = {}, {}, {}
                with torch.no_grad():
                    outp = model(pixel_values=pv, input_ids=ids, attention_mask=am, image_flags=fl)
                lg = outp.logits[0, -1].float().cpu()
                store[a]["pred"].append(int(cand_vals[int(torch.argmax(lg[cand_ids_t]).item())]))
                store[a]["last"].append(torch.stack([st["last"][L] for L in range(nL)]).half())
                store[a]["car"].append(torch.stack([st["car"][L] for L in range(nL)]).half())
                if a == "base":
                    attn_geo.append([st["attn"][L] for L in range(nL)])
            golds.append(gold)
            n += 1
            if n % 20 == 0:
                print(f"  {n}/{args.limit}", flush=True)
        except Exception as exc:
            fails += 1
            print(f"{sd} failed: {type(exc).__name__}: {exc}")
            if fails >= 25 and n == 0:
                raise RuntimeError("25 consecutive failures with 0 successes — aborting")
            continue

    gold_a = np.array(golds)
    from sklearn.linear_model import RidgeClassifier
    rng = np.random.RandomState(0); idx = rng.permutation(n); ntr = int(0.6 * n); te = idx[ntr:]
    lines = [f"=== E5 DIAGNOSIS (n={n}, off{off}) ===",
             "A) attention geography: mean last-token attention mass on [carrier | other question | images] per layer (base):"]
    geo = np.array(attn_geo)  # [n, nL, 3]
    gm = geo.mean(0)
    for L in range(nL):
        bar = "#" * int(gm[L, 0] * 200)
        lines.append(f"  L{L:>2}: car={gm[L,0]:.4f} q={gm[L,1]:.3f} img={gm[L,2]:.3f} {bar}")
    lines.append("B) per-layer count decode from LAST-token state (logit lens), per arm:")
    for a in arms:
        X = torch.stack(store[a]["last"]).float().numpy()  # [n, nL, H]
        accs = []
        for L in range(0, nL, 2):
            clf = RidgeClassifier(alpha=10.0).fit(X[idx[:ntr], L], gold_a[idx[:ntr]])
            accs.append(float(np.mean(clf.predict(X[te, L]) == gold_a[te])))
        lines.append(f"  {a:>8}: " + " ".join(f"L{L}:{acc:.2f}" for L, acc in zip(range(0, nL, 2), accs)))
    lines.append("C) per-layer count decode from CARRIER state, base vs scruball (axis check):")
    for a in ["base", "scruball"]:
        X = torch.stack(store[a]["car"]).float().numpy()
        accs = []
        for L in range(0, nL, 2):
            clf = RidgeClassifier(alpha=10.0).fit(X[idx[:ntr], L], gold_a[idx[:ntr]])
            accs.append(float(np.mean(clf.predict(X[te, L]) == gold_a[te])))
        lines.append(f"  {a:>8}: " + " ".join(f"L{L}:{acc:.2f}" for L, acc in zip(range(0, nL, 2), accs)))
    lines.append("D) emitted per arm: " + "  ".join(
        f"{a}={np.mean(np.array(store[a]['pred'])==gold_a):.3f}" for a in arms))
    rep = "\n".join(lines) + "\n"
    print(rep)
    (out / "report.txt").write_text(rep)
    torch.save({"golds": gold_a, "attn_geo": geo,
                "preds": {a: store[a]["pred"] for a in arms}, "config": vars(args)}, out / "diag.pt")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
