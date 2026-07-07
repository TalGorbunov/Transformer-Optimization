#!/usr/bin/env python3
"""Track B: single-frame (multipass) bench for InternVL2.5-8B — splits LOW-d' causes.

Each frame is shown ALONE with the same question. Measures:
  (1) per-frame PERCEPTION accuracy: single-frame gold is 0/1 ("was C in R in this frame"), the model's
      digit answer scores it directly — is the 448px resize hurting the small name text?
  (2) multipass carrier d': messages into the room token with NO other frames present — joint-pass
      interference removed. multipass d' vs joint d' (1.9) = interference share; multipass d' vs
      Qwen's bench (~3.4) = perception/transport share.
"""
from __future__ import annotations
import argparse, random, sys, time
from pathlib import Path
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluations.helpers.utils import iter_sample_dirs, load_mmred_sample
from evaluations.helpers import utils as eval_utils
from experiments.internvl.baseline_eval import build_transform
from experiments.internvl.carrier_map import rope_cos_sin, rot_half


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/mmred_images_park/seq_len_8/all_uniform")
    ap.add_argument("--layers", default="16,20")
    ap.add_argument("--offset", type=int, default=13)
    ap.add_argument("--limit", type=int, default=200, help="samples (x8 single-frame passes each)")
    ap.add_argument("--model_name", default="OpenGVLab/InternVL2_5-8B")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--output", default="outputs/frame_axis/internvl/multipass_bench")
    args = ap.parse_args()
    out = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    Ls = [int(x) for x in args.layers.replace(",", " ").split()]

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
    tfm = build_transform()
    vdt = next(p_.dtype for p_ in model.vision_model.parameters() if p_.is_floating_point())
    import copy as _copy
    cand = [(d, tok.encode(str(d), add_special_tokens=False)[0]) for d in range(9)]
    cand_vals = [d for d, _ in cand]
    cand_ids_t = torch.tensor([i for _, i in cand], dtype=torch.long)

    qkv_cap = {}
    def mk_qkv(L):
        def h(_m, _i, o):
            qkv_cap[L] = o.detach()[0]
        return h
    for L in Ls:
        lm_layers[L].attention.wqkv.register_forward_hook(mk_qkv(L))

    def build_one(frame, question):
        pv = tfm(frame).unsqueeze(0).to(vdt).cuda()
        tpl = _copy.deepcopy(model.conv_template)
        tpl.append_message(tpl.roles[0], "Frame-1: <image>\n" + question + "\nAnswer with a single number.")
        tpl.append_message(tpl.roles[1], None)
        prompt = tpl.get_prompt().replace(
            "<image>", "<img>" + "<IMG_CONTEXT>" * model.num_image_token + "</img>", 1)
        enc = tok(prompt, return_tensors="pt")
        return pv, enc["input_ids"].cuda(), enc["attention_mask"].cuda()

    msgs = {L: [] for L in Ls}
    labs, perc_ok = [], []
    all_dirs = list(iter_sample_dirs(Path(args.data_root)))
    random.Random(args.sample_seed).shuffle(all_dirs)
    n = 0; fails = 0
    for sd in all_dirs:
        if n >= args.limit:
            break
        try:
            sid, frames, q0, states, a0 = load_mmred_sample(sd)
            evid = set(int(i) for i in eval_utils.collect_evidence_frame_indices(q0, states))
            if not evid:
                continue
            for t, fr in enumerate(frames):
                lab = 1 if t in evid else 0
                pv, ids, am = build_one(fr, q0)
                S = int(ids.shape[1])
                fl = torch.ones(1, dtype=torch.long, device=pv.device)
                qkv_cap.clear()
                with torch.no_grad():
                    outp = model(pixel_values=pv, input_ids=ids, attention_mask=am, image_flags=fl)
                lg = outp.logits[0, -1].float().cpu()
                pred = int(cand_vals[int(torch.argmax(lg[cand_ids_t]).item())])
                perc_ok.append(int(pred == lab))
                idrow = ids[0].cpu()
                grp = torch.nonzero(idrow == IMG_CTX).flatten().tolist()
                car = S - 1 - args.offset
                cos_t, sin_t = rope_cos_sin(S, hd, theta, "cpu")
                for L in Ls:
                    x = qkv_cap[L].float().cpu().view(S, nKV, gs + 2, hd)
                    q = x[:, :, :gs, :].reshape(S, nH, hd).permute(1, 0, 2)
                    k = x[:, :, -2, :].permute(1, 0, 2)
                    v = x[:, :, -1, :].permute(1, 0, 2)
                    q = q * cos_t[None] + rot_half(q) * sin_t[None]
                    k = k * cos_t[None] + rot_half(k) * sin_t[None]
                    k = k.repeat_interleave(gs, 0); v = v.repeat_interleave(gs, 0)
                    sc = torch.einsum("hd,hkd->hk", q[:, car], k) * scale
                    sc[:, car + 1:] = float("-inf")
                    A = torch.softmax(sc, dim=-1)
                    pos = torch.tensor(grp, dtype=torch.long)
                    ctx = torch.einsum("hj,hjd->hd", A[:, pos], v[:, pos, :]).reshape(1, -1)
                    wo = lm_layers[L].attention.wo
                    dv = next(wo.parameters()).device
                    with torch.no_grad():
                        m = wo(ctx.to(device=dv, dtype=torch.bfloat16)).float().cpu().numpy()[0]
                    msgs[L].append(m.astype(np.float16))
                labs.append(lab)
            n += 1
            if n % 20 == 0:
                print(f"  {n}/{args.limit} samples ({len(labs)} frames; perception acc {np.mean(perc_ok):.3f})", flush=True)
        except Exception as exc:
            fails += 1
            print(f"{sd} failed: {type(exc).__name__}: {exc}")
            if fails >= 25 and n == 0:
                raise RuntimeError("25 consecutive failures with 0 successes — aborting")
            continue

    lab_a = np.array(labs)
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    rng = np.random.RandomState(0)
    # sample-disjoint split by blocks of 8 frames
    nb = len(lab_a) // 8
    bidx = rng.permutation(nb); btr = set(bidx[: int(0.6 * nb)])
    tr_m = np.array([i // 8 in btr for i in range(nb * 8)])
    lines = [f"=== InternVL2.5-8B MULTIPASS bench (n={n} samples, {len(lab_a)} single-frame passes) ===",
             f"per-frame PERCEPTION acc (digit vs 0/1 gold): {np.mean(perc_ok):.3f}   (Qwen look-again ref: 0.96-0.99)"]
    for L in Ls:
        M = np.stack(msgs[L])[: nb * 8].astype(np.float32)
        y = lab_a[: nb * 8]
        Xtr, ytr = M[tr_m], y[tr_m]; Xte, yte = M[~tr_m], y[~tr_m]
        sub = rng.permutation(len(Xtr))[:4000]
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xtr[sub], ytr[sub])
        w = lda.coef_[0] / (np.linalg.norm(lda.coef_[0]) + 1e-12)
        pE, pN = Xte[yte == 1] @ w, Xte[yte == 0] @ w
        d = float(abs(pE.mean() - pN.mean()) / (0.5 * (pE.std() + pN.std()) + 1e-12))
        lines.append(f"  L{L}: multipass carrier d' = {d:.2f}   (joint was {'1.79' if L==16 else '1.90'})")
    rep = "\n".join(lines) + "\n"
    print(rep)
    (out / "report.txt").write_text(rep)
    torch.save({"msgs": {L: np.stack(msgs[L]) for L in Ls}, "labels": lab_a,
                "perception_ok": np.array(perc_ok), "config": vars(args)}, out / "bench_cache.pt")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
