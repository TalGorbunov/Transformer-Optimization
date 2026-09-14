"""Training-free fenced judge, VISION (PREREG_AGG 2026-09-07: A1, A3). Frozen 4-bit Qwen2.5-VL-7B; CarrierEngine layout
(question first, one block per frame, M-RoPE per-block reset, fence at all layers) but each frame is followed by a cue
"Is {who} in the {room} in this image? Answer:" + the carrier marker; the verdict is the model's own logit(Yes) - logit(No)
at the ":" token (the position before the marker); count = #(verdict > 0). No trained parameters. Configs "mode:fence"."""
import argparse, json, os, random, re, sys, time
from pathlib import Path
import torch
sys.path.insert(0, "."); sys.path.insert(0, "outputs/_scratch/dbg")
from gnnformer.runtime import load_runtime, get_layers, image_token_groups  # noqa: E402
from gnnformer.engine import CarrierEngine, SDPA_BACKENDS  # noqa: E402
from gnnformer.carriers import make_masks  # noqa: E402
from gnnformer.fencing import find_question_spans, frame_blocks, reset_positions  # noqa: E402
from gnnformer.constants import MASK_MIN, CARRIER_TOKEN, FRAME_RESIZE  # noqa: E402
from torch.nn.attention import sdpa_kernel  # noqa: E402
from train_bindcount_vlm import load_split  # noqa: E402

YES, NO = [" Yes", "Yes", " yes", "yes"], [" No", "No", " no", "no"]


def cue_for(question, mode):
    if mode == "none": return ""
    if mode == "meta": return " Is this image relevant to the question?"
    who, room = re.fullmatch(r"How many frames show (\w+) in the (\w+)\?", question).groups(); return f" Is {who} in the {room} in this image?"


@torch.no_grad()
def prepare_cue(eng, frames, question, cue, resize, posreset):
    NF = len(frames); frames = [f.resize((resize, resize)) for f in frames]
    content = [{"type": "text", "text": question}]
    for f in frames: content += [{"type": "image", "image": f}, {"type": "text", "text": cue + " Answer:" + CARRIER_TOKEN}]
    content.append({"type": "text", "text": question})
    inputs = eng.processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    inputs = {k: (v.to(eng.dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
    ids = inputs["input_ids"][0].tolist(); seq = len(ids)
    fg = image_token_groups(inputs["input_ids"][0].cpu(), expected_num_frames=NF, processor=eng.processor)
    cpos = [p for p, t in enumerate(ids) if t == eng.carrier_id]; vstarts = [p for p, t in enumerate(ids) if t == eng.vision_start_id]
    spans = find_question_spans(ids, eng.tok, question, 2)
    if len(fg) != NF or len(cpos) != NF or len(vstarts) != NF or spans is None: return None
    fin = spans[-1][0]; blocks = frame_blocks(vstarts, fin)
    base_pos, _ = eng.rope_fn(inputs["input_ids"], image_grid_thw=inputs.get("image_grid_thw"), attention_mask=inputs.get("attention_mask"))
    if posreset:
        pos = reset_positions(base_pos, blocks, fin).clone(); b0 = int(pos[:, :, blocks[0][0]:blocks[0][1]].max())
        for i, c in enumerate(cpos): pos[:, :, c] = b0 + 1 + i
        pos[:, :, fin:] += NF
    else:
        pos = base_pos.clone()
    emb = eng.text_model.embed_tokens(inputs["input_ids"]); img = eng.model.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
    img = torch.cat(img, 0) if isinstance(img, (list, tuple)) else img
    im_mask = inputs["input_ids"][0] == eng.model.config.image_token_id; emb = emb.clone(); emb[0, im_mask] = img.to(emb.dtype)
    return dict(emb=emb[0].to(torch.bfloat16), ids=ids, pos=pos, cpos=cpos, blocks=blocks, fin=fin, seq=seq)


@torch.no_grad()
def judge(eng, rec, fenced, yes_ids, no_ids):
    tm = eng.text_model; dev = eng.dev; seq = rec["seq"]; emb = rec["emb"].to(dev).unsqueeze(0); rows = [c - 1 for c in rec["cpos"]]
    if fenced: lo, _ = make_masks(seq, rec["blocks"], rec["cpos"], rec["fin"]); m = lo.to(dev).to(torch.float32).view(1, 1, seq, seq)
    else: m = torch.full((seq, seq), MASK_MIN, device=dev).triu(1).view(1, 1, seq, seq)
    pos = rec["pos"].to(dev); cos_, sin_ = tm.rotary_emb(emb, pos); pe = (cos_.to(emb.dtype), sin_.to(emb.dtype)); h = emb
    with sdpa_kernel(SDPA_BACKENDS):
        for ly in eng.layers: h = ly(h, attention_mask=m, position_embeddings=pe)[0]
    lm = eng.model.lm_head; lg = lm(tm.norm(h)[0, rows].to(lm.weight.dtype)).float()
    z = torch.logsumexp(lg[:, yes_ids], -1) - torch.logsumexp(lg[:, no_ids], -1)
    yn = torch.tensor([int(t in set(yes_ids + no_ids)) for t in lg.argmax(-1).tolist()]); return z.cpu(), yn


def run(a):
    rt = load_runtime(a.model); nL = len(get_layers(rt.model)); dev = rt.model.device; tm = rt.model.model.language_model
    tok = rt.processor.tokenizer; cid = tok.convert_tokens_to_ids(CARRIER_TOKEN)
    with torch.no_grad(): e0 = tm.embed_tokens.weight[cid].float().clone()
    eng = CarrierEngine(rt, l_open=nL, e_c=torch.nn.Parameter(e0.to(dev)))
    yes_ids = sorted({tok(s, add_special_tokens=False).input_ids[0] for s in YES}); no_ids = sorted({tok(s, add_special_tokens=False).input_ids[0] for s in NO})
    odir = Path(a.output) / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"; odir.mkdir(parents=True, exist_ok=True)
    (odir / "config.json").write_text(json.dumps(dict(vars(a), yes_ids=yes_ids, no_ids=no_ids), indent=1)); log = open(odir / "log.txt", "a")
    def P(*x):
        s = " ".join(str(v) for v in x); print(s, flush=True); log.write(s + "\n"); log.flush()
    t0 = time.time(); results = []; checked = False
    for cfg in a.configs.split("+"):
        mode, fence = cfg.split(":"); fence = int(fence)
        for N in [int(x) for x in a.eval_ns.split("+")]:
            items = load_split(f"data/mmred_vfiltered/seq_len_{N}/test", a.limit_eval, random.Random(1))
            ex = mae = vc = vn = skipped = 0; zs, ys, yns, margins = [], [], [], []
            for frames, q, gold, facts, sid in items:
                rec = prepare_cue(eng, frames, q, cue_for(q, mode), a.resize, posreset=bool(fence))
                if rec is None: skipped += 1; continue
                if not checked:
                    b0 = rec["blocks"][0]; P("layout check block0 text:", repr(tok.decode([t for t in rec["ids"][b0[0]:b0[1]] if t != rt.model.config.image_token_id])), "| read token:", repr(tok.decode([rec["ids"][rec["cpos"][0] - 1]]))); checked = True
                z, yn = judge(eng, rec, bool(fence), yes_ids, no_ids); y = torch.zeros(len(z)); y[facts] = 1.0
                pred = int((z > 0).sum()); ex += int(pred == gold); mae += abs(pred - gold); vc += int(((z > 0).float() == y).sum()); vn += len(y)
                zs.append(z); ys.append(y); yns.append(yn)
                if y.sum() > 0 and (1 - y).sum() > 0: margins.append(float(z[y == 1].min() - z[y == 0].max()))
            n = len(items) - skipped; zz, yy = torch.cat(zs), torch.cat(ys); pos, neg = zz[yy == 1], zz[yy == 0]
            auc = float((pos[:, None] > neg[None]).float().mean() + 0.5 * (pos[:, None] == neg[None]).float().mean()) if len(pos) and len(neg) else None
            r = dict(tag=f"vision N={N}", mode=mode, fence=fence, n=n, skipped=skipped, exact=ex / max(1, n), mae=mae / max(1, n), verdict_acc=vc / max(1, vn), verdict_auc=auc,
                     yesno_top1=float(torch.cat(yns).float().mean()), min_margin_med=(float(torch.tensor(margins).median()) if margins else None))
            results.append(r); P(f"[{time.time()-t0:.0f}s] {r}"); (odir / "results.json").write_text(json.dumps(results, indent=1))
    P("->", odir); return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct"); ap.add_argument("--configs", default="direct:1+meta:1+direct:0")
    ap.add_argument("--eval-ns", default="8+16+32+64"); ap.add_argument("--limit-eval", type=int, default=100); ap.add_argument("--resize", type=int, default=FRAME_RESIZE)
    ap.add_argument("--output", default="outputs/judge_fenced_vlm"); return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
