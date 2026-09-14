"""Bind, then count — MMRED VISION. CarrierEngine layout (question first, one carrier token per frame,
block fence, M-RoPE position reset); the fence (mask_lo) at ALL layers; linear verdict head on the
carriers' final states; count = #(verdict > 0). Arms: sum (fenced) | sum_nofence (causal, native pos).
Frozen Qwen2.5-VL-7B (4-bit); trainable: carrier embedding + verdict head (~7k params)."""
import argparse, json, os, random, re, sys, time
from pathlib import Path
import torch, torch.nn.functional as F
sys.path.insert(0, ".")
from gnnformer.runtime import load_runtime, get_layers  # noqa: E402
from gnnformer.engine import CarrierEngine, SDPA_BACKENDS  # noqa: E402
from gnnformer.carriers import make_masks  # noqa: E402
from gnnformer.constants import MASK_MIN, CARRIER_TOKEN, FRAME_RESIZE  # noqa: E402
from gnnformer.data import load_mmred_sample  # noqa: E402
from torch.nn.attention import sdpa_kernel  # noqa: E402


def facts_of(states, q):
    qc, qr = re.fullmatch(r"How many frames show (\w+) in the (\w+)\?", q).groups()
    return [j for j, st in enumerate(states) if qc in st["rooms"].get(qr, [])]


def forward_carriers(eng, rec, e_c, fenced, grad_ckpt):
    tm = eng.text_model; dev = eng.dev; seq = rec["seq"]
    emb = rec["emb"].to(dev).unsqueeze(0).clone()
    cp = torch.tensor(rec["cpos"], device=dev); emb[0, cp] = e_c.to(emb.dtype).unsqueeze(0).expand(len(rec["cpos"]), -1)
    if fenced:
        lo, _ = make_masks(seq, rec["blocks"], rec["cpos"], rec["fin"]); m = lo.to(dev).to(torch.float32).view(1, 1, seq, seq)
    else:
        m = torch.full((seq, seq), MASK_MIN, device=dev).triu(1).view(1, 1, seq, seq)
    pos = rec["pos"].to(dev); cos_, sin_ = tm.rotary_emb(emb, pos); pe = (cos_.to(emb.dtype), sin_.to(emb.dtype))
    def _blk(hh, ly):
        with sdpa_kernel(SDPA_BACKENDS):
            return ly(hh, attention_mask=m, position_embeddings=pe)[0]
    h = emb
    for ly in eng.layers:
        h = torch.utils.checkpoint.checkpoint(_blk, h, ly, use_reentrant=False) if (grad_ckpt and torch.is_grad_enabled()) else _blk(h, ly)
    return tm.norm(h)[0, cp]


def herbench_items(arm_root, split, limit, rng, test_frac=0.3):
    """data/herbench_ac/ev_fill<N>/<qid>/{000.jpg.., meta.json}; split by VIDEO (hash), frame-count wording."""
    import hashlib
    from PIL import Image
    dirs = sorted(d for d in Path(arm_root).iterdir() if (d / "meta.json").exists()); items = []
    for d in dirs:
        m = json.load(open(d / "meta.json")); h = int(hashlib.md5(m["video"].encode()).hexdigest(), 16) % 1000
        if (split == "test") != (h < test_frac * 1000): continue
        frames = [Image.open(d / f"{f['idx']:03d}.jpg").convert("RGB") for f in m["frames"]]
        facts = [f["idx"] for f in m["frames"] if f["label"] == 1]; pair = " ".join(m["pair"]) if isinstance(m["pair"], list) else str(m["pair"])
        q = f"How many frames show the action '{pair}'?"
        items.append((frames, q, m["true_count"], facts, m["qid"]))
    rng.shuffle(items); return items[:limit]


@torch.no_grad()
def frozen_count(rt, frames, q, resize, oracle_idx=None):
    """Frozen VLM baseline: canonical count prompt over all frames (or only oracle frames), greedy."""
    from gnnformer.data import build_count_prompt
    fr = frames if oracle_idx is None else [frames[i] for i in oracle_idx]
    if not fr: fr = frames[:1]
    content = [{"type": "image", "image": f.resize((resize, resize))} for f in fr] + [{"type": "text", "text": build_count_prompt(q, len(fr))}]
    inp = rt.processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    inp = {k: (v.to(rt.model.device) if hasattr(v, "to") else v) for k, v in inp.items()}
    out = rt.model.generate(**inp, max_new_tokens=4, do_sample=False)
    txt = rt.processor.tokenizer.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True); mm = re.search(r"\d+", txt)
    return int(mm.group()) if mm else None


def load_split(root, limit, rng):
    dirs = sorted(Path(root).iterdir()); rng.shuffle(dirs); out = []
    for d in dirs[:limit]:
        try: sid, frames, q, states, ans = load_mmred_sample(d)
        except Exception: continue
        out.append((frames, q, int(str(ans).strip()), facts_of(states, q), sid))
    return out


def run(a):
    rt = load_runtime(a.model); eng0_layers = get_layers(rt.model); nL = len(eng0_layers); dev = rt.model.device
    tm = rt.model.model.language_model; cid = rt.processor.tokenizer.convert_tokens_to_ids(CARRIER_TOKEN)
    with torch.no_grad(): e0 = tm.embed_tokens.weight[cid].float().clone()
    e_c = torch.nn.Parameter(e0.to(dev)); d_model = e0.numel()
    head_w = torch.nn.Parameter(torch.zeros(d_model, device=dev)); head_b = torch.nn.Parameter(torch.zeros(1, device=dev))
    eng = CarrierEngine(rt, l_open=nL, e_c=e_c); fenced = a.arm == "sum"
    odir = Path(a.output) / a.arm / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"; odir.mkdir(parents=True, exist_ok=True)
    (odir / "config.json").write_text(json.dumps(vars(a), indent=1)); log = open(odir / "log.txt", "a")
    def P(*x):
        s = " ".join(str(v) for v in x); print(s, flush=True); log.write(s + "\n"); log.flush()
    rng = random.Random(a.seed)
    if a.dataset == "herbench":
        train = herbench_items(f"data/herbench_ac/ev_fill{a.hb_n}", "train", a.limit_train, rng); dev_set = train[-30:]; train = train[:-30]
    else:
        train = load_split("data/mmred_vfiltered/seq_len_8/train", a.limit_train, rng) + load_split("data/mmred_vfiltered/seq_len_32/train", a.limit_train // 2, rng)
        dev_set = load_split("data/mmred_vfiltered/seq_len_8/dev", 40, rng)
    rng.shuffle(train); P(f"train {len(train)} dev {len(dev_set)} arm {a.arm} resize {a.resize} dataset {a.dataset}")
    opt = torch.optim.AdamW([e_c, head_w, head_b], lr=a.lr, weight_decay=0.0)
    def prep(item):
        frames, q, gold, facts, sid = item
        rec = eng.prepare_sample(frames, q, gold=gold, task="steps", resize=a.resize, qfirst=True, posreset=fenced)
        return rec
    def loss_of(item):
        rec = prep(item)
        if rec is None: return None
        st = forward_carriers(eng, rec, e_c, fenced, grad_ckpt=True).float(); z = st @ head_w + head_b
        y = torch.zeros(len(rec["cpos"]), device=dev); y[item[3]] = 1.0
        return F.binary_cross_entropy_with_logits(z, y)
    @torch.no_grad()
    def evaluate(items, tag):
        ex = vc = vn = 0; margins = []; skipped = 0; allz = []; ally = []
        for it in items:
            rec = prep(it)
            if rec is None: skipped += 1; continue
            z = (forward_carriers(eng, rec, e_c, fenced, grad_ckpt=False).float() @ head_w + head_b).cpu(); y = torch.zeros(len(z)); y[it[3]] = 1.0; allz.append(z); ally.append(y)
            ex += int(int((z > 0).sum()) == it[2]); vc += int(((z > 0).float() == y).sum()); vn += len(y)
            if y.sum() > 0 and (1 - y).sum() > 0: margins.append(float(z[y == 1].min() - z[y == 0].max()))
        auc = None
        if allz:
            zz = torch.cat(allz); yy = torch.cat(ally); pos = zz[yy == 1]; neg = zz[yy == 0]
            if len(pos) and len(neg): auc = float((pos[:, None] > neg[None, :]).float().mean() + 0.5 * (pos[:, None] == neg[None, :]).float().mean())
        return dict(tag=tag, n=len(items) - skipped, skipped=skipped, exact=ex / max(1, len(items) - skipped), verdict_acc=vc / max(1, vn), verdict_auc=auc, min_margin_med=(float(torch.tensor(margins).median()) if margins else None))
    t0 = time.time(); step = 0
    for ep in range(a.epochs):
        rng.shuffle(train)
        for i in range(0, len(train), a.batch):
            opt.zero_grad(); bl = 0.0; nb = 0
            for it in train[i:i + a.batch]:
                loss = loss_of(it)
                if loss is None: continue
                (loss / a.batch).backward(); bl += float(loss); nb += 1
            opt.step(); step += 1
            if step % 10 == 0: P(f"[{time.time()-t0:.0f}s] ep {ep} step {step} loss {bl/max(1,nb):.4f}")
        P(f"[{time.time()-t0:.0f}s] epoch {ep} | dev {evaluate(dev_set, f'dev ep{ep}')}")
    torch.save(dict(e_c=e_c.detach().cpu(), head_w=head_w.detach().cpu(), head_b=head_b.detach().cpu()), odir / "ckpt.pt")
    results = []
    if a.dataset == "herbench":
        for N in [int(x) for x in a.eval_ns.split("+")]:
            items = herbench_items(f"data/herbench_ac/ev_fill{N}", "test", a.limit_eval, random.Random(1))
            if not items: continue
            r = evaluate(items, f"herbench N={N}"); results.append(r); P(f"[{time.time()-t0:.0f}s] {r}")
            if a.frozen_baselines:
                fb = dict(tag=f"herbench N={N} frozen", n=len(items)); fu = [frozen_count(rt, it[0], it[1], a.resize) for it in items]; fo = [frozen_count(rt, it[0], it[1], a.resize, oracle_idx=it[3]) for it in items]
                fb["full_exact"] = sum(int(p_ == it[2]) for p_, it in zip(fu, items)) / len(items); fb["oracle_exact"] = sum(int(p_ == it[2]) for p_, it in zip(fo, items)) / len(items)
                fb["full_mae"] = sum(abs((p_ if p_ is not None else 0) - it[2]) for p_, it in zip(fu, items)) / len(items); results.append(fb); P(f"[{time.time()-t0:.0f}s] {fb}")
    else:
      for N in [int(x) for x in a.eval_ns.split("+")]:
        items = load_split(f"data/mmred_vfiltered/seq_len_{N}/test", a.limit_eval, random.Random(1)); r = evaluate(items, f"vision N={N}"); results.append(r); P(f"[{time.time()-t0:.0f}s] {r}")
    (odir / "summary.json").write_text(json.dumps(results, indent=1)); P("->", odir); return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct"); ap.add_argument("--arm", choices=["sum", "sum_nofence"], default="sum")
    ap.add_argument("--resize", type=int, default=FRAME_RESIZE); ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--limit-train", type=int, default=300); ap.add_argument("--limit-eval", type=int, default=100)
    ap.add_argument("--eval-ns", default="8+16+32+64"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--output", default="outputs/bindcount_vlm")
    ap.add_argument("--dataset", choices=["mmred", "herbench"], default="mmred"); ap.add_argument("--hb-n", type=int, default=16); ap.add_argument("--frozen-baselines", action="store_true")
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
