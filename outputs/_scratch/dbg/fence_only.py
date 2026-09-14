"""Control: fence + per-block position reset ONLY, frozen model answers the count itself at the tail (greedy digits), one forward
per decode step, no per-block question, no carriers/heads. --modality text (Qwen2.5-7B-Instruct, MMRED text, layout of train_bindcount
with cid=None) | vision (4-bit Qwen2.5-VL-7B, CarrierEngine layout with the frozen <|box_start|> marker per frame, lo mask).
Configs: fenced (fence all layers + reset) and open (plain causal, native positions = the frozen baseline on the same prompt)."""
import argparse, json, os, random, re, sys, time
from pathlib import Path
import torch
sys.path.insert(0, "scripts/condmask"); sys.path.insert(0, "."); sys.path.insert(0, "outputs/_scratch/dbg")


def run_text(a, odir, P):
    from modeling import load_text_model
    from train_bindcount import build_rec, forward, mmred_units
    dev = "cuda"; tok, model = load_text_model(a.model or "Qwen/Qwen2.5-7B-Instruct", device=dev); results = []; t0 = time.time()
    @torch.no_grad()
    def decode(rec, fenced):
        cur = dict(rec); toks = []
        for _ in range(4):
            st = forward(model, cur, None, dev, 10 ** 6 if fenced else None, fenced, [cur["seq"] - 1]); t = int(model.lm_head(st.to(model.dtype)).float()[0].argmax())
            if not tok.decode([t]).strip().isdigit(): break
            toks.append(t); cur = dict(cur, ids=cur["ids"] + [t], seq=cur["seq"] + 1)
        txt = tok.decode(toks).strip(); return int(txt) if txt.isdigit() else None
    for N in [int(x) for x in a.ns.split("+")]:
        items = mmred_units(f"data/mmred_filtered/seq_len_{N}/test", a.limit, random.Random(1))
        for fenced in (True, False):
            ex = mae = 0; preds = []
            for kind, units, q, gold, facts, *_ in items:
                rec = build_rec(tok, units, q, gold, facts, None, kind); p = decode(rec, fenced); preds.append(p)
                ex += int(p == gold); mae += abs((p if p is not None else 0) - gold)
            r = dict(tag=f"text N={N}", config="fenced+reset" if fenced else "open", n=len(items), exact=ex / len(items), mae=mae / len(items),
                     none=sum(p is None for p in preds), median_pred=sorted(p for p in preds if p is not None)[len([p for p in preds if p is not None]) // 2] if any(p is not None for p in preds) else None)
            results.append(r); P(f"[{time.time()-t0:.0f}s] {r}"); (odir / "results.json").write_text(json.dumps(results, indent=1))


def run_vision(a, odir, P):
    from gnnformer.runtime import load_runtime, get_layers
    from gnnformer.engine import CarrierEngine, SDPA_BACKENDS
    from gnnformer.carriers import make_masks
    from gnnformer.constants import MASK_MIN, CARRIER_TOKEN, FRAME_RESIZE
    from torch.nn.attention import sdpa_kernel
    from train_bindcount_vlm import load_split
    rt = load_runtime(a.model or "Qwen/Qwen2.5-VL-7B-Instruct"); nL = len(get_layers(rt.model)); dev = rt.model.device; tm = rt.model.model.language_model
    tok = rt.processor.tokenizer; cid = tok.convert_tokens_to_ids(CARRIER_TOKEN)
    with torch.no_grad(): e0 = tm.embed_tokens.weight[cid].float().clone()
    eng = CarrierEngine(rt, l_open=nL, e_c=torch.nn.Parameter(e0.to(dev))); results = []; t0 = time.time(); shown = False
    @torch.no_grad()
    def decode(rec, fenced, max_new=6):
        emb = rec["emb"].to(dev).unsqueeze(0); pos = rec["pos"].to(dev); toks = []
        for _ in range(max_new):
            S = emb.shape[1]
            if fenced: lo, _ = make_masks(S, rec["blocks"], rec["cpos"], rec["fin"]); m = lo.to(dev).to(torch.float32).view(1, 1, S, S)
            else: m = torch.full((S, S), MASK_MIN, device=dev).triu(1).view(1, 1, S, S)
            cos_, sin_ = tm.rotary_emb(emb, pos); pe = (cos_.to(emb.dtype), sin_.to(emb.dtype)); h = emb
            with sdpa_kernel(SDPA_BACKENDS):
                for ly in eng.layers: h = ly(h, attention_mask=m, position_embeddings=pe)[0]
            lm = rt.model.lm_head; t = int(lm(tm.norm(h)[0, -1].to(lm.weight.dtype)).float().argmax()); toks.append(t)
            if t == tok.eos_token_id or tok.decode(toks).count("\n") > 0: break
            emb = torch.cat([emb, tm.embed_tokens(torch.tensor([[t]], device=dev)).to(emb.dtype)], 1); pos = torch.cat([pos, pos[:, :, -1:] + 1], -1)
        txt = tok.decode(toks, skip_special_tokens=True); mm = re.search(r"\d+", txt); return (int(mm.group()) if mm else None), txt
    for N in [int(x) for x in a.ns.split("+")]:
        items = load_split(f"data/mmred_vfiltered/seq_len_{N}/test", a.limit, random.Random(1))
        for fenced in (True, False):
            ex = mae = skipped = 0; preds = []
            for frames, q, gold, facts, sid in items:
                rec = eng.prepare_sample(frames, q + " Answer with the number only.", gold=gold, task="steps", resize=a.resize, qfirst=True, posreset=fenced)
                if rec is None: skipped += 1; continue
                p, txt = decode(rec, fenced); preds.append(p); ex += int(p == gold); mae += abs((p if p is not None else 0) - gold)
                if not shown: P("example output:", repr(txt), "gold", gold); shown = True
            n = len(items) - skipped
            r = dict(tag=f"vision N={N}", config="fenced+reset" if fenced else "open", n=n, skipped=skipped, exact=ex / max(1, n), mae=mae / max(1, n), none=sum(p is None for p in preds))
            results.append(r); P(f"[{time.time()-t0:.0f}s] {r}"); (odir / "results.json").write_text(json.dumps(results, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["text", "vision"], required=True); ap.add_argument("--model", default=""); ap.add_argument("--ns", default="")
    ap.add_argument("--limit", type=int, default=100); ap.add_argument("--resize", type=int, default=392); ap.add_argument("--output", default="outputs/fence_only")
    a = ap.parse_args(); a.ns = a.ns or ("8+16+32+64+128+256+512+1024" if a.modality == "text" else "8+16+32+64")
    odir = Path(a.output) / a.modality / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"; odir.mkdir(parents=True, exist_ok=True)
    (odir / "config.json").write_text(json.dumps(vars(a), indent=1)); log = open(odir / "log.txt", "a")
    def P(*x):
        s = " ".join(str(v) for v in x); print(s, flush=True); log.write(s + "\n"); log.flush()
    (run_text if a.modality == "text" else run_vision)(a, odir, P); P("->", odir); return 0


if __name__ == "__main__":
    sys.exit(main())
