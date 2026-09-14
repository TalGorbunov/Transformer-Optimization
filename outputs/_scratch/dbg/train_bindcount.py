"""Bind, then count — per-unit carriers behind a block-diagonal fence (per-block position reset), a
per-carrier verdict head (linear on the carrier's final state), and a SUM read (count = #verdicts > 0).
Arms: sum (fence all layers), sum_nofence (carriers, full attention), mean_lora (fence below L*, open
above; digit read at the answer row through LoRA r8 on layers >= L*). Frozen backbone (bf16).
Data: MMRED filtered text (train N<=32 from data/mmred_filtered_train; test N 8..1024 from
data/mmred_filtered/<N>/test) and needle stories (train N=32 sentences; test N in {64,128,256})."""
import argparse, ast, glob, json, os, random, re, sys, time
from pathlib import Path
import torch, torch.nn.functional as F
sys.path.insert(0, "scripts/condmask"); sys.path.insert(0, "."); sys.path.insert(0, "outputs/_scratch/dbg")
from modeling import load_text_model, model_parts, SDPA_BACKENDS  # noqa: E402
from torch.nn.attention import sdpa_kernel  # noqa: E402
from gnnformer.constants import MASK_MIN  # noqa: E402
from gnnformer.carriers import attach_lora  # noqa: E402
from textdata import load_text_sample, render_frame_line  # noqa: E402
from needles import PEOPLE, ROOMS  # noqa: E402
from needles_types import story_typed, load_pool  # noqa: E402

CARRIER_TOKEN = "<|fim_pad|>"


# ------------------------------------------------------------------ prompt with carriers
def build_rec(tok, units, question, gold, facts, cid, kind):
    """cid=None -> plain prompt without carrier tokens (digit_lora baseline)."""
    noun = "frames" if kind == "mmred" else "sentences"
    prefix = f"You will be shown {len(units)} {noun}.\nQuestion: {question}"
    ids = list(tok(prefix, add_special_tokens=False).input_ids); pe = len(ids); blocks, carriers = [], []
    for u in units:
        seg = tok("\n" + u, add_special_tokens=False).input_ids + ([cid] if cid is not None else [])
        blocks.append((len(ids), len(ids) + len(seg)))
        if cid is not None: carriers.append(len(ids) + len(seg) - 1)
        ids.extend(seg)
    fin = len(ids); ids.extend(tok(f"\nQuestion: {question}\nAnswer: ", add_special_tokens=False).input_ids)
    return dict(ids=ids, prefix_end=pe, blocks=blocks, carriers=carriers, fin=fin, seq=len(ids), gold=gold, facts=facts,
                is_fact=[int(j in set(facts)) for j in range(len(units))], n=len(units))


def positions(rec, reset):
    S = rec["seq"]; p = list(range(S))
    if reset:
        pe = rec["prefix_end"]; L = max(b - a for a, b in rec["blocks"])
        for a, b in rec["blocks"]:
            for t in range(a, b): p[t] = pe + (t - a)
        for t in range(rec["fin"], S): p[t] = pe + L + (t - rec["fin"])
    return torch.tensor([p])


def fence_mask(rec, dev):
    """[1,1,S,S] additive: causal base; frame rows cannot see other blocks (prefix always visible)."""
    S = rec["seq"]; pos = torch.arange(S, device=dev); q, k = pos.view(-1, 1), pos.view(1, -1)
    base = torch.where(k <= q, 0.0, MASK_MIN)
    blk = torch.full((S,), -1, dtype=torch.long, device=dev)
    for i, (a, b) in enumerate(rec["blocks"]): blk[a:b] = i
    cross = (blk.view(-1, 1) >= 0) & (blk.view(1, -1) >= 0) & (blk.view(-1, 1) != blk.view(1, -1))
    return (base + torch.where(cross, MASK_MIN, 0.0)).clamp(min=MASK_MIN)[None, None], base[None, None]


# ------------------------------------------------------------------ forward
def forward(model, rec, e_c, dev, fence_until, reset, need_rows):
    """Manual stack; e_c overrides the carrier token embeddings. fence_until: layers < it are fenced
    (None = no fence). Returns final normed states at `need_rows` [R, d]."""
    embed, layers, norm, head, rotary = model_parts(model)
    ids = torch.tensor([rec["ids"]], device=dev); h = embed(ids)
    if e_c is not None and rec["carriers"]:
        car = torch.tensor(rec["carriers"], device=dev); h = h.clone(); h[0, car] = e_c.to(h.dtype)
    pos = positions(rec, reset).to(dev); cos, sin = rotary(h, pos); pe = (cos.to(h.dtype), sin.to(h.dtype))
    fenced, open_ = fence_mask(rec, dev); fenced, open_ = fenced.to(h.dtype), open_.to(h.dtype)
    for li, lay in enumerate(layers):
        m = fenced if (fence_until is not None and li < fence_until) else open_
        with sdpa_kernel(SDPA_BACKENDS):
            out = lay(h, attention_mask=m, position_ids=pos, position_embeddings=pe)
        h = out[0] if isinstance(out, tuple) else out
    return norm(h[0, torch.tensor(need_rows, device=dev)])


# ------------------------------------------------------------------ data
def mmred_units(root, limit, rng):
    out = []
    dirs = sorted(Path(root).iterdir()); rng.shuffle(dirs)
    for d in dirs[:limit]:
        s = load_text_sample(d)
        if s is None: continue
        sid, states, q, gold = s
        who, room = re.fullmatch(r"How many frames show (\w+) in the (\w+)\?", q).groups()
        facts = [j for j, st in enumerate(states) if who in ast.literal_eval(st)["rooms"].get(room, [])]
        rooms = []
        for st in states:
            d_ = ast.literal_eval(st)["rooms"]; r_ = next((r for r, cs in d_.items() if who in cs), "absent"); rooms.append(r_)
        out.append((("mmred"), [render_frame_line(st) for st in states], q, gold, facts, rooms))
    return out


def needle_units(n, N, kmax, rng, pool, kinds=("HC", "HN", "HP", "HR")):
    out = []
    for _ in range(n):
        K = rng.randint(0, kmax); kind = rng.choice(kinds)
        if K == 0:
            sents, facts, classes, q = story_typed(1, N, kind, rng, pool)   # N+1 units; drop the needle -> N units, K=0
            j = facts[0]; sents = [s_ for i_, s_ in enumerate(sents) if i_ != j]; facts = []
        else:
            sents, facts, classes, q = story_typed(K, N - K, kind, rng, pool)
        out.append(("needles", sents, q, len(facts), facts, None))
    return out


# ------------------------------------------------------------------ train / eval
def run(a):
    dev = "cuda"; tok, model = load_text_model(a.model, device=dev); d_model = model.config.hidden_size
    cid = tok.convert_tokens_to_ids(CARRIER_TOKEN)
    if cid is None or cid == tok.unk_token_id or not isinstance(cid, int):      # Llama: no <|fim_pad|>; use a reserved special token
        cid = tok.convert_tokens_to_ids("<|reserved_special_token_10|>")
    assert isinstance(cid, int) and cid >= 0, f"no carrier token id for {a.model}"
    rng = random.Random(a.seed); pool = load_pool()
    odir = Path(a.output) / a.arm / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"; odir.mkdir(parents=True, exist_ok=True)
    (odir / "config.json").write_text(json.dumps(vars(a), indent=1)); log = open(odir / "log.txt", "a")
    def P(*x):
        s = " ".join(str(v) for v in x); print(s, flush=True); log.write(s + "\n"); log.flush()
    # data
    train = []
    for N in (8, 16, 32): train += mmred_units(f"data/mmred_filtered_train/seq_len_{N}/train", a.limit_train, rng)
    train += needle_units(a.limit_train, 32, 16, rng, pool)
    dev_set = mmred_units("data/mmred_filtered_train/seq_len_16/dev", 60, rng) + needle_units(40, 32, 16, rng, pool)
    rng.shuffle(train); P(f"train {len(train)} dev {len(dev_set)} arm {a.arm}")
    # params
    with torch.no_grad(): e0 = model.model.embed_tokens.weight[cid].float().clone()
    e_c = torch.nn.Parameter(e0.clone().to(dev)); head_w = torch.nn.Parameter(torch.zeros(d_model, device=dev)); head_b = torch.nn.Parameter(torch.zeros(1, device=dev))
    params = [e_c, head_w, head_b]; lora = None
    fence_until = {"sum": 10 ** 6, "sum_multi": 10 ** 6, "sum_noreset": 10 ** 6, "sum_nofence": None, "mean_lora": a.lstar, "digit_lora": None}[a.arm]; reset = a.arm in ("sum", "sum_multi", "mean_lora")
    ROOMS7 = ["Kitchen", "Bathroom", "Garden", "Office", "Bedroom", "Park", "absent"]
    room_w = torch.nn.Parameter(torch.zeros(d_model, 7, device=dev)); room_b = torch.nn.Parameter(torch.zeros(7, device=dev))
    use_carriers = a.arm != "digit_lora"
    if not use_carriers: cid = None
    groups = [dict(params=[e_c, head_w, head_b] + ([room_w, room_b] if a.arm == "sum_multi" else []), lr=a.lr)]
    if a.arm in ("mean_lora", "digit_lora"):
        lora = attach_lora(model.model.layers, a.lstar, rank=8, alpha=16, device=dev); groups.append(dict(params=lora.parameters(), lr=a.lr_lora))
    digit_ids = torch.tensor([tok(str(d), add_special_tokens=False).input_ids[0] for d in range(10)], device=dev)
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    def loss_of(item):
        kind, units, q, gold, facts = item[:5]; rooms = item[5] if len(item) > 5 else None; rec = build_rec(tok, units, q, gold, facts, cid, kind)
        if a.arm in ("mean_lora", "digit_lora"):
            st = forward(model, rec, e_c if use_carriers else None, dev, fence_until, reset, [rec["seq"] - 1]); lg = model.lm_head(st.to(model.dtype)).float()[0]
            tgt = tok(str(gold), add_special_tokens=False).input_ids[0]; return F.cross_entropy(lg[None], torch.tensor([tgt], device=dev)), None
        st = forward(model, rec, e_c, dev, fence_until, reset, rec["carriers"]).float()
        z = st @ head_w + head_b; y = torch.tensor(rec["is_fact"], device=dev, dtype=torch.float)
        loss = F.binary_cross_entropy_with_logits(z, y) + a.lam * (torch.sigmoid(z).sum() - gold) ** 2 / max(1, len(units))
        if a.arm == "sum_multi" and rooms is not None:
            loss = loss + F.cross_entropy(st @ room_w + room_b, torch.tensor([ROOMS7.index(r_) for r_ in rooms], device=dev))
        return loss, z
    @torch.no_grad()
    def predict(item):
        kind, units, q, gold, facts = item[:5]; rooms = item[5] if len(item) > 5 else None; rec = build_rec(tok, units, q, gold, facts, cid, kind)
        if a.arm in ("mean_lora", "digit_lora"):
            toks = []; cur = dict(rec)
            for _ in range(4):
                st = forward(model, cur, e_c if use_carriers else None, dev, fence_until, reset, [cur["seq"] - 1]); t = int(model.lm_head(st.to(model.dtype)).float()[0].argmax())
                if not tok.decode([t]).strip().isdigit(): break
                toks.append(t); cur = dict(cur, ids=cur["ids"] + [t], seq=cur["seq"] + 1)
            txt = tok.decode(toks).strip(); return (int(txt) if txt.isdigit() else None), None
        st = forward(model, rec, e_c, dev, fence_until, reset, rec["carriers"]).float(); z = st @ head_w + head_b
        extra = (z.cpu(), rec["is_fact"]) if a.arm != "sum_multi" else (z.cpu(), rec["is_fact"], (st @ room_w + room_b).argmax(-1).cpu())
        return int((z > 0).sum()), extra
    def evaluate(items, tag):
        ex = 0; vc = 0; vn = 0; margins = []; sf = {k: [0, 0] for k in ("exists", "first", "last", "distinct_rooms", "most_room", "room_acc")}
        for it in items:
            pred, extra = predict(it); ex += int(pred == it[3])
            if extra is not None:
                z, y = extra[0], extra[1]; y = torch.tensor(y); vc += int(((z > 0).long() == y).sum()); vn += len(y)
                if y.sum() > 0 and (1 - y).sum() > 0: margins.append(float(z[y == 1].min() - z[y == 0].max()))
                if a.arm == "sum_multi" and len(it) > 5 and it[5] is not None:
                    yes = (z > 0).nonzero().flatten().tolist(); gy = y.nonzero().flatten().tolist(); rp = extra[2].tolist(); gr = [ROOMS7.index(r_) for r_ in it[5]]
                    sf["exists"][0] += int((len(yes) > 0) == (len(gy) > 0)); sf["exists"][1] += 1
                    if gy: sf["first"][0] += int(bool(yes) and yes[0] == gy[0]); sf["first"][1] += 1; sf["last"][0] += int(bool(yes) and yes[-1] == gy[-1]); sf["last"][1] += 1
                    pres_p = {r_ for r_ in rp if r_ != 6}; pres_g = {r_ for r_ in gr if r_ != 6}
                    sf["distinct_rooms"][0] += int(len(pres_p) == len(pres_g)); sf["distinct_rooms"][1] += 1
                    if pres_g:
                        mp = max(pres_p, key=lambda r_: rp.count(r_)) if pres_p else -1; mg = max(pres_g, key=lambda r_: gr.count(r_)); sf["most_room"][0] += int(mp == mg); sf["most_room"][1] += 1
                    sf["room_acc"][0] += sum(int(p_ == g_) for p_, g_ in zip(rp, gr)); sf["room_acc"][1] += len(gr)
        out = dict(tag=tag, n=len(items), exact=ex / max(1, len(items)), verdict_acc=(vc / vn if vn else None), min_margin_med=(float(torch.tensor(margins).median()) if margins else None))
        if a.arm == "sum_multi": out.update({k: (round(v[0] / v[1], 3) if v[1] else None) for k, v in sf.items()})
        return out
    # train
    t0 = time.time(); step = 0
    for ep in range(a.epochs):
        rng.shuffle(train); tot = 0.0
        for i in range(0, len(train), a.batch):
            opt.zero_grad(); bl = 0.0
            for it in train[i:i + a.batch]:
                loss, _ = loss_of(it); (loss / a.batch).backward(); bl += float(loss) / a.batch
            opt.step(); tot += bl; step += 1
            if step % 20 == 0: P(f"[{time.time()-t0:.0f}s] ep {ep} step {step} loss {bl:.4f}")
        r = evaluate(dev_set, f"dev ep{ep}"); P(f"[{time.time()-t0:.0f}s] epoch {ep} mean loss {tot/max(1,len(train)//a.batch):.4f} | dev {r}")
    torch.save(dict(e_c=e_c.detach().cpu(), head_w=head_w.detach().cpu(), head_b=head_b.detach().cpu(), lora=(None if lora is None else {k: (v[0].detach().cpu(), v[1].detach().cpu()) for k, v in lora.params.items()} if hasattr(lora, "params") else None)), odir / "ckpt.pt")
    # eval: MMRED test by N, needles by N
    results = []
    for N in [int(x) for x in a.eval_ns.split("+")]:
        root = f"data/mmred_filtered/seq_len_{N}/test"
        if Path(root).exists():
            items = mmred_units(root, a.limit_eval, random.Random(1)); r = evaluate(items, f"mmred N={N}"); results.append(r); P(f"[{time.time()-t0:.0f}s] {r}")
    for N in (64, 128, 256):
        items = needle_units(a.limit_eval, N, min(64, N // 2), random.Random(2), pool, kinds=("HC",)); r = evaluate(items, f"needles N={N} (K<={min(64, N//2)})"); results.append(r); P(f"[{time.time()-t0:.0f}s] {r}")
    (odir / "summary.json").write_text(json.dumps(results, indent=1)); P("->", odir); return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct"); ap.add_argument("--arm", choices=["sum", "sum_nofence", "sum_noreset", "mean_lora", "digit_lora", "sum_multi"], default="sum"); ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--lstar", type=int, default=12); ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--lam", type=float, default=0.05); ap.add_argument("--limit-train", type=int, default=300)
    ap.add_argument("--limit-eval", type=int, default=100); ap.add_argument("--eval-ns", default="8+16+32+64+128+256+512+1024"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="outputs/bindcount"); a = ap.parse_args(); return run(a)


if __name__ == "__main__":
    sys.exit(main())
