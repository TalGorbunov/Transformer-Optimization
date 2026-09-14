"""S1 trace anatomy (PREREG_AGG 2026-09-07: S1a, S1b). Qwen3-8B THINKING on needle stories (needles_types.story_typed,
kind HC = same-domain distractors; J = 0 is the clean regime).
S1a  is the running tally on the tape?  detect a written tally (numerals 1,2,3,... in order), corrupt one numeral by +1,
     regenerate from there: does the next written numeral follow the corruption (v+2) or repair it (v+1)? does the final
     answer shift by +1?
S1b  per-item retrieval: attention share from the first tokens of each per-item chunk (a mention of the queried person in
     the thinking text) onto the context sentence it copies, vs. all context sentences; by J and by outcome."""
import argparse, json, os, random, re, sys, time, bisect
from pathlib import Path
import torch
sys.path.insert(0, "scripts/condmask"); sys.path.insert(0, "."); sys.path.insert(0, "outputs/_scratch/dbg")
from needles import PROMPT  # noqa: E402
from needles_types import story_typed, load_pool  # noqa: E402
from needles_cot import POST, parse  # noqa: E402
from osq import rope  # noqa: E402
from modeling import model_parts  # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402

WORDS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
W2I = {w: i for i, w in enumerate(WORDS)}
NUM = re.compile(r"\b(\d{1,3}|" + "|".join(WORDS) + r")\b", re.I)


def numerals(text):
    return [((int(m.group(1)) if m.group(1).isdigit() else W2I[m.group(1).lower()]), m.start(), m.end(), m.group(1).isdigit()) for m in NUM.finditer(text)]


def tally_chain(text):
    """Longest chain of numeral occurrences with values 1, 2, 3, ... in text order (a written running tally or a numbered list)."""
    occ = numerals(text); best = []
    for i, o in enumerate(occ):
        if o[0] != 1: continue
        chain = [o]; want = 2
        for o2 in occ[i + 1:]:
            if o2[0] == want: chain.append(o2); want += 1
        if len(chain) > len(best): best = chain
    return best


def build_prompt_ids(tok, sents, q):
    user = PROMPT["instruction"] + "\n\n" + PROMPT["examples"] + "\n\n<context>\n@@CTX@@\n</context>\n\nQuestion: " + q + "\n" + POST["direct"]
    txt = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    pre, post = txt.split("@@CTX@@"); ids = tok(pre, add_special_tokens=False).input_ids; spans = []
    for j, s in enumerate(sents):
        seg = tok((" " if j else "") + s, add_special_tokens=False).input_ids; spans.append((len(ids), len(ids) + len(seg))); ids += seg
    ids += tok(post, add_special_tokens=False).input_ids; return ids, spans


@torch.no_grad()
def generate(model, tok, id_lists, max_new, dev):
    L = max(len(x) for x in id_lists); pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.full((len(id_lists), L), pad); am = torch.zeros((len(id_lists), L), dtype=torch.long)
    for i, x in enumerate(id_lists): ids[i, L - len(x):] = torch.tensor(x); am[i, L - len(x):] = 1
    gen = model.generate(input_ids=ids.to(dev), attention_mask=am.to(dev), max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=pad)
    outs = []
    for i in range(len(id_lists)):
        new = gen[i, L:].tolist(); cut = len(new)
        for j, t in enumerate(new):
            if t in (tok.eos_token_id, pad): cut = j; break
        outs.append(new[:cut])
    return outs


def think_part(text):
    return text.split("</think>")[0]


@torch.no_grad()
def row_attn(lay, h, pe, rows):
    """Attention weights of the given rows at this layer: [H, R, S] fp32 (handles Qwen3 q/k norms)."""
    hn = lay.input_layernorm(h); at = lay.self_attn; hd = pe[0].shape[-1]; S = h.shape[1]; nh = at.q_proj.weight.shape[0] // hd; nkv = at.k_proj.weight.shape[0] // hd
    q = at.q_proj(hn[:, rows]).view(1, len(rows), nh, hd); k = at.k_proj(hn).view(1, S, nkv, hd)
    if hasattr(at, "q_norm"): q = at.q_norm(q); k = at.k_norm(k)
    q = rope(q.transpose(1, 2), pe[0][:, rows], pe[1][:, rows])[0]; k = rope(k.transpose(1, 2), *pe)[0].repeat_interleave(nh // nkv, dim=0)
    sc = torch.einsum("htd,hsd->hts", q.float(), k.float()) / hd ** 0.5
    causal = torch.arange(S, device=h.device)[None, :] > torch.tensor(rows, device=h.device)[:, None]
    return torch.softmax(sc.masked_fill(causal[None], float("-inf")), -1)


@torch.no_grad()
def retrieval_shares(model, ids, spans, chunks, layers_set, dev):
    """chunks: list of (rows, target_span_ids). Returns per chunk: mean over layers/heads/rows of share on target, on all
    sentences, on needle sentences (target set given by caller)."""
    embed, layers, norm, head, rotary = model_parts(model); x = torch.tensor([ids], device=dev); pos = torch.arange(len(ids), device=dev)[None]
    h = embed(x); cos, sin = rotary(h, pos); pe = (cos.to(h.dtype), sin.to(h.dtype)); rows = [r for rs, _ in chunks for r in rs]
    S = len(ids); sent_of = torch.full((S,), -1, dtype=torch.long, device=dev)
    for j, (a, b) in enumerate(spans): sent_of[a:b] = j
    acc = [dict(target=0.0, ctx=0.0, n=0) for _ in chunks]
    for li, lay in enumerate(layers):
        if li in layers_set:
            w = row_attn(lay, h, pe, rows).mean(0)                                            # [R, S] head-mean
            per_sent = torch.zeros(len(rows), len(spans), device=dev).index_add_(1, sent_of.clamp(min=0), w * (sent_of >= 0).float()[None])
            k0 = 0
            for ci, (rs, tgt) in enumerate(chunks):
                ps = per_sent[k0:k0 + len(rs)]; k0 += len(rs); acc[ci]["ctx"] += float(ps.sum(-1).mean()); acc[ci]["n"] += 1
                if tgt: acc[ci]["target"] += float(ps[:, tgt].sum(-1).mean())
        o = lay(h, attention_mask=None, position_ids=pos, position_embeddings=pe); h = o[0] if isinstance(o, tuple) else o
    return [dict(target=a_["target"] / max(1, a_["n"]), ctx=a_["ctx"] / max(1, a_["n"])) for a_ in acc]


def find_chunks(tok, gen_ids, think_text, person, sents, n_rows=8):
    """Per-item chunks = mentions of the queried person in the thinking text; target = context sentence(s) whose words all
    appear in the 48 chars after the mention (identical sentences share the target)."""
    pieces = [tok.decode([t]) for t in gen_ids]; starts = []; c = 0
    for p_ in pieces: starts.append(c); c += len(p_)
    words = [set(w.strip(".,").lower() for w in s.split()) for s in sents]; chunks = []
    for m in re.finditer(r"\b" + re.escape(person) + r"\b", think_text):
        if m.start() < 2: continue
        t0 = bisect.bisect_left(starts, m.start()); rows = list(range(t0, min(t0 + n_rows, len(gen_ids))))
        if len(rows) < 3: continue
        win = set(w.strip(".,").lower() for w in think_text[m.start():m.start() + 48].split())
        full = [j for j, ws in enumerate(words) if ws <= win]
        if full:
            best = max(full, key=lambda j: len(words[j])); tgt = [j for j in full if words[j] == words[best]]
        else: tgt = []
        chunks.append((rows, tgt))
    return chunks


def run(a):
    dev = "cuda"; tok = AutoTokenizer.from_pretrained(a.model); model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    pool = load_pool(); nL = len(model_parts(model)[1]); layers_set = set(range(3, nL, 4))
    odir = Path(a.output) / a.model.split("/")[-1] / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"; odir.mkdir(parents=True, exist_ok=True)
    (odir / "config.json").write_text(json.dumps(dict(vars(a), n_layers=nL, layers=sorted(layers_set)), indent=1)); log = open(odir / "log.txt", "a"); fout = open(odir / "rows.jsonl", "w")
    def P(*x):
        s = " ".join(str(v) for v in x); print(s, flush=True); log.write(s + "\n"); log.flush()
    t0 = time.time(); rows_all = []
    for J in [int(x) for x in a.js.split("+")]:
        for K in [int(x) for x in a.ks.split("+")]:
            rng = random.Random(a.seed * 1000 + K * 10 + J); stories = [story_typed(K, J, "HC", rng, pool) for _ in range(a.n)]
            prompts = [build_prompt_ids(tok, s, q) for s, _, _, q in stories]
            gens = generate(model, tok, [p for p, _ in prompts], a.max_new, dev)
            cell = []
            for i, ((sents, facts, classes, q), (pids, spans), g) in enumerate(zip(stories, prompts, gens)):
                text = tok.decode(g, skip_special_tokens=False); tp = think_part(text); pred = parse(text); trunc = len(g) >= a.max_new and "</think>" not in text
                chain = tally_chain(tp); person = q.split("did ")[1].split(" go")[0]
                row = dict(J=J, K=K, i=i, pred=pred, ok=pred == K, truncated=trunc, new_tokens=len(g), chain_len=len(chain), chain_digits=all(o[3] for o in chain) if chain else None,
                           chain_end=(chain[-1][0] if chain else None), tally_written=len(chain) >= 3, text_tail=text[-300:])
                # ---- S1a: corrupt the middle tally numeral by +1 and regenerate
                if len(chain) >= 3:
                    v, ca, cb, isd = chain[len(chain) // 2]; new_num = str(v + 1) if isd else WORDS[v + 1]
                    prefix = tp[:ca] + new_num; pref_ids = tok(prefix, add_special_tokens=False).input_ids
                    cont = generate(model, tok, [pids + pref_ids], a.max_new_cont, dev)[0]; ctext = tok.decode(cont, skip_special_tokens=False)
                    nxt = next((o[0] for o in numerals(ctext) if o[0] in (v, v + 1, v + 2, v + 3)), None)
                    pc = parse(prefix + ctext)
                    row["corrupt"] = dict(at_value=v, next_numeral=nxt, follows=(nxt == v + 2), repairs=(nxt == v + 1), pred_corrupt=pc,
                                          shift=(None if (pc is None or pred is None) else pc - pred), cont_tail=ctext[-200:])
                # ---- S1b: retrieval shares per chunk (teacher-forced forward over prompt + trace)
                chunks = find_chunks(tok, g, tp, person, sents)
                if chunks:
                    off = len(pids); ch = [([off + r for r in rs], tgt) for rs, tgt in chunks]
                    try:
                        sh = retrieval_shares(model, pids + g, spans, ch, layers_set, dev)
                        matched = [(s_, tgt) for s_, (_, tgt) in zip(sh, ch) if tgt]
                        row["s1b"] = dict(n_chunks=len(ch), n_matched=len(matched), target_share=(sum(s_["target"] for s_, _ in matched) / len(matched) if matched else None),
                                          ctx_share=(sum(s_["ctx"] for s_ in sh) / len(sh)), target_ratio=(sum(s_["target"] / max(1e-6, s_["ctx"]) for s_, _ in matched) / len(matched) if matched else None),
                                          on_needle=(sum(int(any(j in set(facts) for j in tgt)) for _, tgt in matched) / len(matched) if matched else None))
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache(); row["s1b"] = dict(oom=True)
                fout.write(json.dumps(row) + "\n"); fout.flush(); cell.append(row); rows_all.append(row)
            acc = sum(r["ok"] for r in cell) / len(cell); tw = sum(r["tally_written"] for r in cell) / len(cell)
            fol = [r["corrupt"]["follows"] for r in cell if "corrupt" in r]; sh = [r["s1b"]["target_ratio"] for r in cell if r.get("s1b", {}).get("target_ratio") is not None]
            P(f"[{time.time()-t0:.0f}s] J={J} K={K:2d} acc {acc:.2f} trunc {sum(r['truncated'] for r in cell)} tokens {sum(r['new_tokens'] for r in cell)/len(cell):.0f} | tally written {tw:.2f} "
              f"follow {sum(fol)/max(1,len(fol)):.2f} (n={len(fol)}) | target ratio med {sorted(sh)[len(sh)//2] if sh else float('nan'):.3f} (n={len(sh)})")
    fout.close(); summarize(odir, rows_all); P("->", odir); return 0


def summarize(odir, rows):
    import statistics
    out = [f"# S1 trace anatomy — {odir}"]
    js = sorted({r["J"] for r in rows}); ks = sorted({r["K"] for r in rows})
    out.append("\n## S1a — written tally and corruption test")
    for J in js:
        sub = [r for r in rows if r["J"] == J]; cor = [r["corrupt"] for r in sub if "corrupt" in r]
        sh = [c["shift"] for c in cor if c["shift"] is not None]
        out.append(f"J={J}: acc {sum(r['ok'] for r in sub)/len(sub):.2f}, tally written {sum(r['tally_written'] for r in sub)/len(sub):.2f}, "
                   f"corruptions {len(cor)}: follow {sum(c['follows'] for c in cor)/max(1,len(cor)):.2f}, repair {sum(c['repairs'] for c in cor)/max(1,len(cor)):.2f}, "
                   f"answer shift +1 {sum(s == 1 for s in sh)/max(1,len(sh)):.2f}, shift 0 {sum(s == 0 for s in sh)/max(1,len(sh)):.2f} (n={len(sh)})")
    out.append("\n## S1b — retrieval share of the chunk onset on its target sentence (fraction of context attention)")
    for J in js:
        for K in ks:
            sub = [r["s1b"] for r in rows if r["J"] == J and r["K"] == K and r.get("s1b", {}).get("target_ratio") is not None]
            if sub: out.append(f"J={J} K={K:2d}: median target ratio {statistics.median(s['target_ratio'] for s in sub):.3f}, target share {statistics.median(s['target_share'] for s in sub):.3f}, on-needle {statistics.mean(s['on_needle'] for s in sub):.2f}, chunks/trace {statistics.mean(s['n_chunks'] for s in sub):.1f} (n={len(sub)})")
    # AUROC of per-trace target ratio for failure within J = max
    Jm = max(js); sub = [r for r in rows if r["J"] == Jm and r.get("s1b", {}).get("target_ratio") is not None]
    fail = [not r["ok"] or r["truncated"] for r in sub]; x = [r["s1b"]["target_ratio"] for r in sub]
    if any(fail) and not all(fail):
        pos = [v for v, f in zip(x, fail) if f]; neg = [v for v, f in zip(x, fail) if not f]      # low share should predict failure
        auc = sum((p < n_) + 0.5 * (p == n_) for p in pos for n_ in neg) / (len(pos) * len(neg))
        out.append(f"\nJ={Jm}: AUROC(low target ratio -> failure) = {auc:.3f} (fail {len(pos)}, ok {len(neg)})")
    txt = "\n".join(out); print(txt); (odir / "ANALYSIS.md").write_text(txt + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B"); ap.add_argument("--ks", default="4+8+16"); ap.add_argument("--js", default="0+64")
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max-new", type=int, default=3072); ap.add_argument("--max-new-cont", type=int, default=1536)
    ap.add_argument("--output", default="outputs/cot_anatomy"); ap.add_argument("--analyze", default="")
    a = ap.parse_args()
    if a.analyze: rows = [json.loads(l) for l in open(Path(a.analyze) / "rows.jsonl") if l.strip()]; summarize(Path(a.analyze), rows); return 0
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
