"""Quick test: does putting tokens BETWEEN the context and the answer repair many-needle counting?
Same stories as needles.py (K needles that all match the count question; H0 = no distractors,
HF = 64 distractors). Arms:
  direct      answer with the number only (the needles.py protocol)              [any model]
  enumerate   first list each matching sentence (numbered), then 'Answer: N'    [any model]
  think_off   Qwen3 hybrid model, enable_thinking=False, number only            [Qwen3]
  think_on    Qwen3 hybrid model, enable_thinking=True (free <think>), then N    [Qwen3]
Batched greedy generation (left padding). Records the emitted number, the length of the
intermediate text and how many times the queried person is named in it (a proxy for
'one retrieval per needle')."""
import argparse, json, os, re, sys, time, random
from pathlib import Path
import torch
sys.path.insert(0, "scripts/condmask"); sys.path.insert(0, ".")
from needles import story, PROMPT, REGIMES  # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402

POST = {
 "direct": "Your answer should contain only the number, written in digits. Do not write anything else after that.",
 "enumerate": "First, list every sentence from the text in which the asked person goes to the asked room, one per line, numbered 1., 2., 3., ... Then, on a new line, write 'Answer: ' followed by the number of listed sentences, in digits.",
}

def user_msg(sents, q, post):
    return (PROMPT["instruction"] + "\n\n" + PROMPT["examples"] + "\n\n<context>\n" + " ".join(sents)
            + "\n</context>\n\nQuestion: " + q + "\n" + post)

def parse(text):
    tail = text.split("</think>")[-1]
    m = re.findall(r"Answer:\s*\**\s*(\d+)", tail)
    if m: return int(m[-1])
    m = re.findall(r"\d+", tail)
    return int(m[-1]) if m else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--arms", required=True)
    ap.add_argument("--ks", default="2+4+8+16+32"); ap.add_argument("--regimes", default="H0+HF")
    ap.add_argument("--n", type=int, default=16); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-think", type=int, default=1536); ap.add_argument("--output", default="outputs/_scratch/needles_cot")
    a = ap.parse_args()
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    odir = Path(a.output) / a.model.split("/")[-1] / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    odir.mkdir(parents=True, exist_ok=True); fout = open(odir / "results.jsonl", "w")
    (odir / "config.json").write_text(json.dumps(vars(a), indent=1))
    t0 = time.time()
    for arm in a.arms.split("+"):
        post = POST["direct" if arm in ("direct", "think_off", "think_on") else "enumerate"]
        for reg in a.regimes.split("+"):
            for K in [int(k) for k in a.ks.split("+")]:
                D = REGIMES[reg](K)
                rng = random.Random(a.seed * 1000 + K)           # same stories across arms/models
                items = [story(K, D, rng) for _ in range(a.n)]
                kw = {}
                if arm in ("think_on", "think_off"): kw["enable_thinking"] = arm == "think_on"
                prompts = [tok.apply_chat_template([{"role": "user", "content": user_msg(s, q, post)}],
                                                   tokenize=False, add_generation_prompt=True, **kw) for s, _, q in items]
                enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
                max_new = 12 if arm in ("direct", "think_off") else (a.max_think if arm == "think_on" else 40 + 16 * K)
                with torch.no_grad():
                    gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
                cell = []
                for j, (s, facts, q) in enumerate(items):
                    new = gen[j, enc.input_ids.shape[1]:]
                    text = tok.decode(new, skip_special_tokens=True)
                    ntok = int((new != (tok.pad_token_id if tok.pad_token_id is not None else -1)).sum())
                    pred = parse(text); person = q.split("did ")[1].split(" go")[0]
                    mid = text.split("Answer:")[0] if "Answer:" in text else text.split("</think>")[0]
                    row = dict(arm=arm, regime=reg, K=K, D=D, i=j, pred=pred, ok=pred == K,
                               err=None if pred is None else abs(pred - K), new_tokens=ntok,
                               truncated=ntok >= max_new, person_mentions=mid.count(person),
                               lines=mid.count("\n"), out=text[-4000:])
                    fout.write(json.dumps(row) + "\n"); cell.append(row)
                acc = sum(r["ok"] for r in cell) / len(cell)
                err = [r["err"] for r in cell if r["err"] is not None]
                print(f"[{time.time()-t0:.0f}s] {arm:9s} {reg} K={K:2d} acc {acc:.2f} err {sum(err)/max(1,len(err)):.1f} "
                      f"new_tok {sum(r['new_tokens'] for r in cell)/len(cell):.0f} trunc {sum(r['truncated'] for r in cell)} "
                      f"mentions {sum(r['person_mentions'] for r in cell)/len(cell):.1f} | e.g. {cell[0]['out'][-60:]!r}", flush=True)
    fout.close(); print("->", odir)

if __name__ == "__main__":
    main()
