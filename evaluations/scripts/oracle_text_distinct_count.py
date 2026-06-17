#!/usr/bin/env python3
"""ORACLE / headroom test: can the frozen 7B count DISTINCT rooms from a CLEAN SYMBOLIC sequence (text),
with no vision? This isolates the dedup/count COMPUTATION from the vision-extraction problem.

For rooms_visited samples we give the per-frame room sequence as text and ask for the distinct count.
For co_occupancy we give a per-frame same-room yes/no sequence and ask how many 'yes'.
If the model scores high here, the set-cardinality computation is NOT the bottleneck (it can do it given
clean symbols) → the limit is reading room identity from frames. If it scores low, dedup-count itself is hard.
"""
from __future__ import annotations
import argparse, ast, re, sys, random
from pathlib import Path
from typing import Any, Dict, List
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import torch
from evaluations.scripts import eval_mmred_qwen25_vl_accuracy as base
from evaluations.helpers import utils as eval_utils


def states_of(qa_path: Path):
    lines = qa_path.read_text(encoding="utf-8").splitlines()
    qi = next(i for i,l in enumerate(lines) if l.strip()=="question:")
    ai = next(i for i,l in enumerate(lines) if l.strip()=="answer:")
    return [ast.literal_eval(l.strip()) for l in lines[qi+1:ai] if l.strip().startswith("{")]


def char_room_at(states, t, char):
    for room, occ in eval_utils.rooms_to_room2chars(states[t].get("rooms", {})).items():
        if char in occ:
            return room
    return None


@torch.inference_mode()
def ask(model, processor, prompt, device, max_new=8):
    messages=[{"role":"user","content":[{"type":"text","text":prompt}]}]
    inputs=processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors="pt")
    inputs=base.move_inputs_to_device(dict(inputs),device)
    plen=int(inputs["input_ids"].shape[-1])
    pad=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    out=model.generate(**inputs,do_sample=False,max_new_tokens=max_new,pad_token_id=pad)
    dec=processor.batch_decode(out[:,plen:],skip_special_tokens=True)[0]
    m=re.search(r"-?\d+",str(dec)); return int(m.group(0)) if m else None


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--task",default="rooms_visited",choices=["rooms_visited","co_occupancy"])
    ap.add_argument("--model-name",default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--data-root",type=Path,default=PROJECT_ROOT/"data"/"mmred_images_park"/"seq_len_8"/"all_uniform")
    ap.add_argument("--max-samples",type=int,default=120)
    ap.add_argument("--load-in-4bit",action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--output",type=Path,default=PROJECT_ROOT/"outputs"/"oracle_text_distinct_count")
    args=ap.parse_args()
    rng=random.Random(args.seed)
    dev=base.resolve_device("cuda"); dt=base.resolve_dtype("bfloat16",dev)
    args.output.mkdir(parents=True,exist_ok=True)
    model,proc=base.load_model_and_processor(args.model_name,dev,dt,bool(args.load_in_4bit))
    dirs=[d for d in sorted(args.data_root.iterdir()) if (d/"qa.txt").is_file()]
    rng.shuffle(dirs); dirs=dirs[:args.max_samples]
    correct=n=0; by_gold={}
    for d in dirs:
        try: states=states_of(d/"qa.txt")
        except Exception: continue
        chars=sorted(eval_utils.extract_characters_from_states(states))
        if not chars: continue
        if args.task=="rooms_visited":
            char=max(chars,key=lambda c:(sum(1 for t in range(len(states)) if char_room_at(states,t,c)),c))
            seq=[char_room_at(states,t,char) for t in range(len(states))]
            seq=[r for r in seq if r]
            if len(seq)<2: continue
            gold=len(set(seq))
            steps="; ".join(f"step {i+1}: {r}" for i,r in enumerate(seq))
            prompt=(f"A person moved through a house. The room they were in at each step was:\n{steps}.\n"
                    f"How many DISTINCT rooms did they visit in total? Output only a single integer.\nAnswer: ")
        else:
            if len(chars)<2: continue
            c1,c2=rng.sample(chars,2)
            flags=[]
            for t in range(len(states)):
                r2c=eval_utils.rooms_to_room2chars(states[t].get("rooms",{}))
                flags.append(any(c1 in o and c2 in o for o in r2c.values()))
            gold=sum(flags)
            steps="; ".join(f"step {i+1}: {'together' if f else 'apart'}" for i,f in enumerate(flags))
            prompt=(f"Two people moved through a house. At each step they were either together (same room) or apart:\n{steps}.\n"
                    f"In how many steps were they together? Output only a single integer.\nAnswer: ")
        pred=ask(model,proc,prompt,dev)
        n+=1
        by_gold.setdefault(gold,[0,0]); by_gold[gold][1]+=1
        if pred==gold: correct+=1; by_gold[gold][0]+=1
        if n%25==0: print(f"  {n}/{len(dirs)} acc={correct/max(1,n):.3f}",flush=True)
    acc=correct/max(1,n)
    print(f"\nTEXT-ORACLE {args.task} (7B, symbolic, no vision): acc={acc:.3f} n={n}")
    for g in sorted(by_gold):
        c,tot=by_gold[g]; print(f"  gold={g}: {c}/{tot}={c/max(1,tot):.2f}")
    (args.output/f"{args.task}_result.txt").write_text(f"acc={acc:.4f} n={n}\n"+
        "\n".join(f"gold={g}: {by_gold[g][0]}/{by_gold[g][1]}" for g in sorted(by_gold))+"\n",encoding="utf-8")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
