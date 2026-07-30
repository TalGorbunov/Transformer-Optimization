#!/usr/bin/env python3
"""One-off CPU analysis: gold-count distribution + diversity for candidate Cat-1 MMReD tasks
on the park dataset. Cat-1 = answer is Sum_t 1[per-frame predicate]. Picks the most count-diverse."""
import math, random, sys, zlib
from collections import Counter
from pathlib import Path

ROOT = Path("/home/tal.gorbunov/projects/Transformer-Optimization")
sys.path.insert(0, str(ROOT))
from evaluations.scripts import eval_mmred_rooms_visited_baseline as rv
from evaluations.scripts import eval_mmred_text_frames_acc as tf

DATA = ROOT / "data" / "mmred_images_park"
SPLIT = "all_uniform"
SEQ_LENS = list(range(1, 9))
PER_SL = 250  # sample cap per seq_len


def occ(state, room):
    return state["rooms"].get(room, [])


def per_frame_indicators(task, dstr, states, rng):
    """Return list of 0/1 per frame for a candidate Cat-1 task, or None to skip."""
    rooms = list(states[0]["rooms"].keys())
    chars = rv.present_characters(states)
    name = Path(dstr).name
    rsel = random.Random(zlib.crc32((name + task).encode()))
    if task == "room_busy":
        R = rsel.choice(rooms)
        return [int(len(occ(s, R)) >= 1) for s in states]
    if task == "room_empty":
        R = rsel.choice(rooms)
        return [int(len(occ(s, R)) == 0) for s in states]
    if task == "room_crowded":
        R = rsel.choice(rooms)
        return [int(len(occ(s, R)) >= 2) for s in states]
    if task == "char_present":
        if not chars:
            return None
        C = rsel.choice(chars)
        return [int(tf.room_of(s, C) != "not present") for s in states]
    if task == "char_alone":
        if not chars:
            return None
        C = rsel.choice(chars)
        out = []
        for s in states:
            r = tf.room_of(s, C)
            out.append(int(r != "not present" and len(occ(s, r)) == 1))
        return out
    if task == "char_accompanied":
        if not chars:
            return None
        C = rsel.choice(chars)
        out = []
        for s in states:
            r = tf.room_of(s, C)
            out.append(int(r != "not present" and len(occ(s, r)) >= 2))
        return out
    if task == "co_occupancy":  # existing, for reference
        if len(chars) < 2:
            return None
        C, D = rsel.sample(chars, 2)
        return [int(tf.room_of(s, C) == tf.room_of(s, D) and tf.room_of(s, C) != "not present") for s in states]
    if task == "steps_in_room":  # existing, for reference (uses metadata target)
        import json
        m = json.loads((Path(dstr) / "metadata.json").read_text())
        C, R = m.get("target_character"), m.get("target_room")
        if not C or not R:
            return None
        return [int(tf.room_of(s, C) == R) for s in states]
    raise ValueError(task)


TASKS = ["room_busy", "room_empty", "room_crowded", "char_present", "char_alone",
         "char_accompanied", "co_occupancy", "steps_in_room"]


def main():
    rng = random.Random(0)
    golds = {t: Counter() for t in TASKS}
    by_sl = {t: {sl: Counter() for sl in SEQ_LENS} for t in TASKS}
    for sl in SEQ_LENS:
        sr = DATA / f"seq_len_{sl}" / SPLIT
        if not sr.is_dir():
            continue
        dirs = [d for d in sorted(sr.iterdir()) if (d / "qa.txt").is_file() and (d / "metadata.json").is_file()]
        rng.shuffle(dirs)
        dirs = dirs[:PER_SL]
        for d in dirs:
            states = rv.states_of(d / "qa.txt")
            if not states:
                continue
            for t in TASKS:
                ind = per_frame_indicators(t, str(d), states, rng)
                if ind is None:
                    continue
                golds[t][sum(ind)] += 1
                by_sl[t][sl][sum(ind)] += 1

    def cstats(c):
        N = sum(c.values())
        if N == 0:
            return 0, 0.0, 0.0
        mean = sum(k * v for k, v in c.items()) / N
        std = (sum((k - mean) ** 2 * v for k, v in c.items()) / N) ** 0.5
        return N, mean, std

    print(f"{'task':16s} {'N':>6s} {'entropy':>8s} {'within_sl_std':>13s} {'p_fire':>7s} "
          f"{'pct_deg':>8s}  per-seq_len gold mean (sl1..sl8)")
    rows = []
    for t in TASKS:
        c = golds[t]
        N = sum(c.values())
        if N == 0:
            continue
        probs = [v / N for v in c.values()]
        ent = -sum(p * math.log2(p) for p in probs if p > 0)
        # within-seq_len std averaged over seq_len: genuine per-frame variation, not just seq_len tracking
        wstds, sl_means, deg = [], [], 0
        tot = 0
        p_fire_num = p_fire_den = 0
        for sl in SEQ_LENS:
            n, m, s = cstats(by_sl[t][sl])
            if n:
                wstds.append(s); sl_means.append(m)
                deg += by_sl[t][sl].get(sl, 0) + by_sl[t][sl].get(0, 0)  # all-true or all-false (degenerate)
                tot += n
                p_fire_num += m; p_fire_den += sl  # avg fire-rate proxy
        w_std = sum(wstds) / max(1, len(wstds))
        p_fire = sum(sl_means[i] / SEQ_LENS[i] for i in range(len(sl_means))) / max(1, len(sl_means))
        pct_deg = deg / max(1, tot)
        slm = " ".join(f"{m:.1f}" for m in sl_means)
        rows.append((w_std, ent, t, N, p_fire, pct_deg, slm))
    for w_std, ent, t, N, p_fire, pct_deg, slm in sorted(rows, reverse=True):
        print(f"{t:16s} {N:6d} {ent:8.3f} {w_std:13.3f} {p_fire:7.2f} {pct_deg:8.2f}  {slm}")
    print("\nwithin_sl_std = avg over seq_len of std(gold|seq_len)  (higher = genuine per-frame variation)")
    print("p_fire = avg per-frame predicate fire-rate;  pct_deg = frac of samples with gold==0 or gold==seq_len (degenerate)")


if __name__ == "__main__":
    main()
