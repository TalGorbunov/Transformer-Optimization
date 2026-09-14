"""Train/dev splits in the data/mmred_filtered grammar (one (room, [character]) per frame; question
'How many frames show C in the R?'; gold = #frames with (C, R), uniform over 0..min(8, N)). The test
splits under data/mmred_filtered are never touched; this writes data/mmred_filtered_train/. Distractor
frames: the queried character in another room (p = .5) or another character in any room."""
import argparse, random
from pathlib import Path
ROOMS = ["Kitchen", "Bathroom", "Garden", "Office", "Bedroom", "Park"]
CHARS = ["Sandra", "Mary", "Michael", "John", "Daniel", "Laura", "Peter", "Emma", "Noah"]

def sample(rng, N, K):
    C, R = rng.choice(CHARS), rng.choice(ROOMS)
    ev = set(rng.sample(range(N), K)); lines = []
    for i in range(N):
        if i in ev:
            r, c = R, C
        elif rng.random() < .5:
            r, c = rng.choice([x for x in ROOMS if x != R]), C
        else:
            r, c = rng.choice(ROOMS), rng.choice([x for x in CHARS if x != C])
        lines.append(f"{{'step_id': {i + 1}, 'rooms': {{'{r}': ['{c}']}}}}")
    return "question:\n" + "\n".join(lines) + f"\nHow many frames show {C} in the {R}?\nanswer:\n{K}\n"

ap = argparse.ArgumentParser(); ap.add_argument("--ns", default="8+16+32"); ap.add_argument("--per-gold", type=int, default=40)
ap.add_argument("--dev-per-gold", type=int, default=10); ap.add_argument("--seed", type=int, default=20260903)
ap.add_argument("--out", default="data/mmred_filtered_train"); a = ap.parse_args()
for N in [int(x) for x in a.ns.split("+")]:
    for split, per, seed_off in (("train", a.per_gold, 0), ("dev", a.dev_per_gold, 1)):
        rng = random.Random(a.seed + 7 * N + seed_off); n = 0
        for K in range(0, min(8, N) + 1):
            for j in range(per):
                d = Path(a.out) / f"seq_len_{N}" / split / f"filtered_N{N}_K{K}_{j:04d}"; d.mkdir(parents=True, exist_ok=True)
                (d / "qa.txt").write_text(sample(rng, N, K)); n += 1
        print(f"{split} N={N}: {n} samples -> {Path(a.out) / f'seq_len_{N}' / split}")
