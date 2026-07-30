#!/usr/bin/env python3
"""MLVU-AC ground-truth recovery: locate inserted needle clips by self-similarity.

The MLVU count task builds each video by inserting the SAME short clip k times into a long
background video (k = gold answer, 1..5). Frames inside different insertions of the same clip
are near-identical, so far-apart near-duplicate thumbnail pairs pinpoint the insertions —
exact per-frame evidence GT with no judge in the loop. (k=1 has nothing to match against and
is handled downstream: behavioral-only, or judge labels.)

Usage (module): segs = detect(video_path)  ->  list of (t_start, t_end) insertion segments.
CLI: --videos a.mp4 b.mp4 [--fps 2] [--ham 6] [--min-gap 20] prints segments per video.
"""
from __future__ import annotations
import argparse
import numpy as np


def dhash_and_texture(img, size=8):
    g = np.asarray(img.convert("L").resize((size + 1, size)), dtype=np.int16)
    h = (g[:, 1:] > g[:, :-1]).flatten()
    return h, float(np.abs(np.diff(g.astype(np.float32), axis=1)).mean())


_POP = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.int16)


def hamming_matrix(H: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Pairwise hamming distances of [n,64] bool hashes, memory-bounded (packbits + chunked
    XOR-popcount): peak extra memory ~ chunk*n*8 int16 instead of n*n*64 bool."""
    P = np.packbits(H.astype(np.uint8), axis=1)          # [n, 8] uint8
    n = len(P)
    D = np.empty((n, n), dtype=np.int16)
    for i in range(0, n, chunk):
        x = P[i:i + chunk, None, :] ^ P[None, :, :]      # [b, n, 8]
        D[i:i + chunk] = _POP[x].sum(-1, dtype=np.int16)
    return D


def decode_thumbs(video_path: str, fps: float = 2.0):
    import av
    hashes, times, tex = [], [], []
    with av.open(str(video_path)) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        step = 1.0 / fps
        next_t = 0.0
        for fr in c.decode(st):
            if fr.time is None:
                continue
            if fr.time >= next_t:
                h, tx = dhash_and_texture(fr.to_image())
                hashes.append(h); times.append(fr.time); tex.append(tx)
                next_t += step
    return np.stack(hashes), np.array(times), np.array(tex)


def detect(video_path: str, fps: float = 2.0, ham: int = 6, min_gap: float = 20.0,
           tex_min: float = 4.0, run_gap: float = 1.6, min_len: float = 0.9):
    """Returns (segments [(t0,t1)...], n_thumbs, diag dict)."""
    H, T, X = decode_thumbs(video_path, fps)
    n = len(H)
    ok = X >= tex_min                       # exclude flat frames (match everything)
    D = hamming_matrix(H)
    far = np.abs(T[:, None] - T[None, :]) >= min_gap
    match = (D <= ham) & far & ok[:, None] & ok[None, :]
    involved = match.any(1)
    # maximal runs of involved thumbs with small gaps
    segs = []
    i = 0
    while i < n:
        if not involved[i]:
            i += 1; continue
        j = i
        last = i
        while j + 1 < n and (T[j + 1] - T[last]) <= run_gap:
            j += 1
            if involved[j]:
                last = j
        segs.append((float(T[i]), float(T[last])))
        i = last + 1
    segs = [(a, b) for a, b in segs if (b - a) >= min_len]
    diag = {"n_thumbs": n, "flat_frac": float(1 - ok.mean()),
            "n_involved": int(involved.sum())}
    return segs, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--ham", type=int, default=6)
    ap.add_argument("--min-gap", type=float, default=20.0)
    args = ap.parse_args()
    for v in args.videos:
        segs, diag = detect(v, fps=args.fps, ham=args.ham, min_gap=args.min_gap)
        print(f"{v}: {len(segs)} segments {[(round(a,1), round(b,1)) for a, b in segs]}  {diag}")


if __name__ == "__main__":
    main()
