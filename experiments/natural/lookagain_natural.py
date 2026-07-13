#!/usr/bin/env python3
"""Per-frame look-again judge on mmred_natural samples (pilot gate: per-frame extraction
>=0.95 REQUIRED before the full build, per plan A4). Single-frame forward,
"Does a <concept> appear in this frame? yes/no" -> P(yes) from yes/no logits.
Writes lookagain.json per sample dir + a summary with per-cell judge accuracy/AUROC vs GT.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/mmred_natural")
    ap.add_argument("--cells", default="ident_far,dist_far,ident_near,dist_near")
    ap.add_argument("--limit", type=int, default=0, help="per cell")
    ap.add_argument("--model_name", "--model", dest="model_name",
                    default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    from sklearn.metrics import roc_auc_score
    from evaluations.helpers import patching_core as tgi
    from evaluations.scripts.patch_importence import group_restoration_importance as gri
    gri.configure_runtime(args.model_name)
    model = gri._model(); processor = gri._processor()
    tok = processor.tokenizer
    yes_ids = [tok.encode(t, add_special_tokens=False)[0] for t in ("yes", "Yes")]
    no_ids = [tok.encode(t, add_special_tokens=False)[0] for t in ("no", "No")]

    summary = {}
    for cell in [c.strip() for c in args.cells.split(",")]:
        root = Path(args.data_root) / cell
        dirs = sorted(d for d in root.iterdir() if (d / "meta.json").exists()) if root.exists() else []
        if args.limit:
            dirs = dirs[: args.limit]
        probs, gts = [], []
        for i, sd in enumerate(dirs):
            meta = json.loads((sd / "meta.json").read_text())
            concept = meta.get("concept", "dog")
            out = {}
            for t, fr in enumerate(meta["frames"]):
                fp = sd / f"frame_{t:02d}.jpg"
                img = Image.open(fp).convert("RGB")
                prompt = (f"Look at this single photograph.\n"
                          f"Does a {concept} appear in this photo? Answer yes or no.\nAnswer: ")
                inputs = tgi.move_inputs_to_model_device(tgi.build_inputs_from_prompt([img], prompt))
                with torch.no_grad():
                    logits = model(**inputs, use_cache=False).logits[0, -1].float()
                py = torch.logsumexp(logits[yes_ids], 0)
                pn = torch.logsumexp(logits[no_ids], 0)
                p = torch.sigmoid(py - pn).item()
                out[fp.name] = round(p, 4)
                probs.append(p); gts.append(int(fr["is_evidence"]))
            (sd / "lookagain.json").write_text(json.dumps(out, indent=1))
            if (i + 1) % 20 == 0:
                print(f"[{cell} {i+1}/{len(dirs)}]", flush=True)
        if probs:
            pr = np.array(probs); gt = np.array(gts)
            acc = float(((pr > 0.5).astype(int) == gt).mean())
            auc = float(roc_auc_score(gt, pr)) if len(set(gt)) > 1 else float("nan")
            fn = float((pr[gt == 1] <= 0.5).mean()) if (gt == 1).any() else float("nan")
            fp_ = float((pr[gt == 0] > 0.5).mean()) if (gt == 0).any() else float("nan")
            summary[cell] = {"n_frames": len(pr), "acc": acc, "auroc": auc, "fn": fn, "fp": fp_,
                             "gate_pass_0.95": bool(acc >= 0.95)}
            print(f"{cell}: acc {acc:.3f} AUROC {auc:.3f} FN {fn:.3f} FP {fp_:.3f} "
                  f"-> {'PASS' if acc >= 0.95 else 'FAIL'}", flush=True)
    outp = Path(args.output) if args.output else Path(args.data_root) / "lookagain_summary.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(summary, indent=1))
    print("Wrote", outp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
