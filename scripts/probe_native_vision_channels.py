"""Diagnostic-only final-query channel interchange for a selected V2 global adapter."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
import sys
import time
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts.native_aggregation import CHECKPOINT_BASE, DATA_BASE, write_json
IDENTITY_ATOL, IDENTITY_RTOL = 0.03125, 0.001
CHANNELS = {"hidden": "query_down", "read": "read_down"}

class FinalQueryChannels:
    """One final query of this forward; independent offsets handle query tiling."""
    def __init__(self, branch, query_count, replacements=None):
        if branch.variant != "global" or branch.mode != "all" or query_count < 1:
            raise ValueError("Expected active global branch and positive query count")
        self.branch, self.query_count = branch, query_count
        self.replacements = {} if replacements is None else dict(replacements)
        if self.replacements.keys() - CHANNELS.keys():
            raise ValueError("Unknown replacement channel")
        self.captured, self.handles = {}, []
        self.offsets = dict.fromkeys(CHANNELS, 0)
    def _hook(self, channel):
        def hook(_module, args):
            import torch
            x = args[0]
            if x.ndim != 3 or x.shape[0] != 1:
                raise ValueError("Expected batch-one [B,Q,D] channel input")
            index = self.query_count - 1 - self.offsets[channel]
            self.offsets[channel] += x.shape[1]
            if not 0 <= index < x.shape[1]:
                return None
            if channel in self.captured:
                raise RuntimeError("Final query visited twice")
            original = x[:, index, :]
            self.captured[channel] = original.detach().clone()
            if channel not in self.replacements:
                return None
            replacement = self.replacements[channel].to(original)
            if replacement.shape != original.shape or not torch.isfinite(replacement).all():
                raise ValueError("Invalid replacement shape/values")
            altered = x.clone()
            altered[:, index, :] = replacement
            return (altered, *args[1:])
        return hook
    def __enter__(self):
        for channel, name in CHANNELS.items():
            self.handles.append(getattr(self.branch, name).register_forward_pre_hook(self._hook(channel)))
        return self
    def __exit__(self, kind, value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if kind is None and (self.captured.keys() != CHANNELS.keys() or
                             any(n != self.query_count for n in self.offsets.values())):
            raise RuntimeError("Hooks did not cover exactly the intended queries")
        return False

def match_recipient_norm(donor, recipient):
    import torch
    dn = torch.linalg.vector_norm(donor.float(), dim=-1, keepdim=True)
    rn = torch.linalg.vector_norm(recipient.float(), dim=-1, keepdim=True)
    if (not torch.isfinite(dn).all() or not torch.isfinite(rn).all()
            or (dn <= 1e-8).any() or (rn <= 1e-8).any()):
        raise ValueError("Zero or nonfinite norm in channel control")
    return (donor.float() * (rn / dn)).to(donor.dtype)

def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()

def validate_record(record, root):
    from scripts.native_aggregation_vlm_v2 import sample_metadata
    directory = Path(record["path"]).resolve()
    if directory.parent != (root / "mmred_vfiltered/seq_len_32/probe").resolve():
        raise ValueError("Sample outside dedicated probe split")
    actual = sample_metadata(directory, 32)
    for key in ("sid", "n_frames", "question", "gold", "qa_sha256"):
        if actual[key] != record[key]:
            raise ValueError(f"Probe metadata mismatch: {key}")
    if len(record["image_files"]) != 32:
        raise ValueError("Expected 32 images")
    for observed, expected in zip(actual["image_files"], record["image_files"]):
        if (Path(observed["path"]).resolve() != Path(expected["path"]).resolve()
                or observed["bytes"] != expected["bytes"]
                or digest(observed["path"]) != expected["sha256"]):
            raise ValueError("Probe image provenance mismatch")

def restore_global(run_dir):
    import torch
    from transformers import __version__ as transformers_version
    from gnnformer.carriers import attach_lora
    from gnnformer.native_aggregation import attach_native_aggregation
    from gnnformer.runtime import get_layers, load_runtime
    config = json.loads((run_dir / "config.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    arch = config["architecture"]
    if (arch["protocol"] != "vision_v2_clean" or arch["arm"] != "global"
            or arch["variant"] != "global" or arch["quantization"] != "nf4"
            or arch["center_messages"] or summary["arm"] != "global"
            or config["branch_operator"]["merge"] != "mean"):
        raise ValueError("Not a V2 global checkpoint")
    if config["transformers_version"] != str(transformers_version):
        raise ValueError("Transformers version changed")
    sources = ("scripts/native_aggregation_vlm_v2.py", "scripts/native_aggregation.py",
               "gnnformer/native_aggregation.py", "gnnformer/runtime.py",
               "gnnformer/data.py", "gnnformer/carriers.py")
    for name in sources:
        if digest(REPO / name) != config["code_sha256"][name]:
            raise ValueError(f"Selected-run source changed: {name}")
    path = Path(summary["selected_checkpoint"]).resolve()
    if not path.is_relative_to(CHECKPOINT_BASE) or not path.is_file():
        raise ValueError("Invalid selected checkpoint path")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["architecture"] != arch or saved["config"]["run_id"] != config["run_id"]:
        raise ValueError("Checkpoint architecture/run mismatch")
    def score(epoch):
        dev = epoch["dev"]
        n = sum(item["n"] for item in dev)
        return (sum(item["correct"] for item in dev) / n,
                -sum(item["gold_first_token_nll"] * item["n"] for item in dev) / n)
    selected_epoch = max(summary["training"], key=score)["epoch"]
    if saved["epoch"] != selected_epoch:
        raise ValueError("Checkpoint does not match dev selection")
    runtime = load_runtime(arch["model"], use_4bit=True, attn_implementation="sdpa", device_map="cuda")
    model = runtime.model
    model.requires_grad_(False)
    branch = attach_native_aggregation(model, arch["layer_index"],
        block_size=arch["block_size"], rank=arch["rank"],
        query_chunk_size=config["query_chunk_size"], center_messages=False,
        **config["branch_operator"])
    branch.load_state_dict(saved["branch"], strict=True)
    layers = get_layers(model)
    lora = attach_lora(layers, len(layers)-arch["lora_layers"], rank=arch["lora_rank"],
                       alpha=arch["lora_alpha"], device=runtime.device)
    if set(saved["lora"]) != {f"{i}.{n}" for i, n in lora.params}:
        raise ValueError("LoRA checkpoint keys differ")
    with torch.no_grad():
        for (index, name), pair in lora.params.items():
            for parameter, stored in zip(pair, saved["lora"][f"{index}.{name}"]):
                if parameter.shape != stored.shape:
                    raise ValueError("LoRA checkpoint shape differs")
                parameter.copy_(stored.to(parameter))
                parameter.requires_grad_(False)
    branch.requires_grad_(False)
    branch.mode = "all"
    model.eval()
    provenance = dict(run_dir=str(run_dir), run_id=config["run_id"], checkpoint=str(path),
        checkpoint_sha256=digest(path), selected_epoch=selected_epoch, architecture=arch,
        source_sha256={name: digest(REPO / name) for name in sources})
    return runtime, branch, config, provenance

def prepare_inputs(runtime, record, resize, max_tokens):
    from gnnformer.data import build_count_prompt, build_prompt_inputs, load_mmred_sample
    from gnnformer.runtime import move_to_device
    sid, frames, question, _states, answer = load_mmred_sample(Path(record["path"]))
    if sid != record["sid"] or question != record["question"] or int(answer) != record["gold"]:
        raise ValueError("Sample changed after validation")
    resized = []
    try:
        resized = [frame.resize((resize, resize)) for frame in frames]
        inputs = build_prompt_inputs(runtime.processor, resized, build_count_prompt(question, len(resized)))
    finally:
        for frame in frames + resized:
            frame.close()
    if inputs["input_ids"].shape[1] > max_tokens or inputs["image_grid_thw"].shape[0] != 32:
        raise ValueError("Invalid input size; no truncation allowed")
    return move_to_device(inputs, runtime.device)

def run(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Run this diagnostic inside a Slurm GPU allocation")
    if not 1 <= args.limit_pairs <= 16 or not args.pairs_manifest.resolve().is_relative_to(DATA_BASE):
        raise SystemExit("Invalid pair count or manifest path")
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA allocation required")
    torch.set_num_threads(max(1, min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))))
    started = time.monotonic()
    raw_manifest = args.pairs_manifest.read_bytes()
    manifest = json.loads(raw_manifest)
    if manifest["schema_version"] != 1 or manifest["purpose"] != "causal_probe" or len(manifest["pairs"]) != 16:
        raise ValueError("Expected registered 16-pair manifest")
    dataset_root = Path(manifest["dataset_root"]).resolve()
    if not dataset_root.is_relative_to(DATA_BASE):
        raise ValueError("Dataset outside DATA_BASE")
    pairs = manifest["pairs"][:args.limit_pairs]
    if len({p["pair_id"] for p in pairs}) != len(pairs):
        raise ValueError("Duplicate pair IDs")
    for pair in pairs:
        low, high = pair["low"], pair["high"]
        if pair["n_frames"] != 32 or low["gold"] != 2 or high["gold"] != 6 or low["question"] != high["question"]:
            raise ValueError("Pair differs from registered N/question/K contrast")
        validate_record(low, dataset_root)
        validate_record(high, dataset_root)
        changes = [i for i, (x, y) in enumerate(zip(low["image_files"], high["image_files"])) if x["sha256"] != y["sha256"]]
        if len(changes) != 4 or changes != sorted(pair["changed_positions"]):
            raise ValueError("Expected exactly the four declared image changes")
    runtime, branch, config, provenance = restore_global(args.run_dir.resolve())
    if Path(config["dataset_root"]).resolve() != dataset_root:
        raise ValueError("Probe and checkpoint dataset roots differ")
    outdir = args.output / f"channels_{time.strftime('%Y%m%d_%H%M%S')}_{os.environ['SLURM_JOB_ID']}"
    outdir.mkdir(parents=True, exist_ok=False)
    (outdir / "pairs_manifest.json").write_bytes(raw_manifest)
    (outdir / "probe_source.py").write_bytes(Path(__file__).read_bytes())
    provenance.update(probe_sha256=digest(__file__), pairs_manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
        limit_pairs=args.limit_pairs, directions="both", forwards_expected=16*len(pairs),
        identity_atol=IDENTITY_ATOL, identity_rtol=IDENTITY_RTOL,
        scope="Artificial branch-input interchange; no training, generation, or new attention code")
    write_json(outdir / "config.json", provenance)
    forwards = 0
    def evaluate(inputs, replacements=None):
        nonlocal forwards
        with FinalQueryChannels(branch, inputs["input_ids"].shape[1], replacements) as context:
            output = runtime.model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True)
            logits = output.logits[0, -1].float().detach().cpu()
        forwards += 1
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite native logits")
        return logits, context.captured
    def metrics(logits, recipient_id, donor_id):
        probabilities = logits.softmax(-1)
        predicted = int(logits.argmax())
        return dict(raw_logit_margin=float(logits[donor_id]-logits[recipient_id]),
            raw_recipient_logit=float(logits[recipient_id]), raw_donor_logit=float(logits[donor_id]),
            recipient_probability=float(probabilities[recipient_id]), donor_probability=float(probabilities[donor_id]),
            argmax_token=predicted, argmax_text=runtime.tokenizer.decode([predicted], skip_special_tokens=False))
    rows, baselines = [], []
    with torch.no_grad():
        for pair in pairs:
            prepared = {side: prepare_inputs(runtime, pair[side], config["resize"], config["max_seq_tokens"]) for side in ("low", "high")}
            for key in ("input_ids", "attention_mask", "image_grid_thw"):
                if not torch.equal(prepared["low"][key], prepared["high"][key]):
                    raise ValueError("Paired textual inputs/image grids differ")
            tokens = {side: runtime.tokenizer(str(pair[side]["gold"]), add_special_tokens=False).input_ids for side in ("low", "high")}
            if any(len(ids) != 1 for ids in tokens.values()) or tokens["low"] == tokens["high"]:
                raise ValueError("Require distinct single-token answers")
            original = {side: evaluate(prepared[side]) for side in ("low", "high")}
            for recipient, donor in (("low", "high"), ("high", "low")):
                recipient_logits, own = original[recipient]
                donor_logits, other = original[donor]
                recipient_id, donor_id = tokens[recipient][0], tokens[donor][0]
                baseline = metrics(recipient_logits, recipient_id, donor_id)
                donor_baseline = metrics(donor_logits, recipient_id, donor_id)
                common = dict(pair_id=pair["pair_id"], direction=f"{recipient}_to_{donor}",
                    recipient_path=pair[recipient]["path"], donor_path=pair[donor]["path"],
                    recipient_gold=pair[recipient]["gold"], donor_gold=pair[donor]["gold"],
                    recipient_token=recipient_id, donor_token=donor_id, n_frames=32,
                    prompt_tokens=prepared[recipient]["input_ids"].shape[1])
                baselines.append(dict(**common, recipient=baseline, donor=donor_baseline,
                    full_context_margin_difference=donor_baseline["raw_logit_margin"]-baseline["raw_logit_margin"],
                    channel_norms={name: dict(recipient=float(own[name].float().norm()), donor=float(other[name].float().norm())) for name in CHANNELS}))
                matched = {name: match_recipient_norm(other[name], own[name]) for name in CHANNELS}
                conditions = (("identity_both", own),
                    ("donor_hidden", {"hidden": other["hidden"]}), ("donor_read", {"read": other["read"]}),
                    ("donor_both", other), ("norm_hidden", {"hidden": matched["hidden"]}),
                    ("norm_read", {"read": matched["read"]}), ("norm_both", matched))
                for condition, replacement in conditions:
                    logits, _ = evaluate(prepared[recipient], replacement)
                    current = metrics(logits, recipient_id, donor_id)
                    row = dict(**common, condition=condition, **current,
                        margin_shift=current["raw_logit_margin"]-baseline["raw_logit_margin"],
                        full_vocabulary_max_logit_change=float((logits-recipient_logits).abs().max()))
                    rows.append(row)
                    if condition == "identity_both":
                        row["identity_passed"] = bool(torch.allclose(logits, recipient_logits, atol=IDENTITY_ATOL, rtol=IDENTITY_RTOL))
                        if not row["identity_passed"]:
                            write_json(outdir / "identity_failure.json", row)
                            raise RuntimeError("Identity changed full native logits beyond registered tolerance")
                print(json.dumps(dict(pair=pair["pair_id"], direction=common["direction"], forwards=forwards)), flush=True)
            write_json(outdir / "baselines.json", baselines)
            write_json(outdir / "interventions.json", rows)
            del prepared, original
    if forwards != 16*len(pairs) or len(rows) != 14*len(pairs):
        raise RuntimeError("Unexpected forward/intervention denominator")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    summary = dict(pairs=len(pairs), directions=2*len(pairs), forwards=forwards, seconds=time.monotonic()-started,
        estimand="Within-recipient change in raw donor-minus-recipient answer-token logit margin",
        scope="Descriptive artificial causal interchange; one checkpoint; two directions within each pair are dependent",
        limitations="Swaps can create off-distribution channel combinations; read effects do not establish extra aggregation bandwidth",
        conditions={name: dict(n=len(items), mean_margin_shift=mean(x["margin_shift"] for x in items),
            positive_margin_shifts=sum(x["margin_shift"] > 0 for x in items),
            maximum_full_vocabulary_logit_change=max(x["full_vocabulary_max_logit_change"] for x in items)) for name, items in grouped.items()})
    write_json(outdir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--pairs-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, default=REPO / "outputs/native_aggregation_vlm/v2/channel_probe")
    p.add_argument("--limit-pairs", type=int, default=16)
    p.add_argument("--directions", choices=("both",), default="both")
    return p

if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
