"""Inspect V1 checkpoint branch magnitudes on four fixed profile imagesets.

This is a forward-only diagnostic, not an efficacy evaluation or intervention.
The zero-read component still receives a contextual hidden state and can contain
visual evidence. Its algebraic decomposition does not identify causal use.
All model and image work requires a Slurm GPU allocation.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.native_aggregation import (  # noqa: E402
    CHECKPOINT_BASE, DATA_BASE, arm_configuration, compatible_architecture, write_json,
)
from scripts.native_aggregation_vlm import sample_metadata  # noqa: E402


ARMS = ("sum", "mean", "post_sum", "post_mean", "global", "hierarchical", "lora")
PROFILE_NS = (16, 64)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=REPO_ROOT / "outputs/native_aggregation_vlm/v1")
    p.add_argument("--manifest", type=Path,
                   default=DATA_BASE / "native_aggregation_vision_pilot/profile_manifest.json")
    a = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Submit this inspection through Slurm with one GPU")
    if not a.manifest.resolve().is_relative_to(DATA_BASE):
        raise SystemExit(f"Manifest must live under {DATA_BASE}")
    staged = json.loads(a.manifest.read_text())
    if staged.get("schema_version") != 1:
        raise ValueError("Expected a version-1 profile manifest")
    records = []
    dataset_root = Path(staged["dataset_root"]).resolve()
    if not dataset_root.is_relative_to(DATA_BASE):
        raise ValueError("Profile data root is outside the requested data storage")
    for n in PROFILE_NS:
        cell = staged["splits"][f"test_N{n}"]["samples"]
        if len(cell) != 2:
            raise ValueError(f"Inspection requires exactly two profile examples at N{n}")
        expected_root = dataset_root / "mmred_vfiltered" / f"seq_len_{n}" / "test"
        for declared in cell:
            directory = Path(declared["path"])
            if directory.resolve().parent != expected_root:
                raise ValueError("Profile example is outside its declared split")
            actual = sample_metadata(directory, n)
            for key in ("sid", "n_frames", "gold", "qa_sha256"):
                if actual[key] != declared[key]:
                    raise ValueError(f"Profile metadata changed: {directory}, {key}")
            if len(declared["image_files"]) != n:
                raise ValueError("Profile image list differs")
            for observed, expected in zip(actual["image_files"], declared["image_files"]):
                if (Path(observed["path"]).resolve() != Path(expected["path"]).resolve()
                        or observed["bytes"] != expected["bytes"]
                        or sha256(observed["path"]) != expected["sha256"]):
                    raise ValueError(f"Profile image differs: {observed['path']}")
            records.append(actual)
    runs = {}
    common_manifest = set()
    for arm in ARMS:
        paths = sorted((a.root / "main" / arm).glob("*/summary.json"))
        if len(paths) != 1:
            raise ValueError(f"Expected one completed main run for {arm}: {paths}")
        summary_path, = paths
        summary = json.loads(summary_path.read_text())
        config = json.loads(summary_path.with_name("config.json").read_text())
        if config["arm"] != arm or config["epochs"] != 9 or len(summary["training"]) != 9:
            raise ValueError("Inspection requires the completed nine-epoch V1 runs")
        checkpoint = Path(summary["selected_checkpoint"])
        if not checkpoint.resolve().is_relative_to(CHECKPOINT_BASE) or not checkpoint.is_file():
            raise ValueError("Selected checkpoint is missing or outside checkpoint storage")
        for source in ("gnnformer/native_aggregation.py", "gnnformer/carriers.py",
                       "gnnformer/runtime.py", "gnnformer/data.py"):
            if sha256(REPO_ROOT / source) != config["code_sha256"][source]:
                raise ValueError(f"Current implementation differs from the trained source: {source}")
        common_manifest.add(config["manifest_sha256"])
        runs[arm] = dict(config=config, checkpoint=str(checkpoint),
                         checkpoint_sha256=sha256(checkpoint), source_run=str(summary_path.parent))
    if len(common_manifest) != 1:
        raise ValueError("Main runs used different data manifests")
    reference = runs["sum"]["config"]
    for run in runs.values():
        for field in ("model", "resize", "resolved_layer_index", "block_size", "seed"):
            if run["config"][field] != reference[field]:
                raise ValueError(f"Main runs disagree on {field}")
    if reference["resize"] != 392 or reference["center_messages"]:
        raise ValueError("This decomposition expects uncentered V1 with 392-pixel images")

    # Heavy imports follow the Slurm guard and input/checkpoint provenance checks.
    import torch
    import torch.nn.functional as F
    from gnnformer.carriers import attach_lora
    from gnnformer.data import build_count_prompt, build_prompt_inputs, load_mmred_sample
    from gnnformer.native_aggregation import attach_native_aggregation
    from gnnformer.runtime import get_layers, load_runtime, move_to_device
    if not torch.cuda.is_available():
        raise SystemExit("A Slurm GPU allocation with CUDA is required")
    torch.set_num_threads(max(1, min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))))
    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{os.environ['SLURM_JOB_ID']}"
    outdir = a.root / "inspection" / run_id
    outdir.mkdir(parents=True, exist_ok=False)
    report = dict(
        run_id=run_id, slurm_job_id=os.environ["SLURM_JOB_ID"],
        model=reference["model"], quantization="nf4", resize=392,
        torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name(0),
        scope="Four fixed profile examples/model; first-token forward only; no efficacy claim",
        caveats=["Zero-read input uses contextual hidden states that can contain visual evidence.",
                 "The additive decomposition is algebraic, not a causal intervention.",
                 "Read-dependent remainder includes interactions with the contextual query.",
                 "Ordinary attention is captured directly at o_proj before branch addition; addition rounding is recorded separately.",
                 "No generation, optimization, model selection, or model-weight writes occur."],
        decomposition="delta_fp32 = W_up(s_R*phi(local messages)); zero = W_up(s_R*phi(W_q*h+b)); remainder = delta_fp32-zero; s_R=R for sum arms, 1 otherwise",
        manifest=str(a.manifest), manifest_sha256=sha256(a.manifest),
        source_sha256={name: sha256(REPO_ROOT / name) for name in
                      ("scripts/inspect_native_vision_v1.py", "gnnformer/native_aggregation.py",
                       "scripts/native_aggregation_vlm.py", "scripts/native_aggregation.py")},
        records=records, main_manifest_sha256=next(iter(common_manifest)), models={},
    )
    write_json(outdir / "inspection.json", report)
    (outdir / "INDEX.md").write_text(
        "# V1 checkpoint inspection\n\n"
        "[inspection.json](inspection.json) records four fixed profile examples per model.\n\n"
        "This is a forward-only algebraic diagnostic, not an efficacy test or causal intervention. "
        "The zero-read component receives a contextual hidden state.\n")

    def norm(vector):
        return float(torch.linalg.vector_norm(vector.float()))

    def ratio(numerator, denominator):
        return numerator / denominator if denominator > 0 else None

    def cosine(x, y):
        nx, ny = norm(x), norm(y)
        return float(torch.dot(x.float(), y.float()) / (nx * ny)) if nx > 0 and ny > 0 else None

    def visible_count(args, block_size):
        _hidden, _query, key, _value, mask, positions, _scaling = args
        length = key.shape[-2]
        position = positions[-1] if positions.ndim == 1 else positions[0, -1]
        visible = torch.arange(length, device=key.device) <= position
        if mask is not None:
            if mask.ndim == 2:
                visible = visible & mask[0, :length].bool()
            elif mask.ndim == 4:
                current = mask[0, :, -1 if mask.shape[-2] > 1 else 0, :length]
                allowed = current if current.dtype == torch.bool else (
                    torch.isfinite(current) & (current > torch.finfo(current.dtype).min / 2))
                visible = visible & allowed.any(dim=0)
            else:
                raise ValueError("Unexpected attention mask dimensions")
        padding = (-length) % block_size
        return int(F.pad(visible, (0, padding), value=False).view(-1, block_size).any(-1).sum())

    def inspect_arm(arm):
        info = runs.get(arm)
        config = reference if info is None else info["config"]
        runtime = load_runtime(config["model"], use_4bit=True,
                               attn_implementation="sdpa", device_map="cuda")
        model, processor, tokenizer = runtime.model, runtime.processor, runtime.tokenizer
        model.requires_grad_(False)
        layers = get_layers(model)
        layer_index = config["resolved_layer_index"]
        branch = lora = saved = None
        selected_epoch = selected_step = None
        handles, rows, capture = [], [], {}
        try:
            if info is not None:
                saved = torch.load(info["checkpoint"], map_location="cpu", weights_only=True)
                selected_epoch, selected_step = saved["epoch"], saved["step"]
                if compatible_architecture(saved["architecture"]) != compatible_architecture(config["architecture"]):
                    raise ValueError("Saved architecture differs from the source run")
                if config["center_messages"]:
                    raise ValueError("Inspection expects uncentered V1 checkpoints")
                if arm != "lora":
                    branch = attach_native_aggregation(
                        model, layer_index=layer_index, block_size=config["block_size"],
                        rank=config["rank"], query_chunk_size=config["query_chunk_size"],
                        center_messages=False,
                        **arm_configuration(arm, config.get("routing_rank", 8)))
                    branch.load_state_dict(saved["branch"], strict=True)
                    branch.requires_grad_(False)
                    branch.mode = "all"
                if config["lora_layers"]:
                    lora = attach_lora(layers, len(layers)-config["lora_layers"],
                                       rank=config["lora_rank"], alpha=config["lora_alpha"],
                                       device=runtime.device, state=saved["lora"])
                    for pair in lora.params.values():
                        for parameter in pair:
                            parameter.requires_grad_(False)
                saved = None
            model.eval()

            if branch is not None:
                def up_hook(_module, _args, output):
                    # Every tile overwrites this; the final tile contains the
                    # final prompt token, before the branch's native-dtype cast.
                    capture["delta_fp32"] = output[0, -1].detach().float()

                def branch_hook(module, args, output):
                    r = visible_count(args, module.block_size)
                    hidden = args[0][0, -1].to(module.up.weight.dtype)
                    baseline = module.query_down(hidden) + module.read_down.bias
                    if module.nonlinear:
                        baseline = F.silu(baseline)
                    scale = r if module.merge == "sum" else int(r > 0)
                    capture["zero"] = module.up.weight @ (baseline * scale)
                    capture["delta_native"] = output[0, -1].detach().float()
                    capture["visible_blocks"] = r
                    capture["scale"] = scale

                handles.append(branch.up.register_forward_hook(up_hook))
                handles.append(branch.register_forward_hook(branch_hook))

            def ordinary_hook(_module, _args, output):
                capture["ordinary"] = output[0, -1].detach().float()

            def attention_hook(_module, _args, output):
                capture["patched"] = output[0][0, -1].detach().float()

            handles.append(layers[layer_index].self_attn.o_proj.register_forward_hook(ordinary_hook))
            handles.append(layers[layer_index].self_attn.register_forward_hook(attention_hook))
            with torch.inference_mode():
                for record in records:
                    capture.clear()
                    sid, original, question, _states, answer = load_mmred_sample(Path(record["path"]))
                    if (sid != record["sid"] or question != record["question"]
                            or int(answer) != record["gold"] or len(original) != record["n_frames"]):
                        raise ValueError("Profile metadata changed during inspection")
                    resized = []
                    try:
                        resized = [frame.resize((392, 392)) for frame in original]
                        inputs = build_prompt_inputs(processor, resized,
                                                     build_count_prompt(question, len(resized)))
                    finally:
                        for frame in original + resized:
                            frame.close()
                    length = inputs["input_ids"].shape[1]
                    if length > config["max_seq_tokens"]:
                        raise ValueError("Inspection would exceed the registered sequence limit")
                    inputs = move_to_device(inputs, runtime.device)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.monotonic()
                    output = model(**inputs, use_cache=False, logits_to_keep=1)
                    torch.cuda.synchronize()
                    elapsed = time.monotonic()-started
                    logits = output.logits[0, -1].float()
                    gold_ids = tokenizer(str(record["gold"]), add_special_tokens=False).input_ids
                    if len(gold_ids) != 1:
                        raise ValueError("This V1 diagnostic expects a one-token gold digit")
                    prediction = int(logits.argmax())
                    ordinary = capture["ordinary"]
                    delta = capture.get("delta_native", torch.zeros_like(ordinary))
                    full = capture.get("delta_fp32", torch.zeros_like(ordinary))
                    zero = capture.get("zero", torch.zeros_like(ordinary))
                    remainder = full-zero
                    ordinary_norm = norm(ordinary)
                    row = dict(
                        arm=arm, path=record["path"], sid=sid, n_frames=record["n_frames"],
                        gold=record["gold"], prompt_tokens=length,
                        gold_token_id=gold_ids[0], first_token_argmax_id=prediction,
                        first_token_argmax_text=tokenizer.decode([prediction]),
                        first_token_matches_gold=prediction == gold_ids[0],
                        gold_first_token_nll=-float(F.log_softmax(logits, dim=-1)[gold_ids[0]]),
                        visible_blocks=capture.get("visible_blocks"), zero_component_scale=capture.get("scale"),
                        branch_delta_norm=norm(delta), branch_delta_before_cast_norm=norm(full),
                        branch_cast_rounding_norm=norm(full-delta), zero_read_component_norm=norm(zero),
                        read_dependent_component_norm=norm(remainder), ordinary_attention_norm=ordinary_norm,
                        branch_addition_rounding_norm=norm(capture["patched"]-(ordinary+delta)),
                        delta_to_ordinary_ratio=ratio(norm(delta), ordinary_norm),
                        zero_to_ordinary_ratio=ratio(norm(zero), ordinary_norm),
                        read_dependent_to_ordinary_ratio=ratio(norm(remainder), ordinary_norm),
                        zero_read_cosine=cosine(zero, remainder), delta_ordinary_cosine=cosine(delta, ordinary),
                        zero_delta_cosine=cosine(zero, full), read_delta_cosine=cosine(remainder, full),
                        model_seconds=elapsed, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                    )
                    rows.append(row)
                    print(json.dumps(row, allow_nan=False), flush=True)
                    del output, inputs, logits
                    capture.clear()
            return dict(source_run=None if info is None else info["source_run"],
                        checkpoint=None if info is None else info["checkpoint"],
                        checkpoint_sha256=None if info is None else info["checkpoint_sha256"],
                        selected_epoch=selected_epoch, selected_step=selected_step,
                        rows=rows)
        finally:
            for handle in handles:
                handle.remove()
            if lora is not None:
                for handle in lora.handles:
                    handle.remove()
                lora.params.clear()
                lora.handles.clear()
            if branch is not None:
                branch.remove()
            capture.clear()

    for arm in (*ARMS, "base"):
        print(f"Inspecting {arm}", flush=True)
        report["models"][arm] = inspect_arm(arm)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        report["models"][arm]["cuda_allocated_after_cleanup"] = torch.cuda.memory_allocated()
        write_json(outdir / "inspection.json", report)
    report["complete"] = True
    write_json(outdir / "inspection.json", report)
    index = outdir.parent / "INDEX.md"
    if not index.exists():
        index.write_text("# V1 checkpoint inspections\n\n")
    with index.open("a") as stream:
        stream.write(f"- [{run_id}]({run_id}/inspection.json): eight models, four profile examples each; algebraic diagnostic only.\n")
    print(f"Completed: {outdir / 'inspection.json'}", flush=True)


if __name__ == "__main__":
    main()
