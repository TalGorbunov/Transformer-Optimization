"""Matched scene-refresh MMReD Vision training experiment (V4), using the ordinary Qwen2.5-VL forward.

Images are resized to 392 x 392, then passed through the existing processor's
chat template: images first, canonical build_count_prompt last. The model's
vision encoder runs inside its standard forward; no image features are cached or
injected manually. Only the final answer and EOS supervise the native vocabulary
head. This new experiment has no established accuracy to reproduce.

All actual runs require a Slurm GPU allocation. Expected dataset layout:
  /mnt/data/gabriele/gnn_transformer/mmred_vfiltered/seq_len_N/{train,dev,test}/SID
Each SID contains qa.txt and 000.png ... . Checkpoints are saved exclusively under
the requested /mnt/ckpts/gabriele/gnn_transformer root. Reports contain metadata,
predictions, and source snapshots, never precomputed vision representations.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# These utilities are intentionally standard-library-only at module import.
from scripts.native_aggregation import (  # noqa: E402
    DATA_BASE,
    arm_configuration,
    compatible_architecture,
    jsonable_config,
    parse_answer,
    parser as text_parser,
    validate_args as validate_legacy_args,
    write_json,
)


V4_ARMS = ("hidden",)


def validate_args(a):
    # Reuse operational guards without relaxing the legacy pilot's count cap.
    import copy
    legacy = copy.copy(a)
    legacy.gold_max = min(a.gold_max, 8)
    validate_legacy_args(legacy)
    if not 0 <= a.gold_max <= 16:
        raise SystemExit("V4 registered support is K<=16")
    if not a.manifest:
        raise SystemExit("V4 requires exact staged manifests")
    if a.count_manifest and (not a.count_manifest.resolve().is_relative_to(DATA_BASE)
                             or not a.count_manifest.is_file()):
        raise SystemExit("Count manifest must exist under DATA_BASE")
    if a.hidden_rank <= 0 or a.middle_rank <= 0:
        raise SystemExit("Control ranks must be positive")
    if a.arm == "middle_lora" and not a.lora_layers:
        raise SystemExit("Middle LoRA control requires the registered upper LoRA")
    if a.arm in ("lora", "middle_lora", "base") and a.eval_modes != "all":
        raise SystemExit("Branch-mode interventions require a branch")

    if a.arm != "hidden" or a.eval_modes != "all":
        raise SystemExit("V4 fixes the hidden-only adapter and ordinary all-token evaluation")
    if a.seed not in (2, 3):
        raise SystemExit("V4 uses fixed training seeds 2 and 3")
    expected = dict(model="Qwen/Qwen2.5-VL-7B-Instruct", layer_index=14, rank=64,
                    hidden_rank=96, lora_layers=4, lora_rank=8, lora_alpha=16.0,
                    lr=0.001, lr_lora=0.0001, max_grad_norm=1.0, accumulation=4,
                    resize=392, max_new_tokens=4, max_seq_tokens=16000)
    if any(getattr(a, field) != value for field, value in expected.items()):
        raise SystemExit("V4 model, optimizer or decoding differs from the fixed protocol")
    if a.max_steps or a.gradient_checkpointing or a.test_epochs:
        raise SystemExit("V4 requires complete blocks and the unchanged V2 training computation")
    if not a.eval_only:
        if a.checkpoint:
            raise SystemExit("V4 training starts from the frozen base, never from a warm start")
        if a.epochs != (2 if a.profile else 9) or a.train_ns != [8, 16] or a.dev_ns != [8, 16]:
            raise SystemExit("V4 requires the complete registered main/profile block schedule")
        if not a.schedule or not a.schedule.is_file() or not a.schedule.resolve().is_relative_to(DATA_BASE):
            raise SystemExit("V4 requires an existing schedule under the user data root")
        if (a.limit_dev, a.limit_eval, a.limit_count) != ((2, 2, 2) if a.profile else (36, 108, 64)):
            raise SystemExit("V4 development/test cell counts differ from the main/profile protocol")
        if a.eval_ns != [16, 32, 64]:
            raise SystemExit("V4 evaluates N16/N32/N64 plus the registered count manifest")


def parser():
    p = text_parser()
    p.description = __doc__
    next(action for action in p._actions if action.dest == "arm").choices = V4_ARMS
    p.add_argument("--count-manifest", type=Path)
    p.add_argument("--condition", choices=("repeat", "refresh"), default="repeat")
    p.add_argument("--schedule", type=Path)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--check-schedule", action="store_true")
    p.add_argument("--hidden-rank", type=int, default=96)
    p.add_argument("--middle-rank", type=int, default=32)
    p.add_argument("--middle-alpha", type=float, default=64.0)
    p.add_argument("--limit-count", type=int, default=64)
    p.set_defaults(model="Qwen/Qwen2.5-VL-7B-Instruct", train_ns=[8, 16],
                   dev_ns=[8, 16], eval_ns=[16, 32, 64], limit_train=90, limit_dev=36,
                   limit_eval=108, gold_max=16, epochs=9, arm="hidden", seed=2, layer_index=14, max_seq_tokens=16000,
                   output=REPO_ROOT / "outputs/native_aggregation_vlm/v4")
    p.add_argument("--resize", type=int, default=392,
                   help="Square PIL resize, matching the existing vision baseline")
    return p


def object_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def validate_schedule(schedule: dict, staged: dict, *, profile: bool = False) -> dict:
    """Pure metadata validation; training content and pair identities never become inputs."""
    blocks, width = (2, 4) if profile else (9, 180)
    counts = {8: 4, 16: 4} if profile else {8: 730, 16: 810}
    if schedule.get("schema_version") != 1 or staged.get("schema_version") != 1:
        raise ValueError("Expected version-1 schedule and data manifests")
    if schedule.get("dataset_root") != staged.get("dataset_root"):
        raise ValueError("Schedule and manifest dataset roots differ")
    conditions = schedule.get("conditions", {})
    if set(conditions) != {"repeat", "refresh"}:
        raise ValueError("Schedule requires exactly repeat and refresh conditions")
    slots = schedule.get("slot_metadata", [])
    if len(slots) != width or [slot.get("slot") for slot in slots] != list(range(width)):
        raise ValueError("Slot metadata must be a complete ordered index")
    indexed = {}
    for n, count in counts.items():
        rows = staged["splits"][f"train_N{n}"]["samples"]
        if len(rows) != count:
            raise ValueError("Unique training cell size differs from registered schedule")
        for row in rows:
            if row["sid"] in indexed or row["n_frames"] != n or row.get("split") != "train":
                raise ValueError("Duplicate training SID or malformed split")
            indexed[row["sid"]] = row
    hashes = [row["content_sha256"] for row in indexed.values()]
    if any(not isinstance(sha, str) or len(sha) != 64 for sha in hashes) or len(set(hashes)) != len(hashes):
        raise ValueError("Training manifest must contain distinct canonical content")
    for condition, schedules in conditions.items():
        if len(schedules) != blocks or any(len(block) != width or len(set(block)) != width for block in schedules):
            raise ValueError("Wrong block count/width or duplicated within-block SID")
        for block in schedules:
            for slot, sid in zip(slots, block):
                if sid not in indexed:
                    raise ValueError("Schedule refers to an absent training SID")
                row = indexed[sid]
                for field in ("n_frames", "gold", "question", "target_character", "target_room"):
                    if row.get(field) != slot.get(field):
                        raise ValueError(f"Block changed fixed slot field {field}")
    original = conditions["repeat"][0]
    if conditions["refresh"][0] != original or any(block != original for block in conditions["repeat"]):
        raise ValueError("Both conditions must share original block zero; repeat cannot change")
    if any(slot.get("original_sid", sid) != sid for slot, sid in zip(slots, original)):
        raise ValueError("Original slot SID differs from block zero")
    saturated = [index for index, slot in enumerate(slots) if slot["n_frames"] == 8 and slot["gold"] == 8]
    if len(saturated) != (0 if profile else 10):
        raise ValueError("Wrong finite-support saturated-slot count")
    if any("saturated" in slot and bool(slot["saturated"]) != (index in saturated)
           for index, slot in enumerate(slots)):
        raise ValueError("Declared saturated slots disagree with N8/K8")
    slot_hist = Counter((slot["n_frames"], slot["gold"]) for slot in slots)
    if not profile and slot_hist != Counter({(n, k): 10 for n in (8, 16) for k in range(9)}):
        raise ValueError("Main slot count/label balance differs")
    if profile and Counter(slot["n_frames"] for slot in slots) != {8: 2, 16: 2}:
        raise ValueError("Profile requires two nonsaturated slots at each length")
    seen = set(original)
    for block in conditions["refresh"][1:]:
        for index, sid in enumerate(block):
            if index in saturated:
                if sid != original[index]:
                    raise ValueError("A deterministic N8/K8 slot changed")
            else:
                if sid in seen:
                    raise ValueError("A refreshed nonsaturated slot reused prior content")
                seen.add(sid)
    if seen != set(indexed) or len(seen) != (8 if profile else 1540):
        raise ValueError("Refresh schedule does not cover the exact unique training set")
    return dict(blocks=blocks, slots=width, saturated_slots=saturated,
                presentations=blocks * width,
                image_frame_presentations=blocks * sum(slot["n_frames"] for slot in slots),
                distinct_by_condition={"repeat": width, "refresh": len(seen)},
                unique_training_cells={str(n): count for n, count in counts.items()},
                schedule_conditions_verified=True)


def load_schedule(a):
    raw = a.schedule.read_bytes()
    schedule = json.loads(raw)
    staged = json.loads(a.manifest.read_bytes())
    if schedule.get("manifest_sha256") != file_sha256(a.manifest):
        raise ValueError("Schedule does not bind the exact main/profile manifest")
    if not a.count_manifest or schedule.get("count_manifest_sha256") != file_sha256(a.count_manifest):
        raise ValueError("Schedule does not bind the exact count manifest")
    sources = schedule.get("source_manifest_sha256", {})
    if not sources:
        raise ValueError("Schedule lacks the frozen source-manifest ledger")
    for name, sha in sources.items():
        path = Path(name)
        if not path.is_absolute() or not path.resolve().is_relative_to(DATA_BASE) or file_sha256(path) != sha:
            raise ValueError("A frozen source manifest changed")
    audit = validate_schedule(schedule, staged, profile=a.profile)
    token_info = schedule.get("prompt_token_audit", {})
    token_path = Path(token_info.get("path", ""))
    if (not token_path.is_absolute() or not token_path.resolve().is_relative_to(DATA_BASE)
            or not token_path.is_file() or file_sha256(token_path) != token_info.get("sha256")
            or token_info.get("all_matched_slot_input_ids_and_grids_equal") is not True):
        raise ValueError("Schedule lacks a valid independent CPU prompt-token audit")
    token_raw = token_path.read_bytes()
    token_records = json.loads(token_raw)["records"]
    train_sids = {row["sid"] for n in (8, 16) for row in staged["splits"][f"train_N{n}"]["samples"]}
    if not train_sids.issubset(token_records):
        raise ValueError("CPU token audit does not cover every selected training SID")
    return raw, schedule, audit, token_raw, token_records


def slot_order(width: int, seed: int, block_index: int) -> list[int]:
    result = list(range(width))
    random.Random(seed + block_index).shuffle(result)
    return result


def initial_parameter_hashes(branch, lora) -> dict:
    """Hash exact initialized trainable tensors before any data-dependent forward."""
    tensors = [("branch." + name, parameter) for name, parameter in sorted(branch.named_parameters())]
    tensors += [(f"lora.{index}.{name}.{suffix}", parameter)
                for (index, name), pair in sorted(lora.params.items())
                for suffix, parameter in zip(("A", "B"), pair)]
    groups = {"branch": hashlib.sha256(), "lora": hashlib.sha256()}
    combined = hashlib.sha256()
    records = []
    for name, parameter in tensors:
        value = parameter.detach().cpu().contiguous()
        header = dict(name=name, shape=list(value.shape), dtype=str(value.dtype))
        data = value.numpy().tobytes()
        payload = json.dumps(header, sort_keys=True).encode() + b"\0" + data
        groups[name.split(".")[0]].update(payload)
        combined.update(payload)
        records.append(dict(**header, sha256=hashlib.sha256(data).hexdigest(), numel=value.numel()))
    return dict(combined_sha256=combined.hexdigest(),
                branch_sha256=groups["branch"].hexdigest(),
                lora_sha256=groups["lora"].hexdigest(), tensors=records)


def sample_metadata(sample_dir: Path, n_frames: int) -> dict[str, Any]:
    """Read only question/answer and frame paths; do not derive evidence labels."""
    source = (sample_dir / "qa.txt").read_text()
    lines = source.splitlines()
    q_index = next((i for i, line in enumerate(lines) if line.strip() == "question:"), -1)
    a_index = next((i for i, line in enumerate(lines) if line.strip() == "answer:"), -1)
    if q_index < 0 or a_index <= q_index:
        raise ValueError(f"Invalid question/answer boundaries in {sample_dir}")
    block = [line.strip() for line in lines[q_index + 1:a_index] if line.strip()]
    state_lines = [line for line in block if line.startswith("{") and line.endswith("}")]
    question_lines = [line for line in block if not (line.startswith("{") and line.endswith("}"))]
    answer = next((line.strip() for line in lines[a_index + 1:] if line.strip()), "")
    if len(state_lines) != n_frames or len(question_lines) != 1 or not answer.isdigit():
        raise ValueError(f"Unexpected metadata or frame count in {sample_dir}")
    frame_paths = [sample_dir / f"{index:03d}.png" for index in range(n_frames)]
    if any(not path.is_file() for path in frame_paths):
        raise FileNotFoundError(f"Missing rendered frame in {sample_dir}")
    return dict(path=str(sample_dir), sid=sample_dir.name, n_frames=n_frames,
                question=question_lines[0], gold=int(answer),
                qa_sha256=hashlib.sha256(source.encode()).hexdigest(),
                image_files=[dict(path=str(path), bytes=path.stat().st_size) for path in frame_paths])


def run(a) -> int:
    validate_args(a)
    schedule_raw = schedule = schedule_audit = token_audit_raw = None
    token_audit_records = {}
    if not a.eval_only:
        schedule_raw, schedule, schedule_audit, token_audit_raw, token_audit_records = load_schedule(a)
    if a.check_schedule:
        if schedule_audit is None:
            raise SystemExit("Schedule-only validation requires a training configuration")
        print(json.dumps(schedule_audit, indent=2))
        return 0
    if a.test_epochs:
        raise SystemExit("Fixed-epoch diagnostics are currently implemented in the text harness only")
    if a.resize <= 0:
        raise SystemExit("--resize must be positive")
    # No torch, processor, or image loading happens before the Slurm guard.
    import torch
    import torch.nn.functional as F
    from transformers import __version__ as transformers_version
    from gnnformer.carriers import attach_lora
    from gnnformer.aggregation_controls import attach_hidden_only_adapter, attach_middle_and_upper_lora
    from gnnformer.data import build_count_prompt, build_prompt_inputs, iter_sample_dirs, load_mmred_sample
    from gnnformer.native_aggregation import attach_native_aggregation
    from gnnformer.runtime import get_layers, load_runtime, move_to_device

    if not torch.cuda.is_available():
        raise SystemExit("A Slurm GPU allocation with CUDA is required")
    torch.set_num_threads(max(1, min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))))
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    run_id = f"{a.condition}_seed{a.seed}_{time.strftime('%Y%m%d_%H%M%S')}_{os.environ['SLURM_JOB_ID']}_{os.getpid()}"
    outdir = a.output / run_id
    outdir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = a.checkpoint_root / "native_aggregation_vlm_v4" / run_id
    code_dir = outdir / "code"
    code_dir.mkdir()
    code_hashes = {}
    for name in ("scripts/native_aggregation_vlm_v4.py", "scripts/native_aggregation_vlm_v2.py", "gnnformer/aggregation_controls.py", "scripts/native_aggregation.py",
                 "gnnformer/native_aggregation.py", "gnnformer/runtime.py",
                 "gnnformer/data.py", "gnnformer/carriers.py"):
        source = (REPO_ROOT / name).read_bytes()
        code_hashes[name] = hashlib.sha256(source).hexdigest()
        (code_dir / name.replace("/", "_")).write_bytes(source)
    config = jsonable_config(a)
    config.update(run_id=run_id, slurm_job_id=os.environ["SLURM_JOB_ID"],
                  code_sha256=code_hashes, torch_version=str(torch.__version__),
                  transformers_version=str(transformers_version),
                  quantization="load_runtime(use_4bit=True): nf4, double quantization, bf16 compute",
                  supervision="Final answer and EOS only, native full-vocabulary cross-entropy",
                  prompt="Images first, then canonical build_count_prompt; normal processor chat template",
                  image_encoding="Standard VLM forward on pixel_values; no precomputed visual features",
                  decoding="Ordinary greedy cached model.generate, unrestricted vocabulary, repetition_penalty=1.0",
                  generation_policy=dict(do_sample=False, repetition_penalty=1.0,
                                         max_new_tokens=a.max_new_tokens, output_logits=True),
                  checkpoint_loading="Weight warm start; optimizer and RNG are not restored")
    write_json(outdir / "config.json", config)
    if schedule_raw is not None:
        (outdir / "schedule.json").write_bytes(schedule_raw)
        (outdir / "staged_prompt_token_audit.json").write_bytes(token_audit_raw)
        config.update(schedule_sha256=hashlib.sha256(schedule_raw).hexdigest(), schedule_audit=schedule_audit,
                      prompt_token_audit=schedule["prompt_token_audit"])
        write_json(outdir / "config.json", config)
    log_file = (outdir / "log.txt").open("a", buffering=1)

    def log(value: Any) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {value}"
        print(line, flush=True)
        log_file.write(line + "\n")

    log(f"Loading {a.model} in nf4; reports {outdir}")
    runtime = load_runtime(a.model, use_4bit=True, attn_implementation="sdpa", device_map="cuda")
    model, processor, tokenizer = runtime.model, runtime.processor, runtime.tokenizer
    # load_runtime selects eval mode but does not itself freeze every weight.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    backbone_parameters = model.num_parameters()
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    layers = get_layers(model)
    layer_index = len(layers) // 2 if a.layer_index == -1 else a.layer_index
    if not 0 <= layer_index < len(layers) or a.lora_layers > len(layers):
        raise ValueError("Invalid branch or LoRA layer range")
    branch = None
    if a.arm == "hidden":
        branch = attach_hidden_only_adapter(model, layer_index=layer_index, rank=a.hidden_rank)
    elif a.arm == "global":
        branch = attach_native_aggregation(
            model, layer_index=layer_index, block_size=a.block_size, rank=a.rank,
            **arm_configuration(a.arm, a.routing_rank),
            query_chunk_size=a.query_chunk_size, center_messages=a.center_messages,
        )
        branch.mode = "all"
    lora = None
    if a.arm != "base" and a.lora_layers:
        torch.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)
        if a.arm == "middle_lora":
            lora = attach_middle_and_upper_lora(
                layers, middle_layer_index=layer_index, upper_layers=a.lora_layers,
                middle_rank=a.middle_rank, middle_alpha=a.middle_alpha,
                upper_rank=a.lora_rank, upper_alpha=a.lora_alpha, device=runtime.device)
        else:
            lora = attach_lora(layers, len(layers) - a.lora_layers,
                               rank=a.lora_rank, alpha=a.lora_alpha, device=runtime.device)
    groups = []
    if branch is not None:
        groups.append(dict(params=list(branch.parameters()), lr=a.lr))
    if lora is not None:
        groups.append(dict(params=list(lora.parameters()), lr=a.lr_lora))
    trainable = [parameter for group in groups for parameter in group["params"]]
    if len({id(parameter) for parameter in trainable}) != len(trainable):
        raise ValueError("Duplicate optimizer parameters")
    parameters = dict(backbone=backbone_parameters,
                      branch=0 if branch is None else sum(p.numel() for p in branch.parameters()),
                      lora=0 if lora is None else lora.num_parameters(),
                      total_trainable=sum(p.numel() for p in trainable))
    image_settings = {name: getattr(processor.image_processor, name, None)
                      for name in ("size", "min_pixels", "max_pixels", "patch_size",
                                   "temporal_patch_size", "merge_size")}
    config.update(parameters=parameters, resolved_layer_index=layer_index,
                  resolved_lora_start=len(layers) - a.lora_layers,
                  image_processor_settings=image_settings, gpu=torch.cuda.get_device_name(0))
    write_json(outdir / "config.json", config)
    log(dict(parameters=parameters, layer=layer_index, image_settings=image_settings))
    operator = arm_configuration(a.arm, a.routing_rank) if a.arm == "global" else {"variant": a.arm}
    architecture = dict(protocol="vision_v4_diversity", hidden_rank=a.hidden_rank,
                        middle_rank=a.middle_rank, middle_alpha=a.middle_alpha, model=a.model, arm=a.arm, layer_index=layer_index,
                        block_size=a.block_size, rank=a.rank, lora_layers=a.lora_layers,
                        lora_rank=a.lora_rank, lora_alpha=a.lora_alpha, resize=a.resize,
                        quantization="nf4", center_messages=a.center_messages,
                        variant=operator.get("variant"), routing_rank=operator.get("routing_rank"))
    initialization = initial_parameter_hashes(branch, lora)
    write_json(outdir / "initialization.json", initialization)
    config.update(architecture=architecture,
                  initial_parameter_sha256=initialization["combined_sha256"],
                  branch_operator=operator)
    write_json(outdir / "config.json", config)

    def restore(path: Path | str) -> None:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["architecture"] != architecture:
            raise ValueError("Checkpoint architecture differs from the current vision configuration")
        if branch is not None:
            branch.load_state_dict(saved["branch"])
        if lora is not None:
            with torch.no_grad():
                for (index, name), (matrix_a, matrix_b) in lora.params.items():
                    saved_a, saved_b = saved["lora"][f"{index}.{name}"]
                    matrix_a.copy_(saved_a.to(matrix_a))
                    matrix_b.copy_(saved_b.to(matrix_b))

    if a.checkpoint:
        restore(a.checkpoint)
        log(f"Restored {a.checkpoint}")
    manifest = {}
    staged = None
    if a.manifest:
        raw_manifest = a.manifest.read_bytes()
        staged = json.loads(raw_manifest)
        if staged.get("schema_version") != 1 or not isinstance(staged.get("splits"), dict):
            raise ValueError("Expected a version-1 staged manifest with a splits object")
        if Path(staged["dataset_root"]).resolve() != a.dataset_root.resolve():
            raise ValueError("Manifest dataset_root differs from --dataset-root")
        if schedule is not None and hashlib.sha256(raw_manifest).hexdigest() != schedule["manifest_sha256"]:
            raise ValueError("Manifest changed after schedule validation")
        config.update(manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
                      sample_selection="Exact manifest lists; no filtering, shuffling, or truncation",
                      manifest_image_verification="Staging SHA256; runtime validates paths, byte sizes, and qa SHA256")
        (outdir / "staged_manifest.json").write_bytes(raw_manifest)
        write_json(outdir / "config.json", config)
    staged_count = None
    if a.count_manifest:
        raw_count = a.count_manifest.read_bytes()
        staged_count = json.loads(raw_count)
        if staged_count.get("schema_version") != 1 or Path(staged_count["dataset_root"]).resolve() != a.dataset_root.resolve():
            raise ValueError("Invalid count manifest schema or dataset root")
        if schedule is not None and hashlib.sha256(raw_count).hexdigest() != schedule["count_manifest_sha256"]:
            raise ValueError("Count manifest changed after schedule validation")
        config["count_manifest_sha256"] = hashlib.sha256(raw_count).hexdigest()
        (outdir / "staged_count_manifest.json").write_bytes(raw_count)
        write_json(outdir / "config.json", config)
    selected_paths = set()

    def load_split(ns, split, limit, source=None, family="length"):
        source = staged if source is None else source
        sets = {}
        for n in ns:
            root = a.dataset_root / "mmred_vfiltered" / f"seq_len_{n}" / split
            if not root.is_dir():
                raise FileNotFoundError(f"Required staged data split absent: {root}")
            key = f"{split}_N{n}"
            declared = None
            if source is not None:
                cell = source["splits"].get(key)
                if not isinstance(cell, dict) or not isinstance(cell.get("samples"), list):
                    raise ValueError(f"Manifest is missing {key}")
                declared = cell["samples"]
                if split == "train":
                    limit = len(declared)
                if len(declared) != limit:
                    raise ValueError(f"Manifest {key} has {len(declared)} samples; requested exactly {limit}")
                directories = [Path(record["path"]) for record in declared]
            else:
                directories = iter_sample_dirs(root)
                random.Random(a.data_seed + n).shuffle(directories)
            records = []
            for index, directory in enumerate(directories):
                resolved = directory.resolve()
                if not resolved.is_relative_to(DATA_BASE) or resolved.parent != root.resolve():
                    raise ValueError(f"Sample is outside its declared split: {directory}")
                record = sample_metadata(directory, n)
                record["split"] = split
                if declared is not None:
                    expected = declared[index]
                    for field in ("sid", "n_frames", "gold", "qa_sha256"):
                        if expected.get(field) != record[field]:
                            raise ValueError(f"Manifest {field} differs for {directory}")
                    if expected.get("split", split) != split:
                        raise ValueError(f"Manifest split differs for {directory}")
                    expected_images = expected.get("image_files", [])
                    if len(expected_images) != n:
                        raise ValueError(f"Manifest image count differs for {directory}")
                    for actual_image, expected_image in zip(record["image_files"], expected_images):
                        if (Path(actual_image["path"]).resolve() != Path(expected_image["path"]).resolve()
                                or actual_image["bytes"] != expected_image.get("bytes")):
                            raise ValueError(f"Manifest image path/size differs for {directory}")
                        if "sha256" in expected_image:
                            actual_image["sha256"] = expected_image["sha256"]
                    for field in ("pair_id", "anchor_id", "test_family", "parent_n_frames", "parent_positions", "anchor_positions", "target_character", "target_room", "content_sha256"):
                        if field in expected:
                            record[field] = expected[field]
                if split in ("train", "dev") and record["gold"] > 8:
                    raise ValueError("Training/development cannot include unseen counts")
                if record["gold"] > a.gold_max:
                    if declared is not None:
                        raise ValueError(f"Manifest sample exceeds --gold-max: {directory}")
                    continue
                if resolved in selected_paths:
                    raise ValueError(f"Duplicate selected sample: {directory}")
                selected_paths.add(resolved)
                record["target_ids"] = tokenizer(str(record["gold"]), add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
                records.append(record)
                if declared is None and len(records) == limit:
                    break
            if len(records) != limit:
                raise ValueError(f"Requested {limit} valid samples, found {len(records)} in {root}")
            cell_key = f"{family}_N{n}" if split == "test" else key
            sets[cell_key] = records
            manifest[cell_key] = dict(
                root=str(root), n=len(records),
                gold_histogram=dict(sorted(Counter(str(record["gold"]) for record in records).items())),
                samples=[{key: value for key, value in record.items() if key != "target_ids"} for record in records],
            )
        return sets

    training_enabled = a.arm != "base" and not a.eval_only and a.epochs > 0
    train_sets = load_split(a.train_ns, "train", a.limit_train) if training_enabled else {}
    dev_sets = load_split(a.dev_ns, "dev", a.limit_dev) if training_enabled else {}
    test_sets = load_split(a.eval_ns, "test", a.limit_eval)
    if staged_count is not None:
        count_ns = sorted(int(key.split("_N")[1]) for key in staged_count["splits"] if key.startswith("test_N"))
        test_sets.update(load_split(count_ns, "test", a.limit_count, source=staged_count, family="unseen_count"))
    write_json(outdir / "data_manifest.json", manifest)
    log({key: dict(n=value["n"], gold_histogram=value["gold_histogram"]) for key, value in manifest.items()})
    layout_rows = {}
    slot_layouts = {}
    presentation_records = []
    training_order_records = []

    def prepare(record):
        sid, original, question, _states, answer = load_mmred_sample(Path(record["path"]))
        if sid != record["sid"] or question != record["question"] or int(answer) != record["gold"]:
            raise ValueError("Sample metadata changed after selection")
        if len(original) != record["n_frames"]:
            raise ValueError("Sample frame count changed after selection")
        resized = []
        try:
            resized = [frame.resize((a.resize, a.resize)) for frame in original]
            inputs = build_prompt_inputs(processor, resized, build_count_prompt(question, len(resized)))
        finally:
            for frame in original + resized:
                frame.close()
        length = inputs["input_ids"].shape[1]
        if length + len(record["target_ids"]) > a.max_seq_tokens:
            raise ValueError(f"Sample {sid} exceeds --max-seq-tokens; no truncation is permitted")
        grids = inputs.get("image_grid_thw")
        if grids is None or grids.shape[0] != record["n_frames"]:
            raise ValueError("Processor did not preserve the expected number of images")
        image_tokens = int((inputs["input_ids"] == model.config.image_token_id).sum())
        layout_rows[record["path"]] = dict(path=record["path"], prompt_tokens=length,
                                          input_ids_sha256=object_sha256(inputs["input_ids"][0].tolist()),
                                          image_tokens=image_tokens,
                                          image_grid_thw=grids.tolist())
        if "training_slot" in record:
            checked = token_audit_records[record["sid"]]
            actual = layout_rows[record["path"]]
            if any(actual[key] != checked[key] for key in
                   ("input_ids_sha256", "prompt_tokens", "image_grid_thw")):
                raise ValueError("Actual training input differs from the independent CPU processor audit")
        return move_to_device(inputs, runtime.device)

    def answer_loss(record):
        inputs = prepare(record)
        target = torch.tensor([record["target_ids"]], device=runtime.device)
        inputs["input_ids"] = torch.cat((inputs["input_ids"], target[:, :-1]), dim=1)
        if "attention_mask" in inputs:
            inputs["attention_mask"] = torch.cat((inputs["attention_mask"], torch.ones_like(target[:, :-1])), dim=1)
        if "token_type_ids" in inputs:
            inputs["token_type_ids"] = torch.cat((inputs["token_type_ids"], torch.zeros_like(target[:, :-1])), dim=1)
        output = model(**inputs, use_cache=False, logits_to_keep=target.shape[1])
        if output.logits.shape[1] != target.shape[1]:
            raise RuntimeError("VLM does not honor logits_to_keep")
        return F.cross_entropy(output.logits[0].float(), target[0])

    @torch.no_grad()
    def evaluate(sets, tag, mode="all"):
        model.eval()
        if branch is not None:
            branch.mode = mode
        metrics, rows = [], []
        for cell_key, records in sets.items():
            n = records[0]["n_frames"]
            correct = parsed = generated_count = prompt_count = image_count = 0
            absolute_error = signed_error = nll_total = preprocess_seconds = model_seconds = 0.0
            torch.cuda.reset_peak_memory_stats()
            for record in records:
                torch.cuda.synchronize()
                start = time.monotonic()
                inputs = prepare(record)
                torch.cuda.synchronize()
                preprocessing = time.monotonic() - start
                start = time.monotonic()
                generated = model.generate(
                    **inputs, use_cache=True, do_sample=False, max_new_tokens=a.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True, output_logits=True, output_scores=False,
                    temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                )
                torch.cuda.synchronize()
                elapsed = time.monotonic() - start
                length = inputs["input_ids"].shape[1]
                answer_ids = generated.sequences[0, length:]
                text = tokenizer.decode(answer_ids, skip_special_tokens=True)
                prediction = parse_answer(text)
                exact = prediction == record["gold"]
                nll = -float(F.log_softmax(generated.logits[0][0].float(), dim=-1)[record["target_ids"][0]])
                correct += int(exact)
                parsed += int(prediction is not None)
                if prediction is not None:
                    absolute_error += abs(prediction - record["gold"])
                    signed_error += prediction - record["gold"]
                nll_total += nll
                generated_count += len(answer_ids)
                prompt_count += length
                image_count += layout_rows[record["path"]]["image_tokens"]
                preprocess_seconds += preprocessing
                model_seconds += elapsed
                rows.append(dict(tag=tag, mode=mode, cell=cell_key,
                                 pair_id=record.get("pair_id"), test_family=record.get("test_family"), path=record["path"], sid=record["sid"],
                                 n_frames=n, gold=record["gold"], prediction=prediction, exact=exact,
                                 output_text=text, generated_token_ids=answer_ids.cpu().tolist(),
                                 prompt_tokens=length, generated_tokens=len(answer_ids),
                                 gold_first_token_nll=nll, preprocessing_seconds=preprocessing,
                                 model_seconds=elapsed))
                del generated, inputs
            count = len(records)
            metric = dict(tag=tag, mode=mode, cell=cell_key, n_frames=n, n=count, correct=correct,
                          exact=correct / count, parsed=parsed, parse_rate=parsed / count,
                          mae_parsed=absolute_error / parsed if parsed else None,
                          bias_parsed=signed_error / parsed if parsed else None,
                          gold_first_token_nll=nll_total / count,
                          mean_prompt_tokens=prompt_count / count, mean_image_tokens=image_count / count,
                          mean_generated_tokens=generated_count / count,
                          mean_preprocessing_seconds=preprocess_seconds / count,
                          mean_model_seconds=model_seconds / count,
                          mean_total_seconds=(preprocess_seconds + model_seconds) / count,
                          peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
            metrics.append(metric)
            log(metric)
        if branch is not None:
            branch.mode = "all"
        write_json(outdir / "input_layouts.json", list(layout_rows.values()))
        return metrics, rows

    def save(filename, epoch, step, dev):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / filename
        temporary = path.with_suffix(".tmp")
        torch.save(dict(architecture=architecture, config=config, epoch=epoch, step=step, dev=dev,
                        branch=None if branch is None else {k: v.detach().cpu() for k, v in branch.state_dict().items()},
                        lora=None if lora is None else lora.state()), temporary)
        temporary.replace(path)
        return path

    history = []
    selected = str(a.checkpoint) if a.checkpoint else None
    if training_enabled:
        if not trainable:
            raise ValueError("No parameters to train")
        if a.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
        training = [record for records in train_sets.values() for record in records]
        by_sid = {record["sid"]: record for record in training}
        if len(by_sid) != len(training):
            raise ValueError("Training SIDs are not unique")
        actual_staged = dict(schema_version=1, dataset_root=str(a.dataset_root),
                             splits={key: dict(samples=records) for key, records in train_sets.items()})
        validate_schedule(schedule, actual_staged, profile=a.profile)
        best = (-1.0, -math.inf)
        step = 0
        stop = False
        for epoch in range(a.epochs):
            ordered_slots = slot_order(schedule_audit["slots"], a.seed, epoch)
            block_sids = schedule["conditions"][a.condition][epoch]
            order = [dict(by_sid[block_sids[slot]], training_slot=slot) for slot in ordered_slots]
            training_order_records.append(dict(block_index=epoch, epoch=epoch + 1,
                                               ordered_slot_indices=ordered_slots,
                                               ordered_sids=[record["sid"] for record in order]))
            write_json(outdir / "training_order.json", training_order_records)
            model.train()
            # The frozen vision encoder has no trainable behavior in this pilot.
            model.model.visual.eval()
            if branch is not None:
                branch.mode = "all"
            total_loss = 0.0
            seen = 0
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            for offset in range(0, len(order), a.accumulation):
                batch = order[offset:offset + a.accumulation]
                optimizer.zero_grad(set_to_none=True)
                batch_loss = 0.0
                for record in batch:
                    loss = answer_loss(record)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite loss: {record['path']}")
                    (loss / len(batch)).backward()
                    value = float(loss.detach())
                    slot = record["training_slot"]
                    layout = layout_rows[record["path"]]
                    signature = {key: layout[key] for key in ("prompt_tokens", "input_ids_sha256", "image_tokens", "image_grid_thw")}
                    signature["target_ids"] = record["target_ids"]
                    if slot in slot_layouts and slot_layouts[slot] != signature:
                        raise ValueError("Actual processed tokens changed for a fixed training slot")
                    slot_layouts.setdefault(slot, signature)
                    presentation = dict(block_index=epoch, epoch=epoch + 1, optimizer_step=step + 1,
                                        order_index=seen, slot=slot, sid=record["sid"], path=record["path"],
                                        n_frames=record["n_frames"], gold=record["gold"],
                                        qa_sha256=record["qa_sha256"], content_sha256=record["content_sha256"],
                                        question_sha256=hashlib.sha256(record["question"].encode()).hexdigest(),
                                        prompt_tokens=layout["prompt_tokens"], input_ids_sha256=layout["input_ids_sha256"],
                                        target_ids=record["target_ids"], answer_token_ce=value)
                    presentation_records.append(presentation)
                    with (outdir / "presentations.jsonl").open("a") as stream:
                        stream.write(json.dumps(presentation, allow_nan=False) + "\n")
                    total_loss += value
                    batch_loss += value / len(batch)
                    seen += 1
                norm = torch.nn.utils.clip_grad_norm_(trainable, a.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                step += 1
                if step == 1 or step % a.log_every == 0:
                    log(dict(epoch=epoch + 1, step=step, loss=batch_loss, grad_norm=float(norm)))
                if a.max_steps and step >= a.max_steps:
                    stop = True
                    break
            torch.cuda.synchronize()
            optimizer_state_steps = sorted({int(state["step"].item()) for state in optimizer.state.values()})
            if optimizer_state_steps != [step] or len(optimizer.state) != len(trainable):
                raise ValueError("Adam state did not persist across the registered block schedule")
            write_json(outdir / "training_slot_layouts.json", slot_layouts)
            train_result = dict(epoch=epoch + 1, block_index=epoch, step=step, n=seen,
                                optimizer_state_steps=optimizer_state_steps,
                                optimizer_state_entries=len(optimizer.state),
                                ordered_slots_sha256=object_sha256(ordered_slots),
                                ordered_sids_sha256=object_sha256([record["sid"] for record in order]),
                                mean_answer_token_ce=total_loss / seen,
                                seconds=time.monotonic() - start,
                                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
            log(train_result)
            dev, dev_rows = evaluate(dev_sets, f"dev_epoch{epoch + 1}")
            write_json(outdir / f"dev_epoch{epoch + 1}.json", dict(metrics=dev, predictions=dev_rows))
            count = sum(item["n"] for item in dev)
            score = (sum(item["correct"] for item in dev) / count,
                     -sum(item["gold_first_token_nll"] * item["n"] for item in dev) / count)
            if score > best:
                best = score
                selected = str(save("best.pt", epoch + 1, step, dev))
            save("last.pt", epoch + 1, step, dev)
            history.append(dict(**train_result, dev=dev))
            write_json(outdir / "training.json", history)
            if stop:
                break
        if a.gradient_checkpointing:
            model.gradient_checkpointing_disable()
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        restore(selected)
        log(f"Selected on in-range development data: {selected}")
    presentation_audit = None
    if training_enabled:
        expected_unique = schedule_audit["distinct_by_condition"][a.condition]
        if (len(presentation_records) != schedule_audit["presentations"]
                or sum(row["n_frames"] for row in presentation_records) != schedule_audit["image_frame_presentations"]
                or len({row["sid"] for row in presentation_records}) != expected_unique):
            raise ValueError("Observed training presentations differ from the complete schedule")
        presentation_audit = dict(presentations=len(presentation_records),
                                  image_frame_presentations=sum(row["n_frames"] for row in presentation_records),
                                  unique_training_sids_seen=expected_unique,
                                  presentations_sha256=file_sha256(outdir / "presentations.jsonl"),
                                  training_order_sha256=file_sha256(outdir / "training_order.json"),
                                  training_slot_layouts_sha256=file_sha256(outdir / "training_slot_layouts.json"),
                                  all_observed_slot_token_layouts_equal=True)
        write_json(outdir / "presentation_audit.json", presentation_audit)
    metrics, predictions = [], []
    for mode in a.eval_modes.split("+"):
        current_metrics, current_predictions = evaluate(test_sets, "test", mode)
        metrics.extend(current_metrics)
        predictions.extend(current_predictions)
    write_json(outdir / "predictions.json", predictions)
    write_json(outdir / "summary.json", dict(
        run_id=run_id, arm=a.arm, model=a.model, parameters=parameters,
        selected_checkpoint=selected, training=history, results=metrics,
        condition=a.condition, profile=a.profile, schedule_sha256=config.get("schedule_sha256"),
        initial_parameter_sha256=initialization["combined_sha256"], presentation_audit=presentation_audit,
        metric_notes=dict(
            exact="Every selected example remains in the denominator; unparsable output is incorrect",
            mae_parsed="Conditional on strict whole-integer parsing; report together with parse_rate",
            gold_first_token_nll="Raw native full-vocabulary first-answer-token NLL from output_logits; not whole-answer likelihood",
            timing="Batch one; preprocessing includes image loading/CPU processor/H2D; model includes vision encoding and language prefill/decode",
            scope="Separate paired length cells K0..8 and unseen-count cells K9..16; count extrapolation also changes answer support",
        ),
    ))
    log(f"Completed: {outdir / 'summary.json'}")
    log_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
