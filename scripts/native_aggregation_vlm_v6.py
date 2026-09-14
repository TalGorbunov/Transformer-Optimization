"""V6 training-only local teacher alignment; unchanged V6 SUM at inference.

Copied from frozen V6; all original sources remain unchanged. The aligned arm
propagates the same auxiliary KL that the control learns from detached messages.
Only CPU --check-schedule may execute without a Slurm GPU allocation.
"""
from __future__ import annotations
from collections import Counter
import copy
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
from scripts.native_aggregation import (
    DATA_BASE, jsonable_config, parse_answer, parser as text_parser,
    validate_args as validate_legacy_args, write_json,
)
from scripts.native_aggregation_vlm_v4 import (
    validate_schedule, load_schedule, slot_order, initial_parameter_hashes,
    sample_metadata, object_sha256, file_sha256,
)

DATASET = DATA_BASE / "v4_diversity"
FRESH_DATASET = DATA_BASE / "v6_fresh"
TEACHER_INDEX = DATA_BASE / "v6_local_teacher/teacher_cache.json"
SOURCES = (
    "scripts/native_aggregation_vlm_v6.py", "scripts/profile_native_vision_v5_checks.py", "gnnformer/independent_vision_aggregation.py",
    "scripts/native_aggregation_vlm_v4.py", "scripts/native_aggregation_vlm_v2.py",
    "scripts/native_aggregation.py", "gnnformer/runtime.py", "gnnformer/data.py",
    "gnnformer/carriers.py", "gnnformer/constants.py",
    "slurm/native_aggregation_vision_v6_check.sbatch",
    "slurm/native_aggregation_vision_v6_profile.sbatch",
    "slurm/native_aggregation_vision_v6_main.sbatch",
    "tests/test_independent_vision_aggregation.py",
    "gnnformer/local_evidence_distillation.py", "tests/test_local_evidence_distillation.py",
    "scripts/native_aggregation_vlm_v5.py", "scripts/probe_native_vision_v2_prefix.py",
)


def source_hashes():
    return {name: file_sha256(REPO_ROOT / name) for name in SOURCES}


def validate_args(a):
    legacy = copy.copy(a)
    legacy.gold_max = 8
    legacy.arm = "global"
    validate_legacy_args(legacy)
    if a.arm != "independent" or a.seed not in (6, 7):
        raise SystemExit("V6 fixes independent memory and training seeds6/7")
    expected = dict(model="Qwen/Qwen2.5-VL-7B-Instruct", layer_index=14, rank=96,
                    lora_layers=4, lora_rank=8, lora_alpha=16.0, lr=0.001,
                    lr_lora=0.0001, max_grad_norm=1.0, accumulation=4, resize=392,
                    max_new_tokens=4, max_seq_tokens=16000, gold_max=16, eval_modes="all")
    if any(getattr(a, key) != value for key, value in expected.items()):
        raise SystemExit("V6 model/training/decoding differs from registration")
    if a.dataset_root.resolve() != DATASET or a.center_messages:
        raise SystemExit("V6 fixes dataset and internal centered operator, no legacy centering flag")
    if a.max_steps or a.gradient_checkpointing or a.test_epochs or a.eval_only or a.checkpoint:
        raise SystemExit("V6 requires complete training with ordinary non-checkpointed forwards")
    if a.epochs != (2 if a.profile else 9) or a.train_ns != [8,16] or a.dev_ns != [8,16] or a.eval_ns != [16,32,64]:
        raise SystemExit("V6 complete main/profile schedule differs")
    if (a.limit_dev,a.limit_eval,a.limit_count) != ((2,2,2) if a.profile else (36,108,64)):
        raise SystemExit("V6 evaluation cell counts differ")
    names = ("profile_manifest.json","profile_count_manifest.json","profile_schedule.json") if a.profile else ("main_manifest.json","count_manifest.json","schedule.json")
    if any(getattr(a,field).resolve() != DATASET / name for field,name in zip(("manifest","count_manifest","schedule"),names)):
        raise SystemExit("V6 requires the exact immutable V4 manifests/schedule")
    test_root = DATASET if a.profile else FRESH_DATASET
    test_names = ("profile_manifest.json", "profile_count_manifest.json") if a.profile else ("main_manifest.json", "count_manifest.json")
    if any(getattr(a,field).resolve() != test_root/name for field,name in
           zip(("test_manifest", "test_count_manifest"), test_names)):
        raise SystemExit("V6 requires frozen fresh main tests or original software-profile tests")
    if a.teacher_index.resolve() != TEACHER_INDEX or a.data_seed != 20260914:
        raise SystemExit("V6 teacher index/training data seed differs")


def parser():
    p=text_parser()
    p.description=__doc__
    next(action for action in p._actions if action.dest=="arm").choices=("independent",)
    p.add_argument("--condition",choices=("control","aligned"),required=True)
    p.add_argument("--count-manifest",type=Path,default=DATASET/"count_manifest.json")
    p.add_argument("--test-manifest",type=Path,default=FRESH_DATASET/"main_manifest.json")
    p.add_argument("--test-count-manifest",type=Path,default=FRESH_DATASET/"count_manifest.json")
    p.add_argument("--teacher-index",type=Path,default=TEACHER_INDEX)
    p.add_argument("--schedule",type=Path,default=DATASET/"schedule.json")
    p.add_argument("--source-ledger",type=Path)
    p.add_argument("--profile",action="store_true")
    p.add_argument("--check-schedule",action="store_true")
    p.add_argument("--resize",type=int,default=392)
    p.add_argument("--limit-count",type=int,default=64)
    p.set_defaults(model="Qwen/Qwen2.5-VL-7B-Instruct",arm="independent",seed=6,
                   dataset_root=DATASET,manifest=DATASET/"main_manifest.json",
                   layer_index=14,rank=96,train_ns=[8,16],dev_ns=[8,16],eval_ns=[16,32,64],
                   limit_train=90,limit_dev=36,limit_eval=108,gold_max=16,epochs=9,
                   lr=0.001,lr_lora=0.0001,lora_layers=4,lora_rank=8,lora_alpha=16.0,
                   max_seq_tokens=16000,max_new_tokens=4,accumulation=4,max_grad_norm=1.0,
                   data_seed=20260914,eval_modes="all",output=REPO_ROOT/"outputs/native_aggregation_vlm/v6")
    return p


def load_teacher_index(path):
    """Consume cached probabilities only; semantic teacher-audit labels are never read."""
    raw = path.read_bytes()
    cache = json.loads(raw)
    if cache.get("schema_version") != 1 or cache.get("passed_quality_gate") is not True:
        raise ValueError("V6 cannot train without the complete passing teacher-quality gate")
    plan_path = Path(cache["plan_file"])
    if not plan_path.resolve().is_relative_to(DATA_BASE / "v6_local_teacher"):
        raise ValueError("Teacher plan outside registered data root")
    plan_raw = plan_path.read_bytes()
    if hashlib.sha256(plan_raw).hexdigest() != cache["plan_sha256"]:
        raise ValueError("Teacher plan hash differs")
    plan = json.loads(plan_raw)
    if path.with_suffix(".sha256").is_file() and path.with_suffix(".sha256").read_text().strip() != hashlib.sha256(raw).hexdigest():
        raise ValueError("Teacher index sidecar hash differs")
    if cache.get("quality_gate", {}).get("passed") is not True:
        raise ValueError("Teacher quality-gate details do not confirm eligibility")
    for field in ("model", "runtime", "processor", "image_processor_settings", "resize", "quantization", "attention", "temperature", "source_files", "source_sha256"):
        if cache.get(field) != plan.get(field):
            raise ValueError(f"Teacher cache/plan {field} mismatch")
    for name,expected in (("training_manifest", DATASET/"main_manifest.json"), ("training_schedule", DATASET/"schedule.json")):
        binding = cache["source_files"][name]
        if Path(binding["path"]).resolve() != expected or binding["sha256"] != file_sha256(expected):
            raise ValueError("Teacher was not built from the exact original V4 training inputs")
    for name in ("gnnformer/runtime.py", "gnnformer/data.py"):
        if cache["source_sha256"].get(name) != file_sha256(REPO_ROOT/name):
            raise ValueError("Teacher/native runtime or count-prompt source differs")
    targets, scenes = cache.get("targets", {}), cache.get("scenes", {})
    if len(targets) != 9980 or len(scenes) != 1540:
        raise ValueError("Teacher must retain all9980 pairs and1540 original scenes")
    training = json.loads((DATASET / "main_manifest.json").read_text())
    records = [row for n in (8,16) for row in training["splits"][f"train_N{n}"]["samples"]]
    if set(scenes) != {row["sid"] for row in records}:
        raise ValueError("Teacher contains missing/extra training scenes")
    occurrences, used = 0, set()
    for row in records:
        ids = scenes[row["sid"]]
        if len(ids) != row["n_frames"]:
            raise ValueError("Teacher frame order/count differs")
        question_sha = object_sha256(row["question"])
        for image, pair_id in zip(row["image_files"], ids):
            target = targets[pair_id]
            if (target["image_sha256"] != image["sha256"] or target["question_sha256"] != question_sha
                    or target["question"] != row["question"]):
                raise ValueError("Teacher image/question correspondence differs")
            p = target["probabilities"]
            if (len(p) != 3 or any(not math.isfinite(x) or x < 0 or x > 1 for x in p)
                    or abs(sum(p)-1) > 2e-6):
                raise ValueError("Invalid full-vocabulary teacher probability distribution")
            occurrences += 1
            used.add(pair_id)
    if occurrences != 18800 or used != set(targets):
        raise ValueError("Teacher occurrence/target coverage differs")
    binding = dict(index_path=str(path), index_sha256=hashlib.sha256(raw).hexdigest(),
                   plan_path=str(plan_path), plan_sha256=cache["plan_sha256"],
                   targets=len(targets), scenes=len(scenes), original_frame_occurrences=occurrences,
                   passed_quality_gate=True)
    return raw, cache, plan_raw, plan, binding


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
    if not a.source_ledger or json.loads(a.source_ledger.read_text()) != source_hashes():
        raise ValueError("V6 CPU-frozen source ledger differs")
    teacher_raw, teacher, teacher_plan_raw, teacher_plan, teacher_binding = load_teacher_index(a.teacher_index)
    # No torch, processor, or image loading happens before the Slurm guard.
    import torch
    import torch.nn.functional as F
    from transformers import __version__ as transformers_version
    from gnnformer.carriers import attach_lora
    from gnnformer.independent_vision_aggregation import attach_independent_vision_aggregation
    from gnnformer.local_evidence_distillation import LocalEvidenceHead, LastPromptMessageObserver, local_teacher_kl
    from gnnformer.data import build_count_prompt, build_prompt_inputs, iter_sample_dirs, load_mmred_sample
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
    checkpoint_dir = a.checkpoint_root / "native_aggregation_vlm_v6" / run_id
    code_dir = outdir / "code"
    code_dir.mkdir()
    code_hashes = {}
    for name in SOURCES:
        source = (REPO_ROOT / name).read_bytes()
        code_hashes[name] = hashlib.sha256(source).hexdigest()
        (code_dir / name.replace("/", "_")).write_bytes(source)
    (outdir / "source_hashes.json").write_bytes(a.source_ledger.read_bytes())
    (outdir / "teacher_cache.json").write_bytes(teacher_raw)
    (outdir / "teacher_plan.json").write_bytes(teacher_plan_raw)
    write_json(outdir / "teacher_binding.json", teacher_binding)
    config = jsonable_config(a)
    config.update(run_id=run_id, slurm_job_id=os.environ["SLURM_JOB_ID"],
                  code_sha256=code_hashes, torch_version=str(torch.__version__),
                  transformers_version=str(transformers_version),
                  quantization="load_runtime(use_4bit=True): nf4, double quantization, bf16 compute",
                  supervision="Native final-answer/EOS CE plus training-only mean-image teacher KL at last original prompt query",
                  prompt="Images first, then canonical build_count_prompt; normal processor chat template",
                  image_encoding="Standard VLM forward on pixel_values; no precomputed visual features",
                  decoding="Ordinary greedy cached model.generate, unrestricted vocabulary, repetition_penalty=1.0",
                  generation_policy=dict(do_sample=False, repetition_penalty=1.0,
                                         max_new_tokens=a.max_new_tokens, output_logits=True),
                  checkpoint_loading="Selected branch/LoRA weights; all405 training updates share one optimizer",
                  training_schedule_condition="refresh", test_status="software_profile" if a.profile else "fresh_V6_test_after_dev_selection",
                  fresh_test_seed=None if a.profile else 20260919, teacher_binding=teacher_binding,
                  auxiliary_optimizer=dict(lr=.001, weight_decay=0.0, max_grad_norm=1.0, separate=True),
                  interpretation="Training-only auxiliary objective contrast; identical deployed V5 SUM operator, no reasoning-composition claim")
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
    import importlib.metadata
    from huggingface_hub import try_to_load_from_cache
    from scripts.probe_native_vision_v2_prefix import fingerprint
    expected_model = teacher["model"]
    cached_config = try_to_load_from_cache(a.model, "config.json")
    if not isinstance(cached_config, str) or Path(cached_config).parent.resolve() != Path(expected_model["path"]).resolve():
        raise ValueError("Training resolves a different frozen Qwen snapshot than the teacher")
    model_directory = Path(expected_model["path"])
    for name,sha in expected_model["metadata_sha256"].items():
        if file_sha256(model_directory/name) != sha:
            raise ValueError("Frozen model metadata changed since teacher generation")
    for name,record in expected_model["weight_shard_stat"].items():
        weight_path=model_directory/name; stat=weight_path.stat()
        actual=dict(path=str(weight_path.resolve()),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,inode=stat.st_ino)
        if actual != record:
            raise ValueError("Frozen model weight shard identity changed")
    installed_runtime = dict(torch_version=str(torch.__version__), transformers_version=str(transformers_version),
                             bitsandbytes_version=importlib.metadata.version("bitsandbytes"))
    if teacher["runtime"] != installed_runtime:
        raise ValueError("Teacher/native training runtime versions differ")
    runtime = load_runtime(a.model, use_4bit=True, attn_implementation="sdpa", device_map="cuda")
    model, processor, tokenizer = runtime.model, runtime.processor, runtime.tokenizer
    if teacher["processor"] != fingerprint(processor, str(transformers_version)):
        raise ValueError("Teacher/native processor or tokenizer differs")
    if (teacher["resize"] != a.resize or teacher["quantization"] != "nf4_double_bf16"
            or teacher["attention"] != "sdpa" or teacher["temperature"] != 1):
        raise ValueError("Teacher/native preprocessing or probability policy differs")
    config.update(teacher_runtime=installed_runtime, teacher_processor=teacher["processor"],
                  frozen_model_identity=expected_model)
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
    branch = attach_independent_vision_aggregation(
        model, layer_index=layer_index, rank=a.rank, merge="sum")
    branch.mode = "all"
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    lora = attach_lora(layers, len(layers) - a.lora_layers,
                      rank=a.lora_rank, alpha=a.lora_alpha, device=runtime.device)
    def rng_hashes():
        return dict(cpu=hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
                    cuda=[hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()
                          for state in torch.cuda.get_rng_state_all()])
    rng_before = rng_hashes()
    auxiliary_head = LocalEvidenceHead(a.rank, seed=a.seed, device=runtime.device)
    rng_after = rng_hashes()
    if rng_before != rng_after:
        raise ValueError("Auxiliary initialization consumed the native RNG stream")
    auxiliary_parameters = list(auxiliary_head.parameters())
    if sum(p.numel() for p in auxiliary_parameters) != 291:
        raise ValueError("Auxiliary head parameter count differs")
    auxiliary_initialization = dict(seed=a.seed, stream_seed=a.seed+1000003, weight_std=.01,
        bias_zero=True, parameters=291, main_rng_before=rng_before, main_rng_after=rng_after,
        main_rng_unchanged=True, state_sha256={name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                                            for name,value in auxiliary_head.state_dict().items()})
    write_json(outdir / "auxiliary_initialization.json", auxiliary_initialization)
    observer = LastPromptMessageObserver(branch)
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
                      total_trainable=sum(p.numel() for p in trainable),
                      auxiliary_trainable=291, total_trainable_including_auxiliary=sum(p.numel() for p in trainable)+291)
    if len(trainable) != 37 or {id(p) for p in trainable} & {id(p) for p in auxiliary_parameters}:
        raise ValueError("Native37 parameters and auxiliary2 tensors must be disjoint")
    image_settings = {name: getattr(processor.image_processor, name, None)
                      for name in ("size", "min_pixels", "max_pixels", "patch_size",
                                   "temporal_patch_size", "merge_size")}
    if image_settings != teacher["image_processor_settings"]:
        raise ValueError("Teacher/native image processor settings differ")
    config.update(parameters=parameters, resolved_layer_index=layer_index,
                  resolved_lora_start=len(layers) - a.lora_layers,
                  image_processor_settings=image_settings, gpu=torch.cuda.get_device_name(0))
    write_json(outdir / "config.json", config)
    log(dict(parameters=parameters, layer=layer_index, image_settings=image_settings))
    operator = dict(variant="independent_visual_memory", merge="sum",
                    memory="ordinary_final_visual_output_after_window_restoration",
                    normalized_inputs="fixed_tokenwise_RMS_eps1e-6", centered_messages=True,
                    keys_values="shared_learned_projection", language_only=True,
                    visibility="only_complete_images_before_each_query")
    distillation = dict(condition=a.condition, rank=96, classes=3, parameters=291,
                        temperature=1.0, weight=1.0, detach_messages=a.condition == "control",
                        teacher_index_sha256=teacher_binding["index_sha256"],
                        query="last_original_prompt_token", deployed=False)
    architecture = dict(protocol="vision_v6_local_teacher", model=a.model, arm=a.arm,
                        layer_index=layer_index, rank=a.rank, lora_layers=a.lora_layers,
                        lora_rank=a.lora_rank, lora_alpha=a.lora_alpha, resize=a.resize,
                        quantization="nf4", operator=operator, local_distillation=distillation)
    initialization = initial_parameter_hashes(branch, lora)
    write_json(outdir / "initialization.json", initialization)
    config.update(architecture=architecture,
                  initial_parameter_sha256=initialization["combined_sha256"],
                  branch_operator=operator, local_distillation=distillation,
                  auxiliary_initialization=auxiliary_initialization)
    write_json(outdir / "config.json", config)

    def restore(path: Path | str) -> None:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["architecture"] != architecture:
            raise ValueError("Checkpoint architecture differs from the current vision configuration")
        auxiliary_head.load_state_dict(saved["auxiliary_head"], strict=True)
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
    staged_tests = {}
    for field, output_name, allowed_root in (
            ("test_manifest", "staged_test_manifest.json", DATASET if a.profile else FRESH_DATASET),
            ("test_count_manifest", "staged_test_count_manifest.json", DATASET if a.profile else FRESH_DATASET)):
        path = getattr(a, field)
        raw = path.read_bytes()
        content = json.loads(raw)
        if content.get("schema_version") != 1 or Path(content["dataset_root"]).resolve() != allowed_root:
            raise ValueError("Test manifest schema/root differs")
        staged_tests[field] = content
        config[field + "_sha256"] = hashlib.sha256(raw).hexdigest()
        (outdir / output_name).write_bytes(raw)
    write_json(outdir / "config.json", config)
    selected_paths = set()

    def load_split(ns, split, limit, source=None, family="length"):
        source = staged if source is None else source
        sets = {}
        for n in ns:
            root = Path(source["dataset_root"]) / "mmred_vfiltered" / f"seq_len_{n}" / split
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
    test_sets = load_split(a.eval_ns, "test", a.limit_eval, source=staged_tests["test_manifest"])
    count_source = staged_tests["test_count_manifest"]
    count_ns = sorted(int(key.split("_N")[1]) for key in count_source["splits"] if key.startswith("test_N"))
    if count_ns != [32,64]:
        raise ValueError("V6 count-test cells differ")
    test_sets.update(load_split(count_ns, "test", a.limit_count, source=count_source, family="unseen_count"))
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

    auxiliary_gradient_route = None

    def answer_loss(record):
        nonlocal auxiliary_gradient_route
        inputs = prepare(record)
        prompt_length = inputs["input_ids"].shape[1]
        target = torch.tensor([record["target_ids"]], device=runtime.device)
        inputs["input_ids"] = torch.cat((inputs["input_ids"], target[:, :-1]), dim=1)
        if "attention_mask" in inputs:
            inputs["attention_mask"] = torch.cat((inputs["attention_mask"], torch.ones_like(target[:, :-1])), dim=1)
        if "token_type_ids" in inputs:
            inputs["token_type_ids"] = torch.cat((inputs["token_type_ids"], torch.zeros_like(target[:, :-1])), dim=1)
        observer.begin(prompt_length=prompt_length, total_length=inputs["input_ids"].shape[1],
                       expected_images=record["n_frames"])
        try:
            output = model(**inputs, use_cache=False, logits_to_keep=target.shape[1])
            messages, observed = observer.consume()
        except BaseException:
            observer.cancel()
            raise
        if output.logits.shape[1] != target.shape[1]:
            raise RuntimeError("VLM does not honor logits_to_keep")
        count_loss = F.cross_entropy(output.logits[0].float(), target[0])
        pair_ids = teacher["scenes"][record["sid"]]
        probabilities = torch.tensor([teacher["targets"][pair]["probabilities"] for pair in pair_ids],
                                     dtype=torch.float32, device=messages.device)
        auxiliary_loss, local_logits = local_teacher_kl(auxiliary_head, messages, probabilities,
                                                        detach_messages=a.condition == "control")
        if auxiliary_gradient_route is None:
            gradients = torch.autograd.grad(auxiliary_loss, trainable, allow_unused=True, retain_graph=True)
            connected = [index for index,grad in enumerate(gradients) if grad is not None]
            if a.condition == "control" and connected:
                raise ValueError("Detached auxiliary objective reached native parameters")
            if a.condition == "aligned" and connected != [0,1,2,3]:
                raise ValueError("Aligned auxiliary must reach query/memory/read only, not up/upper LoRA")
            auxiliary_gradient_route = dict(condition=a.condition, native_parameter_order="branch5_then_upper_lora32",
                connected_native_indices=connected,
                native_gradient_norms=[None if grad is None else float(grad.detach().float().norm()) for grad in gradients],
                detached_control_native_gradient_is_structurally_zero=a.condition == "control",
                count_only_equivalence="Exact gradient identity tested on CPU; detach yields no native auxiliary graph")
            write_json(outdir / "auxiliary_gradient_route.json", auxiliary_gradient_route)
            del gradients
        detached = messages.detach()
        observed.update(query_position=prompt_length-1, all_images_visible=True,
                        message_mean_norm=float(detached.norm(dim=-1).mean()),
                        student_teacher_argmax_agreement=float((local_logits.detach().argmax(-1) == probabilities.argmax(-1)).float().mean()),
                        teacher_pair_ids_sha256=object_sha256(pair_ids),
                        student_mean_probabilities=F.softmax(local_logits.detach().float(), -1).mean(0).cpu().tolist())
        return count_loss, auxiliary_loss, observed

    diagnostic_dir = DATA_BASE / "v6_diagnostics" / run_id

    @torch.no_grad()
    def evaluate(sets, tag, mode="all"):
        model.eval()
        if observer.expected is not None or observer.messages is not None:
            raise ValueError("Training observer must be inactive during generation")
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
                captured_prefill = []
                capture_handle = None
                if tag == "test":
                    branch.capture_last_query_messages = True
                    def capture_prefill(module, positional, output):
                        if not captured_prefill:
                            value = branch.export_last_query_diagnostics(cpu=False)
                            if value is None:
                                raise ValueError("No final-prompt branch messages were captured")
                            captured_prefill.append(value)
                    capture_handle = model.register_forward_hook(capture_prefill)
                try:
                    generated = model.generate(
                        **inputs, use_cache=True, do_sample=False, max_new_tokens=a.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                        return_dict_in_generate=True, output_logits=True, output_scores=False,
                        temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                    )
                finally:
                    if capture_handle is not None:
                        capture_handle.remove()
                    branch.capture_last_query_messages = False
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
                diagnostic_metadata = {}
                if tag == "test":
                    if len(captured_prefill) != 1:
                        raise ValueError("Expected exactly one first-prefill diagnostic")
                    diagnostic = {key: value.detach().cpu() for key,value in captured_prefill[0].items()}
                    if (diagnostic['frame_messages'].shape != (n,a.rank)
                            or not bool(diagnostic['visible'].all())
                            or int(diagnostic['query_position']) != length-1):
                        raise ValueError("Captured diagnostic does not describe final prompt and every image")
                    diagnostic_dir.mkdir(parents=True,exist_ok=True)
                    diagnostic_path = diagnostic_dir / f"{record['sid']}.pt"
                    torch.save(diagnostic,diagnostic_path)
                    native_norm=float(diagnostic['native_output_norm'])
                    residual_norm=float(diagnostic['residual'].float().norm())
                    diagnostic_metadata=dict(branch_diagnostics_path=str(diagnostic_path),
                        branch_diagnostics_sha256=file_sha256(diagnostic_path),
                        branch_residual_norm=residual_norm,native_attention_output_norm=native_norm,
                        branch_to_native_attention_norm_ratio=residual_norm/native_norm if native_norm else None)
                rows.append(dict(tag=tag, mode=mode, cell=cell_key,
                                 pair_id=record.get("pair_id"), test_family=record.get("test_family"), path=record["path"], sid=record["sid"],
                                 n_frames=n, gold=record["gold"], prediction=prediction, exact=exact,
                                 output_text=text, generated_token_ids=answer_ids.cpu().tolist(),
                                 prompt_tokens=length, generated_tokens=len(answer_ids),
                                 gold_first_token_nll=nll, preprocessing_seconds=preprocessing,
                                 model_seconds=elapsed, **diagnostic_metadata))
                del generated, inputs, captured_prefill
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
                        lora=None if lora is None else lora.state(),
                        auxiliary_head={k:v.detach().cpu() for k,v in auxiliary_head.state_dict().items()},
                        auxiliary_training=dict(parameters=291, optimizer_step=step, lr=.001, weight_decay=0.0,
                                                max_grad_norm=1.0, detached=a.condition == "control")), temporary)
        temporary.replace(path)
        return path

    profile_checks = {}
    if a.profile:
        from scripts.profile_native_vision_v5_checks import zero_initialization
        profile_checks['zero_initialization'] = zero_initialization(model, branch, prepare, train_sets['train_N8'][0])
        write_json(outdir / "profile_checks.json", profile_checks)
    history = []
    gradient_audit = []
    auxiliary_gradient_audit = []
    selected = str(a.checkpoint) if a.checkpoint else None
    if training_enabled:
        if not trainable:
            raise ValueError("No parameters to train")
        if a.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
        auxiliary_optimizer = torch.optim.AdamW(auxiliary_parameters, lr=.001, weight_decay=0.0)
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
            block_sids = schedule["conditions"]["refresh"][epoch]
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
            total_loss = total_auxiliary_loss = 0.0
            seen = 0
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            for offset in range(0, len(order), a.accumulation):
                batch = order[offset:offset + a.accumulation]
                optimizer.zero_grad(set_to_none=True)
                auxiliary_optimizer.zero_grad(set_to_none=True)
                batch_loss = 0.0
                for record in batch:
                    count_loss, auxiliary_loss, local_diagnostics = answer_loss(record)
                    loss = count_loss + auxiliary_loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite loss: {record['path']}")
                    (loss / len(batch)).backward()
                    value = float(count_loss.detach())
                    auxiliary_value = float(auxiliary_loss.detach())
                    combined_value = float(loss.detach())
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
                                        target_ids=record["target_ids"], answer_token_ce=value,
                                        auxiliary_kl=auxiliary_value, total_loss=combined_value,
                                        local_evidence=local_diagnostics)
                    presentation_records.append(presentation)
                    with (outdir / "presentations.jsonl").open("a") as stream:
                        stream.write(json.dumps(presentation, allow_nan=False) + "\n")
                    total_loss += value
                    total_auxiliary_loss += auxiliary_value
                    batch_loss += combined_value / len(batch)
                    del count_loss, auxiliary_loss, loss
                    seen += 1
                branch_norm = torch.linalg.vector_norm(torch.stack([
                    p.grad.detach().float().norm() for p in branch.parameters() if p.grad is not None]))
                lora_norm = torch.linalg.vector_norm(torch.stack([
                    p.grad.detach().float().norm() for p in lora.parameters() if p.grad is not None]))
                if step == 0:
                    first_gradients = {}
                    for label, module in (("branch", branch), ("lora", lora)):
                        for index, parameter in enumerate(module.parameters()):
                            if parameter.grad is not None:
                                data = parameter.grad.detach().float().cpu().contiguous().numpy().tobytes()
                                first_gradients[f"{label}.{index}"] = hashlib.sha256(data).hexdigest()
                    write_json(outdir / "first_step_gradient_hashes.json", first_gradients)
                norm = torch.nn.utils.clip_grad_norm_(trainable, a.max_grad_norm, error_if_nonfinite=True)
                gradient_audit.append(dict(step=step + 1, branch_preclip_norm=float(branch_norm),
                                           lora_preclip_norm=float(lora_norm), combined_preclip_norm=float(norm),
                                           clipped=bool(float(norm) > a.max_grad_norm)))
                auxiliary_norm = torch.nn.utils.clip_grad_norm_(auxiliary_parameters, 1.0, error_if_nonfinite=True)
                auxiliary_gradient_audit.append(dict(step=step+1, combined_preclip_norm=float(auxiliary_norm),
                                                     clipped=bool(float(auxiliary_norm)>1.0), parameters=291, tensors=2))
                optimizer.step()
                auxiliary_optimizer.step()
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
            auxiliary_state_steps = sorted({int(state["step"].item()) for state in auxiliary_optimizer.state.values()})
            if auxiliary_state_steps != [step] or len(auxiliary_optimizer.state) != 2:
                raise ValueError("Auxiliary Adam state did not persist alongside native optimizer")
            write_json(outdir / "gradient_audit.json", gradient_audit)
            write_json(outdir / "auxiliary_gradient_audit.json", auxiliary_gradient_audit)
            write_json(outdir / "training_slot_layouts.json", slot_layouts)
            train_result = dict(epoch=epoch + 1, block_index=epoch, step=step, n=seen,
                                optimizer_state_steps=optimizer_state_steps,
                                optimizer_state_entries=len(optimizer.state),
                                ordered_slots_sha256=object_sha256(ordered_slots),
                                ordered_sids_sha256=object_sha256([record["sid"] for record in order]),
                                mean_answer_token_ce=total_loss / seen,
                                mean_auxiliary_kl=total_auxiliary_loss/seen,
                                mean_total_loss=(total_loss+total_auxiliary_loss)/seen,
                                auxiliary_optimizer_state_steps=auxiliary_state_steps,
                                auxiliary_optimizer_state_entries=len(auxiliary_optimizer.state),
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
        auxiliary_optimizer.zero_grad(set_to_none=True)
        del optimizer, auxiliary_optimizer
        restore(selected)
        log(f"Selected on in-range development data: {selected}")
    presentation_audit = None
    if training_enabled:
        expected_unique = schedule_audit["distinct_by_condition"]["refresh"]
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
    # The auxiliary observer/head never participates in deployment or test generation.
    if a.profile:
        model.eval()
        software = prepare(train_sets["train_N8"][0])
        length = software["input_ids"].shape[1]
        branch.capture_last_query_messages = True
        observer.begin(prompt_length=length, total_length=length, expected_images=8)
        output = model(**software, use_cache=False, logits_to_keep=1)
        live, observed = observer.consume()
        actual = branch.export_last_query_diagnostics()["frame_messages"]
        difference = live.detach() - actual
        if not torch.allclose(live.detach(), actual, rtol=1e-4, atol=1e-5):
            raise ValueError("Live original-query messages differ from actual branch formula")
        profile_checks["live_messages"] = dict(passed=True, max_abs=float(difference.abs().max()),
            rms=float(difference.square().mean().sqrt()), observation=observed,
            rtol=1e-4, atol=1e-5, native_model_forwards=1)
        del output, live, actual, difference
        branch.capture_last_query_messages = False
        # Match grad mode for exact removal: the live-message proof above needs
        # autograd, whereas these two otherwise identical inference calls do not.
        with torch.no_grad():
            reference_logits = model(**software, use_cache=False, logits_to_keep=1).logits.detach().clone()
        observer.remove()
        with torch.no_grad():
            removed = model(**software, use_cache=False, logits_to_keep=1).logits
        if not torch.equal(reference_logits, removed):
            raise ValueError("Removing the inactive training observer changed native inference")
        profile_checks["observer_removal"] = dict(passed=True, exact_native_logits=True)
        profile_checks["gradient_routing"] = dict(passed=True, **auxiliary_gradient_route)
        profile_checks["rng_preservation"] = dict(passed=True, main_rng_unchanged=True)
        profile_checks["finite_updates_and_selected_restore"] = dict(passed=True, updates=step,
            selected_checkpoint=selected, native_parameters=1762400, auxiliary_parameters=291)
        profile_checks["passed"] = all(item["passed"] for item in profile_checks.values())
        profile_checks["numeric_policy"] = "Native V5 numerical limitation retained; fixed selected-checkpoint audit is separate"
        write_json(outdir / "profile_checks.json", profile_checks)
        del software, reference_logits, removed
    else:
        observer.remove()
    auxiliary_head.requires_grad_(False)
    if file_sha256(a.teacher_index) != teacher_binding["index_sha256"]:
        raise ValueError("Teacher cache changed during training")
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
        teacher_binding=teacher_binding, auxiliary_gradient_route=auxiliary_gradient_route,
        auxiliary_head_removed_from_inference=True,
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
