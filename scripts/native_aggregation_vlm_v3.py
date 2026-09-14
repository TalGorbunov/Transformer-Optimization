"""Native nonlinear-value placement comparison on clean MMReD Vision (V3), using the ordinary Qwen2.5-VL forward.

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
    jsonable_config,
    parse_answer,
    parser as text_parser,
    validate_args as validate_legacy_args,
    write_json,
)


V3_ARMS = ("lift_pre", "lift_post")


def validate_args(a):
    # Reuse operational guards without relaxing the legacy pilot's count cap.
    import copy
    legacy = copy.copy(a)
    legacy.gold_max = min(a.gold_max, 8)
    validate_legacy_args(legacy)
    if not 0 <= a.gold_max <= 16:
        raise SystemExit("V3 registered support is K<=16")
    if not a.manifest:
        raise SystemExit("V3 requires exact staged manifests")
    if a.count_manifest and (not a.count_manifest.resolve().is_relative_to(DATA_BASE)
                             or not a.count_manifest.is_file()):
        raise SystemExit("Count manifest must exist under DATA_BASE")
    if a.lift_rank <= 0:
        raise SystemExit("Lift rank must be positive")
    if not a.lora_layers:
        raise SystemExit("V3 requires the registered upper LoRA")


def parser():
    p = text_parser()
    p.description = __doc__
    next(action for action in p._actions if action.dest == "arm").choices = V3_ARMS
    p.add_argument("--count-manifest", type=Path)
    p.add_argument("--lift-rank", type=int, default=8)
    p.add_argument("--limit-count", type=int, default=64)
    p.set_defaults(model="Qwen/Qwen2.5-VL-7B-Instruct", train_ns=[8, 16],
                   dev_ns=[8, 16], eval_ns=[16, 32, 64], limit_train=90, limit_dev=36,
                   limit_eval=108, gold_max=16, epochs=9, arm="lift_pre", max_seq_tokens=16000,
                   output=REPO_ROOT / "outputs/native_aggregation_vlm/v3")
    p.add_argument("--resize", type=int, default=392,
                   help="Square PIL resize, matching the existing vision baseline")
    return p


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
    if a.test_epochs:
        raise SystemExit("Fixed-epoch diagnostics are currently implemented in the text harness only")
    if a.resize <= 0:
        raise SystemExit("--resize must be positive")
    # No torch, processor, or image loading happens before the Slurm guard.
    import torch
    import torch.nn.functional as F
    from transformers import __version__ as transformers_version
    from gnnformer.carriers import attach_lora
    from gnnformer.value_lifting import attach_value_lifting
    from gnnformer.data import build_count_prompt, build_prompt_inputs, iter_sample_dirs, load_mmred_sample
    from gnnformer.runtime import get_layers, load_runtime, move_to_device

    if not torch.cuda.is_available():
        raise SystemExit("A Slurm GPU allocation with CUDA is required")
    torch.set_num_threads(max(1, min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))))
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    run_id = f"{a.arm}_seed{a.seed}_{time.strftime('%Y%m%d_%H%M%S')}_{os.environ['SLURM_JOB_ID']}_{os.getpid()}"
    outdir = a.output / run_id
    outdir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = a.checkpoint_root / "native_aggregation_vlm_v3" / run_id
    code_dir = outdir / "code"
    code_dir.mkdir()
    code_hashes = {}
    for name in ("scripts/native_aggregation_vlm_v3.py", "gnnformer/value_lifting.py", "scripts/native_aggregation.py",
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
    placement = "before" if a.arm == "lift_pre" else "after"
    branch = attach_value_lifting(model, layer_index=layer_index, rank=a.rank,
                                  lift_rank=a.lift_rank, placement=placement, activation="silu")
    branch.mode = "all"
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
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
    operator = dict(variant="value_lifting", placement=placement, activation="silu", lift_rank=a.lift_rank,
                    cached_features="Native cached V re-lifted each forward; no transformed-value cache")
    architecture = dict(protocol="vision_v3_value_lifting", model=a.model, arm=a.arm,
                        layer_index=layer_index, rank=a.rank, lift_rank=a.lift_rank,
                        placement=placement, activation="silu", lora_layers=a.lora_layers,
                        lora_rank=a.lora_rank, lora_alpha=a.lora_alpha, resize=a.resize,
                        quantization="nf4")
    config.update(architecture=architecture,
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
                    for field in ("pair_id", "anchor_id", "test_family", "parent_n_frames", "parent_positions", "anchor_positions"):
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

    training_enabled = not a.eval_only and a.epochs > 0
    train_sets = load_split(a.train_ns, "train", a.limit_train) if training_enabled else {}
    dev_sets = load_split(a.dev_ns, "dev", a.limit_dev) if training_enabled else {}
    test_sets = load_split(a.eval_ns, "test", a.limit_eval)
    if staged_count is not None:
        count_ns = sorted(int(key.split("_N")[1]) for key in staged_count["splits"] if key.startswith("test_N"))
        test_sets.update(load_split(count_ns, "test", a.limit_count, source=staged_count, family="unseen_count"))
    write_json(outdir / "data_manifest.json", manifest)
    log({key: dict(n=value["n"], gold_histogram=value["gold_histogram"]) for key, value in manifest.items()})
    layout_rows = {}

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
                                          image_tokens=image_tokens,
                                          image_grid_thw=grids.tolist())
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
        best = (-1.0, -math.inf)
        step = 0
        stop = False
        for epoch in range(a.epochs):
            order = list(training)
            random.Random(a.seed + epoch).shuffle(order)
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
            train_result = dict(epoch=epoch + 1, step=step, n=seen,
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
    metrics, predictions = [], []
    for mode in a.eval_modes.split("+"):
        current_metrics, current_predictions = evaluate(test_sets, "test", mode)
        metrics.extend(current_metrics)
        predictions.extend(current_predictions)
    write_json(outdir / "predictions.json", predictions)
    write_json(outdir / "summary.json", dict(
        run_id=run_id, arm=a.arm, model=a.model, parameters=parameters,
        selected_checkpoint=selected, training=history, results=metrics,
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
