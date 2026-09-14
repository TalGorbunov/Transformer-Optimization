"""Train/evaluate native text aggregation with the existing vocabulary head.

This is the first feasibility harness for docs/paper/NATIVE_AGGREGATION_PROPOSAL.md;
there is no established accuracy to reproduce. Only final answer tokens supervise
the model. Evaluation calls ordinary cached ``generate`` without digit-restricted
decoding. Fixed-token region boundaries are owned by the model adapter, not data.

Actual runs require a Slurm allocation. Example (inside an allocated GPU job):
  python scripts/native_aggregation.py --arm sum --dataset-root /mnt/data/gabriele/gnn_transformer \
    --checkpoint-root /mnt/ckpts/gabriele/gnn_transformer --limit-train 24 --limit-eval 16
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_BASE = Path("/mnt/ckpts/gabriele/gnn_transformer")
DATA_BASE = Path("/mnt/data/gabriele/gnn_transformer")
ARMS = ("base", "lora", "sum", "mean", "post_sum", "post_mean",
        "global", "hierarchical", "linear_sum", "linear_mean")


def arm_configuration(arm: str, routing_rank: int = 8) -> dict[str, Any]:
    """One shared mapping for text, vision, configuration, and checkpoints."""
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    if arm in ("base", "lora"):
        return {}
    variant = ("local_post" if arm.startswith("post_") else
               arm if arm in ("global", "hierarchical") else "local_pre")
    merge = "mean" if arm in ("global", "hierarchical") else arm.split("_")[-1]
    return dict(variant=variant, merge=merge,
                nonlinear=not arm.startswith("linear_"), routing_rank=routing_rank)


def compatible_architecture(saved: dict[str, Any]) -> dict[str, Any]:
    """Fill fields absent from pre-positioning pilot checkpoints, without mutation."""
    result = dict(saved)
    result.setdefault("center_messages", False)
    old_options = arm_configuration(result["arm"])
    result.setdefault("variant", old_options.get("variant"))
    result.setdefault("routing_rank", old_options.get("routing_rank"))
    return result


def integer_list(text: str) -> list[int]:
    values = [int(value) for value in text.split("+")]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("Expected positive integers separated by +")
    return values


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--arm", choices=ARMS, default="sum")
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="Root containing mmred_filtered_train and mmred_filtered")
    p.add_argument("--manifest", type=Path,
                   help="Vision only: exact staged split/sample manifest; no reselection")
    p.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_BASE)
    p.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/native_aggregation")
    p.add_argument("--checkpoint", type=Path,
                   help="Restore adapter/LoRA weights for evaluation or a recorded warm start")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--train-ns", type=integer_list, default=[8, 16, 32])
    p.add_argument("--dev-ns", type=integer_list, default=[8, 32])
    p.add_argument("--eval-ns", type=integer_list, default=[8, 32, 64, 128])
    p.add_argument("--gold-max", type=int, default=8)
    p.add_argument("--limit-train", type=int, default=24, help="Examples per training N")
    p.add_argument("--limit-dev", type=int, default=12, help="Examples per development N")
    p.add_argument("--limit-eval", type=int, default=16, help="Examples per test N")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--accumulation", type=int, default=4,
                   help="Unpadded batch-one forwards per optimizer step")
    p.add_argument("--max-steps", type=int, default=0, help="0 runs all selected epochs")
    p.add_argument("--layer-index", type=int, default=-1, help="-1 selects middle decoder layer")
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--routing-rank", type=int, default=8,
                   help="Per-head hierarchical score correction bottleneck")
    p.add_argument("--query-chunk-size", type=int, default=32)
    p.add_argument("--center-messages", action="store_true", help="Subtract the zero-read response")
    p.add_argument("--test-epochs", type=integer_list, default=[], help="Fixed epochs to evaluate without test selection")
    p.add_argument("--lora-layers", type=int, default=4)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-lora", type=float, default=1e-4)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=4)
    p.add_argument("--max-seq-tokens", type=int, default=12000,
                   help="Fail, rather than truncate, if any selected prompt exceeds this")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--eval-modes", default="all",
                   help="+ separated branch inference modes: all/off/prefill/decode")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-seed", type=int, default=1234)
    p.add_argument("--log-every", type=int, default=5)
    return p


def validate_args(a: argparse.Namespace) -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Run this trainer/evaluator inside a Slurm GPU allocation.")
    for field in ("checkpoint_root", "dataset_root"):
        base = CHECKPOINT_BASE if field == "checkpoint_root" else DATA_BASE
        if not getattr(a, field).resolve().is_relative_to(base):
            raise SystemExit(f"--{field.replace('_', '-')} must be within {base}")
    if a.manifest and not a.manifest.resolve().is_relative_to(DATA_BASE):
        raise SystemExit(f"Manifest reads must use {DATA_BASE}")
    if a.manifest and not a.manifest.is_file():
        raise SystemExit(f"Manifest does not exist: {a.manifest}")
    if a.checkpoint and not a.checkpoint.resolve().is_relative_to(CHECKPOINT_BASE):
        raise SystemExit(f"Checkpoint reads must use {CHECKPOINT_BASE}")
    for field in ("limit_train", "limit_dev", "limit_eval", "accumulation", "rank",
                  "block_size", "query_chunk_size", "routing_rank", "lora_rank", "max_new_tokens",
                  "max_seq_tokens", "log_every"):
        if getattr(a, field) <= 0:
            raise SystemExit(f"{field} must be positive")
    if a.epochs < 0 or a.max_steps < 0 or a.lora_layers < 0 or a.gold_max < 0:
        raise SystemExit("Epochs, maximum steps, LoRA layers, and gold maximum must be nonnegative")
    if any(epoch > a.epochs for epoch in a.test_epochs):
        raise SystemExit("A test epoch exceeds the training duration")
    if max(a.train_ns + a.dev_ns) > 32:
        raise SystemExit("This feasibility protocol restricts training/development to N <= 32")
    if a.gold_max > 8:
        raise SystemExit("This pilot uses K <= 8; register a separate numerical-extrapolation run")
    modes = a.eval_modes.split("+")
    if any(mode not in ("all", "off", "prefill", "decode") for mode in modes):
        raise SystemExit("Unknown --eval-modes value")
    if a.arm in ("base", "lora") and modes != ["all"]:
        raise SystemExit("Branch mode ablations require an aggregation arm")
    if a.center_messages and a.arm not in ("sum", "mean", "linear_sum", "linear_mean"):
        raise SystemExit("--center-messages is restricted to legacy local-pre arms")
    if a.arm == "base" and a.checkpoint:
        raise SystemExit("The base arm does not load trainable checkpoints")
    if a.arm == "lora" and a.lora_layers == 0:
        raise SystemExit("The lora arm requires at least one adapted layer")


def jsonable_config(a: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(a).items()}


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parse_answer(text: str) -> int | None:
    match = re.fullmatch(r"\s*([0-9]+)\s*", text)
    return int(match.group(1)) if match else None


def run(a: argparse.Namespace) -> int:
    validate_args(a)
    if a.manifest:
        raise SystemExit("--manifest is implemented by the vision harness only")
    # Keep --help, source inspection, and syntax checks free of heavy imports.
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "scripts/condmask"))
    import torch
    import torch.nn.functional as F
    from transformers import __version__ as transformers_version
    from gnnformer.carriers import attach_lora
    from gnnformer.native_aggregation import attach_native_aggregation
    from modeling import load_text_model
    from textdata import prep_root

    if not torch.cuda.is_available():
        raise SystemExit("A Slurm GPU allocation with CUDA is required")
    torch.set_num_threads(max(1, min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))))
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    run_id = f"{a.arm}_seed{a.seed}_{time.strftime('%Y%m%d_%H%M%S')}_{os.environ['SLURM_JOB_ID']}_{os.getpid()}"
    outdir = a.output / run_id
    outdir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = a.checkpoint_root / "native_aggregation" / run_id
    config = jsonable_config(a)
    code_dir = outdir / "code"
    code_dir.mkdir()
    code_hashes = {}
    for source_name in ("scripts/native_aggregation.py", "gnnformer/native_aggregation.py"):
        source = (REPO_ROOT / source_name).read_bytes()
        code_hashes[source_name] = hashlib.sha256(source).hexdigest()
        (code_dir / source_name.replace("/", "_")).write_bytes(source)
    config.update(run_id=run_id, slurm_job_id=os.environ["SLURM_JOB_ID"],
                  code_sha256=code_hashes, transformers_version=str(transformers_version),
                  supervision="final answer plus EOS, full-vocabulary cross-entropy",
                  prompt="scripts.condmask.textdata.build_segmented_prompt; raw canonical text",
                  decoding="ordinary greedy model.generate(use_cache=True), no vocabulary restriction",
                  checkpoint_loading="weight warm start; optimizer and RNG are not restored")
    write_json(outdir / "config.json", config)
    log_file = (outdir / "log.txt").open("a", buffering=1)

    def log(message: Any) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        log_file.write(line + "\n")

    log(f"Loading {a.model}; reports {outdir}")
    tokenizer, model = load_text_model(a.model, device="cuda")
    if tokenizer.eos_token_id is None:
        raise ValueError("Native answer training requires a tokenizer EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # load_text_model freezes the base BEFORE either adapter is attached.
    layer_count = len(model.model.layers)
    layer_index = layer_count // 2 if a.layer_index == -1 else a.layer_index
    if not 0 <= layer_index < layer_count:
        raise ValueError(f"Invalid layer {layer_index}; model has {layer_count} decoder layers")
    if a.lora_layers > layer_count:
        raise ValueError("More LoRA layers requested than the model contains")
    branch = None
    if a.arm not in ("base", "lora"):
        branch = attach_native_aggregation(
            model, layer_index=layer_index, block_size=a.block_size, rank=a.rank,
            **arm_configuration(a.arm, a.routing_rank),
            query_chunk_size=a.query_chunk_size, center_messages=a.center_messages,
        )
        branch.mode = "all"
    lora = None
    if a.arm != "base" and a.lora_layers:
        # Branch construction consumes draws; match LoRA initialization across arms.
        torch.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)
        lora = attach_lora(model.model.layers, layer_count - a.lora_layers,
                           rank=a.lora_rank, alpha=a.lora_alpha, device="cuda")
    groups = []
    if branch is not None:
        groups.append({"params": list(branch.parameters()), "lr": a.lr})
    if lora is not None:
        groups.append({"params": list(lora.parameters()), "lr": a.lr_lora})
    trainable = [parameter for group in groups for parameter in group["params"]]
    if len({id(parameter) for parameter in trainable}) != len(trainable):
        raise RuntimeError("A trainable parameter appears in more than one optimizer group")
    parameter_counts = dict(
        backbone=sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad),
        branch=0 if branch is None else sum(parameter.numel() for parameter in branch.parameters()),
        lora=0 if lora is None else lora.num_parameters(),
        total_trainable=sum(parameter.numel() for parameter in trainable),
    )
    config.update(resolved_layer_index=layer_index, resolved_lora_start=layer_count - a.lora_layers,
                  parameters=parameter_counts, torch_version=str(torch.__version__),
                  gpu=torch.cuda.get_device_name(0))
    write_json(outdir / "config.json", config)
    log(f"Parameters {parameter_counts}; branch layer {layer_index}")

    architecture = dict(model=a.model, arm=a.arm, layer_index=layer_index,
                        block_size=a.block_size, rank=a.rank, lora_layers=a.lora_layers,
                        lora_rank=a.lora_rank, lora_alpha=a.lora_alpha,
                        center_messages=a.center_messages,
                        variant=arm_configuration(a.arm, a.routing_rank).get("variant"),
                        routing_rank=arm_configuration(a.arm, a.routing_rank).get("routing_rank"))
    config.update(architecture=architecture,
                  branch_operator=arm_configuration(a.arm, a.routing_rank))
    write_json(outdir / "config.json", config)
    if a.checkpoint:
        checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
        checkpoint["architecture"] = compatible_architecture(checkpoint["architecture"])
        if checkpoint["architecture"] != architecture:
            raise ValueError(f"Checkpoint architecture differs: {checkpoint['architecture']} != {architecture}")
        if branch is not None:
            branch.load_state_dict(checkpoint["branch"])
        if lora is not None:
            state = checkpoint["lora"]
            with torch.no_grad():
                for (index, name), (matrix_a, matrix_b) in lora.params.items():
                    saved_a, saved_b = state[f"{index}.{name}"]
                    matrix_a.copy_(saved_a.to(matrix_a))
                    matrix_b.copy_(saved_b.to(matrix_b))
        log(f"Restored weights from {a.checkpoint}")

    manifest: dict[str, Any] = {}

    def load_split(ns: list[int], split: str, limit: int) -> dict[int, list[dict[str, Any]]]:
        folder = "mmred_filtered" if split == "test" else "mmred_filtered_train"
        sets = {}
        for n in ns:
            root = a.dataset_root / folder / f"seq_len_{n}" / split
            if not root.is_dir():
                raise FileNotFoundError(f"Required data split absent: {root}")
            records = prep_root(tokenizer, root, limit, seed=a.data_seed + n,
                                gold_max=a.gold_max)
            if not records:
                raise ValueError(f"No valid selected samples in {root}")
            if len(records) < limit:
                raise ValueError(f"Requested {limit} valid samples, found {len(records)} in {root}")
            for record in records:
                if record["n_frames"] != n:
                    raise ValueError(f"Unexpected frame count in {root / record['sid']}")
                answer_ids = tokenizer(str(record["gold"]), add_special_tokens=False).input_ids
                if not answer_ids:
                    raise ValueError("Empty tokenized answer")
                record["target_ids"] = answer_ids + [tokenizer.eos_token_id]
                if len(record["ids"]) + len(record["target_ids"]) > a.max_seq_tokens:
                    raise ValueError(f"Selected sample {record['sid']} exceeds --max-seq-tokens; never truncated")
            sets[n] = records
            manifest[f"{split}_N{n}"] = dict(
                root=str(root), n=len(records),
                gold_histogram=dict(sorted(Counter(str(record["gold"]) for record in records).items())),
                min_tokens=min(len(record["ids"]) for record in records),
                max_tokens=max(len(record["ids"]) for record in records),
                samples=[dict(sid=record["sid"], gold=record["gold"],
                              prompt_tokens=len(record["ids"])) for record in records],
            )
        return sets

    training_enabled = a.arm != "base" and not a.eval_only and a.epochs > 0
    train_sets = load_split(a.train_ns, "train", a.limit_train) if training_enabled else {}
    dev_sets = load_split(a.dev_ns, "dev", a.limit_dev) if training_enabled else {}
    test_sets = load_split(a.eval_ns, "test", a.limit_eval)
    write_json(outdir / "data_manifest.json", manifest)
    log({key: {k: v for k, v in entry.items() if k != "samples"} for key, entry in manifest.items()})

    def answer_loss(record: dict[str, Any]):
        target = record["target_ids"]
        ids = torch.tensor([record["ids"] + target[:-1]], device="cuda")
        # Last len(target) causal positions predict precisely the complete answer
        # and EOS. This avoids materializing [all prompt tokens, vocabulary].
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                       use_cache=False, logits_to_keep=len(target))
        if output.logits.shape[1] != len(target):
            raise RuntimeError("Backbone does not honor logits_to_keep")
        return F.cross_entropy(output.logits[0].float(), torch.tensor(target, device="cuda"))

    @torch.no_grad()
    def evaluate(sets: dict[int, list[dict[str, Any]]], tag: str, mode: str = "all"):
        model.eval()
        if branch is not None:
            branch.mode = mode
        metrics, rows = [], []
        for n, records in sets.items():
            exact = valid = generated_tokens = prompt_tokens = 0
            absolute_error = signed_error = first_nll = total_seconds = 0.0
            torch.cuda.reset_peak_memory_stats()
            for record in records:
                ids = torch.tensor([record["ids"]], device="cuda")
                torch.cuda.synchronize()
                start = time.monotonic()
                generated = model.generate(
                    input_ids=ids, attention_mask=torch.ones_like(ids),
                    use_cache=True, do_sample=False, max_new_tokens=a.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True, output_scores=True,
                    temperature=None, top_p=None, top_k=None,
                )
                torch.cuda.synchronize()
                elapsed = time.monotonic() - start
                answer = generated.sequences[0, ids.shape[1]:]
                text = tokenizer.decode(answer, skip_special_tokens=True)
                prediction = parse_answer(text)
                correct = prediction == record["gold"]
                nll = -float(F.log_softmax(generated.scores[0][0].float(), dim=-1)[record["target_ids"][0]])
                exact += int(correct)
                valid += int(prediction is not None)
                if prediction is not None:
                    absolute_error += abs(prediction - record["gold"])
                    signed_error += prediction - record["gold"]
                first_nll += nll
                generated_tokens += len(answer)
                prompt_tokens += ids.shape[1]
                total_seconds += elapsed
                rows.append(dict(tag=tag, mode=mode, n_frames=n, sid=record["sid"],
                                 gold=record["gold"], prediction=prediction, exact=correct,
                                 output_text=text, generated_token_ids=answer.cpu().tolist(),
                                 prompt_tokens=ids.shape[1], generated_tokens=len(answer),
                                 gold_first_token_nll=nll, seconds=elapsed))
            count = len(records)
            result = dict(tag=tag, mode=mode, n_frames=n, n=count, correct=exact,
                          exact=exact / count, parsed=valid, parse_rate=valid / count,
                          mae_parsed=absolute_error / valid if valid else None,
                          bias_parsed=signed_error / valid if valid else None,
                          gold_first_token_nll=first_nll / count,
                          mean_prompt_tokens=prompt_tokens / count,
                          mean_generated_tokens=generated_tokens / count,
                          mean_seconds=total_seconds / count, total_seconds=total_seconds,
                          peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
            metrics.append(result)
            log(result)
        if branch is not None:
            branch.mode = "all"
        return metrics, rows

    def save_checkpoint(filename: str, epoch: int, step: int, dev_metrics: list[dict[str, Any]]) -> Path:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / filename
        temporary = path.with_suffix(".tmp")
        state = dict(architecture=architecture, epoch=epoch, step=step, dev_metrics=dev_metrics,
                     branch=None if branch is None else {key: value.detach().cpu() for key, value in branch.state_dict().items()},
                     lora=None if lora is None else lora.state(), config=config)
        torch.save(state, temporary)
        temporary.replace(path)
        return path

    training_metrics = []
    selected_checkpoint = str(a.checkpoint) if a.checkpoint else None
    if training_enabled:
        if not trainable:
            raise ValueError("Training requested without trainable parameters")
        if a.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
        training = [record for records in train_sets.values() for record in records]
        best_score = (-1.0, -math.inf)
        step = 0
        stop = False
        for epoch in range(a.epochs):
            order = list(training)
            random.Random(a.seed + epoch).shuffle(order)
            model.train()
            if branch is not None:
                branch.mode = "all"
            total_loss = 0.0
            examples_seen = 0
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            for offset in range(0, len(order), a.accumulation):
                batch = order[offset:offset + a.accumulation]
                optimizer.zero_grad(set_to_none=True)
                batch_loss = 0.0
                for record in batch:
                    loss = answer_loss(record)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite training loss for {record['sid']}")
                    (loss / len(batch)).backward()
                    value = float(loss.detach())
                    batch_loss += value / len(batch)
                    total_loss += value
                    examples_seen += 1
                norm = torch.nn.utils.clip_grad_norm_(trainable, a.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                step += 1
                if step % a.log_every == 0 or step == 1:
                    log(dict(epoch=epoch + 1, step=step, loss=batch_loss, grad_norm=float(norm)))
                if a.max_steps and step >= a.max_steps:
                    stop = True
                    break
            torch.cuda.synchronize()
            train_result = dict(epoch=epoch + 1, step=step, n=examples_seen,
                                mean_answer_token_ce=total_loss / examples_seen,
                                seconds=time.monotonic() - start,
                                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
            log(train_result)
            dev_metrics, dev_rows = evaluate(dev_sets, f"dev_epoch{epoch + 1}")
            write_json(outdir / f"dev_epoch{epoch + 1}.json", dict(metrics=dev_metrics, predictions=dev_rows))
            count = sum(metric["n"] for metric in dev_metrics)
            score = (sum(metric["correct"] for metric in dev_metrics) / count,
                     -sum(metric["gold_first_token_nll"] * metric["n"] for metric in dev_metrics) / count)
            if score > best_score:
                best_score = score
                selected_checkpoint = str(save_checkpoint("best.pt", epoch + 1, step, dev_metrics))
            save_checkpoint("last.pt", epoch + 1, step, dev_metrics)
            training_metrics.append(dict(**train_result, dev=dev_metrics))
            write_json(outdir / "training.json", training_metrics)
            if epoch + 1 in a.test_epochs:
                fixed_metrics, fixed_rows = evaluate(test_sets, f"test_epoch{epoch + 1}")
                write_json(outdir / f"test_epoch{epoch + 1}.json",
                           dict(metrics=fixed_metrics, predictions=fixed_rows))
                save_checkpoint(f"epoch{epoch + 1}.pt", epoch + 1, step, dev_metrics)
            if stop:
                break
        if a.gradient_checkpointing:
            model.gradient_checkpointing_disable()
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        # Development selection only; test results never choose an epoch.
        chosen = torch.load(selected_checkpoint, map_location="cpu", weights_only=True)
        if branch is not None:
            branch.load_state_dict(chosen["branch"])
        if lora is not None:
            with torch.no_grad():
                for (index, name), (matrix_a, matrix_b) in lora.params.items():
                    saved_a, saved_b = chosen["lora"][f"{index}.{name}"]
                    matrix_a.copy_(saved_a.to(matrix_a))
                    matrix_b.copy_(saved_b.to(matrix_b))
        log(f"Selected by in-range development data: {selected_checkpoint}")

    all_metrics, all_predictions = [], []
    for mode in a.eval_modes.split("+"):
        metrics, predictions = evaluate(test_sets, "test", mode=mode)
        all_metrics.extend(metrics)
        all_predictions.extend(predictions)
    write_json(outdir / "predictions.json", all_predictions)
    summary = dict(run_id=run_id, arm=a.arm, model=a.model, parameters=parameter_counts,
                   selected_checkpoint=selected_checkpoint, training=training_metrics,
                   results=all_metrics, metric_notes={
                       "exact": "All selected samples are in denominator; unparsable outputs are incorrect",
                       "mae_parsed": "MAE only for outputs parsed as a whole nonnegative integer; also report parse_rate",
                       "gold_first_token_nll": "Native full-vocabulary probability at first answer position, not full-answer likelihood",
                       "timing": "Synchronized batch-one generation, includes prefill; no warm-up exclusion",
                   })
    write_json(outdir / "summary.json", summary)
    log(f"Completed: {outdir / 'summary.json'}")
    log_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
