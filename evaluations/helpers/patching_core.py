import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from nnsight import LanguageModel

from evaluations.helpers import utils as eval_utils
from models.model import model as base_model, processor

iter_sample_dirs = eval_utils.iter_sample_dirs
load_mmred_sample = eval_utils.load_mmred_sample
parse_layer_selection = eval_utils.parse_layer_selection


def token_ids_of_answer(answer_text: str) -> List[int]:
    ids = processor.tokenizer.encode(str(answer_text).strip(), add_special_tokens=False)
    if not ids:
        raise ValueError(f"Answer text tokenized to empty: {answer_text!r}")
    return [int(token_id) for token_id in ids]


def build_prompt(question: str, num_frames: int) -> str:
    return (
        f"You will be shown {num_frames} frames describing steps in a house.\n"
        f"Respond with a single integer from 0 to {num_frames} (0 is allowed). Output only the integer.\n"
        f"Question: {question}\n"
        "Answer: "
    )


def build_inputs_from_prompt(frames: Sequence[Any], prompt: str) -> Dict[str, torch.Tensor]:
    messages = [{
        "role": "user",
        "content": (
            [{"type": "image", "image": image} for image in frames] +
            [{"type": "text", "text": prompt}]
        ),
    }]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return dict(inputs)


def build_inputs(frames: Sequence[Any], question: str) -> Dict[str, torch.Tensor]:
    return build_inputs_from_prompt(frames, build_prompt(question, num_frames=len(frames)))


def move_inputs_to_model_device(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    device = next(base_model.parameters()).device
    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in inputs.items()}


def repeat_inputs_for_batch(inputs: Dict[str, torch.Tensor], batch_size: int) -> Dict[str, torch.Tensor]:
    if batch_size <= 1:
        return inputs

    repeated: Dict[str, torch.Tensor] = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            repeated[key] = value
            continue
        if value.dim() == 0:
            repeated[key] = value.repeat(batch_size)
            continue
        if int(value.shape[0]) == 1:
            repeated[key] = value.repeat(batch_size, *([1] * (value.dim() - 1)))
            continue
        if key in {"pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"}:
            repeated[key] = torch.cat([value] * batch_size, dim=0)
            continue
        raise ValueError(f"Cannot batch-repeat input {key!r} with shape={tuple(value.shape)}")
    return repeated


def concatenate_inputs_for_batch(inputs_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not inputs_list:
        raise ValueError("inputs_list must be non-empty")
    if len(inputs_list) == 1:
        return inputs_list[0]

    out: Dict[str, torch.Tensor] = {}
    keys = list(inputs_list[0].keys())
    for key in keys:
        values = [inputs[key] for inputs in inputs_list]
        first_value = values[0]
        if not torch.is_tensor(first_value):
            out[key] = first_value
            continue
        if first_value.dim() == 0:
            out[key] = torch.stack(values, dim=0)
            continue
        if int(first_value.shape[0]) == 1 or key in {"pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"}:
            out[key] = torch.cat(values, dim=0)
            continue
        raise ValueError(f"Cannot concatenate input {key!r} with shape={tuple(first_value.shape)}")
    return out


def _materialize_saved(x: Any) -> Any:
    return x.value if hasattr(x, "value") else x


def _to_hidden_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)) and x:
        return _to_hidden_tensor(x[0])
    raise TypeError(f"Unsupported layer output type for corruption: {type(x)}")


def _normalize_token_positions(token_positions: Sequence[int], prompt_len: int) -> List[int]:
    normalized: List[int] = []
    for position in token_positions:
        position_int = int(position)
        if position_int < 0:
            position_int = int(prompt_len) + position_int
        normalized.append(position_int)
    return normalized


def append_answer_tokens_for_scoring(
    inputs: Dict[str, torch.Tensor],
    answer_token_ids: List[int],
) -> Dict[str, torch.Tensor]:
    if not answer_token_ids:
        raise ValueError("answer_token_ids must be non-empty")

    input_ids = inputs["input_ids"]
    if input_ids.dim() != 2:
        raise ValueError(f"Expected input_ids to be rank-2, got shape={tuple(input_ids.shape)}")

    batch_size = int(input_ids.shape[0])
    answer_tokens = torch.tensor(
        answer_token_ids,
        dtype=input_ids.dtype,
        device=input_ids.device,
    ).unsqueeze(0).repeat(batch_size, 1)

    scored_inputs = dict(inputs)
    scored_inputs["input_ids"] = torch.cat([input_ids, answer_tokens], dim=1)

    if "attention_mask" in inputs and torch.is_tensor(inputs["attention_mask"]):
        attention_mask = inputs["attention_mask"]
        suffix_attention = torch.ones(
            (batch_size, len(answer_token_ids)),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        scored_inputs["attention_mask"] = torch.cat([attention_mask, suffix_attention], dim=1)

    return scored_inputs


def sequence_logprob_from_logits(
    logits: torch.Tensor,
    prompt_len: int,
    answer_token_ids: List[int],
) -> torch.Tensor:
    if logits.dim() != 3:
        raise ValueError(f"Expected logits rank-3 [batch, seq, vocab], got {tuple(logits.shape)}")
    if prompt_len <= 0:
        raise ValueError("prompt_len must be >= 1")
    if not answer_token_ids:
        raise ValueError("answer_token_ids must be non-empty")

    batch_size = int(logits.shape[0])
    device = logits.device
    answer_len = len(answer_token_ids)
    token_positions = torch.arange(answer_len, device=device, dtype=torch.long) + (prompt_len - 1)
    target_token_ids = torch.tensor(answer_token_ids, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    selected_logits = logits[:, token_positions, :]
    log_probs = torch.log_softmax(selected_logits, dim=-1)
    target_log_probs = torch.gather(log_probs, dim=-1, index=target_token_ids.unsqueeze(-1)).squeeze(-1)
    return target_log_probs.sum(dim=1)


def run_clean_sequence_logprob(
    lm: LanguageModel,
    scoring_inputs: Dict[str, torch.Tensor],
    prompt_len: int,
    answer_token_ids: List[int],
) -> float:
    with torch.inference_mode():
        with lm.trace(scoring_inputs):
            saved_logits = lm.output.logits.save()
    logits = _materialize_saved(saved_logits)
    scores = sequence_logprob_from_logits(logits, prompt_len=prompt_len, answer_token_ids=answer_token_ids)
    return float(scores[0].item())


def run_layer_multi_group_corrupted_sequence_logprob(
    lm: LanguageModel,
    layers: Any,
    clean_batched_scoring_inputs: Dict[str, torch.Tensor],
    control_batched_scoring_inputs: Dict[str, torch.Tensor],
    layer_idx: int,
    clean_token_positions_by_batch: List[List[int]],
    control_token_positions_by_batch: List[List[int]],
    prompt_len: int,
    answer_token_ids: List[int],
) -> torch.Tensor:
    with torch.no_grad():
        with lm.trace(control_batched_scoring_inputs):
            control_layer_saved = _to_hidden_tensor(layers[layer_idx].output).save()

        with lm.trace(clean_batched_scoring_inputs):
            clean_layer_out = _to_hidden_tensor(layers[layer_idx].output)
            control_layer_out = _materialize_saved(control_layer_saved)
            for batch_idx, (clean_positions, control_positions) in enumerate(
                zip(clean_token_positions_by_batch, control_token_positions_by_batch)
            ):
                clean_positions = _normalize_token_positions(clean_positions, prompt_len=prompt_len)
                control_positions = _normalize_token_positions(control_positions, prompt_len=prompt_len)
                if clean_positions and control_positions:
                    clean_layer_out[batch_idx, clean_positions, :] = control_layer_out[batch_idx, control_positions, :]
            saved_logits = lm.output.logits.save()

    logits = _materialize_saved(saved_logits)
    return sequence_logprob_from_logits(logits, prompt_len=prompt_len, answer_token_ids=answer_token_ids)


def run_layer_corrupted_sequence_logprob(
    lm: LanguageModel,
    layers: Any,
    clean_scoring_inputs: Dict[str, torch.Tensor],
    control_scoring_inputs: Dict[str, torch.Tensor],
    layer_idx: int,
    clean_token_positions: Sequence[int],
    control_token_positions: Sequence[int],
    prompt_len: int,
    answer_token_ids: List[int],
) -> float:
    scores = run_layer_multi_group_corrupted_sequence_logprob(
        lm=lm,
        layers=layers,
        clean_batched_scoring_inputs=clean_scoring_inputs,
        control_batched_scoring_inputs=control_scoring_inputs,
        layer_idx=layer_idx,
        clean_token_positions_by_batch=[list(clean_token_positions)],
        control_token_positions_by_batch=[list(control_token_positions)],
        prompt_len=prompt_len,
        answer_token_ids=answer_token_ids,
    )
    return float(scores[0].item())


def normalize_to_probabilities(values: List[float]) -> List[float]:
    total = float(sum(values))
    if total <= 0.0:
        return [0.0 for _ in values]
    return [float(value) / total for value in values]


def entropy_from_probabilities(probs: List[float]) -> float:
    return -sum(float(prob) * math.log(float(prob)) for prob in probs if prob > 0.0)


def normalize_entropy(entropy: float, num_groups: int) -> float:
    if num_groups <= 1:
        return 0.0
    return float(entropy / math.log(num_groups))


def load_clean_score_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    return eval_utils.load_clean_score_cache(path)


def save_clean_score_cache(path: Path, cache: Dict[str, Dict[str, Any]]) -> None:
    eval_utils.save_clean_score_cache(path, cache)


def write_metrics_json(sample_metrics: List[Dict[str, Any]], output_dir: Path) -> Path:
    return eval_utils.write_metrics_json(sample_metrics, output_dir)


def score_valid_numeric_answers(
    lm: LanguageModel,
    inputs: Dict[str, torch.Tensor],
    prompt_len: int,
    num_frames: int,
) -> Dict[str, Any]:
    scores_by_answer: Dict[str, float] = {}
    for value in range(num_frames + 1):
        answer_text = str(value)
        answer_ids = token_ids_of_answer(answer_text)
        scoring_inputs = append_answer_tokens_for_scoring(inputs, answer_ids)
        scores_by_answer[answer_text] = run_clean_sequence_logprob(
            lm=lm,
            scoring_inputs=scoring_inputs,
            prompt_len=prompt_len,
            answer_token_ids=answer_ids,
        )

    score_values = torch.tensor(list(scores_by_answer.values()), dtype=torch.float64)
    log_denom = torch.logsumexp(score_values, dim=0)
    probs_by_answer = {
        answer_text: float(torch.exp(torch.tensor(score, dtype=torch.float64) - log_denom).item())
        for answer_text, score in scores_by_answer.items()
    }
    best_answer_text = max(scores_by_answer.items(), key=lambda item: item[1])[0]
    return {
        "scores_by_answer": scores_by_answer,
        "probs_by_answer": probs_by_answer,
        "best_answer_text": best_answer_text,
    }
