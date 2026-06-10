"""AF1 scoring, per-sample rows, and grid-level aggregation."""

import json
from typing import Any, Dict, List, Optional, Sequence

import torch

from evaluations.helpers import patching_core as core
from evaluations.scripts.af1.common import (
    PER_SAMPLE_FIELDS,
    AttentionPolicy,
    PreparedSample,
    SampleLayout,
)
from evaluations.scripts.af1.kernel import (
    move_inputs_to_model_device,
    run_clean_model,
    run_model_with_intervention,
)


def _empty_row(model_name: str, sample_id: str, seq_len: int) -> Dict[str, Any]:
    row = {field: "" for field in PER_SAMPLE_FIELDS}
    row["model"] = model_name
    row["sample_id"] = sample_id
    row["seq_len"] = int(seq_len)
    return row


def _required_score_of_answer_text(metrics: Dict[str, Any], answer_text: str, label: str) -> float:
    if answer_text not in metrics["scores_by_answer"]:
        raise KeyError(f"{label} scores are missing answer {answer_text!r}")
    return float(metrics["scores_by_answer"][answer_text])


def compute_clean_top1_score_drop(clean_metrics: Dict[str, Any], af1_metrics: Dict[str, Any]) -> Dict[str, float]:
    clean_top1 = str(clean_metrics["best_answer_text"]).strip()
    clean_top1_score = float(clean_metrics["best_score"])
    af1_clean_top1_score = _required_score_of_answer_text(
        af1_metrics,
        clean_top1,
        label="Intervention",
    )
    return {
        "af1_clean_top1_score": af1_clean_top1_score,
        "clean_top1_score_drop": float(clean_top1_score - af1_clean_top1_score),
    }


def compute_gold_answer_score_drop(
    sample: PreparedSample,
    clean_metrics: Dict[str, Any],
    af1_metrics: Dict[str, Any],
) -> Dict[str, float]:
    gold_answer = str(sample.gold_answer).strip()
    clean_gold_answer_score = _required_score_of_answer_text(
        clean_metrics,
        gold_answer,
        label="Clean",
    )
    af1_gold_answer_score = _required_score_of_answer_text(
        af1_metrics,
        gold_answer,
        label="Intervention",
    )
    return {
        "gold_answer_score_drop": float(clean_gold_answer_score - af1_gold_answer_score),
    }


def sequence_logprob_from_outputs(outputs: Any, prompt_len: int, answer_token_ids: List[int]) -> float:
    return float(
        core.sequence_logprob_from_logits(
            outputs.logits,
            prompt_len=prompt_len,
            answer_token_ids=answer_token_ids,
        )[0].item()
    )


def score_valid_numeric_answers_with_runner(
    inputs: Dict[str, Any],
    prompt_len: int,
    num_frames: int,
    runner: Any,
    *,
    processor: Any,
) -> Dict[str, Any]:
    scores_by_answer: Dict[str, float] = {}
    for value in range(num_frames + 1):
        answer_text = str(value)
        answer_ids = core.token_ids_of_answer(answer_text, processor=processor)
        scoring_inputs = core.append_answer_tokens_for_scoring(inputs, answer_ids)
        outputs = runner(scoring_inputs, answer_ids)
        scores_by_answer[answer_text] = sequence_logprob_from_outputs(
            outputs,
            prompt_len=prompt_len,
            answer_token_ids=answer_ids,
        )

    ranked_scores = sorted(scores_by_answer.items(), key=lambda item: item[1], reverse=True)
    best_answer_text, best_answer_score = ranked_scores[0]
    second_best_score = ranked_scores[1][1] if len(ranked_scores) > 1 else float("-inf")
    score_values = torch.tensor(list(scores_by_answer.values()), dtype=torch.float64)
    log_denom = torch.logsumexp(score_values, dim=0)
    probs_by_answer = {
        answer_text: float(torch.exp(torch.tensor(score, dtype=torch.float64) - log_denom).item())
        for answer_text, score in scores_by_answer.items()
    }
    return {
        "scores_by_answer": scores_by_answer,
        "probs_by_answer": probs_by_answer,
        "best_answer_text": str(best_answer_text),
        "best_score": float(best_answer_score),
        "margin_over_second": float(best_answer_score - second_best_score),
    }


def run_clean_sample(sample: PreparedSample, *, runtime: Any) -> Dict[str, Any]:
    """Score all valid numeric answers on the untouched base model."""
    clean_inputs = move_inputs_to_model_device(sample.inputs_cpu, model_obj=runtime.model)
    return score_valid_numeric_answers_with_runner(
        clean_inputs,
        prompt_len=sample.layout.prompt_len,
        num_frames=sample.layout.seq_len,
        runner=lambda scoring_inputs, answer_ids: run_clean_model(scoring_inputs, model_obj=runtime.model),
        processor=runtime.processor,
    )


def run_intervention_sample(
    sample: PreparedSample,
    frame_group_means: Optional[Dict[int, Any]],
    non_frame_prompt_mean: Optional[Any],
    policy: AttentionPolicy,
    mode: str,
    *,
    runtime: Any,
) -> Dict[str, Any]:
    """Score all valid numeric answers after applying the selected intervention."""
    intervention_inputs = move_inputs_to_model_device(sample.inputs_cpu, model_obj=runtime.model)
    return score_valid_numeric_answers_with_runner(
        intervention_inputs,
        prompt_len=sample.layout.prompt_len,
        num_frames=sample.layout.seq_len,
        runner=lambda scoring_inputs, answer_ids: run_model_with_intervention(
            scoring_inputs,
            layout=sample.layout,
            frame_group_means=frame_group_means,
            non_frame_prompt_mean=non_frame_prompt_mean,
            policy=policy,
            mode=mode,
            model_obj=runtime.model,
        ),
        processor=runtime.processor,
    )


def evaluated_row(
    sample: PreparedSample,
    clean_metrics: Dict[str, Any],
    af1_metrics: Dict[str, Any],
    donor_ids: Sequence[str],
    policy: AttentionPolicy,
    k_donors_requested: int,
    mode: str,
    *,
    model_name: str,
) -> Dict[str, Any]:
    clean_pred = str(clean_metrics["best_answer_text"]).strip()
    af1_pred = str(af1_metrics["best_answer_text"]).strip()
    clean_correct = int(clean_pred == sample.gold_answer)
    af1_correct = int(af1_pred == sample.gold_answer)
    clean_top1_score_drop_metrics = compute_clean_top1_score_drop(clean_metrics, af1_metrics)
    gold_answer_score_drop_metrics = compute_gold_answer_score_drop(sample, clean_metrics, af1_metrics)
    row = _empty_row(model_name, sample_id=sample.sample_id, seq_len=sample.layout.seq_len)
    row.update(
        {
            "mode": mode,
            "used": 1,
            "gold_answer": sample.gold_answer,
            "clean_pred": clean_pred,
            "clean_correct": clean_correct,
            "clean_gold_prob": float(clean_metrics["probs_by_answer"].get(sample.gold_answer, 0.0)),
            "clean_best_score": float(clean_metrics["best_score"]),
            "clean_margin_over_second": float(clean_metrics["margin_over_second"]),
            "af1_clean_top1_score": float(clean_top1_score_drop_metrics["af1_clean_top1_score"]),
            "clean_top1_score_drop": float(clean_top1_score_drop_metrics["clean_top1_score_drop"]),
            "gold_answer_score_drop": float(gold_answer_score_drop_metrics["gold_answer_score_drop"]),
            "af1_pred": af1_pred,
            "af1_correct": af1_correct,
            "af1_gold_prob": float(af1_metrics["probs_by_answer"].get(sample.gold_answer, 0.0)),
            "af1_best_score": float(af1_metrics["best_score"]),
            "af1_margin_over_second": float(af1_metrics["margin_over_second"]),
            "carrier_index": int(sample.layout.carrier_index),
            "carrier_token": sample.layout.carrier_token_text,
            "wait_layer": int(policy.wait_layer),
            "transfer_layers": int(policy.transfer_layers),
            "transfer_layer_indices": json.dumps(list(policy.transfer_layer_indices)),
            "k_donors": int(k_donors_requested),
            "num_frames": int(sample.layout.seq_len),
            "num_frame_groups": int(len(sample.layout.frame_groups)),
            "prompt_len": int(sample.layout.prompt_len),
            "image_tokens_per_frame": json.dumps(list(sample.layout.image_tokens_per_frame)),
            "room_text": sample.layout.room_text,
            "skipped_reason": "",
            "donor_ids": json.dumps(list(donor_ids)),
            "layout_match_status": "exact_match",
            "layout_match_details": "exact_match",
        }
    )
    return row


def skipped_row(
    model_name: str,
    mode: str,
    sample_id: str,
    seq_len: int,
    gold_answer: str,
    skipped_reason: str,
    room_text: str = "",
    layout: Optional[SampleLayout] = None,
    donor_ids: Optional[Sequence[str]] = None,
    wait_layer: Optional[int] = None,
    transfer_layers: Optional[int] = None,
    k_donors: Optional[int] = None,
    layout_status: str = "skipped",
    layout_details: Optional[str] = None,
) -> Dict[str, Any]:
    row = _empty_row(model_name, sample_id=sample_id, seq_len=seq_len)
    row.update(
        {
            "mode": mode,
            "used": 0,
            "gold_answer": gold_answer,
            "room_text": room_text,
            "skipped_reason": skipped_reason,
            "layout_match_status": layout_status,
            "layout_match_details": layout_details or skipped_reason,
            "donor_ids": json.dumps(list(donor_ids)) if donor_ids is not None else "",
            "wait_layer": "" if wait_layer is None else int(wait_layer),
            "transfer_layers": "" if transfer_layers is None else int(transfer_layers),
            "transfer_layer_indices": (
                ""
                if wait_layer is None or transfer_layers is None
                else json.dumps(list(range(int(wait_layer), int(wait_layer) + int(transfer_layers))))
            ),
            "k_donors": "" if k_donors is None else int(k_donors),
        }
    )
    if layout is not None:
        row.update(
            {
                "carrier_index": int(layout.carrier_index),
                "carrier_token": layout.carrier_token_text,
                "num_frames": int(layout.seq_len),
                "num_frame_groups": int(len(layout.frame_groups)),
                "prompt_len": int(layout.prompt_len),
                "image_tokens_per_frame": json.dumps(list(layout.image_tokens_per_frame)),
            }
        )
    return row


def materialize_skipped_row(
    row_template: Dict[str, Any],
    mode: str,
    wait_layer: int,
    transfer_layers: int,
    k_donors: int,
) -> Dict[str, Any]:
    row = dict(row_template)
    row.update(
        {
            "mode": mode,
            "wait_layer": int(wait_layer),
            "transfer_layers": int(transfer_layers),
            "transfer_layer_indices": json.dumps(
                list(range(int(wait_layer), int(wait_layer) + int(transfer_layers)))
            ),
            "k_donors": int(k_donors),
        }
    )
    return row


def summarize_grid_point_results(
    model_name: str,
    mode: str,
    seq_len: int,
    wait_layer: int,
    transfer_layers: int,
    sample_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    used_rows = [row for row in sample_rows if int(row.get("used") or 0)]
    n_total = len(sample_rows)
    n_used = len(used_rows)
    n_clean_correct = sum(int(row.get("clean_correct") or 0) for row in used_rows)
    n_af1_correct = sum(int(row.get("af1_correct") or 0) for row in used_rows)
    n_both_correct = sum(
        int(bool(int(row.get("clean_correct") or 0)) and bool(int(row.get("af1_correct") or 0)))
        for row in used_rows
    )
    clean_acc = (n_clean_correct / float(n_used)) if n_used else 0.0
    af1_acc = (n_af1_correct / float(n_used)) if n_used else 0.0
    af1_faith = (n_both_correct / float(n_clean_correct)) if n_clean_correct else 0.0
    clean_top1_score_drops = [
        float(row["clean_top1_score_drop"])
        for row in used_rows
        if row.get("clean_top1_score_drop") not in {"", None}
    ]
    mean_clean_top1_score_drop = (
        sum(clean_top1_score_drops) / float(len(clean_top1_score_drops))
        if clean_top1_score_drops
        else 0.0
    )
    gold_answer_score_drops = [
        float(row["gold_answer_score_drop"])
        for row in used_rows
        if row.get("gold_answer_score_drop") not in {"", None}
    ]
    mean_gold_answer_score_drop = (
        sum(gold_answer_score_drops) / float(len(gold_answer_score_drops))
        if gold_answer_score_drops
        else 0.0
    )
    return {
        "model": model_name,
        "mode": mode,
        "seq_len": int(seq_len),
        "wait_layer": int(wait_layer),
        "transfer_layers": int(transfer_layers),
        "n_total": int(n_total),
        "n_used": int(n_used),
        "n_clean_correct": int(n_clean_correct),
        "clean_acc": float(clean_acc),
        "af1_acc": float(af1_acc),
        "af1_faith": float(af1_faith),
        "mean_clean_top1_score_drop": float(mean_clean_top1_score_drop),
        "mean_gold_answer_score_drop": float(mean_gold_answer_score_drop),
    }
