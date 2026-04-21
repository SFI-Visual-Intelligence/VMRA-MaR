from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from vmra_mar.data import MAX_HORIZON, PatientSample


def masked_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if pos_weight is not None:
        pos_weight = pos_weight.to(device=logits.device, dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight,
    )
    weighted = loss * mask
    return weighted.sum() / mask.sum().clamp(min=1.0)


def _ensure_mirai_import_path(mirai_package_root: str | Path | None) -> None:
    if mirai_package_root is None:
        return
    package_root = str(Path(mirai_package_root))
    if package_root not in sys.path:
        sys.path.insert(0, package_root)


def _mirai_target(sample: PatientSample) -> tuple[int, int] | None:
    if sample.case_label:
        if sample.years_to_event is None or sample.years_to_event < 1 or sample.years_to_event > MAX_HORIZON:
            return None
        return 1, sample.years_to_event - 1

    if sample.followup_years <= 0:
        return None
    return 0, min(sample.followup_years, MAX_HORIZON) - 1


def _collect_mirai_eval_inputs(
    samples: Sequence[PatientSample],
    probs: Sequence[Sequence[float]] | torch.Tensor,
) -> tuple[list[list[float]], list[int], list[int], list[str]]:
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().tolist()

    filtered_probs: list[list[float]] = []
    censor_times: list[int] = []
    golds: list[int] = []
    patient_ids: list[str] = []
    for sample, sample_probs in zip(samples, probs):
        target = _mirai_target(sample)
        if target is None:
            continue
        gold, censor_time = target
        filtered_probs.append(list(sample_probs))
        censor_times.append(censor_time)
        golds.append(gold)
        patient_ids.append(sample.patient_id)
    return filtered_probs, censor_times, golds, patient_ids


def build_mirai_censoring_distribution(
    samples: Sequence[PatientSample],
    min_survival_prob: float = 1e-3,
) -> dict[int, float] | None:
    _, censor_times, golds, _ = _collect_mirai_eval_inputs(samples, [[0.0] * MAX_HORIZON for _ in samples])
    if not censor_times:
        return None

    try:
        from lifelines import KaplanMeierFitter
    except Exception:
        return None

    kmf = KaplanMeierFitter()
    kmf.fit(np.asarray(censor_times, dtype=float), np.asarray(golds, dtype=float))
    return {
        int(time): max(float(kmf.predict(time)), min_survival_prob)
        for time in sorted(set(censor_times))
    }


def roc_auc_score(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None

    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        tie_end = index + 1
        while tie_end < len(ordered) and ordered[tie_end][0] == ordered[index][0]:
            tie_end += 1
        average_rank = (index + 1 + tie_end) / 2.0
        positive_count = sum(label for _, label in ordered[index:tie_end])
        rank_sum += average_rank * positive_count
        index = tie_end

    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _pair_score(event_score: float, later_score: float) -> float:
    if event_score > later_score:
        return 1.0
    if event_score == later_score:
        return 0.5
    return 0.0


def _fallback_concordance_index(censor_times: Sequence[int], golds: Sequence[int], probs: Sequence[Sequence[float]]) -> float | None:
    comparable = 0
    concordant = 0.0
    end_scores = [float(prob[-1]) for prob in probs]
    for left_index, (left_time, left_gold) in enumerate(zip(censor_times, golds)):
        for right_index in range(left_index + 1, len(censor_times)):
            right_time = censor_times[right_index]
            right_gold = golds[right_index]

            if left_gold and left_time < right_time:
                comparable += 1
                concordant += _pair_score(end_scores[left_index], end_scores[right_index])
            elif right_gold and right_time < left_time:
                comparable += 1
                concordant += _pair_score(end_scores[right_index], end_scores[left_index])
    return concordant / comparable if comparable else None


def _fallback_compute_auc_x_year_auc(
    probs: Sequence[Sequence[float]],
    censor_times: Sequence[int],
    golds: Sequence[int],
    followup: int,
) -> tuple[float | None, list[int]]:
    probs_for_eval = []
    golds_for_eval = []
    for prob_arr, censor_time, gold in zip(probs, censor_times, golds):
        valid_pos = bool(gold) and censor_time <= followup
        valid_neg = censor_time >= followup
        if valid_pos or valid_neg:
            probs_for_eval.append(prob_arr[followup])
            golds_for_eval.append(int(valid_pos))
    return roc_auc_score(golds_for_eval, probs_for_eval), golds_for_eval


def _fallback_compute_auc_metrics_given_curve(
    probs: Sequence[Sequence[float]],
    censor_times: Sequence[int],
    golds: Sequence[int],
    years: Sequence[int],
    max_followup: int,
    censor_distribution: dict[int, float] | None,
) -> tuple[dict[object, object], dict[object, list[int]]]:
    del years, censor_distribution

    metrics: dict[object, object] = {}
    sample_sizes: dict[object, list[int]] = {}
    for followup in range(max_followup):
        auc, golds_for_eval = _fallback_compute_auc_x_year_auc(probs, censor_times, golds, followup)
        metrics[followup + 1] = auc
        sample_sizes[followup + 1] = golds_for_eval

    c_index = _fallback_concordance_index(censor_times, golds, probs)
    metrics["c_index"] = c_index

    total_cases = int(sum(golds))
    if total_cases == 0 or not probs:
        metrics["decile_recall"] = None
    else:
        end_probs = np.asarray(probs)[:, -1].tolist()
        sorted_golds = [gold for _, gold in sorted(zip(end_probs, golds))]
        top_count = max(len(sorted_golds) // 10, 1)
        metrics["decile_recall"] = float(sum(sorted_golds[-top_count:]) / total_cases)
    return metrics, sample_sizes


def _resolve_mirai_metric_impl(
    mirai_package_root: str | Path | None,
) -> tuple[Callable[..., tuple[dict[object, object], dict[object, list[int]]]], str]:
    try:
        _ensure_mirai_import_path(mirai_package_root)
        from onconet.learn.utils import compute_auc_metrics_given_curve as mirai_compute_auc_metrics_given_curve

        return mirai_compute_auc_metrics_given_curve, "vendor_mirai"
    except Exception:
        return _fallback_compute_auc_metrics_given_curve, "local_fallback"


def _normalize_metric(value: object) -> float | None:
    if value in (None, "NA"):
        return None
    if isinstance(value, np.generic):
        value = float(value.item())
        return None if math.isnan(value) else value
    if isinstance(value, (float, int)):
        value = float(value)
        return None if math.isnan(value) else value
    return None


def evaluate_predictions(
    samples: Sequence[PatientSample],
    probs: Sequence[Sequence[float]] | torch.Tensor,
    *,
    train_samples: Sequence[PatientSample] | None = None,
    mirai_package_root: str | Path | None = None,
    censoring_distribution: dict[int, float] | None = None,
) -> dict[str, object]:
    filtered_probs, censor_times, golds, patient_ids = _collect_mirai_eval_inputs(samples, probs)

    if not filtered_probs:
        return {
            "metric_source": "mirai_compatible_empty_split",
            "included_patients": 0,
            "excluded_patients": len(samples),
            "year_auc": {f"year_{year}": None for year in range(1, MAX_HORIZON + 1)},
            "sample_sizes": {
                f"year_{year}": {"n": 0, "cases": 0}
                for year in range(1, MAX_HORIZON + 1)
            },
            "c_index": None,
            "decile_recall": None,
        }

    if censoring_distribution is None:
        censor_source = train_samples if train_samples is not None else samples
        censoring_distribution = build_mirai_censoring_distribution(censor_source)

    metric_fn, metric_source = _resolve_mirai_metric_impl(mirai_package_root)
    try:
        metrics, sample_sizes = metric_fn(
            filtered_probs,
            censor_times,
            golds,
            [0] * len(filtered_probs),
            MAX_HORIZON,
            censoring_distribution,
        )
    except Exception:
        metrics, sample_sizes = _fallback_compute_auc_metrics_given_curve(
            filtered_probs,
            censor_times,
            golds,
            [0] * len(filtered_probs),
            MAX_HORIZON,
            censoring_distribution,
        )
        metric_source = f"{metric_source}_fallback"

    year_auc: dict[str, float | None] = {}
    year_sample_sizes: dict[str, dict[str, int]] = {}
    for followup in range(MAX_HORIZON):
        key = followup + 1
        labels = sample_sizes.get(key, [])
        year_name = f"year_{key}"
        year_auc[year_name] = _normalize_metric(metrics.get(key))
        year_sample_sizes[year_name] = {
            "n": len(labels),
            "cases": int(sum(labels)),
        }

    return {
        "metric_source": metric_source,
        "included_patients": len(patient_ids),
        "excluded_patients": max(len(samples) - len(patient_ids), 0),
        "year_auc": year_auc,
        "sample_sizes": year_sample_sizes,
        "c_index": _normalize_metric(metrics.get("c_index")),
        "decile_recall": _normalize_metric(metrics.get("decile_recall")),
    }
