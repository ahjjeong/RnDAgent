"""Evaluate performance-prediction outputs against paper-outcome labels.

The final moderator output predicts future paper performance:

    performance_score                  0~1 expected percentile-like score
    expected_performance_level          high|middle|low
    predicted_high_performance_top20    bool
    confidence                          0~1

The dataset already contains outcome labels computed from 5-year paper windows.
This module compares agent predictions against the three matching criteria:

    direct              conservative main label
    direct_plus_lead    includes lead-project matching
    expanded            broader matching for robustness checks
"""
from __future__ import annotations

from math import ceil, sqrt
from typing import Any

from .config import ID_COL


MATCHING_CRITERIA = ("direct", "direct_plus_lead", "expanded")
TOP_THRESHOLDS = (10, 20, 30)
MAIN_CRITERION = "direct"
MAIN_TOP = 20


def normalize_project_id(value: Any) -> str:
    """Normalize project ids so pandas numeric ids and JSON ids can match."""
    if _is_missing(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(value != value)
    except Exception:
        return str(value) in {"<NA>", "NaT", "nan", "None"}


def _bool_or_none(value: Any) -> bool | None:
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "high"}:
        return True
    if text in {"0", "false", "f", "no", "n", "low", "middle"}:
        return False
    return None


def _float_or_none(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def _clip01(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, value))


def build_outcome_labels(df: Any) -> dict[str, dict[str, Any]]:
    """Return project-id keyed outcome labels and percentile targets."""
    labels: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        pid = normalize_project_id(row.get(ID_COL))
        if not pid:
            continue

        outcome: dict[str, Any] = {}
        for criterion in MATCHING_CRITERIA:
            for top in TOP_THRESHOLDS:
                col = f"{criterion}_high_performance_top{top}"
                outcome[col] = _bool_or_none(row.get(col))

            pct_col = f"{criterion}_performance_percentile"
            outcome[pct_col] = _clip01(_float_or_none(row.get(pct_col)))

            group_col = f"{criterion}_performance_group_3class"
            outcome[group_col] = row.get(group_col)

        outcome["label_comparison_group"] = row.get("label_comparison_group")
        outcome["label_comparison_group_size"] = _float_or_none(row.get("label_comparison_group_size"))
        outcome["budget_group"] = row.get("budget_group")
        labels[pid] = outcome
    return labels


def _normalize_prediction(verdict: dict[str, Any]) -> dict[str, Any]:
    score = _clip01(_float_or_none(verdict.get("performance_score")))
    pred_top20 = _bool_or_none(verdict.get("predicted_high_performance_top20"))
    if pred_top20 is None and score is not None:
        pred_top20 = score >= 0.80

    level = str(verdict.get("expected_performance_level", "")).strip().lower()
    if level not in {"high", "middle", "low"}:
        if score is None:
            level = ""
        elif score >= 0.80:
            level = "high"
        elif score >= 0.40:
            level = "middle"
        else:
            level = "low"

    return {
        "performance_score": score,
        "expected_performance_level": level,
        "predicted_high_performance_top20": pred_top20,
        "confidence": _clip01(_float_or_none(verdict.get("confidence"))),
        "key_reasons": verdict.get("key_reasons", ""),
    }


def compare(verdict: dict[str, Any], outcome: dict[str, Any] | None) -> dict[str, Any]:
    """Return per-project validation details for all label variants."""
    prediction = _normalize_prediction(verdict)
    validation: dict[str, Any] = {"prediction": prediction}
    if not outcome:
        validation["has_outcome"] = False
        return validation

    validation["has_outcome"] = True
    validation["outcome"] = outcome

    predicted = prediction["predicted_high_performance_top20"]
    for criterion in MATCHING_CRITERIA:
        top20_col = f"{criterion}_high_performance_top20"
        percentile_col = f"{criterion}_performance_percentile"
        actual_top20 = outcome.get(top20_col)
        percentile = outcome.get(percentile_col)

        validation[criterion] = {
            "actual_high_performance_top20": actual_top20,
            "actual_performance_percentile": percentile,
            "aligned_top20": (
                None if predicted is None or actual_top20 is None else predicted == actual_top20
            ),
            "absolute_percentile_error": (
                None if prediction["performance_score"] is None or percentile is None
                else abs(prediction["performance_score"] - percentile)
            ),
        }

    validation["main"] = validation[MAIN_CRITERION]
    return validation


def _confusion(y_true: list[bool], y_pred: list[bool]) -> dict[str, int]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _safe_div(num: float, den: float) -> float | None:
    return None if den == 0 else num / den


def binary_metrics(y_true: list[bool], y_pred: list[bool]) -> dict[str, Any]:
    cm = _confusion(y_true, y_pred)
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = None if precision is None or recall is None or precision + recall == 0 else (
        2 * precision * recall / (precision + recall)
    )
    return {
        "n": len(y_true),
        "positive_rate": _safe_div(sum(y_true), len(y_true)),
        "predicted_positive_rate": _safe_div(sum(y_pred), len(y_pred)),
        **cm,
        "accuracy": _safe_div(tp + tn, len(y_true)),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": (
            None if recall is None or specificity is None else (recall + specificity) / 2
        ),
    }


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-based ranks
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def roc_auc(y_true: list[bool], scores: list[float]) -> float | None:
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _average_ranks(scores)
    rank_sum_pos = sum(rank for rank, is_pos in zip(ranks, y_true) if is_pos)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def average_precision(y_true: list[bool], scores: list[float]) -> float | None:
    n_pos = sum(y_true)
    if n_pos == 0:
        return None
    pairs = sorted(zip(scores, y_true), key=lambda x: x[0], reverse=True)
    hit = 0
    precision_sum = 0.0
    for rank, (_, is_pos) in enumerate(pairs, start=1):
        if is_pos:
            hit += 1
            precision_sum += hit / rank
    return precision_sum / n_pos


def precision_at_fraction(y_true: list[bool], scores: list[float], fraction: float) -> float | None:
    if not y_true:
        return None
    k = max(1, ceil(len(y_true) * fraction))
    pairs = sorted(zip(scores, y_true), key=lambda x: x[0], reverse=True)
    return sum(is_pos for _, is_pos in pairs[:k]) / k


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    mx, my = sum(x) / len(x), sum(y) / len(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den_x = sqrt(sum((a - mx) ** 2 for a in x))
    den_y = sqrt(sum((b - my) ** 2 for b in y))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    return pearson(_average_ranks(x), _average_ranks(y))


def score_metrics(y_true: list[bool], percentiles: list[float], scores: list[float]) -> dict[str, Any]:
    errors = [abs(s - p) for s, p in zip(scores, percentiles)]
    squared_errors = [(s - p) ** 2 for s, p in zip(scores, percentiles)]
    brier = [(s - float(t)) ** 2 for s, t in zip(scores, y_true)]
    return {
        "n": len(scores),
        "mae_vs_percentile": None if not errors else sum(errors) / len(errors),
        "rmse_vs_percentile": None if not squared_errors else sqrt(sum(squared_errors) / len(squared_errors)),
        "brier_top20": None if not brier else sum(brier) / len(brier),
        "spearman_vs_percentile": spearman(scores, percentiles),
        "pearson_vs_percentile": pearson(scores, percentiles),
        "roc_auc_top20": roc_auc(y_true, scores),
        "pr_auc_average_precision_top20": average_precision(y_true, scores),
        "precision_at_top10pct_by_score": precision_at_fraction(y_true, scores, 0.10),
        "precision_at_top20pct_by_score": precision_at_fraction(y_true, scores, 0.20),
        "precision_at_top30pct_by_score": precision_at_fraction(y_true, scores, 0.30),
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate validation metrics from result records."""
    summary: dict[str, Any] = {"n_results": len(results), "criteria": {}}
    for criterion in MATCHING_CRITERIA:
        y_true: list[bool] = []
        y_pred: list[bool] = []
        score_y_true: list[bool] = []
        scores: list[float] = []
        percentiles: list[float] = []

        for result in results:
            validation = result.get("validation", {})
            pred = validation.get("prediction", {})
            outcome = validation.get("outcome", {})

            actual = outcome.get(f"{criterion}_high_performance_top20")
            score = pred.get("performance_score")
            predicted = pred.get("predicted_high_performance_top20")
            percentile = outcome.get(f"{criterion}_performance_percentile")

            if actual is not None and predicted is not None:
                y_true.append(bool(actual))
                y_pred.append(bool(predicted))
            if actual is not None and score is not None and percentile is not None:
                score_y_true.append(bool(actual))
                scores.append(float(score))
                percentiles.append(float(percentile))

        criterion_metrics: dict[str, Any] = {
            "binary_top20": binary_metrics(y_true, y_pred) if y_true else {"n": 0},
        }
        if scores:
            criterion_metrics["score_and_ranking"] = score_metrics(score_y_true, percentiles, scores)
        else:
            criterion_metrics["score_and_ranking"] = {"n": 0}
        summary["criteria"][criterion] = criterion_metrics

    summary["main"] = summary["criteria"][MAIN_CRITERION]
    return summary
