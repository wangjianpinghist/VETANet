"""Aggregate saved classification metrics across cross-validation folds."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


SCALAR_METRICS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
)
PER_CLASS_METRICS = ("precision", "recall", "f1")


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Metric '{field}' must be a number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Metric '{field}' must be finite.")
    return number


def _read_fold(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"Fold result '{path}' must contain a JSON object.")
    for metric in SCALAR_METRICS:
        _finite_number(result.get(metric), metric)
    class_order = result.get("class_order")
    per_class = result.get("per_class")
    if not isinstance(class_order, list) or not all(isinstance(label, str) for label in class_order):
        raise ValueError(f"Fold result '{path}' must contain a string 'class_order' list.")
    if not isinstance(per_class, list) or len(per_class) != len(class_order):
        raise ValueError(f"Fold result '{path}' must contain one per-class result for each class.")
    for index, item in enumerate(per_class):
        if not isinstance(item, dict) or item.get("class_name") != class_order[index]:
            raise ValueError(f"Fold result '{path}' has a per-class order inconsistent with class_order.")
        for metric in PER_CLASS_METRICS:
            _finite_number(item.get(metric), f"{class_order[index]}.{metric}")
        support = item.get("support")
        if isinstance(support, bool) or not isinstance(support, int) or support < 0:
            raise ValueError(f"Fold result '{path}' has an invalid support for '{class_order[index]}'.")
    return result


def _mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "sd": float(statistics.pstdev(values)),
    }


def aggregate_fold_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Return means and population standard deviations for two or more folds."""
    if len(results) < 2:
        raise ValueError("At least two fold result files are required to summarize fold variability.")

    class_order = results[0]["class_order"]
    for index, result in enumerate(results[1:], start=2):
        if result["class_order"] != class_order:
            raise ValueError(f"Fold {index} uses a different class order.")

    metrics = {
        name: _mean_sd([_finite_number(result[name], name) for result in results])
        for name in SCALAR_METRICS
    }
    per_class = []
    for class_index, class_name in enumerate(class_order):
        fold_items = [result["per_class"][class_index] for result in results]
        item = {
            "class_name": class_name,
            **{
                metric: _mean_sd([_finite_number(fold[metric], f"{class_name}.{metric}") for fold in fold_items])
                for metric in PER_CLASS_METRICS
            },
            "support_mean": float(statistics.mean([fold["support"] for fold in fold_items])),
        }
        per_class.append(item)

    return {
        "num_folds": len(results),
        "standard_deviation": "population (n)",
        "class_order": class_order,
        "metrics": metrics,
        "per_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-json",
        nargs="+",
        required=True,
        help="Per-fold JSON files written by evaluation.evaluate_predictions.",
    )
    parser.add_argument("--output-json", help="Optional output path; otherwise print JSON to stdout.")
    args = parser.parse_args()

    results = [_read_fold(path) for path in args.input_json]
    summary = json.dumps(aggregate_fold_results(results), ensure_ascii=False, indent=2) + "\n"
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(summary, encoding="utf-8")
    else:
        print(summary, end="")


if __name__ == "__main__":
    main()
