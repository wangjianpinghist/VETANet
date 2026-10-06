"""Classification metrics for a fixed, explicitly ordered set of classes."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    class_labels: Sequence[str],
) -> dict:
    labels = [str(label) for label in class_labels]
    truth = [str(label) for label in y_true]
    prediction = [str(label) for label in y_pred]

    if not labels or len(set(labels)) != len(labels):
        raise ValueError("class_labels must be a non-empty sequence of unique labels.")
    if not truth or len(truth) != len(prediction):
        raise ValueError("y_true and y_pred must be non-empty sequences of equal length.")
    unknown = (set(truth) | set(prediction)) - set(labels)
    if unknown:
        raise ValueError(f"Found labels not listed in class_labels: {sorted(unknown)}")

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        truth, prediction, labels=labels, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        truth, prediction, labels=labels, average="weighted", zero_division=0
    )
    per_p, per_r, per_f1, support = precision_recall_fscore_support(
        truth, prediction, labels=labels, average=None, zero_division=0
    )

    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "per_class": [
            {
                "class_name": label,
                "precision": float(per_p[index]),
                "recall": float(per_r[index]),
                "f1": float(per_f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        ],
        "confusion_matrix": confusion_matrix(truth, prediction, labels=labels).tolist(),
        "class_order": labels,
        "num_samples": len(truth),
    }
