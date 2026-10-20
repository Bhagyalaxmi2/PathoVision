"""Small dependency-light classification metrics for validation reporting."""
from __future__ import annotations

import numpy as np


def classification_metrics(y_true, y_pred, class_names):
    """Return accuracy, macro F1, per-class scores, and a confusion matrix.

    Labels are integer indices into ``class_names``. Validate inputs so a
    truncated zip or a negative NumPy index cannot silently distort reports.
    """
    class_names = list(class_names)
    y_true = list(y_true)
    y_pred = list(y_pred)
    if not class_names:
        raise ValueError("class_names must contain at least one class")
    if not all(isinstance(name, str) and name.strip() for name in class_names):
        raise ValueError("class_names must contain non-empty strings")
    if len(set(class_names)) != len(class_names):
        raise ValueError("class_names must be unique")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same number of labels")
    n = len(class_names)
    matrix = np.zeros((n, n), dtype=np.int64)
    for actual, predicted in zip(y_true, y_pred):
        try:
            actual_index, predicted_index = int(actual), int(predicted)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("labels must be integer class indices") from exc
        if actual_index != actual or predicted_index != predicted:
            raise ValueError("labels must be integer class indices")
        if not 0 <= actual_index < n or not 0 <= predicted_index < n:
            raise ValueError(f"labels must be between 0 and {n - 1}")
        matrix[actual_index, predicted_index] += 1
    rows = []
    for i, name in enumerate(class_names):
        tp = int(matrix[i, i])
        precision = tp / int(matrix[:, i].sum()) if matrix[:, i].sum() else 0.0
        recall = tp / int(matrix[i, :].sum()) if matrix[i, :].sum() else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"class": name, "precision": precision, "recall": recall, "f1": f1, "support": int(matrix[i, :].sum())})
    accuracy = float(np.trace(matrix) / matrix.sum()) if matrix.sum() else 0.0
    macro_f1 = float(np.mean([row["f1"] for row in rows])) if rows else 0.0
    return {"accuracy": accuracy, "macro_f1": macro_f1, "classes": class_names,
            "per_class": rows, "confusion_matrix": matrix.tolist()}
