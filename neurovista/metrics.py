"""Auditable metrics for actual arrays, never random paper-performance substitutes."""
import numpy as np


def binary_overlap(truth, prediction, *, threshold=0.5, empty_score=1.0):
    """Compute Dice/IoU for one mask. Truth must contain only 0/1 values.

    Both-empty masks score 1 by default (explicit convention). Aggregate across
    patients separately; do not silently flatten a batch into one global metric.
    """
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape or truth.size == 0:
        raise ValueError("Masks must have the same nonempty shape")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Masks must be finite")
    if not np.isin(truth, [0, 1]).all() or np.any((prediction < 0) | (prediction > 1)):
        raise ValueError("Expected binary truth and probability-range prediction")
    if not 0 <= threshold <= 1 or not 0 <= empty_score <= 1:
        raise ValueError("Invalid threshold or empty-mask score")
    a, b = truth.astype(bool), prediction >= threshold
    intersection = int(np.logical_and(a, b).sum())
    total, union = int(a.sum() + b.sum()), int(np.logical_or(a, b).sum())
    return {"dice": 2 * intersection / total if total else empty_score,
            "iou": intersection / union if union else empty_score}


def classification_report(truth, prediction, labels):
    """Confusion matrix and per-class precision/recall/F1 from label IDs.

    Undefined ratios are zero; macro scores include every supplied class.
    Accuracy is overall multiclass accuracy, not one-vs-rest class accuracy.
    """
    a, b = np.asarray(truth), np.asarray(prediction)
    labels = tuple(labels)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Provide unique nonempty label names")
    if a.ndim != 1 or a.shape != b.shape or not a.size:
        raise ValueError("Expected matching nonempty one-dimensional label IDs")
    n = len(labels)
    if not np.isin(a, range(n)).all() or not np.isin(b, range(n)).all():
        raise ValueError("Label IDs outside the supplied class order")
    matrix = np.zeros((n, n), dtype=int)
    np.add.at(matrix, (a.astype(int), b.astype(int)), 1)
    rows = []
    for i, label in enumerate(labels):
        tp, support, predicted = int(matrix[i,i]), int(matrix[i].sum()), int(matrix[:,i].sum())
        precision, recall = tp / predicted if predicted else 0., tp / support if support else 0.
        rows.append({"label": label, "support": support, "precision": precision,
                     "recall": recall, "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.})
    return {"accuracy": float(np.trace(matrix)/a.size), "confusion_matrix": matrix.tolist(),
            "per_class": rows, "macro_f1": float(np.mean([r["f1"] for r in rows]))}
