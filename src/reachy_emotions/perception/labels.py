"""Shared label and probability validation for the three CREMA-D votes."""

import math

EXPRESSION_LABELS = ("ANG", "DIS", "FEA", "HAP", "NEU", "SAD")
UNKNOWN = "unknown"
ALL_LABELS = EXPRESSION_LABELS + (UNKNOWN,)


def normalize_distribution(scores):
    """Return a complete six-class distribution or raise on invalid scores."""
    if not isinstance(scores, dict):
        raise ValueError("scores must be a label-to-number object")
    if set(scores) - set(EXPRESSION_LABELS):
        raise ValueError("unexpected expression labels in scores")
    values = {}
    for label in EXPRESSION_LABELS:
        value = float(scores.get(label, 0.0))
        if not math.isfinite(value) or value < 0:
            raise ValueError("scores must contain finite nonnegative values")
        values[label] = value
    total = sum(values.values())
    if total <= 0:
        raise ValueError("scores must have positive total mass")
    return {label: value / total for label, value in values.items()}


def one_hot(label, smoothing=0.0):
    if label not in EXPRESSION_LABELS:
        raise ValueError("not a supervised expression label: %r" % label)
    if not 0 <= smoothing < 1:
        raise ValueError("smoothing must be in [0, 1)")
    low = smoothing / len(EXPRESSION_LABELS)
    return {name: low + (1.0 - smoothing if name == label else 0.0) for name in EXPRESSION_LABELS}


def best_label(scores):
    distribution = normalize_distribution(scores)
    return max(EXPRESSION_LABELS, key=lambda label: distribution[label])
