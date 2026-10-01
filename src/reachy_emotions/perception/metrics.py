"""Clip-level metrics with explicit abstention accounting."""

from .labels import ALL_LABELS, EXPRESSION_LABELS, UNKNOWN


def evaluate_predictions(rows, target_key, prediction_key="presented_expression"):
    """Evaluate one prediction per sample and summarize by held-out actor."""
    evaluated = []
    seen = set()
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in seen:
            raise ValueError("duplicate sample_id in evaluation: %s" % sample_id)
        seen.add(sample_id)
        truth = row.get(target_key)
        predicted = row.get(prediction_key)
        if truth not in ALL_LABELS or predicted not in ALL_LABELS:
            raise ValueError("invalid truth or prediction for %s" % sample_id)
        evaluated.append((truth, predicted, str(row["actor_id"])))
    if not evaluated:
        raise ValueError("no rows to evaluate")
    known = [(y, p, actor) for y, p, actor in evaluated if y in EXPRESSION_LABELS]
    if not known:
        raise ValueError("no unambiguous labeled clips to evaluate")
    confusion = {label: {prediction: 0 for prediction in ALL_LABELS} for label in EXPRESSION_LABELS}
    for truth, predicted, _ in known:
        confusion[truth][predicted] += 1
    per_class = {}
    for label in EXPRESSION_LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in EXPRESSION_LABELS if other != label)
        fn = sum(confusion[label][other] for other in ALL_LABELS if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(confusion[label].values())}
    by_actor = {}
    for actor in sorted({actor for _, _, actor in known}):
        actor_rows = [(y, p) for y, p, a in known if a == actor]
        by_actor[actor] = {"clips": len(actor_rows), "accuracy": sum(y == p for y, p in actor_rows) / len(actor_rows),
                           "abstention_rate": sum(p == UNKNOWN for _, p in actor_rows) / len(actor_rows)}
    return {
        "clips": len(evaluated), "labeled_clips": len(known),
        "accuracy": sum(y == p for y, p, _ in known) / len(known),
        "macro_f1": sum(item["f1"] for item in per_class.values()) / len(EXPRESSION_LABELS),
        "abstention_rate": sum(p == UNKNOWN for _, p, _ in known) / len(known),
        "ambiguous_abstention_rate": (sum(y == UNKNOWN and p == UNKNOWN for y, p, _ in evaluated)
                                      / sum(y == UNKNOWN for y, _, _ in evaluated)
                                      if any(y == UNKNOWN for y, _, _ in evaluated) else None),
        "per_class": per_class, "confusion": confusion, "by_actor": by_actor,
    }
