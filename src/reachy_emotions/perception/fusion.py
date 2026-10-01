"""Small validation-calibrated weighted audiovisual fusion baseline."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .labels import EXPRESSION_LABELS, UNKNOWN, best_label, normalize_distribution


@dataclass(frozen=True)
class FusionConfig:
    visual_weight: float = 0.5
    visual_temperature: float = 1.0
    audio_temperature: float = 1.0
    min_quality: float = 0.25
    min_confidence: float = 0.42
    min_margin: float = 0.08
    disagreement_confidence: float = 0.60

    def validate(self):
        for key in ("visual_weight", "min_quality", "min_confidence", "min_margin", "disagreement_confidence"):
            value = getattr(self, key)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("%s must be finite and in [0, 1]" % key)
        for key in ("visual_temperature", "audio_temperature"):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("%s must be finite and positive" % key)
        return self

    def save(self, path):
        self.validate()
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8"))).validate()


def aggregate_frames(frame_scores, frame_qualities=None):
    """Aggregate three single-image predictions into one clip distribution."""
    if len(frame_scores) != 3:
        raise ValueError("exactly three frame predictions are required")
    qualities = [1.0] * 3 if frame_qualities is None else list(frame_qualities)
    if len(qualities) != 3 or any(not math.isfinite(float(q)) or not 0 <= float(q) <= 1 for q in qualities):
        raise ValueError("three frame qualities in [0, 1] are required")
    total = sum(qualities)
    if total == 0:
        return None, 0.0
    distributions = [normalize_distribution(scores) if q > 0 else None
                     for q, scores in zip(qualities, frame_scores)]
    merged = {label: sum(q * distribution[label] for q, distribution in zip(qualities, distributions) if q > 0) / total
              for label in EXPRESSION_LABELS}
    return merged, total / 3.0


def apply_temperature(scores, temperature):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    distribution = normalize_distribution(scores)
    powered = {label: max(probability, 1e-12) ** (1 / temperature)
               for label, probability in distribution.items()}
    return normalize_distribution(powered)


def fit_temperature(validation_rows, score_key, target_key,
                    temperatures=(0.5, 0.75, 1.0, 1.5, 2.0, 3.0)):
    """Choose a temperature from held-out modality labels by log loss."""
    examples = []
    for row in validation_rows:
        if row.get("split") != "validation":
            raise ValueError("temperatures must be fitted on validation records only")
        if row.get(target_key) in EXPRESSION_LABELS and row.get(score_key) is not None:
            examples.append((row[score_key], row[target_key]))
    if not examples:
        raise ValueError("no labeled validation clips for %s" % target_key)
    return min(temperatures, key=lambda value: sum(-math.log(max(apply_temperature(scores, value)[target], 1e-12))
                                                    for scores, target in examples) / len(examples))


def fuse_predictions(visual_scores, audio_scores, visual_quality=1.0, audio_quality=1.0, config=None):
    """Return calibrated vote and abstention, degrading to one valid modality."""
    cfg = (config or FusionConfig()).validate()
    for quality in (visual_quality, audio_quality):
        if not math.isfinite(float(quality)) or not 0 <= float(quality) <= 1:
            raise ValueError("modality quality must be in [0, 1]")
    visual = apply_temperature(visual_scores, cfg.visual_temperature) if visual_scores is not None and visual_quality >= cfg.min_quality else None
    audio = apply_temperature(audio_scores, cfg.audio_temperature) if audio_scores is not None and audio_quality >= cfg.min_quality else None
    if visual is None and audio is None:
        return {"presented_expression": UNKNOWN, "source": "none", "abstained": True,
                "scores": None, "reason": "insufficient_quality"}
    if visual is not None and audio is not None:
        weight_v = cfg.visual_weight * float(visual_quality)
        weight_a = (1 - cfg.visual_weight) * float(audio_quality)
        if weight_v + weight_a == 0:
            raise ValueError("fusion weights have zero mass")
        if weight_v == 0:
            scores, source, disagreement = audio, "audio", False
        elif weight_a == 0:
            scores, source, disagreement = visual, "visual", False
        else:
            scores = {label: (weight_v * visual[label] + weight_a * audio[label]) / (weight_v + weight_a)
                      for label in EXPRESSION_LABELS}
            source = "audio_visual"
            disagreement = best_label(visual) != best_label(audio)
    else:
        scores = visual if visual is not None else audio
        source = "visual" if visual is not None else "audio"
        disagreement = False
    ranked = sorted(scores.values(), reverse=True)
    confidence = ranked[0]
    margin = ranked[0] - ranked[1]
    reason = None
    if confidence < cfg.min_confidence or margin < cfg.min_margin:
        reason = "uncertain"
    elif disagreement and confidence < cfg.disagreement_confidence:
        reason = "modality_disagreement"
    return {"presented_expression": UNKNOWN if reason else best_label(scores), "source": source,
            "abstained": reason is not None, "scores": scores, "reason": reason}


def fit_fusion(validation_rows, weights=None, config=None):
    """Choose visual weight by validation macro-F1, counting abstentions as misses."""
    cfg = (config or FusionConfig()).validate()
    # The deployable default always uses nonzero evidence from both experts.
    # Callers can pass endpoint weights explicitly for unimodal ablations.
    candidates = weights if weights is not None else [i / 20 for i in range(2, 19)]
    best = None
    for weight in candidates:
        candidate = FusionConfig(visual_weight=weight,
                                 visual_temperature=cfg.visual_temperature,
                                 audio_temperature=cfg.audio_temperature,
                                 min_quality=cfg.min_quality,
                                 min_confidence=cfg.min_confidence,
                                 min_margin=cfg.min_margin,
                                 disagreement_confidence=cfg.disagreement_confidence).validate()
        truth, predictions = [], []
        for row in validation_rows:
            if row.get("split") != "validation":
                raise ValueError("fusion weights must be fitted on validation records only")
            target = row.get("multimodal_vote")
            if target not in EXPRESSION_LABELS:
                continue
            prediction = fuse_predictions(row.get("visual_scores"), row.get("audio_scores"),
                                          row.get("visual_quality", 1.0), row.get("audio_quality", 1.0), candidate)
            truth.append(target)
            predictions.append(prediction["presented_expression"])
        if not truth:
            raise ValueError("no labeled validation clips for fusion")
        f1_scores = []
        for label in EXPRESSION_LABELS:
            true_positives = sum(y == label and p == label for y, p in zip(truth, predictions))
            false_positives = sum(y != label and p == label for y, p in zip(truth, predictions))
            false_negatives = sum(y == label and p != label for y, p in zip(truth, predictions))
            denominator = 2 * true_positives + false_positives + false_negatives
            f1_scores.append(2 * true_positives / denominator if denominator else 0.0)
        macro_f1 = sum(f1_scores) / len(f1_scores)
        accuracy = sum(y == p for y, p in zip(truth, predictions)) / len(truth)
        key = (macro_f1, accuracy, -abs(weight - 0.5))
        if best is None or key > best[0]:
            best = (key, candidate)
    return best[1]
