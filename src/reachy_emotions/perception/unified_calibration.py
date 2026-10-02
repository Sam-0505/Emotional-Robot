"""Validation-only calibration for joint decoder likelihood scores."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .fusion import apply_temperature, fit_temperature
from .labels import UNKNOWN, best_label
from .metrics import evaluate_predictions
from .unified_config import clip_quality


@dataclass(frozen=True)
class UnifiedCalibration:
    temperature: float
    min_confidence: float
    min_margin: float
    checkpoint_id: str
    manifest_sha256: str
    selection_split: str = "validation"

    def validate(self):
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("invalid unified temperature")
        if any(not math.isfinite(value) or not 0 <= value <= 1
               for value in (self.min_confidence, self.min_margin)):
            raise ValueError("invalid unified abstention threshold")
        if self.selection_split != "validation" or any(
                len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
                for value in (self.checkpoint_id, self.manifest_sha256)):
            raise ValueError("invalid unified calibration provenance")
        return self

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self.validate()), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8"))).validate()

    def check(self, checkpoint_id, manifest_sha256):
        self.validate()
        if self.checkpoint_id != checkpoint_id or self.manifest_sha256 != manifest_sha256:
            raise ValueError("calibration does not match unified checkpoint or manifest")


def calibrated_prediction(scores, calibration, quality_accepted=True):
    result = {"presented_expression": UNKNOWN, "source": "audio_visual", "abstained": True,
              "scores": None, "model_kind": "unified_audio_visual", "calibrated": calibration is not None}
    if not quality_accepted or scores is None:
        return {**result, "reason": "insufficient_paired_quality"}
    if calibration is None:
        return {**result, "reason": "uncalibrated_joint_model"}
    calibration.validate()
    probabilities = apply_temperature(scores, calibration.temperature)
    ranked = sorted(probabilities.values(), reverse=True)
    uncertain = ranked[0] < calibration.min_confidence or ranked[0] - ranked[1] < calibration.min_margin
    return {**result, "scores": probabilities, "abstained": uncertain,
            "presented_expression": UNKNOWN if uncertain else best_label(probabilities),
            "reason": "uncertain" if uncertain else None}


def fit_unified_calibration(rows, checkpoint_id, manifest_sha256):
    if not rows or any(row.get("split") != "validation" for row in rows):
        raise ValueError("unified calibration requires validation records only")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate calibration sample")
    usable = [row for row in rows if clip_quality(row)["accepted"]]
    temperature = fit_temperature(usable, "scores", "multimodal_vote")
    best = None
    for confidence in (0.0, 0.3, 0.42, 0.55, 0.7):
        for margin in (0.0, 0.05, 0.1, 0.2):
            config = UnifiedCalibration(temperature, confidence, margin, checkpoint_id, manifest_sha256).validate()
            predictions = [{**row, "presented_expression": calibrated_prediction(
                row.get("scores"), config, clip_quality(row)["accepted"])["presented_expression"]} for row in rows]
            metrics = evaluate_predictions(predictions, "multimodal_vote")
            key = (metrics["macro_f1"], metrics["ambiguous_abstention_rate"] or 0,
                   metrics["accuracy"], confidence, margin)
            if best is None or key > best[0]:
                best = (key, config)
    return best[1]
