"""Clip-level audiovisual expression perception."""

from .fusion import FusionConfig, fuse_predictions, fit_fusion
from .labels import EXPRESSION_LABELS, UNKNOWN
from .metrics import evaluate_predictions

__all__ = [
    "EXPRESSION_LABELS",
    "UNKNOWN",
    "FusionConfig",
    "fuse_predictions",
    "fit_fusion",
    "evaluate_predictions",
]
