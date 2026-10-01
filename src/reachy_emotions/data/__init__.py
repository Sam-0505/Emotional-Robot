"""CREMA-D preparation and manifest validation."""

from .cremad import load_manifest, prepare_dataset, validate_manifest

__all__ = ["load_manifest", "prepare_dataset", "validate_manifest"]
