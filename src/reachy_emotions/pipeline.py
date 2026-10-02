"""Bridge one synchronized CREMA-D clip to the guarded response harness."""

from __future__ import annotations

import time
import math
import hashlib
from pathlib import Path
from typing import Any, Mapping

from .perception.fusion import FusionConfig, fuse_predictions
from .perception.manifest import resolve_media_path


def make_observation(
    record: Mapping[str, Any],
    visual_result: Mapping[str, Any],
    audio_result: Mapping[str, Any],
    config: FusionConfig | None = None,
    *,
    observed_at: float | None = None,
) -> dict[str, Any]:
    """Fuse model evidence; preserve abstention and require both modalities for action.

    A single-modality result remains useful for evaluation, but the current robot
    guard accepts only an audio-visual observation from a synchronized clip.
    """
    if not record.get("sample_id"):
        raise ValueError("a manifest sample_id is required")
    if record.get("source_dataset") != "CREMA-D":
        raise ValueError("this prototype accepts CREMA-D records only")
    if record.get("split") not in {"train", "validation", "test"}:
        raise ValueError("a valid manifest split is required")
    visual_quality = float(visual_result["visual_quality"])
    audio_quality = float(audio_result["audio_quality"])
    fused = fuse_predictions(
        visual_result.get("scores"),
        audio_result.get("scores"),
        visual_quality=visual_quality,
        audio_quality=audio_quality,
        config=config,
    )
    audio_metadata = record.get("audio_quality") or {}
    speech_activity = audio_metadata.get("active_frame_fraction", 0.0)
    accepted = (
        fused["source"] == "audio_visual"
        and not fused["abstained"]
        and record.get("av_alignment_status") == "accepted"
        and record.get("audio_quality_status") == "accepted"
        and type(speech_activity) in (int, float)
        and math.isfinite(speech_activity)
        and speech_activity >= 0.05
        and visual_quality >= 2 / 3
        and record.get("paired_usable") is True
    )
    return {
        **fused,
        "input_origin": "crema_d",
        "observation_id": str(record["sample_id"]),
        "observed_at": time.time() if observed_at is None else observed_at,
        "quality": "accepted" if accepted else "rejected",
        "modality_quality": {"visual": visual_quality, "audio": audio_quality},
    }


def with_conversation_context(
    observation: Mapping[str, Any], transcript: str | None = None,
    recent_context: str | None = None,
) -> dict[str, Any]:
    """Attach short, explicitly supplied dialogue context for the reasoning agent."""
    result = dict(observation)
    for key, value in (("transcript", transcript), ("recent_context", recent_context)):
        if value is None:
            continue
        if (type(value) is not str or not value.strip() or len(value) > 500
                or any(ord(character) < 32 or ord(character) == 127 for character in value)):
            raise ValueError(f"{key} must be one printable line of at most 500 characters")
        result[key] = value.strip()
    return result


def make_unified_observation(record, result, calibration=None, *, observed_at=None):
    """Joint scores enter the same execution guard without late fusion."""
    from .perception.unified_config import clip_quality
    from .perception.unified_calibration import calibrated_prediction

    if not record.get("sample_id") or record.get("source_dataset") != "CREMA-D":
        raise ValueError("a CREMA-D manifest sample is required")
    quality = clip_quality(record)
    prediction = calibrated_prediction(result.get("scores"), calibration, quality["accepted"])
    return {**prediction, "input_origin": "crema_d", "observation_id": str(record["sample_id"]),
            "observed_at": time.time() if observed_at is None else observed_at,
            "quality": "accepted" if quality["accepted"] and not prediction["abstained"] else "rejected",
            "modality_quality": {"visual": quality["visual_quality"], "audio": quality["audio_quality"]}}


def infer_unified_manifest_record(manifest_path, record, model, calibration):
    from .perception.unified import file_hash, predict_unified_record

    if not model.reload_verified:
        raise ValueError("unified checkpoint must pass reload verification before integration")
    manifest_hash = file_hash(manifest_path)
    if manifest_hash != model.manifest_sha256:
        raise ValueError("unified manifest differs from training actor splits")
    calibration.check(model.checkpoint_id, manifest_hash)
    result = predict_unified_record(model, manifest_path, record)
    return make_unified_observation(record, result, calibration)


def infer_manifest_record(
    manifest_path: str | Path,
    record: Mapping[str, Any],
    visual_expert: tuple[Any, Any, Any],
    audio_expert: tuple[Any, Any, Any, str],
    config: FusionConfig | None = None,
) -> dict[str, Any]:
    """Run both actual experts on one paired manifest record, then fuse them."""
    from .perception.audio import predict_audio_clip
    from .perception.visual import predict_visual_clip

    frames = [resolve_media_path(manifest_path, frame) for frame in record["frame_paths"]]
    audio = resolve_media_path(manifest_path, record["audio_path"])
    if len(frames) != 3 or not all(frame.is_file() for frame in frames) or not audio.is_file():
        raise FileNotFoundError("paired manifest media is missing")
    expected_hashes = list(record.get("frame_sha256s", [])) + [record.get("audio_sha256")]
    if len(expected_hashes) != 4 or any(not isinstance(value, str) or len(value) != 64
                                        for value in expected_hashes):
        raise ValueError("paired manifest media hashes are missing")
    for path, expected in zip(frames + [audio], expected_hashes):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError("prepared media hash does not match manifest: %s" % path.name)
    frame_metadata = record.get("frame_quality", [])
    # Full frames and missed faces are useful as audit samples, not evidence
    # strong enough to authorize a robot response.
    face_quality = [1.0 if isinstance(item, dict) and item.get("face_status") == "detected" else 0.0
                    for item in frame_metadata]
    if len(face_quality) != 3:
        face_quality = [0.0, 0.0, 0.0]
    visual_result = predict_visual_clip(*visual_expert, frames, face_quality)
    audio_result = predict_audio_clip(*audio_expert, audio)
    return make_observation(record, visual_result, audio_result, config)
