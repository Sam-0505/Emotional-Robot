"""Architecture and clip eligibility for the joint audio-visual decoder."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class UnifiedConfig:
    visual_tokens_per_frame: int = 64
    audio_tokens: int = 16
    projector_hidden_size: int = 512
    lora_rank: int = 8
    lora_alpha: int = 16
    gradient_checkpointing: bool = True
    prompt_version: str = "cremad_joint_v1"

    def validate(self):
        for name in ("visual_tokens_per_frame", "audio_tokens", "projector_hidden_size",
                     "lora_rank", "lora_alpha"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("%s must be a positive integer" % name)
        side = math.isqrt(self.visual_tokens_per_frame)
        if side * side != self.visual_tokens_per_frame or self.visual_tokens_per_frame > 256:
            raise ValueError("visual_tokens_per_frame must be a square number <= 256")
        if self.audio_tokens > 128 or type(self.gradient_checkpointing) is not bool:
            raise ValueError("invalid audio token budget or checkpointing flag")
        if self.prompt_version != "cremad_joint_v1":
            raise ValueError("unsupported joint prompt version")
        return self

    def to_dict(self):
        return asdict(self.validate())

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8"))).validate()


def clip_quality(record):
    """Quality is derived from measurements, never from a target label."""
    frames = record.get("frame_quality", [])
    faces = [isinstance(item, dict) and item.get("face_status") == "detected" for item in frames]
    if len(faces) != 3:
        faces = [False] * 3
    activity = (record.get("audio_quality") or {}).get("active_frame_fraction", 0)
    speech = (type(activity) in (int, float) and math.isfinite(activity) and activity >= 0.05
              and record.get("audio_quality_status") == "accepted")
    accepted = (record.get("source_dataset") == "CREMA-D" and record.get("paired_usable") is True
                and record.get("av_alignment_status") == "accepted" and speech and sum(faces) >= 2)
    return {"accepted": bool(accepted), "faces": faces,
            "visual_quality": sum(faces) / 3, "audio_quality": float(bool(speech))}
