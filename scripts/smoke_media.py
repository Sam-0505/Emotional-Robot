"""Check one official CREMA-D video/WAV pair before full HPRC preparation.

The default check needs only Python's standard library. --decode additionally
exercises the production FFmpeg frame/audio extraction in a temporary folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reachy_emotions.data.cremad import (  # noqa: E402
    _extract, _lfs_pointer, _read_vote_details, _video_duration_ms, read_summary,
    validate_vote_metadata,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_pair(dataset_root: Path, video: Path, audio: Path, *, decode: bool = False) -> dict:
    """Verify identity, media bytes, votes, and optionally the FFmpeg path."""
    dataset_root, video, audio = map(lambda path: Path(path).resolve(), (dataset_root, video, audio))
    if video.stem != audio.stem:
        raise ValueError("video and WAV filenames must have the same CREMA-D clip ID")
    clip_id = video.stem
    if video.suffix.lower() != ".flv" or audio.suffix.lower() != ".wav":
        raise ValueError("expected one .flv video and its matching .wav audio")
    rows = read_summary(dataset_root)
    details = _read_vote_details(dataset_root)
    validate_vote_metadata(rows, details)
    match = [row for row in rows if row["clip_id"] == clip_id]
    if len(match) != 1:
        raise ValueError(f"clip {clip_id} is not in the official CREMA-D metadata")
    row = match[0]
    hashes = {}
    lfs_verified = {}
    for media_type, path in (("video", video), ("audio", audio)):
        if not path.is_file():
            raise FileNotFoundError(path)
        if _lfs_pointer(path) is not None:
            raise ValueError(f"{path} is a Git LFS pointer, not playable media")
        digest = _sha256(path)
        official = dataset_root / ("VideoFlash" if media_type == "video" else "AudioWAV") / path.name
        expected = None
        if official.is_file():
            expected = _lfs_pointer(official)
            if expected is not None and (digest, path.stat().st_size) != expected:
                raise ValueError(f"{media_type} does not match the official Git LFS pointer")
        hashes[media_type] = digest
        lfs_verified[media_type] = expected is not None
    with video.open("rb") as stream:
        header = stream.read(9)
    if len(header) != 9 or header[:4] != b"FLV\x01" or header[4] & 0x05 != 0x05:
        raise ValueError("video is not an FLV stream declaring both audio and video")
    with wave.open(str(audio), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
            raise ValueError("WAV must be mono PCM16 at 16 kHz")
        audio_duration_ms = round(wav.getnframes() / wav.getframerate() * 1000)
    if audio_duration_ms <= 0:
        raise ValueError("WAV contains no samples")
    report = {
        "clip_id": clip_id, "actor_id": row["actor_id"],
        "labels": {key: row[key] for key in ("face_vote", "voice_vote", "multimodal_vote")},
        "vote_agreement": {modality: details[(clip_id, modality)][0]
                           for modality in ("face", "voice", "multimodal")},
        "sha256": hashes, "lfs_hash_verified": lfs_verified,
        "audio_duration_ms": audio_duration_ms,
        "media_status": ("verified_bytes_and_metadata" if all(lfs_verified.values())
                         else "headers_and_metadata_only"),
        "decode_status": "not_requested",
    }
    if decode:
        duration_ms = _video_duration_ms(video)
        with tempfile.TemporaryDirectory(prefix="cremad-smoke-") as temp:
            extracted = _extract(video, audio, Path(temp), clip_id, duration_ms)
        report["video_duration_ms"] = duration_ms
        report["source_audio_video_duration_delta_ms"] = abs(audio_duration_ms - duration_ms)
        report["decode_status"] = "passed"
        report["extraction_quality"] = {
            key: extracted[key] for key in (
                "frame_timestamps_ms", "frame_quality", "audio_quality",
                "visual_usable", "audio_quality_status", "av_alignment_status", "paired_usable",
            )
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/CREMA-D"))
    parser.add_argument("--video", type=Path, default=Path("data/1001_DFA_ANG_XX.flv"))
    parser.add_argument("--audio", type=Path, default=Path("data/1001_DFA_ANG_XX.wav"))
    parser.add_argument("--decode", action="store_true", help="Exercise FFmpeg extraction in a temporary directory")
    args = parser.parse_args()
    try:
        report = inspect_pair(args.dataset_root, args.video, args.audio, decode=args.decode)
    except (OSError, RuntimeError, ValueError, KeyError, wave.Error) as exc:
        parser.exit(2, f"CREMA-D pair check failed: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
