"""Prepare paired CREMA-D face frames and speech for actor-disjoint experiments.

Paths in the manifest are relative to the output directory. The source dataset
must be obtained separately from the official CREMA-D repository with Git LFS.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import wave
from array import array
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


VOTE_CODES = {"A": "ANG", "D": "DIS", "F": "FEA", "H": "HAP", "N": "NEU", "S": "SAD"}
LABELS = frozenset(VOTE_CODES.values())
FRAME_FRACTIONS = (0.35, 0.50, 0.65)
PREPROCESSING_VERSION = "cremad-av-v6"
SPLIT_STRATEGY = "actor_demographic_marginals_256_v1"
SOURCE_URL = "https://github.com/CheyneyComputerScience/CREMA-D"
AUDIO_TARGET_RMS = 0.1
AUDIO_MAX_GAIN = 4.0
AUDIO_PEAK_CEILING = 0.95


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required executable is missing: {command[0]}. Install FFmpeg (ffmpeg and ffprobe).") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "unknown error").strip()
        raise RuntimeError(f"{command[0]} failed: {detail}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vote(value: str | None) -> str:
    """A tie or absent official vote is an abstention, not a class label."""
    if not value or not value.strip() or value.strip().upper() in {"NA", "N/A"}:
        return "unknown"
    code = value.strip().upper()
    if ":" in code:
        if all(part in VOTE_CODES for part in code.split(":")):
            return "unknown"
        raise ValueError(f"Invalid CREMA-D vote: {value!r}")
    if code not in VOTE_CODES:
        raise ValueError(f"Invalid CREMA-D vote: {value!r}")
    return VOTE_CODES[code]


def _clip_parts(clip_id: str) -> tuple[str, str]:
    parts = clip_id.split("_")
    if len(parts) != 4 or len(parts[0]) != 4 or not parts[0].isdigit() or parts[2] not in LABELS:
        raise ValueError(f"Invalid CREMA-D clip name: {clip_id!r}")
    return parts[0], parts[2]


def read_summary(dataset_root: Path) -> list[dict[str, Any]]:
    """Read official summaryTable.csv; the columns called Level are not agreement."""
    path = dataset_root / "processedResults" / "summaryTable.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing official CREMA-D vote table: {path}")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"FileName", "FaceVote", "VoiceVote", "MultiModalVote"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        for row in reader:
            clip_id = (row["FileName"] or "").strip()
            actor_id, performed = _clip_parts(clip_id)
            if clip_id in seen:
                raise ValueError(f"Duplicate clip in summaryTable.csv: {clip_id}")
            seen.add(clip_id)
            records.append({
                "clip_id": clip_id,
                "actor_id": actor_id,
                "performed_label": performed,
                "face_vote_raw": (row["FaceVote"] or "").strip(),
                "voice_vote_raw": (row["VoiceVote"] or "").strip(),
                "multimodal_vote_raw": (row["MultiModalVote"] or "").strip(),
                "face_vote": _vote(row["FaceVote"]),
                "voice_vote": _vote(row["VoiceVote"]),
                "multimodal_vote": _vote(row["MultiModalVote"]),
            })
    if not records:
        raise ValueError(f"No clips in {path}")
    return records


def _read_vote_details(dataset_root: Path) -> dict[tuple[str, str], tuple[float, str]]:
    """Use official tabulatedVotes row prefixes: 1 voice, 2 face, 3 combined."""
    path = dataset_root / "processedResults" / "tabulatedVotes.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing official CREMA-D vote counts: {path}")
    modes = {"1": "voice", "2": "face", "3": "multimodal"}
    details: dict[tuple[str, str], tuple[float, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        count_columns = tuple(VOTE_CODES)
        if not {"fileName", "agreement", "emoVote", "numResponses", *count_columns}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} is missing required columns")
        index_column = (reader.fieldnames or [""])[0]
        for row in reader:
            row_id = (row[index_column] or "").strip()
            modality = modes.get(row_id[:1])
            if modality is None:
                raise ValueError(f"Unexpected vote row ID {row_id!r} in {path}")
            key = ((row["fileName"] or "").strip(), modality)
            if key in details:
                raise ValueError(f"Duplicate {modality} agreement for {key[0]}")
            value = float(row["agreement"])
            if not 0 <= value <= 1:
                raise ValueError(f"Agreement outside [0, 1] for {key[0]}")
            counts = {code: int(row[code]) for code in count_columns}
            total = int(row["numResponses"])
            if total <= 0 or any(count < 0 for count in counts.values()) or sum(counts.values()) != total:
                raise ValueError(f"Invalid vote counts for {key[0]} ({modality})")
            winners = {code for code, count in counts.items() if count == max(counts.values())}
            vote = (row["emoVote"] or "").strip().upper()
            if set(vote.split(":")) != winners or abs(value - max(counts.values()) / total) > 1e-5:
                raise ValueError(f"Vote or agreement conflicts with counts for {key[0]} ({modality})")
            details[key] = (value, vote)
    return details


def read_agreements(dataset_root: Path) -> dict[tuple[str, str], float]:
    return {key: detail[0] for key, detail in _read_vote_details(dataset_root).items()}


def read_demographics(dataset_root: Path, actor_ids: Iterable[str]) -> dict[str, dict[str, str | int]]:
    """Read official actor metadata for split balancing, never model input."""
    path = dataset_root / "VideoDemographics.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing official CREMA-D actor demographics: {path}")
    records: dict[str, dict[str, str | int]] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"ActorID", "Age", "Sex", "Race", "Ethnicity"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} is missing required columns")
        for row in reader:
            actor_id = (row["ActorID"] or "").strip()
            if actor_id in records:
                raise ValueError(f"Duplicate actor demographics for {actor_id}")
            age = int(row["Age"])
            if not actor_id.isdigit() or not 18 <= age <= 100:
                raise ValueError(f"Invalid actor demographics for {actor_id}")
            values = {key: (row[key] or "").strip() for key in ("Sex", "Race", "Ethnicity")}
            if not all(values.values()):
                raise ValueError(f"Incomplete actor demographics for {actor_id}")
            records[actor_id] = {"age": age, **values}
    expected = set(actor_ids)
    if set(records) != expected:
        raise ValueError(f"Actor demographics disagree with clips: {len(expected - set(records))} missing, "
                         f"{len(set(records) - expected)} extra")
    return records


def validate_vote_metadata(rows: list[dict[str, Any]], details: dict[tuple[str, str], tuple[float, str]]) -> None:
    """Catch swapped modalities, missing rows, and mismatched summary votes."""
    if not details:
        return
    expected = {(row["clip_id"], modality) for row in rows for modality in ("face", "voice", "multimodal")}
    if set(details) != expected:
        missing = expected - set(details)
        extra = set(details) - expected
        raise ValueError(f"Vote tables disagree: {len(missing)} missing and {len(extra)} extra modality rows")
    for row in rows:
        for modality in ("face", "voice", "multimodal"):
            summary_vote = row[f"{modality}_vote_raw"].upper()
            detailed_vote = details[(row["clip_id"], modality)][1]
            if set(summary_vote.split(":")) != set(detailed_vote.split(":")):
                raise ValueError(f"Vote tables disagree for {row['clip_id']} ({modality})")


def split_balance(rows: list[dict[str, Any]], splits: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Report class/tie distribution before fitting any model."""
    report: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        group = [row for row in rows if splits[row["actor_id"]] == split]
        report[split] = {
            "actors": len({row["actor_id"] for row in group}),
            "clips": len(group),
            "performed_label": dict(sorted(Counter(row["performed_label"] for row in group).items())),
            "face_vote": dict(sorted(Counter(row["face_vote"] for row in group).items())),
            "voice_vote": dict(sorted(Counter(row["voice_vote"] for row in group).items())),
            "multimodal_vote": dict(sorted(Counter(row["multimodal_vote"] for row in group).items())),
        }
    return report


def _age_band(age: int) -> str:
    return f"{min(age // 10 * 10, 70)}+" if age >= 70 else f"{age // 10 * 10}s"


def demographic_balance(demographics: dict[str, dict[str, str | int]],
                        splits: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Aggregate only, so actor attributes do not enter model artifacts."""
    report = {}
    for split in ("train", "validation", "test"):
        group = [demographics[actor] for actor in sorted(splits) if splits[actor] == split]
        report[split] = {
            "actors": len(group),
            "sex": dict(sorted(Counter(str(row["Sex"]) for row in group).items())),
            "race": dict(sorted(Counter(str(row["Race"]) for row in group).items())),
            "ethnicity": dict(sorted(Counter(str(row["Ethnicity"]) for row in group).items())),
            "age_band": dict(sorted(Counter(_age_band(int(row["age"])) for row in group).items())),
        }
    return report


def actor_splits(actor_ids: Iterable[str], seed: int = 42,
                 demographics: dict[str, dict[str, str | int]] | None = None) -> dict[str, str]:
    actors = sorted(set(actor_ids))
    if not actors:
        raise ValueError("Cannot split an empty actor set")
    count = len(actors)
    if count >= 3:
        train_count = min(count - 2, max(1, round(count * 64 / 91)))
        validation_count = min(count - train_count - 1, max(1, round(count * 13 / 91)))
    elif count == 2:
        train_count, validation_count = 1, 0
    else:
        train_count, validation_count = 1, 0
    def assign(order: list[str]) -> dict[str, str]:
        return {actor: ("train" if index < train_count else "validation"
                        if index < train_count + validation_count else "test")
                for index, actor in enumerate(order)}

    rng = random.Random(seed)
    if demographics is None:
        rng.shuffle(actors)
        return assign(actors)
    if set(demographics) != set(actors):
        raise ValueError("demographics must cover exactly the split actors")
    features = {}
    for actor in actors:
        row = demographics[actor]
        features[actor] = (("sex", str(row["Sex"])), ("race", str(row["Race"])),
                           ("ethnicity", str(row["Ethnicity"])), ("age_band", _age_band(int(row["age"]))))
    totals = Counter(feature for actor in actors for feature in features[actor])
    sizes = {"train": train_count, "validation": validation_count,
             "test": count - train_count - validation_count}
    best_score, best_splits = float("inf"), None
    for _ in range(256):
        order = list(actors)
        rng.shuffle(order)
        candidate = assign(order)
        observed = Counter((candidate[actor], feature) for actor in actors for feature in features[actor])
        score = 0.0
        for feature, total in totals.items():
            for split, size in sizes.items():
                if size == 0:
                    continue
                expected = total * size / count
                actual = observed[(split, feature)]
                score += (actual - expected) ** 2 / max(expected, 1.0)
                if split != "train" and total >= 3 and actual == 0:
                    score += 2.0
        if score < best_score:
            best_score, best_splits = score, candidate
    return best_splits


def _video_duration_ms(path: Path) -> int:
    result = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)])
    try:
        duration = round(float(result.stdout.strip()) * 1000)
    except ValueError as exc:
        raise RuntimeError(f"Could not determine video duration: {path}") from exc
    if duration <= 0:
        raise RuntimeError(f"Video has no positive duration: {path}")
    return duration


def _lfs_pointer(path: Path) -> tuple[str, int] | None:
    with path.open("rb") as stream:
        data = stream.read(200)
    if not data.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
        return None
    values = dict(line.split(" ", 1) for line in data.decode("ascii").splitlines()[1:] if " " in line)
    oid = values.get("oid", "")
    digest = oid[7:] if oid.startswith("sha256:") else ""
    size = values.get("size", "")
    if len(digest) != 64 or not size.isdigit():
        raise RuntimeError(f"Git LFS pointer is malformed: {path}")
    return digest, int(size)


def _media_source(dataset_root: Path, clip_id: str, kind: str, override_dir: Path | None = None) -> Path:
    if kind == "video":
        folder, suffixes = "VideoFlash", (".flv", ".FLV")
    elif kind == "audio":
        folder, suffixes = "AudioWAV", (".wav", ".WAV")
    else:
        raise ValueError(f"Unknown CREMA-D media kind: {kind}")
    official_dir = dataset_root / folder
    source_dir = override_dir or official_dir
    for suffix in suffixes:
        path = source_dir / f"{clip_id}{suffix}"
        if path.is_file():
            if _lfs_pointer(path) is not None:
                raise RuntimeError(f"{path} is a Git LFS pointer. Run git lfs pull in the official CREMA-D checkout.")
            if source_dir.resolve() != official_dir.resolve():
                official = official_dir / path.name
                if not official.is_file():
                    raise FileNotFoundError(f"Cannot verify override against official CREMA-D media: {official}")
                expected = _lfs_pointer(official)
                expected_hash = expected[0] if expected else _sha256(official)
                expected_size = expected[1] if expected else official.stat().st_size
                if (_sha256(path), path.stat().st_size) != (expected_hash, expected_size):
                    raise ValueError(f"{kind.capitalize()} override does not match official CREMA-D media: {path}")
            return path
    raise FileNotFoundError(f"Missing official {kind}: {source_dir / (clip_id + suffixes[0])}")


def _source_commit(dataset_root: Path) -> str | None:
    if not (dataset_root / ".git").exists() or shutil.which("git") is None:
        return None
    result = subprocess.run(["git", "-C", str(dataset_root), "rev-parse", "HEAD"], capture_output=True, text=True)
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and len(commit) == 40 else None


def _tool_versions() -> dict[str, str | None]:
    ffmpeg = _run(["ffmpeg", "-version"]).stdout.splitlines()[0]
    ffprobe = _run(["ffprobe", "-version"]).stdout.splitlines()[0]
    try:
        import cv2  # type: ignore[import-not-found]
        opencv = cv2.__version__
    except ImportError:
        opencv = None
    return {"ffmpeg": ffmpeg, "ffprobe": ffprobe, "opencv": opencv}


def _face_crop(path: Path, padding: float = 0.20) -> dict[str, Any]:
    """Align face by eye centers when possible, then crop with head context.

    OpenCV is optional. A failed detection retains the full frame and records
    quality status so callers can exclude it from face-only training.
    """
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return {"face_status": "opencv_unavailable", "aligned": False}
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"OpenCV could not decode frame: {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
    if cascade.empty():
        raise RuntimeError("OpenCV face cascade is unavailable")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    if len(faces) == 0:
        return {"face_status": "not_detected", "aligned": False, "blur_variance": blur_variance}
    x, y, width, height = max(faces, key=lambda box: int(box[2]) * int(box[3]))
    x, y, width, height = map(int, (x, y, width, height))
    aligned = False
    eye_cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_eye.xml"))
    if not eye_cascade.empty():
        eyes = eye_cascade.detectMultiScale(gray[y:y + height // 2, x:x + width], scaleFactor=1.1, minNeighbors=5)
        eyes = sorted(eyes, key=lambda box: int(box[2]) * int(box[3]), reverse=True)[:2]
        if len(eyes) == 2:
            left, right = sorted(eyes, key=lambda box: box[0])
            p1 = (x + left[0] + left[2] / 2, y + left[1] + left[3] / 2)
            p2 = (x + right[0] + right[2] / 2, y + right[1] + right[3] / 2)
            angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
            if abs(angle) <= 20:
                center = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                aligned = True
    extra = round(max(width, height) * padding)
    left = max(0, x - extra)
    top = max(0, y - extra)
    right = min(image.shape[1], x + width + extra)
    bottom = min(image.shape[0], y + height + extra)
    crop = image[top:bottom, left:right]
    if crop.size == 0 or not cv2.imwrite(str(path), crop):
        raise RuntimeError(f"Could not write face crop: {path}")
    return {"face_status": "detected", "aligned": aligned, "face_box_xywh": [x, y, width, height], "blur_variance": blur_variance}


def _audio_quality(path: Path) -> dict[str, Any]:
    """Simple signal diagnostics; energy contrast is not a calibrated SNR."""
    with wave.open(str(path), "rb") as wav:
        samples = array("h")
        samples.frombytes(wav.readframes(wav.getnframes()))
        if sys.byteorder != "little":
            samples.byteswap()
    if not samples:
        raise RuntimeError(f"No audio samples in {path}")
    block_size = 320  # 20 ms at 16 kHz
    block_rms = []
    for start in range(0, len(samples), block_size):
        block = samples[start:start + block_size]
        block_rms.append(math.sqrt(sum(value * value for value in block) / len(block)) / 32768)
    noise_floor = sorted(block_rms)[max(0, len(block_rms) // 10 - 1)]
    threshold = max(0.01, 3 * noise_floor)
    active_blocks = [level for level in block_rms if level >= threshold]
    level_threshold = max(0.002, 0.25 * max(block_rms))
    level_blocks = [level for level in block_rms if level >= level_threshold]
    clipping_fraction = sum(abs(value) >= 32760 for value in samples) / len(samples)
    energy_contrast_db = 20 * math.log10((max(block_rms) + 1e-8) / (noise_floor + 1e-8))
    return {
        "rms": round(math.sqrt(sum(value * value for value in samples) / len(samples)) / 32768, 6),
        "active_rms": round(math.sqrt(sum(level * level for level in level_blocks) / len(level_blocks)), 6)
                      if level_blocks else 0.0,
        "clipping_fraction": round(clipping_fraction, 6),
        "active_frame_fraction": round(len(active_blocks) / len(block_rms), 6),
        "energy_contrast_db": round(energy_contrast_db, 2),
    }


def _normalize_audio_volume(path: Path, source_quality: dict[str, Any] | None = None) -> dict[str, float]:
    """Use high-energy block level to set one bounded gain without changing timing."""
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getframerate() != 16000 or wav.getsampwidth() != 2:
            raise RuntimeError(f"Unexpected WAV format: {path}")
        params = wav.getparams()
        samples = array("h")
        samples.frombytes(wav.readframes(wav.getnframes()))
    if not samples:
        raise RuntimeError(f"No audio samples in {path}")
    if sys.byteorder != "little":
        samples.byteswap()
    active_rms = (source_quality or _audio_quality(path))["active_rms"]
    peak = max(abs(value) for value in samples) / 32768
    if active_rms and peak:
        gain = min(AUDIO_TARGET_RMS / active_rms, AUDIO_MAX_GAIN, AUDIO_PEAK_CEILING / peak)
    else:
        gain = min(1.0, AUDIO_PEAK_CEILING / peak) if peak else 1.0
    peak_limit = math.floor(AUDIO_PEAK_CEILING * 32768)
    normalized = array("h", (max(-peak_limit, min(peak_limit, round(value * gain))) for value in samples))
    if sys.byteorder != "little":
        normalized.byteswap()
    pending = path.with_name(path.stem + ".normalized.wav")
    try:
        with wave.open(str(pending), "wb") as wav:
            wav.setparams(params)
            wav.writeframes(normalized.tobytes())
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)
    return {"target_rms": AUDIO_TARGET_RMS, "max_gain": AUDIO_MAX_GAIN,
            "peak_ceiling": AUDIO_PEAK_CEILING, "applied_gain": round(gain, 6)}


def _visual_usable(frame_quality: list[dict[str, Any]]) -> bool:
    return sum(item.get("face_status") == "detected" for item in frame_quality) >= 2


def _extract(video: Path, source_audio: Path, output_root: Path, clip_id: str,
             duration_ms: int) -> dict[str, Any]:
    frame_dir = output_root / "frames" / clip_id
    audio_dir = output_root / "audio"
    frame_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    timestamps = [round(duration_ms * fraction) for fraction in FRAME_FRACTIONS]
    if len(set(timestamps)) != 3:
        raise RuntimeError(f"Video too short to sample three distinct timestamps: {video}")
    frame_paths: list[str] = []
    frame_hashes: list[str] = []
    frame_quality: list[dict[str, Any]] = []
    for index, timestamp in enumerate(timestamps):
        frame = frame_dir / f"frame_{index}.png"
        _run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-ss", f"{timestamp / 1000:.3f}", "-frames:v", "1", str(frame)])
        if not frame.is_file() or frame.stat().st_size == 0:
            raise RuntimeError(f"No frame extracted at {timestamp} ms from {video}")
        frame_quality.append(_face_crop(frame))
        frame_paths.append(frame.relative_to(output_root).as_posix())
        frame_hashes.append(_sha256(frame))
    audio = audio_dir / f"{clip_id}.wav"
    _run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source_audio), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(audio)])
    with wave.open(str(audio), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getframerate() != 16000 or wav.getsampwidth() != 2:
            raise RuntimeError(f"Unexpected WAV format: {audio}")
        audio_duration_ms = round(wav.getnframes() / wav.getframerate() * 1000)
    if audio_duration_ms <= 0:
        raise RuntimeError(f"No audio samples extracted from {video}")
    alignment_delta_ms = abs(audio_duration_ms - duration_ms)
    source_audio_quality = _audio_quality(audio)
    audio_normalization = _normalize_audio_volume(audio, source_audio_quality)
    audio_quality = _audio_quality(audio)
    visual_usable = _visual_usable(frame_quality)
    audio_review = (source_audio_quality["active_rms"] < 0.01
                    or source_audio_quality["active_frame_fraction"] < 0.05
                    or source_audio_quality["clipping_fraction"] > 0.01)
    audio_quality_status = "review" if audio_review else "accepted"
    av_alignment_status = "accepted" if alignment_delta_ms <= 500 else "review"
    return {
        "frame_paths": frame_paths,
        "frame_timestamps_ms": timestamps,
        "frame_sha256s": frame_hashes,
        "frame_quality": frame_quality,
        "visual_usable": visual_usable,
        "audio_path": audio.relative_to(output_root).as_posix(),
        "audio_sha256": _sha256(audio),
        "audio_sample_rate": 16000,
        "audio_duration_ms": audio_duration_ms,
        "audio_quality_before_normalization": source_audio_quality,
        "audio_normalization": audio_normalization,
        "audio_quality": audio_quality,
        "audio_quality_status": audio_quality_status,
        "av_duration_delta_ms": alignment_delta_ms,
        "av_alignment_status": av_alignment_status,
        "paired_usable": visual_usable and audio_quality_status == "accepted" and av_alignment_status == "accepted",
    }


def prepare_dataset(dataset_root: Path | str, output_root: Path | str, *, dry_run: bool = False,
                    seed: int = 42, limit: int | None = None, clip_ids: Iterable[str] | None = None,
                    video_dir: Path | str | None = None,
                    audio_dir: Path | str | None = None) -> dict[str, Any]:
    """Prepare paired examples and return counts/paths; dry-run writes nothing."""
    dataset_root = Path(dataset_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"CREMA-D directory does not exist: {dataset_root}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    requested = list(clip_ids) if clip_ids is not None else None
    if requested is not None and (not requested or len(requested) != len(set(requested))):
        raise ValueError("clip IDs must be nonempty and unique")
    if requested is not None and limit is not None:
        raise ValueError("choose clip IDs or limit, not both")
    if (video_dir is not None or audio_dir is not None) and requested is None:
        raise ValueError("custom media directories require explicit clip IDs")
    resolved_video_dir = Path(video_dir).expanduser().resolve() if video_dir is not None else None
    resolved_audio_dir = Path(audio_dir).expanduser().resolve() if audio_dir is not None else None
    rows = read_summary(dataset_root)
    demographics = read_demographics(dataset_root, (row["actor_id"] for row in rows))
    splits = actor_splits((row["actor_id"] for row in rows), seed, demographics)
    vote_details = _read_vote_details(dataset_root)
    validate_vote_metadata(rows, vote_details)
    agreements = {key: item[0] for key, item in vote_details.items()}
    metadata_sha256 = _sha256(dataset_root / "processedResults" / "summaryTable.csv")
    vote_counts_sha256 = _sha256(dataset_root / "processedResults" / "tabulatedVotes.csv")
    demographics_sha256 = _sha256(dataset_root / "VideoDemographics.csv")
    license_sha256 = _sha256(dataset_root / "LICENSE.txt")
    source_commit = _source_commit(dataset_root)
    selected = [row for row in rows if row["clip_id"] in set(requested)] if requested is not None else rows[:limit]
    if requested is not None and len(selected) != len(requested):
        missing = sorted(set(requested) - {row["clip_id"] for row in selected})
        raise ValueError(f"Unknown CREMA-D clip IDs: {missing}")
    # Fail before writing any output when the checkout is incomplete.
    media = {row["clip_id"]: (
        _media_source(dataset_root, row["clip_id"], "video", resolved_video_dir),
        _media_source(dataset_root, row["clip_id"], "audio", resolved_audio_dir),
    ) for row in selected}
    summary: dict[str, Any] = {
        "clips": len(selected),
        "total_metadata_clips": len(rows),
        "actors": len(splits),
        "split_actors": {name: sorted(actor for actor, split in splits.items() if split == name) for name in ("train", "validation", "test")},
        "split_clips": dict(Counter(splits[row["actor_id"]] for row in selected)),
        "vote_agreement_available": bool(agreements),
        "split_balance": split_balance(rows, splits),
        "demographic_balance": demographic_balance(demographics, splits),
        "selected_split_balance": split_balance(selected, splits),
        "metadata_sha256": {"summaryTable.csv": metadata_sha256, "tabulatedVotes.csv": vote_counts_sha256,
                            "VideoDemographics.csv": demographics_sha256},
        "source_license_sha256": license_sha256,
        "source_commit": source_commit,
        "selected_clip_ids": [row["clip_id"] for row in selected] if requested is not None else None,
        "video_source_dir": str(resolved_video_dir) if resolved_video_dir is not None else "VideoFlash",
        "audio_source_dir": str(resolved_audio_dir) if resolved_audio_dir is not None else "AudioWAV",
        "dry_run": dry_run,
        "manifest_path": str(output_root / "manifest.jsonl"),
    }
    if dry_run:
        return summary
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"Required executable is missing: {executable}. Install FFmpeg (ffmpeg and ffprobe).")
    versions = _tool_versions()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / "manifest.jsonl"
    if manifest.exists():
        raise FileExistsError(f"Manifest already exists at {manifest}; use a fresh output directory for a new preparation run")
    pending_manifest = output_root / "manifest.jsonl.pending"
    records: list[dict[str, Any]] = []
    try:
        with pending_manifest.open("w", encoding="utf-8") as stream:
            for row in selected:
                clip_id = row["clip_id"]
                video, source_audio = media[clip_id]
                try:
                    video_path = video.relative_to(dataset_root).as_posix()
                except ValueError:
                    video_path = str(video)
                try:
                    source_audio_path = source_audio.relative_to(dataset_root).as_posix()
                except ValueError:
                    source_audio_path = str(source_audio)
                duration_ms = _video_duration_ms(video)
                extracted = _extract(video, source_audio, output_root, clip_id, duration_ms)
                record = {
                    "sample_id": clip_id,
                    "source_dataset": "CREMA-D",
                    "source_url": SOURCE_URL,
                    "source_version_or_commit": source_commit or f"summaryTable-sha256:{metadata_sha256}",
                    "source_metadata_sha256": summary["metadata_sha256"],
                    "license": "ODbL-1.0 / DbCL-1.0",
                    "actor_id": row["actor_id"],
                    "clip_id": clip_id,
                    "video_path": video_path,
                    "video_sha256": _sha256(video),
                    "source_audio_path": source_audio_path,
                    "source_audio_sha256": _sha256(source_audio),
                    "video_duration_ms": duration_ms,
                    "performed_label": row["performed_label"],
                    "split": splits[row["actor_id"]],
                    "preprocessing_version": PREPROCESSING_VERSION,
                    "prompt_version": "unset",
                    **{key: row[key] for key in ("face_vote", "voice_vote", "multimodal_vote", "face_vote_raw", "voice_vote_raw", "multimodal_vote_raw")},
                    "face_vote_agreement": agreements.get((clip_id, "face")),
                    "voice_vote_agreement": agreements.get((clip_id, "voice")),
                    "multimodal_vote_agreement": agreements.get((clip_id, "multimodal")),
                    "visual_target_json": {"presented_expression": row["face_vote"]},
                    "audio_target": row["voice_vote"],
                    "fusion_target": row["multimodal_vote"],
                    **extracted,
                }
                records.append(record)
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        validate_manifest(records, root=output_root, verify_files=True)
        pending_manifest.replace(manifest)
        (output_root / "actor_splits.json").write_text(json.dumps(summary["split_actors"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output_root / "split_balance.json").write_text(json.dumps(summary["split_balance"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output_root / "demographic_balance.json").write_text(
            json.dumps(summary["demographic_balance"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        attribution = {
            "dataset": "CREMA-D", "source_url": SOURCE_URL, "source_commit": source_commit,
            "source_license_file": "LICENSE.txt", "source_license_sha256": license_sha256,
            "database_license": "ODbL-1.0", "contents_license": "DbCL-1.0",
            "source_metadata_sha256": summary["metadata_sha256"],
            "notice": "CREMA-D media and metadata are third-party materials; do not commit raw or prepared media.",
        }
        (output_root / "dataset_attribution.json").write_text(
            json.dumps(attribution, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        provenance = {
            "source_url": SOURCE_URL,
            "source_commit": source_commit,
            "metadata_sha256": summary["metadata_sha256"],
            "source_license_sha256": license_sha256,
            "preprocessing_version": PREPROCESSING_VERSION,
            "frame_fractions": FRAME_FRACTIONS,
            "face_padding_fraction": 0.20,
            "audio_sample_rate": 16000,
            "audio_volume_normalization": {
                "basis": "high_energy_20ms_rms", "target_rms": AUDIO_TARGET_RMS,
                "max_gain": AUDIO_MAX_GAIN, "peak_ceiling": AUDIO_PEAK_CEILING,
            },
            "actor_split_seed": seed,
            "actor_split_strategy": SPLIT_STRATEGY,
            "selected_clip_ids": summary["selected_clip_ids"],
            "video_source_dir": summary["video_source_dir"],
            "audio_source_dir": summary["audio_source_dir"],
            "tool_versions": versions,
        }
        (output_root / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception:
        pending_manifest.unlink(missing_ok=True)
        raise
    summary["quality_counts"] = {
        "visual_usable": sum(bool(row["visual_usable"]) for row in records),
        "paired_usable": sum(bool(row["paired_usable"]) for row in records),
        "alignment_review": sum(row["av_alignment_status"] == "review" for row in records),
        "audio_review": sum(row["audio_quality_status"] == "review" for row in records),
    }
    return summary


def load_manifest(path: Path | str) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_manifest(records: Iterable[dict[str, Any]], *, root: Path | str | None = None, verify_files: bool = False) -> dict[str, int]:
    """Reject actor/clip leakage, malformed AV pairs, and stale file hashes."""
    base = Path(root).resolve() if root is not None else None
    if verify_files and base is None:
        raise ValueError("root is required when verify_files=True")
    actor_split: dict[str, str] = {}
    seen: set[str] = set()
    seen_media: set[str] = set()
    split_counts: Counter[str] = Counter()
    for row in records:
        sample_id = str(row["sample_id"])
        actor_id = str(row["actor_id"])
        split = row["split"]
        if row.get("source_dataset") != "CREMA-D":
            raise ValueError(f"Unsupported source dataset for {sample_id}")
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split for {sample_id}: {split}")
        if sample_id in seen:
            raise ValueError(f"Repeated sample ID: {sample_id}")
        seen.add(sample_id)
        if sample_id != row["clip_id"]:
            raise ValueError(f"Sample ID does not match clip ID: {sample_id}")
        if actor_id in actor_split and actor_split[actor_id] != split:
            raise ValueError(f"Actor leakage: {actor_id} appears in multiple splits")
        actor_split[actor_id] = split
        clip_actor, performed_label = _clip_parts(str(row["clip_id"]))
        if clip_actor != actor_id:
            raise ValueError(f"Actor ID does not match clip ID: {sample_id}")
        if row.get("performed_label") != performed_label:
            raise ValueError(f"Performed label does not match clip ID: {sample_id}")
        for label in ("face_vote", "voice_vote", "multimodal_vote"):
            if row[label] not in LABELS | {"unknown"}:
                raise ValueError(f"Invalid {label} for {sample_id}")
        if (row.get("visual_target_json") != {"presented_expression": row["face_vote"]}
                or row.get("audio_target") != row["voice_vote"]
                or row.get("fusion_target") != row["multimodal_vote"]):
            raise ValueError(f"Supervision targets disagree with votes for {sample_id}")
        timestamps = row["frame_timestamps_ms"]
        paths = row["frame_paths"]
        hashes = row["frame_sha256s"]
        duration = row["video_duration_ms"]
        if len(timestamps) != 3 or len(paths) != 3 or len(hashes) != 3:
            raise ValueError(f"Expected three paired frames for {sample_id}")
        quality = row.get("frame_quality")
        if quality is not None and len(quality) != 3:
            raise ValueError(f"Expected three frame quality records for {sample_id}")
        if quality is not None and row.get("visual_usable") != _visual_usable(quality):
            raise ValueError(f"Visual usability conflicts with face detections for {sample_id}")
        if all(key in row for key in ("paired_usable", "visual_usable", "audio_quality_status", "av_alignment_status")):
            expected_paired = (row["visual_usable"] and row["audio_quality_status"] == "accepted"
                               and row["av_alignment_status"] == "accepted")
            if row["paired_usable"] != expected_paired:
                raise ValueError(f"Paired usability conflicts with quality statuses for {sample_id}")
        if Path(row["audio_path"]).stem != sample_id or any(Path(path).parent.name != sample_id for path in paths):
            raise ValueError(f"Media paths do not match clip ID: {sample_id}")
        for media_path in paths + [row["audio_path"]]:
            path = Path(media_path)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe relative media path: {media_path}")
            if media_path in seen_media:
                raise ValueError(f"Media path reused across clips: {media_path}")
            seen_media.add(media_path)
        if any(len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest) for digest in hashes + [row["audio_sha256"]]):
            raise ValueError(f"Invalid media SHA256 for {sample_id}")
        for source_key in ("video_sha256", "source_audio_sha256"):
            digest = row.get(source_key)
            if digest is not None and (not isinstance(digest, str) or len(digest) != 64
                                       or any(char not in "0123456789abcdef" for char in digest)):
                raise ValueError(f"Invalid {source_key} for {sample_id}")
        if row.get("source_audio_path") is not None and Path(row["source_audio_path"]).stem != sample_id:
            raise ValueError(f"Source audio does not match clip ID: {sample_id}")
        if not (0 <= timestamps[0] < timestamps[1] < timestamps[2] < duration):
            raise ValueError(f"Frame timestamps outside video duration for {sample_id}")
        if int(row["audio_sample_rate"]) != 16000 or int(row["audio_duration_ms"]) <= 0:
            raise ValueError(f"Invalid audio properties for {sample_id}")
        if abs(int(row["audio_duration_ms"]) - int(duration)) > 1000:
            raise ValueError(f"AV duration differs by over one second for {sample_id}")
        if verify_files and base is not None:
            for relative, digest in zip(paths + [row["audio_path"]], hashes + [row["audio_sha256"]]):
                candidate = (base / relative).resolve()
                if base != candidate and base not in candidate.parents:
                    raise ValueError(f"Media path escapes output root: {relative}")
                if not candidate.is_file() or _sha256(candidate) != digest:
                    raise ValueError(f"Missing or corrupted output media: {candidate}")
        split_counts[split] += 1
    if not seen:
        raise ValueError("Manifest is empty")
    return dict(split_counts)
