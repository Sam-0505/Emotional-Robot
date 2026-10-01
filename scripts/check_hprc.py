"""Read-only CREMA-D and training-environment preflight for an HPRC worker."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reachy_emotions.data.cremad import (  # noqa: E402
    _lfs_pointer, _read_vote_details, read_demographics, read_summary, validate_vote_metadata,
)
from reachy_emotions.perception.revision import require_commit_sha  # noqa: E402


def media_counts(dataset_root: Path, clip_ids: list[str]) -> dict[str, dict[str, int]]:
    """Count usable media, unresolved LFS pointers, and absent files."""
    report = {}
    for folder, suffix in (("VideoFlash", ".flv"), ("AudioWAV", ".wav")):
        counts = {"ready": 0, "lfs_pointer": 0, "missing": 0}
        for clip_id in clip_ids:
            path = dataset_root / folder / f"{clip_id}{suffix}"
            if not path.is_file():
                counts["missing"] += 1
            elif _lfs_pointer(path) is not None:
                counts["lfs_pointer"] += 1
            else:
                counts["ready"] += 1
        report[folder] = counts
    return report


def preflight(dataset_root: Path, *, training: bool = False,
              visual_revision: str | None = None, audio_revision: str | None = None) -> dict:
    dataset_root = Path(dataset_root).resolve()
    issues = []
    tools = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe", "git-lfs")}
    packages = {name: importlib.util.find_spec(name) is not None
                for name in ("cv2", "torch", "transformers", "peft", "PIL")}
    if sys.version_info < (3, 10):
        issues.append("Python 3.10 or newer is required for this project")
    for name in ("ffmpeg", "ffprobe"):
        if tools[name] is None:
            issues.append(f"{name} is missing")
    if not packages["cv2"]:
        issues.append("OpenCV is missing; face crops would be unusable")
    try:
        rows = read_summary(dataset_root)
        if not (dataset_root / "LICENSE.txt").is_file():
            raise FileNotFoundError("official LICENSE.txt is missing")
        read_demographics(dataset_root, (row["actor_id"] for row in rows))
        details = _read_vote_details(dataset_root)
        validate_vote_metadata(rows, details)
        counts = media_counts(dataset_root, [row["clip_id"] for row in rows])
        for folder, values in counts.items():
            if values["missing"] or values["lfs_pointer"]:
                issues.append(f"{folder}: {values['missing']} missing, {values['lfs_pointer']} Git LFS pointers")
    except (OSError, RuntimeError, ValueError) as exc:
        rows, counts = [], {}
        issues.append(f"CREMA-D metadata/media check failed: {exc}")
    cuda_available = None
    if training:
        for name in ("torch", "transformers", "peft", "PIL"):
            if not packages[name]:
                issues.append(f"training dependency {name} is missing")
        if packages["torch"]:
            try:
                import torch
                cuda_available = torch.cuda.is_available()
            except Exception as exc:
                issues.append(f"PyTorch CUDA check failed: {exc}")
            if cuda_available is False:
                issues.append("PyTorch cannot see a CUDA GPU")
        for label, revision in (("visual", visual_revision), ("audio", audio_revision)):
            if revision is None:
                issues.append(f"{label} checkpoint commit SHA was not supplied")
            else:
                try:
                    require_commit_sha(revision)
                except ValueError as exc:
                    issues.append(f"{label} revision: {exc}")
    return {
        "ready": not issues, "stage": "training" if training else "data",
        "dataset_root": str(dataset_root), "metadata_clips": len(rows),
        "media": counts, "python": sys.version.split()[0],
        "tools": tools, "packages": packages, "cuda_available": cuda_available,
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--training", action="store_true", help="Also require CUDA and pinned model revisions")
    parser.add_argument("--visual-revision", help="Full 40-character NVIDIA checkpoint commit SHA")
    parser.add_argument("--audio-revision", help="Full 40-character WavLM checkpoint commit SHA")
    args = parser.parse_args()
    report = preflight(args.dataset_root, training=args.training,
                       visual_revision=args.visual_revision, audio_revision=args.audio_revision)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
