"""Select one deterministic prepared CREMA-D clip for an integration smoke test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reachy_emotions.perception.manifest import read_manifest  # noqa: E402


def select_clip(manifest: Path, split: str = "validation", paired_only: bool = True) -> str:
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    candidates = [str(row["sample_id"]) for row in read_manifest(manifest)
                  if row["split"] == split and (not paired_only or row.get("paired_usable") is True)]
    if not candidates:
        raise ValueError(f"no {'paired-usable ' if paired_only else ''}{split} clip in {manifest}")
    return min(candidates)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--allow-unusable", action="store_true", help="Select even if quality gates rejected the pair")
    args = parser.parse_args()
    try:
        chosen = select_clip(args.manifest, args.split, not args.allow_unusable)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Clip selection failed: {exc}\n")
    print(chosen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
