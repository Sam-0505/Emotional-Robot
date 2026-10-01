"""Prepare an official local CREMA-D checkout without downloading media."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Also support running this repository script before an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reachy_emotions.data import load_manifest, prepare_dataset, validate_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="Official CREMA-D checkout containing processedResults/ and VideoFlash/")
    parser.add_argument("output_root", type=Path, help="Local processed-media directory (keep it out of Git)")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic actor split seed")
    parser.add_argument("--limit", type=int, help="Process only the first N metadata clips for a smoke test")
    parser.add_argument("--clip-id", action="append", help="Prepare only this official clip ID; repeat for a pilot")
    parser.add_argument("--video-dir", type=Path, help="Directory with verified video files for --clip-id smoke tests")
    parser.add_argument("--audio-dir", type=Path, help="Directory with verified WAV files for --clip-id smoke tests")
    parser.add_argument("--dry-run", action="store_true", help="Check metadata/media presence and show split counts without decoding or writing")
    parser.add_argument("--validate", action="store_true", help="Validate an existing output manifest and file hashes")
    args = parser.parse_args()
    try:
        if args.validate:
            path = args.output_root / "manifest.jsonl"
            result = {"manifest_path": str(path), "split_clips": validate_manifest(load_manifest(path), root=args.output_root, verify_files=True)}
        else:
            result = prepare_dataset(args.dataset_root, args.output_root, dry_run=args.dry_run,
                                     seed=args.seed, limit=args.limit, clip_ids=args.clip_id,
                                     video_dir=args.video_dir, audio_dir=args.audio_dir)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        parser.exit(2, f"CREMA-D preparation failed: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
