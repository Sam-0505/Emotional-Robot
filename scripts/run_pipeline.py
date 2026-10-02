"""Run paired CREMA-D perception and save a guarded-harness observation.

Requires a CUDA host, prepared media, and either a joint checkpoint/calibration
pair or the trained baseline experts. Use the JSON with `scripts/run_demo.py --live` after the
simulator and speech services are available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reachy_emotions.perception.cli import add_perception_arguments, validate_perception_arguments, infer_from_arguments
from reachy_emotions.perception.manifest import read_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("sample_id", help="Exact sample_id from manifest.jsonl")
    add_perception_arguments(parser)
    parser.add_argument("--output", type=Path, help="Optional output JSON path")
    args = parser.parse_args()
    validate_perception_arguments(parser, args)

    records = [record for record in read_manifest(args.manifest)
               if record["sample_id"] == args.sample_id]
    if len(records) != 1:
        parser.error("sample_id must identify exactly one manifest record")
    observation = infer_from_arguments(args, records[0])
    serialized = json.dumps(observation, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
