"""Run paired CREMA-D perception and save a guarded-harness observation.

Requires a CUDA host, pinned model revisions, prepared media, and a trained
audio head. Use the saved JSON with `scripts/run_demo.py --live` after the
simulator and speech services are available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reachy_emotions.perception.audio import load_audio_expert
from reachy_emotions.perception.fusion import FusionConfig
from reachy_emotions.perception.manifest import read_manifest
from reachy_emotions.perception.visual import load_visual_expert
from reachy_emotions.pipeline import infer_manifest_record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("sample_id", help="Exact sample_id from manifest.jsonl")
    parser.add_argument("--visual-revision", required=True, help="Pinned NVIDIA model commit")
    parser.add_argument("--audio-revision", required=True, help="Pinned WavLM model commit")
    parser.add_argument("--visual-adapter", type=Path, help="Saved decoder LoRA adapter")
    parser.add_argument("--audio-head", type=Path, required=True, help="Trained WavLM head directory")
    parser.add_argument("--fusion-config", type=Path, help="Validation-calibrated fusion JSON")
    parser.add_argument("--output", type=Path, help="Optional output JSON path")
    args = parser.parse_args()

    records = [record for record in read_manifest(args.manifest)
               if record["sample_id"] == args.sample_id]
    if len(records) != 1:
        parser.error("sample_id must identify exactly one manifest record")
    config = FusionConfig.load(args.fusion_config) if args.fusion_config else FusionConfig()
    visual = load_visual_expert(args.visual_revision, args.visual_adapter)
    audio = load_audio_expert(args.audio_revision, args.audio_head)
    observation = infer_manifest_record(args.manifest, records[0], visual, audio, config)
    serialized = json.dumps(observation, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
