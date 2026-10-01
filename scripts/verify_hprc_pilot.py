"""Check saved one-step reloads and paired inference before full HPRC training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def verify_pilot(run_root: Path, visual_revision: str, audio_revision: str,
                 stage: str = "pilot") -> None:
    if stage not in {"pilot", "full"}:
        raise ValueError(f"unsupported verification stage: {stage}")
    manifest = run_root / "prepared" / "manifest.jsonl"
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    for name, revision, subdir in (
        ("visual", visual_revision, "visual_adapter"),
        ("audio", audio_revision, "audio_head"),
    ):
        path = run_root / stage / subdir / "reload_verification.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("reload_verified") is not True:
            raise ValueError(f"{name} subprocess reload is not verified: {path}")
        if result.get("revision") != revision or result.get("manifest_sha256") != manifest_sha256:
            raise ValueError(f"{name} {stage} checkpoint or manifest differs from this run")
    if stage == "pilot":
        observation_path = run_root / "pilot" / "observation.json"
        observation = json.loads(observation_path.read_text(encoding="utf-8"))
        if observation.get("source") != "audio_visual":
            raise ValueError(f"pilot did not exercise both modalities: {observation_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--visual-revision", required=True)
    parser.add_argument("--audio-revision", required=True)
    parser.add_argument("--stage", choices=("pilot", "full"), default="pilot")
    args = parser.parse_args()
    try:
        verify_pilot(args.run_root, args.visual_revision, args.audio_revision, args.stage)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Pilot gate failed: {exc}\n")
    print(f"{args.stage.capitalize()} artifact verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
