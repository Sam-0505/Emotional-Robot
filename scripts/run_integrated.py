"""Infer a prepared CREMA-D clip and immediately run the simulator harness.

This live path requires a CUDA host, a local Reachy MuJoCo daemon, a Magpie
Speech NIM endpoint, and a Nebius Token Factory key/model. No motion or speech
occurs unless --execute is passed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from reachy_emotions.perception.audio import load_audio_expert
from reachy_emotions.perception.fusion import FusionConfig
from reachy_emotions.perception.manifest import read_manifest
from reachy_emotions.perception.visual import load_visual_expert
from reachy_emotions.pipeline import infer_manifest_record, with_conversation_context


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("sample_id")
    parser.add_argument("--visual-revision", required=True)
    parser.add_argument("--visual-adapter", type=Path, required=True)
    parser.add_argument("--audio-revision", required=True)
    parser.add_argument("--audio-head", type=Path, required=True)
    parser.add_argument("--fusion-config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Connect to live services and allow a guarded simulator response")
    parser.add_argument("--nebius-model", help="Available Token Factory text Nemotron model ID")
    parser.add_argument("--tts-url", help="Running Magpie Speech NIM HTTP base URL")
    parser.add_argument("--speaker-prefix", default="Magpie-Multilingual.EN-US.Jason")
    parser.add_argument("--timeout-seconds", type=float, default=90.0,
                        help="Maximum response cycle time before stop is requested")
    parser.add_argument("--allow-move", action="append", default=[], help="Reviewed Reachy move; repeat per move")
    parser.add_argument("--transcript", help="Optional one-line supplied utterance; sent to the Nebius reasoning model")
    parser.add_argument("--recent-context", help="Optional one-line session summary; sent to the Nebius reasoning model")
    args = parser.parse_args()
    try:
        context = with_conversation_context({}, args.transcript, args.recent_context)
    except ValueError as exc:
        parser.error(str(exc))

    records = [row for row in read_manifest(args.manifest) if row["sample_id"] == args.sample_id]
    if len(records) != 1:
        parser.error("sample_id must identify exactly one prepared clip")
    if args.execute and not all((args.nebius_model, args.tts_url, args.allow_move, os.environ.get("NEBIUS_API_KEY"))):
        parser.error("--execute requires --nebius-model, --tts-url, --allow-move, and NEBIUS_API_KEY")

    from reachy_emotions.runtime import ExecutionGuard, ResponseCoordinator
    from reachy_emotions.runtime.adapters import MagpieSpeech, NebiusPlanner, ReachyRobot

    robot = None
    try:
        # Establish the slow external connections before recording observation
        # freshness; the guard requires inference to be recent at response time.
        if args.execute:
            planner = NebiusPlanner(args.nebius_model)
            speech = MagpieSpeech(args.tts_url)
            robot = ReachyRobot.connect(connection_mode="localhost_only")
        visual = load_visual_expert(args.visual_revision, args.visual_adapter)
        audio = load_audio_expert(args.audio_revision, args.audio_head)
        config = FusionConfig.load(args.fusion_config)
        observation = infer_manifest_record(args.manifest, records[0], visual, audio, config)
        observation.update(context)
        if not args.execute:
            print(json.dumps({"mode": "perception_only", "observation": observation}, indent=2))
            return 0
        guard = ExecutionGuard(set(args.allow_move), args.speaker_prefix)
        result = ResponseCoordinator(planner, speech, robot, guard,
                                     timeout_seconds=args.timeout_seconds).respond(observation)
        print(json.dumps({
            "mode": "live_simulator", "observation": observation,
            "status": result.status, "reason": result.reason,
            "trace": result.trace, "voice": result.voice,
            "plan": result.plan.__dict__ if result.plan else None,
            "elapsed_seconds": result.elapsed_seconds,
            "playback": result.playback.__dict__ if result.playback else None,
        }, indent=2))
        return 0 if result.status == "completed" else 1
    finally:
        if robot is not None:
            robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
