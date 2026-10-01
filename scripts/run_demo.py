"""Run the guarded agent harness offline, or against a local Reachy simulator.

Offline: python scripts/run_demo.py
Live: see --help; requires a running Reachy MuJoCo daemon, Magpie NIM, and Nebius key.
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import sys
import threading
import time
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reachy_emotions.runtime import ExecutionGuard, ResponseCoordinator
from reachy_emotions.pipeline import with_conversation_context


class OfflinePlanner:
    def propose(self, observation, moves, styles):
        return {
            "response_intent": "acknowledge", "robot_affect": "calm",
            "spoken_text": "I'm here if you'd like to continue.", "speech_style": "Calm",
            "move": moves[0], "should_act": True,
            "decision_summary": "Offer a gentle acknowledgment to an acted expression.",
        }


class OfflineSpeech:
    def list_voices(self):
        return ("Magpie-Multilingual.EN-US.Jason.Calm",)

    def synthesize(self, text, voice, sample_rate_hz):
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate_hz)
            wav.writeframes(b"\x00\x00" * (sample_rate_hz // 4))
        return buffer.getvalue()


class OfflineRobot:
    def __init__(self):
        self.events = []

    def list_moves(self):
        return ("mock_understanding",)

    def output_sample_rate(self):
        return 22050

    def play(self, move, wav_bytes, stop_event: threading.Event):
        self.events.append({"move": move, "wav_bytes": len(wav_bytes), "stopped": stop_event.is_set()})

    def stop(self):
        self.events.append({"stop": True})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Use live Nebius, Magpie NIM, and the local Reachy simulator")
    parser.add_argument("--nebius-model", help="Exact text Nemotron model ID from the current Token Factory catalog")
    parser.add_argument("--tts-url", help="Running NVIDIA Speech NIM HTTP base URL, e.g. http://localhost:9000")
    parser.add_argument("--speaker-prefix", default="Magpie-Multilingual.EN-US.Jason")
    parser.add_argument("--allow-move", action="append", default=[], help="Reviewed emotions-library move; repeat for each allowed move")
    parser.add_argument("--observation-json", type=Path, help="Fused observation JSON for live simulation")
    parser.add_argument("--transcript", help="Optional one-line supplied utterance; sent to the Nebius reasoning model in live mode")
    parser.add_argument("--recent-context", help="Optional one-line session summary; sent to the Nebius reasoning model in live mode")
    args = parser.parse_args()
    try:
        context = with_conversation_context({}, args.transcript, args.recent_context)
    except ValueError as exc:
        parser.error(str(exc))

    if args.live:
        if not all((args.nebius_model, args.tts_url, args.allow_move, args.observation_json)):
            parser.error("--live requires --nebius-model, --tts-url, --allow-move, and --observation-json")
        if not os.environ.get("NEBIUS_API_KEY"):
            parser.error("--live requires NEBIUS_API_KEY")
        from reachy_emotions.runtime.adapters import MagpieSpeech, NebiusPlanner, ReachyRobot

        with args.observation_json.open("r", encoding="utf-8") as handle:
            observation = json.load(handle)
        planner = NebiusPlanner(args.nebius_model)
        speech = MagpieSpeech(args.tts_url)
        robot = ReachyRobot.connect(connection_mode="localhost_only")
        allowed = set(args.allow_move)
    else:
        observation = {
            "presented_expression": "SAD", "source": "audio_visual", "input_origin": "crema_d",
            "abstained": False, "quality": "accepted", "observed_at": time.time(),
            "observation_id": "offline-example-1",
        }
        planner, speech, robot = OfflinePlanner(), OfflineSpeech(), OfflineRobot()
        allowed = {"mock_understanding"}

    observation.update(context)

    try:
        guard = ExecutionGuard(allowed, args.speaker_prefix)
        result = ResponseCoordinator(planner, speech, robot, guard).respond(observation)
        print(json.dumps({
            "mode": "live_simulator" if args.live else "offline_dry_run",
            "status": result.status, "reason": result.reason, "trace": result.trace,
            "voice": result.voice, "plan": result.plan.__dict__ if result.plan else None,
            "playback": result.playback.__dict__ if result.playback else None,
            "robot_events": getattr(robot, "events", None),
        }, indent=2))
        if result.status != "completed":
            raise SystemExit(1)
    finally:
        if args.live:
            robot.close()


if __name__ == "__main__":
    main()
