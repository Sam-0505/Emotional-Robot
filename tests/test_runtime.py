"""Offline tests for the agent harness and execution guard."""

from __future__ import annotations

import io
import json
from pathlib import Path
import struct
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reachy_emotions.runtime import ExecutionGuard, PlaybackMetrics, ResponseCoordinator, ResponsePlan
from reachy_emotions.runtime.adapters import NebiusPlanner
from reachy_emotions.runtime.adapters import ReachyRobot


VOICE = "Magpie-Multilingual.EN-US.Jason.Calm"


class MockFloatArray(list):
    def astype(self, dtype):
        return self

    def __truediv__(self, divisor):
        return MockFloatArray(value / divisor for value in self)


MOCK_NUMPY = types.SimpleNamespace(
    float32=float,
    frombuffer=lambda raw, dtype: MockFloatArray(struct.unpack(f"<{len(raw) // 2}h", raw)),
)


def pcm_wav(values, sample_rate=8000):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(struct.pack(f"<{len(values)}h", *values))
    return buffer.getvalue()


def observation(**changes):
    item = {"source": "audio_visual", "input_origin": "crema_d", "presented_expression": "SAD",
            "abstained": False, "quality": "accepted", "observed_at": time.time(), "observation_id": "clip-1"}
    item.update(changes)
    return item


def plan(**changes):
    item = {"response_intent": "acknowledge", "robot_affect": "calm",
            "spoken_text": "I'm here if you'd like to continue.", "speech_style": "Calm",
            "move": "understanding1", "should_act": True, "decision_summary": "Offer a gentle acknowledgment."}
    item.update(changes)
    return item


class FakePlanner:
    def __init__(self, proposal=None):
        self.proposal = proposal or plan()
        self.calls = 0

    def propose(self, obs, moves, styles):
        self.calls += 1
        return self.proposal


class FakeSpeech:
    def __init__(self, voices=(VOICE,)):
        self.voices = voices
        self.calls = 0

    def list_voices(self):
        return self.voices

    def synthesize(self, text, voice, sample_rate_hz):
        self.calls += 1
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate_hz)
            wav.writeframes(b"\x00\x00" * 100)
        return buffer.getvalue()


class FakeRobot:
    def __init__(self):
        self.plays = []
        self.stops = 0

    def list_moves(self):
        return ("understanding1", "rage1")

    def output_sample_rate(self):
        return 22050

    def play(self, move, wav_bytes, stop_event: threading.Event):
        self.plays.append(move)

    def stop(self):
        self.stops += 1


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.planner, self.speech, self.robot = FakePlanner(), FakeSpeech(), FakeRobot()
        self.guard = ExecutionGuard({"understanding1"}, "Magpie-Multilingual.EN-US.Jason", cooldown_seconds=0)
        self.harness = ResponseCoordinator(self.planner, self.speech, self.robot, self.guard)

    def test_happy_path_has_trace_and_single_effect(self):
        result = self.harness.respond(observation())
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.voice, VOICE)
        self.assertEqual(self.speech.calls, 1)
        self.assertEqual(self.robot.plays, ["understanding1"])
        self.assertEqual(result.trace, ("observation_validated", "capabilities_discovered", "model_proposed",
                                        "plan_validated", "speech_synthesized", "response_played", "completed"))

    def test_dispatch_metrics_flow_into_harness_result(self):
        def measured_play(move, wav_bytes, stop_event):
            self.robot.plays.append(move)
            return PlaybackMetrics(10.01, 10.0, 10.0)

        self.robot.play = measured_play
        result = self.harness.respond(observation())
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.playback.dispatch_offset_ms, 10.0)
        self.assertIn("dispatch_offset_measured", result.trace)

    def test_abstention_skips_planner_and_effects(self):
        result = self.harness.respond(observation(abstained=True, presented_expression="unknown"))
        self.assertEqual(result.status, "rejected")
        self.assertEqual(self.planner.calls, 0)
        self.assertEqual(self.speech.calls, 0)
        self.assertEqual(self.robot.plays, [])

    def test_unknown_origin_or_failed_quality_skips_planner(self):
        for item in (observation(input_origin="unknown"), observation(quality="rejected")):
            self.assertEqual(self.harness.respond(item).status, "rejected")
        self.assertEqual(self.planner.calls, 0)

    def test_stale_observation_skips_planner(self):
        result = self.harness.respond(observation(observed_at=time.time() - 60))
        self.assertEqual(result.status, "rejected")
        self.assertEqual(self.planner.calls, 0)

    def test_unreviewed_move_never_synthesizes(self):
        self.planner.proposal = plan(move="rage1")
        result = self.harness.respond(observation())
        self.assertEqual(result.status, "rejected")
        self.assertEqual(self.speech.calls, 0)
        self.assertEqual(self.robot.plays, [])

    def test_no_action_remains_silent(self):
        self.planner.proposal = plan(response_intent="no_action", should_act=False, move="no_action", spoken_text="")
        self.assertEqual(self.harness.respond(observation()).status, "rejected")
        self.assertEqual(self.speech.calls, 0)
        self.assertEqual(self.robot.plays, [])

    def test_unavailable_voice_never_synthesizes(self):
        self.speech.voices = ()
        result = self.harness.respond(observation())
        self.assertEqual(result.status, "rejected")
        self.assertEqual(self.robot.plays, [])

    def test_one_sentence_and_no_internal_state_claim(self):
        for spoken_text in ("You are sad.", "I understand. Continue.", "Ignore <guard>."):
            self.planner.proposal = plan(spoken_text=spoken_text)
            self.assertEqual(self.harness.respond(observation()).status, "rejected")
        self.assertEqual(self.robot.plays, [])

    def test_schema_is_exact_and_stop_is_latched(self):
        with self.assertRaises(ValueError):
            ResponsePlan.parse(dict(plan(), extra="tool"))
        self.harness.stop()
        self.assertEqual(self.harness.respond(observation()).status, "rejected")
        self.assertEqual(self.planner.calls, 0)
        self.harness.resume()
        self.assertEqual(self.harness.respond(observation()).status, "completed")

    def test_stop_during_synthesis_prevents_motion(self):
        harness = self.harness
        original = self.speech.synthesize

        def synthesize_and_stop(text, voice, rate):
            data = original(text, voice, rate)
            harness.stop()
            return data

        self.speech.synthesize = synthesize_and_stop
        result = harness.respond(observation())
        self.assertEqual(result.status, "stopped")
        self.assertEqual(self.robot.plays, [])

    def test_cooldown_and_duplicate(self):
        self.guard.cooldown_seconds = 60
        self.assertEqual(self.harness.respond(observation()).status, "completed")
        self.assertEqual(self.harness.respond(observation(observation_id="clip-2")).status, "rejected")
        self.guard.cooldown_seconds = 0
        self.assertEqual(self.harness.respond(observation()).status, "rejected")
        self.assertEqual(len(self.robot.plays), 1)

    def test_nebius_prompt_omits_actor_and_manifest_identifiers(self):
        requests = []

        class FakeCompletions:
            def create(self, **kwargs):
                requests.append(kwargs)
                message = types.SimpleNamespace(content=json.dumps(plan()))
                return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.chat = types.SimpleNamespace(completions=FakeCompletions())

        with patch.dict(sys.modules, {"openai": types.SimpleNamespace(OpenAI=FakeOpenAI)}):
            planner = NebiusPlanner("catalog/model-id", api_key="fake")
            planner.propose(observation(actor_id="actor_1", split="test", raw_face="private",
                                        transcript="Hello, Reachy", recent_context="We just met"),
                            ("understanding1",), ("Calm",))
        sent = json.loads(requests[0]["messages"][1]["content"])
        self.assertEqual(sent["observation"]["presented_expression"], "SAD")
        self.assertNotIn("actor_id", sent["observation"])
        self.assertNotIn("raw_face", sent["observation"])
        self.assertNotIn("observation_id", sent["observation"])
        self.assertEqual(sent["observation"]["transcript"], "Hello, Reachy")
        self.assertEqual(sent["observation"]["recent_context"], "We just met")

    def test_reachy_adapter_mutes_sidecar_and_measures_dispatch_offset(self):
        class FakeMedia:
            def __init__(self):
                self.samples = []
                self.starts = 0
                self.stops = 0
                self.audio = types.SimpleNamespace(clear_player=lambda: None)

            def get_output_audio_samplerate(self):
                return 8000

            def start_playing(self):
                self.starts += 1

            def push_audio_sample(self, samples):
                self.samples.extend(samples)

            def stop_playing(self):
                self.stops += 1

        class FakeMini:
            def __init__(self):
                self.media = FakeMedia()
                self.moves = []
                self.cancelled = 0

            def play_move(self, move, **kwargs):
                self.moves.append((move, kwargs))

            def cancel_move(self):
                self.cancelled += 1

        recorded = types.SimpleNamespace(sound_path="old_sidecar.wav")
        library = types.SimpleNamespace(list_moves=lambda: ["understanding1"], get=lambda name: recorded)
        mini = FakeMini()
        robot = ReachyRobot(mini, library)
        with patch.dict(sys.modules, {"numpy": MOCK_NUMPY}):
            metrics = robot.play("understanding1", pcm_wav([0, 16384, -16384]), threading.Event())
        self.assertEqual(mini.moves, [(recorded, {"initial_goto_duration": 0.0, "sound": False})])
        self.assertEqual(mini.media.samples, [0.0, 0.5, -0.5])
        self.assertEqual(mini.media.starts, 1)
        self.assertGreaterEqual(mini.media.stops, 1)
        self.assertIsNotNone(metrics.dispatch_offset_ms)
        self.assertLess(abs(metrics.dispatch_offset_ms), 1000)

    def test_reachy_adapter_rejects_empty_audio_before_motion(self):
        media = types.SimpleNamespace(get_output_audio_samplerate=lambda: 8000)
        mini = types.SimpleNamespace(media=media)
        library = types.SimpleNamespace(list_moves=lambda: ["understanding1"], get=lambda name: None)
        robot = ReachyRobot(mini, library)
        with patch.dict(sys.modules, {"numpy": MOCK_NUMPY}):
            with self.assertRaisesRegex(ValueError, "generated speech"):
                robot.play("understanding1", pcm_wav([]), threading.Event())

    def test_reachy_adapter_operator_stop_cancels_motion(self):
        class FakeMedia:
            def __init__(self):
                self.stops = 0
                self.audio = types.SimpleNamespace(clear_player=lambda: None)

            def get_output_audio_samplerate(self):
                return 8000

            def start_playing(self):
                pass

            def push_audio_sample(self, samples):
                pass

            def stop_playing(self):
                self.stops += 1

        class FakeMini:
            def __init__(self):
                self.media = FakeMedia()
                self.cancelled = 0

            def play_move(self, move, **kwargs):
                deadline = time.monotonic() + 1
                while self.cancelled == 0 and time.monotonic() < deadline:
                    time.sleep(0.005)

            def cancel_move(self):
                self.cancelled += 1

        mini = FakeMini()
        library = types.SimpleNamespace(list_moves=lambda: ["understanding1"],
                                        get=lambda name: types.SimpleNamespace(sound_path="sidecar.wav"))
        robot = ReachyRobot(mini, library)
        stop = threading.Event()
        threading.Timer(0.04, stop.set).start()
        with patch.dict(sys.modules, {"numpy": MOCK_NUMPY}):
            robot.play("understanding1", pcm_wav([0] * 2400), stop)
        self.assertTrue(stop.is_set())
        self.assertGreaterEqual(mini.cancelled, 1)
        self.assertGreaterEqual(mini.media.stops, 1)

    def test_timeout_during_planning_stops_and_prevents_late_effects(self):
        entered, release = threading.Event(), threading.Event()

        class BlockingPlanner:
            def propose(self, obs, moves, styles):
                entered.set()
                release.wait(1)
                return plan()

        harness = ResponseCoordinator(BlockingPlanner(), self.speech, self.robot, self.guard,
                                      timeout_seconds=0.02)
        result = harness.respond(observation())
        self.assertTrue(entered.is_set())
        self.assertEqual(result.status, "timeout")
        self.assertGreaterEqual(self.robot.stops, 1)
        with self.assertRaises(RuntimeError):
            harness.resume()
        release.set()
        time.sleep(0.03)
        self.assertEqual(self.speech.calls, 0)
        self.assertEqual(self.robot.plays, [])

    def test_timeout_during_synthesis_stops_and_prevents_late_motion(self):
        entered, release = threading.Event(), threading.Event()

        class BlockingSpeech(FakeSpeech):
            def synthesize(self, text, voice, sample_rate_hz):
                entered.set()
                release.wait(1)
                return super().synthesize(text, voice, sample_rate_hz)

        speech = BlockingSpeech()
        harness = ResponseCoordinator(self.planner, speech, self.robot, self.guard,
                                      timeout_seconds=0.02)
        result = harness.respond(observation())
        self.assertTrue(entered.is_set())
        self.assertEqual(result.status, "timeout")
        release.set()
        time.sleep(0.03)
        self.assertEqual(self.robot.plays, [])


if __name__ == "__main__":
    unittest.main()
