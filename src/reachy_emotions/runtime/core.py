"""Small, testable boundary between perception, a reasoning model, and robot effects."""

from __future__ import annotations

import re
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol


LABELS = frozenset({"ANG", "DIS", "FEA", "HAP", "NEU", "SAD", "unknown"})
STYLES = frozenset({"Neutral", "Calm", "Happy"})
INTENTS = frozenset({"acknowledge", "encourage", "greet", "celebrate", "no_action"})
_CLAIM = re.compile(
    r"\b(?:you are|you're|you seem|you look|i know you (?:are|feel)|"
    r"you have|you suffer from)\b[^.!?]{0,50}\b"
    r"(?:angry|sad|afraid|depressed|anxious|autistic|sick|ill|diagnos\w*)\b",
    re.IGNORECASE,
)
_CONTROL = re.compile(r"[\x00-\x1f\x7f]|[<>`{}]|https?://", re.IGNORECASE)


@dataclass(frozen=True)
class ResponsePlan:
    response_intent: str
    robot_affect: str
    spoken_text: str
    speech_style: str
    move: str
    should_act: bool
    decision_summary: str

    @classmethod
    def parse(cls, value: Mapping[str, Any]) -> "ResponsePlan":
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("response plan must contain exactly the required fields")
        for key in set(value) - {"should_act"}:
            if type(value[key]) is not str:
                raise ValueError(f"{key} must be a string")
        if type(value["should_act"]) is not bool:
            raise ValueError("should_act must be a boolean")
        return cls(**value)


@dataclass(frozen=True)
class PlaybackMetrics:
    """Client dispatch timestamps, not audible onset or physical motion onset."""

    audio_dispatch_monotonic: float | None
    motion_dispatch_monotonic: float | None
    dispatch_offset_ms: float | None


@dataclass(frozen=True)
class RunResult:
    status: str
    reason: str
    plan: ResponsePlan | None = None
    voice: str | None = None
    elapsed_seconds: float = 0.0
    trace: tuple[str, ...] = ()
    playback: PlaybackMetrics | None = None


class Planner(Protocol):
    def propose(self, observation: Mapping[str, Any], moves: tuple[str, ...], styles: tuple[str, ...]) -> Mapping[str, Any]: ...


class Speech(Protocol):
    def list_voices(self) -> tuple[str, ...]: ...
    def synthesize(self, text: str, voice: str, sample_rate_hz: int) -> bytes: ...


class Robot(Protocol):
    def list_moves(self) -> tuple[str, ...]: ...
    def output_sample_rate(self) -> int: ...
    def play(self, move: str, wav_bytes: bytes, stop_event: threading.Event) -> PlaybackMetrics | None: ...
    def stop(self) -> None: ...


class ExecutionGuard:
    """Validate model output and local state before speech or motion begins.

    Pattern checks are a narrow prototype constraint, not a general content
    safety classifier. Rejected plans remain silent.
    """

    def __init__(self, allowed_moves: set[str], speaker_prefix: str, max_age_seconds: float = 10.0,
                 cooldown_seconds: float = 3.0, max_words: int = 24, max_chars: int = 160):
        if not allowed_moves or not speaker_prefix.startswith("Magpie-Multilingual.EN-US."):
            raise ValueError("configure a reviewed move allowlist and an English Magpie speaker prefix")
        self.allowed_moves = frozenset(allowed_moves)
        self.speaker_prefix = speaker_prefix
        self.max_age_seconds = max_age_seconds
        self.cooldown_seconds = cooldown_seconds
        self.max_words = max_words
        self.max_chars = max_chars
        self._last_completed = float("-inf")
        self._last_key: tuple[Any, ...] | None = None
        self._stopped = False
        self._active = False
        self._lock = threading.RLock()

    def stop(self) -> None:
        with self._lock:
            self._stopped = True

    def resume(self) -> None:
        with self._lock:
            if self._active:
                raise RuntimeError("cannot resume while a response is active")
            self._stopped = False

    def check_observation(self, observation: Mapping[str, Any], now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._lock:
            if self._stopped:
                raise ValueError("operator stop is active")
            if self._active:
                raise ValueError("another response is active")
            if not isinstance(observation, Mapping):
                raise ValueError("observation must be a mapping")
            if observation.get("abstained") is not False or observation.get("presented_expression") not in LABELS - {"unknown"}:
                raise ValueError("perception abstained or has no valid expression")
            if observation.get("quality", "accepted") != "accepted":
                raise ValueError("input quality gate failed")
            if observation.get("source") != "audio_visual" or observation.get("input_origin") not in {"crema_d", "live"}:
                raise ValueError("unrecognized observation modality or input origin")
            timestamp = observation.get("observed_at")
            if type(timestamp) not in (int, float) or not 0 <= now - timestamp <= self.max_age_seconds:
                raise ValueError("observation is missing, stale, or future dated")

    def check(self, observation: Mapping[str, Any], plan: ResponsePlan, available_moves: set[str],
              available_voices: set[str], now: float | None = None) -> str:
        now = time.time() if now is None else now
        self.check_observation(observation, now)
        with self._lock:
            if not plan.should_act or plan.response_intent == "no_action":
                raise ValueError("plan requested no action")
            if plan.response_intent not in INTENTS or plan.robot_affect not in {"calm", "neutral", "happy"}:
                raise ValueError("unsupported intent or robot affect")
            if plan.speech_style not in STYLES:
                raise ValueError("unsupported speech style")
            if plan.move not in self.allowed_moves or plan.move not in available_moves:
                raise ValueError("move is not reviewed and available")
            voice = f"{self.speaker_prefix}.{plan.speech_style}"
            if voice not in available_voices:
                raise ValueError("selected Magpie voice is unavailable")
            words = plan.spoken_text.split()
            if (not words or len(words) > self.max_words or len(plan.spoken_text) > self.max_chars
                    or len(plan.decision_summary) > 160):
                raise ValueError("response text is empty or too long")
            if _CONTROL.search(plan.spoken_text) or _CLAIM.search(plan.spoken_text):
                raise ValueError("response text contains a disallowed claim or control character")
            if len(re.findall(r"[.!?]", plan.spoken_text)) != 1 or not re.search(r"[.!?]$", plan.spoken_text):
                raise ValueError("response must be one complete sentence")
            if now - self._last_completed < self.cooldown_seconds:
                raise ValueError("response cooldown is active")
            key = (observation.get("input_origin"), observation.get("observation_id"))
            if observation.get("observation_id") is not None and key == self._last_key:
                raise ValueError("duplicate observation response")
            return voice

    def begin(self) -> None:
        with self._lock:
            if self._stopped or self._active:
                raise ValueError("response is stopped or already active")
            self._active = True

    def finish(self, observation: Mapping[str, Any], plan: ResponsePlan, completed: bool) -> None:
        with self._lock:
            self._active = False
            if completed:
                self._last_completed = time.time()
                self._last_key = (observation.get("input_origin"), observation.get("observation_id"))


class ResponseCoordinator:
    def __init__(self, planner: Planner, speech: Speech, robot: Robot, guard: ExecutionGuard,
                 timeout_seconds: float = 90.0):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        self.planner, self.speech, self.robot, self.guard = planner, speech, robot, guard
        self.timeout_seconds = timeout_seconds
        self._stop_event = threading.Event()
        self._inflight = False
        self._state_lock = threading.Lock()

    def stop(self) -> None:
        self.guard.stop()
        self._stop_event.set()
        self.robot.stop()

    def resume(self) -> None:
        with self._state_lock:
            if self._inflight:
                raise RuntimeError("cannot resume while a response is still running")
        self.guard.resume()
        self._stop_event.clear()

    def respond(self, observation: Mapping[str, Any]) -> RunResult:
        with self._state_lock:
            if self._inflight:
                return RunResult("rejected", "another response is running", trace=("rejected",))
            self._inflight = True
        started = time.monotonic()
        results: list[RunResult] = []

        def run() -> None:
            try:
                results.append(self._respond_once(observation))
            except Exception as exc:
                results.append(RunResult("error", f"response worker failed: {exc}",
                                         elapsed_seconds=time.monotonic() - started, trace=("worker_error",)))
            finally:
                with self._state_lock:
                    self._inflight = False

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(self.timeout_seconds)
        if worker.is_alive():
            stop_error = None
            try:
                self.stop()
            except Exception as exc:
                stop_error = str(exc)
            reason = "response deadline exceeded; stop requested"
            if stop_error:
                reason += f"; robot stop failed: {stop_error}"
            return RunResult("timeout", reason,
                             elapsed_seconds=time.monotonic() - started, trace=("timeout", "stop_requested"))
        return results[0]

    def _respond_once(self, observation: Mapping[str, Any]) -> RunResult:
        start = time.monotonic()
        trace: list[str] = []
        try:
            self.guard.check_observation(observation)
            trace.append("observation_validated")
            moves = tuple(sorted(set(self.robot.list_moves()) & self.guard.allowed_moves))
            voices = set(self.speech.list_voices())
            styles = tuple(sorted(style for style in STYLES if f"{self.guard.speaker_prefix}.{style}" in voices))
            trace.append("capabilities_discovered")
            if not moves or not styles:
                raise ValueError("no reviewed moves or supported Magpie styles are available")
            raw = self.planner.propose(observation, moves, styles)
            trace.append("model_proposed")
            plan = ResponsePlan.parse(raw)
            voice = self.guard.check(observation, plan, set(moves), voices)
            trace.append("plan_validated")
            self.guard.begin()
        except (ValueError, TypeError, KeyError) as exc:
            trace.append("rejected")
            return RunResult("rejected", str(exc), elapsed_seconds=time.monotonic() - start, trace=tuple(trace))
        except Exception as exc:
            trace.append("planning_error")
            return RunResult("error", f"planning or capability discovery failed: {exc}", elapsed_seconds=time.monotonic() - start, trace=tuple(trace))
        completed = False
        try:
            wav_bytes = self.speech.synthesize(plan.spoken_text, voice, self.robot.output_sample_rate())
            trace.append("speech_synthesized")
            if self._stop_event.is_set():
                trace.append("stopped")
                return RunResult("stopped", "operator stop during synthesis", plan, voice, time.monotonic() - start, tuple(trace))
            playback = self.robot.play(plan.move, wav_bytes, self._stop_event)
            trace.append("response_played")
            if playback is not None:
                trace.append("dispatch_offset_measured")
            if self._stop_event.is_set():
                trace.append("stopped")
                return RunResult("stopped", "operator stop during playback", plan, voice, time.monotonic() - start, tuple(trace), playback)
            completed = True
            trace.append("completed")
            return RunResult("completed", "speech and motion completed", plan, voice, time.monotonic() - start, tuple(trace), playback)
        except Exception as exc:
            stop_error = None
            try:
                self.robot.stop()
            except Exception as stop_exc:
                stop_error = str(stop_exc)
            trace.append("execution_error")
            reason = f"synthesis or playback failed: {exc}"
            if stop_error:
                reason += f"; robot stop failed: {stop_error}"
            return RunResult("error", reason, plan, voice, time.monotonic() - start, tuple(trace))
        finally:
            self.guard.finish(observation, plan, completed)
