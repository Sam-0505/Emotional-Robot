"""Fast bridge tests requiring neither media nor a GPU."""

from __future__ import annotations

from pathlib import Path
import hashlib
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reachy_emotions.pipeline import infer_manifest_record, make_observation, with_conversation_context


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.record = {"sample_id": "1001_IEO_HAP_HI", "source_dataset": "CREMA-D",
                       "actor_id": "1001", "split": "test",
                       "av_alignment_status": "accepted", "audio_quality_status": "accepted",
                       "paired_usable": True,
                       "audio_quality": {"active_frame_fraction": 0.5}}
        self.happy = {"ANG": 0.01, "DIS": 0.01, "FEA": 0.01,
                      "HAP": 0.95, "NEU": 0.01, "SAD": 0.01}

    def test_paired_observation_is_action_eligible(self):
        result = make_observation(
            self.record,
            {"scores": self.happy, "visual_quality": 1.0},
            {"scores": self.happy, "audio_quality": 1.0},
            observed_at=123.0,
        )
        self.assertEqual(result["source"], "audio_visual")
        self.assertEqual(result["quality"], "accepted")
        self.assertEqual(result["presented_expression"], "HAP")
        self.assertEqual(result["observation_id"], self.record["sample_id"])

    def test_missing_modality_cannot_trigger_robot(self):
        result = make_observation(
            self.record,
            {"scores": self.happy, "visual_quality": 1.0},
            {"scores": None, "audio_quality": 0.0},
        )
        self.assertEqual(result["source"], "visual")
        self.assertEqual(result["quality"], "rejected")

    def test_both_missing_abstains(self):
        result = make_observation(
            self.record,
            {"scores": None, "visual_quality": 0.0},
            {"scores": None, "audio_quality": 0.0},
        )
        self.assertTrue(result["abstained"])
        self.assertEqual(result["presented_expression"], "unknown")

    def test_alignment_or_speech_activity_failure_cannot_trigger_robot(self):
        for change in ({"av_alignment_status": "review"},
                       {"audio_quality_status": "review"},
                       {"audio_quality": {"active_frame_fraction": 0.0}}):
            record = dict(self.record, **change)
            result = make_observation(record,
                                      {"scores": self.happy, "visual_quality": 1.0},
                                      {"scores": self.happy, "audio_quality": 1.0})
            self.assertEqual(result["quality"], "rejected")

    def test_only_one_usable_face_frame_cannot_trigger_robot(self):
        result = make_observation(self.record,
                                  {"scores": self.happy, "visual_quality": 1 / 3},
                                  {"scores": self.happy, "audio_quality": 1.0})
        self.assertEqual(result["quality"], "rejected")

    def test_non_cremad_record_is_out_of_scope(self):
        with self.assertRaisesRegex(ValueError, "CREMA-D"):
            make_observation(dict(self.record, source_dataset="other"),
                             {"scores": self.happy, "visual_quality": 1.0},
                             {"scores": self.happy, "audio_quality": 1.0})

    def test_typed_conversation_context_is_bounded_and_nonmutating(self):
        original = {"presented_expression": "HAP"}
        result = with_conversation_context(original, "Hello there", "We just met")
        self.assertEqual(result["transcript"], "Hello there")
        self.assertEqual(result["recent_context"], "We just met")
        self.assertNotIn("transcript", original)
        for text in ("", "bad\ncommand", "x" * 501):
            with self.assertRaisesRegex(ValueError, "transcript"):
                with_conversation_context(original, text)

    def test_inference_bridge_passes_face_quality_and_keeps_pairing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / f"frame_{index}.png" for index in range(3)]
            audio_path = root / "audio.wav"
            for path in paths + [audio_path]:
                path.write_bytes(b"fixture")
            record = dict(self.record,
                          frame_paths=[path.name for path in paths],
                          audio_path=audio_path.name,
                          frame_sha256s=[hashlib.sha256(path.read_bytes()).hexdigest() for path in paths],
                          audio_sha256=hashlib.sha256(audio_path.read_bytes()).hexdigest(),
                          frame_quality=[{"face_status": "detected"},
                                         {"face_status": "detected"},
                                         {"face_status": "not_detected"}])
            visual_result = {"scores": self.happy, "visual_quality": 2 / 3}
            audio_result = {"scores": self.happy, "audio_quality": 1.0}
            with patch("reachy_emotions.perception.visual.predict_visual_clip", return_value=visual_result) as visual, \
                 patch("reachy_emotions.perception.audio.predict_audio_clip", return_value=audio_result) as audio:
                result = infer_manifest_record(root / "manifest.jsonl", record,
                                               ("model", "tokenizer", "processor"),
                                               ("encoder", "head", "extractor", "cpu"))
            self.assertEqual(visual.call_args.args[-1], [1.0, 1.0, 0.0])
            self.assertEqual(audio.call_args.args[-1], audio_path.resolve())
            self.assertEqual(result["quality"], "accepted")

    def test_inference_bridge_rejects_tampered_media(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = [root / f"frame_{index}.png" for index in range(3)]
            audio_path = root / "audio.wav"
            for path in frames + [audio_path]:
                path.write_bytes(b"fixture")
            record = dict(self.record,
                          frame_paths=[path.name for path in frames],
                          audio_path=audio_path.name,
                          frame_sha256s=["0" * 64] * 3,
                          audio_sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "hash does not match"):
                infer_manifest_record(root / "manifest.jsonl", record,
                                      ("model", "tokenizer", "processor"),
                                      ("encoder", "head", "extractor", "cpu"))


if __name__ == "__main__":
    unittest.main()
