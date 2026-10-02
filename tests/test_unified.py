"""Joint input, real CPU autograd, provenance, and held-out calibration tests."""

import importlib.util
import json
import tempfile
import unittest
import subprocess
import sys
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from reachy_emotions.perception.labels import EXPRESSION_LABELS, one_hot
from reachy_emotions.perception.unified import (joint_training_records, checkpoint_metadata,
                                               checkpoint_identity, file_hash, require_unified_pilot)
from reachy_emotions.perception.unified_config import UnifiedConfig, clip_quality
from reachy_emotions.perception.unified_calibration import (UnifiedCalibration, fit_unified_calibration,
                                                           calibrated_prediction)
from reachy_emotions.pipeline import make_unified_observation, infer_unified_manifest_record


def record(sample_id="one", split="train", target="HAP"):
    return {"sample_id": sample_id, "actor_id": split + "-actor", "split": split,
            "source_dataset": "CREMA-D", "paired_usable": True,
            "av_alignment_status": "accepted", "audio_quality_status": "accepted",
            "audio_quality": {"active_frame_fraction": .8},
            "frame_quality": [{"face_status": "detected"}] * 3,
            "frame_paths": ["1.png", "2.png", "3.png"], "audio_path": "one.wav",
            "face_vote": "SAD", "voice_vote": "ANG", "multimodal_vote": target}


class UnifiedSelectionTests(unittest.TestCase):
    def test_joint_targets_train_only_and_measured_pair_quality(self):
        eligible = record()
        ambiguous = record("ambiguous", target="unknown")
        misaligned = {**record("misaligned"), "av_alignment_status": "review"}
        rows = [eligible, record("val", "validation"), record("test", "test"), ambiguous, misaligned]
        self.assertEqual(joint_training_records(rows), [eligible])
        eligible["face_vote"], eligible["voice_vote"] = "unknown", "unknown"
        self.assertEqual(joint_training_records(rows), [eligible])

    def test_invalid_quality_and_token_budgets(self):
        for value in (None, float("nan"), False, .001):
            row = record()
            row["audio_quality"] = {"active_frame_fraction": value}
            self.assertFalse(clip_quality(row)["accepted"])
        for count in (0, 12, 1024, True):
            with self.assertRaises(ValueError):
                UnifiedConfig(visual_tokens_per_frame=count).validate()

    def test_calibration_is_validation_only_and_bound_to_artifacts(self):
        rows = [{**record(str(i), "validation", label), "scores": one_hot(label, .01)}
                for i, label in enumerate(EXPRESSION_LABELS)]
        config = fit_unified_calibration(rows, "a" * 64, "b" * 64)
        config.check("a" * 64, "b" * 64)
        with self.assertRaises(ValueError):
            config.check("c" * 64, "b" * 64)
        with self.assertRaisesRegex(ValueError, "validation"):
            fit_unified_calibration([{**rows[0], "split": "test"}], "a" * 64, "b" * 64)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            fit_unified_calibration(rows + rows[:1], "a" * 64, "b" * 64)

    def test_uncalibrated_or_missing_modality_cannot_authorize_response(self):
        row = record()
        prediction = {"scores": one_hot("HAP", .01)}
        self.assertEqual(make_unified_observation(row, prediction)["quality"], "rejected")
        config = UnifiedCalibration(1, .4, .1, "a" * 64, "b" * 64)
        observation = make_unified_observation(row, prediction, config, observed_at=123)
        self.assertEqual(observation["quality"], "accepted")
        self.assertEqual(observation["observed_at"], 123)
        row["paired_usable"] = False
        observation = make_unified_observation(row, prediction, config)
        self.assertTrue(observation["abstained"])
        self.assertEqual(observation["presented_expression"], "unknown")
        self.assertEqual(calibrated_prediction(None, config)["presented_expression"], "unknown")

    def test_checkpoint_integrity_and_pilot_gate(self):
        from reachy_emotions.perception.unified import VISUAL_ID, AUDIO_ID
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "decoder_lora").mkdir()
            names = ("decoder_lora/adapter_config.json", "decoder_lora/adapter_model.safetensors", "audio_projector.pt")
            for name in names:
                (root / name).write_bytes(b"test artifact")
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(record()) + "\n")
            metadata = {"architecture": "nemotron_wavlm_joint_v1", "target": "MultiModalVote",
                        "visual_model_id": VISUAL_ID, "audio_model_id": AUDIO_ID,
                        "visual_revision": "a" * 40, "audio_revision": "b" * 40,
                        "config": UnifiedConfig().to_dict(), "manifest_sha256": file_hash(manifest),
                        "artifact_sha256s": {name: file_hash(root / name) for name in names},
                        "reload_verified": True, "modality_wiring_passed": True, "gradient_check_passed": True}
            (root / "unified_metadata.json").write_text(json.dumps(metadata))
            actual = require_unified_pilot(root, manifest, "a" * 40, "b" * 40, UnifiedConfig())
            self.assertEqual(checkpoint_identity(actual), checkpoint_identity(metadata))
            with self.assertRaisesRegex(ValueError, "revision"):
                checkpoint_metadata(root, "c" * 40)
            with self.assertRaisesRegex(ValueError, "matching verified"):
                require_unified_pilot(root, manifest, "a" * 40, "b" * 40, UnifiedConfig(audio_tokens=8))
            (root / "audio_projector.pt").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                checkpoint_metadata(root)

    def test_runtime_checks_reload_manifest_and_calibration_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.jsonl"
            manifest.write_text(json.dumps(record()) + "\n")
            digest = file_hash(manifest)
            model = SimpleNamespace(reload_verified=False, manifest_sha256=digest, checkpoint_id="a" * 64)
            config = UnifiedCalibration(1, .4, .1, "a" * 64, digest)
            with patch("reachy_emotions.perception.unified.predict_unified_record") as predict:
                with self.assertRaisesRegex(ValueError, "reload verification"):
                    infer_unified_manifest_record(manifest, record(), model, config)
                model.reload_verified = True
                model.manifest_sha256 = "b" * 64
                with self.assertRaisesRegex(ValueError, "manifest differs"):
                    infer_unified_manifest_record(manifest, record(), model, config)
                model.manifest_sha256 = digest
                model.checkpoint_id = "c" * 64
                with self.assertRaisesRegex(ValueError, "calibration does not match"):
                    infer_unified_manifest_record(manifest, record(), model, config)
                predict.assert_not_called()
                model.checkpoint_id = "a" * 64
                predict.return_value = {"scores": one_hot("HAP", .01)}
                observation = infer_unified_manifest_record(manifest, record(), model, config)
                self.assertEqual(observation["presented_expression"], "HAP")
                predict.assert_called_once()

    def test_evaluation_selects_on_validation_and_requires_saved_config_for_test(self):
        from scripts.evaluate_unified import evaluate
        from reachy_emotions.perception.unified import VISUAL_ID, AUDIO_ID
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [record(split + str(i), split, label) for split in ("validation", "test")
                    for i, label in enumerate(EXPRESSION_LABELS)]
            manifest = root / "manifest.jsonl"
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
            metadata = {"architecture": "nemotron_wavlm_joint_v1", "config": UnifiedConfig().to_dict(),
                        "visual_revision": "a" * 40, "audio_revision": "b" * 40,
                        "artifact_sha256s": {}, "manifest_sha256": file_hash(manifest), "reload_verified": True}
            identity = checkpoint_identity(metadata)
            for split in ("validation", "test"):
                path = root / (split + ".jsonl")
                path.write_text("".join(json.dumps({"sample_id": row["sample_id"], "checkpoint_id": identity,
                                                    "manifest_sha256": file_hash(manifest),
                                                    "scores": one_hot(row["multimodal_vote"], .01)}) + "\n"
                                        for row in rows if row["split"] == split))
            with patch("scripts.evaluate_unified.checkpoint_metadata", return_value=metadata):
                metrics = evaluate(manifest, root / "validation.jsonl", root / "val", root)
                self.assertEqual(metrics["unified_multimodal_vote"]["accuracy"], 1)
                with self.assertRaisesRegex(ValueError, "saved validation"):
                    evaluate(manifest, root / "test.jsonl", root / "test-out", root, "test")
                with patch("scripts.evaluate_unified.fit_unified_calibration", side_effect=AssertionError("test refit")):
                    metrics = evaluate(manifest, root / "test.jsonl", root / "test-out", root, "test",
                                       root / "val/unified_calibration.json")
                self.assertEqual(metrics["unified_multimodal_vote"]["accuracy"], 1)
                with self.assertRaisesRegex(ValueError, "exactly match"):
                    evaluate(manifest, root / "validation.jsonl", root / "wrong", root, "test")
                metadata["reload_verified"] = False
                with self.assertRaisesRegex(ValueError, "reload verification"):
                    evaluate(manifest, root / "validation.jsonl", root / "unverified", root)

    def test_colab_launcher_uses_current_interpreter_and_separate_artifacts(self):
        from scripts.run_unified_colab import build_command
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nemotron-revision.txt").write_text("a" * 40)
            (root / "wavlm-revision.txt").write_text("b" * 40)
            manifest = root / "manifest.jsonl"
            manifest.touch()
            args = SimpleNamespace(base=root, manifest=manifest, run_root=None, config=None, stage="pilot")
            command, run = build_command(args, Path(__file__).resolve().parents[1])
            self.assertEqual(command[0], sys.executable)
            self.assertIn(str(root.resolve() / "unified-001/pilot"), command)
            self.assertFalse(run.exists())
            args.stage = "full"
            args.baseline_evaluation = None
            with self.assertRaisesRegex(ValueError, "baseline-evaluation"):
                build_command(args, root)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def tiny_model():
    import torch
    from torch import nn
    from reachy_emotions.perception.unified_model import AudioVisualModel

    class Tokenizer:
        eos_token = "~"

        def __call__(self, text, **kwargs):
            return {"input_ids": [ord(char) for char in text]}

        def apply_chat_template(self, messages, **kwargs):
            return "User: " + messages[0]["content"] + "Assistant: "

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.Embedding(128, 16)
            self.output = nn.Linear(16, 128)
            self.lora_A = nn.Linear(16, 4, bias=False)
            self.lora_B = nn.Linear(4, 128, bias=False)
            nn.init.zeros_(self.lora_B.weight)
            self.requires_grad_(False)
            self.lora_A.requires_grad_(True)
            self.lora_B.requires_grad_(True)

        def get_input_embeddings(self):
            return self.embeddings

        def forward(self, inputs_embeds, attention_mask, labels, **kwargs):
            # A tiny causal decoder that mixes all unmasked preceding tokens.
            weights = attention_mask.unsqueeze(-1)
            hidden = (inputs_embeds * weights).cumsum(1) / weights.cumsum(1).clamp_min(1)
            logits = self.output(hidden) + self.lora_B(self.lora_A(hidden))
            loss = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 128), labels[:, 1:].reshape(-1))
            return SimpleNamespace(loss=loss)

    class Visual(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = Decoder()
            self.vision_model = nn.Linear(1, 16)
            self.mlp1 = nn.Linear(16, 16)

        def extract_feature(self, pixels):
            return self.mlp1(self.vision_model(pixels.float()))

    class Audio(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=8)
            self.encoder = nn.Linear(1, 8)

        def forward(self, input_values, attention_mask=None):
            return SimpleNamespace(last_hidden_state=self.encoder(input_values.unsqueeze(-1)))

        def _get_feature_vector_attention_mask(self, length, mask):
            return mask

    def processor(images, **kwargs):
        return {"pixel_values": images[0]}

    def extractor(samples, **kwargs):
        return {"input_values": torch.tensor([samples]), "attention_mask": torch.ones(1, len(samples), dtype=torch.long)}

    torch.manual_seed(41)
    return AudioVisualModel(Visual(), Audio(), Tokenizer(), processor, extractor,
                            UnifiedConfig(visual_tokens_per_frame=4, audio_tokens=2, projector_hidden_size=8))


@unittest.skipUnless(TORCH_AVAILABLE, "install torch to run joint CPU tensor/autograd tests")
class UnifiedTensorTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        torch.set_num_threads(1)
        self.model = tiny_model()
        images = [torch.full((1, 4, 1), float(i + 1)) for i in range(3)]
        self.features = self.model.encode(images, [.1, .2, .3, .4])

    def test_shapes_loss_mask_and_modality_mask(self):
        visual, audio = self.features
        self.assertEqual(tuple(visual.shape), (1, 12, 16))
        self.assertEqual(tuple(audio.shape), (1, 2, 8))
        self.assertFalse(visual.requires_grad)
        self.assertFalse(audio.requires_grad)
        inputs = self.model.decoder_inputs(*self.features, "HAP", face_mask=(True, False, True))
        labels = inputs["labels"][0]
        first_answer = int((labels != -100).nonzero()[0])
        self.assertTrue((labels[:first_answer] == -100).all())
        self.assertEqual(int((inputs["attention_mask"] == 0).sum()), 4)
        masked = self.model.decoder_inputs(*self.features, "HAP", ablation="audio")
        self.assertEqual(int((masked["attention_mask"] == 0).sum()), 2)
        self.assertEqual(inputs["inputs_embeds"].shape[1], labels.numel())
        self.assertEqual(inputs["position_ids"].shape, inputs["attention_mask"].shape)

    def test_joint_backward_updates_projector_and_lora_only(self):
        torch = self.torch
        self.model.train()
        self.assertFalse(self.model.audio.training)
        self.assertFalse(self.model.visual.vision_model.training)
        before = {name: p.detach().clone() for name, p in self.model.named_parameters()}
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=.01)
        self.model(*self.features, "HAP").loss.backward()
        report = self.model.gradient_report()
        self.assertTrue(report["frozen_encoders_have_no_gradients"])
        self.assertTrue(report["audio_projector"]["nonzero"])
        self.assertTrue(report["decoder_lora"]["nonzero"])
        optimizer.step()
        changed = [name for name, p in self.model.named_parameters() if not torch.equal(before[name], p)]
        self.assertTrue(any(name.startswith("audio_projector") for name in changed))
        self.assertTrue(any("lora_" in name for name in changed))
        self.assertTrue(all("lora_" in name or name.startswith("audio_projector") for name in changed))

    def test_both_modalities_change_scores_and_trainables_reload(self):
        torch = self.torch
        scores = self.model.scores(*self.features)
        self.assertAlmostEqual(sum(scores.values()), 1)
        for modality in ("audio", "video"):
            changed = self.model.scores(*self.features, ablation=modality)
            self.assertGreater(max(abs(scores[key] - changed[key]) for key in scores), 1e-7)
        optimizer = torch.optim.SGD([p for p in self.model.parameters() if p.requires_grad], lr=.1)
        self.model.train()
        self.model(*self.features, "SAD").loss.backward()
        optimizer.step()
        expected = self.model.scores(*self.features)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trainables.pt"
            torch.save({name: p.detach() for name, p in self.model.named_parameters() if p.requires_grad}, path)
            reloaded = tiny_model()
            state = reloaded.state_dict()
            state.update(torch.load(path, weights_only=True))
            reloaded.load_state_dict(state)
            self.assertEqual(expected, reloaded.scores(*self.features))
            code = """
import json, sys, torch
from test_unified import tiny_model
torch.set_num_threads(1)
model = tiny_model()
state = model.state_dict()
state.update(torch.load(sys.argv[1], weights_only=True))
model.load_state_dict(state)
features = model.encode([torch.full((1, 4, 1), float(i + 1)) for i in range(3)], [.1, .2, .3, .4])
print(json.dumps(model.scores(*features)))
"""
            env = os.environ.copy()
            root = Path(__file__).resolve().parents[1]
            env["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root / "tests")))
            process = subprocess.run([sys.executable, "-c", code, str(path)], env=env,
                                     capture_output=True, text=True, check=True)
            self.assertEqual(expected, json.loads(process.stdout))

    def test_prediction_does_not_use_any_crowd_vote(self):
        from reachy_emotions.perception.unified import predict_unified_record
        row = record()
        with patch("reachy_emotions.perception.unified.encode_record", return_value=self.features):
            first = predict_unified_record(self.model, "unused", row)
            row.update(face_vote="ANG", voice_vote="NEU", multimodal_vote="SAD")
            second = predict_unified_record(self.model, "unused", row)
        self.assertEqual(first, second)

    def test_input_shape_mismatch_fails_before_decoder(self):
        with self.assertRaisesRegex(ValueError, "visual feature shape"):
            self.model.decoder_inputs(self.features[0][:, :-1], self.features[1], "HAP")


if __name__ == "__main__":
    unittest.main()
