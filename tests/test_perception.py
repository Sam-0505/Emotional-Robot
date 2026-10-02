"""Offline checks for paired clip evaluation and abstention behavior."""

import json
import hashlib
import importlib.util
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from reachy_emotions.perception.fusion import (FusionConfig, aggregate_frames,
                                                fit_fusion, fit_temperature, fuse_predictions)
from reachy_emotions.perception.labels import EXPRESSION_LABELS, UNKNOWN, one_hot
from reachy_emotions.perception.manifest import assert_actor_disjoint, read_manifest
from reachy_emotions.perception.metrics import evaluate_predictions
from reachy_emotions.perception.audio import augment_waveform, load_audio_expert, verify_audio_head
from reachy_emotions.perception.revision import require_commit_sha
from reachy_emotions.perception.visual import (discover_decoder_targets, load_visual_expert,
                                               training_schedule, augment_training_image,
                                               select_visual_training_records)
from scripts.evaluate_perception import evaluate


def record(sample_id, actor_id, split, label="HAP"):
    return {"sample_id": sample_id, "actor_id": actor_id, "split": split,
            "frame_paths": ["frames/a.png", "frames/b.png", "frames/c.png"],
            "audio_path": "audio/a.wav", "face_vote": label, "voice_vote": label,
            "multimodal_vote": label}


class ManifestTests(unittest.TestCase):
    def test_actor_cannot_cross_splits(self):
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            assert_actor_disjoint([record("one", "actor1", "train"),
                                   record("two", "actor1", "test")])

    def test_three_frames_are_one_clip_record(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.jsonl"
            rows = [record("one", "actor1", "train"), record("two", "actor2", "test")]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            self.assertEqual(len(read_manifest(path)), 2)

    def test_duplicate_clip_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate sample"):
            assert_actor_disjoint([record("same", "a", "train"), record("same", "b", "test")])


class FusionTests(unittest.TestCase):
    def test_three_frames_aggregate_once(self):
        scores, quality = aggregate_frames([one_hot("HAP"), one_hot("HAP"), one_hot("SAD")],
                                           [1, 1, 0])
        self.assertAlmostEqual(scores["HAP"], 1.0)
        self.assertAlmostEqual(quality, 2 / 3)

    def test_invalid_face_frame_is_ignored(self):
        scores, quality = aggregate_frames([one_hot("HAP"), None, None], [1, 0, 0])
        self.assertEqual(scores["HAP"], 1.0)
        self.assertAlmostEqual(quality, 1 / 3)

    def test_missing_modalities_abstain(self):
        result = fuse_predictions(None, None)
        self.assertEqual(result["presented_expression"], UNKNOWN)
        self.assertTrue(result["abstained"])
        self.assertEqual(result["source"], "none")

    def test_valid_audio_can_fallback(self):
        result = fuse_predictions(one_hot("SAD"), one_hot("HAP"),
                                  visual_quality=0.0, audio_quality=1.0)
        self.assertEqual(result["presented_expression"], "HAP")
        self.assertEqual(result["source"], "audio")

    def test_zero_weight_is_not_reported_as_paired_fusion(self):
        result = fuse_predictions(one_hot("SAD"), one_hot("HAP"),
                                  config=FusionConfig(visual_weight=0.0))
        self.assertEqual(result["source"], "audio")
        result = fuse_predictions(one_hot("SAD"), one_hot("HAP"),
                                  config=FusionConfig(visual_weight=1.0))
        self.assertEqual(result["source"], "visual")

    def test_disagreement_abstains_when_weak(self):
        result = fuse_predictions(one_hot("HAP", 0.3), one_hot("SAD", 0.3))
        self.assertTrue(result["abstained"])

    def test_fitting_rejects_test_rows(self):
        row = {"split": "test", "multimodal_vote": "HAP",
               "visual_scores": one_hot("HAP"), "audio_scores": one_hot("HAP")}
        with self.assertRaisesRegex(ValueError, "validation"):
            fit_fusion([row])
        with self.assertRaisesRegex(ValueError, "validation"):
            fit_temperature([row], "visual_scores", "multimodal_vote")

    def test_fusion_selection_prioritizes_macro_f1_over_majority_accuracy(self):
        rows = []
        for index in range(6):
            truth = "ANG" if index == 5 else "HAP"
            visual = "ANG" if index in (3, 4, 5) else "HAP"
            rows.append({"split": "validation", "multimodal_vote": truth,
                         "visual_scores": one_hot(visual), "audio_scores": one_hot("HAP")})
        selected = fit_fusion(rows, weights=[0.0, 1.0])
        self.assertEqual(selected.visual_weight, 1.0)

    def test_config_round_trip(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fusion.json"
            config = FusionConfig(visual_weight=0.6, audio_temperature=1.5)
            config.save(path)
            self.assertEqual(FusionConfig.load(path), config)


class MetricsTests(unittest.TestCase):
    def test_clip_level_metrics_count_abstention_as_error(self):
        rows = []
        for index, label in enumerate(EXPRESSION_LABELS):
            rows.append({"sample_id": str(index), "actor_id": "actor1",
                         "truth": label, "prediction": label if index != 0 else UNKNOWN})
        metrics = evaluate_predictions(rows, "truth", "prediction")
        self.assertEqual(metrics["clips"], 6)
        self.assertAlmostEqual(metrics["accuracy"], 5 / 6)
        self.assertAlmostEqual(metrics["abstention_rate"], 1 / 6)
        self.assertEqual(metrics["confusion"]["ANG"][UNKNOWN], 1)


class AudioTests(unittest.TestCase):
    def test_training_augmentation_is_seeded_and_does_not_mutate_input(self):
        samples = [0.0, 0.25, -0.25, 0.5] * 20
        first = augment_waveform(samples, random.Random(17))
        second = augment_waveform(samples, random.Random(17))
        self.assertEqual(first, second)
        self.assertNotEqual(first, samples)
        self.assertEqual(samples[:4], [0.0, 0.25, -0.25, 0.5])
        self.assertTrue(all(-1 <= value <= 1 for value in first))

    def test_audio_reload_checks_provenance_before_loading_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "manifest.jsonl"
            manifest.write_text("{}\n", encoding="utf-8")
            revision = "a" * 40
            metadata = {"model_id": "microsoft/wavlm-base-plus", "revision": revision,
                        "manifest_sha256": "0" * 64}
            (root / "audio_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest changed"):
                verify_audio_head(manifest, root, revision)
            metadata["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            metadata["model_id"] = "wrong"
            (root / "audio_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                verify_audio_head(manifest, root, revision)


class VisualTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow is an optional training dependency")
    def test_training_image_augmentation_is_seeded(self):
        from PIL import Image

        image = Image.new("RGB", (32, 32))
        image.putdata([(x * 7, y * 7, (x + y) * 3) for y in range(32) for x in range(32)])
        first = augment_training_image(image, random.Random(17))
        second = augment_training_image(image, random.Random(17))
        self.assertEqual(first.size, image.size)
        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertNotEqual(first.tobytes(), image.tobytes())

    def test_decoder_target_discovery_uses_model_tree(self):
        class Tree:
            def named_modules(self):
                return iter((("layer.self_attn.q_proj", object()),
                             ("layer.self_attn.v_proj", object())))

        class Model:
            language_model = Tree()

        self.assertEqual(discover_decoder_targets(Model()), ["q_proj", "v_proj"])

    def test_seeded_schedule_and_accumulation_limit(self):
        rows = []
        for index in range(8):
            row = record(str(index), "actor%d" % index, "train")
            row["frame_quality"] = [{"face_status": "detected"},
                                    {"face_status": "not_detected"},
                                    {"face_status": "detected"}]
            rows.append(row)
        first = training_schedule(rows, seed=17, epochs=2, max_steps=3,
                                  gradient_accumulation=2)
        second = training_schedule(rows, seed=17, epochs=2, max_steps=3,
                                   gradient_accumulation=2)
        self.assertEqual([(row["sample_id"], index) for row, index in first],
                         [(row["sample_id"], index) for row, index in second])
        self.assertEqual(len(first), 6)
        self.assertTrue(all(index in (0, 2) for _, index in first))
        different = training_schedule(rows, seed=18, epochs=2, max_steps=3,
                                      gradient_accumulation=2)
        self.assertNotEqual([row["sample_id"] for row, _ in first],
                            [row["sample_id"] for row, _ in different])

    def test_schedule_rejects_non_train_clip(self):
        with self.assertRaisesRegex(ValueError, "non-training"):
            training_schedule([record("one", "actor", "test")])

    def test_visual_trainer_selects_only_training_split(self):
        rows = [record("train", "actor1", "train"),
                record("validation", "actor2", "validation"),
                record("test", "actor3", "test")]
        selected = select_visual_training_records(rows)
        self.assertEqual([row["sample_id"] for row in selected], ["train"])

    def test_visual_loader_overrides_flash_attention_default(self):
        loader = Mock()
        loaded_model = loader.from_pretrained.return_value.cuda.return_value
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True),
                                     bfloat16=object())
        fake_transformers = SimpleNamespace(AutoModel=loader, AutoTokenizer=Mock(),
                                            AutoImageProcessor=Mock())
        with patch.dict("sys.modules", {"torch": fake_torch,
                                        "transformers": fake_transformers,
                                        "PIL": SimpleNamespace(Image=object())}):
            model, _, _ = load_visual_expert("a" * 40)
        self.assertIs(model, loaded_model)
        self.assertEqual(loader.from_pretrained.call_args.kwargs["attn_implementation"], "eager")

    def test_pinned_revision_and_adapter_provenance(self):
        revision = "a" * 40
        self.assertEqual(require_commit_sha(revision), revision)
        for moving in ("main", "v1.0", "a" * 7, "A" * 40):
            with self.assertRaisesRegex(ValueError, "commit SHA"):
                load_visual_expert(moving)
            with self.assertRaisesRegex(ValueError, "commit SHA"):
                load_audio_expert(moving)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "missing training_metadata"):
                load_visual_expert(revision, adapter_path=temp)
            (Path(temp) / "training_metadata.json").write_text(
                json.dumps({"model_id": "wrong", "revision": revision}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_visual_expert(revision, adapter_path=temp)


class EvaluationTests(unittest.TestCase):
    def test_paired_evaluation_uses_validation_then_test(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            predictions = []
            for split, actor in (("validation", "actor_val"), ("test", "actor_test")):
                for label in EXPRESSION_LABELS:
                    sample_id = "%s_%s" % (split, label)
                    rows.append(record(sample_id, actor, split, label))
                    predictions.append({"sample_id": sample_id, "scores": one_hot(label, 0.01)})
            manifest = root / "manifest.jsonl"
            visual = root / "visual.jsonl"
            audio = root / "audio.jsonl"
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            visual.write_text("".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8")
            audio.write_text("".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8")
            metrics = evaluate(manifest, visual, audio, root / "out", visual)
            self.assertEqual(metrics["fusion_multimodal_vote"]["clips"], 6)
            self.assertEqual(metrics["fusion_multimodal_vote"]["macro_f1"], 1.0)
            self.assertEqual(metrics["visual_zero_shot_face_vote"]["macro_f1"], 1.0)
            self.assertEqual(metrics["visual_face_vote_macro_f1_gain_over_zero_shot"], 0.0)
            self.assertTrue((root / "out" / "fusion_config.json").exists())
            self.assertEqual(metrics["validation_selection"]["fusion_multimodal_vote"]["macro_f1"], 1.0)
            self.assertTrue((root / "out" / "validation_metrics.json").exists())

    def test_paired_evaluation_rejects_extra_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [record("val", "actor_val", "validation"),
                    record("test", "actor_test", "test")]
            manifest = root / "manifest.jsonl"
            visual = root / "visual.jsonl"
            audio = root / "audio.jsonl"
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            predictions = [{"sample_id": row["sample_id"], "scores": one_hot("HAP")}
                           for row in rows]
            audio.write_text("".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8")
            predictions.append({"sample_id": "stale_clip", "scores": one_hot("HAP")})
            visual.write_text("".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prediction IDs do not match"):
                evaluate(manifest, visual, audio, root / "out")


if __name__ == "__main__":
    unittest.main()
