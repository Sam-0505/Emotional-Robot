"""Schedule integrity and real CPU optimizer/RNG resume equivalence."""

import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from test_unified import record, tiny_model
from reachy_emotions.perception.unified import file_hash
from reachy_emotions.perception.unified_checkpoint import (resolve_resume, read_resume_metadata,
    training_windows, schedule_digest, validate_resume_plan)


class ResumeScheduleTests(unittest.TestCase):
    def test_extended_epochs_keep_prefix_and_flush_partial_accumulation(self):
        rows = [record(str(i)) for i in range(5)]
        windows = training_windows(rows, 4, 42, 2)
        self.assertEqual([len(item["rows"]) for item in windows[:3]], [2, 2, 1])
        self.assertEqual((windows[2]["next_epoch"], windows[2]["next_clip_index"]), (1, 0))
        prefix = [row for item in windows[:4] for row in item["rows"]]
        metadata = {"training_settings": {"seed": 42}, "progress": {
            "optimizer_steps": 4, "completed_microsteps": len(prefix),
            "next_epoch": windows[3]["next_epoch"], "next_clip_index": windows[3]["next_clip_index"],
            "schedule_sha256": schedule_digest(prefix)}}
        self.assertEqual(validate_resume_plan(metadata, {"seed": 42}, windows), 4)
        with self.assertRaisesRegex(ValueError, "settings mismatch.*seed"):
            validate_resume_plan(metadata, {"seed": 43}, windows)
        with self.assertRaisesRegex(ValueError, "no remaining"):
            validate_resume_plan(metadata, {"seed": 42}, windows[:4])
        metadata["progress"]["next_clip_index"] += 1
        with self.assertRaisesRegex(ValueError, "training position"):
            validate_resume_plan(metadata, {"seed": 42}, windows)

    def test_pointer_cannot_escape_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "last_checkpoint.json").write_text('{"checkpoint": "../elsewhere"}')
            with self.assertRaisesRegex(ValueError, "escapes"):
                resolve_resume(root)

    def test_colab_resume_uses_saved_settings_and_a_new_run_root(self):
        from scripts.run_unified_colab import build_command
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nemotron-revision.txt").write_text("a" * 40)
            (root / "wavlm-revision.txt").write_text("b" * 40)
            (root / "manifest.jsonl").touch()
            args = SimpleNamespace(base=root, manifest=root / "manifest.jsonl", run_root=root / "new",
                config=None, stage="full", resume=root / "old/full", epochs=4, gradient_accumulation=None,
                baseline_evaluation=None, save_every=25)
            metadata = {"training_settings": {"gradient_accumulation": 3, "learning_rate": .001,
                                               "seed": 24, "augment": True}}
            with patch("reachy_emotions.perception.unified_checkpoint.read_resume_metadata", return_value=(root, metadata)):
                command, run = build_command(args, root)
            for flag, value in (("--epochs", "4"), ("--gradient-accumulation", "3"), ("--seed", "24"),
                                ("--learning-rate", "0.001"), ("--save-every", "25")):
                self.assertEqual(command[command.index(flag) + 1], value)
            self.assertIn("--resume", command)
            self.assertNotIn("--config", command)
            self.assertNotIn("--pilot", command)
            self.assertFalse(run.exists())


def run_cpu_training(manifest, output, epochs, max_steps=None, resume=None, fail_at=None):
    """Run the production loop/serializer with tiny encoders, no CUDA or HF downloads."""
    import numpy as np
    import torch
    from reachy_emotions.perception.unified import train_unified
    torch.set_num_threads(1)
    config = tiny_model().architecture
    calls = [0]

    def load(visual_revision, audio_revision, config, checkpoint=None, training=False):
        model = tiny_model()
        if checkpoint:
            root = Path(checkpoint)
            state = torch.load(root / "decoder_lora/adapter_model.safetensors", weights_only=True)
            model.visual.language_model.load_state_dict(state, strict=False)
            model.audio_projector.load_state_dict(torch.load(root / "audio_projector.pt", weights_only=True))
        return model

    def encode(model, manifest, row, rng=None):
        if rng:
            calls[0] += 1
            if fail_at is not None and calls[0] == fail_at:
                raise RuntimeError("simulated interrupted microstep")
            # Exercise every saved RNG, including augmentation and Torch stochasticity.
            scale = rng.uniform(.8, 1.2) + random.random() / 10 + np.random.random() / 10 + float(torch.rand(())) / 10
        else:
            scale = 1.
        images = [torch.full((1, 4, 1), float(i + 1) * scale) for i in range(3)]
        return model.encode(images, [.1 * scale, .2, .3, .4])

    with ExitStack() as stack:
        stack.enter_context(patch("reachy_emotions.perception.unified.load_unified", side_effect=load))
        stack.enter_context(patch("reachy_emotions.perception.unified.encode_record", side_effect=encode))
        stack.enter_context(patch("reachy_emotions.perception.unified.require_unified_pilot"))
        for name in ("synchronize", "reset_peak_memory_stats", "manual_seed_all"):
            stack.enter_context(patch("torch.cuda." + name))
        stack.enter_context(patch("torch.cuda.max_memory_allocated", return_value=0))
        stack.enter_context(patch("torch.cuda.get_device_name", return_value="tiny CPU fixture"))
        return train_unified(manifest, output, "a" * 40, "b" * 40, config, max_steps, epochs,
                             gradient_accumulation=2, learning_rate=.002, augment=True,
                             pilot=Path(manifest).parent / "pilot",
                             baseline_evaluation=Path(manifest).parent / "baseline.json",
                             resume=resume, save_every=2)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "install torch for optimizer resume tests")
class ResumeTensorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        rows = [record(str(i), target=label) for i, label in enumerate(("HAP", "SAD", "ANG", "NEU", "FEA"))]
        rows += [record("val", "validation")]
        self.manifest = self.root / "manifest.jsonl"
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        (self.root / "baseline.json").write_text(json.dumps({
            "visual_face_vote": {}, "audio_voice_vote": {}, "fusion_multimodal_vote": {},
            "provenance": {"manifest_sha256": file_hash(self.manifest), "selection_split": "validation", "sample_ids": ["val"]}}))

    def assert_weights_equal(self, left, right):
        import torch
        for name in ("audio_projector.pt", "decoder_lora/adapter_model.safetensors"):
            a = torch.load(Path(left) / name, weights_only=True)
            b = torch.load(Path(right) / name, weights_only=True)
            self.assertEqual(set(a), set(b))
            for key in a:
                self.assertTrue(torch.equal(a[key], b[key]), name + ": " + key)

    def test_fresh_process_resume_matches_uninterrupted_training_and_keeps_history(self):
        uninterrupted = self.root / "uninterrupted"
        partial, continued = self.root / "partial", self.root / "continued"
        run_cpu_training(self.manifest, uninterrupted, epochs=3)
        run_cpu_training(self.manifest, partial, epochs=2, max_steps=4)
        checkpoint, metadata = read_resume_metadata(partial)
        self.assertEqual(metadata["progress"]["optimizer_steps"], 4)
        self.assertEqual(metadata["progress"]["next_epoch"], 1)
        self.assertEqual(metadata["progress"]["next_clip_index"], 2)
        self.assertEqual(len(list((partial / "checkpoints").glob("step-*"))), 3)
        state_hash = file_hash(checkpoint / "training_state.pt")
        env = os.environ.copy()
        repo = Path(__file__).resolve().parents[1]
        env["PYTHONPATH"] = os.pathsep.join([str(repo / "src"), str(repo / "tests")])
        result = subprocess.run([sys.executable, "-c",
            "import sys; from test_unified_resume import run_cpu_training; run_cpu_training(sys.argv[1], sys.argv[2], 3, resume=sys.argv[3])",
            str(self.manifest), str(continued), str(partial)], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assert_weights_equal(uninterrupted, continued)
        a = json.loads((uninterrupted / "unified_metadata.json").read_text())
        b = json.loads((continued / "unified_metadata.json").read_text())
        self.assertEqual(a["losses"], b["losses"])
        self.assertEqual(a["schedule_sha256"], b["schedule_sha256"])
        self.assertEqual(b["optimizer_steps"], 9)
        self.assertEqual(b["completed_epochs"], 3)
        self.assertEqual(file_hash(checkpoint / "training_state.pt"), state_hash)

    def test_mid_microstep_failure_recovers_from_completed_optimizer_boundary(self):
        partial = self.root / "interrupted"
        with self.assertRaisesRegex(RuntimeError, "interrupted microstep"):
            # Epoch 0 saved at step 3; interrupt the second microstep of epoch 1's first window.
            run_cpu_training(self.manifest, partial, epochs=2, fail_at=7)
        _, metadata = read_resume_metadata(partial)
        self.assertEqual(metadata["progress"]["optimizer_steps"], 3)
        self.assertFalse((partial / "unified_metadata.json").exists())
        continued, full = self.root / "recovered", self.root / "reference"
        run_cpu_training(self.manifest, continued, epochs=2, resume=partial)
        run_cpu_training(self.manifest, full, epochs=2)
        self.assert_weights_equal(full, continued)

    def test_incomplete_snapshot_does_not_replace_last_checkpoint(self):
        import torch
        from reachy_emotions.perception.unified_checkpoint import save_training_checkpoint
        partial = self.root / "partial"
        run_cpu_training(self.manifest, partial, epochs=1, max_steps=2)
        _, metadata = read_resume_metadata(partial)
        pointer = (partial / "last_checkpoint.json").read_bytes()
        model = tiny_model()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        progress = {**metadata["progress"], "optimizer_steps": 3}
        with patch("reachy_emotions.perception.unified_checkpoint.save_model_weights", side_effect=RuntimeError("write interrupted")):
            with self.assertRaisesRegex(RuntimeError, "write interrupted"):
                save_training_checkpoint(partial, model, optimizer, metadata, progress, {}, random.Random(42))
        self.assertEqual((partial / "last_checkpoint.json").read_bytes(), pointer)
        self.assertEqual(read_resume_metadata(partial)[1]["progress"]["optimizer_steps"], 2)
        self.assertFalse((partial / "checkpoints/step-00000003").exists())

    def test_tampered_state_changed_settings_and_existing_output_rejected(self):
        partial = self.root / "partial"
        run_cpu_training(self.manifest, partial, epochs=1, max_steps=2)
        with self.assertRaisesRegex(FileExistsError, "new output/run root"):
            run_cpu_training(self.manifest, partial, epochs=2, resume=partial)
        (self.root / "baseline.json").write_text((self.root / "baseline.json").read_text() + " ")
        with self.assertRaisesRegex(ValueError, "settings mismatch.*baseline"):
            run_cpu_training(self.manifest, self.root / "bad", epochs=2, resume=partial)
        self.assertFalse((self.root / "bad").exists())
        checkpoint, _ = read_resume_metadata(partial)
        (checkpoint / "training_state.pt").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "training-state hash"):
            read_resume_metadata(partial)


if __name__ == "__main__":
    unittest.main()
