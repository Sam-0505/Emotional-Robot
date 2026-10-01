"""Offline checks for the Grace submission and pilot gates."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.select_prepared_clip import select_clip
from scripts.verify_hprc_pilot import verify_pilot


REPO = Path(__file__).resolve().parents[1]
SHA_A = "a" * 40
SHA_B = "b" * 40


class HprcScriptsTests(unittest.TestCase):
    def test_select_paired_validation_clip(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.jsonl"
            rows = [
                {"sample_id": "B", "actor_id": "1", "split": "validation", "frame_paths": ["a", "b", "c"],
                 "audio_path": "audio", "face_vote": "HAP", "voice_vote": "HAP", "multimodal_vote": "HAP",
                 "paired_usable": True},
                {"sample_id": "A", "actor_id": "1", "split": "validation", "frame_paths": ["a", "b", "c"],
                 "audio_path": "audio", "face_vote": "HAP", "voice_vote": "HAP", "multimodal_vote": "HAP",
                 "paired_usable": False},
            ]
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            self.assertEqual(select_clip(manifest), "B")
            self.assertEqual(select_clip(manifest, paired_only=False), "A")

    def test_pilot_checks_revisions_manifest_and_paired_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared"
            prepared.mkdir()
            manifest = prepared / "manifest.jsonl"
            manifest.write_text("sample\n", encoding="utf-8")
            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            for subdir, revision in (("visual_adapter", SHA_A), ("audio_head", SHA_B)):
                folder = root / "pilot" / subdir
                folder.mkdir(parents=True)
                (folder / "reload_verification.json").write_text(json.dumps({
                    "reload_verified": True, "revision": revision, "manifest_sha256": digest,
                }), encoding="utf-8")
            observation = root / "pilot" / "observation.json"
            observation.write_text('{"source": "audio_visual"}', encoding="utf-8")
            verify_pilot(root, SHA_A, SHA_B)
            for subdir in ("visual_adapter", "audio_head"):
                folder = root / "full" / subdir
                folder.mkdir(parents=True)
                source = root / "pilot" / subdir / "reload_verification.json"
                (folder / "reload_verification.json").write_bytes(source.read_bytes())
            verify_pilot(root, SHA_A, SHA_B, stage="full")
            with self.assertRaisesRegex(ValueError, "checkpoint or manifest differs"):
                verify_pilot(root, SHA_B, SHA_A)
            manifest.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checkpoint or manifest differs"):
                verify_pilot(root, SHA_A, SHA_B)
            manifest.write_text("sample\n", encoding="utf-8")
            observation.write_text('{"source": "visual"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "both modalities"):
                verify_pilot(root, SHA_A, SHA_B)

    def test_submit_dry_run_does_not_create_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "CREMA-D"
            dataset.mkdir()
            venv = root / "venv"
            (venv / "bin").mkdir(parents=True)
            (venv / "bin" / "activate").touch()
            run_root = root / "run"
            process = subprocess.run(["bash", str(REPO / "scripts" / "hprc_submit.sh"),
                                      "--stage", "pilot", "--dataset", str(dataset),
                                      "--run-root", str(run_root), "--venv", str(venv),
                                      "--visual-revision", SHA_A, "--audio-revision", SHA_B,
                                      "--dry-run"], capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn("gpu:a100:1", process.stdout)
            self.assertFalse(run_root.exists())

    def test_prepare_requires_explicit_cpu_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "CREMA-D"
            dataset.mkdir()
            venv = root / "venv"
            (venv / "bin").mkdir(parents=True)
            (venv / "bin" / "activate").touch()
            process = subprocess.run(["bash", str(REPO / "scripts" / "hprc_submit.sh"),
                                      "--stage", "prepare", "--dataset", str(dataset),
                                      "--run-root", str(root / "run"), "--venv", str(venv),
                                      "--dry-run"], capture_output=True, text=True)
            self.assertEqual(process.returncode, 2)
            self.assertIn("--partition", process.stderr)

    def test_setup_dry_run_does_not_create_venv_or_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "CREMA-D"
            dataset.mkdir()
            venv = root / "venv"
            run_root = root / "run"
            process = subprocess.run(["bash", str(REPO / "scripts" / "hprc_submit.sh"),
                                      "--stage", "prepare", "--dataset", str(dataset),
                                      "--run-root", str(run_root), "--venv", str(venv),
                                      "--partition", "cpu", "--setup-env", "--dry-run"],
                                     capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn("requirements-hprc.txt", process.stdout)
            self.assertFalse(venv.exists())
            self.assertFalse(run_root.exists())

    def test_submission_checks_venv_before_queuing_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "CREMA-D"
            dataset.mkdir()
            venv_bin = root / "venv" / "bin"
            venv_bin.mkdir(parents=True)
            (venv_bin / "activate").touch()
            fake_python = venv_bin / "python"
            fake_python.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
            fake_python.chmod(0o755)
            fake_bin = root / "fake-bin"
            fake_bin.mkdir()
            for name in ("sbatch", "ffmpeg", "ffprobe"):
                executable = fake_bin / name
                executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                executable.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"
            run_root = root / "run"
            process = subprocess.run(["bash", str(REPO / "scripts" / "hprc_submit.sh"),
                                      "--stage", "prepare", "--dataset", str(dataset),
                                      "--run-root", str(run_root), "--venv", str(venv_bin.parent),
                                      "--partition", "cpu"], capture_output=True, text=True, env=env)
            self.assertEqual(process.returncode, 2)
            self.assertIn("--setup-env", process.stderr)
            self.assertFalse(run_root.exists())


if __name__ == "__main__":
    unittest.main()
