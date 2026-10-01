"""Checks for CREMA-D's real vote formats and paired manifest invariants."""

import hashlib
import json
import sys
import tempfile
import unittest
import wave
from array import array
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reachy_emotions.data.cremad import (_audio_quality, _extract, _normalize_audio_volume, _visual_usable,
                                         actor_splits, load_manifest,
                                         prepare_dataset, read_agreements, read_demographics,
                                         read_summary, validate_manifest)


class CremaDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "processedResults").mkdir()
        (self.root / "VideoFlash").mkdir()
        (self.root / "AudioWAV").mkdir()
        (self.root / "VideoDemographics.csv").write_text(
            '"ActorID","Age","Sex","Race","Ethnicity"\n'
            '1001,51,"Male","Caucasian","Not Hispanic"\n'
            '1002,21,"Female","Asian","Hispanic"\n', encoding="utf-8")
        (self.root / "LICENSE.txt").write_text("fixture ODbL and DbCL terms\n", encoding="utf-8")
        (self.root / "processedResults" / "summaryTable.csv").write_text(
            '"","FileName","VoiceVote","VoiceLevel","FaceVote","FaceLevel","MultiModalVote","MultiModalLevel"\n'
            '"1","1001_IEO_HAP_LO","N","71.67","H","57","H","57.38"\n'
            '"2","1001_IEO_SAD_HI","N","60.14","A:F","30.67:51.00","N","63.5"\n'
            '"3","1002_IEO_NEU_XX","N","69.1","N","92.22","N","64.78"\n',
            encoding="utf-8",
        )
        (self.root / "processedResults" / "tabulatedVotes.csv").write_text(
            '"","A","D","F","H","N","S","fileName","numResponses","agreement","emoVote"\n'
            '"100001",0,0,0,3,6,0,"1001_IEO_HAP_LO",9,0.6666667,"N"\n'
            '"100002",0,0,0,0,8,0,"1001_IEO_SAD_HI",8,1,"N"\n'
            '"100003",0,0,0,0,9,0,"1002_IEO_NEU_XX",9,1,"N"\n'
            '"200001",0,0,0,9,1,0,"1001_IEO_HAP_LO",10,0.9,"H"\n'
            '"200002",3,0,3,0,2,0,"1001_IEO_SAD_HI",8,0.375,"A:F"\n'
            '"200003",0,0,0,0,8,0,"1002_IEO_NEU_XX",8,1,"N"\n'
            '"300001",0,0,0,8,1,0,"1001_IEO_HAP_LO",9,0.8888889,"H"\n'
            '"300002",0,0,0,0,8,0,"1001_IEO_SAD_HI",8,1,"N"\n'
            '"300003",0,0,0,0,9,0,"1002_IEO_NEU_XX",9,1,"N"\n',
            encoding="utf-8",
        )
        for clip in ("1001_IEO_HAP_LO", "1001_IEO_SAD_HI", "1002_IEO_NEU_XX"):
            (self.root / "VideoFlash" / (clip + ".flv")).write_bytes(b"FLV sample fixture")
            (self.root / "AudioWAV" / (clip + ".wav")).write_bytes(b"WAV sample fixture")

    def test_official_votes_and_agreements_are_distinct(self):
        rows = read_summary(self.root)
        self.assertEqual(rows[0]["face_vote"], "HAP")
        self.assertEqual(rows[0]["voice_vote"], "NEU")
        self.assertEqual(rows[1]["face_vote"], "unknown")
        agreements = read_agreements(self.root)
        self.assertAlmostEqual(agreements[(rows[0]["clip_id"], "face")], 0.9)
        self.assertAlmostEqual(agreements[(rows[0]["clip_id"], "voice")], 0.6666667)
        self.assertNotEqual(agreements[(rows[0]["clip_id"], "face")], float(57))

    def test_split_is_stable_and_actor_disjoint(self):
        actors = [str(i) for i in range(1001, 1092)]
        splits = actor_splits(actors, seed=17)
        self.assertEqual(splits, actor_splits(reversed(actors), seed=17))
        self.assertEqual({name: list(splits.values()).count(name) for name in {"train", "validation", "test"}},
                         {"train": 64, "validation": 13, "test": 14})

    def test_demographic_balancing_keeps_rare_groups_in_held_out_splits(self):
        actors = [str(i) for i in range(1001, 1092)]
        demographics = {
            actor: {"age": 20 + index % 50, "Sex": "Male" if index % 2 else "Female",
                    "Race": "Asian" if index < 7 else "Caucasian",
                    "Ethnicity": "Hispanic" if 7 <= index < 17 else "Not Hispanic"}
            for index, actor in enumerate(actors)
        }
        splits = actor_splits(actors, seed=42, demographics=demographics)
        self.assertEqual(splits, actor_splits(reversed(actors), seed=42, demographics=demographics))
        for split in ("validation", "test"):
            self.assertTrue(any(splits[actor] == split and demographics[actor]["Race"] == "Asian"
                                for actor in actors))
            self.assertTrue(any(splits[actor] == split and demographics[actor]["Ethnicity"] == "Hispanic"
                                for actor in actors))

    def test_official_demographics_cover_all_actors(self):
        demographics = read_demographics(self.root, ("1001", "1002"))
        self.assertEqual(demographics["1001"]["age"], 51)
        with self.assertRaisesRegex(ValueError, "disagree with clips"):
            read_demographics(self.root, ("1001",))

    def test_dry_run_does_not_decode_or_write(self):
        out = self.root / "prepared"
        summary = prepare_dataset(self.root, out, dry_run=True, limit=2)
        self.assertEqual(summary["clips"], 2)
        self.assertEqual(summary["total_metadata_clips"], 3)
        self.assertEqual(sum(group["clips"] for group in summary["split_balance"].values()), 3)
        self.assertEqual(sum(group["face_vote"].get("unknown", 0) for group in summary["split_balance"].values()), 1)
        self.assertEqual(len(summary["metadata_sha256"]["summaryTable.csv"]), 64)
        self.assertFalse(out.exists())

    def test_missing_vote_row_is_rejected_before_extraction(self):
        path = self.root / "processedResults" / "tabulatedVotes.csv"
        text = path.read_text(encoding="utf-8")
        path.write_text("\n".join(line for line in text.splitlines() if not line.startswith('"300003"')) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Vote tables disagree"):
            prepare_dataset(self.root, self.root / "prepared", dry_run=True)

    def test_mismatched_vote_counts_are_rejected(self):
        path = self.root / "processedResults" / "tabulatedVotes.csv"
        text = path.read_text(encoding="utf-8").replace('"200001",0,0,0,9,1,0', '"200001",0,0,0,8,1,0')
        path.write_text(text, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid vote counts"):
            prepare_dataset(self.root, self.root / "prepared", dry_run=True)

    def test_preparation_writes_auditable_manifest_without_real_decoder(self):
        def fake_extract(_video, _source_audio, output_root, clip_id, _duration):
            frame_dir = output_root / "frames" / clip_id
            frame_dir.mkdir(parents=True)
            paths = []
            hashes = []
            for index in range(3):
                path = frame_dir / ("frame_%d.png" % index)
                path.write_bytes(("image%d" % index).encode())
                paths.append(path.relative_to(output_root).as_posix())
                hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
            audio = output_root / "audio" / (clip_id + ".wav")
            audio.parent.mkdir()
            audio.write_bytes(b"wave fixture")
            return {
                "frame_paths": paths, "frame_timestamps_ms": [350, 500, 650], "frame_sha256s": hashes,
                "frame_quality": [{"face_status": "detected", "aligned": True}] * 3,
                "visual_usable": True,
                "audio_path": audio.relative_to(output_root).as_posix(),
                "audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                "audio_sample_rate": 16000, "audio_duration_ms": 1000,
                "audio_quality": {"rms": 0.1, "clipping_fraction": 0},
                "audio_quality_status": "accepted", "av_duration_delta_ms": 0,
                "av_alignment_status": "accepted", "paired_usable": True,
            }

        output = self.root / "prepared"
        with patch("reachy_emotions.data.cremad.shutil.which", return_value="/fake/ffmpeg"), \
             patch("reachy_emotions.data.cremad._tool_versions", return_value={"ffmpeg": "test", "ffprobe": "test", "opencv": None}), \
             patch("reachy_emotions.data.cremad._video_duration_ms", return_value=1000), \
             patch("reachy_emotions.data.cremad._extract", side_effect=fake_extract):
            summary = prepare_dataset(self.root, output, seed=17, limit=1)
        records = load_manifest(output / "manifest.jsonl")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["source_dataset"], "CREMA-D")
        self.assertEqual(records[0]["visual_target_json"], {"presented_expression": "HAP"})
        self.assertEqual(validate_manifest(records, root=output, verify_files=True), summary["split_clips"])
        with self.assertRaisesRegex(ValueError, "Paired usability conflicts"):
            validate_manifest([dict(records[0], paired_usable=False)])
        provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["actor_split_seed"], 17)
        self.assertEqual(provenance["tool_versions"]["ffmpeg"], "test")
        self.assertEqual(summary["quality_counts"]["paired_usable"], 1)
        self.assertTrue((output / "split_balance.json").is_file())
        self.assertTrue((output / "demographic_balance.json").is_file())
        attribution = json.loads((output / "dataset_attribution.json").read_text(encoding="utf-8"))
        self.assertEqual(attribution["dataset"], "CREMA-D")
        self.assertEqual(len(attribution["source_license_sha256"]), 64)

    def test_lfs_pointer_fails_clearly(self):
        (self.root / "VideoFlash" / "1001_IEO_HAP_LO.flv").write_text(
            "version https://git-lfs.github.com/spec/v1\noid sha256:fixture\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(RuntimeError, "Git LFS pointer"):
            prepare_dataset(self.root, self.root / "prepared", dry_run=True, limit=1)

    def test_named_clip_can_use_hash_verified_smoke_video(self):
        video_dir = self.root / "sample_videos"
        video_dir.mkdir()
        sample = video_dir / "1001_IEO_HAP_LO.flv"
        sample.write_bytes((self.root / "VideoFlash" / sample.name).read_bytes())
        report = prepare_dataset(self.root, self.root / "prepared", dry_run=True,
                                 clip_ids=[sample.stem], video_dir=video_dir)
        self.assertEqual(report["selected_clip_ids"], [sample.stem])
        self.assertEqual(report["clips"], 1)
        sample.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "does not match official"):
            prepare_dataset(self.root, self.root / "prepared", dry_run=True,
                            clip_ids=[sample.stem], video_dir=video_dir)

    def test_custom_video_dir_requires_explicit_clip(self):
        with self.assertRaisesRegex(ValueError, "require explicit clip IDs"):
            prepare_dataset(self.root, self.root / "prepared", dry_run=True, video_dir=self.root)

    def test_named_clip_can_use_hash_verified_wav(self):
        audio_dir = self.root / "sample_audio"
        audio_dir.mkdir()
        sample = audio_dir / "1001_IEO_HAP_LO.wav"
        sample.write_bytes((self.root / "AudioWAV" / sample.name).read_bytes())
        report = prepare_dataset(self.root, self.root / "prepared", dry_run=True,
                                 clip_ids=[sample.stem], audio_dir=audio_dir)
        self.assertEqual(report["clips"], 1)
        sample.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "Audio override does not match"):
            prepare_dataset(self.root, self.root / "prepared", dry_run=True,
                            clip_ids=[sample.stem], audio_dir=audio_dir)

    def test_audio_quality_flags_clipping(self):
        path = self.root / "clipped.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(array("h", [0] * 320 + [32767] * 320).tobytes())
        quality = _audio_quality(path)
        self.assertGreater(quality["clipping_fraction"], 0.4)

    def test_volume_normalization_limits_gain_and_preserves_wav_timing(self):
        path = self.root / "quiet.wav"
        original = array("h", [655] * 16000)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(original.tobytes())
        before = _audio_quality(path)
        settings = _normalize_audio_volume(path)
        after = _audio_quality(path)
        with wave.open(str(path), "rb") as wav:
            self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()),
                             (1, 2, 16000, 16000))
        self.assertAlmostEqual(settings["applied_gain"], 4.0)
        self.assertAlmostEqual(after["rms"], before["rms"] * 4, places=4)

    def test_internal_pause_does_not_change_speech_gain(self):
        gains = []
        for gap_seconds in (0, 4):
            path = self.root / ("gap_%d.wav" % gap_seconds)
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(array("h", [1000] * 16000 + [0] * (gap_seconds * 16000)).tobytes())
            gains.append(_normalize_audio_volume(path)["applied_gain"])
            with wave.open(str(path), "rb") as wav:
                samples = array("h")
                samples.frombytes(wav.readframes(wav.getnframes()))
            self.assertEqual(len(samples), (1 + gap_seconds) * 16000)
            self.assertTrue(all(value == 0 for value in samples[16000:]))
        self.assertAlmostEqual(gains[0], gains[1])

    def test_volume_normalization_respects_peak_ceiling(self):
        path = self.root / "peaky.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(array("h", [1000] * 999 + [20000]).tobytes())
        settings = _normalize_audio_volume(path)
        with wave.open(str(path), "rb") as wav:
            samples = array("h")
            samples.frombytes(wav.readframes(wav.getnframes()))
        self.assertLess(settings["applied_gain"], 2)
        self.assertLessEqual(max(abs(value) for value in samples) / 32768, 0.95)

    def test_sparse_source_stays_under_review_after_normalization(self):
        def fake_run(command):
            output = Path(command[-1])
            if output.suffix == ".png":
                output.write_bytes(b"frame")
            else:
                with wave.open(str(output), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(16000)
                    wav.writeframes(array("h", [1000] * 320 + [0] * 15680).tobytes())

        with patch("reachy_emotions.data.cremad._run", side_effect=fake_run), \
             patch("reachy_emotions.data.cremad._face_crop", return_value={"face_status": "detected"}):
            result = _extract(self.root / "source.flv", self.root / "source.wav", self.root / "output",
                              "1001_IEO_HAP_LO", 1000)
        self.assertGreater(result["audio_quality"]["rms"], result["audio_quality_before_normalization"]["rms"])
        self.assertEqual(result["audio_quality_status"], "review")
        self.assertFalse(result["paired_usable"])

    def test_two_detected_face_frames_are_usable(self):
        frames = [{"face_status": "detected"}, {"face_status": "detected"}, {"face_status": "not_detected"}]
        self.assertTrue(_visual_usable(frames))
        frames[1]["face_status"] = "opencv_unavailable"
        self.assertFalse(_visual_usable(frames))

    def test_manifest_rejects_actor_leakage_and_bad_alignment(self):
        template = {
            "sample_id": "1001_IEO_HAP_LO", "actor_id": "1001", "clip_id": "1001_IEO_HAP_LO", "split": "train",
            "source_dataset": "CREMA-D", "performed_label": "HAP",
            "face_vote": "HAP", "voice_vote": "NEU", "multimodal_vote": "HAP",
            "visual_target_json": {"presented_expression": "HAP"}, "audio_target": "NEU", "fusion_target": "HAP",
            "frame_timestamps_ms": [350, 500, 650], "frame_paths": ["frames/1001_IEO_HAP_LO/a.png", "frames/1001_IEO_HAP_LO/b.png", "frames/1001_IEO_HAP_LO/c.png"],
            "frame_sha256s": ["a" * 64, "b" * 64, "c" * 64], "video_duration_ms": 1000,
            "audio_path": "audio/1001_IEO_HAP_LO.wav", "audio_sha256": "d" * 64, "audio_sample_rate": 16000,
            "audio_duration_ms": 1000,
        }
        self.assertEqual(validate_manifest([template]), {"train": 1})
        second = dict(template, sample_id="1001_IEO_SAD_HI", clip_id="1001_IEO_SAD_HI", performed_label="SAD", split="test",
                      frame_paths=["frames/1001_IEO_SAD_HI/a.png", "frames/1001_IEO_SAD_HI/b.png", "frames/1001_IEO_SAD_HI/c.png"],
                      audio_path="audio/1001_IEO_SAD_HI.wav")
        with self.assertRaisesRegex(ValueError, "Actor leakage"):
            validate_manifest([template, second])
        with self.assertRaisesRegex(ValueError, "AV duration"):
            validate_manifest([dict(template, audio_duration_ms=2100)])
        with self.assertRaisesRegex(ValueError, "Media paths do not match"):
            validate_manifest([dict(template, audio_path="audio/another_clip.wav")])
        with self.assertRaisesRegex(ValueError, "Source audio does not match"):
            validate_manifest([dict(template, source_audio_path="AudioWAV/another_clip.wav")])
        with self.assertRaisesRegex(ValueError, "Unsupported source dataset"):
            validate_manifest([dict(template, source_dataset="other")])
        with self.assertRaisesRegex(ValueError, "Unsafe relative media path"):
            validate_manifest([dict(template, audio_path="../audio/1001_IEO_HAP_LO.wav")])


if __name__ == "__main__":
    unittest.main()
