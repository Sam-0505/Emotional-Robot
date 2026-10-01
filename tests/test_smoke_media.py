"""Offline sample-pair checks without CREMA-D downloads or FFmpeg."""

import hashlib
import tempfile
import unittest
import wave
from array import array
from pathlib import Path
from unittest.mock import patch

from scripts.smoke_media import inspect_pair
from scripts.check_hprc import media_counts


class SampleMediaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.dataset = root / "CREMA-D"
        (self.dataset / "VideoFlash").mkdir(parents=True)
        (self.dataset / "AudioWAV").mkdir()
        self.video = root / "1001_DFA_ANG_XX.flv"
        self.audio = root / "1001_DFA_ANG_XX.wav"
        self.video.write_bytes(b"FLV\x01\x05\x00\x00\x00\x09" + b"\x00" * 12)
        with wave.open(str(self.audio), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(array("h", [100] * 1600).tobytes())
        for source, directory in ((self.video, "VideoFlash"), (self.audio, "AudioWAV")):
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            (self.dataset / directory / source.name).write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                f"oid sha256:{digest}\nsize {source.stat().st_size}\n", encoding="ascii")
        self.rows = [{"clip_id": self.video.stem, "actor_id": "1001", "face_vote": "ANG",
                      "voice_vote": "ANG", "multimodal_vote": "ANG"}]
        self.details = {(self.video.stem, modality): (0.8, "A")
                        for modality in ("face", "voice", "multimodal")}

    def inspect(self):
        with patch("scripts.smoke_media.read_summary", return_value=self.rows), \
             patch("scripts.smoke_media._read_vote_details", return_value=self.details), \
             patch("scripts.smoke_media.validate_vote_metadata"):
            return inspect_pair(self.dataset, self.video, self.audio)

    def test_pair_hashes_and_wav_header(self):
        result = self.inspect()
        self.assertEqual(result["media_status"], "verified_bytes_and_metadata")
        self.assertEqual(result["audio_duration_ms"], 100)
        self.assertEqual(result["labels"]["face_vote"], "ANG")

    def test_wrong_lfs_hash_is_rejected(self):
        (self.dataset / "VideoFlash" / self.video.name).write_text(
            "version https://git-lfs.github.com/spec/v1\n"
            f"oid sha256:{'0' * 64}\nsize {self.video.stat().st_size}\n", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.inspect()

    def test_pointer_is_not_mistaken_for_video(self):
        self.video.write_bytes((self.dataset / "VideoFlash" / self.video.name).read_bytes())
        with self.assertRaisesRegex(ValueError, "Git LFS pointer"):
            self.inspect()

    def test_hprc_scan_distinguishes_media_from_pointers(self):
        report = media_counts(self.dataset, [self.video.stem, "1002_DFA_ANG_XX"])
        self.assertEqual(report["VideoFlash"], {"ready": 0, "lfs_pointer": 1, "missing": 1})
        (self.dataset / "VideoFlash" / self.video.name).write_bytes(self.video.read_bytes())
        report = media_counts(self.dataset, [self.video.stem])
        self.assertEqual(report["VideoFlash"]["ready"], 1)


if __name__ == "__main__":
    unittest.main()
