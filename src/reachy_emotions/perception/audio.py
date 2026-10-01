"""Frozen WavLM encoder with a trainable VoiceVote classification head."""

import hashlib
import json
import math
import random
import wave
from array import array
from pathlib import Path

from .labels import EXPRESSION_LABELS
from .manifest import read_manifest, resolve_media_path
from .revision import require_commit_sha

MODEL_ID = "microsoft/wavlm-base-plus"


def read_pcm16_wav(path):
    """Read prepared CREMA-D audio, preserving amplitude and timing."""
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2 or handle.getframerate() != 16000:
            raise ValueError("audio must be mono, PCM16, 16 kHz WAV: %s" % path)
        raw = handle.readframes(handle.getnframes())
    samples = array("h")
    samples.frombytes(raw)
    if not samples:
        raise ValueError("audio file is empty: %s" % path)
    # array('h') follows host endian; the prepared WAV is little endian.
    import sys
    if sys.byteorder != "little":
        samples.byteswap()
    return [value / 32768.0 for value in samples]


def waveform_quality(samples):
    """Conservative clip-level gate for silence and severe clipping."""
    if not samples:
        return 0.0
    rms = math.sqrt(sum(value * value for value in samples) / len(samples))
    clipping = sum(abs(value) >= 0.98 for value in samples) / len(samples)
    if rms < 0.005 or clipping > 0.10:
        return 0.0
    return 1.0


def augment_waveform(samples, rng):
    """Training-only mild gain, noise and bandwidth variation; never use at eval."""
    gain = rng.uniform(0.85, 1.15)
    noise_scale = rng.uniform(0.0, 0.003)
    lowpass = rng.random() < 0.5
    previous = 0.0
    augmented = []
    for value in samples:
        changed = gain * value + rng.uniform(-noise_scale, noise_scale)
        if lowpass:
            previous = 0.8 * changed + 0.2 * previous
            changed = previous
        augmented.append(max(-1.0, min(1.0, changed)))
    return augmented


def _create_head(torch, hidden_size):
    class AudioHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = torch.nn.Dropout(0.1)
            self.classifier = torch.nn.Linear(hidden_size, len(EXPRESSION_LABELS))

        def forward(self, features, attention_mask=None):
            if attention_mask is None:
                pooled = features.mean(dim=1)
            else:
                weights = attention_mask.unsqueeze(-1).to(features.dtype)
                pooled = (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
            return self.classifier(self.dropout(pooled))

    return AudioHead()


def load_audio_expert(revision, head_path=None, device=None):
    """Load a pinned frozen speech encoder; an untrained head is for fitting only."""
    require_commit_sha(revision)
    try:
        import torch
        from transformers import AutoFeatureExtractor, AutoModel
    except ImportError as exc:
        raise RuntimeError("Audio expert requires torch and transformers") from exc
    chosen_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        encoder = AutoModel.from_pretrained(MODEL_ID, revision=revision).to(chosen_device).eval()
        extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID, revision=revision)
    except Exception as exc:
        raise RuntimeError("Unable to load pinned WavLM checkpoint; verify revision and model access") from exc
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    head = _create_head(torch, encoder.config.hidden_size).to(chosen_device)
    if head_path:
        metadata_path = Path(head_path) / "audio_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["model_id"] != MODEL_ID or metadata["revision"] != revision:
            raise ValueError("audio head metadata does not match requested checkpoint")
        try:
            state = torch.load(Path(head_path) / "audio_head.pt", map_location=chosen_device, weights_only=True)
        except TypeError as exc:
            raise RuntimeError("torch.load(weights_only=True) requires a recent PyTorch release") from exc
        head.load_state_dict(state)
        head.eval()
    return encoder, head, extractor, chosen_device


def _forward_logits(encoder, head, extractor, samples, device):
    import torch

    inputs = extractor(samples, sampling_rate=16000, return_tensors="pt", padding=True,
                       return_attention_mask=True)
    input_values = inputs["input_values"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    with torch.no_grad():
        features = encoder(input_values=input_values, attention_mask=attention_mask).last_hidden_state
    if attention_mask is not None and hasattr(encoder, "_get_feature_vector_attention_mask"):
        pooled_mask = encoder._get_feature_vector_attention_mask(features.shape[1], attention_mask)
    else:
        pooled_mask = None
    return head(features, pooled_mask)


def train_audio_head(manifest_path, output_dir, revision, epochs=1, max_steps=None,
                     learning_rate=1e-3, device=None, seed=42, augment=False):
    """Train the small head on VoiceVote while the WavLM encoder stays frozen."""
    import torch

    require_commit_sha(revision)
    records = [row for row in read_manifest(manifest_path)
               if row["split"] == "train" and row["voice_vote"] in EXPRESSION_LABELS]
    if not records:
        raise ValueError("manifest contains no unambiguous VoiceVote training clips")
    if epochs < 1 or (max_steps is not None and max_steps < 1):
        raise ValueError("epochs and max_steps must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("audio head output directory is not empty: %s" % output)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    encoder, head, extractor, chosen_device = load_audio_expert(revision, device=device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate)
    head.train()
    steps = 0
    final_loss = None
    used_clips = set()
    reference_record = None
    for _ in range(epochs):
        epoch_records = list(records)
        rng.shuffle(epoch_records)
        for record in epoch_records:
            samples = read_pcm16_wav(resolve_media_path(manifest_path, record["audio_path"]))
            if waveform_quality(samples) == 0:
                continue
            if augment:
                samples = augment_waveform(samples, rng)
            target = torch.tensor([EXPRESSION_LABELS.index(record["voice_vote"])],
                                  dtype=torch.long, device=chosen_device)
            optimizer.zero_grad(set_to_none=True)
            logits = _forward_logits(encoder, head, extractor, samples, chosen_device)
            loss = torch.nn.functional.cross_entropy(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError("audio head loss is nonfinite")
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach())
            steps += 1
            used_clips.add(record["sample_id"])
            if reference_record is None:
                reference_record = record
            if max_steps is not None and steps >= max_steps:
                break
        if max_steps is not None and steps >= max_steps:
            break
    if steps == 0:
        raise ValueError("all VoiceVote training clips failed audio quality checks")
    output.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), output / "audio_head.pt")
    # Verify that the saved classifier, rather than the in-memory head, can
    # reproduce logits on one training waveform with the same frozen encoder.
    sample = read_pcm16_wav(resolve_media_path(manifest_path, reference_record["audio_path"]))
    head.eval()
    with torch.no_grad():
        expected = _forward_logits(encoder, head, extractor, sample, chosen_device)
    reloaded = _create_head(torch, encoder.config.hidden_size).to(chosen_device)
    try:
        saved_state = torch.load(output / "audio_head.pt", map_location=chosen_device, weights_only=True)
    except TypeError as exc:
        raise RuntimeError("torch.load(weights_only=True) requires a recent PyTorch release") from exc
    reloaded.load_state_dict(saved_state)
    reloaded.eval()
    with torch.no_grad():
        actual = _forward_logits(encoder, reloaded, extractor, sample, chosen_device)
    if not torch.allclose(expected, actual, atol=1e-5, rtol=1e-5):
        raise RuntimeError("saved audio head did not reproduce logits after reload")
    (output / "audio_metadata.json").write_text(json.dumps({
        "model_id": MODEL_ID, "revision": revision, "target": "VoiceVote",
        "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
        "train_actors": sorted({str(row["actor_id"]) for row in records}),
        "eligible_training_clips": len(records), "training_clips": len(used_clips),
        "steps": steps, "epochs": epochs, "max_steps": max_steps,
        "seed": seed, "augmentation": "gain_noise_bandwidth_v1" if augment else "none",
        "final_loss": final_loss,
        "labels": EXPRESSION_LABELS, "head_in_process_reload_verified": True,
        "head_subprocess_reload_verified": False,
        "reload_reference": {"sample_id": reference_record["sample_id"],
                             "audio_path": reference_record["audio_path"],
                             "logits": expected.squeeze(0).tolist()},
    }, indent=2) + "\n", encoding="utf-8")
    return output


def verify_audio_head(manifest_path, output_dir, revision, tolerance=1e-4, device=None):
    """Reload frozen encoder and saved head in a new process and compare logits."""
    require_commit_sha(revision)
    output = Path(output_dir)
    metadata_path = output / "audio_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("model_id") != MODEL_ID or metadata.get("revision") != revision:
        raise ValueError("audio head metadata does not match requested checkpoint")
    if hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest() != metadata.get("manifest_sha256"):
        raise ValueError("manifest changed since audio head training")
    import torch

    encoder, head, extractor, chosen_device = load_audio_expert(revision, output, device)
    reference = metadata["reload_reference"]
    samples = read_pcm16_wav(resolve_media_path(manifest_path, reference["audio_path"]))
    with torch.no_grad():
        actual = _forward_logits(encoder, head, extractor, samples, chosen_device).squeeze(0).tolist()
    expected = reference["logits"]
    if len(expected) != len(EXPRESSION_LABELS) or len(actual) != len(expected):
        raise ValueError("audio reload reference has the wrong label count")
    maximum_difference = max(abs(got - want) for got, want in zip(actual, expected))
    if maximum_difference > tolerance:
        raise RuntimeError("audio head reload changed logits by %.6f" % maximum_difference)
    verification = {"reload_verified": True, "max_logit_difference": maximum_difference,
                    "tolerance": tolerance, "model_id": MODEL_ID, "revision": revision,
                    "manifest_sha256": metadata["manifest_sha256"]}
    (output / "reload_verification.json").write_text(json.dumps(verification, indent=2) + "\n",
                                                     encoding="utf-8")
    metadata["head_subprocess_reload_verified"] = True
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return verification


def predict_audio_clip(encoder, head, extractor, device, wav_path):
    import torch

    samples = read_pcm16_wav(wav_path)
    quality = waveform_quality(samples)
    if quality == 0:
        return {"scores": None, "audio_quality": 0.0, "calibrated": False}
    head.eval()
    with torch.no_grad():
        logits = _forward_logits(encoder, head, extractor, samples, device)
        probabilities = torch.softmax(logits, dim=-1).squeeze(0).tolist()
    return {"scores": dict(zip(EXPRESSION_LABELS, probabilities)),
            "audio_quality": quality, "calibrated": False}
