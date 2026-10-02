"""Train, reload and score the U0 joint model using paired CREMA-D clips."""

import hashlib
import json
import math
import random
import time
import sys
from pathlib import Path

from .audio import MODEL_ID as AUDIO_ID, read_pcm16_wav, augment_waveform, waveform_quality
from .visual import MODEL_ID as VISUAL_ID, augment_training_image
from .labels import EXPRESSION_LABELS
from .manifest import read_manifest, verified_media
from .revision import require_commit_sha
from .unified_config import UnifiedConfig, clip_quality


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def joint_training_records(records):
    return [row for row in records if row["split"] == "train"
            and row.get("multimodal_vote") in EXPRESSION_LABELS and clip_quality(row)["accepted"]]


def checkpoint_metadata(path, visual_revision=None, audio_revision=None):
    root = Path(path)
    metadata = json.loads((root / "unified_metadata.json").read_text(encoding="utf-8"))
    if (metadata.get("architecture") != "nemotron_wavlm_joint_v1"
            or metadata.get("visual_model_id") != VISUAL_ID or metadata.get("audio_model_id") != AUDIO_ID
            or metadata.get("target") != "MultiModalVote"):
        raise ValueError("not a unified Nemotron/WavLM checkpoint")
    for key, revision in (("visual_revision", visual_revision), ("audio_revision", audio_revision)):
        require_commit_sha(metadata[key])
        if revision is not None and revision != metadata[key]:
            raise ValueError("unified checkpoint revision mismatch: " + key)
    UnifiedConfig(**metadata["config"]).validate()
    artifacts = metadata.get("artifact_sha256s", {})
    if not {"audio_projector.pt", "decoder_lora/adapter_config.json", "decoder_lora/adapter_model.safetensors"} <= set(artifacts):
        raise ValueError("unified checkpoint is incomplete")
    for relative, expected in artifacts.items():
        target = (root / relative).resolve()
        if root.resolve() not in target.parents or file_hash(target) != expected:
            raise ValueError("unified checkpoint artifact hash mismatch: " + relative)
    return metadata


def checkpoint_identity(metadata):
    stable = {key: metadata[key] for key in ("architecture", "visual_revision", "audio_revision",
                                           "config", "artifact_sha256s", "manifest_sha256")}
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


def load_unified(visual_revision, audio_revision, config=None, checkpoint=None, training=False):
    from importlib.metadata import version
    for package, required in (("transformers", "4.57.3"), ("peft", "0.18.0")):
        if version(package) != required:
            raise RuntimeError("Joint model requires %s==%s; install the repository's perception dependencies in %s"
                               % (package, required, sys.executable))
    import torch
    from peft import PeftModel
    from transformers import AutoModel, AutoFeatureExtractor
    from .visual import load_visual_expert, attach_decoder_lora
    from .unified_model import AudioVisualModel

    require_commit_sha(visual_revision)
    require_commit_sha(audio_revision)
    if not torch.cuda.is_available():
        raise RuntimeError("joint Nemotron training/inference requires a CUDA GPU")
    metadata = checkpoint_metadata(checkpoint, visual_revision, audio_revision) if checkpoint else None
    if metadata:
        saved = UnifiedConfig(**metadata["config"]).validate()
        if config is not None and config != saved:
            raise ValueError("configuration differs from saved unified checkpoint")
        config = saved
    config = (config or UnifiedConfig()).validate()
    visual, tokenizer, processor = load_visual_expert(visual_revision)
    visual.requires_grad_(False)
    if checkpoint:
        visual.language_model = PeftModel.from_pretrained(
            visual.language_model, Path(checkpoint) / "decoder_lora", is_trainable=training)
    elif training:
        attach_decoder_lora(visual, config.lora_rank, config.lora_alpha)
    else:
        raise ValueError("joint inference requires a trained audio projector and decoder adapter")
    audio = AutoModel.from_pretrained(AUDIO_ID, revision=audio_revision).to("cuda").eval()
    extractor = AutoFeatureExtractor.from_pretrained(AUDIO_ID, revision=audio_revision)
    model = AudioVisualModel(visual, audio, tokenizer, processor, extractor, config)
    model.checkpoint_id = checkpoint_identity(metadata) if metadata else None
    model.manifest_sha256 = metadata["manifest_sha256"] if metadata else None
    model.reload_verified = bool(metadata and metadata.get("reload_verified"))
    if checkpoint:
        state = torch.load(Path(checkpoint) / "audio_projector.pt", map_location="cpu", weights_only=True)
        model.audio_projector.load_state_dict(state, strict=True)
    if training and config.gradient_checkpointing:
        visual.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    visual.language_model.config.use_cache = False
    model.train(training)
    return model


def encode_record(model, manifest, record, rng=None):
    from PIL import Image
    frames, audio = verified_media(manifest, record)
    images = []
    for path in frames:
        with Image.open(path) as source:
            image = source.convert("RGB")
        images.append(augment_training_image(image, rng) if rng else image)
    samples = read_pcm16_wav(audio)
    if waveform_quality(samples) == 0:
        raise ValueError("paired waveform is silent or severely clipped")
    if rng:
        samples = augment_waveform(samples, rng)
    return model.encode(images, samples)


def predict_unified_record(model, manifest, record, ablations=False):
    quality = clip_quality(record)
    result = {"scores": None, "calibrated": False, "model_kind": "unified_audio_visual",
              "visual_quality": quality["visual_quality"], "audio_quality": quality["audio_quality"]}
    if not quality["accepted"]:
        return {**result, "reason": "insufficient_paired_quality"}
    model.eval()
    features = encode_record(model, manifest, record)
    result["scores"] = model.scores(*features, face_mask=quality["faces"])
    if ablations:
        for modality in ("audio", "video"):
            result[modality + "_ablated_scores"] = model.scores(
                *features, face_mask=quality["faces"], ablation=modality)
    return result


def require_unified_pilot(pilot, manifest, visual_revision, audio_revision, config):
    metadata = checkpoint_metadata(pilot, visual_revision, audio_revision)
    if (metadata["manifest_sha256"] != file_hash(manifest) or metadata["config"] != config.to_dict()
            or not metadata.get("reload_verified") or not metadata.get("modality_wiring_passed")
            or not metadata.get("gradient_check_passed")):
        raise ValueError("full joint training requires a matching verified unified pilot")
    return metadata


def train_unified(manifest, output, visual_revision, audio_revision, config=None, max_steps=1,
                  epochs=1, gradient_accumulation=1, learning_rate=2e-4, seed=42,
                  augment=False, log_every=10, pilot=None, baseline_evaluation=None,
                  resume=None, save_every=100):
    import torch
    import numpy as np
    from .unified_checkpoint import (read_resume_metadata, training_windows, validate_resume_plan,
                                     save_training_checkpoint, restore_training_state,
                                     save_model_weights, schedule_digest)

    resume_root, resume_metadata = read_resume_metadata(resume) if resume else (None, None)
    if resume_metadata:
        evidence = resume_metadata.get("training_evidence", {})
        pilot = pilot or evidence.get("pilot")
        baseline_evaluation = baseline_evaluation or evidence.get("baseline_evaluation")
        if config is None:
            config = UnifiedConfig(**resume_metadata["config"])
    config = (config or UnifiedConfig()).validate()
    require_commit_sha(visual_revision)
    require_commit_sha(audio_revision)
    if (epochs < 1 or gradient_accumulation < 1 or log_every < 1 or save_every < 1
            or (max_steps is not None and max_steps < 1)
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("invalid training schedule or learning rate")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("unified output directory must be empty; resume into a new output/run root: %s" % output)
    if max_steps != 1:
        if not pilot or not baseline_evaluation:
            raise ValueError("full joint training requires --pilot and --baseline-evaluation")
        require_unified_pilot(pilot, manifest, visual_revision, audio_revision, config)
        baselines = json.loads(Path(baseline_evaluation).read_text(encoding="utf-8"))
        if not {"visual_face_vote", "audio_voice_vote", "fusion_multimodal_vote"} <= set(baselines):
            raise ValueError("baseline evaluation must contain visual/audio/fusion validation metrics")
        provenance = baselines.get("provenance", {})
        validation_ids = sorted(row["sample_id"] for row in read_manifest(manifest) if row["split"] == "validation")
        if (provenance.get("selection_split") != "validation"
                or provenance.get("manifest_sha256") != file_hash(manifest)
                or provenance.get("sample_ids") != validation_ids):
            raise ValueError("baseline validation metrics must match the prepared manifest and validation clips")
    records = joint_training_records(read_manifest(manifest))
    if not records:
        raise ValueError("no usable paired training clips with unambiguous MultiModalVote")
    augmentation_rng = random.Random(seed + 1)
    windows = training_windows(records, epochs, seed, gradient_accumulation, max_steps)
    settings = {"manifest_sha256": file_hash(manifest), "visual_revision": visual_revision,
                "audio_revision": audio_revision, "config": config.to_dict(), "seed": seed,
                "gradient_accumulation": gradient_accumulation, "learning_rate": learning_rate,
                "augment": augment, "eligible_sample_ids_sha256": schedule_digest(records),
                "baseline_evaluation_sha256": file_hash(baseline_evaluation) if baseline_evaluation else None}
    initial_step = validate_resume_plan(resume_metadata, settings, windows) if resume_metadata else 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("Loading joint Nemotron/WavLM model; optimizer steps %d -> %d" % (initial_step, len(windows)), flush=True)
    model = load_unified(visual_revision, audio_revision, config, checkpoint=resume_root, training=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    output.mkdir(parents=True, exist_ok=True)
    total = len(windows)
    losses, durations, reports = [], [], []
    peak_vram = 0
    if resume_metadata:
        history = restore_training_state(resume_root, model, optimizer, augmentation_rng, resume_metadata)
        losses, durations, reports = history["losses"], history["step_seconds"], history["gradient_reports"]
        peak_vram = history["peak_vram_bytes"]
        print("Restored optimizer, RNG, history and next batch from %s" % resume_root, flush=True)
    base_metadata = {
        "architecture": "nemotron_wavlm_joint_v1", "target": "MultiModalVote", "config": config.to_dict(),
        "visual_model_id": VISUAL_ID, "visual_revision": visual_revision,
        "audio_model_id": AUDIO_ID, "audio_revision": audio_revision,
        "manifest_sha256": settings["manifest_sha256"], "training_settings": settings,
        "training_evidence": {"pilot": str(Path(pilot).resolve()) if pilot else None,
                              "baseline_evaluation": str(Path(baseline_evaluation).resolve()) if baseline_evaluation else None},
        "seed": seed, "epochs": epochs, "gradient_accumulation": gradient_accumulation,
        "learning_rate": learning_rate, "gradient_check_passed": True,
        "augmentation": "framing_mirror_photometric_gain_noise_bandwidth_v1" if augment else "none",
    }
    consumed = [row for window in windows[:initial_step] for row in window["rows"]]
    torch.cuda.reset_peak_memory_stats()
    for position in range(initial_step, total):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        item = windows[position]
        window = item["rows"]
        torch.cuda.synchronize()
        started = time.perf_counter()
        window_loss = 0.0
        for row in window:
            features = encode_record(model, manifest, row, augmentation_rng if augment else None)
            loss = model(*features, row["multimodal_vote"], clip_quality(row)["faces"]).loss
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("joint decoder loss is missing or nonfinite")
            (loss / len(window)).backward()
            window_loss += float(loss.detach()) / len(window)
        report = model.gradient_report()
        if (not report["frozen_encoders_have_no_gradients"]
                or not all(report[key]["nonzero"] for key in ("audio_projector", "decoder_lora"))):
            raise RuntimeError("joint gradient check failed: %s" % report)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)
        losses.append(window_loss)
        consumed.extend(window)
        peak_vram = max(peak_vram, torch.cuda.max_memory_allocated())
        if not reports:
            reports.append(report)
        step = len(losses)
        if step == 1 or step % log_every == 0 or step == total:
            print("Joint step %d/%d loss=%.6f seconds=%.2f peak_vram_gib=%.2f" % (
                step, total, window_loss, durations[-1], torch.cuda.max_memory_allocated() / 2**30), flush=True)
        progress = {"optimizer_steps": step, "next_epoch": item["next_epoch"],
                    "next_clip_index": item["next_clip_index"], "completed_microsteps": len(consumed),
                    "schedule_sha256": schedule_digest(consumed)}
        if step % save_every == 0 or item["next_clip_index"] == 0 or step == total:
            path = save_training_checkpoint(output, model, optimizer, base_metadata, progress,
                                             {"losses": losses, "step_seconds": durations,
                                              "gradient_reports": reports, "peak_vram_bytes": peak_vram}, augmentation_rng)
            print("Saved resumable checkpoint: %s" % path, flush=True)
    model.eval()
    reference = consumed[0]
    features = encode_record(model, manifest, reference)
    scores = model.scores(*features, face_mask=clip_quality(reference)["faces"])
    del optimizer
    ablations = {}
    for modality in ("audio", "video"):
        changed = model.scores(*features, face_mask=clip_quality(reference)["faces"], ablation=modality)
        delta = max(abs(scores[label] - changed[label]) for label in EXPRESSION_LABELS)
        ablations[modality] = {"scores": changed, "max_score_difference": delta, "affects_scores": delta > 1e-7}
    artifacts = save_model_weights(model, output)
    metadata = {
        **base_metadata, "artifact_sha256s": artifacts, "optimizer_steps": len(losses),
        "completed_epochs": progress["next_epoch"], "training_progress": progress,
        "resumed_from": str(resume_root) if resume_root else None, "save_every": save_every,
        "train_actors": sorted({str(row["actor_id"]) for row in consumed}),
        "training_clips": len({row["sample_id"] for row in consumed}),
        "schedule_sha256": schedule_digest(consumed),
        "losses": losses, "step_seconds": durations,
        "peak_vram_bytes": max(peak_vram, torch.cuda.max_memory_allocated()), "gradient_report": reports[0],
        "gradient_check_passed": True, "modality_ablations": ablations,
        "modality_wiring_passed": all(item["affects_scores"] for item in ablations.values()),
        "feature_shapes": {"visual": list(features[0].shape), "audio": list(features[1].shape)},
        "reload_reference": {"sample_id": reference["sample_id"], "scores": scores},
        "reload_verified": False,
        "baseline_evaluation_sha256": file_hash(baseline_evaluation) if baseline_evaluation else None,
        "environment": {"python": sys.version, "torch": torch.__version__,
                        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()},
    }
    (output / "unified_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if not metadata["modality_wiring_passed"]:
        raise RuntimeError("saved diagnostic checkpoint, but a modality ablation did not affect scores")
    print("Saved unified decoder LoRA, audio projector, and provenance to %s" % output, flush=True)
    return output


def verify_unified(manifest, checkpoint, visual_revision, audio_revision, tolerance=1e-3):
    metadata = checkpoint_metadata(checkpoint, visual_revision, audio_revision)
    if metadata["manifest_sha256"] != file_hash(manifest):
        raise ValueError("manifest changed since unified training")
    reference = metadata["reload_reference"]
    rows = [row for row in read_manifest(manifest) if row["sample_id"] == reference["sample_id"]]
    if len(rows) != 1 or rows[0]["split"] != "train":
        raise ValueError("invalid training reload reference")
    model = load_unified(visual_revision, audio_revision, checkpoint=checkpoint)
    actual = predict_unified_record(model, manifest, rows[0], ablations=True)
    if actual["scores"] is None:
        raise ValueError("reload reference failed paired quality")
    differences = [abs(actual["scores"][key] - reference["scores"][key]) for key in EXPRESSION_LABELS]
    for modality in ("audio", "video"):
        differences.extend(abs(actual[modality + "_ablated_scores"][key]
                               - metadata["modality_ablations"][modality]["scores"][key])
                           for key in EXPRESSION_LABELS)
    delta = max(differences)
    if not math.isfinite(delta) or delta > tolerance:
        raise RuntimeError("unified reload changed joint/ablated scores by %.6f" % delta)
    result = {"reload_verified": True, "max_score_difference": delta, "tolerance": tolerance,
              "checkpoint_id": checkpoint_identity(metadata), "manifest_sha256": metadata["manifest_sha256"]}
    root = Path(checkpoint)
    (root / "reload_verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    metadata["reload_verified"] = True
    (root / "unified_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return result
