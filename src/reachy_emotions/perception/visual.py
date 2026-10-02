"""Nemotron Nano VL visual expert and decoder-only LoRA feasibility path.

The pinned model revision must be tested on a CUDA host before a full run. The
checkpoint uses custom remote code and a custom image token layout.
"""

import json
import math
import hashlib
import random
from pathlib import Path

from .fusion import aggregate_frames
from .labels import EXPRESSION_LABELS
from .manifest import resolve_media_path
from .revision import require_commit_sha

MODEL_ID = "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1"
QUESTION = "<image>\nWhich facial expression is presented? Reply with JSON using only presented_expression and one of ANG, DIS, FEA, HAP, NEU, SAD."


def discover_decoder_targets(model):
    """Inspect actual decoder modules before selecting LoRA projections."""
    language_model = getattr(model, "language_model", None)
    if language_model is None:
        raise RuntimeError("Pinned Nemotron model has no language_model attribute; inspect its remote code before training")
    leaf_names = {name.rsplit(".", 1)[-1] for name, _ in language_model.named_modules()}
    targets = [name for name in ("q_proj", "k_proj", "v_proj", "o_proj") if name in leaf_names]
    if not targets:
        raise RuntimeError("No decoder attention projections found; inspect model.named_modules() before attaching LoRA")
    return targets


def load_visual_expert(revision, adapter_path=None, training=False):
    """Load a pinned NVIDIA checkpoint; dependencies are imported only here."""
    require_commit_sha(revision)
    if adapter_path:
        metadata_path = Path(adapter_path) / "training_metadata.json"
        if not metadata_path.is_file():
            raise ValueError("visual adapter is missing training_metadata.json provenance")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("model_id") != MODEL_ID or metadata.get("revision") != revision:
            raise ValueError("visual adapter metadata does not match requested base checkpoint")
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
        from PIL import Image  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Visual training requires torch, transformers, Pillow and PEFT (for adapters)") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("The Nemotron visual expert requires a CUDA GPU for this implementation")
    try:
        model = AutoModel.from_pretrained(MODEL_ID, revision=revision, trust_remote_code=True,
                                          torch_dtype=torch.bfloat16).cuda()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
        processor = AutoImageProcessor.from_pretrained(MODEL_ID, revision=revision,
                                                       trust_remote_code=True, device="cuda")
    except Exception as exc:
        raise RuntimeError("Unable to load pinned Nemotron VL checkpoint; verify revision, model access, GPU memory and Transformers version") from exc
    if adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("PEFT is required to load a visual adapter") from exc
        model.language_model = PeftModel.from_pretrained(model.language_model, adapter_path,
                                                         is_trainable=training)
    if not training:
        model.eval()
    return model, tokenizer, processor


def attach_decoder_lora(model, rank=8, alpha=16):
    """Keep vision and projector frozen; wrap only the language decoder."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError("PEFT is required for Nemotron decoder LoRA") from exc
    targets = discover_decoder_targets(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    config = LoraConfig(r=rank, lora_alpha=alpha, target_modules=targets,
                        lora_dropout=0.05, bias="none")
    model.language_model = get_peft_model(model.language_model, config)
    model.train()
    model.vision_model.eval()
    model.mlp1.eval()
    return targets


def augment_training_image(image, rng):
    """Mild training-only camera/framing variation on a padded face crop."""
    from PIL import ImageEnhance, ImageOps

    width, height = image.size
    shift_x = int(width * rng.uniform(0.0, 0.04))
    shift_y = int(height * rng.uniform(0.0, 0.04))
    if width - 2 * shift_x > 0 and height - 2 * shift_y > 0:
        image = image.crop((shift_x, shift_y, width - shift_x, height - shift_y)).resize((width, height))
    if rng.random() < 0.5:
        image = ImageOps.mirror(image)
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.85, 1.15))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.85, 1.15))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.9, 1.1))
    return image


def _image_and_tokens(model, tokenizer, processor, image_path, answer, image_rng=None):
    import torch
    from PIL import Image

    if not hasattr(model, "_format_image_token"):
        raise RuntimeError("Pinned Nemotron revision changed image token formatting; inspect model.chat before training")
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        if image_rng is not None:
            image = augment_training_image(image, image_rng)
        features = processor([image])
    pixel_values = features["pixel_values"].to(device="cuda", dtype=torch.bfloat16)
    num_patches = features["num_patches"]
    if len(num_patches) != 1:
        raise RuntimeError("expected one image in visual sample")
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": QUESTION}],
                                           tokenize=False, add_generation_prompt=True)
    prompt = model._format_image_token(prompt, num_patches, "<image>")
    completion = json.dumps({"presented_expression": answer}, separators=(",", ":")) + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise RuntimeError("tokenizer changed the prompt token boundary; inspect chat template")
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<image>")
    if model.img_context_token_id is None:
        raise RuntimeError("Nemotron image context token was not found")
    input_ids = torch.tensor([full_ids], dtype=torch.long, device="cuda")
    labels = torch.tensor([[-100] * len(prompt_ids) + full_ids[len(prompt_ids):]],
                          dtype=torch.long, device="cuda")
    image_flags = torch.ones((pixel_values.shape[0], 1), dtype=torch.long, device="cuda")
    return {"pixel_values": pixel_values, "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids), "image_flags": image_flags,
            "labels": labels}


def _ensure_single_rank_group():
    """NVIDIA's custom forward calls torch.distributed.get_rank unconditionally."""
    import os
    import tempfile
    import torch

    if torch.distributed.is_initialized():
        return None
    descriptor, path = tempfile.mkstemp(prefix="nemotron-rank-")
    os.close(descriptor)
    torch.distributed.init_process_group(backend="nccl", init_method="file://" + path,
                                         rank=0, world_size=1)
    return path


def training_schedule(records, seed=42, epochs=1, max_steps=None, gradient_accumulation=1):
    """Seeded actor-safe clip order with one detected frame per clip per epoch.

    A training record represents one clip. Sampling a frame does not duplicate
    a clip across actor splits or turn frames into separate evaluation examples.
    """
    if epochs < 1 or gradient_accumulation < 1 or (max_steps is not None and max_steps < 1):
        raise ValueError("epochs, max_steps and gradient_accumulation must be positive")
    eligible = []
    for row in records:
        if row.get("split") != "train":
            raise ValueError("visual training schedule received a non-training clip")
        if row.get("face_vote") not in EXPRESSION_LABELS:
            continue
        quality = row.get("frame_quality", [])
        if len(row.get("frame_paths", [])) != 3 or len(quality) != 3:
            raise ValueError("each visual training clip needs three frames and quality records")
        valid = [index for index, item in enumerate(quality) if item.get("face_status") == "detected"]
        if valid:
            eligible.append((row, valid))
    if not eligible:
        raise ValueError("no unambiguous FaceVote clips with detected face crops")
    rng = random.Random(seed)
    schedule = []
    for _ in range(epochs):
        epoch_rows = list(eligible)
        rng.shuffle(epoch_rows)
        schedule.extend((row, rng.choice(valid)) for row, valid in epoch_rows)
    if max_steps is not None:
        schedule = schedule[:max_steps * gradient_accumulation]
    return schedule


def select_visual_training_records(records):
    """Use only the training split from a complete prepared manifest."""
    return [row for row in records if row.get("split") == "train"]


def _score_visual_frame(model, tokenizer, processor, frame_path):
    """Real model likelihoods for six fixed answers; no label-derived scores."""
    losses = {}
    for label in EXPRESSION_LABELS:
        inputs = _image_and_tokens(model, tokenizer, processor, frame_path, label)
        losses[label] = float(model(**inputs).loss)
    maximum = max(-value for value in losses.values())
    shifted = {label: math.exp(-loss - maximum) for label, loss in losses.items()}
    total = sum(shifted.values())
    return {label: value / total for label, value in shifted.items()}


def train_visual_adapter(manifest_path, output_dir, revision, max_steps=1, learning_rate=2e-4,
                         epochs=1, gradient_accumulation=1, seed=42, augment=False):
    """Train decoder LoRA using a seeded clip schedule and saved provenance."""
    import torch
    from .manifest import read_manifest

    require_commit_sha(revision)
    manifest_file = Path(manifest_path)
    records = select_visual_training_records(read_manifest(manifest_file))
    schedule = training_schedule(records, seed=seed, epochs=epochs, max_steps=max_steps,
                                 gradient_accumulation=gradient_accumulation)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("visual adapter output directory is not empty: %s" % output)
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    random.seed(seed)
    image_rng = random.Random(seed + 1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, tokenizer, processor = load_visual_expert(revision, training=True)
    targets = attach_decoder_lora(model)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                                  lr=learning_rate)
    rank_file = _ensure_single_rank_group()
    try:
        optimizer_steps = 0
        losses = []
        for start in range(0, len(schedule), gradient_accumulation):
            window = schedule[start:start + gradient_accumulation]
            optimizer.zero_grad(set_to_none=True)
            window_losses = []
            for record, frame_index in window:
                frame = record["frame_paths"][frame_index]
                inputs = _image_and_tokens(model, tokenizer, processor,
                                           resolve_media_path(manifest_file, frame), record["face_vote"],
                                           image_rng if augment else None)
                try:
                    loss = model(**inputs).loss
                    if loss is None or not torch.isfinite(loss):
                        raise RuntimeError("visual loss is missing or nonfinite")
                    (loss / len(window)).backward()
                    window_losses.append(float(loss.detach()))
                except Exception as exc:
                    raise RuntimeError("Nemotron forward/backward gate failed; inspect pinned remote code, token layout and GPU memory") from exc
            optimizer.step()
            optimizer_steps += 1
            losses.append(sum(window_losses) / len(window_losses))
        reference_record, reference_index = schedule[0]
        reference_frame = reference_record["frame_paths"][reference_index]
        model.eval()
        with torch.no_grad():
            reference_scores = _score_visual_frame(model, tokenizer, processor,
                                                   resolve_media_path(manifest_file, reference_frame))
        output.mkdir(parents=True, exist_ok=True)
        model.language_model.save_pretrained(output)
        order = "\n".join("%s:%d" % (row["sample_id"], index) for row, index in schedule)
        (output / "training_metadata.json").write_text(json.dumps({
            "model_id": MODEL_ID, "revision": revision, "max_steps": max_steps,
            "epochs": epochs, "gradient_accumulation": gradient_accumulation,
            "seed": seed, "learning_rate": learning_rate,
            "augmentation": "framing_mirror_photometric_v1" if augment else "none",
            "microsteps": len(schedule), "optimizer_steps": optimizer_steps,
            "mean_training_loss": sum(losses) / len(losses),
            "schedule_sha256": hashlib.sha256(order.encode("utf-8")).hexdigest(),
            "manifest_sha256": hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
            "train_actors": sorted({str(row["actor_id"]) for row, _ in schedule}),
            "target_modules": targets, "visual_target": "FaceVote",
            "training_clips": len({row["sample_id"] for row, _ in schedule}),
            "reload_reference": {"sample_id": reference_record["sample_id"],
                                 "frame_path": reference_frame, "scores": reference_scores},
            "reload_verified": False,
        }, indent=2) + "\n", encoding="utf-8")
        return output
    finally:
        if rank_file is not None:
            torch.distributed.destroy_process_group()
            Path(rank_file).unlink(missing_ok=True)


def verify_visual_adapter(manifest_path, output_dir, revision, tolerance=1e-3):
    """Reload the saved adapter and reproduce six-label scores in a new process."""
    import torch

    require_commit_sha(revision)
    output = Path(output_dir)
    metadata = json.loads((output / "training_metadata.json").read_text(encoding="utf-8"))
    if metadata["model_id"] != MODEL_ID or metadata["revision"] != revision:
        raise ValueError("adapter metadata does not match requested base checkpoint")
    manifest_file = Path(manifest_path)
    if hashlib.sha256(manifest_file.read_bytes()).hexdigest() != metadata["manifest_sha256"]:
        raise ValueError("manifest changed since adapter training")
    model, tokenizer, processor = load_visual_expert(revision, adapter_path=output)
    rank_file = _ensure_single_rank_group()
    try:
        with torch.no_grad():
            actual = _score_visual_frame(model, tokenizer, processor,
                                         resolve_media_path(manifest_file, metadata["reload_reference"]["frame_path"]))
        expected = metadata["reload_reference"]["scores"]
        maximum_difference = max(abs(actual[label] - expected[label]) for label in EXPRESSION_LABELS)
        if maximum_difference > tolerance:
            raise RuntimeError("adapter reload changed label scores by %.6f" % maximum_difference)
        verification = {"reload_verified": True, "max_score_difference": maximum_difference,
                        "tolerance": tolerance, "model_id": MODEL_ID, "revision": revision,
                        "manifest_sha256": metadata["manifest_sha256"]}
        (output / "reload_verification.json").write_text(json.dumps(verification, indent=2) + "\n",
                                                          encoding="utf-8")
        metadata["reload_verified"] = True
        (output / "training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                                     encoding="utf-8")
        return verification
    finally:
        if rank_file is not None:
            torch.distributed.destroy_process_group()
            Path(rank_file).unlink(missing_ok=True)


def predict_visual_clip(model, tokenizer, processor, frame_paths, frame_qualities=None):
    """Return an uncalibrated clip distribution from three frame label likelihoods."""
    import torch

    if len(frame_paths) != 3:
        raise ValueError("three frame paths are required")
    qualities = [1.0] * 3 if frame_qualities is None else list(frame_qualities)
    if len(qualities) != 3:
        raise ValueError("three frame quality values are required")
    if all(quality == 0 for quality in qualities):
        return {"scores": None, "visual_quality": 0.0,
                "frame_scores": [None, None, None], "calibrated": False}
    rank_file = _ensure_single_rank_group()
    try:
        model.eval()
        frame_scores = []
        with torch.no_grad():
            for frame_path, quality in zip(frame_paths, qualities):
                if quality == 0:
                    frame_scores.append(None)
                    continue
                frame_scores.append(_score_visual_frame(model, tokenizer, processor, frame_path))
        scores, clip_quality = aggregate_frames(frame_scores, qualities)
        return {"scores": scores, "visual_quality": clip_quality, "frame_scores": frame_scores,
                "calibrated": False}
    finally:
        if rank_file is not None:
            torch.distributed.destroy_process_group()
            Path(rank_file).unlink(missing_ok=True)
