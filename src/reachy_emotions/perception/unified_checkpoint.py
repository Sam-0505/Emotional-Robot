"""Immutable optimizer-boundary checkpoints for continued joint training."""

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path


def resolve_resume(path):
    """Accept a checkpoint folder or a training output's last-checkpoint pointer."""
    root = Path(path).resolve()
    if (root / "last_checkpoint.json").is_file():
        relative = json.loads((root / "last_checkpoint.json").read_text())["checkpoint"]
        target = (root / relative).resolve()
        if root not in target.parents:
            raise ValueError("last-checkpoint pointer escapes training output")
        root = target
    return root


def read_resume_metadata(path):
    from .unified import checkpoint_metadata, file_hash
    root = resolve_resume(path)
    metadata = checkpoint_metadata(root)
    if metadata.get("training_checkpoint_version") != 1:
        raise ValueError("checkpoint has no resumable optimizer state; use a checkpoint from the new trainer")
    if file_hash(root / "training_state.pt") != metadata.get("training_state_sha256"):
        raise ValueError("training-state hash mismatch")
    return root, metadata


def save_model_weights(model, root):
    import torch
    from .unified import file_hash
    root = Path(root)
    model.visual.language_model.save_pretrained(root / "decoder_lora", safe_serialization=True)
    torch.save({key: value.detach().cpu() for key, value in model.audio_projector.state_dict().items()},
               root / "audio_projector.pt")
    artifacts = {str(path.relative_to(root)): file_hash(path)
                 for path in sorted((root / "decoder_lora").iterdir()) if path.is_file()}
    artifacts["audio_projector.pt"] = file_hash(root / "audio_projector.pt")
    return artifacts


def parameter_names(model):
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def capture_random_states(augmentation_rng):
    import numpy as np
    import torch
    numpy_state = np.random.get_state()
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python": random.getstate(), "augmentation": augmentation_rng.getstate(),
            "numpy": {"kind": numpy_state[0], "keys": numpy_state[1].tolist(),
                      "position": numpy_state[2], "has_gauss": numpy_state[3], "cached_gaussian": numpy_state[4]}}


def restore_random_states(state, augmentation_rng):
    import numpy as np
    import torch
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(state["cuda"]) != count:
        raise ValueError("resume requires the same number of CUDA devices as the saved run")
    torch.set_rng_state(state["torch"])
    if count:
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    augmentation_rng.setstate(state["augmentation"])
    saved = state["numpy"]
    np.random.set_state((saved["kind"], np.asarray(saved["keys"], dtype=np.uint32),
                         saved["position"], saved["has_gauss"], saved["cached_gaussian"]))


def schedule_digest(rows):
    return hashlib.sha256("\n".join(row["sample_id"] for row in rows).encode()).hexdigest()


def training_windows(records, epochs, seed, accumulation, max_steps=None):
    """Deterministic shuffles with the last accumulation window flushed each epoch."""
    rng = random.Random(seed)
    windows = []
    for epoch in range(epochs):
        order = list(records)
        rng.shuffle(order)
        for start in range(0, len(order), accumulation):
            end = min(start + accumulation, len(order))
            windows.append({"epoch": epoch, "rows": order[start:end],
                            "next_epoch": epoch + 1 if end == len(order) else epoch,
                            "next_clip_index": 0 if end == len(order) else end})
            if max_steps is not None and len(windows) >= max_steps:
                return windows
    return windows


def validate_resume_plan(metadata, settings, windows):
    if metadata["training_settings"] != settings:
        changed = sorted(key for key in set(metadata["training_settings"]) | set(settings)
                         if metadata["training_settings"].get(key) != settings.get(key))
        raise ValueError("resume training settings mismatch: " + ", ".join(changed))
    progress = metadata["progress"]
    step = progress["optimizer_steps"]
    if not isinstance(step, int) or step < 1 or step >= len(windows):
        raise ValueError("resume has no remaining steps; increase total --epochs or --max-steps")
    consumed = [row for window in windows[:step] for row in window["rows"]]
    previous = windows[step - 1]
    if (progress["schedule_sha256"] != schedule_digest(consumed)
            or progress["completed_microsteps"] != len(consumed)
            or progress["next_epoch"] != previous["next_epoch"]
            or progress["next_clip_index"] != previous["next_clip_index"]):
        raise ValueError("resume training position does not match the deterministic schedule")
    return step


def save_training_checkpoint(output, model, optimizer, metadata, progress, history, augmentation_rng):
    """Publish only complete snapshots; never modify an older checkpoint folder."""
    import torch
    from .unified import file_hash
    output = Path(output)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    name = "step-%08d" % progress["optimizer_steps"]
    target = checkpoints / name
    if target.exists():
        raise FileExistsError("checkpoint already exists: %s" % target)
    # An interrupted temporary snapshot is not referenced by last_checkpoint.json.
    temporary = Path(tempfile.mkdtemp(prefix="." + name + "-", dir=checkpoints))
    state = {"optimizer": optimizer.state_dict(), "parameter_names": parameter_names(model),
             "random_states": capture_random_states(augmentation_rng), "progress": progress,
             "history": history, "torch_version": str(torch.__version__)}
    artifacts = save_model_weights(model, temporary)
    torch.save(state, temporary / "training_state.pt")
    saved = {**metadata, "artifact_sha256s": artifacts, "training_checkpoint_version": 1,
             "training_state_sha256": file_hash(temporary / "training_state.pt"),
             "progress": progress, "optimizer_steps": progress["optimizer_steps"],
             "reload_verified": False}
    (temporary / "unified_metadata.json").write_text(json.dumps(saved, indent=2) + "\n")
    os.replace(temporary, target)
    descriptor, pointer = tempfile.mkstemp(prefix=".last-checkpoint-", suffix=".json", dir=output)
    with os.fdopen(descriptor, "w") as stream:
        json.dump({"checkpoint": str(target.relative_to(output)),
                   "optimizer_steps": progress["optimizer_steps"]}, stream)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pointer, output / "last_checkpoint.json")
    return target


def restore_training_state(path, model, optimizer, augmentation_rng, metadata):
    import torch
    root, checked = read_resume_metadata(path)
    if checked != metadata:
        raise ValueError("resume checkpoint changed during loading")
    state = torch.load(root / "training_state.pt", map_location="cpu", weights_only=True)
    if state["torch_version"] != str(torch.__version__):
        raise ValueError("resume requires the saved PyTorch version: " + state["torch_version"])
    if state["parameter_names"] != parameter_names(model):
        raise ValueError("resume trainable parameter order changed")
    if state["progress"] != metadata["progress"]:
        raise ValueError("resume progress disagrees with checkpoint metadata")
    history = state["history"]
    step = metadata["progress"]["optimizer_steps"]
    if len(history["losses"]) != step or len(history["step_seconds"]) != step or not history["gradient_reports"]:
        raise ValueError("resume metric history does not match optimizer step")
    optimizer.load_state_dict(state["optimizer"])
    # Model/optimizer construction can consume RNG; restore last, before the next batch.
    restore_random_states(state["random_states"], augmentation_rng)
    return history
