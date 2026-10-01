"""Minimal manifest adapter; every record represents one synchronized clip."""

import json
from pathlib import Path

from .labels import ALL_LABELS


def read_manifest(path):
    manifest_path = Path(path)
    records = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("invalid JSON on manifest line %d" % line_number) from exc
            for field in ("sample_id", "actor_id", "split", "frame_paths", "audio_path"):
                if field not in record:
                    raise ValueError("manifest line %d missing %s" % (line_number, field))
            if len(record["frame_paths"]) != 3:
                raise ValueError("manifest line %d must have exactly three frames" % line_number)
            for field in ("face_vote", "voice_vote", "multimodal_vote"):
                if record.get(field) not in ALL_LABELS:
                    raise ValueError("manifest line %d has invalid %s" % (line_number, field))
            records.append(record)
    if not records:
        raise ValueError("manifest is empty")
    assert_actor_disjoint(records)
    return records


def assert_actor_disjoint(records):
    actor_splits = {}
    sample_ids = set()
    for record in records:
        actor = str(record["actor_id"])
        split = record["split"]
        if split not in ("train", "validation", "test"):
            raise ValueError("invalid split %r" % split)
        if actor in actor_splits and actor_splits[actor] != split:
            raise ValueError("actor %s occurs in multiple splits" % actor)
        actor_splits[actor] = split
        sample_id = record["sample_id"]
        if sample_id in sample_ids:
            raise ValueError("duplicate sample_id %s" % sample_id)
        sample_ids.add(sample_id)


def resolve_media_path(manifest_path, relative_path):
    """Resolve a manifest relative path without permitting traversal outside its root."""
    root = Path(manifest_path).resolve().parent
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("media path escapes manifest directory: %s" % relative_path) from exc
    return candidate
