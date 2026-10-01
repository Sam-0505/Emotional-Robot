"""Immutable Hugging Face model revision validation."""

import re


def require_commit_sha(revision):
    """Reject moving refs such as main, tags and short hashes for model loads."""
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("model revision must be a full 40-character lowercase commit SHA")
    return revision
