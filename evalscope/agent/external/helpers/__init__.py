"""Reusable helpers for benchmark-side external-agent integrations.

Each helper is a small standalone async function that takes an
:class:`AgentEnvironment` and produces a benchmark-specific artifact
(patch text, file payload, ...). They live here so multiple adapters
can share the post-run extraction logic without re-importing from one
benchmark package into another.
"""

from .patch import extract_patch

__all__ = ['extract_patch']
