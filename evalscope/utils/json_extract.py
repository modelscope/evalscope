"""Extract the last JSON object from a model reply that may wrap it in prose."""

import json
from typing import Any, Optional

__all__ = ['extract_last_json_object']


def extract_last_json_object(text: str) -> Optional[dict[str, Any]]:
    """Return the last JSON object that parses, or ``None``.

    Benchmark extractors currently hand-roll a first-brace-to-last-brace span (or a greedy
    ``\\{.*\\}`` regex). Both break as soon as the model shows an example object before its
    answer, because the span then covers more than one object.

    Scanning candidate ``{`` positions from the end and keeping the last one that
    :meth:`json.JSONDecoder.raw_decode` accepts keeps the model's final answer. Callers that need a
    hard failure keep raising on ``None`` themselves, so the failure mode does not change.
    """
    decoder = json.JSONDecoder()
    for start in range(len(text) - 1, -1, -1):
        if text[start] != '{':
            continue
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None
