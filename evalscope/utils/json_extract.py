"""Extract the last JSON object from a model reply that may wrap it in prose."""

import json
from typing import Any, Optional

__all__ = ['extract_last_json_object']


def extract_last_json_object(text: str) -> Optional[dict[str, Any]]:
    """Return the last JSON object that parses, or ``None``.

    Benchmark extractors currently hand-roll a first-brace-to-last-brace span (or a greedy
    ``\\{.*\\}`` regex). Both break as soon as the model shows an example object before its
    answer, because the span then covers more than one object.

    Candidate ``{`` positions are decoded from the end; the candidate whose object reaches
    furthest into the text wins, so a nested object cannot shadow the outer payload that
    contains it. Callers that need a hard failure keep raising on ``None`` themselves.
    """
    decoder = json.JSONDecoder()
    best: Optional[dict[str, Any]] = None
    best_end = -1
    position = len(text)
    while True:
        position = text.rfind('{', 0, position)
        if position == -1:
            return best
        try:
            value, end = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and end > best_end:
            best, best_end = value, end
