# Copyright (c) Alibaba, Inc. and its affiliates.
"""Extract the final JSON object from free-form model output."""

import json
from typing import Any, Dict, Optional


def extract_last_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Return the last JSON object in ``text`` that parses, or ``None``.

    Model replies often show an example object, or repeat a draft, before the actual answer. A
    ``text[text.find('{'):text.rfind('}') + 1]`` slice then spans several objects and parses as
    nothing, so decode candidate objects instead: try every ``{`` from the end of the text and
    keep the last object that ``json.JSONDecoder.raw_decode`` accepts.

    The candidate whose object reaches furthest into the text wins, so a nested object never
    shadows the outer payload that contains it.

    Args:
        text (str): Raw model output; surrounding prose and Markdown fences are tolerated.

    Returns:
        Optional[Dict[str, Any]]: The last parseable JSON object, or ``None`` when ``text``
        contains no parseable object.
    """
    decoder = json.JSONDecoder()
    best: Optional[Dict[str, Any]] = None
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
