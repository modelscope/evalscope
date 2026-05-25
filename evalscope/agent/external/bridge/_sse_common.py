"""Constants shared by the anthropic and openai SSE synthesizers.

Both protocols slice the same ``ModelOutput`` into wire frames; keeping the
chunking + keep-alive cadence in one place ensures the two paths stay in
lock-step (a change to ``_TEXT_CHUNK`` for one would otherwise silently
drift from the other).
"""

from typing import List

#: Max characters per text/reasoning delta frame.
TEXT_CHUNK = 48

#: Max characters per tool-call ``arguments`` delta frame.
TOOL_INPUT_CHUNK = 20

#: How long to wait between keep-alive frames while the upstream model is
#: still generating. Most HTTP intermediaries close idle connections after
#: ~30s; 5s leaves plenty of headroom.
PING_INTERVAL_S = 5.0


def iter_chunks(text: str, max_len: int) -> List[str]:
    """Slice ``text`` into ``max_len`` segments; always yields at least one
    entry (an empty string when ``text`` is empty)."""
    return [text[i:i + max_len] for i in range(0, len(text), max_len)] or ['']
