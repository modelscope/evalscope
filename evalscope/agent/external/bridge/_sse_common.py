"""Constants shared by the anthropic and openai SSE synthesizers.

Both protocols slice the same ``ModelOutput`` into wire frames; keeping the
chunking + keep-alive cadence in one place ensures the two paths stay in
lock-step (a change to ``_TEXT_CHUNK`` for one would otherwise silently
drift from the other).
"""

from typing import List

from evalscope.api.model import ModelUsage

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
    return [text[i : i + max_len] for i in range(0, len(text), max_len)] or ['']


def prompt_tokens(usage: ModelUsage) -> int:
    """Input tokens including cached ones (the OpenAI and Gemini convention).

    Providers disagree on ``ModelUsage.input_tokens``: Anthropic leaves cache
    reads and writes out of it, OpenAI-compatible ones count them in. Only the
    former adds up to ``total_tokens`` once the cache counts are included.
    """
    cached = (usage.input_tokens_cache_read or 0) + (usage.input_tokens_cache_write or 0)
    if cached and usage.input_tokens + cached + usage.output_tokens == usage.total_tokens:
        return usage.input_tokens + cached
    return usage.input_tokens


def uncached_input_tokens(usage: ModelUsage) -> int:
    """Input tokens excluding cached ones (the Anthropic convention)."""
    cached = (usage.input_tokens_cache_read or 0) + (usage.input_tokens_cache_write or 0)
    return max(prompt_tokens(usage) - cached, 0)
