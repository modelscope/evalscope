import re
from typing import Optional

from .content import ContentReasoning


def parse_content_with_reasoning(content: str) -> tuple[str, Optional[ContentReasoning]]:
    """
    Looks for and extracts <think/> tags into reasoning text.

    Returns a tuple:
    - The first element is the input content with the <think> tag and its contents fully removed.
    - The second element is a ContentReasoning object (or None if no <think> tag is found).
    """
    # Match <think> tag with optional attributes anywhere in the string
    pattern = (r'<think(?:\s+signature="([^"]*)")?(?:\s+redacted="(true)")?\s*>(.*?)</think>')
    match = re.search(pattern, content, re.DOTALL)

    if match:
        signature = match.group(1)  # This will be None if not present
        redacted_value = match.group(2)  # This will be "true" or None
        reasoning = match.group(3).strip()
        # Remove the matched <think>...</think> from the input
        start, end = match.span()

        return (
            (content[:start] + content[end:]).strip(),
            ContentReasoning(
                reasoning=reasoning,
                signature=signature,
                redacted=redacted_value == 'true',
            ),
        )
    else:
        return content, None
