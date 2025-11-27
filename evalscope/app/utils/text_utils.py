"""
Text processing utilities for the Evalscope dashboard.
"""
import json
import os
import re
from typing import Any, List, Optional

from evalscope.utils.logger import get_logger

logger = get_logger()


def convert_html_tags(text: str) -> str:
    """Convert HTML-like tags in the text to markdown-like tags.

    This function:
    - Converts <tag> to [tag] and </tag> to [/tag].
    - Preserves content within protected blocks: <code ...>...</code>, <audio ...>...</audio>, <image ...>...</image>.
    - Supports tags with attributes and self-closing forms.

    Args:
        text: The input text that may contain HTML-like tags.

    Returns:
        The text with tags converted to markdown-like forms, with protected blocks left unchanged.
    """
    # Return early for non-string or empty inputs
    if not isinstance(text, str) or not text:
        return text if isinstance(text, str) else str(text)

    # Step 1: Protect blocks like <code ...>...</code>, <audio ...>...</audio>, <image ...>...</image>
    protected_pattern = r'<(code|audio|image)\b[^>]*>.*?</\1\s*>'
    placeholders = []

    def replace_with_placeholder(match: re.Match) -> str:
        placeholders.append(match.group(0))
        return f'__PROTECTED_{len(placeholders) - 1}__'

    working = re.sub(protected_pattern, replace_with_placeholder, text, flags=re.IGNORECASE | re.DOTALL)

    # Step 2: Convert closing tags </tag> -> [/tag]
    working = re.sub(r'</\s*([a-zA-Z][\w-]*)\s*>', lambda m: f'[/{m.group(1).lower()}]', working)

    # Step 3: Convert opening/self-closing tags <tag ...> -> [tag]
    working = re.sub(r'<\s*([a-zA-Z][\w-]*)(?:\s[^>]*)?\s*/?>', lambda m: f'[{m.group(1).lower()}]', working)

    # Step 4: Restore protected blocks
    for i, original in enumerate(placeholders):
        working = working.replace(f'__PROTECTED_{i}__', original)

    return working


def process_string(string: str, max_length: Optional[int] = None) -> str:
    """Normalize a string for display by converting tags and truncating.

    Steps:
    1) Convert HTML-like tags to markdown-like tags for display.
    2) If max_length is provided and exceeded, truncate from middle with ellipsis.

    Args:
        string: The input string to process.
        max_length: Maximum allowed length. When exceeded, the string is middle-truncated.

    Returns:
        The processed string suitable for UI display.
    """
    string = convert_html_tags(string)  # for display labels e.g.
    if max_length and len(string) > max_length:
        return f'{string[:max_length // 2]}......{string[-max_length // 2:]}'
    return string


def process_model_prediction(item: Any, max_length: Optional[int] = None) -> str:
    """Render a model prediction as a markdown-friendly string.

    - Dicts/lists are pretty-printed as JSON inside a fenced code block.
    - Other values are stringified.
    - Final output passes through process_string for tag conversion and truncation.

    Args:
        item: The model prediction (can be dict, list, or any other type).
        max_length: Optional max output length for middle-truncation.

    Returns:
        A markdown-friendly string representation of the prediction.
    """
    if isinstance(item, (dict, list)):
        result = json.dumps(item, ensure_ascii=False, indent=2)
        result = f'```json\n{result}\n```'
    else:
        result = str(item)

    # Apply HTML tag conversion and truncation only at the final output
    return process_string(result, max_length)


def convert_markdown_image(text: str) -> str:
    """Convert image inputs to markdown image syntax when possible.

    Behavior:
    - If text is a data URI (base64, starts with "data:image"), return a markdown image.
    - If text points to an existing file and has a known image extension, convert to a
        markdown image with a gradio-compatible path.
    - Otherwise, fall back to process_model_prediction.

    Args:
        text: Either a data URI, a filesystem path, or any other string.

    Returns:
        A markdown-friendly string (image tag when resolvable).
    """
    if text.startswith('data:image'):
        # Convert base64 image data to a markdown image tag
        image_tag = f'![image]({text})'
        logger.debug(f'Converting base64 image data to markdown: {text[:30]}... -> {image_tag[:40]}...')
        return image_tag
    elif os.path.isfile(text):
        # Convert the image path to a markdown image tag
        if text.endswith('.png') or text.endswith('.jpg') or text.endswith('.jpeg'):
            text = os.path.abspath(text)
            image_tag = f'![image](gradio_api/file={text})'
            logger.debug(f'Converting image path to markdown: {text} -> {image_tag}')
            return image_tag
    return process_model_prediction(text)


def process_json_content(content: Any) -> str:
    """
    Process JSON-like content and return a pretty-printed JSON string.

    If content is a plain string, it will be wrapped as {"content": "<string>"} to
    keep a consistent schema for downstream consumers.

    Args:
        content: Arbitrary content that can be serialized to JSON.

    Returns:
        A JSON-formatted string (indent=2, ensure_ascii=False).
    """
    # If a raw string is provided, wrap it into a JSON object for schema consistency
    if isinstance(content, str):
        content = {'content': content}

    content_json = json.dumps(content, ensure_ascii=False, indent=2)
    return content_json
