"""
Text processing utilities for the Evalscope dashboard.
"""
import json
import os
import re
from typing import Any, Dict, List, Optional

from evalscope.utils.logger import get_logger
from ..constants import LATEX_DELIMITERS

logger = get_logger()


def convert_markdown_image(text: str):
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
    return text


def convert_html_tags(text):
    # match begin label
    text = re.sub(r'<(\w+)>', r'[\1]', text)
    # match end label
    text = re.sub(r'</(\w+)>', r'[/\1]', text)
    return text


def process_string(string: str, max_length: int = 2048) -> str:
    string = convert_html_tags(string)  # for display labels e.g.
    if max_length and len(string) > max_length:
        return f'{string[:max_length // 2]}......{string[-max_length // 2:]}'
    return string


def dict_to_markdown(data) -> str:
    markdown_lines = []

    for key, value in data.items():
        bold_key = f'**{key}**'

        if isinstance(value, list):
            value_str = '\n' + '\n'.join([f'- {process_model_prediction(item, max_length=None)}' for item in value])
        elif isinstance(value, dict):
            value_str = dict_to_markdown(value)
        else:
            value_str = str(value)

        value_str = process_string(value_str, max_length=None)  # Convert HTML tags but don't truncate
        markdown_line = f'{bold_key}:\n{value_str}'
        markdown_lines.append(markdown_line)

    return '\n\n'.join(markdown_lines)


def process_model_prediction_old(item: Any, max_length: int = 2048) -> str:
    """
    Process model prediction output into a formatted string.

    Args:
        item: The item to process. Can be a string, list, or dictionary.
        max_length: The maximum length of the output string.

    Returns:
        A formatted string representation of the input.
    """
    if isinstance(item, dict):
        result = dict_to_markdown(item)
    elif isinstance(item, list):
        result = '\n'.join([f'- {process_model_prediction(i, max_length=None)}' for i in item])
    else:
        result = str(item)

    # Apply HTML tag conversion and truncation only at the final output
    if max_length is not None:
        return process_string(result, max_length)
    return result


def process_model_prediction(item: Any, max_length: Optional[int] = None) -> str:
    if isinstance(item, (dict, list)):
        result = json.dumps(item, ensure_ascii=False, indent=2)
        result = f'```json\n{result}\n```'
    else:
        result = str(item)

    # Apply HTML tag conversion and truncation only at the final output
    if max_length is not None:
        return process_string(result, max_length)

    return result


def process_json_content(content: Any) -> str:
    """
    Process JSON content to convert it into a markdown-friendly format.

    Args:
        content (str): The JSON content as a string.

    Returns:
        str: The processed content formatted for markdown display.
    """

    if isinstance(content, str):
        content = {'content': content}

    content_json = json.dumps(content, ensure_ascii=False, indent=2)
    return content_json
