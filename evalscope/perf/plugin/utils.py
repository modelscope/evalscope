from typing import Dict, List


def tokenize_chat_messages(tokenizer, messages: List[Dict], add_generation_prompt: bool = True) -> List[int]:
    """Apply a tokenizer chat template and return a plain ``List[int]`` of token IDs.

    Normalises the return value of ``tokenizer.apply_chat_template`` so callers
    always receive a plain Python list regardless of the installed transformers
    version.  transformers >= 4.46 changed ``apply_chat_template(tokenize=True)``
    to return a ``BatchEncoding`` dict-like object instead of ``List[int]``.

    Args:
        tokenizer: A HuggingFace / ModelScope tokenizer instance.
        messages: Chat messages in OpenAI format (list of ``{'role': ..., 'content': ...}`` dicts).
        add_generation_prompt: Whether to append the assistant generation prompt.

    Returns:
        List[int]: Flat list of token IDs.
    """
    result = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt)
    if isinstance(result, list):
        return result
    # BatchEncoding (transformers >= 4.46): extract input_ids
    if hasattr(result, 'input_ids'):
        ids = result.input_ids
        return ids.tolist() if hasattr(ids, 'tolist') else list(ids)
    return list(result)
