import numpy as np
from typing import Dict, List, Tuple


def tokenize_chat_messages(tokenizer, messages: List[Dict], add_generation_prompt: bool = True) -> List[int]:
    """Apply a tokenizer chat template and return a plain ``List[int]`` of token IDs.

    Normalises the return value of ``tokenizer.apply_chat_template`` so callers
    always receive a flat Python list of ints regardless of the installed transformers
    version.  transformers >= 4.46 changed ``apply_chat_template(tokenize=True)``
    to return a ``BatchEncoding`` dict-like object instead of ``List[int]``.

    Args:
        tokenizer: A HuggingFace / ModelScope tokenizer instance.
        messages: Chat messages in OpenAI format (list of ``{'role': ..., 'content': ...}`` dicts).
        add_generation_prompt: Whether to append the assistant generation prompt.

    Returns:
        List[int]: Flat list of token IDs.

    Raises:
        TypeError: If the tokenizer returns an unexpected type that cannot be converted.
    """
    result = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt)

    # Old transformers: returns List[int] directly.
    if isinstance(result, list):
        # Guard against an unexpected batch dimension: [[token_ids]] -> [token_ids]
        if result and isinstance(result[0], list):
            return result[0]
        return result

    # transformers >= 4.46: returns BatchEncoding (dict-like) with an input_ids field.
    if hasattr(result, 'input_ids'):
        ids = result.input_ids
        ids = ids.tolist() if hasattr(ids, 'tolist') else list(ids)
        # Guard against batch dimension from tensor conversion: [[ids]] -> [ids]
        if ids and isinstance(ids[0], list):
            return ids[0]
        return ids

    raise TypeError(
        f'tokenize_chat_messages: unexpected return type {type(result)!r} from '
        'tokenizer.apply_chat_template. Expected List[int] or BatchEncoding with input_ids.'
    )


def gen_prompt_decode_to_target_len(
    tokenizer,
    token_sequence: List[int],
    target_token_len: int,
    max_retry: int = 10,
    add_special_tokens: bool = False,
    rng: np.random.Generator = None,
    allowed_tokens: np.ndarray = None,
) -> Tuple[str, List[int], int]:
    """
    Ensure decoded-then-encoded prompt length matches the target token length.

    This function decodes an initial token sequence to text and re-encodes it,
    iteratively adjusting the token sequence length to match a target.
    This is necessary because some tokenizers do not guarantee a 1:1 mapping
    between consecutive tokens and the decoded-then-encoded sequence length.
    For example, for GPT2Tokenizer:
    [6880, 6881] -> ['Ġcalls', 'here'] ->
    [1650, 939, 486] -> ['Ġcall', 'sh', 'ere']

    Returns a tuple of the final prompt string, adjusted token sequence, and token mismatch.
    """
    remain_num_try = max_retry
    token_mismatch = 0
    vocab_size = len(tokenizer)

    # Build the pool of tokens to use when filling gaps; exclude special tokens if possible
    if allowed_tokens is None:
        prohibited = set(tokenizer.all_special_ids)
        allowed_tokens = np.array([t for t in range(vocab_size) if t not in prohibited])
        if len(allowed_tokens) == 0:
            allowed_tokens = np.arange(vocab_size)

    while True:
        prompt = tokenizer.decode(token_sequence)
        token_sequence = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)

        if remain_num_try <= 0:
            if len(token_sequence) != target_token_len:
                token_mismatch = len(token_sequence) - target_token_len
            break

        if len(token_sequence) == target_token_len:
            break
        elif len(token_sequence) < target_token_len:
            # Generate extra tokens to reach target length, only from allowed (non-special) tokens
            fill_size = target_token_len - len(token_sequence)
            if rng is not None:
                indices = rng.integers(0, len(allowed_tokens), size=fill_size)
            else:
                indices = np.random.randint(0, len(allowed_tokens), size=fill_size)
            extra_tokens = allowed_tokens[indices].tolist()
            token_sequence.extend(extra_tokens)
        elif len(token_sequence) > target_token_len:
            # Truncate to target length
            token_sequence = token_sequence[:target_token_len]

        remain_num_try -= 1

    return prompt, token_sequence, token_mismatch
