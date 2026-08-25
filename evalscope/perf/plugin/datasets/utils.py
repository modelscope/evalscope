from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from evalscope.utils.logger import get_logger

logger = get_logger()

# Guidance appended to the error raised when `apply_chat_template` fails.
_NO_CHAT_TEMPLATE_HINT = (
    'Some checkpoints ship no Jinja chat template at all (DeepSeek-V3.2 / V4 provide `encoding` scripts '
    'instead, and base/pretrain checkpoints have none), and some templates reject the message shape used '
    'here. The template is applied client-side only to count prompt tokens the way the model service does. '
    'Options:\n'
    '  1. drop `--tokenizer-path` and let the `usage` reported by the model service provide token counts;\n'
    '  2. use a tokenizer that ships a chat template and shares the vocabulary of the served model (a '
    'different vocabulary silently distorts token counts);\n'
    '  3. pass `--no-apply-chat-template` and benchmark a text-completions endpoint instead.'
)


def load_tokenizer(tokenizer_path: str) -> object:
    """Load a tokenizer from the given path, with a fallback for models that lack ``max_position_embeddings``.

    Some model configs (e.g. DeepSeek-V3) do not define ``max_position_embeddings``, which causes
    ``transformers >= 5.x`` to raise an ``AttributeError`` inside ``standardize_rope_params()`` when
    ``trust_remote_code=True`` is used.  This helper retries with ``trust_remote_code=False`` in that
    case so evaluation can continue without manual intervention.

    Args:
        tokenizer_path (str): Local path or ModelScope/HuggingFace model ID for the tokenizer.

    Returns:
        The loaded tokenizer instance.

    Raises:
        AttributeError: Re-raised if the error is unrelated to ``max_position_embeddings``.
        Exception: Any other exception from ``AutoTokenizer.from_pretrained`` is propagated as-is.
    """
    from modelscope import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except AttributeError as e:
        if 'max_position_embeddings' in str(e):
            logger.warning(
                f'Tokenizer loading with trust_remote_code=True failed: {e}. '
                'Retrying with trust_remote_code=False.'
            )
            return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)
        raise


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
        ImportError: Propagated unchanged when an optional dependency (e.g. jinja2) is missing.
        ValueError: If the tokenizer has no chat template usable for these messages.
        TypeError: If the tokenizer returns an unexpected type that cannot be converted.
    """
    try:
        result = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt)
    except ImportError:
        # Missing optional dependency (e.g. jinja2): the template exists, so surface the
        # real dependency error instead of the misleading no-template hint.
        raise
    except Exception as e:
        name = getattr(tokenizer, 'name_or_path', None) or type(tokenizer).__name__
        raise ValueError(
            f'Failed to apply the chat template of tokenizer `{name}`: {e}\n{_NO_CHAT_TEMPLATE_HINT}'
        ) from e

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


def converge_to_token_len(
    tokenizer,
    token_ids: List[int],
    target_len: int,
    fill: Callable[[int], List[int]],
    add_special_tokens: bool = False,
    skip_special_tokens: bool = False,
    max_retry: int = 10,
) -> Tuple[str, List[int], int]:
    """Decode/re-encode ``token_ids`` until the re-encoded length matches ``target_len``.

    Tokenizers do not guarantee that decoding then re-encoding a token sequence
    preserves its length.  For example, with GPT2Tokenizer:
    ``[6880, 6881] -> ['Ġcalls', 'here'] -> [1650, 939, 486] -> ['Ġcall', 'sh', 'ere']``.
    This helper iteratively truncates over-length re-encodes and tops up
    under-length ones until the target is hit or ``max_retry`` is exhausted.

    The token source used for topping up is supplied by the caller via ``fill``,
    which is what distinguishes the two use cases: random prompt generation
    samples arbitrary vocabulary tokens, while long-context prefix injection
    continues deterministically through real text.

    Args:
        tokenizer: A HuggingFace / ModelScope tokenizer instance.
        token_ids: Initial token IDs to converge.
        target_len: Desired token count after decode/re-encode.
        fill: Returns ``n`` extra token IDs to append when the sequence is short.
        add_special_tokens: Whether to add special tokens when re-encoding.
        skip_special_tokens: Whether to skip special tokens when decoding.
        max_retry: Maximum decode/re-encode refinement rounds.

    Returns:
        Tuple of the final text, its re-encoded token IDs, and the token
        mismatch (``len(ids) - target_len``, ``0`` when converged).
    """
    for attempt in range(max_retry + 1):
        text = tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        token_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
        mismatch = len(token_ids) - target_len
        # Report the mismatch on the final round instead of adjusting again, so
        # the returned text / ids / mismatch stay consistent with each other.
        if mismatch == 0 or attempt == max_retry:
            return text, token_ids, mismatch
        token_ids = token_ids[:target_len] if mismatch > 0 else token_ids + fill(-mismatch)


def gen_prompt_decode_to_target_len(
    tokenizer,
    token_sequence: List[int],
    target_token_len: int,
    max_retry: int = 10,
    add_special_tokens: bool = False,
    allowed_tokens: np.ndarray = None,
) -> Tuple[str, List[int], int]:
    """
    Ensure decoded-then-encoded prompt length matches the target token length.

    Thin wrapper over :func:`converge_to_token_len` that fills length gaps with
    random tokens drawn from the allowed (non-special) vocabulary, used for
    synthetic ``random`` dataset prompts.

    Returns a tuple of the final prompt string, adjusted token sequence, and token mismatch.
    """
    # Build the pool of tokens to use when filling gaps; exclude special tokens if possible
    if allowed_tokens is None:
        vocab_size = len(tokenizer)
        prohibited = set(tokenizer.all_special_ids)
        allowed_tokens = np.array([t for t in range(vocab_size) if t not in prohibited])
        if len(allowed_tokens) == 0:
            allowed_tokens = np.arange(vocab_size)

    def fill(size: int) -> List[int]:
        return allowed_tokens[np.random.randint(0, len(allowed_tokens), size=size)].tolist()

    return converge_to_token_len(
        tokenizer=tokenizer,
        token_ids=token_sequence,
        target_len=target_token_len,
        fill=fill,
        add_special_tokens=add_special_tokens,
        max_retry=max_retry,
    )


def truncate_text_to_token_len(text: str, target_len: int, tokenizer, add_special_tokens: bool = False) -> str:
    """Truncate ``text`` so it occupies at most ``target_len`` tokens.

    Encodes with ``add_special_tokens=False`` by default (bare content tokens).
    Text that already fits within ``target_len`` is returned unchanged; longer
    text is truncated to the first ``target_len`` ids and decoded back.

    Args:
        text: The input text to truncate.
        target_len: Maximum number of tokens to keep.
        tokenizer: A HuggingFace / ModelScope tokenizer instance.
        add_special_tokens: Whether to add special tokens when encoding.

    Returns:
        The truncated text.
    """
    ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    if len(ids) <= target_len:
        return text
    return tokenizer.decode(ids[:target_len], skip_special_tokens=True)


def fit_text_to_token_len(
    text: str,
    target_len: int,
    mode: str,
    tokenizer,
    add_special_tokens: bool = False,
) -> Optional[str]:
    """Fit ``text`` to ``target_len`` tokens according to ``mode``.

    Over-length text is always truncated to ``target_len``.  The handling of
    text shorter than ``target_len`` depends on ``mode``:

    - ``cap``: keep the shorter text as-is (result <= target).
    - ``drop``: return ``None`` so the caller skips it (exact length only).

    Args:
        text: The input text.
        target_len: Target token length.
        mode: One of ``'cap'``, ``'drop'``.
        tokenizer: A HuggingFace / ModelScope tokenizer instance.
        add_special_tokens: Whether to add special tokens when encoding.

    Returns:
        The adjusted text, or ``None`` when the prompt should be skipped.

    Raises:
        ValueError: If ``mode`` is not one of the supported values.
    """
    ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    if len(ids) > target_len:
        return tokenizer.decode(ids[:target_len], skip_special_tokens=True)
    # Already at or below the target: avoid a redundant decode/re-encode round-trip
    # (which could otherwise alter the token count).
    if mode == 'cap' or len(ids) == target_len:
        return text
    if mode == 'drop':
        return None
    raise ValueError(f"Unknown input_len_mode: {mode!r}. Expected one of 'cap', 'drop'.")


def fit_prefix_to_budget(prefix_ids: List[int], budget: int, tokenizer, max_retry: int = 10) -> str:
    """Slice (tiling if needed) ``prefix_ids`` into text of exactly ``budget`` tokens.

    Used by the long-context prefix injection mechanism (issue #1524): the
    prefix must fill the remaining token budget ``target_input_len - prompt_len``
    so the total request input hits the target length.  When ``prefix_ids`` is
    shorter than the budget it is repeated (tiled) to cover it.

    Thin wrapper over :func:`converge_to_token_len` that fills length gaps by
    continuing deterministically through the tiled prefix pool, so the result
    stays real low-entropy text (unlike the random filler used for synthetic
    prompts).

    Args:
        prefix_ids: Token IDs of the full prefix text (``add_special_tokens=False``).
        budget: Target token count; ``<= 0`` returns an empty string.
        tokenizer: A HuggingFace / ModelScope tokenizer instance.
        max_retry: Maximum decode/re-encode refinement rounds.

    Returns:
        The prefix text occupying (as close as possible to) ``budget`` tokens.

    Raises:
        ValueError: If ``prefix_ids`` is empty while ``budget`` is positive.
    """
    if budget <= 0:
        return ''
    if not prefix_ids:
        raise ValueError('prefix_ids must not be empty when budget > 0.')

    pool = list(prefix_ids)
    while len(pool) < budget:
        pool.extend(prefix_ids)
    cursor = budget

    def fill(size: int) -> List[int]:
        """Take the next unconsumed slice of the pool, extending it on demand."""
        nonlocal cursor
        while cursor + size > len(pool):
            pool.extend(prefix_ids)
        chunk = pool[cursor:cursor + size]
        cursor += size
        return chunk

    text, _, mismatch = converge_to_token_len(
        tokenizer=tokenizer,
        token_ids=pool[:budget],
        target_len=budget,
        fill=fill,
        skip_special_tokens=True,
        max_retry=max_retry,
    )
    if mismatch != 0:
        logger.warning(
            f'fit_prefix_to_budget: prefix converged to {budget + mismatch} tokens instead of the '
            f'{budget}-token budget after {max_retry} retries (tokenizer round-trip drift).'
        )
    return text
