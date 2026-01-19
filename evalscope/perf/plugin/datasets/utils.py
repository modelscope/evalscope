import numpy as np
from typing import List, Tuple


def gen_prompt_decode_to_target_len(
    tokenizer,
    token_sequence: List[int],
    target_token_len: int,
    max_retry: int = 10,
    add_special_tokens: bool = False,
    rng: np.random.Generator = None,
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
    vocab_size = tokenizer.vocab_size

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
            # Generate extra tokens to reach target length
            if rng is not None:
                extra_tokens = rng.integers(
                    0,
                    vocab_size,
                    size=target_token_len - len(token_sequence),
                ).tolist()
            else:
                extra_tokens = np.random.randint(
                    0,
                    vocab_size,
                    size=target_token_len - len(token_sequence),
                ).tolist()
            token_sequence.extend(extra_tokens)
        elif len(token_sequence) > target_token_len:
            # Truncate to target length
            token_sequence = token_sequence[:target_token_len]

        remain_num_try -= 1

    return prompt, token_sequence, token_mismatch
