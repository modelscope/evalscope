import numpy as np
from typing import Dict, Iterator, List, Tuple

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.registry import register_dataset
from evalscope.utils import get_logger

logger = get_logger()


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


@register_dataset('random')
class RandomDatasetPlugin(DatasetPluginBase):
    """Read dataset and return prompt."""

    def __init__(self, query_parameters: Arguments):
        assert query_parameters.tokenizer_path, 'Tokenizer path is required for random data generation, please provide it with `--tokenizer-path`.'  # noqa: E501
        super().__init__(query_parameters)

        self.prefix_length = self.query_parameters.prefix_length
        self.number = self.query_parameters.number or 1
        # Use numpy's default_rng for deterministic sampling
        self._rng = np.random.default_rng(None)

        # Filter out special tokens from vocabulary
        vocab_size = self.tokenizer.vocab_size
        prohibited_tokens = set(self.tokenizer.all_special_ids)
        all_tokens = np.arange(vocab_size)
        self.allowed_tokens = np.array(list(set(all_tokens) - prohibited_tokens))

        # Generate prefix once using allowed tokens
        self.prefix_ids = self.get_random_inputs(self.prefix_length)

        logger.info(f'Using {len(self.allowed_tokens)} allowed tokens out of {vocab_size} total tokens')

    def build_messages(self) -> Iterator[List[Dict]]:
        if self.query_parameters.apply_chat_template:
            template_len = self.get_template_len()
            min_prompt_length = self.query_parameters.min_prompt_length - template_len
            max_prompt_length = self.query_parameters.max_prompt_length - template_len + 1
            assert min_prompt_length >= 0, f'min_prompt_length should be greater than or equal to the template length {template_len}.'  # noqa: E501
        else:
            min_prompt_length = self.query_parameters.min_prompt_length
            max_prompt_length = self.query_parameters.max_prompt_length + 1

        assert max_prompt_length >= min_prompt_length, 'max_prompt_length should be greater than or equal to min_prompt_length.'  # noqa: E501

        logger.info(f'Sampling input lengths from [{min_prompt_length}, {max_prompt_length})')

        # Sample input lengths
        input_lens = self._rng.integers(min_prompt_length, max_prompt_length, size=self.number)
        offsets = self._rng.integers(0, len(self.allowed_tokens), size=self.number)

        token_mismatch_total = 0
        for i in range(self.number):
            prompt, total_input_len, token_mismatch = self.generate_token_sequence(
                input_len=int(input_lens[i]),
                offset=int(offsets[i]),
                index=i,
            )
            token_mismatch_total += token_mismatch

            if self.query_parameters.apply_chat_template:
                message = self.create_message(prompt)
                yield [message]
            else:
                yield prompt

        if token_mismatch_total != 0:
            sign = 'more' if token_mismatch_total > 0 else 'fewer'
            logger.warning(
                f'Across all generated prompts, there were {abs(token_mismatch_total)} {sign} tokens '
                'than expected after decoding and re-encoding. This is expected due to the '
                'imperfect nature of the sampling procedure.'
            )

    def generate_token_sequence(
        self,
        input_len: int,
        offset: int,
        index: int,
    ) -> Tuple[str, int, int]:
        """
        Generate a token sequence and return (prompt, total_input_len, token_mismatch).

        Uses a deterministic sequence based on offset and index to ensure reproducibility,
        then decodes and re-encodes to handle tokenizer inconsistencies.
        """
        # Build the inner sequence by sampling sequentially from allowed tokens
        inner_seq = self.allowed_tokens[(offset + index + np.arange(input_len)) % len(self.allowed_tokens)].tolist()
        token_sequence = self.prefix_ids + inner_seq

        # Decode, then re-encode with retry logic to match target length
        total_input_len = self.prefix_length + int(input_len)
        prompt, adjusted_token_sequence, token_mismatch = gen_prompt_decode_to_target_len(
            tokenizer=self.tokenizer,
            token_sequence=token_sequence,
            target_token_len=total_input_len,
            add_special_tokens=False,
            rng=self._rng,
        )
        total_input_len = len(adjusted_token_sequence)

        return prompt, total_input_len, token_mismatch

    def get_random_inputs(self, length: int) -> List[int]:
        """Generate random prefix tokens from allowed vocabulary."""
        if length <= 0:
            return []
        return self.allowed_tokens[self._rng.integers(0, len(self.allowed_tokens), size=length)].tolist()

    def get_template_len(self):
        empty_message = [self.create_message(text='')]
        template = self.tokenizer.apply_chat_template(empty_message, tokenize=True, add_generation_prompt=True)
        return len(template)
