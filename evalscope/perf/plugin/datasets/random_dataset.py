import numpy as np
import re
from typing import Dict, Iterator, List, Tuple, Union

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.datasets.utils import gen_prompt_decode_to_target_len
from evalscope.perf.plugin.registry import register_dataset
from evalscope.utils import get_logger

logger = get_logger()


@register_dataset('random')
class RandomDatasetPlugin(DatasetPluginBase):
    """Read dataset and return prompt."""

    def __init__(self, query_parameters: Arguments):
        assert query_parameters.tokenizer_path, 'Tokenizer path is required for random data generation, please provide it with `--tokenizer-path`.'  # noqa: E501
        super().__init__(query_parameters)

        assert self.tokenizer is not None, 'Tokenizer should be initialized for random data generation.'  # noqa: E501
        self.prefix_length = self.query_parameters.prefix_length
        self.number = self.query_parameters.number or 1
        # Use numpy's default_rng for deterministic sampling
        self._rng = np.random.default_rng(None)

        # Filter out special tokens and byte-fallback tokens from vocabulary.
        # Byte-fallback tokens (e.g. <0xE4>) are NOT in all_special_ids but decode to
        # raw bytes that produce Mojibake when decoded as Latin-1 and re-encoded by the
        # server tokenizer, causing 3-5x token count inflation for CJK/multi-byte content.
        full_vocab_size = len(self.tokenizer)
        prohibited_tokens = set(self.tokenizer.all_special_ids)
        prohibited_tokens.update(self._get_byte_fallback_token_ids())
        all_tokens = np.arange(full_vocab_size)
        self.allowed_tokens = np.array(list(set(all_tokens) - prohibited_tokens))

        # Generate prefix once using allowed tokens
        self.prefix_ids = self.get_random_inputs(self.prefix_length)

        logger.info(
            f'Using {len(self.allowed_tokens)} allowed tokens out of {full_vocab_size} total tokens '
            f'(excluded {len(prohibited_tokens)} special/byte-fallback tokens)'
        )

    def build_messages(self) -> Iterator[Union[List[Dict], List[int]]]:
        """Yield prompts as text messages or, when --tokenize-prompt is set, as raw
        token-ID lists that bypass the decode/re-encode round-trip entirely."""
        tokenize_prompt = self.query_parameters.tokenize_prompt

        if tokenize_prompt:
            # Subtract prefix_length so that the final prompt
            # (prefix_ids + inner_seq) stays within [min, max]_prompt_length.
            min_prompt_length = self.query_parameters.min_prompt_length - self.prefix_length
            max_prompt_length = self.query_parameters.max_prompt_length - self.prefix_length + 1
            if min_prompt_length < 0:
                logger.warning(
                    f'min_prompt_length is less than prefix_length {self.prefix_length}, '
                    'setting min_prompt_length to 0.'
                )
                min_prompt_length = 0
        elif self.query_parameters.apply_chat_template:
            template_len = self.get_template_len()
            min_prompt_length = self.query_parameters.min_prompt_length - template_len
            max_prompt_length = self.query_parameters.max_prompt_length - template_len + 1
            if min_prompt_length < 0:
                logger.warning(
                    f'min_prompt_length is less than the template length {template_len}, '
                    'setting min_prompt_length to 0.'
                )
                min_prompt_length = 0
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
            if tokenize_prompt:
                # Fast path: yield token IDs directly, no decode/re-encode step.
                # The API plugin will send them as `prompt=[int, ...]` to /v1/completions.
                token_ids = self.generate_token_ids_only(
                    input_len=int(input_lens[i]),
                    offset=int(offsets[i]),
                    index=i,
                )
                yield token_ids
            else:
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

        if not tokenize_prompt and token_mismatch_total != 0:
            sign = 'more' if token_mismatch_total > 0 else 'fewer'
            logger.warning(
                f'Across all generated prompts, there were {abs(token_mismatch_total)} {sign} tokens '
                'than expected after decoding and re-encoding. This is expected due to the '
                'imperfect nature of the sampling procedure.'
            )

    def generate_token_ids_only(
        self,
        input_len: int,
        offset: int,
        index: int,
    ) -> List[int]:
        """Return a raw token-ID list of exactly `input_len` tokens (+ prefix).

        Unlike `generate_token_sequence`, this method never decodes tokens to text
        and therefore avoids any token ID → text → token ID round-trip inflation.
        The result is intended to be sent directly as `prompt=[int, ...]` to the
        /v1/completions endpoint via the --tokenize-prompt path.
        """
        inner_seq = self.allowed_tokens[(offset + index + np.arange(input_len)) % len(self.allowed_tokens)].tolist()
        return self.prefix_ids + inner_seq

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
            allowed_tokens=self.allowed_tokens,
        )
        total_input_len = len(adjusted_token_sequence)

        return prompt, total_input_len, token_mismatch

    def _get_byte_fallback_token_ids(self) -> set:
        """Return the set of token IDs that are byte-fallback tokens.

        Byte-fallback tokens (e.g. <0x00>–<0xFF>) are used by sentencepiece/tiktoken
        BPE tokenizers to represent raw bytes when a character sequence is out of
        vocabulary. They are NOT listed in `all_special_ids` but decode to a single
        raw byte. When a sequence of such tokens is decoded client-side and then
        re-encoded by the server tokenizer, the byte sequences are interpreted as
        Latin-1 / Mojibake, causing CJK or other multi-byte characters to expand
        into 3-5x more tokens than expected.

        Detection strategy: a token is a byte-fallback token if
        `tokenizer.convert_ids_to_tokens(id)` matches the pattern `<0xHH>`.
        """
        byte_pattern = re.compile(r'^<0x[0-9A-Fa-f]{2}>$')
        byte_ids = set()
        try:
            vocab = self.tokenizer.get_vocab()  # {str_token: int_id}
            for token_str, token_id in vocab.items():
                if byte_pattern.match(token_str):
                    byte_ids.add(token_id)
        except Exception:
            # Fallback: iterate over all IDs (use len(tokenizer), not vocab_size,
            # to cover added special tokens whose IDs are >= base vocab_size)
            for i in range(len(self.tokenizer)):
                try:
                    tok_str = self.tokenizer.convert_ids_to_tokens(i)
                    if tok_str and byte_pattern.match(tok_str):
                        byte_ids.add(i)
                except Exception:
                    pass
        return byte_ids

    def get_random_inputs(self, length: int) -> List[int]:
        """Generate random prefix tokens from allowed vocabulary."""
        if length <= 0:
            return []
        return self.allowed_tokens[self._rng.integers(0, len(self.allowed_tokens), size=length)].tolist()

    def get_template_len(self):
        empty_message = [self.create_message(text='')]
        template = self.tokenizer.apply_chat_template(empty_message, tokenize=True, add_generation_prompt=True)
        return len(template)
