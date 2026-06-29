import math
import multiprocessing
import numpy as np
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.datasets.utils import gen_prompt_decode_to_target_len, load_tokenizer, tokenize_chat_messages
from evalscope.perf.plugin.registry import register_dataset
from evalscope.utils import get_logger
from evalscope.utils.tqdm_utils import TqdmLogging as tqdm

logger = get_logger()

_RANDOM_GENERATION_WORKER_STATE: Optional['_RandomGenerationWorkerState'] = None
_MIN_AUTO_PARALLEL_PROMPT_LENGTH = 8192
_MIN_AUTO_PARALLEL_TOKEN_WORK = 4 * 1024 * 1024


@dataclass
class _RandomGenerationWorkerState:
    """Process-local state initialized once per random generation worker."""

    tokenizer: Optional[Any]
    tokenize_prompt: bool
    apply_chat_template: bool
    prefix_ids: List[int]
    allowed_tokens: np.ndarray


def _get_random_generation_context() -> BaseContext:
    """Return the spawn context used for random request generation workers."""
    return multiprocessing.get_context('spawn')


def _init_random_generation_worker(
    tokenizer_path: str,
    tokenize_prompt: bool,
    apply_chat_template: bool,
    prefix_ids: List[int],
    allowed_tokens: np.ndarray,
) -> None:
    """Initialize per-process state for random request generation."""
    global _RANDOM_GENERATION_WORKER_STATE
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    tokenizer = None if tokenize_prompt else load_tokenizer(tokenizer_path)
    _RANDOM_GENERATION_WORKER_STATE = _RandomGenerationWorkerState(
        tokenizer=tokenizer,
        tokenize_prompt=tokenize_prompt,
        apply_chat_template=apply_chat_template,
        prefix_ids=prefix_ids,
        allowed_tokens=np.asarray(allowed_tokens),
    )


def _build_random_message_worker(
    task: Tuple[int, int, int, Optional[int]],
) -> Tuple[int, Union[List[Dict], List[int], str], int]:
    """Build one random message from an index-level generation task."""
    if _RANDOM_GENERATION_WORKER_STATE is None:
        raise RuntimeError('Random request generation worker state is not initialized.')

    index, input_len, offset, seed = task
    state = _RANDOM_GENERATION_WORKER_STATE
    allowed_tokens = state.allowed_tokens
    prefix_ids = state.prefix_ids
    inner_seq = allowed_tokens[(offset + index + np.arange(input_len)) % len(allowed_tokens)].tolist()
    token_sequence = prefix_ids + inner_seq

    if state.tokenize_prompt:
        return index, token_sequence, 0

    if seed is None:
        raise ValueError('Parallel random request generation requires a per-item seed.')
    np.random.seed(seed)
    target_token_len = len(prefix_ids) + int(input_len)
    prompt, adjusted_token_sequence, token_mismatch = gen_prompt_decode_to_target_len(
        tokenizer=state.tokenizer,
        token_sequence=token_sequence,
        target_token_len=target_token_len,
        add_special_tokens=False,
        allowed_tokens=allowed_tokens,
    )
    if len(adjusted_token_sequence) != target_token_len:
        token_mismatch = len(adjusted_token_sequence) - target_token_len

    if state.apply_chat_template:
        return index, [{'role': 'user', 'content': prompt}], token_mismatch
    return index, prompt, token_mismatch


@register_dataset('random')
class RandomDatasetPlugin(DatasetPluginBase):
    """Read dataset and return prompt."""

    def __init__(self, query_parameters: Arguments):
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        assert query_parameters.tokenizer_path, 'Tokenizer path is required for random data generation, please provide it with `--tokenizer-path`.'  # noqa: E501
        super().__init__(query_parameters)

        assert self.tokenizer is not None, 'Tokenizer should be initialized for random data generation.'  # noqa: E501
        self.prefix_length = self.query_parameters.prefix_length
        # Include warmup_count so the generator produces enough unique items
        # to cover both warmup and benchmark requests without cycling reuse.
        self.number = self.query_parameters.total_count or 1

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

    def build_messages(self) -> Iterator[Union[List[Dict], List[int], str]]:
        """Yield prompts as text messages or, when --tokenize-prompt is set, as raw
        token-ID lists that bypass the decode/re-encode round-trip entirely."""
        yield from self._build_messages_for_count(self.number)

    def _build_messages_for_count(self, total_count: int) -> Iterator[Union[List[Dict], List[int], str]]:
        """Yield random prompts for a fixed item count."""
        plan = self._create_generation_plan(total_count, include_seeds=not self.query_parameters.tokenize_prompt)
        token_mismatch_total = 0
        for index, input_len, offset, seed in self._iter_generation_plan(plan):
            message, token_mismatch = self._build_random_message(input_len, offset, index, seed)
            token_mismatch_total += token_mismatch
            yield message

        self._log_token_mismatch(token_mismatch_total)

    def supports_parallel_message_generation(self) -> bool:
        """Random prompts are independent by index and can be generated in worker processes."""
        if self.query_parameters.num_workers > 1:
            return True
        if self.query_parameters.tokenize_prompt:
            return False
        prompt_length_estimate = (
            self.query_parameters.min_prompt_length + self.query_parameters.max_prompt_length
        ) // 2
        token_work_estimate = prompt_length_estimate * self.number
        return (
            prompt_length_estimate >= _MIN_AUTO_PARALLEL_PROMPT_LENGTH
            and token_work_estimate >= _MIN_AUTO_PARALLEL_TOKEN_WORK
        )

    def build_messages_parallel(self, total_count: int, workers: int) -> List[Union[List[Dict], List[int], str]]:
        """Build random request messages across multiple worker processes."""
        if total_count <= 0:
            return []

        workers = max(1, min(workers, total_count))
        if workers == 1:
            return list(self._build_messages_for_count(total_count))

        process_context = _get_random_generation_context()
        plan = self._create_generation_plan(total_count, include_seeds=not self.query_parameters.tokenize_prompt)
        messages: List[Union[List[Dict], List[int], str, None]] = [None] * total_count
        token_mismatch_total = 0
        chunk_size = max(1, min(64, math.ceil(total_count / (workers * 8))))

        with tqdm(total=total_count, desc='Generating[requests]', logger=logger) as pbar:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=process_context,
                initializer=_init_random_generation_worker,
                initargs=(
                    self.query_parameters.tokenizer_path,
                    self.query_parameters.tokenize_prompt,
                    self.query_parameters.apply_chat_template,
                    self.prefix_ids,
                    self.allowed_tokens,
                ),
            ) as executor:
                for index, message, token_mismatch in executor.map(
                    _build_random_message_worker,
                    self._iter_generation_plan(plan),
                    chunksize=chunk_size,
                ):
                    messages[index] = message
                    token_mismatch_total += token_mismatch
                    pbar.update(1)

        missing = [index for index, message in enumerate(messages) if message is None]
        if missing:
            raise RuntimeError(f'Parallel random request generation missed {len(missing)} items.')

        self._log_token_mismatch(token_mismatch_total)
        return [message for message in messages if message is not None]

    def _resolve_prompt_length_bounds(self) -> Tuple[int, int]:
        """Resolve the random prompt length sampling interval."""
        if self.query_parameters.apply_chat_template and not self.query_parameters.tokenize_prompt:
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
        return min_prompt_length, max_prompt_length

    def _create_generation_plan(
        self,
        total_count: int,
        include_seeds: bool,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Pre-sample all per-index random inputs in the parent process."""
        min_prompt_length, max_prompt_length = self._resolve_prompt_length_bounds()
        input_lens = np.random.randint(min_prompt_length, max_prompt_length, size=total_count)
        global_offset = self.query_parameters.dataset_offset
        offsets = (np.random.randint(0, len(self.allowed_tokens), size=total_count)
                   + global_offset) % len(self.allowed_tokens)
        seeds = None
        if include_seeds:
            seeds = np.random.randint(0, np.iinfo(np.uint32).max, size=total_count, dtype=np.uint32)
        return input_lens, offsets, seeds

    def _iter_generation_plan(
        self,
        plan: Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]],
    ) -> Iterator[Tuple[int, int, int, Optional[int]]]:
        """Iterate over pre-sampled generation tasks."""
        input_lens, offsets, seeds = plan
        for index in range(len(input_lens)):
            seed = None if seeds is None else int(seeds[index])
            yield index, int(input_lens[index]), int(offsets[index]), seed

    def _build_random_message(
        self,
        input_len: int,
        offset: int,
        index: int,
        seed: Optional[int] = None,
    ) -> Tuple[Union[List[Dict], List[int], str], int]:
        """Build one random message in the current process."""
        if self.query_parameters.tokenize_prompt:
            token_ids = self.generate_token_ids_only(input_len=input_len, offset=offset, index=index)
            return token_ids, 0

        if seed is None:
            prompt, _, token_mismatch = self.generate_token_sequence(
                input_len=input_len,
                offset=offset,
                index=index,
            )
        else:
            random_state = np.random.get_state()
            np.random.seed(seed)
            try:
                prompt, _, token_mismatch = self.generate_token_sequence(
                    input_len=input_len,
                    offset=offset,
                    index=index,
                )
            finally:
                np.random.set_state(random_state)

        if self.query_parameters.apply_chat_template:
            message = self.create_message(prompt)
            return [message], token_mismatch
        return prompt, token_mismatch

    def _log_token_mismatch(self, token_mismatch_total: int) -> None:
        """Log aggregate token mismatch from decode/re-encode prompt generation."""
        if not self.query_parameters.tokenize_prompt and token_mismatch_total != 0:
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
        """Return a raw token-ID list of exactly `prefix_length + input_len` tokens.

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
        return self.allowed_tokens[np.random.randint(0, len(self.allowed_tokens), size=length)].tolist()

    def get_template_len(self):
        empty_message = [self.create_message(text='')]
        return len(tokenize_chat_messages(self.tokenizer, empty_message))
