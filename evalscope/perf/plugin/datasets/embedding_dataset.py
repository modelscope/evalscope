"""Embedding dataset plugin for evalscope perf.

This plugin provides datasets suitable for embedding model performance testing.
"""

import json
import numpy as np
import os
from typing import Iterator, List

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.datasets.utils import gen_prompt_decode_to_target_len
from evalscope.perf.plugin.registry import register_dataset
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_dataset(['random_embedding'])
class RandomEmbeddingDatasetPlugin(DatasetPluginBase):
    """Dataset plugin for random embedding generation."""

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)

        if not self.tokenizer:
            raise ValueError('Tokenizer is required for random embedding generation. Please provide --tokenizer-path.')

        # Use numpy's default_rng for sampling
        self._rng = np.random.default_rng(None)

        # Filter out special tokens from vocabulary
        vocab_size = self.tokenizer.vocab_size
        prohibited_tokens = set(self.tokenizer.all_special_ids)
        all_tokens = np.arange(vocab_size)
        self.allowed_tokens = np.array(list(set(all_tokens) - prohibited_tokens))
        logger.info(
            f'Using {len(self.allowed_tokens)} allowed tokens out of {vocab_size} total tokens for random generation.'
        )

    def _generate_random_text(self) -> str:
        """Generate a random text of reasonable length for embedding using tokenizer."""
        min_len = self.query_parameters.min_prompt_length
        max_len = self.query_parameters.max_prompt_length

        # Ensure min <= max
        if min_len > max_len:
            min_len, max_len = max_len, min_len

        # Ensure positive length
        min_len = max(1, min_len)
        max_len = max(1, max_len)

        target_length = self._rng.integers(min_len, max_len + 1)

        # Generate random tokens
        tokens = self.allowed_tokens[self._rng.integers(0, len(self.allowed_tokens), size=target_length)].tolist()

        # Decode and re-encode to ensure length match
        prompt, _, _ = gen_prompt_decode_to_target_len(
            tokenizer=self.tokenizer,
            token_sequence=tokens,
            target_token_len=target_length,
            add_special_tokens=False,
            rng=self._rng,
        )

        return prompt

    def build_messages(self) -> Iterator[str]:
        """Build random embedding input texts."""
        for _ in range(self.query_parameters.number):
            text = self._generate_random_text()
            yield text


@register_dataset(['embedding', 'embed'])
class EmbeddingDatasetPlugin(DatasetPluginBase):
    """Dataset plugin for embedding model testing from file.

    Supports multiple input formats:
    1. Line-by-line text file: each line is a text to embed
    2. JSON file with list of strings
    3. JSON file with list of objects containing 'text' or 'input' field
    """

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)
        self.texts = []
        self._load_texts()

        if not self.texts:
            raise ValueError(f'No texts loaded from dataset path: {query_parameters.dataset_path}')

    def _load_texts(self):
        """Load texts from dataset file."""
        dataset_path = self.query_parameters.dataset_path

        if not dataset_path:
            logger.warning('No dataset path provided for EmbeddingDatasetPlugin.')
            return

        if not os.path.exists(dataset_path):
            logger.error(f'Dataset file not found: {dataset_path}')
            return

        if dataset_path.endswith('.txt'):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                self.texts = [line.strip() for line in f if line.strip()]
            logger.info(f'Loaded {len(self.texts)} texts from TXT file: {dataset_path}')

        elif dataset_path.endswith('.json'):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        self.texts.append(item)
                    elif isinstance(item, dict):
                        text = item.get('text') or item.get('input') or item.get('sentence') or item.get('content', '')
                        if text:
                            self.texts.append(text)
            logger.info(f'Loaded {len(self.texts)} texts from JSON file: {dataset_path}')

        elif dataset_path.endswith('.jsonl'):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                        if isinstance(item, dict):
                            text = item.get('text') or item.get('input') or item.get('sentence'
                                                                                     ) or item.get('content', '')
                            if text:
                                self.texts.append(text)
                        elif isinstance(item, str):
                            self.texts.append(item)
                    except json.JSONDecodeError:
                        continue
            logger.info(f'Loaded {len(self.texts)} texts from JSONL file: {dataset_path}')

        else:
            raise ValueError(f'Unsupported dataset file format: {dataset_path}, need .txt, .json, or .jsonl')

    def build_messages(self) -> Iterator[str]:
        """Build embedding input texts from loaded file."""
        for text in self.texts:
            is_valid, _ = self.check_prompt_length(text)
            if is_valid:
                yield text


@register_dataset(['random_embedding_batch'])
class RandomEmbeddingBatchDatasetPlugin(RandomEmbeddingDatasetPlugin):
    """Dataset plugin for random batch embedding testing."""

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)
        # Default batch size from extra_args or 8
        self.batch_size = 8
        if query_parameters.extra_args:
            self.batch_size = query_parameters.extra_args.get('batch_size', 8)

    def build_messages(self) -> Iterator[List[str]]:
        """Build batches of random embedding input texts."""
        for _ in range(self.query_parameters.number):
            batch = [self._generate_random_text() for _ in range(self.batch_size)]
            yield batch


@register_dataset(['embedding_batch'])
class EmbeddingBatchDatasetPlugin(EmbeddingDatasetPlugin):
    """Dataset plugin for batch embedding testing from file."""

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)
        # Default batch size from extra_args or 8
        self.batch_size = 8
        if query_parameters.extra_args:
            self.batch_size = query_parameters.extra_args.get('batch_size', 8)

    def build_messages(self) -> Iterator[List[str]]:
        """Build batches of embedding input texts from loaded file."""
        batch = []
        for text in self.texts:
            is_valid, _ = self.check_prompt_length(text)
            if is_valid:
                batch.append(text)
                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []
        # Yield remaining texts
        if batch:
            yield batch
