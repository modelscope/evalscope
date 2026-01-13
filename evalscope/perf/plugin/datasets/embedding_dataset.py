"""Embedding dataset plugin for evalscope perf.

This plugin provides datasets suitable for embedding model performance testing.
"""

import json
import random
from typing import Dict, Iterator, List, Union

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.registry import register_dataset
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_dataset(['random_embedding', 'embedding', 'embed'])
class RandomEmbeddingDatasetPlugin(DatasetPluginBase):
    """Dataset plugin for embedding model testing.

    Supports multiple input formats:
    1. Line-by-line text file: each line is a text to embed
    2. JSON file with list of strings
    3. JSON file with list of objects containing 'text' or 'input' field
    4. Random text generation for stress testing (default)

    If no dataset_path is provided, generates random texts.
    """

    # Sample texts for random generation
    SAMPLE_TEXTS = [
        'The quick brown fox jumps over the lazy dog.',
        'Machine learning is a subset of artificial intelligence.',
        'Natural language processing enables computers to understand human language.',
        'Deep learning models can learn complex patterns from data.',
        'Embeddings are dense vector representations of text.',
        'Semantic search uses embeddings to find similar documents.',
        'Transformers have revolutionized natural language processing.',
        'BERT and GPT are popular language models.',
        'Vector databases store and retrieve embeddings efficiently.',
        'Cosine similarity measures the angle between two vectors.',
        'Neural networks are inspired by the human brain structure.',
        'Attention mechanisms allow models to focus on relevant parts.',
        'Pre-training on large corpora improves model performance.',
        'Fine-tuning adapts models to specific downstream tasks.',
        'Tokenization splits text into smaller units for processing.',
    ]

    # Reasonable defaults for embedding models (most have 512 token limit)
    DEFAULT_MIN_LENGTH = 10
    DEFAULT_MAX_LENGTH = 256

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)
        self.texts = []
        self._load_texts()

    def _load_texts(self):
        """Load texts from dataset file or use random generation."""
        dataset_path = self.query_parameters.dataset_path

        if dataset_path:
            try:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()

                # Try to parse as JSON first
                try:
                    data = json.loads(content)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, str):
                                self.texts.append(item)
                            elif isinstance(item, dict):
                                # Support various field names
                                text = item.get('text') or item.get('input') or item.get('sentence'
                                                                                         ) or item.get('content', '')
                                if text:
                                    self.texts.append(text)
                    logger.info(f'Loaded {len(self.texts)} texts from JSON file: {dataset_path}')
                except json.JSONDecodeError:
                    # Treat as line-by-line text file
                    self.texts = [line.strip() for line in content.split('\n') if line.strip()]
                    logger.info(f'Loaded {len(self.texts)} texts from line-by-line file: {dataset_path}')

            except Exception as e:
                logger.warning(f'Failed to load dataset from {dataset_path}: {e}. Using random texts.')
                self.texts = []

        if not self.texts:
            # Generate random texts for testing
            logger.info('No dataset provided, generating random texts for embedding testing.')

    def _generate_random_text(self) -> str:
        """Generate a random text of reasonable length for embedding."""
        # Use reasonable defaults, ignore the potentially huge max_prompt_length
        min_len = max(self.DEFAULT_MIN_LENGTH, self.query_parameters.min_prompt_length)
        max_len = min(self.DEFAULT_MAX_LENGTH, self.query_parameters.max_prompt_length)

        # Ensure min <= max
        if min_len > max_len:
            min_len, max_len = max_len, min_len

        # Start with a random sentence
        text = random.choice(self.SAMPLE_TEXTS)

        # Extend if needed
        target_length = random.randint(min_len, max_len)
        attempts = 0
        while len(text) < target_length and attempts < 20:
            text += ' ' + random.choice(self.SAMPLE_TEXTS)
            attempts += 1

        return text[:max_len] if len(text) > max_len else text

    def build_messages(self) -> Iterator[Union[str, List[str]]]:
        """Build embedding input texts.

        Yields:
            Iterator[str]: Text strings for embedding.
        """
        if self.texts:
            # Use loaded texts
            for text in self.texts:
                is_valid, _ = self.check_prompt_length(text)
                if is_valid:
                    yield text
        else:
            # Generate random texts - yield immediately without pre-generating
            count = 0
            max_count = 100000  # Large but finite limit
            while count < max_count:
                text = self._generate_random_text()
                yield text
                count += 1


@register_dataset('random_embedding_batch')
class RandomEmbeddingBatchDatasetPlugin(RandomEmbeddingDatasetPlugin):
    """Dataset plugin for batch embedding testing.

    Similar to RandomEmbeddingDatasetPlugin but yields batches of texts.
    """

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)
        # Default batch size from extra_args or 8
        self.batch_size = 8
        if query_parameters.extra_args:
            self.batch_size = query_parameters.extra_args.get('batch_size', 8)

    def build_messages(self) -> Iterator[List[str]]:
        """Build batches of embedding input texts.

        Yields:
            Iterator[List[str]]: Batches of text strings for embedding.
        """
        batch = []

        if self.texts:
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
        else:
            # Generate random batches
            count = 0
            max_count = 10000  # Number of batches
            while count < max_count:
                batch = [self._generate_random_text() for _ in range(self.batch_size)]
                yield batch
                count += 1
