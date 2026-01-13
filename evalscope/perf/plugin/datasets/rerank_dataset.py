"""Rerank dataset plugin for evalscope perf.

This plugin provides datasets suitable for rerank model performance testing.
"""

import json
import numpy as np
import random
from typing import Dict, Iterator, List

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.datasets.utils import gen_prompt_decode_to_target_len
from evalscope.perf.plugin.registry import register_dataset
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_dataset(['random_rerank', 'rerank'])
class RandomRerankDatasetPlugin(DatasetPluginBase):
    """Dataset plugin for rerank model testing.

    Supports multiple input formats:
    1. JSON file with list of objects containing 'query' and 'documents' fields
    2. JSONL file where each line is a query-documents pair
    3. Random generation for stress testing (default)

    Expected JSON format:
    [
        {
            "query": "What is machine learning?",
            "documents": ["ML is a subset of AI.", "Deep learning uses neural networks."]
        },
        ...
    ]

    If no dataset_path is provided, generates random query-document pairs.
    """

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)
        self.pairs = []
        self._load_pairs()

        # Number of documents per query (from extra_args or default 10)
        self.num_documents = 10
        self.document_length_ratio = 5
        if query_parameters.extra_args:
            self.num_documents = query_parameters.extra_args.get('num_documents', 10)
            self.document_length_ratio = query_parameters.extra_args.get('document_length_ratio', 5)

        if not self.pairs:
            if not self.tokenizer:
                raise ValueError(
                    'Tokenizer is required for random rerank generation when no dataset path is provided. Please provide --tokenizer-path.'  # noqa: E501
                )

            # Use numpy's default_rng for sampling
            self._rng = np.random.default_rng(None)

            # Filter out special tokens from vocabulary
            vocab_size = self.tokenizer.vocab_size
            prohibited_tokens = set(self.tokenizer.all_special_ids)
            all_tokens = np.arange(vocab_size)
            self.allowed_tokens = np.array(list(set(all_tokens) - prohibited_tokens))
            logger.info(
                f'Using {len(self.allowed_tokens)} allowed tokens out of {vocab_size} total tokens for random generation.'  # noqa: E501
            )

    def _load_pairs(self):
        """Load query-document pairs from dataset file."""
        dataset_path = self.query_parameters.dataset_path

        if dataset_path:
            try:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()

                # Try to parse as JSON
                try:
                    data = json.loads(content)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                query = item.get('query', '')
                                documents = item.get('documents', [])
                                # Also support 'passages' or 'texts' as field names
                                if not documents:
                                    documents = item.get('passages', []) or item.get('texts', [])
                                if query and documents:
                                    self.pairs.append({'query': query, 'documents': documents})
                    logger.info(f'Loaded {len(self.pairs)} query-document pairs from JSON: {dataset_path}')
                except json.JSONDecodeError:
                    # Try JSONL format
                    for line in content.split('\n'):
                        line = line.strip()
                        if line:
                            try:
                                item = json.loads(line)
                                query = item.get('query', '')
                                documents = item.get('documents', []) or item.get('passages', [])
                                if query and documents:
                                    self.pairs.append({'query': query, 'documents': documents})
                            except json.JSONDecodeError:
                                continue
                    logger.info(f'Loaded {len(self.pairs)} query-document pairs from JSONL: {dataset_path}')

            except Exception as e:
                logger.warning(f'Failed to load dataset from {dataset_path}: {e}. Using random pairs.')
                self.pairs = []

        if not self.pairs:
            logger.info('No dataset provided, generating random query-document pairs for rerank testing.')

    def _generate_token_sequence(self, length: int) -> str:
        """Generate a random string with specific token length."""
        tokens = self.allowed_tokens[self._rng.integers(0, len(self.allowed_tokens), size=length)].tolist()
        prompt, _, _ = gen_prompt_decode_to_target_len(
            tokenizer=self.tokenizer,
            token_sequence=tokens,
            target_token_len=length,
            add_special_tokens=False,
            rng=self._rng,
        )
        return prompt

    def _generate_random_pair(self) -> Dict:
        """Generate a random query-document pair."""
        # Calculate lengths
        min_len = self.query_parameters.min_prompt_length
        max_len = self.query_parameters.max_prompt_length
        if min_len > max_len:
            min_len, max_len = max_len, min_len
        min_len = max(1, min_len)
        max_len = max(1, max_len)

        query_len = self._rng.integers(min_len, max_len + 1)
        doc_len = int(query_len * self.document_length_ratio)
        if doc_len < 1:
            doc_len = 1

        query = self._generate_token_sequence(query_len)
        documents = [self._generate_token_sequence(doc_len) for _ in range(self.num_documents)]

        return {'query': query, 'documents': documents}

    def build_messages(self) -> Iterator[Dict]:
        """Build rerank input pairs.

        Yields:
            Iterator[Dict]: Dict with 'query' and 'documents' keys.
        """
        if self.pairs:
            for pair in self.pairs:
                yield pair
        else:
            # Generate random pairs
            for _ in range(self.query_parameters.number):
                yield self._generate_random_pair()
