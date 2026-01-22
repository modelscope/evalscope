"""Rerank dataset plugin for evalscope perf.

This plugin provides datasets suitable for rerank model performance testing.
"""

import json
import numpy as np
import os
from typing import Dict, Iterator, List

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.datasets.utils import gen_prompt_decode_to_target_len
from evalscope.perf.plugin.registry import register_dataset
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_dataset('rerank')
class RerankDatasetPlugin(DatasetPluginBase):
    """Dataset plugin for loading rerank data from files.

    Supports input formats determined by file extension:
    1. .json: List of objects containing 'query' and 'documents' fields
    2. .jsonl / .txt: Each line is a JSON object with query-documents pair

    Expected JSON/JSON Object format:
    {
        "query": "What is machine learning?",
        "documents": ["ML is a subset of AI.", "Deep learning uses neural networks."]
    }
    """

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)
        self.pairs = []
        self._load_pairs()

    def _load_pairs(self):
        """Load query-document pairs from dataset file based on extension."""
        dataset_path = self.query_parameters.dataset_path

        if not dataset_path:
            logger.warning('No dataset path provided for rerank dataset.')
            return

        if not os.path.exists(dataset_path):
            logger.error(f'Dataset file not found: {dataset_path}')
            return

        ext = os.path.splitext(dataset_path)[-1].lower()
        raw_items = []

        try:
            if ext == '.json':
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    if isinstance(content, list):
                        raw_items = content
                    else:
                        logger.error(f'JSON file {dataset_path} must contain a list of objects.')
            elif ext in ['.jsonl', '.txt']:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                raw_items.append(json.loads(line))
                            except json.JSONDecodeError:
                                logger.warning(f'Skipping invalid JSON line in {dataset_path}')
            else:
                logger.warning(
                    f'Unsupported file extension {ext} for dataset {dataset_path}. Supported: .json, .jsonl, .txt'
                )
                return
        except Exception as e:
            logger.error(f'Failed to load dataset from {dataset_path}: {e}')
            return

        # Normalize and store pairs
        for item in raw_items:
            if isinstance(item, dict):
                query = item.get('query', '')
                documents = item.get('documents', [])
                # Also support 'passages' or 'texts' as field names
                if not documents:
                    documents = item.get('passages', []) or item.get('texts', [])
                if query and documents:
                    self.pairs.append({'query': query, 'documents': documents})

        logger.info(f'Loaded {len(self.pairs)} query-document pairs from {dataset_path}')

    def build_messages(self) -> Iterator[Dict]:
        """Build rerank input pairs from loaded file.

        Yields:
            Iterator[Dict]: Dict with 'query' and 'documents' keys.
        """
        for pair in self.pairs:
            yield pair


@register_dataset('random_rerank')
class RandomRerankDatasetPlugin(DatasetPluginBase):
    """Dataset plugin for generating random rerank data.

    Generates random token sequences for stress testing.
    Requires a tokenizer to be provided via --tokenizer-path.
    """

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)

        # Number of documents per query (from extra_args or default 10)
        self.num_documents = 10
        self.document_length_ratio = 5
        if query_parameters.extra_args:
            self.num_documents = query_parameters.extra_args.get('num_documents', 10)
            self.document_length_ratio = query_parameters.extra_args.get('document_length_ratio', 5)

        if not self.tokenizer:
            raise ValueError('Tokenizer is required for random rerank generation. Please provide --tokenizer-path.')

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

        query_len = int(self._rng.integers(min_len, max_len + 1))
        doc_len = int(query_len * self.document_length_ratio)
        if doc_len < 1:
            doc_len = 1

        query = self._generate_token_sequence(query_len)
        documents = [self._generate_token_sequence(doc_len) for _ in range(self.num_documents)]

        return {'query': query, 'documents': documents}

    def build_messages(self) -> Iterator[Dict]:
        """Build random rerank input pairs.

        Yields:
            Iterator[Dict]: Dict with 'query' and 'documents' keys.
        """
        for _ in range(self.query_parameters.number):
            yield self._generate_random_pair()
