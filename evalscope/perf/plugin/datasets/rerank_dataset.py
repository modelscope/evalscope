"""Rerank dataset plugin for evalscope perf.

This plugin provides datasets suitable for rerank model performance testing.
"""

import json
import random
from typing import Dict, Iterator, List

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
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

    # Sample queries and documents for random generation
    SAMPLE_QUERIES = [
        'What is machine learning?',
        'How does natural language processing work?',
        'What are embeddings used for?',
        'Explain deep learning architectures.',
        'What is semantic search?',
        'How do transformers work?',
        'What is the difference between BERT and GPT?',
        'How to implement vector search?',
        'What is retrieval augmented generation?',
        'Explain attention mechanism in transformers.',
    ]

    SAMPLE_DOCUMENTS = [
        'Machine learning is a subset of artificial intelligence that enables systems to learn from data.',
        'Natural language processing (NLP) is a field of AI focused on the interaction between computers and humans through language.',
        'Embeddings are dense vector representations that capture semantic meaning of text.',
        'Deep learning uses neural networks with multiple layers to learn hierarchical representations.',
        'Semantic search goes beyond keyword matching to understand the intent and context of queries.',
        'Transformers are neural network architectures that use self-attention mechanisms.',
        'BERT is a bidirectional encoder while GPT is an autoregressive decoder model.',
        'Vector search enables similarity-based retrieval using embedding representations.',
        'RAG combines retrieval systems with language models to generate grounded responses.',
        'Attention mechanisms allow models to focus on relevant parts of the input sequence.',
        'Convolutional neural networks are primarily used for image processing tasks.',
        'Recurrent neural networks process sequential data by maintaining hidden states.',
        'The softmax function converts logits into probability distributions.',
        'Gradient descent optimizes model parameters by minimizing the loss function.',
        'Transfer learning allows models to leverage knowledge from pre-training.',
    ]

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)
        self.pairs = []
        self._load_pairs()

        # Number of documents per query (from extra_args or default 10)
        self.num_documents = 10
        if query_parameters.extra_args:
            self.num_documents = query_parameters.extra_args.get('num_documents', 10)

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

    def _generate_random_pair(self) -> Dict:
        """Generate a random query-document pair."""
        query = random.choice(self.SAMPLE_QUERIES)
        # Select random documents, ensuring some variety
        num_docs = min(self.num_documents, len(self.SAMPLE_DOCUMENTS))
        documents = random.sample(self.SAMPLE_DOCUMENTS, num_docs)
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
            count = 0
            max_count = 100000  # Large but finite limit
            while count < max_count:
                yield self._generate_random_pair()
                count += 1
