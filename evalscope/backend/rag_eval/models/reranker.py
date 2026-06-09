"""Reranker model implementations.

Provides CrossEncoderReranker (local) and APIReranker (OpenAI-compatible rerank API).
Both implement the BaseReranker interface aligned with MTEB 2.x CrossEncoderProtocol.
"""
from __future__ import annotations

import os
import requests
import time
import torch
from torch import Tensor
from typing import Any, Dict, List, Optional, Union

from evalscope.backend.rag_eval.models.base import Array, BaseReranker
from evalscope.backend.rag_eval.models.utils import resolve_model_path
from evalscope.constants import HubType
from evalscope.utils.argument_utils import get_supported_params
from evalscope.utils.logger import get_logger

logger = get_logger()


class CrossEncoderReranker(BaseReranker):
    """Reranker wrapping sentence-transformers CrossEncoder.

    Implements MTEB 2.x CrossEncoderProtocol: predict(), mteb_model_meta.
    """

    def __init__(
        self,
        model_name_or_path: str,
        max_seq_length: int = 512,
        prompt: Optional[str] = None,
        prompts: Optional[Dict[str, str]] = None,
        revision: Optional[str] = 'master',
        hub: str = HubType.MODELSCOPE,
        model_kwargs: Optional[Dict[str, Any]] = None,
        config_kwargs: Optional[Dict[str, Any]] = None,
        encode_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize CrossEncoderReranker.

        Args:
            model_name_or_path: Model name or local path.
            max_seq_length: Maximum sequence length for tokenization.
            prompt: A prompt to prepend to queries.
            prompts: Dict mapping task names to prompts.
            revision: Model revision/branch.
            hub: Download hub ('modelscope' or 'huggingface').
            model_kwargs: Additional kwargs passed to the CrossEncoder automodel.
            config_kwargs: Additional kwargs for model config (unused, kept for interface parity).
            encode_kwargs: Default kwargs for predict() calls.
            **kwargs: Extra keyword arguments (ignored).
        """
        from sentence_transformers.cross_encoder import CrossEncoder

        # Resolve model path (download if needed)
        self.model_name_or_path = resolve_model_path(model_name_or_path, hub=hub, revision=revision)
        self.revision = revision
        self.max_seq_length = max_seq_length
        self.prompt = prompt or ''
        self.prompts = prompts or {}
        self.framework = ['Sentence Transformers', 'PyTorch']

        _model_kwargs = model_kwargs or {}

        self._encode_kwargs: Dict[str, Any] = encode_kwargs or {}
        self._encode_kwargs.setdefault('convert_to_tensor', True)

        # Build the CrossEncoder model
        self.model = CrossEncoder(
            self.model_name_or_path,
            trust_remote_code=True,
            max_length=self.max_seq_length,
            automodel_args=_model_kwargs,
        )

        # Ensure pad token is set
        self.tokenizer = self.model.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if ('pad_token_id' not in self.model.config) or (self.model.config.pad_token_id is None):
            self.model.config.update({'pad_token_id': self.tokenizer.eos_token_id})

        self._supported_predict_params = get_supported_params(self.model.predict)

    def predict(self, inputs1, inputs2=None, **kwargs: Any) -> Array:
        """Predict relevance scores for sentence pairs using CrossEncoder.

        Supports both MTEB 2.x DataLoader[BatchedInput] format (inputs1, inputs2)
        and legacy List[List[str]] format for backward compatibility.

        Args:
            inputs1: DataLoader of query batches (MTEB 2.x) or List[List[str]] (legacy).
            inputs2: DataLoader of document batches (MTEB 2.x), None for legacy format.
            **kwargs: Additional parameters.

        Returns:
            Tensor of relevance scores with shape [n_pairs].
        """
        from torch.utils.data import DataLoader

        # Pop MTEB metadata kwargs that we don't use
        kwargs.pop('task_metadata', None)
        kwargs.pop('hf_split', None)
        kwargs.pop('hf_subset', None)
        kwargs.pop('prompt_type', None)

        # Handle MTEB 2.x DataLoader input
        if isinstance(inputs1, DataLoader):
            queries = [text for batch in inputs1 for text in batch['text']]
            docs = [text for batch in inputs2 for text in batch['text']]
            sentences = list(zip(queries, docs))
        else:
            # Legacy: inputs1 is already List[List[str]]
            sentences = inputs1

        # Filter unsupported kwargs
        for key in list(kwargs.keys()):
            if key not in self._supported_predict_params:
                kwargs.pop(key)

        predict_kwargs = {**self._encode_kwargs, **kwargs}

        # Prepend prompt to queries for retrieval tasks
        if len(sentences) > 0 and len(sentences[0]) == 2 and self.prompt:
            sentences = [(self.prompt + query, doc) for query, doc in sentences]

        embeddings = self.model.predict(sentences, **predict_kwargs)
        assert isinstance(embeddings, Tensor)
        return embeddings


class APIReranker(BaseReranker):
    """Reranker using an OpenAI-compatible rerank API.

    Supports batched inference with configurable batch_size.
    """

    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key: Optional[str] = None,
        prompt: Optional[str] = None,
        prompts: Optional[Dict[str, str]] = None,
        revision: Optional[str] = 'master',
        batch_size: int = 10,
        timeout: int = 60,
        max_seq_length: int = 512,
        **kwargs: Any,
    ) -> None:
        """Initialize APIReranker.

        Args:
            model_name: Model identifier for the API.
            api_base: Base URL for the rerank API endpoint.
            api_key: API key for authentication.
            prompt: A prompt to prepend to queries.
            prompts: Dict mapping task names to prompts.
            revision: Model revision (metadata only for API models).
            batch_size: Number of documents per API request.
            timeout: Request timeout in seconds.
            max_seq_length: Maximum sequence length; texts exceeding this (in estimated tokens) are truncated.
            **kwargs: Extra keyword arguments (ignored).
        """
        if not api_base:
            raise ValueError('api_base is required for API reranker model.')

        self.model_name_or_path = model_name
        self.model_name = model_name
        self.revision = revision
        self.prompt = prompt or ''
        self.prompts = prompts or {}
        self.framework = ['API']
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_seq_length = max_seq_length
        self._max_chars = max_seq_length * 3

        # Build the rerank URL (default /rerank, DashScope uses /reranks)
        self.rerank_url = api_base.rstrip('/')
        if not self.rerank_url.endswith(('/rerank', '/reranks')):
            if 'dashscope.aliyuncs.com' in self.rerank_url:
                self.rerank_url = f'{self.rerank_url}/reranks'
            else:
                self.rerank_url = f'{self.rerank_url}/rerank'

        # Set up headers
        self.headers: Dict[str, str] = {'Content-Type': 'application/json'}
        resolved_api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not resolved_api_key and 'dashscope.aliyuncs.com' in self.rerank_url:
            resolved_api_key = os.getenv('DASHSCOPE_API_KEY')
        if resolved_api_key:
            self.headers['Authorization'] = f'Bearer {resolved_api_key}'

        self.session = requests.Session()

    def _truncate_text(self, text: str) -> str:
        """Truncate text that exceeds max_seq_length (estimated by character count)."""
        if self.max_seq_length > 0 and len(text) > self._max_chars:
            self._truncated_count += 1
            return text[:self._max_chars]
        return text

    def predict(self, inputs1, inputs2=None, **kwargs: Any) -> Array:
        """Predict relevance scores via the rerank API.

        Supports both MTEB 2.x DataLoader[BatchedInput] format and legacy format.

        Args:
            inputs1: DataLoader of query batches (MTEB 2.x) or List[List[str]] (legacy).
            inputs2: DataLoader of document batches (MTEB 2.x), None for legacy format.
            **kwargs: Additional parameters (batch_size override, etc.).

        Returns:
            Tensor of relevance scores with shape [n_pairs].
        """
        from torch.utils.data import DataLoader

        # Pop MTEB metadata kwargs
        kwargs.pop('task_metadata', None)
        kwargs.pop('hf_split', None)
        kwargs.pop('hf_subset', None)
        kwargs.pop('prompt_type', None)

        # Handle MTEB 2.x DataLoader input
        if isinstance(inputs1, DataLoader):
            if not isinstance(inputs2, DataLoader):
                raise ValueError(
                    'APIReranker.predict: when inputs1 is a DataLoader (MTEB 2.x), '
                    'inputs2 must also be a DataLoader of documents.'
                )
            queries = [text for batch in inputs1 for text in batch['text']]
            docs = [text for batch in inputs2 for text in batch['text']]
            sentences = list(zip(queries, docs))
        else:
            sentences = inputs1

        batch_size = kwargs.pop('batch_size', self.batch_size)

        if not sentences:
            return torch.tensor([])

        scores: List[float] = [0.0] * len(sentences)
        grouped_sentences: Dict[str, List] = {}
        prompt = self.prompt
        self._truncated_count = 0

        # Group by query
        for idx, sentence in enumerate(sentences):
            if len(sentence) == 2:
                query, document = sentence
            elif len(sentence) == 3:
                query, document, instruction = sentence
                if instruction is not None:
                    query = f'{query} {instruction}'.strip()
            else:
                raise ValueError('API reranker expects query-document pairs (2 or 3 elements).')
            query = self._truncate_text(prompt + query)
            document = self._truncate_text(document)
            grouped_sentences.setdefault(query, []).append((idx, document))

        if self._truncated_count:
            logger.warning(
                f'Truncated {self._truncated_count} texts to {self._max_chars} chars '
                f'(max_seq_length={self.max_seq_length}).'
            )

        # Process each query group in batches
        max_retries = 3
        for query, pairs in grouped_sentences.items():
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                documents = [doc for _, doc in batch_pairs]
                payload = {
                    'model': self.model_name,
                    'query': query,
                    'documents': documents,
                    'top_n': len(documents),
                    'return_documents': False,
                }
                response = None
                for attempt in range(max_retries):
                    try:
                        response = self.session.post(
                            self.rerank_url, headers=self.headers, json=payload, timeout=self.timeout
                        )
                    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                        if attempt < max_retries - 1:
                            wait_time = 2**attempt
                            logger.warning(
                                f'Rerank API request failed: {e}, retrying in {wait_time}s '
                                f'(attempt {attempt + 1}/{max_retries})'
                            )
                            time.sleep(wait_time)
                            continue
                        raise
                    if response.status_code < 500:
                        break
                    if attempt < max_retries - 1:
                        wait_time = 2**attempt
                        logger.warning(
                            f'Rerank API returned {response.status_code}, retrying in {wait_time}s '
                            f'(attempt {attempt + 1}/{max_retries})'
                        )
                        time.sleep(wait_time)
                response.raise_for_status()
                data = response.json()
                results = data.get('results')
                if results is None:
                    raise ValueError(f'Invalid rerank response: {data}')

                for result_idx, result in enumerate(results):
                    doc_idx = result.get('index')
                    if doc_idx is None:
                        doc_idx = result_idx
                    else:
                        try:
                            doc_idx = int(doc_idx)
                        except (TypeError, ValueError):
                            logger.warning(f'Rerank result has non-integer index: {result.get("index")}, skipping.')
                            continue
                    if doc_idx < 0 or doc_idx >= len(batch_pairs):
                        logger.warning(f'Rerank result index {doc_idx} out of range [0, {len(batch_pairs)}), skipping.')
                        continue
                    score = result.get('relevance_score', result.get('score', 0.0))
                    scores[batch_pairs[doc_idx][0]] = score

        return torch.tensor(scores)
