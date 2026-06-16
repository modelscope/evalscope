"""Encoder model implementations.

Provides SentenceTransformerEncoder (local) and APIEncoder (OpenAI-compatible API).
Both implement the BaseEncoder interface aligned with MTEB 2.x EncoderProtocol.
"""
from __future__ import annotations

import torch
from torch import Tensor
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Union

from evalscope.backend.rag_eval.models.base import Array, BaseEncoder
from evalscope.backend.rag_eval.models.utils import resolve_model_path
from evalscope.constants import HubType
from evalscope.utils.argument_utils import get_supported_params
from evalscope.utils.logger import get_logger

logger = get_logger()

try:
    from mteb.types import PromptType
    _PROMPT_TYPE_QUERY = PromptType.query
except ImportError:
    _PROMPT_TYPE_QUERY = 'query'


class SentenceTransformerEncoder(BaseEncoder):
    """Encoder wrapping sentence-transformers SentenceTransformer.

    Implements MTEB 2.x EncoderProtocol: encode(), similarity(), similarity_pairwise(), mteb_model_meta.
    """

    def __init__(
        self,
        model_name_or_path: str,
        pooling_mode: Optional[str] = None,
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
        """Initialize SentenceTransformerEncoder.

        Args:
            model_name_or_path: Model name or local path.
            pooling_mode: Pooling mode override (cls, mean, max, etc.). None uses model default.
            max_seq_length: Maximum sequence length for tokenization.
            prompt: A single prompt to prepend to queries.
            prompts: Dict mapping task names to prompts.
            revision: Model revision/branch.
            hub: Download hub ('modelscope' or 'huggingface').
            model_kwargs: Additional kwargs passed to the transformer model.
            config_kwargs: Additional kwargs for model config.
            encode_kwargs: Default kwargs for encode() calls.
            **kwargs: Extra keyword arguments (ignored).
        """
        from sentence_transformers import models
        from sentence_transformers.SentenceTransformer import SentenceTransformer

        # Resolve model path (download if needed)
        self.model_name_or_path = resolve_model_path(model_name_or_path, hub=hub, revision=revision)
        self.revision = revision
        self.max_seq_length = max_seq_length
        self.prompt = prompt
        self.prompts = prompts or {}
        self.framework = ['Sentence Transformers', 'PyTorch']

        _model_kwargs = model_kwargs or {}
        _model_kwargs['trust_remote_code'] = True

        _config_kwargs = config_kwargs or {}
        _config_kwargs['trust_remote_code'] = True

        self._encode_kwargs: Dict[str, Any] = encode_kwargs or {}
        self._encode_kwargs.setdefault('convert_to_tensor', True)

        # Build the SentenceTransformer model
        if not pooling_mode:
            self.model = SentenceTransformer(
                self.model_name_or_path,
                config_kwargs=_config_kwargs,
                model_kwargs=_model_kwargs,
            )
        else:
            word_embedding_model = models.Transformer(
                self.model_name_or_path,
                config_args=_config_kwargs,
                model_args=_model_kwargs,
            )
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(),
                pooling_mode=pooling_mode,
            )
            self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        self.model.max_seq_length = self.max_seq_length
        self._supported_encode_params = get_supported_params(self.model.encode)

    def encode(self, inputs, **kwargs: Any) -> Array:
        """Encode texts into embeddings using SentenceTransformer.

        Supports both MTEB 2.x DataLoader[BatchedInput] format and legacy
        str/List[str] format for backward compatibility.

        Args:
            inputs: DataLoader of batched inputs (MTEB 2.x) or str/list[str] (legacy).
            **kwargs: Additional parameters (task_metadata, prompt_type, hf_split, etc.).

        Returns:
            Torch tensor of shape (n_texts, embed_dim) on CPU.
        """
        from torch.utils.data import DataLoader

        # Extract texts from DataLoader (MTEB 2.x API) or use directly
        if isinstance(inputs, DataLoader):
            texts = [text for batch in inputs for text in batch['text']]
        elif isinstance(inputs, str):
            texts = [inputs]
        else:
            texts = list(inputs)

        # Extract prompt_type and task_metadata for prompt resolution
        prompt = None
        prompt_type = kwargs.pop('prompt_type', None)
        task_metadata = kwargs.pop('task_metadata', None)
        kwargs.pop('hf_split', None)
        kwargs.pop('hf_subset', None)

        if prompt_type == _PROMPT_TYPE_QUERY:
            task_name = getattr(task_metadata, 'name', '') if task_metadata else ''
            prompt = self.get_prompt(task_name)

        # Filter unsupported params for ST encode
        encode_kwargs = {**self._encode_kwargs}
        for key, val in kwargs.items():
            if key in self._supported_encode_params:
                encode_kwargs[key] = val

        embeddings = self.model.encode(texts, prompt=prompt, **encode_kwargs)
        assert isinstance(embeddings, Tensor)
        return embeddings.cpu().detach()

    def similarity(self, embeddings1: Array, embeddings2: Array) -> Array:
        """Compute similarity using SentenceTransformer's built-in method if available."""
        if hasattr(self.model, 'similarity') and callable(self.model.similarity):
            return self.model.similarity(embeddings1, embeddings2)
        return super().similarity(embeddings1, embeddings2)

    def similarity_pairwise(self, embeddings1: Array, embeddings2: Array) -> Array:
        """Compute pairwise similarity using SentenceTransformer's built-in method if available."""
        if hasattr(self.model, 'similarity_pairwise') and callable(self.model.similarity_pairwise):
            return self.model.similarity_pairwise(embeddings1, embeddings2)
        return super().similarity_pairwise(embeddings1, embeddings2)


class APIEncoder(BaseEncoder):
    """Encoder using an OpenAI-compatible embedding API.

    Implements batched inference with configurable batch_size. Returns numpy arrays.
    """

    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
        max_seq_length: int = 512,
        prompt: Optional[str] = None,
        prompts: Optional[Dict[str, str]] = None,
        revision: Optional[str] = 'master',
        batch_size: int = 10,
        check_embedding_ctx_length: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize APIEncoder.

        Args:
            model_name: Model identifier for the API.
            api_base: Base URL for the OpenAI-compatible API.
            api_key: API key for authentication.
            dimensions: Embedding dimensions (if supported by API).
            max_seq_length: Maximum sequence length; texts exceeding this (in estimated tokens) are truncated.
            prompt: A single prompt to prepend to queries.
            prompts: Dict mapping task names to prompts.
            revision: Model revision (metadata only for API models).
            batch_size: Number of texts per API request.
            check_embedding_ctx_length: Whether to check embedding context length.
            **kwargs: Extra keyword arguments (ignored).
        """
        from langchain_openai.embeddings import OpenAIEmbeddings

        self.model_name_or_path = model_name
        self.revision = revision
        self.prompt = prompt
        self.prompts = prompts or {}
        self.framework = ['API']
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

        # Conservative estimate: 1 token ≈ 3 characters (reasonable for mixed CJK/Latin text).
        self._max_chars = self.max_seq_length * 3

        self._client = OpenAIEmbeddings(
            model=model_name,
            base_url=api_base,
            api_key=api_key,
            dimensions=dimensions,
            check_embedding_ctx_length=check_embedding_ctx_length,
        )
        self._supported_encode_params = get_supported_params(self._client.embed_documents)

    def _truncate_texts(self, texts: List[str]) -> List[str]:
        """Truncate texts that exceed max_seq_length (estimated by character count)."""
        if self.max_seq_length <= 0:
            return texts
        truncated = []
        truncated_count = 0
        for text in texts:
            if len(text) > self._max_chars:
                truncated_count += 1
                text = text[:self._max_chars]
            truncated.append(text)
        if truncated_count:
            logger.warning(
                f'Truncated {truncated_count}/{len(texts)} texts to {self._max_chars} chars '
                f'(max_seq_length={self.max_seq_length}).'
            )
        return truncated

    def encode(self, inputs, **kwargs: Any) -> Array:
        """Encode texts into embeddings using API.

        Supports both MTEB 2.x DataLoader[BatchedInput] format and legacy
        str/List[str] format for backward compatibility.

        Args:
            inputs: DataLoader of batched inputs (MTEB 2.x) or str/list[str] (legacy).
            **kwargs: Additional parameters (task_metadata, prompt_type, hf_split, etc.).

        Returns:
            Torch tensor of shape (n_texts, embed_dim).
        """
        from torch.utils.data import DataLoader

        # Extract texts from DataLoader (MTEB 2.x API) or use directly
        if isinstance(inputs, DataLoader):
            texts = [text for batch in inputs for text in batch['text']]
        elif isinstance(inputs, str):
            texts = [inputs]
        else:
            texts = list(inputs)

        # Extract MTEB metadata before filtering
        prompt_type = kwargs.pop('prompt_type', None)
        task_metadata = kwargs.pop('task_metadata', None)
        kwargs.pop('hf_split', None)
        kwargs.pop('hf_subset', None)

        # Resolve prompt based on prompt_type
        prompt = None
        if prompt_type == _PROMPT_TYPE_QUERY:
            task_name = getattr(task_metadata, 'name', '') if task_metadata else ''
            prompt = self.get_prompt(task_name)

        # Filter and pass through supported encode parameters
        encode_kwargs: Dict[str, Any] = {}
        for key, val in kwargs.items():
            if key in self._supported_encode_params:
                encode_kwargs[key] = val

        embeddings: List[List[float]] = []
        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch_texts = texts[i:i + self.batch_size]
            if prompt is not None:
                batch_texts = [prompt + text for text in batch_texts]
            batch_texts = self._truncate_texts(batch_texts)
            response = self._client.embed_documents(batch_texts, chunk_size=self.batch_size, **encode_kwargs)
            embeddings.extend(response)
        return torch.tensor(embeddings)
