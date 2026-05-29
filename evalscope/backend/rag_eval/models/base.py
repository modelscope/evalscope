"""Abstract base classes for encoder and reranker models.

These protocols align with MTEB 2.x (EncoderProtocol, CrossEncoderProtocol) without
importing from mteb directly (since it is an optional dependency).
"""
from __future__ import annotations

import numpy as np
import os
import torch
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from torch import Tensor
from typing import Any, Dict, List, Optional, Union

# Type alias matching MTEB's Array type
Array = Union[Tensor, NDArray[np.float32]]


class BaseEncoder(ABC):
    """Abstract base class for embedding/encoder models.

    Aligns with MTEB 2.x EncoderProtocol: encode(), similarity(), similarity_pairwise(), mteb_model_meta.
    """

    model_name_or_path: str
    revision: Optional[str]
    framework: List[str]

    @abstractmethod
    def encode(self, texts: Union[str, List[str]], **kwargs: Any) -> Array:
        """Encode texts into embeddings.

        Args:
            texts: Single text or list of texts to encode.
            **kwargs: Additional encoding parameters.

        Returns:
            Numpy array or torch tensor of shape (n_texts, embed_dim).
        """
        ...

    def similarity(self, embeddings1: Array, embeddings2: Array) -> Array:
        """Compute pairwise cosine similarity matrix between two collections.

        Args:
            embeddings1: Shape [num_embeddings_1, embedding_dim].
            embeddings2: Shape [num_embeddings_2, embedding_dim].

        Returns:
            Similarity matrix of shape [num_embeddings_1, num_embeddings_2].
        """
        e1 = torch.as_tensor(embeddings1, dtype=torch.float32)
        e2 = torch.as_tensor(embeddings2, dtype=torch.float32)
        e1_norm = torch.nn.functional.normalize(e1, p=2, dim=-1)
        e2_norm = torch.nn.functional.normalize(e2, p=2, dim=-1)
        return e1_norm @ e2_norm.T

    def similarity_pairwise(self, embeddings1: Array, embeddings2: Array) -> Array:
        """Compute element-wise cosine similarity between corresponding pairs.

        Args:
            embeddings1: Shape [num_embeddings, embedding_dim].
            embeddings2: Shape [num_embeddings, embedding_dim].

        Returns:
            Similarity scores of shape [num_embeddings].
        """
        e1 = torch.as_tensor(embeddings1, dtype=torch.float32)
        e2 = torch.as_tensor(embeddings2, dtype=torch.float32)
        e1_norm = torch.nn.functional.normalize(e1, p=2, dim=-1)
        e2_norm = torch.nn.functional.normalize(e2, p=2, dim=-1)
        return (e1_norm * e2_norm).sum(dim=-1)

    @property
    def mteb_model_meta(self) -> Any:
        """Model metadata for MTEB. Returns a ModelMeta instance from mteb."""
        from mteb.models.model_meta import ModelMeta

        return ModelMeta(
            loader=None,
            name='eval/' + os.path.basename(self.model_name_or_path),
            revision=self.revision,
            languages=None,
            release_date=None,
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=None,
            embed_dim=None,
            license=None,
            open_weights=None,
            public_training_code=None,
            public_training_data=None,
            similarity_fn_name=None,
            use_instructions=None,
            training_datasets=None,
            framework=self.framework,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs. Compatible with langchain Embeddings interface.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings as float lists.
        """
        result = self.encode(texts)
        return result.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed query text. Compatible with langchain Embeddings interface.

        Args:
            text: Text to embed.

        Returns:
            Embedding as float list.
        """
        result = self.encode(text)
        return result.tolist()

    def get_prompt(self, task_name: str) -> Optional[str]:
        """Get prompt for the given task name.

        Args:
            task_name: Name of the task.

        Returns:
            Prompt string or None.
        """
        if hasattr(self, 'prompt') and self.prompt:
            return self.prompt
        if hasattr(self, 'prompts') and self.prompts:
            return self.prompts.get(task_name, None)
        return None


class BaseReranker(ABC):
    """Abstract base class for reranker/cross-encoder models.

    Aligns with MTEB 2.x CrossEncoderProtocol: predict(), mteb_model_meta.
    """

    model_name_or_path: str
    revision: Optional[str]
    framework: List[str]

    @abstractmethod
    def predict(self, sentences: List[List[str]], **kwargs: Any) -> Array:
        """Predict relevance scores for sentence pairs.

        Args:
            sentences: List of [query, document] or [query, document, instruction] pairs.
            **kwargs: Additional prediction parameters.

        Returns:
            Tensor of relevance scores with shape [n_pairs].
        """
        ...

    @property
    def mteb_model_meta(self) -> Any:
        """Model metadata for MTEB. Returns a ModelMeta instance from mteb."""
        from mteb.models.model_meta import ModelMeta

        return ModelMeta(
            loader=None,
            name='eval/' + os.path.basename(self.model_name_or_path),
            revision=self.revision,
            languages=None,
            release_date=None,
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=None,
            embed_dim=None,
            license=None,
            open_weights=None,
            public_training_code=None,
            public_training_data=None,
            similarity_fn_name=None,
            use_instructions=None,
            training_datasets=None,
            framework=self.framework,
        )
