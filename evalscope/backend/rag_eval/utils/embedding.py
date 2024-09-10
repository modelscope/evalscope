import os
from typing import List, Optional, Dict, Any
from langchain_core.embeddings import Embeddings
from sentence_transformers import models
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from torch import Tensor
from evalscope.utils.logger import get_logger

logger = get_logger()


class EmbeddingModel(Embeddings):
    """Custom embeddings"""

    def __init__(
        self,
        model_name_or_path: str = "",
        is_cross_encoder: bool = False,
        pooling_mode: Optional[str] = None,
        model_kwargs: Dict[str, Any] = None,
        encode_kwargs: Dict[str, Any] = None,
        hub: str = "modelscope",
    ):
        # Initialize model name or path
        self.model_name_or_path = model_name_or_path
        # Initialize whether it is a cross encoder
        self.is_cross_encoder = is_cross_encoder
        # Initialize pooling mode for sentence embeddings
        self.pooling_mode = pooling_mode
        # Initialize model keyword arguments
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.model_kwargs["trust_remote_code"] = True

        # Initialize encoding keyword arguments
        self.encode_kwargs = encode_kwargs if encode_kwargs is not None else {}
        self.encode_kwargs["convert_to_tensor"] = True

        # Get and initialize the model
        self.model = self._get_model(hub=hub)

        # Update model information
        self.update_model_info()

    @property
    def mteb_model_meta(self):
        """Model metadata for MTEB (Multilingual Task Embeddings Benchmark)"""
        from mteb import ModelMeta

        # Return model metadata
        return ModelMeta(
            name=self.model_name,
            revision=self.model_revision,
            languages=None,
            release_date=None,
        )

    def update_model_info(self):
        """Update model information from model card"""
        model_card = self.model.model_card_data

        # Update model name
        self.model_name = model_card.model_name or os.path.basename(
            self.model_name_or_path
        )

        # Update model revision
        self.model_revision = model_card.base_model_revision or "v1"

    def _get_model(self, hub):
        """Load model based on model name or path

        Args:
            hub (str): Model hub, default is "modelscope"

        Returns:
            Model instance
        """
        # If model path does not exist and hub is modelscope, download model
        if not os.path.exists(self.model_name_or_path) and hub == "modelscope":
            from modelscope import snapshot_download

            logger.info(f"Loading model {self.model_name_or_path} from modelscope")
            self.model_name_or_path = snapshot_download(self.model_name_or_path)

        # Return different model instances based on whether it is a cross encoder and pooling mode
        if self.is_cross_encoder:
            return CrossEncoder(self.model_name_or_path, **self.model_kwargs)

        if not self.pooling_mode:
            return SentenceTransformer(self.model_name_or_path, **self.model_kwargs)
        else:
            word_embedding_model = models.Transformer(self.model_name_or_path)
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(),
                pooling_mode=self.pooling_mode,
            )

            return SentenceTransformer(
                modules=[word_embedding_model, pooling_model], **self.model_kwargs
            )

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text

        Args:
            text (str): Query text

        Returns:
            List[float]: Embedding vector
        """
        return self.encode([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document texts

        Args:
            texts (List[str]): List of document texts

        Returns:
            List[List[float]]: List of embedding vectors
        """
        return self.encode(texts)

    def encode(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Encode texts into embeddings

        Args:
            texts (List[str]): List of texts to encode
            **kwargs: Additional encoding arguments

        Returns:
            List[List[float]]: List of embedding vectors

        Raises:
            ValueError: If model type is unknown
        """
        # Update encoding arguments
        self.encode_kwargs.update(kwargs)

        # Encode based on model type
        if isinstance(self.model, SentenceTransformer):
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                **self.encode_kwargs,
            )
        elif isinstance(self.model, CrossEncoder):
            embeddings = self.model.predict(texts, **self.encode_kwargs)
        else:
            raise ValueError("Unknown model type")

        # Ensure embeddings is a tensor
        assert isinstance(embeddings, Tensor)
        return embeddings