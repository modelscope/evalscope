import os
import torch
from langchain_core.embeddings import Embeddings
from sentence_transformers import models
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.SentenceTransformer import SentenceTransformer
from torch import Tensor
from typing import Dict, List, Optional, Union

from evalscope.backend.rag_eval.utils.tools import download_model
from evalscope.constants import HubType
from evalscope.utils.logger import get_logger

logger = get_logger()


class BaseModel(Embeddings):

    def __init__(
        self,
        model_name_or_path: str,
        max_seq_length: int = 512,
        prompt: str = '',
        revision: Optional[str] = None,
        **kwargs,
    ):
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.model_kwargs = kwargs.pop('model_kwargs', {})
        self.model_kwargs['trust_remote_code'] = True

        self.config_kwargs = kwargs.pop('config_kwargs', {})
        self.config_kwargs['trust_remote_code'] = True

        self.encode_kwargs = kwargs.pop('encode_kwargs', {})
        self.encode_kwargs['convert_to_tensor'] = True

        self.prompt = prompt
        self.revision = revision

    @property
    def mteb_model_meta(self):
        """Model metadata for MTEB (Multilingual Task Embeddings Benchmark)"""
        from mteb import ModelMeta

        return ModelMeta(
            name=os.path.basename(self.model_name_or_path),
            revision=self.revision,
            languages=None,
            release_date=None,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs. Compact langchain.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """
        return self.encode_corpus(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed query text. Compact langchain.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """
        return self.encode_queries(text).tolist()

    def encode(self, texts: Union[str, List[str]], **kwargs) -> List[List[float]]:
        """Embed text."""
        raise NotImplementedError

    def encode_queries(self, queries: List[str], **kwargs) -> list[torch.Tensor]:
        """Embed query text. Compact mteb."""
        raise NotImplementedError

    def encode_corpus(self, corpus: Union[List[str], List[Dict[str, str]]], **kwargs) -> list[torch.Tensor]:
        """Embed search docs . Compact mteb."""
        raise NotImplementedError


class SentenceTransformerModel(BaseModel):

    def __init__(self, model_name_or_path: str, pooling_mode: Optional[str] = None, **kwargs):
        super().__init__(model_name_or_path, **kwargs)

        if not pooling_mode:
            self.model = SentenceTransformer(
                self.model_name_or_path,
                config_kwargs=self.config_kwargs,
                model_kwargs=self.model_kwargs,
            )
        else:
            word_embedding_model = models.Transformer(
                self.model_name_or_path,
                config_args=self.config_kwargs,
                model_args=self.model_kwargs,
            )
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(),
                pooling_mode=pooling_mode,
            )
            self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model], )

        self.model.max_seq_length = self.max_seq_length

    def encode(self, texts: Union[str, List[str]], prompt=None, **kwargs) -> List[torch.Tensor]:
        kwargs.pop('prompt_name', '')  # remove prompt name, use prompt
        self.encode_kwargs.update(kwargs)

        embeddings = self.model.encode(texts, prompt=prompt, **self.encode_kwargs)
        assert isinstance(embeddings, Tensor)
        return embeddings.cpu().detach()

    def encode_queries(self, queries, **kwargs):
        return self.encode(queries, prompt=self.prompt)

    def encode_corpus(self, corpus, **kwargs):
        if isinstance(corpus[0], dict):
            input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        else:
            input_texts = corpus
        return self.encode(input_texts)


class CrossEncoderModel(BaseModel):

    def __init__(self, model_name_or_path: str, **kwargs):
        super().__init__(model_name_or_path, **kwargs)
        self.model = CrossEncoder(
            self.model_name_or_path,
            trust_remote_code=True,
            max_length=self.max_seq_length,
        )

    def predict(self, sentences: List[List[str]], **kwargs) -> List[List[float]]:
        self.encode_kwargs.update(kwargs)

        if len(sentences[0]) == 3:  # Note: For mteb retrieval task
            processed_sentences = []
            for query, docs, instruction in sentences:
                if isinstance(docs, dict):
                    docs = docs['text']
                processed_sentences.append((self.prompt + query, docs))
            sentences = processed_sentences
        embeddings = self.model.predict(sentences, **self.encode_kwargs)
        assert isinstance(embeddings, Tensor)
        return embeddings


class EmbeddingModel:
    """Custom embeddings"""

    @staticmethod
    def load(
        model_name_or_path: str = '',
        is_cross_encoder: bool = False,
        hub: str = HubType.MODELSCOPE,
        revision: Optional[str] = 'master',
        **kwargs,
    ):
        # If model path does not exist and hub is 'modelscope', download the model
        if not os.path.exists(model_name_or_path) and hub == HubType.MODELSCOPE:
            model_name_or_path = download_model(model_name_or_path, revision)

        # Return different model instances based on whether it is a cross-encoder and pooling mode
        if is_cross_encoder:
            return CrossEncoderModel(
                model_name_or_path,
                revision=revision,
                **kwargs,
            )
        else:
            return SentenceTransformerModel(
                model_name_or_path,
                revision=revision,
                **kwargs,
            )
