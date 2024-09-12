import os
from typing import List, Optional, Dict, Any
from sentence_transformers import models
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from torch import Tensor
from evalscope.utils.logger import get_logger

logger = get_logger()


class BaseModel:
    def __init__(self, model_name_or_path: str, max_seq_length: int, **kwargs):
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.model_kwargs = kwargs.pop("model_kwargs", {})
        self.model_kwargs["trust_remote_code"] = True

        self.config_kwargs = kwargs.pop("config_kwargs", {})
        self.config_kwargs["trust_remote_code"] = True

        self.encode_kwargs = kwargs.pop("encode_kwargs", {})
        self.encode_kwargs["convert_to_tensor"] = True

    @property
    def mteb_model_meta(self):
        """Model metadata for MTEB (Multilingual Task Embeddings Benchmark)"""
        from mteb import ModelMeta

        return ModelMeta(
            name=self.model_name,
            revision=self.model_revision,
            languages=None,
            release_date=None,
        )


class SentenceTransformerModel(BaseModel):
    def __init__(
        self, model_name_or_path: str, pooling_mode: Optional[str] = None, **kwargs
    ):
        super().__init__(model_name_or_path, **kwargs)

        if not pooling_mode:
            self.model = SentenceTransformer(
                self.model_name_or_path,
                config_kwargs=self.config_kwargs,
                model_kwargs=self.model_kwargs,
                prompts=kwargs.pop("prompts", None),
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
            self.model = SentenceTransformer(
                modules=[word_embedding_model, pooling_model],
                prompts=kwargs.pop("prompts", None),
            )

        self.model.max_seq_length = self.max_seq_length
        self.update_model_info()

    def encode(self, texts: List[str], **kwargs) -> List[List[float]]:
        self.encode_kwargs.update(kwargs)
        embeddings = self.model.encode(texts, **self.encode_kwargs)
        assert isinstance(embeddings, Tensor)
        return embeddings

    def update_model_info(self):
        """Update model information from model card"""
        model_card = self.model.model_card_data
        self.model_name = model_card.model_name or os.path.basename(
            self.model_name_or_path
        )
        self.model_revision = model_card.base_model_revision or "v1"


class CrossEncoderModel(BaseModel):
    def __init__(self, model_name_or_path: str, **kwargs):
        super().__init__(model_name_or_path, **kwargs)
        self.model = CrossEncoder(
            self.model_name_or_path,
            trust_remote_code=True,
            max_length=self.max_seq_length,
        )
        self.update_model_info()

    def update_model_info(self):
        self.model_name = os.path.basename(self.model_name_or_path)
        self.model_revision = "v1"

    def encode(self, texts: List[str], **kwargs) -> List[List[float]]:
        self.encode_kwargs.update(kwargs)
        embeddings = self.model.predict(texts, **self.encode_kwargs)
        assert isinstance(embeddings, Tensor)
        return embeddings

    def predict(self, sentences: List[str], **kwargs) -> List[List[float]]:
        self.encode_kwargs.update(kwargs)
        embeddings = self.model.predict(sentences, **self.encode_kwargs)
        assert isinstance(embeddings, Tensor)
        return embeddings


class EmbeddingModel:
    """Custom embeddings"""

    @staticmethod
    def from_pretrained(
        model_name_or_path: str = "",
        is_cross_encoder: bool = False,
        hub: str = "modelscope",
        **kwargs,
    ):
        # If model path does not exist and hub is 'modelscope', download the model
        if not os.path.exists(model_name_or_path) and hub == "modelscope":
            from modelscope import snapshot_download

            logger.info(f"Loading model {model_name_or_path} from modelscope")
            model_name_or_path = snapshot_download(model_name_or_path)

        # Return different model instances based on whether it is a cross-encoder and pooling mode
        if is_cross_encoder:
            return CrossEncoderModel(
                model_name_or_path,
                **kwargs,
            )
        else:
            return SentenceTransformerModel(
                model_name_or_path,
                **kwargs,
            )
