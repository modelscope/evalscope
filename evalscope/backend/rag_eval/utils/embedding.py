import os
import torch
from langchain_core.embeddings import Embeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from mteb.encoder_interface import PromptType
from sentence_transformers import models
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.SentenceTransformer import SentenceTransformer
from torch import Tensor
from tqdm import tqdm
from typing import Dict, List, Optional, Union

from evalscope.backend.rag_eval.utils.tools import download_model
from evalscope.constants import HubType
from evalscope.utils.logger import get_logger
from evalscope.utils.utils import get_supported_params

logger = get_logger()


class BaseModel(Embeddings):

    def __init__(
        self,
        model_name_or_path: str = '',
        max_seq_length: int = 512,
        prompt: Optional[str] = None,
        prompts: Optional[Dict[str, str]] = None,
        revision: Optional[str] = 'master',
        **kwargs,
    ):
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.model_kwargs = kwargs.pop('model_kwargs', {})

        self.config_kwargs = kwargs.pop('config_kwargs', {})
        self.config_kwargs['trust_remote_code'] = True

        self.encode_kwargs = kwargs.pop('encode_kwargs', {})
        self.encode_kwargs['convert_to_tensor'] = True

        self.prompt = prompt
        self.prompts = prompts if prompts else {}
        self.revision = revision
        self.framework = ['PyTorch']

    @property
    def mteb_model_meta(self):
        """Model metadata for MTEB (Multilingual Task Embeddings Benchmark)"""
        from mteb import ModelMeta

        return ModelMeta(
            name='eval/' + os.path.basename(self.model_name_or_path),  # Ensure the name contains a slash
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
        """Embed search docs. Compact langchain.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """
        return self.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed query text. Compact langchain.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """
        return self.encode(text).tolist()

    def encode(self, texts: Union[str, List[str]], **kwargs) -> List[List[float]]:
        """Embed text."""
        raise NotImplementedError

    def get_prompt(self, task_name: str) -> Optional[str]:
        """Get prompt for the given task name."""
        if self.prompt:
            return self.prompt
        return self.prompts.get(task_name, None)


class SentenceTransformerModel(BaseModel):

    def __init__(self, model_name_or_path: str, pooling_mode: Optional[str] = None, **kwargs):
        super().__init__(model_name_or_path, **kwargs)

        self.framework = ['Sentence Transformers', 'PyTorch']

        self.model_kwargs['trust_remote_code'] = True
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

        self.supported_encode_params = get_supported_params(self.model.encode)

    def encode(self, texts: Union[str, List[str]], **kwargs) -> List[torch.Tensor]:
        # pop unused kwargs
        extra_params = {}
        for key in list(kwargs.keys()):
            if key not in self.supported_encode_params:
                extra_params[key] = kwargs.pop(key)
        self.encode_kwargs.update(kwargs)

        # set prompt if provided
        prompt = None
        prompt_type = extra_params.pop('prompt_type', '')
        task_name = extra_params.pop('task_name', '')
        if prompt_type and prompt_type == PromptType.query:
            prompt = self.get_prompt(task_name)

        embeddings = self.model.encode(texts, prompt=prompt, **self.encode_kwargs)
        assert isinstance(embeddings, Tensor)
        return embeddings.cpu().detach()


class CrossEncoderModel(BaseModel):

    def __init__(self, model_name_or_path: str, **kwargs):
        super().__init__(model_name_or_path, **kwargs)

        self.framework = ['Sentence Transformers', 'PyTorch']

        self.model = CrossEncoder(
            self.model_name_or_path,
            trust_remote_code=True,
            max_length=self.max_seq_length,
            automodel_args=self.model_kwargs,
        )
        self.supported_encode_params = get_supported_params(self.model.predict)

    def predict(self, sentences: List[List[str]], **kwargs) -> Tensor:
        for key in list(kwargs.keys()):
            if key not in self.supported_encode_params:
                kwargs.pop(key)
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


class APIEmbeddingModel(BaseModel):

    def __init__(self, **kwargs):
        self.model_name = kwargs.get('model_name')
        self.openai_api_base = kwargs.get('api_base')
        self.openai_api_key = kwargs.get('api_key')
        self.dimensions = kwargs.get('dimensions')
        self.framework = ['API']

        self.model = OpenAIEmbeddings(
            model=self.model_name,
            openai_api_base=self.openai_api_base,
            openai_api_key=self.openai_api_key,
            dimensions=self.dimensions,
            check_embedding_ctx_length=False)

        super().__init__(model_name_or_path=self.model_name, **kwargs)

        self.batch_size = self.encode_kwargs.get('batch_size', 10)

        self.supported_encode_params = get_supported_params(self.model.embed_documents)

    def encode(self, texts: Union[str, List[str]], **kwargs) -> Tensor:
        # pop unused kwargs
        extra_params = {}
        for key in list(kwargs.keys()):
            if key not in self.supported_encode_params:
                extra_params[key] = kwargs.pop(key)
        self.encode_kwargs.update(kwargs)

        # set prompt if provided
        prompt = None
        prompt_type = extra_params.pop('prompt_type', '')
        task_name = extra_params.pop('task_name', '')
        if prompt_type and prompt_type == PromptType.query:
            prompt = self.get_prompt(task_name)

        if isinstance(texts, str):
            texts = [texts]

        embeddings: List[List[float]] = []
        for i in tqdm(range(0, len(texts), self.batch_size)):
            # set prompt if provided
            if prompt is not None:
                batch_texts = [prompt + text for text in texts[i:i + self.batch_size]]
            else:
                batch_texts = texts[i:i + self.batch_size]
            response = self.model.embed_documents(batch_texts, chunk_size=self.batch_size)
            embeddings.extend(response)
        return torch.tensor(embeddings)


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
        if kwargs.get('model_name'):
            # If model_name is provided, use OpenAIEmbeddings
            return APIEmbeddingModel(**kwargs)

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
