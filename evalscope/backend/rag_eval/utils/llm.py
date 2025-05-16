import os
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM as BaseLLM
from langchain_openai import ChatOpenAI
from transformers.generation.configuration_utils import GenerationConfig
from typing import Any, Dict, Iterator, List, Mapping, Optional

from evalscope.constants import DEFAULT_MODEL_REVISION
from evalscope.models import ChatGenerationModelAdapter, LocalModel


class LLM:

    @staticmethod
    def load(**kw):
        api_base = kw.get('api_base', None)
        if api_base:
            return ChatOpenAI(
                model=kw.get('model_name', ''),
                base_url=api_base,
                api_key=kw.get('api_key', 'EMPTY'),
            )
        else:
            return LocalLLM(**kw)


class LocalLLM(BaseLLM):
    """A custom LLM that loads a model from a given path and performs inference."""

    model_name_or_path: str
    model_revision: str = DEFAULT_MODEL_REVISION
    template_type: Optional[str] = None
    model_name: Optional[str]
    model: Optional[ChatGenerationModelAdapter]
    generation_config: Optional[Dict]

    def __init__(self, **kw):
        super().__init__(**kw)
        self.model_name = os.path.basename(self.model_name_or_path)
        self.model = ChatGenerationModelAdapter(
            model=LocalModel(model_id=self.model_name_or_path, model_revision=self.model_revision),
            generation_config=GenerationConfig(**self.generation_config) if self.generation_config else None,
        )

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input."""
        infer_cfg = {'stop': stop}

        response, _ = self.model.predict([{'data': [prompt]}], infer_cfg=infer_cfg)
        return response[0][0]

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            'model_name': self.model_name,
            'revision': self.model_revision,
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return self.model_name
