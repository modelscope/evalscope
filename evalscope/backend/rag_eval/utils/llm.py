import os
from typing import Any, Dict, Iterator, List, Mapping, Optional
from modelscope.utils.hf_util import GenerationConfig
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM as BaseLLM
from evalscope.models.model_adapter import ChatGenerationModelAdapter
from langchain_openai import ChatOpenAI


class LLM:
    @staticmethod
    def load(**kw):
        api_base = kw.get('api_base', None)
        if api_base:
            return ChatOpenAI(
                model_name=kw.get('model_name', ''),
                openai_api_base=api_base,
                openai_api_key=kw.get('api_key', 'EMPTY'),
            )
        else:
            return LocalLLM(**kw)


class LocalLLM(BaseLLM):
    """A custom LLM that loads a model from a given path and performs inference."""

    model_name_or_path: str
    model_revision: str = 'master'
    template_type: str = 'default'
    model_name: Optional[str]
    model: Optional[ChatGenerationModelAdapter]
    generation_config: Optional[Dict]

    def __init__(self, **kw):
        super().__init__(**kw)
        self.model_name = os.path.basename(self.model_name_or_path)
        self.model = ChatGenerationModelAdapter(
            model_id=self.model_name_or_path,
            model_revision=self.model_revision,
            template_type=self.template_type,
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

        response = self.model._model_generate(prompt, infer_cfg)
        return response

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
