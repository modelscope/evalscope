import logging
import os
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM as BaseLLM
from langchain_openai import ChatOpenAI
from typing import Any, Dict, List, Optional

from evalscope.api.model import GenerateConfig, Model, get_model
from evalscope.constants import DEFAULT_MODEL_REVISION, EvalType

logger = logging.getLogger(__name__)


class LLM:
    """Factory for creating LLM instances (LangChain-compatible)."""

    @staticmethod
    def load(**kw):
        """Load an LLM instance based on config.

        If api_base is provided, creates a ChatOpenAI (remote API).
        Otherwise, creates a LocalLLM (local checkpoint).
        """
        api_base = kw.get('api_base', None)
        if api_base:
            return ChatOpenAI(
                model=kw.get('model_name', ''),
                base_url=api_base,
                api_key=kw.get('api_key', 'EMPTY'),
                temperature=kw.get('temperature', 0.0),
            )
        else:
            return LocalLLM(**kw)

    @staticmethod
    def load_ragas_llm(**kw):
        """Load a ragas-native LLM using llm_factory (ragas 0.4.x).

        Falls back to LangchainLLMWrapper if llm_factory is unavailable.
        """
        api_base = kw.get('api_base', None)
        model_name = kw.get('model_name', '')
        api_key = kw.get('api_key', None)

        if api_base:
            try:
                from openai import OpenAI
                from ragas.llms import llm_factory

                client_kwargs: Dict[str, Any] = {'base_url': api_base}
                if api_key:
                    client_kwargs['api_key'] = api_key
                client = OpenAI(**client_kwargs)
                return llm_factory(model_name, client=client)
            except (ImportError, Exception) as e:
                logger.warning(f'Failed to use ragas llm_factory: {e}, falling back to LangchainLLMWrapper')
                from ragas.llms.base import LangchainLLMWrapper
                langchain_llm = ChatOpenAI(
                    model=model_name,
                    base_url=api_base,
                    api_key=api_key or 'EMPTY',
                    temperature=kw.get('temperature', 0.0),
                )
                return LangchainLLMWrapper(langchain_llm)
        else:
            # Local model wrapped for ragas
            from ragas.llms.base import LangchainLLMWrapper
            local_llm = LocalLLM(**kw)
            return LangchainLLMWrapper(local_llm)


class LocalLLM(BaseLLM):
    """A custom LLM that loads a model from a given path and performs inference."""

    model_name_or_path: str
    model_revision: str = DEFAULT_MODEL_REVISION
    template_type: Optional[str] = None
    model_name: Optional[str] = None
    model: Optional[Model] = None
    generation_config: Optional[Dict] = {}

    def __init__(self, **kw):
        super().__init__(**kw)
        self.model_name = os.path.basename(self.model_name_or_path)

        # Create and initialize the local model
        self.model = get_model(
            model=self.model_name_or_path,
            eval_type=EvalType.CHECKPOINT,
            config=GenerateConfig(**self.generation_config),
        )

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input."""
        response = self.model.generate(input=prompt)
        return response.completion

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            'model_name': self.model_name,
            'revision': self.model_revision,
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return self.model_name
