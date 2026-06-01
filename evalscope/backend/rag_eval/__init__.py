from typing import TYPE_CHECKING

from evalscope.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from evalscope.backend.rag_eval.backend_manager import RAGEvalBackendManager, Tools
    from evalscope.backend.rag_eval.utils.clip import VisionModel
    from evalscope.backend.rag_eval.utils.embedding import EmbeddingModel
    from evalscope.backend.rag_eval.utils.llm import LLM, ChatOpenAI, LocalLLM

else:
    _import_structure = {
        'backend_manager': [
            'RAGEvalBackendManager',
            'Tools',
        ],
        'utils.clip': [
            'VisionModel',
        ],
        'utils.embedding': [
            'EmbeddingModel',
        ],
        'utils.llm': [
            'LLM',
            'ChatOpenAI',
            'LocalLLM',
        ],
        'cmteb': [],
        'ragas': [],
        'clip_benchmark': [],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
