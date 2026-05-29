from evalscope.backend.rag_eval.backend_manager import RAGEvalBackendManager, Tools
from evalscope.backend.rag_eval.models import (
    APIEncoder,
    APIReranker,
    CrossEncoderReranker,
    SentenceTransformerEncoder,
    load_model,
)

# Backward-compatible aliases
try:
    from evalscope.backend.rag_eval.utils.embedding import EmbeddingModel
except ImportError:
    EmbeddingModel = None

try:
    from evalscope.backend.rag_eval.utils.clip import VisionModel
except ImportError:
    VisionModel = None

try:
    from evalscope.backend.rag_eval.utils.llm import LLM, ChatOpenAI, LocalLLM
except ImportError:
    LLM = None
    ChatOpenAI = None
    LocalLLM = None
