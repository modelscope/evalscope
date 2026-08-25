# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Union

from evalscope.backend.base import BackendManager
from evalscope.utils.import_utils import is_module_installed
from evalscope.utils.logger import get_logger

logger = get_logger()


class Tools:
    MTEB = 'mteb'
    RAGAS = 'ragas'
    CLIP_BENCHMARK = 'clip_benchmark'


# Rationale appended to the sentence-transformers floor error, since an older version fails
# silently rather than loudly and the symptom on its own looks like a model quality problem.
_SENTENCE_TRANSFORMERS_REASON = (
    'Generative (CausalLM-based) rerankers such as Qwen3-Reranker need the CrossEncoder '
    'LogitScore module added in 5.4.0; older versions silently fall back to a '
    'randomly-initialized classification head and produce meaningless scores.'
)


def require_min_version(
    package: str, installed_version: str, min_version: str, install_spec: str, reason: str = ''
) -> None:
    """Raise ImportError if `installed_version` is below `min_version`.

    Pre-releases of the floor are accepted: a dev/rc build of `min_version` already carries the
    feature the floor exists for. Comparing against `<min_version>.dev0` rather than
    `min_version` also keeps a two-component version such as '5.4' from reading as older than
    '5.4.0'. An unparseable version fails closed.

    Args:
        package: Distribution name as shown to the user.
        installed_version: Version string reported by the installed package.
        min_version: Lowest supported release, as a plain 'X.Y.Z' string.
        install_spec: Requirement specifier to put in the suggested pip command.
        reason: Optional explanation of why the floor exists.
    """
    from packaging.version import InvalidVersion, Version, parse
    try:
        supported = parse(installed_version) >= Version(f'{min_version}.dev0')
    except InvalidVersion:
        supported = False
    if not supported:
        detail = f'{reason} ' if reason else ''
        raise ImportError(
            f'{package} >= {min_version} is required (got {installed_version}). '
            f'{detail}Please upgrade: pip install "{install_spec}"'
        )


class RAGEvalBackendManager(BackendManager):

    def __init__(self, config: Union[str, dict], **kwargs):
        """BackendManager for RAG Evaluation.

        Args:
            config: Configuration as yaml file path, dict, or Pydantic model.
        """
        super().__init__(config, **kwargs)

    @staticmethod
    def _check_env(module_name: str) -> None:
        if is_module_installed(module_name):
            logger.info(f'Check `{module_name}` Installed')
        else:
            raise RuntimeError(f'`{module_name}` is not installed. Please install it with: pip install {module_name}')

    @staticmethod
    def run_mteb(config) -> None:
        """Run MTEB evaluation.

        Args:
            config: MTEBToolConfig instance or dict with MTEB configuration.
        """
        import mteb
        import sentence_transformers
        require_min_version('MTEB', mteb.__version__, '2.7.0', 'mteb>=2.7.0,<3.0.0')
        require_min_version(
            'sentence-transformers',
            sentence_transformers.__version__,
            '5.4.0',
            'sentence-transformers>=5.4.0',
            reason=_SENTENCE_TRANSFORMERS_REASON,
        )
        from evalscope.backend.rag_eval.mteb import MTEBToolConfig, run_mteb_eval

        if isinstance(config, dict):
            config = MTEBToolConfig(**config)
        run_mteb_eval(config)

    @staticmethod
    def run_ragas(config) -> None:
        """Run RAGAS evaluation and/or testset generation.

        Args:
            config: RAGASToolConfig instance or dict with RAGAS configuration.
        """
        import ragas
        require_min_version('RAGAS', ragas.__version__, '0.4.0', 'ragas>=0.4.0,<0.5.0')
        from evalscope.backend.rag_eval.ragas import RAGASToolConfig, rag_eval
        from evalscope.backend.rag_eval.ragas.tasks import generate_testset

        if isinstance(config, dict):
            config = RAGASToolConfig(**config)

        if config.testset_generation is not None:
            generate_testset(config.testset_generation)
        if config.eval is not None:
            rag_eval(config.eval)

    @staticmethod
    def run_clip_benchmark(config) -> None:
        """Run CLIP Benchmark evaluation.

        Args:
            config: ClipBenchmarkToolConfig instance.
        """
        from evalscope.backend.rag_eval.clip_benchmark import evaluate

        evaluate(config.eval)

    def run(self, *args, **kwargs) -> None:
        """Run the RAG evaluation pipeline based on tool type."""
        from evalscope.backend.rag_eval.clip_benchmark.arguments import ClipBenchmarkToolConfig
        from evalscope.backend.rag_eval.mteb.arguments import MTEBToolConfig
        from evalscope.backend.rag_eval.ragas.arguments import RAGASToolConfig

        config = self.config_d

        if isinstance(config, MTEBToolConfig):
            self._check_env('mteb')
            self.run_mteb(config)
        elif isinstance(config, RAGASToolConfig):
            self._check_env('ragas')
            self.run_ragas(config)
        elif isinstance(config, ClipBenchmarkToolConfig):
            self._check_env('webdataset')
            self.run_clip_benchmark(config)
        else:
            raise ValueError(
                f'Unsupported config type: {type(config)}. '
                f'Expected MTEBToolConfig, RAGASToolConfig, or ClipBenchmarkToolConfig.'
            )
