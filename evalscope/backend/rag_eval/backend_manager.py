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
        from packaging.version import InvalidVersion, Version, parse
        try:
            mteb_version = parse(mteb.__version__)
        except InvalidVersion:
            raise ImportError(
                f'MTEB >= 2.7.0 is required (got {mteb.__version__}). '
                'Please upgrade: pip install "mteb>=2.7.0,<3.0.0"'
            )
        if mteb_version < Version('2.7.0'):
            raise ImportError(
                f'MTEB >= 2.7.0 is required (got {mteb.__version__}). '
                'Please upgrade: pip install "mteb>=2.7.0,<3.0.0"'
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
        from packaging.version import InvalidVersion, Version, parse
        try:
            ragas_version = parse(ragas.__version__)
        except InvalidVersion:
            raise ImportError(
                f'RAGAS >= 0.4.0 is required (got {ragas.__version__}). '
                'Please upgrade: pip install "ragas>=0.4.0,<0.5.0"'
            )
        if ragas_version < Version('0.4.0'):
            raise ImportError(
                f'RAGAS >= 0.4.0 is required (got {ragas.__version__}). '
                'Please upgrade: pip install "ragas>=0.4.0,<0.5.0"'
            )
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
